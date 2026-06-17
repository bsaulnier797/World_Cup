"""
wc_model.py
===========
All the math for the World Cup predictor. NO streamlit imports belong here so
this stays testable from the command line (python wc_model.py).

Pipeline:
    1. load_results / get_training_data   -- data
    2. compute_elo                        -- a strength rating per team
    3. match_probabilities                -- Elo gap -> P(home/draw/away)
    4. get_fixtures_for_date              -- the games on a given day
    5. single_kelly / optimize_stakes     -- how much to bet on each game
    6. backtest                           -- "what % would you be up?" (point-in-time)
"""

import itertools

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.special import gammaln

try:
    import cvxpy as cp
    HAVE_CVXPY = True
except ImportError:
    HAVE_CVXPY = False


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

BASE_RATING = 1500.0
HOME_ADVANTAGE_DEFAULT = 100.0

# Bigger competitions move ratings more; friendlies barely count.
K_BY_TOURNAMENT = {
    "FIFA World Cup": 60,
    "Copa América": 50,
    "UEFA Euro": 50,
    "African Cup of Nations": 50,
    "AFC Asian Cup": 50,
    "Gold Cup": 50,
    "Confederations Cup": 45,
    "FIFA World Cup qualification": 40,
    "UEFA Nations League": 40,
    "CONCACAF Nations League": 35,
    "Friendly": 20,
}
K_DEFAULT = 30


# ---------------------------------------------------------------------------
# PHASE 1: DATA
# ---------------------------------------------------------------------------

def load_results():
    """Download the full international results dataset (date parsed as datetime)."""
    return pd.read_csv(DATA_URL, parse_dates=["date"])


def get_training_data(df):
    """Completed matches only (both scores present), sorted oldest-to-newest."""
    completed = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    return completed.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# PHASE 2: ELO ENGINE
# ---------------------------------------------------------------------------

def _k_factor(tournament):
    return K_BY_TOURNAMENT.get(tournament, K_DEFAULT)


def _goal_diff_multiplier(home_score, away_score):
    """World Football Elo goal-difference index G (bigger wins -> bigger update)."""
    gd = abs(int(home_score) - int(away_score))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def _is_neutral(value):
    """`neutral` can load as a real bool or as 'TRUE'/'FALSE'. Coerce safely."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in ("true", "1")


def expected_score(r_home_effective, r_away):
    """Home win-expectancy (draw counts 0.5). Inputs already include home adv."""
    return 1.0 / (1.0 + 10.0 ** ((r_away - r_home_effective) / 400.0))


def update_elo(home_elo, away_elo, home_score, away_score,
               tournament="Friendly", neutral=True,
               home_advantage=HOME_ADVANTAGE_DEFAULT):
    """
    New ratings for both teams after one match. Per-match building block.
    Home advantage shifts only the *effective* rating used for the expectation;
    the stored rating never bakes it in.
    """
    adj = 0.0 if neutral else home_advantage
    exp_home = expected_score(home_elo + adj, away_elo)

    if home_score > away_score:
        actual_home = 1.0
    elif home_score < away_score:
        actual_home = 0.0
    else:
        actual_home = 0.5

    k = _k_factor(tournament)
    g = _goal_diff_multiplier(home_score, away_score)
    delta = k * g * (actual_home - exp_home)

    return home_elo + delta, away_elo - delta  # zero-sum


def compute_elo(results_df, home_advantage=HOME_ADVANTAGE_DEFAULT):
    """
    Walk completed matches in date order; return current ratings + match counts.
        ratings[team] = float
        counts[team]  = int  (< 30 => provisional, trust less)
    """
    ratings, counts = {}, {}
    for row in results_df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        rh = ratings.get(home, BASE_RATING)
        ra = ratings.get(away, BASE_RATING)

        new_rh, new_ra = update_elo(
            rh, ra, row.home_score, row.away_score,
            tournament=row.tournament,
            neutral=_is_neutral(row.neutral),
            home_advantage=home_advantage,
        )
        ratings[home], ratings[away] = new_rh, new_ra
        counts[home] = counts.get(home, 0) + 1
        counts[away] = counts.get(away, 0) + 1
    return ratings, counts


# ---------------------------------------------------------------------------
# PHASE 3: ELO -> H/D/A PROBABILITIES
# ---------------------------------------------------------------------------

def match_probabilities(r_home, r_away, neutral=True,
                        home_advantage=HOME_ADVANTAGE_DEFAULT, draw_max=0.30):
    """
    Split the Elo win-expectancy into (P_home, P_draw, P_away).

    Improvements over a flat draw probability:
      - draw probability PEAKS for evenly matched teams and shrinks with the gap
      - the construction preserves the Elo expectation exactly:
            P_home + 0.5*P_draw == We
      - probabilities always sum to 1 and stay non-negative
    """
    adj = 0.0 if neutral else home_advantage
    we = expected_score(r_home + adj, r_away)

    p_draw = draw_max * (1.0 - 2.0 * abs(we - 0.5))
    p_draw = min(p_draw, 2.0 * min(we, 1.0 - we))   # keep the next two >= 0
    p_home = we - p_draw / 2.0
    p_away = 1.0 - we - p_draw / 2.0
    return float(p_home), float(p_draw), float(p_away)


# ---------------------------------------------------------------------------
# PHASE 3b: DIXON-COLES SCORELINE MODEL  (the better predictor)
# ---------------------------------------------------------------------------
# Instead of forcing one Elo number into three outcomes with a hand-built draw
# rule, model the GOALS each side scores as Poisson, driven by the Elo gap, with
# the Dixon-Coles low-score correction. Home/draw/away then fall out of the score
# matrix, and you get exact-scoreline and over/under probabilities for free.
#
#   log(lambda_home) = a + b * (elo_diff / 100) + h * (plays_at_home)
#   log(lambda_away) = a - b * (elo_diff / 100)
# where elo_diff is the venue-neutral strength gap. The 4 parameters (a, b, h,
# rho) are fit once by maximum likelihood -- few parameters, so not overfit-prone.

def _poisson_logpmf(k, lam):
    lam = np.maximum(lam, 1e-10)
    return k * np.log(lam) - lam - gammaln(k + 1.0)


def _dc_tau(hg, ag, lh, la, rho):
    """Dixon-Coles low-score correction on the (0/1 x 0/1) cells; 1 elsewhere."""
    tau = np.ones_like(lh, dtype=float)
    tau = np.where((hg == 0) & (ag == 0), 1.0 - lh * la * rho, tau)
    tau = np.where((hg == 0) & (ag == 1), 1.0 + lh * rho, tau)
    tau = np.where((hg == 1) & (ag == 0), 1.0 + la * rho, tau)
    tau = np.where((hg == 1) & (ag == 1), 1.0 - rho, tau)
    return tau


def _build_fit_arrays(train_df, home_advantage=HOME_ADVANTAGE_DEFAULT):
    """Walk Elo point-in-time and return arrays (d, venue, home_goals, away_goals)."""
    ratings = {}
    d_list, venue_list, hg_list, ag_list = [], [], [], []
    for row in train_df.itertuples(index=False):
        h, a = row.home_team, row.away_team
        rh = ratings.get(h, BASE_RATING)
        ra = ratings.get(a, BASE_RATING)
        neutral = _is_neutral(row.neutral)

        d_list.append((rh - ra) / 100.0)          # venue-neutral strength gap
        venue_list.append(0.0 if neutral else 1.0)
        hg_list.append(int(row.home_score))
        ag_list.append(int(row.away_score))

        new_rh, new_ra = update_elo(rh, ra, row.home_score, row.away_score,
                                    tournament=row.tournament, neutral=neutral,
                                    home_advantage=home_advantage)
        ratings[h], ratings[a] = new_rh, new_ra
    return (np.array(d_list), np.array(venue_list),
            np.array(hg_list), np.array(ag_list))


def _scoreline_nll(theta, d, venue, hg, ag):
    """Negative log-likelihood of the Dixon-Coles + Elo-covariate model."""
    a_, b_, h_, rho_ = theta
    lh = np.exp(a_ + b_ * d + h_ * venue)
    la = np.exp(a_ - b_ * d)
    tau = np.maximum(_dc_tau(hg, ag, lh, la, rho_), 1e-10)
    ll = _poisson_logpmf(hg, lh) + _poisson_logpmf(ag, la) + np.log(tau)
    return -np.sum(ll)


def fit_scoreline_model(train_df, home_advantage=HOME_ADVANTAGE_DEFAULT):
    """
    Fit the Dixon-Coles + Elo-covariate goals model by maximum likelihood.
    Returns a params dict {a, b, h, rho} for use with predict_scoreline().
    """
    d, venue, hg, ag = _build_fit_arrays(train_df, home_advantage)
    x0 = np.array([0.2, 0.3, 0.25, -0.05])
    bounds = [(-1.0, 1.0), (0.0, 2.0), (-0.5, 1.0), (-0.2, 0.2)]
    res = minimize(_scoreline_nll, x0, args=(d, venue, hg, ag),
                   method="L-BFGS-B", bounds=bounds)
    a_, b_, h_, rho_ = res.x
    return {"a": float(a_), "b": float(b_), "h": float(h_), "rho": float(rho_)}


def _numeric_hessian(f, x, eps=1e-4):
    """Finite-difference Hessian of scalar f at x (small dimension only)."""
    n = len(x)
    H = np.zeros((n, n))
    fx = f(x)
    for i in range(n):
        for j in range(i, n):
            xi, xj, xij = x.copy(), x.copy(), x.copy()
            xi[i] += eps
            xj[j] += eps
            xij[i] += eps
            xij[j] += eps
            H[i, j] = H[j, i] = (f(xij) - f(xi) - f(xj) + fx) / (eps * eps)
    return H


def fit_scoreline_model_with_cov(train_df, home_advantage=HOME_ADVANTAGE_DEFAULT):
    """
    Like fit_scoreline_model, but also returns the parameter covariance matrix
    (inverse observed Fisher information = inverse Hessian of the NLL at the MLE).
    This covariance is what lets us put a confidence interval / p-value on a
    prediction by sampling plausible parameter values.
    Returns (params_dict, cov 4x4 array, param_order list).
    """
    d, venue, hg, ag = _build_fit_arrays(train_df, home_advantage)
    x0 = np.array([0.2, 0.3, 0.25, -0.05])
    bounds = [(-1.0, 1.0), (0.0, 2.0), (-0.5, 1.0), (-0.2, 0.2)]
    res = minimize(_scoreline_nll, x0, args=(d, venue, hg, ag),
                   method="L-BFGS-B", bounds=bounds)
    theta = res.x

    H = _numeric_hessian(lambda t: _scoreline_nll(t, d, venue, hg, ag), theta)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    # numerical guard: symmetrise and nudge to positive semi-definite
    cov = (cov + cov.T) / 2.0

    params = {"a": float(theta[0]), "b": float(theta[1]),
              "h": float(theta[2]), "rho": float(theta[3])}
    return params, cov, ["a", "b", "h", "rho"]




def _score_matrix(lh, la, rho, max_goals=10):
    """Dixon-Coles probability matrix where matrix[i, j] = P(home i, away j)."""
    i = np.arange(0, max_goals + 1)
    ph = np.exp(_poisson_logpmf(i, lh))
    pa = np.exp(_poisson_logpmf(i, la))
    matrix = np.outer(ph, pa)
    matrix[0, 0] *= 1.0 - lh * la * rho
    matrix[0, 1] *= 1.0 + lh * rho
    matrix[1, 0] *= 1.0 + la * rho
    matrix[1, 1] *= 1.0 - rho
    matrix = np.maximum(matrix, 0.0)
    return matrix / matrix.sum()


def _lambdas(r_home, r_away, params, neutral):
    """Expected goals for home and away from the fitted model."""
    d = (r_home - r_away) / 100.0
    venue = 0.0 if neutral else 1.0
    lh = float(np.exp(params["a"] + params["b"] * d + params["h"] * venue))
    la = float(np.exp(params["a"] - params["b"] * d))
    return lh, la


def _hda_from_matrix(matrix):
    p_home = float(np.tril(matrix, -1).sum())   # home goals > away goals
    p_away = float(np.triu(matrix, 1).sum())    # away goals > home goals
    p_draw = float(np.trace(matrix))
    return p_home, p_draw, p_away


def predict_scoreline(r_home, r_away, params, neutral=True, max_goals=10):
    """
    Full prediction from the fitted goals model. Returns a dict with H/D/A
    probabilities, expected goals for each side, and the most likely scoreline.
    """
    lh, la = _lambdas(r_home, r_away, params, neutral)
    matrix = _score_matrix(lh, la, params["rho"], max_goals)
    p_home, p_draw, p_away = _hda_from_matrix(matrix)
    best = np.unravel_index(np.argmax(matrix), matrix.shape)
    return {
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "exp_home_goals": lh, "exp_away_goals": la,
        "likely_score": (int(best[0]), int(best[1])),
    }


def match_probabilities_dc(r_home, r_away, params, neutral=True):
    """Drop-in (P_home, P_draw, P_away) from the Dixon-Coles model."""
    r = predict_scoreline(r_home, r_away, params, neutral=neutral)
    return r["p_home"], r["p_draw"], r["p_away"]


# ---------------------------------------------------------------------------
# STATISTICAL INFERENCE: confidence interval, probability, p-value
# ---------------------------------------------------------------------------

def predict_with_inference(r_home, r_away, params, cov, neutral=True,
                           home_name="Home", away_name="Away",
                           n_samples=3000, ci=0.95, seed=0, max_goals=10):
    """
    Predict a match AND quantify how sure we are, separating the two kinds of
    uncertainty:

    ESTIMATION uncertainty (how well we know the probabilities) -- by drawing
    parameter vectors from N(theta_hat, cov) and recomputing the prediction:
        - winner, winner_prob (point estimate)
        - prob_ci             : confidence interval on the winner's win probability
        - p_value             : two-sided test of H0 'teams evenly matched'
                                (expected goal supremacy == 0); significant<0.05
                                means the favourite is genuinely stronger

    OUTCOME uncertainty (how random the match is, even with perfect knowledge):
        - p_home/p_draw/p_away, likely_score
        - margin_interval     : predictive interval on goal margin (home - away)

    Returns one dict with everything.
    """
    rng = np.random.default_rng(seed)
    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q

    # --- point estimate ---
    lh0, la0 = _lambdas(r_home, r_away, params, neutral)
    matrix0 = _score_matrix(lh0, la0, params["rho"], max_goals)
    p_home, p_draw, p_away = _hda_from_matrix(matrix0)
    labels = [home_name, "Draw", away_name]
    point = [p_home, p_draw, p_away]
    win_idx = int(np.argmax(point))
    winner = labels[win_idx]
    winner_prob = point[win_idx]
    supremacy_hat = lh0 - la0

    # --- estimation uncertainty: sample parameters ---
    theta_hat = np.array([params["a"], params["b"], params["h"], params["rho"]])
    try:
        draws = rng.multivariate_normal(theta_hat, cov, size=n_samples)
    except Exception:
        draws = np.repeat(theta_hat[None, :], n_samples, axis=0)

    d = (r_home - r_away) / 100.0
    venue = 0.0 if neutral else 1.0
    winner_probs, supremacies = [], []
    for a_, b_, h_, rho_ in draws:
        lh = np.exp(a_ + b_ * d + h_ * venue)
        la = np.exp(a_ - b_ * d)
        mat = _score_matrix(lh, la, rho_, max_goals)
        winner_probs.append(_hda_from_matrix(mat)[win_idx])
        supremacies.append(lh - la)
    winner_probs = np.array(winner_probs)
    supremacies = np.array(supremacies)

    prob_ci = (float(np.quantile(winner_probs, lo_q)),
               float(np.quantile(winner_probs, hi_q)))

    # two-sided p-value for H0: supremacy == 0 (teams evenly matched)
    if supremacy_hat >= 0:
        p_value = 2.0 * float(np.mean(supremacies <= 0.0))
    else:
        p_value = 2.0 * float(np.mean(supremacies >= 0.0))
    p_value = min(1.0, p_value)

    # --- outcome uncertainty: predictive interval on goal margin ---
    g = np.arange(0, max_goals + 1)
    margins = g[:, None] - g[None, :]          # margin for each (i, j)
    flat_m = margins.ravel()
    flat_p = matrix0.ravel()
    order = np.argsort(flat_m)
    m_sorted, p_sorted = flat_m[order], flat_p[order]
    cdf = np.cumsum(p_sorted)
    margin_lo = int(m_sorted[np.searchsorted(cdf, lo_q)])
    margin_hi = int(m_sorted[np.searchsorted(cdf, hi_q)])
    best = np.unravel_index(np.argmax(matrix0), matrix0.shape)

    return {
        "home_name": home_name, "away_name": away_name,
        "winner": winner,
        "winner_prob": winner_prob,
        "prob_ci": prob_ci,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "exp_home_goals": lh0, "exp_away_goals": la0,
        "likely_score": (int(best[0]), int(best[1])),
        "margin_interval": (margin_lo, margin_hi),
        "ci_level": ci,
    }


# ---------------------------------------------------------------------------
# PROPER SCORING RULES  (to prove one model beats another)
# ---------------------------------------------------------------------------

def log_loss(p_home, p_draw, p_away, outcome):
    """Negative log of the probability assigned to what actually happened."""
    p = {"Home": p_home, "Draw": p_draw, "Away": p_away}[outcome]
    return -np.log(max(p, 1e-12))


def rps(p_home, p_draw, p_away, outcome):
    """Ranked Probability Score for ordered outcomes (Home, Draw, Away)."""
    preds = np.array([p_home, p_draw, p_away])
    obs = np.array([1.0 if outcome == o else 0.0 for o in ("Home", "Draw", "Away")])
    cum_p = np.cumsum(preds)
    cum_o = np.cumsum(obs)
    return float(np.sum((cum_p[:-1] - cum_o[:-1]) ** 2) / (len(preds) - 1))


# ---------------------------------------------------------------------------
# PHASE 4: FIXTURES
# ---------------------------------------------------------------------------

def get_fixtures_for_date(df, date, tournament="FIFA World Cup", unplayed_only=True):
    """
    World Cup fixtures on a given date. `date` may be a string ('2026-06-15') or
    a date/Timestamp. By default returns only unplayed games (NA scores).
    """
    date = pd.Timestamp(date).normalize()
    mask = (df["date"].dt.normalize() == date) & (df["tournament"] == tournament)
    fixtures = df.loc[mask].copy()
    if unplayed_only:
        fixtures = fixtures[fixtures["home_score"].isna()]
    return fixtures[["date", "home_team", "away_team", "neutral", "city"]].reset_index(drop=True)


def upcoming_wc_dates(df, today, tournament="FIFA World Cup"):
    """Sorted list of future dates that still have unplayed WC fixtures."""
    today = pd.Timestamp(today).normalize()
    wc = df[(df["tournament"] == tournament) & (df["home_score"].isna())]
    days = sorted({d.normalize() for d in wc["date"]})
    return [d for d in days if d >= today]


# ---------------------------------------------------------------------------
# PHASE 6: KELLY BET SIZING
# ---------------------------------------------------------------------------

def single_kelly(p_win, odds):
    """Closed-form Kelly fraction for ONE bet. 0 if there's no edge."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (p_win * odds - 1.0) / b)


def best_edge_bet(p_home, p_draw, p_away, odds_home, odds_draw, odds_away):
    """
    For one game, pick the single outcome with the largest POSITIVE edge
    (edge = p*odds - 1). Returns a dict or None if no outcome is +EV.
    """
    options = [
        ("Home", p_home, odds_home),
        ("Draw", p_draw, odds_draw),
        ("Away", p_away, odds_away),
    ]
    best = None
    for outcome, p, odds in options:
        if odds is None or pd.isna(odds) or odds <= 1.0:
            continue
        edge = p * odds - 1.0
        if best is None or edge > best["edge"]:
            best = {"outcome": outcome, "p": p, "odds": float(odds), "edge": edge}
    if best is None or best["edge"] <= 0:
        return None
    return best


def _solve_joint_kelly(ps, odds):
    """
    Simultaneous-bet Kelly: maximise E[log wealth] over the 2^N joint win/lose
    outcomes, subject to fractions >= 0 and sum <= 1. Returns full-Kelly
    fractions, or None if cvxpy is unavailable / fails.
    """
    n = len(ps)
    if not HAVE_CVXPY or n == 0 or n > 16:
        return None
    ps = np.asarray(ps, float)
    b = np.asarray(odds, float) - 1.0

    f = cp.Variable(n, nonneg=True)
    log_growth = 0.0
    for combo in itertools.product([1, 0], repeat=n):
        combo = np.asarray(combo)
        prob = float(np.prod(np.where(combo == 1, ps, 1.0 - ps)))
        if prob == 0.0:
            continue
        ret = cp.sum(cp.multiply(f, np.where(combo == 1, b, -1.0)))
        log_growth += prob * cp.log(1.0 + ret)

    try:
        cp.Problem(cp.Maximize(log_growth), [cp.sum(f) <= 1.0]).solve()
    except Exception:
        return None
    return None if f.value is None else np.clip(f.value, 0.0, None)


def optimize_stakes(games, bankroll, kelly_fraction=0.5):
    """
    Given today's games, recommend a stake per game using simultaneous Kelly.

    `games` is a list of dicts, each with keys:
        game, p_home, p_draw, p_away, odds_home, odds_draw, odds_away
    Returns a DataFrame: one row per game with pick, edge and recommended stake.
    Games with no +EV outcome get a stake of 0.
    """
    picks, idx = [], []
    for i, g in enumerate(games):
        b = best_edge_bet(g["p_home"], g["p_draw"], g["p_away"],
                          g["odds_home"], g["odds_draw"], g["odds_away"])
        if b is not None:
            picks.append(b)
            idx.append(i)

    fractions = np.zeros(len(picks))
    if picks:
        f = _solve_joint_kelly([b["p"] for b in picks], [b["odds"] for b in picks])
        if f is None:  # fallback: independent single-bet Kelly, capped at bankroll
            f = np.array([single_kelly(b["p"], b["odds"]) for b in picks])
            if f.sum() > 1.0:
                f = f / f.sum()
        fractions = np.clip(f * kelly_fraction, 0.0, None)

    stake_by_game = {}
    for b, frac, gi in zip(picks, fractions, idx):
        stake_by_game[gi] = (b, frac * bankroll)

    rows = []
    for i, g in enumerate(games):
        if i in stake_by_game:
            b, stake = stake_by_game[i]
            rows.append({
                "Game": g["game"], "Pick": b["outcome"],
                "Model prob": round(b["p"], 3), "Odds": round(b["odds"], 2),
                "Edge": round(b["edge"], 3), "Stake": round(stake, 2),
            })
        else:
            rows.append({
                "Game": g["game"], "Pick": "No bet",
                "Model prob": None, "Odds": None, "Edge": None, "Stake": 0.0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PHASE 7: BACKTEST  ("what % would you be up?")
# ---------------------------------------------------------------------------

def _synthetic_odds(p_home, p_draw, p_away, margin, rng):
    """Plausible bookmaker odds = fair probs + vig + a little noise, then inverted."""
    probs = np.array([p_home, p_draw, p_away]) * (1.0 + margin)
    probs = np.clip(probs * rng.uniform(0.92, 1.08, size=3), 1e-3, 0.99)
    return tuple(1.0 / probs)


def backtest(results_df, starting_bankroll=1000.0, kelly_fraction=0.5,
             home_advantage=HOME_ADVANTAGE_DEFAULT, draw_max=0.30,
             bet_tournament="FIFA World Cup", odds_margin=0.05, seed=42,
             real_odds=None):
    """
    Replay history match-by-match and bet the model's suggestions using ONLY
    information available before each match (no lookahead). Reports the
    cumulative bankroll curve and the percentage you'd be up or down.

    Critical ordering per match:  PREDICT (from current ratings) -> BET ->
    LEARN (apply the Elo update). Predicting before the update is what keeps
    the rating from ever 'seeing' the result it's being asked to predict.

    real_odds: optional dict keyed by (date_str, home, away) -> (oh, od, oa).
               If absent, synthetic odds are used -- which is CIRCULAR and only
               good for confirming the machinery (the % should hover near 0).
    """
    rng = np.random.default_rng(seed)
    ratings = {}
    bankroll = starting_bankroll

    curve = []          # (date, bankroll)
    n_bets = wins = 0
    total_staked = 0.0

    for row in results_df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        rh = ratings.get(home, BASE_RATING)
        ra = ratings.get(away, BASE_RATING)
        neutral = _is_neutral(row.neutral)

        # --- BET (only on the target tournament) ---
        if row.tournament == bet_tournament:
            p_home, p_draw, p_away = match_probabilities(
                rh, ra, neutral=neutral, home_advantage=home_advantage, draw_max=draw_max)

            if real_odds is not None:
                key = (pd.Timestamp(row.date).strftime("%Y-%m-%d"), home, away)
                odds = real_odds.get(key)
            else:
                odds = _synthetic_odds(p_home, p_draw, p_away, odds_margin, rng)

            if odds is not None:
                oh, od, oa = odds
                bet = best_edge_bet(p_home, p_draw, p_away, oh, od, oa)
                if bet is not None:
                    stake = bankroll * kelly_fraction * single_kelly(bet["p"], bet["odds"])
                    if stake > 0:
                        # actual outcome from the real scores
                        if row.home_score > row.away_score:
                            actual = "Home"
                        elif row.home_score < row.away_score:
                            actual = "Away"
                        else:
                            actual = "Draw"

                        if bet["outcome"] == actual:
                            bankroll += stake * (bet["odds"] - 1.0)
                            wins += 1
                        else:
                            bankroll -= stake
                        n_bets += 1
                        total_staked += stake
                        curve.append((pd.Timestamp(row.date), bankroll))

        # --- LEARN (always, with the real result) ---
        new_rh, new_ra = update_elo(
            rh, ra, row.home_score, row.away_score,
            tournament=row.tournament, neutral=neutral, home_advantage=home_advantage)
        ratings[home], ratings[away] = new_rh, new_ra

    final_pct = (bankroll - starting_bankroll) / starting_bankroll * 100.0
    roi = (bankroll - starting_bankroll) / total_staked * 100.0 if total_staked else 0.0
    curve_df = pd.DataFrame(curve, columns=["date", "bankroll"])
    max_dd = _max_drawdown(curve_df["bankroll"]) if not curve_df.empty else 0.0

    return {
        "final_bankroll": bankroll,
        "final_pct": final_pct,
        "n_bets": n_bets,
        "win_rate": (wins / n_bets * 100.0) if n_bets else 0.0,
        "roi": roi,
        "max_drawdown": max_dd,
        "curve": curve_df,
        "used_real_odds": real_odds is not None,
    }


def _max_drawdown(series):
    """Largest peak-to-trough drop in a bankroll series, as a percentage."""
    if series.empty:
        return 0.0
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    return float(drawdown.min() * 100.0)


# ---------------------------------------------------------------------------
# PREDICTION ACCURACY  (how often the predicted winner actually won)
# ---------------------------------------------------------------------------

def tournament_accuracy(results_df, params, tournament="FIFA World Cup", year=None,
                        home_advantage=HOME_ADVANTAGE_DEFAULT):
    """
    Walk completed matches in date order (point-in-time, no lookahead) and, for
    every match in the chosen tournament/year, predict the team more likely to
    win and check whether it actually won.

    A draw counts as a miss (we predicted a team to win, nobody did).
    Returns: n (matches), correct, win_pct, and a per-match `rows` list.
    """
    ratings = {}
    n = correct = 0
    rows = []

    for row in results_df.itertuples(index=False):
        h, a = row.home_team, row.away_team
        rh = ratings.get(h, BASE_RATING)
        ra = ratings.get(a, BASE_RATING)
        neutral = _is_neutral(row.neutral)

        in_scope = (row.tournament == tournament and
                    (year is None or pd.Timestamp(row.date).year == year))
        if in_scope:
            ph, _pd, pa = match_probabilities_dc(rh, ra, params, neutral=neutral)
            predicted = h if ph >= pa else a

            if row.home_score > row.away_score:
                actual = h
            elif row.home_score < row.away_score:
                actual = a
            else:
                actual = None  # draw

            is_correct = predicted == actual
            n += 1
            correct += int(is_correct)
            rows.append({
                "Date": pd.Timestamp(row.date).date(),
                "Match": f"{h} vs {a}",
                "Predicted winner": predicted,
                "Result": "Correct" if is_correct else ("Draw" if actual is None else "Wrong"),
            })

        new_rh, new_ra = update_elo(rh, ra, row.home_score, row.away_score,
                                    tournament=row.tournament, neutral=neutral,
                                    home_advantage=home_advantage)
        ratings[h], ratings[a] = new_rh, new_ra

    return {
        "n": n, "correct": correct,
        "win_pct": (correct / n * 100.0) if n else 0.0,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# CHECKPOINTS  -- run with: python wc_model.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 1: data")
    df = load_results()
    train = get_training_data(df)
    print(f"  {len(df)} rows, {len(train)} completed, sorted={train['date'].is_monotonic_increasing}")

    print("=" * 60)
    print("PHASE 2: Elo  (top 8)")
    ratings, counts = compute_elo(train)
    for i, (t, r) in enumerate(sorted(ratings.items(), key=lambda kv: -kv[1])[:8], 1):
        print(f"  {i}. {t:20s} {r:7.1f}")

    print("=" * 60)
    print("PHASE 3: probabilities")
    for h, a in [("Spain", "Cape Verde"), ("Belgium", "Egypt")]:
        ph, pd_, pa = match_probabilities(ratings[h], ratings[a])
        print(f"  {h} vs {a}: H={ph:.0%} D={pd_:.0%} A={pa:.0%} (sum={ph+pd_+pa:.3f})")

    print("=" * 60)
    print("PHASE 3b: Dixon-Coles scoreline model")
    params = fit_scoreline_model(train)
    print(f"  fitted: a={params['a']:.3f} b={params['b']:.3f} h={params['h']:.3f} rho={params['rho']:.3f}")
    for h, a in [("Spain", "Cape Verde"), ("Belgium", "Egypt")]:
        r = predict_scoreline(ratings[h], ratings[a], params, neutral=True)
        print(f"  {h} vs {a}: H={r['p_home']:.0%} D={r['p_draw']:.0%} A={r['p_away']:.0%}"
              f"  | likely score {r['likely_score'][0]}-{r['likely_score'][1]}"
              f"  (xG {r['exp_home_goals']:.1f}-{r['exp_away_goals']:.1f})")

    print("=" * 60)
    print("INFERENCE: probability, confidence interval, p-value")
    _params, _cov, _ = fit_scoreline_model_with_cov(train)
    for h, a in [("Spain", "Cape Verde"), ("Belgium", "Egypt")]:
        res = predict_with_inference(ratings[h], ratings[a], _params, _cov,
                                     neutral=True, home_name=h, away_name=a)
        lo, hi = res["prob_ci"]
        pv = "<0.001" if res["p_value"] < 0.001 else f"{res['p_value']:.3f}"
        print(f"  {h} vs {a}: winner={res['winner']}  P(win)={res['winner_prob']:.0%}"
              f"  95% CI=[{lo:.0%},{hi:.0%}]  p={pv}"
              f"  ({'significant' if res['significant'] else 'not significant'})")

    print("=" * 60)
    print("PHASE 6: Kelly")
    print(f"  single_kelly(0.55, 2.10) = {single_kelly(0.55, 2.10):.4f}  (expect 0.1409)")

    print("=" * 60)
    print("PHASE 7: backtest on historical World Cup matches (synthetic odds)")
    bt = backtest(train, kelly_fraction=0.5)
    print(f"  bets placed : {bt['n_bets']}")
    print(f"  win rate    : {bt['win_rate']:.1f}%")
    print(f"  final return: {bt['final_pct']:+.1f}%   (synthetic odds -> expect near 0 / slightly negative)")
    print(f"  max drawdown: {bt['max_drawdown']:.1f}%")