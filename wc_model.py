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
    print("PHASE 6: Kelly")
    print(f"  single_kelly(0.55, 2.10) = {single_kelly(0.55, 2.10):.4f}  (expect 0.1409)")

    print("=" * 60)
    print("PHASE 7: backtest on historical World Cup matches (synthetic odds)")
    bt = backtest(train, kelly_fraction=0.5)
    print(f"  bets placed : {bt['n_bets']}")
    print(f"  win rate    : {bt['win_rate']:.1f}%")
    print(f"  final return: {bt['final_pct']:+.1f}%   (synthetic odds -> expect near 0 / slightly negative)")
    print(f"  max drawdown: {bt['max_drawdown']:.1f}%")