"""
app.py -- Streamlit UI for the World Cup 2026 predictor.

No math lives here -- it imports wc_model and arranges widgets. Predictions now
come from the Dixon-Coles scoreline model (a goals-based predictor that beats the
old Elo-split on held-out log loss and RPS).

Run:  streamlit run app.py
"""

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

import wc_model as m

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="wide")


# ---------------------------------------------------------------------------
# CACHED DATA + MODEL
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner="Loading history, rating teams, fitting model...")
def load_data(home_advantage):
    df = m.load_results()
    train = m.get_training_data(df)
    ratings, counts = m.compute_elo(train, home_advantage=home_advantage)
    params = m.fit_scoreline_model(train, home_advantage=home_advantage)
    return df, train, ratings, counts, params


@st.cache_data(ttl=1800, show_spinner="Replaying history for the track record...")
def run_backtest(home_advantage, kelly_fraction, draw_max):
    _, train, _, _, _ = load_data(home_advantage)
    return m.backtest(train, kelly_fraction=kelly_fraction,
                      home_advantage=home_advantage, draw_max=draw_max)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

st.sidebar.header("Controls")
bankroll = st.sidebar.slider("Bankroll for today ($)", 10, 10_000, 1_000, 10)
kelly_fraction = st.sidebar.slider("Risk level (fraction of Kelly)", 0.10, 1.00, 0.50, 0.05)
home_adv = st.sidebar.slider("Home advantage (Elo pts, host nations only)", 0, 150, 100, 10)
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

st.title("World Cup 2026 -- Predictions & Bet Sizer")
st.caption("Win probabilities from a Dixon-Coles goals model driven by Elo strength ratings.")

df, train, ratings, counts, params = load_data(float(home_adv))

# Reproducible demo date. For a live site use: dt.date.today()
today = dt.date(2026, 6, 15)

fixtures = m.get_fixtures_for_date(df, today)
if not fixtures.empty:
    day = today
    st.subheader(f"Fixtures for today -- {day:%A, %B %d, %Y}")
else:
    options = m.upcoming_wc_dates(df, today)
    if not options:
        st.info("No upcoming World Cup fixtures in the dataset. Try Refresh.")
        st.stop()
    day = st.selectbox("No unplayed games today -- pick a match day:",
                       options, format_func=lambda d: f"{d:%A, %B %d}")
    fixtures = m.get_fixtures_for_date(df, day)
    st.subheader(f"Fixtures for {day:%A, %B %d, %Y}")


# Build predictions (Dixon-Coles) + synthetic demo odds for each game.
rng = np.random.default_rng(int(pd.Timestamp(day).strftime("%Y%m%d")))
games, skipped = [], []
for r in fixtures.itertuples(index=False):
    if r.home_team not in ratings or r.away_team not in ratings:
        skipped.append(f"{r.home_team}/{r.away_team}")
        continue
    neutral = m._is_neutral(r.neutral)
    pred = m.predict_scoreline(ratings[r.home_team], ratings[r.away_team],
                               params, neutral=neutral)
    oh, od, oa = m._synthetic_odds(pred["p_home"], pred["p_draw"], pred["p_away"], 0.05, rng)
    games.append({
        "game": f"{r.home_team} vs {r.away_team}",
        "home_team": r.home_team, "away_team": r.away_team,
        "p_home": pred["p_home"], "p_draw": pred["p_draw"], "p_away": pred["p_away"],
        "likely_score": f"{pred['likely_score'][0]}-{pred['likely_score'][1]}",
        "xg": f"{pred['exp_home_goals']:.1f}-{pred['exp_away_goals']:.1f}",
        "odds_home": round(oh, 2), "odds_draw": round(od, 2), "odds_away": round(oa, 2),
    })

if skipped:
    st.warning("Skipped (no Elo history under this exact name): " + ", ".join(skipped))
if not games:
    st.stop()


# ---------------------------------------------------------------------------
# 1 -- PREDICTIONS  (percent chance of each outcome)
# ---------------------------------------------------------------------------

st.markdown("### 1 - Match predictions")
st.caption("Each row is the model's percent chance of a home win, draw, or away win, "
           "plus the single most likely scoreline and expected goals.")
pred_rows = []
for g in games:
    winner = [g["home_team"], "Draw", g["away_team"]][
        int(np.argmax([g["p_home"], g["p_draw"], g["p_away"]]))]
    pred_rows.append({"Match": g["game"], "Most likely": winner,
                      "P(home)": g["p_home"], "P(draw)": g["p_draw"], "P(away)": g["p_away"],
                      "Likely score": g["likely_score"], "xG": g["xg"]})
pred_df = pd.DataFrame(pred_rows)
st.dataframe(
    pred_df.style.format({"P(home)": "{:.0%}", "P(draw)": "{:.0%}", "P(away)": "{:.0%}"}),
    use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 2 -- ODDS (editable)
# ---------------------------------------------------------------------------

st.markdown("### 2 - Odds")
st.caption("Synthetic demo odds shown -- edit any cell with real decimal odds to find "
           "real edges. Skip this section entirely if you only want the predictions above.")
odds_df = pd.DataFrame([{"Match": g["game"], "Home": g["odds_home"],
                         "Draw": g["odds_draw"], "Away": g["odds_away"]} for g in games])
edited = st.data_editor(
    odds_df, use_container_width=True, hide_index=True, key="odds",
    column_config={c: st.column_config.NumberColumn(min_value=1.01, step=0.01, format="%.2f")
                   for c in ["Home", "Draw", "Away"]})
for g, (_, row) in zip(games, edited.iterrows()):
    g["odds_home"], g["odds_draw"], g["odds_away"] = row["Home"], row["Draw"], row["Away"]


# ---------------------------------------------------------------------------
# 3 -- BET ALLOCATION
# ---------------------------------------------------------------------------

st.markdown("### 3 - How much to bet on each game")
alloc = m.optimize_stakes(games, bankroll=float(bankroll), kelly_fraction=float(kelly_fraction))
total_staked = alloc["Stake"].sum()
n_bets = int((alloc["Stake"] > 0.005).sum())

c1, c2 = st.columns(2)
c1.metric("Recommended total stake", f"${total_staked:,.0f}")
c2.metric("Bets placed", f"{n_bets} of {len(alloc)}")

if n_bets == 0:
    st.success("No positive-edge bets today. With these odds the model can't beat the "
               "market, so the growth-optimal move is to bet nothing. (Expected with "
               "synthetic odds -- enter real odds to find real edges.)")
else:
    show = alloc.copy()
    show["Stake"] = show["Stake"].map(lambda v: f"${v:,.2f}")
    st.dataframe(show, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 4 -- TRACK RECORD (backtest)
# ---------------------------------------------------------------------------

st.markdown("### 4 - Track record: what % would you be up?")
bt = run_backtest(float(home_adv), float(kelly_fraction), 0.30)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Return", f"{bt['final_pct']:+.1f}%")
c2.metric("Bets", f"{bt['n_bets']}")
c3.metric("Win rate", f"{bt['win_rate']:.0f}%")
c4.metric("Max drawdown", f"{bt['max_drawdown']:.0f}%")
if not bt["curve"].empty:
    st.line_chart(bt["curve"].set_index("date")["bankroll"])

st.warning("This track record uses SYNTHETIC odds, so it's a machinery check, not a real "
           "edge -- a negative/near-zero result is expected and confirms no lookahead "
           "leakage. Pass real closing odds via backtest(real_odds=...) for a meaningful number.")

with st.expander("How the predictor works"):
    st.markdown("""
Win probabilities come from a **Dixon-Coles** model: each team's Elo strength rating
(built from ~150 years of internationals, with goal-difference weighting, tournament
importance, and home advantage) drives a Poisson model of the goals each side scores.
Home/draw/away probabilities are read off the full score matrix, with a low-score
correction. On held-out matches (2018+), this beats the simpler Elo-split on both log
loss and RPS -- modestly, because Elo was already strong. It also yields a most-likely
scoreline and expected goals. Not financial advice.
""")