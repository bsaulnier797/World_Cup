"""
app.py -- World Cup 2026 match predictor with statistical inference.

For each fixture it reports:
  - the predicted winner and their probability of winning
  - a confidence interval on that probability (estimation uncertainty)
  - a p-value testing whether the teams are genuinely unevenly matched
  - the expected goal-margin range (outcome uncertainty)

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
def load_model(home_advantage):
    df = m.load_results()
    train = m.get_training_data(df)
    ratings, _ = m.compute_elo(train, home_advantage=home_advantage)
    params, cov, _ = m.fit_scoreline_model_with_cov(train, home_advantage=home_advantage)
    return df, train, ratings, params, cov


@st.cache_data(ttl=1800, show_spinner="Scoring the model's track record...")
def accuracy(home_advantage):
    _df, train, _ratings, params, _cov = load_model(home_advantage)
    current = m.tournament_accuracy(train, params, year=2026, home_advantage=home_advantage)
    all_time = m.tournament_accuracy(train, params, year=None, home_advantage=home_advantage)
    return current, all_time


@st.cache_data(ttl=1800, show_spinner="Running inference...")
def predictions_for(home_advantage, ci, day_str):
    df, train, ratings, params, cov = load_model(home_advantage)
    fixtures = m.get_fixtures_for_date(df, day_str)
    out, skipped = [], []
    for r in fixtures.itertuples(index=False):
        if r.home_team not in ratings or r.away_team not in ratings:
            skipped.append(f"{r.home_team} / {r.away_team}")
            continue
        res = m.predict_with_inference(
            ratings[r.home_team], ratings[r.away_team], params, cov,
            neutral=m._is_neutral(r.neutral),
            home_name=r.home_team, away_name=r.away_team,
            ci=ci, n_samples=2000)
        out.append(res)
    return out, skipped


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

st.sidebar.header("Settings")
home_adv = st.sidebar.slider("Home advantage (Elo pts)", 0, 150, 100, 10,
    help="Applied only to host nations USA, Mexico, Canada at home.")
ci_pct = st.sidebar.select_slider("Confidence level", options=[90, 95, 99], value=95)
ci = ci_pct / 100.0
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()


# ---------------------------------------------------------------------------
# HEADER + DATE
# ---------------------------------------------------------------------------

st.title("⚽ World Cup 2026 — Who Wins, and How Sure Are We?")
st.caption("Win probability, a confidence interval, and a significance test for each match.")

df, train, ratings, params, cov = load_model(float(home_adv))

# --- model win rate, shown at the top ---
current, all_time = accuracy(float(home_adv))
if current["n"]:
    st.metric("Model win rate — 2026 World Cup",
              f"{current['win_pct']:.0f}%",
              help="Share of completed 2026 World Cup matches where the model's "
                   "predicted winner actually won. Draws count as misses.")
    st.caption(f"Correctly predicted the winner in **{current['correct']} of "
               f"{current['n']}** completed matches so far. "
               f"For context, across all {all_time['n']} World Cup matches in history "
               f"the model is right {all_time['win_pct']:.0f}% of the time.")
else:
    st.metric("Model win rate — World Cup (all-time)", f"{all_time['win_pct']:.0f}%",
              help="Share of historical World Cup matches the model called correctly.")
    st.caption(f"No completed 2026 matches yet — this is across all {all_time['n']} "
               f"World Cup matches in history.")

today = dt.date(2026, 6, 15)            # live site: dt.date.today()

if not m.get_fixtures_for_date(df, today).empty:
    day = today
    st.subheader(f"Today — {day:%A, %B %d, %Y}")
else:
    options = m.upcoming_wc_dates(df, today)
    if not options:
        st.info("No upcoming World Cup fixtures found. Try Refresh.")
        st.stop()
    day = st.selectbox("No unplayed games today — pick a match day:",
                       options, format_func=lambda d: f"{d:%A, %B %d}")
    st.subheader(f"{day:%A, %B %d, %Y}")

results, skipped = predictions_for(float(home_adv), ci, pd.Timestamp(day).strftime("%Y-%m-%d"))
if skipped:
    st.warning("No rating data for: " + ", ".join(skipped))
if not results:
    st.stop()


def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------------------
# SUMMARY TABLE
# ---------------------------------------------------------------------------

rows = []
for res in results:
    lo, hi = res["prob_ci"]
    rows.append({
        "Match": f"{res['home_name']} vs {res['away_name']}",
        "Predicted winner": res["winner"],
        "P(win)": res["winner_prob"],
        f"{ci_pct}% CI": f"{lo:.0%} – {hi:.0%}",
        "p-value": pfmt(res["p_value"]),
        "Significant?": "Yes" if res["significant"] else "No",
    })
st.dataframe(
    pd.DataFrame(rows).style.format({"P(win)": "{:.0%}"}),
    use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# PER-MATCH DETAIL
# ---------------------------------------------------------------------------

st.markdown("### Match detail")
for res in results:
    lo, hi = res["prob_ci"]
    ml, mh = res["margin_interval"]
    home, away = res["home_name"], res["away_name"]

    with st.container(border=True):
        st.markdown(f"#### {home} vs {away}")
        st.markdown(f"**Predicted winner: {res['winner']}**")

        c1, c2, c3 = st.columns(3)
        c1.metric(f"P({res['winner']} wins)", f"{res['winner_prob']:.0%}",
                  help="Point estimate from the model.")
        c2.metric(f"{ci_pct}% confidence interval", f"{lo:.0%} – {hi:.0%}",
                  help="How precisely we know that probability. Narrow because the "
                       "model is trained on ~49,000 matches. It does NOT mean the "
                       "match itself is near-certain.")
        c3.metric("p-value (H0: even match)", pfmt(res["p_value"]),
                  help="Two-sided test that the two teams are equally strong. "
                       "< 0.05 means the strength gap is statistically real.")

        if res["significant"]:
            st.success(f"Significant at the {100-ci_pct:.0f}% level (p < 0.05): "
                       f"{res['winner']} is genuinely the stronger side — though that "
                       f"is not the same as being certain to win.")
        else:
            st.info("Not significant (p ≥ 0.05): the two teams are too evenly matched "
                    "to call one genuinely stronger.")

        st.write(
            f"Full probabilities — **{home} {res['p_home']:.0%}**, "
            f"**Draw {res['p_draw']:.0%}**, **{away} {res['p_away']:.0%}**. "
            f"Most likely score **{res['likely_score'][0]}–{res['likely_score'][1]}** "
            f"(expected goals {res['exp_home_goals']:.1f} – {res['exp_away_goals']:.1f}).")

        st.caption(
            f"Expected goal margin ({home} − {away}), {ci_pct}% range: "
            f"**{ml:+d} to {mh:+d}** goals. This is the spread of plausible *results* — "
            f"the real uncertainty about who wins lives here and in the probabilities "
            f"above, not in the (narrow) confidence interval.")


# ---------------------------------------------------------------------------
# EXPLAINER
# ---------------------------------------------------------------------------

with st.expander("What these numbers mean (read this)"):
    st.markdown("""
There are **two different kinds of uncertainty** here, and they answer different
questions:

**1. How sure are we *who wins*?**
That's the **win probability** (and the **expected goal-margin range**). A 90% favourite
still loses 10% of the time — that is the match's own randomness and is irreducible.
This is the uncertainty that actually matters for predicting a result.

**2. How well do we *know* that probability?**
That's the **confidence interval** and the **p-value**. Because the model is fit on
~49,000 international matches, its parameters are pinned down very precisely, so:
- the confidence interval on the probability is **narrow** — a tight CI means we know
  the probability well, NOT that the match is a foregone conclusion;
- the p-value (testing H0: the teams are evenly matched) is **significant for almost any
  real strength gap**. "Significant" means the gap is statistically real, **not** that the
  outcome is certain.

**Method.** Win probabilities come from a Dixon-Coles goals model driven by Elo ratings.
The confidence interval and p-value are produced by sampling plausible model parameters
from their estimated covariance (the inverse Hessian of the likelihood at the fit) and
recomputing the prediction. The margin range comes from the predictive score distribution.

**One honest limitation:** the confidence interval reflects uncertainty in the *global
model parameters*, not in the two teams' individual Elo ratings — so for teams with very
few matches the true estimation uncertainty is somewhat larger than shown.
""")