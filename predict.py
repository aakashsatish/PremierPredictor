"""Generate predictions for the current season and write the report.

Outcome probabilities come from a blend of two models that disagree in useful
ways: a calibrated Random Forest over form and Elo features, and a Dixon-Coles
Poisson model over goals. The forest is stronger alone, so it carries most of
the weight. Predicted scorelines come from the Poisson grid, which the
classifier cannot produce at all.

Two training regimes are used on purpose:

  * Completed matches of the current season are scored by a model trained
    only on earlier seasons, so the accuracy quoted in the report is genuinely
    out-of-sample. The previous version reported accuracy over matches the
    model had trained on, and then matched two thirds of them to the wrong
    fixture via a date-less fallback lookup.

  * Upcoming fixtures are predicted by a model trained on everything played
    so far, including this season.

Predictions are joined by match identity, never by a name-only fallback.
"""

from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import config
import features
import model as M
from poisson_model import PoissonGoals

warnings.filterwarnings("ignore")

# Outcome probabilities blend the classifier with the goals model. The forest
# is the stronger of the two on its own (0.990 vs 1.004 log loss walk-forward),
# so it carries most of the weight; the blend measured 0.987, an improvement
# too small to call significant on six seasons but consistently in the right
# direction, which is the usual behaviour of combining decorrelated models.
POISSON_WEIGHT = 0.35
POISSON_XI = 0.002    # ~9 month half-life
POISSON_RIDGE = 1.0

OUT_CSV = config.DATA / f"predictions_{config.CURRENT_SEASON.replace('-', '_')}.csv"
OUT_TXT = config.ROOT / f"predictions_{config.CURRENT_SEASON.replace('-', '_')}.txt"

LABEL = {"H": "HOME WIN", "D": "DRAW", "A": "AWAY WIN"}


def confidence(p: float) -> str:
    if p >= 0.60:
        return "High"
    if p >= 0.45:
        return "Medium"
    return "Low"


def _blend(classifier, poisson, target: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Combine the two models' outcome probabilities."""
    rf = pd.DataFrame(
        classifier.predict_proba(target[cols]),
        columns=classifier.classes_, index=target.index,
    )[M.CLASSES]
    po = poisson.predict_proba(target)
    return POISSON_WEIGHT * po + (1 - POISSON_WEIGHT) * rf


def _likely_scores(poisson, target: pd.DataFrame, n: int = 3) -> pd.Series:
    """Most probable scorelines, e.g. '2-1 (11%), 1-1 (10%), 2-0 (9%)'.

    This is the one thing the classifier cannot do at all: it predicts a label,
    while the goals model carries a distribution over every scoreline.
    """
    out = {}
    for idx, m in target.iterrows():
        g = poisson.score_grid(m["home"], m["away"])
        flat = [(g[i, j], i, j) for i in range(6) for j in range(6)]
        flat.sort(reverse=True)
        out[idx] = ", ".join(f"{i}-{j} ({p:.0%})" for p, i, j in flat[:n])
    return pd.Series(out)


def build_predictions() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    matches = pd.read_csv(config.MATCHES_CSV, parse_dates=["date"])
    fixtures = pd.read_csv(config.FIXTURES_CSV, parse_dates=["date"])
    df = features.build(matches, fixtures)
    cols = features.feature_columns(df)

    season = config.CURRENT_SEASON
    played = df[(df["is_fixture"] == 0) & df["result"].notna()]

    # --- out-of-sample scoring of matches already played this season
    prior = played[played["season"] < season]
    current_played = played[played["season"] == season].copy()

    honest = M.make_models()[M.DEFAULT_MODEL]
    honest.fit(prior[cols], prior["result"])
    honest_po = PoissonGoals(xi=POISSON_XI, ridge=POISSON_RIDGE).fit(
        prior, ref_date=current_played["date"].min() if len(current_played) else None
    )
    if len(current_played):
        proba = _blend(honest, honest_po, current_played, cols)
        for c in M.CLASSES:
            current_played[f"p_{c}"] = proba[c]
        current_played["prediction"] = proba.idxmax(axis=1)
        current_played["correct"] = current_played["prediction"] == current_played["result"]

    # --- forward predictions, trained on everything available
    final = M.fit_final(df, cols)
    final_po = PoissonGoals(xi=POISSON_XI, ridge=POISSON_RIDGE).fit(played)
    upcoming = df[df["is_fixture"] == 1].copy()
    proba = _blend(final, final_po, upcoming, cols)
    for c in M.CLASSES:
        upcoming[f"p_{c}"] = proba[c]
    upcoming["likely_scores"] = _likely_scores(final_po, upcoming)
    upcoming["prediction"] = proba.idxmax(axis=1)
    # Consistent by construction: the label IS the argmax of these numbers.
    upcoming["top_prob"] = proba.max(axis=1)
    upcoming["confidence"] = upcoming["top_prob"].map(confidence)

    stats = {"n_prior": len(prior), "n_current": len(current_played),
             "n_upcoming": len(upcoming)}

    if len(current_played):
        from sklearn.metrics import log_loss
        y = current_played["result"]
        p = current_played[[f"p_{c}" for c in M.CLASSES]].values
        stats["accuracy"] = current_played["correct"].mean()
        stats["log_loss"] = log_loss(y, p, labels=M.CLASSES)
        stats["always_home"] = (y == "H").mean()
        mkt = current_played.dropna(subset=["odds_home", "odds_draw", "odds_away"])
        if len(mkt):
            mp = M.market_probabilities(mkt)
            stats["market_accuracy"] = (
                np.array(M.CLASSES)[mp.values.argmax(1)] == mkt["result"].values
            ).mean()
            stats["market_n"] = len(mkt)

    return current_played, upcoming, stats


def write_report(current_played, upcoming, stats) -> None:
    L = []
    add = L.append
    season_txt = config.CURRENT_SEASON.replace("-", "/")

    add("=" * 78)
    add(f"PREMIER LEAGUE PREDICTIONS - {season_txt}")
    add("=" * 78)
    add(f"Generated: {datetime.now():%Y-%m-%d %H:%M}")
    add("Model: calibrated Random Forest blended with a Dixon-Coles Poisson")
    add(f"       goals model ({1-POISSON_WEIGHT:.0%} / {POISSON_WEIGHT:.0%}), "
        f"three-way outcome (home / draw / away)")
    add(f"Trained on {stats['n_prior']:,} matches from earlier seasons")
    add("")
    add("Probabilities are model outputs and sum to 1. Bookmaker odds are used")
    add("only as a benchmark and are never an input to the model.")
    add("Scorelines come from the Poisson model's grid over every possible score.")
    add("")

    if len(current_played):
        add("-" * 78)
        add("RESULTS SO FAR THIS SEASON")
        add("-" * 78)
        add("Scored out-of-sample: these matches were predicted by a model")
        add("trained only on earlier seasons.")
        add("")
        for mw, grp in current_played.groupby("matchweek"):
            add(f"  Matchweek {int(mw)}")
            for _, m in grp.sort_values("date").iterrows():
                mark = "OK  " if m["correct"] else "MISS"
                add(f"    [{mark}] {m['date']:%d %b}  {m['home']} {int(m['home_goals'])}-"
                    f"{int(m['away_goals'])} {m['away']}")
                add(f"           predicted {LABEL[m['prediction']]:9s} "
                    f"(H {m['p_H']:.0%} / D {m['p_D']:.0%} / A {m['p_A']:.0%})"
                    f"   actual {LABEL[m['result']]}")
            add("")

        add(f"  Accuracy: {stats['accuracy']:.1%} "
            f"({int(stats['accuracy'] * stats['n_current'])}/{stats['n_current']})")
        add(f"  Log loss: {stats['log_loss']:.4f}")
        add(f"  For reference, always predicting a home win would score "
            f"{stats['always_home']:.1%}")
        if "market_accuracy" in stats:
            add(f"  Bet365 favourite over the same {stats['market_n']} matches: "
                f"{stats['market_accuracy']:.1%}")
        add("")
        add("  Note: a few dozen matches is far too small a sample to judge a")
        add("  model on. The walk-forward backtest in model.py is the real test.")
        add("")

    add("-" * 78)
    add(f"UPCOMING FIXTURES ({len(upcoming)})")
    add("-" * 78)
    add("")
    for mw, grp in upcoming.groupby("matchweek"):
        add(f"  Matchweek {int(mw)}")
        for _, m in grp.sort_values("date").iterrows():
            t = m["time"] if pd.notna(m["time"]) else "  :  "
            add(f"    {m['date']:%a %d %b} {t}  {m['home']} vs {m['away']}")
            add(f"        {LABEL[m['prediction']]:9s} [{m['confidence']}]"
                f"   H {m['p_H']:.0%} / D {m['p_D']:.0%} / A {m['p_A']:.0%}")
            add(f"        likely scores: {m['likely_scores']}")
        add("")

    add("=" * 78)
    add("SUMMARY")
    add("=" * 78)
    counts = upcoming["prediction"].value_counts()
    for c in ["H", "D", "A"]:
        add(f"  {LABEL[c]:9s} predicted in {counts.get(c, 0):3d} of {len(upcoming)} fixtures")
    add(f"  Confidence: " + ", ".join(
        f"{k} {v}" for k, v in upcoming["confidence"].value_counts().items()))
    add("")
    if counts.get("D", 0) == 0:
        add("")
        add("  No fixture lists DRAW as its single most likely outcome. That is")
        add("  expected rather than a fault: draw probability peaks around a third,")
        add("  so a draw is rarely the modal result even when it is well priced.")
        add("  The probabilities, not the label, are the useful output.")
    add("")
    add("  Mean probabilities across upcoming fixtures: "
        f"H {upcoming['p_H'].mean():.1%} / D {upcoming['p_D'].mean():.1%} / "
        f"A {upcoming['p_A'].mean():.1%}")
    add("=" * 78)

    OUT_TXT.write_text("\n".join(L))


if __name__ == "__main__":
    current_played, upcoming, stats = build_predictions()

    keep = ["season", "matchweek", "date", "time", "home", "away",
            "p_H", "p_D", "p_A", "prediction", "confidence", "likely_scores",
            "home_elo", "away_elo", "elo_diff"]
    upcoming[keep].to_csv(OUT_CSV, index=False)
    write_report(current_played, upcoming, stats)

    print(f"Trained on {stats['n_prior']:,} prior matches")
    if stats["n_current"]:
        print(f"\nCurrent season, scored out-of-sample ({stats['n_current']} matches):")
        print(f"  accuracy   {stats['accuracy']:.1%}")
        print(f"  log loss   {stats['log_loss']:.4f}")
        print(f"  always-home would score {stats['always_home']:.1%}")
        if "market_accuracy" in stats:
            print(f"  Bet365 favourite scores {stats['market_accuracy']:.1%} "
                  f"on the {stats['market_n']} matches with odds")
    print(f"\nWrote {OUT_CSV.name} ({len(upcoming)} fixtures)")
    print(f"Wrote {OUT_TXT.name}")
