"""Three-way match outcome model (home win / draw / away win).

Two deliberate departures from the previous version:

  * The target is three-way. The old model predicted only win vs not-win,
    which is why draw probability had to be invented afterwards as a flat
    0.15 for every fixture in the file.

  * Evaluation is walk-forward and scored against the bookmaker. Accuracy
    alone is close to meaningless here -- always predicting a home win
    scores about 44% -- so log loss carries the real signal, and Bet365's
    closing odds provide the benchmark any honest model has to be measured
    against. Odds are never used as an input, only as a yardstick.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# sklearn's log_loss sorts labels internally, so probability columns must be
# in sorted order. Keeping CLASSES sorted everywhere avoids a silent column
# permutation that makes a good model look worse than a coin flip.
CLASSES = ["A", "D", "H"]  # sorted; do not reorder


def market_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Bookmaker odds -> probabilities, with the overround divided out."""
    inv = pd.DataFrame({
        "A": 1.0 / df["odds_away"],
        "D": 1.0 / df["odds_draw"],
        "H": 1.0 / df["odds_home"],
    })[CLASSES]
    return inv.div(inv.sum(axis=1), axis=0)


DEFAULT_MODEL = "RandomForest"


def make_models() -> dict:
    """Candidate models, all compared honestly in the walk-forward backtest.

    RandomForest is wrapped in a sigmoid calibrator: a forest's vote shares
    are not probabilities, and calibrating them is what makes the reported
    draw and win percentages mean what they say. Gradient boosting overfits
    badly on a few thousand rows even with early stopping, and is kept only
    so the comparison stays visible.
    """
    return {
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            CalibratedClassifierCV(
                RandomForestClassifier(
                    n_estimators=400, min_samples_leaf=20, max_features="sqrt",
                    random_state=42, n_jobs=-1,
                ),
                method="sigmoid", cv=3,
            ),
        ),
        "LogisticRegression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=0.3, multi_class="multinomial"),
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=600, learning_rate=0.04, min_samples_leaf=40,
            l2_regularization=2.0, early_stopping=True,
            validation_fraction=0.18, n_iter_no_change=30, random_state=42,
        ),
    }


def _score(name, y, proba, classes) -> dict:
    proba = pd.DataFrame(proba, columns=classes)[CLASSES].values
    pred = np.array(CLASSES)[proba.argmax(axis=1)]
    return {
        "model": name,
        "n": len(y),
        "accuracy": accuracy_score(y, pred),
        "log_loss": log_loss(y, proba, labels=CLASSES),
    }


def walk_forward(df: pd.DataFrame, feature_cols: list[str],
                 test_seasons: list[str]) -> pd.DataFrame:
    """Train on everything before each test season, predict that season.

    No information from the test season -- or any later one -- reaches the
    model that predicts it.
    """
    played = df[(df["is_fixture"] == 0) & df["result"].notna()].copy()
    rows = []

    for season in test_seasons:
        train = played[played["season"] < season]
        test = played[played["season"] == season]
        if len(train) < 500 or test.empty:
            continue

        Xtr, ytr = train[feature_cols], train["result"]
        Xte, yte = test[feature_cols], test["result"]

        for name, model in make_models().items():
            model.fit(Xtr, ytr)
            proba = model.predict_proba(Xte)
            r = _score(name, yte, proba, list(model.classes_))
            r["season"] = season
            rows.append(r)

        # Baselines, scored on exactly the same matches.
        # Base rates in CLASSES order: away, draw, home.
        always_home = np.tile([[0.321, 0.233, 0.446]], (len(yte), 1))
        r = _score("baseline: always home", yte, always_home, CLASSES)
        r["season"] = season
        rows.append(r)

        mkt = test[["odds_home", "odds_draw", "odds_away"]].dropna()
        if len(mkt) > 0:
            mp = market_probabilities(test.loc[mkt.index])
            r = _score("benchmark: Bet365", test.loc[mkt.index, "result"],
                       mp.values, CLASSES)
            r["season"] = season
            rows.append(r)

    return pd.DataFrame(rows)


def fit_final(df: pd.DataFrame, feature_cols: list[str], model_name: str | None = None):
    """Train the chosen model on every completed match."""
    played = df[(df["is_fixture"] == 0) & df["result"].notna()]
    model = make_models()[model_name or DEFAULT_MODEL]
    model.fit(played[feature_cols], played["result"])
    return model


if __name__ == "__main__":
    import warnings
    import config
    import features

    warnings.filterwarnings("ignore")

    matches = pd.read_csv(config.MATCHES_CSV, parse_dates=["date"])
    fixtures = pd.read_csv(config.FIXTURES_CSV, parse_dates=["date"])
    df = features.build(matches, fixtures)
    cols = features.feature_columns(df)

    test_seasons = [f"{y}-{y+1}" for y in range(2020, 2026)]
    print(f"Walk-forward backtest over {len(test_seasons)} seasons, "
          f"{len(cols)} features\n")

    res = walk_forward(df, cols, test_seasons)

    print("=== per season (log loss; lower is better) ===")
    piv = res.pivot_table(index="model", columns="season", values="log_loss")
    print(piv.round(4).to_string())

    print("\n=== pooled across all test seasons ===")
    agg = (res.groupby("model")
              .apply(lambda g: pd.Series({
                  "matches": int(g["n"].sum()),
                  "accuracy": np.average(g["accuracy"], weights=g["n"]),
                  "log_loss": np.average(g["log_loss"], weights=g["n"]),
              }))
              .sort_values("log_loss"))
    print(agg.round(4).to_string())
