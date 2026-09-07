# PremierPredictor

**Live dashboard: https://aakashsatish.github.io/PremierPredictor/**

Predicts Premier League match outcomes as three-way probabilities — home win,
draw, away win — for every remaining fixture of the season.

## Results

Walk-forward backtest: for each season, the model is trained only on seasons
before it, then predicts that season cold. 2,280 matches across 2020/21–2025/26.

| | Accuracy | Log loss |
|---|---|---|
| Bet365 closing odds *(benchmark)* | 54.6% | 0.968 |
| **Ensemble (shipped)** | **53.0%** | **0.987** |
| Random Forest alone | 53.2% | 0.990 |
| Poisson goals model alone | 50.7% | 1.004 |
| Always predict a home win | 43.1% | 1.070 |

The ensemble's edge over the forest alone is real in direction but small — it
wins 4 of 6 seasons, which is not distinguishable from chance. It is shipped
because combining decorrelated models reliably helps on average, and because
the Poisson half supplies scorelines regardless.

Log loss is the number that matters; accuracy is a poor guide when one class
wins 45% of the time. The model closes roughly three quarters of the distance
between knowing nothing and the bookmaker's price, without ever seeing odds.

Bookmaker odds are a **benchmark only** and are never a model input. Feeding
them in would raise the score while telling you nothing about whether the
football features work.

## Usage

```bash
pip install -r requirements.txt
cp .env.example .env                    # add your ScraperAPI key
git config core.hooksPath .githooks     # blocks committing credentials
python update.py                        # refresh results, rebuild, re-predict
```

A GitHub Actions workflow runs the same refresh every Tuesday and commits
the results, so the predictions stay current without anyone running anything.
It needs a `SCRAPERAPI_KEY` repository secret.

`update.py` re-fetches only the current season (~10 credits). Earlier seasons
come from the on-disk cache and cost nothing.

Outputs: `data/predictions_<season>.csv` and a readable
`predictions_<season>.txt` grouped by matchweek.

## How it works

| File | Role |
|---|---|
| `config.py` | Paths, seasons, cached fetch layer, canonical team names |
| `scraper.py` | fbref schedules and football-data.co.uk season CSVs |
| `dataset.py` | Joins both sources into one row per match |
| `features.py` | Elo, rolling form, rest days, head-to-head |
| `model.py` | Classifier comparison and the walk-forward backtest |
| `poisson_model.py` | Dixon-Coles goals model; scorelines and team ratings |
| `predict.py` | Trains the final model and writes the report |
| `update.py` | Entry point for a weekly refresh |

### Data

3,828 played matches, 2016/17 to date, from two independent sources that agree
on all 3,820 scores they share.

- **fbref** — schedules, matchweeks, and the forward fixture list. The only
  source for matches not yet played.
- **football-data.co.uk** — shots, shots on target, corners, fouls, cards,
  half-time scores, bookmaker odds.

Both need the ScraperAPI proxy: fbref returns 403 to direct requests and
football-data returns a site-wide 503.

> fbref removed xG from its free tier at some point after January 2026 — the
> attribute is gone from team pages and match logs alike. Nothing here depends
> on it. football-data started publishing xG in 2026/27, which is too little
> history to train on but worth revisiting later.

### The two models

**Random Forest** over the features below. Strong, but opaque, and it can only
emit a label plus three probabilities.

**Dixon-Coles Poisson** models goals instead of outcomes. Each club gets an
attack and a defence rating; the two expected-goal figures produce a grid over
every scoreline, and home/draw/away come from summing below, on, and above its
diagonal. It fits in about a second and is readable — on current data it puts
home advantage at ×1.19, Arsenal's defence at 0.64 (opponents score 36% below
their norm) and Ipswich's at 1.33. It also yields correct-score probabilities,
which the classifier cannot produce.

Its fitted low-score correction (rho = −0.075) confirms that real football
produces more 0-0 and 1-1 draws than plain Poisson predicts.

### Features

All computed strictly from information available before kickoff. Rolling
windows are shifted by one match; Elo is recorded before the match, then
updated.

- **Elo ratings** — margin-scaled updates, home advantage, regression to the
  mean between seasons. Promoted clubs enter below average, which is what lets
  the model say anything sensible about sides with no recent top-flight record.
- **Rolling form for both teams** over 5 and 10 matches — goals, shots, shots
  on target, corners, points, win rate — plus the differential between sides.
- **Venue-specific form** — the home side's home record, the away side's away record.
- **Rest days**, **head-to-head**, **matchweek**, **kickoff hour**, **promoted flag**.

## Known limits

- A draw is never the single most likely outcome, so the predicted *label* is
  never "draw". This was tested, not assumed. Across the 300 most draw-likely
  matches in the backtest (mean draw probability 28.5%), draws occurred 25.7%
  of the time while the better of home/away came in at 41.5%. Forcing draw
  predictions costs accuracy at every threshold — −0.13% at p(draw) ≥ 0.30,
  −4.5% at ≥ 0.26, −8.7% at ≥ 0.24. The draw probabilities themselves are
  well calibrated (22.8% predicted vs 23.6% actual), so use them; it is only
  the collapse to a single label that discards the information.
- ~54% is close to the ceiling for football outcome prediction. The bookmaker
  manages 54.6% with vastly more information. Treat anything claiming much
  more with suspicion.
- Current-season accuracy in the report covers only a few dozen matches and is
  mostly noise. The walk-forward backtest is the real measure.
