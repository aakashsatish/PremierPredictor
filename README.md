# PremierPredictor

Predicts Premier League match outcomes as three-way probabilities — home win,
draw, away win — for every remaining fixture of the season.

## Results

Walk-forward backtest: for each season, the model is trained only on seasons
before it, then predicts that season cold. 2,280 matches across 2020/21–2025/26.

| | Accuracy | Log loss |
|---|---|---|
| Bet365 closing odds *(benchmark)* | 54.6% | 0.968 |
| **This model** | **52.8%** | **0.993** |
| Always predict a home win | 43.1% | 1.070 |

Log loss is the number that matters; accuracy is a poor guide when one class
wins 45% of the time. The model closes roughly three quarters of the distance
between knowing nothing and the bookmaker's price, without ever seeing odds.

Bookmaker odds are a **benchmark only** and are never a model input. Feeding
them in would raise the score while telling you nothing about whether the
football features work.

## Usage

```bash
pip install -r requirements.txt
cp .env.example .env        # add your ScraperAPI key
python update.py            # refresh results, rebuild, re-predict
```

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
| `model.py` | Model comparison and the walk-forward backtest |
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
  never "draw". That is a property of the sport, not a bug — draw probability
  peaks around a third. Use the probabilities, not the label.
- ~54% is close to the ceiling for football outcome prediction. The bookmaker
  manages 54.6% with vastly more information. Treat anything claiming much
  more with suspicion.
- Current-season accuracy in the report covers only a few dozen matches and is
  mostly noise. The walk-forward backtest is the real measure.
