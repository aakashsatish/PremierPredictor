"""Build the unified match dataset from the two sources.

football-data.co.uk supplies match statistics and bookmaker odds; fbref's
schedule pages supply matchweek numbers and, crucially, the forward fixture
list (football-data only publishes matches once they have been played).

Output is one row per match -- not the previous two-rows-per-match,
team-perspective layout. That layout is what made it possible to predict a
fixture without the home team's identity ever entering the model, and to
score an away row against the home team's prediction.
"""

from __future__ import annotations

import glob
import warnings

import numpy as np
import pandas as pd

import config
import scraper

warnings.filterwarnings("ignore")

# football-data column -> our name. Home/away pairs are unpacked together.
FD_STATS = {
    "FTHG": "home_goals", "FTAG": "away_goals",
    "HTHG": "home_ht_goals", "HTAG": "away_ht_goals",
    "HS": "home_shots", "AS": "away_shots",
    "HST": "home_sot", "AST": "away_sot",
    "HC": "home_corners", "AC": "away_corners",
    "HF": "home_fouls", "AF": "away_fouls",
    "HY": "home_yellow", "AY": "away_yellow",
    "HR": "home_red", "AR": "away_red",
    "B365H": "odds_home", "B365D": "odds_draw", "B365A": "odds_away",
}


def load_football_data() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(str(config.CACHE / "fd_*.csv"))):
        season = path.split("fd_")[1].replace(".csv", "")
        raw = pd.read_csv(path).dropna(subset=["HomeTeam"])
        df = pd.DataFrame(index=raw.index)
        df["season"] = season
        df["date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
        df["home"] = raw["HomeTeam"].map(config.normalise_team)
        df["away"] = raw["AwayTeam"].map(config.normalise_team)
        df["referee"] = raw["Referee"] if "Referee" in raw else None
        for src, dst in FD_STATS.items():
            df[dst] = pd.to_numeric(raw[src], errors="coerce") if src in raw else pd.NA
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["date", "home", "away"])


def load_fbref_schedule() -> pd.DataFrame:
    seasons = config.RESULTS_SEASONS + [config.CURRENT_SEASON]
    df = scraper.scrape_all_schedules(seasons)
    df["home"] = df["home_raw"].map(config.normalise_team)
    df["away"] = df["away_raw"].map(config.normalise_team)
    return df[[
        "season", "matchweek", "date", "time", "day", "home", "away",
        "home_goals", "away_goals", "attendance", "venue",
    ]]


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading football-data.co.uk (stats + odds) ...")
    fd = load_football_data()
    print(f"  {len(fd):,} matches")

    print("Loading fbref schedules (matchweek + fixtures) ...")
    fb = load_fbref_schedule()
    print(f"  {len(fb):,} matches ({fb.home_goals.notna().sum():,} played)")

    # fbref is the spine: it has every fixture, played or not.
    merged = fb.merge(
        fd.drop(columns=["date"]),
        on=["season", "home", "away"],
        how="left",
        suffixes=("", "_fd"),
        validate="one_to_one",
    )

    played = merged[merged["home_goals"].notna()].copy()

    # Cross-check: two independent sources should agree on every score.
    both = played[played["home_goals_fd"].notna()]
    disagree = both[
        (both["home_goals"] != both["home_goals_fd"])
        | (both["away_goals"] != both["away_goals_fd"])
    ]
    print(f"\nScore cross-check on {len(both):,} matches present in both sources:")
    print(f"  disagreements: {len(disagree)}")
    if len(disagree):
        print(disagree[["season", "date", "home", "away", "home_goals",
                        "home_goals_fd", "away_goals", "away_goals_fd"]].head(10).to_string(index=False))

    merged = merged.drop(columns=["home_goals_fd", "away_goals_fd"])

    # Result from the home team's perspective: H / D / A.
    merged["result"] = None
    m = merged["home_goals"].notna()
    merged.loc[m & (merged.home_goals > merged.away_goals), "result"] = "H"
    merged.loc[m & (merged.home_goals == merged.away_goals), "result"] = "D"
    merged.loc[m & (merged.home_goals < merged.away_goals), "result"] = "A"

    merged = merged.sort_values(["date", "time"]).reset_index(drop=True)
    matches = merged[merged["result"].notna()].copy()
    fixtures = merged[merged["result"].isna()].copy()
    return matches, fixtures


def build_current_season() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refresh only the current season, reusing committed history.

    Completed seasons never change, so re-deriving them from the HTML cache
    every run is wasted effort -- and impossible in CI, where the gitignored
    cache does not exist. This keeps the existing rows for earlier seasons
    and rebuilds just the current one, which costs two requests instead of
    twenty-two.
    """
    season = config.CURRENT_SEASON
    prior_matches = pd.read_csv(config.MATCHES_CSV, parse_dates=["date"])
    prior_matches = prior_matches[prior_matches["season"] != season]
    print(f"Keeping {len(prior_matches):,} matches from completed seasons")

    fb = scraper.scrape_schedule(season, force=False)
    fb["home"] = fb["home_raw"].map(config.normalise_team)
    fb["away"] = fb["away_raw"].map(config.normalise_team)
    fb = fb[["season", "matchweek", "date", "time", "day", "home", "away",
             "home_goals", "away_goals", "attendance", "venue"]]

    try:
        scraper.fetch_football_data(season)
        fd = load_football_data()
        fd = fd[fd["season"] == season].drop(columns=["date"])
        cur = fb.merge(fd, on=["season", "home", "away"], how="left",
                       suffixes=("", "_fd"), validate="one_to_one")
        cur = cur.drop(columns=[c for c in ("home_goals_fd", "away_goals_fd")
                                if c in cur.columns])
    except (RuntimeError, KeyError, ValueError) as e:
        # football-data publishes on a lag; fbref results alone are still
        # enough to keep fixtures and outcomes current.
        print(f"  football-data unavailable ({e}); using fbref results only.")
        cur = fb
    for col in prior_matches.columns:
        if col not in cur.columns:
            cur[col] = np.nan
    cur = cur[prior_matches.columns]

    cur["result"] = None
    m = cur["home_goals"].notna()
    cur.loc[m & (cur.home_goals > cur.away_goals), "result"] = "H"
    cur.loc[m & (cur.home_goals == cur.away_goals), "result"] = "D"
    cur.loc[m & (cur.home_goals < cur.away_goals), "result"] = "A"
    print(f"  {season}: {m.sum()} played, {(~m).sum()} upcoming")

    allm = pd.concat([prior_matches, cur], ignore_index=True)
    allm = allm.sort_values(["date", "time"]).reset_index(drop=True)
    return allm[allm["result"].notna()].copy(), allm[allm["result"].isna()].copy()


if __name__ == "__main__":
    import sys
    if "--full" in sys.argv:
        matches, fixtures = build()
    else:
        matches, fixtures = build_current_season()

    matches.to_csv(config.MATCHES_CSV, index=False)
    fixtures.to_csv(config.FIXTURES_CSV, index=False)

    print(f"\nWrote {config.MATCHES_CSV.name}: {len(matches):,} played matches")
    print(f"Wrote {config.FIXTURES_CSV.name}: {len(fixtures):,} upcoming fixtures")

    print("\nPer-season coverage:")
    cov = matches.groupby("season").agg(
        matches=("result", "size"),
        with_odds=("odds_home", lambda s: s.notna().sum()),
        with_shots=("home_shots", lambda s: s.notna().sum()),
        with_mw=("matchweek", lambda s: s.notna().sum()),
    )
    print(cov.to_string())

    print("\nResult distribution (home perspective):")
    print(matches["result"].value_counts(normalize=True).round(4).to_string())
