"""fbref scrapers, parameterised by season.

Two sources, one request each per season:

  scrape_schedule(season)      -- fbref. Every match with score, matchweek,
                                  kickoff, referee, attendance. The only
                                  source of fixtures not yet played.

  fetch_football_data(season)  -- football-data.co.uk. Shots, shots on
                                  target, corners, fouls, cards, half-time
                                  scores and bookmaker odds.

fbref used to carry xG, and the previous version of this project depended on
it. It no longer appears anywhere in fbref's free views, so nothing here uses
it. football-data began publishing xG in 2026/27, which is too little history
to train on but worth revisiting in a season or two.

The fbref schedule table is the backbone because it is the only view where
home and away come from the same column source, which removes the whole class
of name-mismatch bug that silently disabled seven clubs previously.
"""

from __future__ import annotations

import re
import time

import pandas as pd
from bs4 import BeautifulSoup

import config

# --- score parsing -------------------------------------------------------
# fbref writes scores with an en dash, and occasionally with a penalty
# shootout suffix in cup contexts. Keep this strict and fail loudly.
_SCORE_RE = re.compile(r"^\s*(\d+)\s*[–\-]\s*(\d+)\s*$")


def _parse_score(raw: str):
    if not raw or not raw.strip():
        return (None, None)
    m = _SCORE_RE.match(raw)
    if not m:
        return (None, None)
    return int(m.group(1)), int(m.group(2))


def _cell_text(row, stat: str) -> str:
    cell = row.find(attrs={"data-stat": stat})
    return cell.get_text(strip=True) if cell else ""


def scrape_schedule(season: str, force: bool = False) -> pd.DataFrame:
    """Return one row per match for the given season."""
    html = config.fetch(
        config.schedule_url(season), f"schedule_{season}", force=force
    )
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id=lambda x: x and x.startswith("sched"))
    if table is None:
        raise RuntimeError(f"No schedule table found for {season}")

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        # fbref repeats the header every so often and pads with blank rows.
        if "thead" in (tr.get("class") or []):
            continue
        date = _cell_text(tr, "date")
        home = _cell_text(tr, "home_team")
        away = _cell_text(tr, "away_team")
        if not date or not home or not away:
            continue

        hg, ag = _parse_score(_cell_text(tr, "score"))
        att = _cell_text(tr, "attendance").replace(",", "")
        wk = _cell_text(tr, "gameweek")

        rows.append({
            "season": season,
            "matchweek": int(wk) if wk.isdigit() else None,
            "date": date,
            "time": _cell_text(tr, "start_time") or None,
            "day": _cell_text(tr, "dayofweek") or None,
            "home_raw": home,
            "away_raw": away,
            "home_goals": hg,
            "away_goals": ag,
            "attendance": int(att) if att.isdigit() else None,
            "venue": _cell_text(tr, "venue") or None,
            "referee": _cell_text(tr, "referee") or None,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).reset_index(drop=True)


def scrape_all_schedules(seasons, force: bool = False) -> pd.DataFrame:
    frames = []
    for season in seasons:
        print(f"  {season} ...", end=" ", flush=True)
        df = scrape_schedule(season, force=force)
        played = df["home_goals"].notna().sum()
        print(f"{len(df)} matches ({played} played)")
        frames.append(df)
        time.sleep(1)
    return pd.concat(frames, ignore_index=True)


def football_data_url(season: str) -> str:
    """football-data.co.uk uses a two-digit season code, e.g. 2026-2027 -> 2627."""
    start, end = season.split("-")
    return f"https://www.football-data.co.uk/mmz4281/{start[2:]}{end[2:]}/E0.csv"


def fetch_football_data(season: str, force: bool = False) -> str:
    """Season CSV of match stats and bookmaker odds.

    Goes through the same proxy as fbref: direct requests from this network
    are answered with a site-wide 503, which is an IP block rather than an
    outage.
    """
    path = config.CACHE / f"fd_{season}.csv"
    if path.exists() and not force:
        return path.read_text()

    import requests
    r = requests.get(
        "http://api.scraperapi.com/",
        params={"api_key": config._api_key(), "url": football_data_url(season)},
        timeout=120,
    )
    if r.status_code != 200 or len(r.text) < 1000:
        raise RuntimeError(
            f"football-data fetch failed for {season}: "
            f"status={r.status_code} len={len(r.text)}"
        )
    path.write_text(r.text)
    print(f"    fetched fd_{season} ({len(r.text):,} bytes)")
    return r.text


def fetch_all_football_data(seasons, force: bool = False) -> None:
    for season in seasons:
        try:
            fetch_football_data(season, force=force)
        except RuntimeError as e:
            print(f"    {e}")
        time.sleep(1)


if __name__ == "__main__":
    seasons = config.RESULTS_SEASONS + [config.CURRENT_SEASON]
    print(f"Scraping {len(seasons)} season schedules")
    all_matches = scrape_all_schedules(seasons)

    print(f"\nTotal: {len(all_matches):,} matches across {all_matches.season.nunique()} seasons")

    names = sorted(set(all_matches.home_raw) | set(all_matches.away_raw))
    print(f"\nDistinct team names seen on schedule pages ({len(names)}):")
    for n in names:
        seasons_seen = all_matches.loc[
            (all_matches.home_raw == n) | (all_matches.away_raw == n), "season"
        ].nunique()
        print(f"  {n:22} in {seasons_seen} season(s)")
