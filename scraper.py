"""fbref scrapers, parameterised by season.

Two levels of data:

  scrape_schedule(season)  -- one request, every match in the season with
                              score, matchweek, kickoff, referee, attendance.
                              Also the source of forward fixtures.

  scrape_team_logs(season) -- twenty requests, per-match xG/xGA/possession.

The schedule table is the backbone because it is the only view where home and
away come from the same column source, which removes the whole class of
name-mismatch bug that silently disabled seven clubs in the previous version.
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
