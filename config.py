"""Central configuration: paths, seasons, and the fbref fetch layer.

Every network call in this project goes through fetch() so that caching and
credit accounting happen in exactly one place. fbref returns 403 to direct
requests, so ScraperAPI is mandatory, and it bills fbref at 5 credits per
request rather than 1 -- see CREDITS_PER_REQUEST.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = DATA / "raw_cache"
DATA.mkdir(exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

MATCHES_CSV = DATA / "matches.csv"
FIXTURES_CSV = DATA / "fixtures.csv"

# Season labels as fbref writes them in URLs.
def season_label(start_year: int) -> str:
    return f"{start_year}-{start_year + 1}"

# Results (cheap: one schedule page each) go back further than xG (20 team
# pages each), because Elo only needs scores while form features need xG.
RESULTS_SEASONS = [season_label(y) for y in range(2016, 2026)]  # 2016-17 .. 2025-26
XG_SEASONS = [season_label(y) for y in range(2020, 2026)]       # 2020-21 .. 2025-26
CURRENT_SEASON = season_label(2026)                             # 2026-27

CREDITS_PER_REQUEST = 5  # fbref is a ScraperAPI "protected domain"

FBREF = "https://fbref.com"
COMP_ID = 9  # Premier League


def schedule_url(season: str) -> str:
    return (
        f"{FBREF}/en/comps/{COMP_ID}/{season}/schedule/"
        f"{season}-Premier-League-Scores-and-Fixtures"
    )


def _api_key() -> str:
    key = os.environ.get("SCRAPERAPI_KEY", "").strip()
    if not key:
        # Minimal .env reader; avoids a hard dependency on python-dotenv.
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line.startswith("SCRAPERAPI_KEY=") and not line.startswith("#"):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        raise RuntimeError(
            "No SCRAPERAPI_KEY found. Copy .env.example to .env and add your key."
        )
    return key


def fetch(url: str, cache_name: str, force: bool = False, retries: int = 3) -> str:
    """Return page HTML, reading from the on-disk cache when possible.

    Cached pages cost nothing, so re-running the pipeline during development is
    free. Pass force=True only when you genuinely want fresh data.
    """
    path = CACHE / f"{cache_name}.html"
    if path.exists() and not force:
        return path.read_text()

    key = _api_key()
    last = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(
                "http://api.scraperapi.com/",
                params={"api_key": key, "url": url},
                timeout=120,
            )
            if r.status_code == 200 and len(r.text) > 5000:
                path.write_text(r.text)
                print(f"    fetched {cache_name} ({len(r.text):,} bytes, "
                      f"~{CREDITS_PER_REQUEST} credits)")
                return r.text
            last = f"status={r.status_code} len={len(r.text)}"
        except requests.RequestException as e:
            last = str(e)
        print(f"    attempt {attempt}/{retries} failed for {cache_name}: {last}")
        if attempt < retries:
            time.sleep(5 * attempt)
    raise RuntimeError(f"Could not fetch {url}: {last}")


def credits_remaining() -> dict:
    r = requests.get(
        "http://api.scraperapi.com/account",
        params={"api_key": _api_key()},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# --- team name canonicalisation -----------------------------------------
# The two sources spell clubs differently, and fbref is even inconsistent
# with itself between page types. Everything is mapped to one canonical
# name here. normalise_team() RAISES on an unknown name rather than falling
# back to a default -- the previous version's silent .get(name, default)
# is exactly how seven clubs ended up with a hardcoded 0.3 win rate.

CANONICAL_TEAMS = {
    # football-data.co.uk spellings
    "Cardiff": "Cardiff City", "Coventry": "Coventry City", "Hull": "Hull City",
    "Ipswich": "Ipswich Town", "Leeds": "Leeds United", "Leicester": "Leicester City",
    "Luton": "Luton Town", "Man City": "Manchester City", "Man United": "Manchester United",
    "Norwich": "Norwich City", "Nott'm Forest": "Nottingham Forest",
    "Stoke": "Stoke City", "Swansea": "Swansea City",
    # fbref schedule-page spellings
    "Manchester Utd": "Manchester United", "Newcastle": "Newcastle United",
    "Nottingham": "Nottingham Forest",
    # fbref team-page spellings (kept so old matches.csv can still be read)
    "Brighton and Hove Albion": "Brighton", "Newcastle Utd": "Newcastle United",
    "Nott'ham Forest": "Nottingham Forest", "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves",
    "West Bromwich Albion": "West Brom", "Sheffield Utd": "Sheffield United",
    "Huddersfield Town": "Huddersfield",
}

# Names that are already canonical.
_CANONICAL_SET = {
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Burnley",
    "Cardiff City", "Chelsea", "Coventry City", "Crystal Palace", "Everton",
    "Fulham", "Huddersfield", "Hull City", "Ipswich Town", "Leeds United",
    "Leicester City", "Liverpool", "Luton Town", "Manchester City",
    "Manchester United", "Middlesbrough", "Newcastle United", "Norwich City",
    "Nottingham Forest", "Sheffield United", "Southampton", "Stoke City",
    "Sunderland", "Swansea City", "Tottenham", "Watford", "West Brom",
    "West Ham", "Wolves",
}


def normalise_team(name: str) -> str:
    """Map any source spelling to the canonical club name.

    Raises on anything unrecognised. A promoted club that neither source has
    seen before should surface as a loud failure at scrape time, not as a
    silently wrong feature value at prediction time.
    """
    if name is None:
        raise ValueError("normalise_team got None")
    name = str(name).strip()
    if name in _CANONICAL_SET:
        return name
    if name in CANONICAL_TEAMS:
        return CANONICAL_TEAMS[name]
    raise KeyError(
        f"Unknown team name {name!r}. Add it to config.CANONICAL_TEAMS "
        f"(or _CANONICAL_SET if it is already canonical)."
    )
