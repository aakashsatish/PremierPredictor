"""Feature engineering.

Every feature here is computable strictly from information available before
kickoff. That is not a stylistic preference: the previous model's headline
accuracy came from a backtest that used each match's own rolling stats, while
production substituted the constant 0.5, so the reported number described a
model that was never shipped.

Two rules enforce it:
  * rolling form is always shifted by one match, so a row never sees itself;
  * Elo ratings are recorded *before* the match is played, then updated.

Upcoming fixtures get the team's current state -- the rolling average over
their most recent completed matches -- which is genuinely all that is known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Elo settings --------------------------------------------------------
ELO_START = 1500.0
ELO_PROMOTED = 1420.0   # promoted sides start below the mean; they underperform
ELO_K = 20.0
ELO_HFA = 65.0          # home advantage, in rating points
ELO_REGRESS = 0.75      # between seasons, pull 25% of the way back to the mean

ROLL_WINDOWS = (5, 10)

# Per-team, per-match quantities that rolling form is built from.
STAT_COLS = [
    "goals_for", "goals_against", "shots", "shots_against",
    "sot", "sot_against", "corners", "points", "won",
]


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match (two rows per match)."""
    def side(is_home: bool) -> pd.DataFrame:
        p, o = ("home", "away") if is_home else ("away", "home")
        out = pd.DataFrame({
            "match_id": df.index,
            "season": df["season"],
            "date": df["date"],
            "team": df[p],
            "opponent": df[o],
            "is_home": int(is_home),
            "goals_for": df[f"{p}_goals"],
            "goals_against": df[f"{o}_goals"],
            "shots": df.get(f"{p}_shots"),
            "shots_against": df.get(f"{o}_shots"),
            "sot": df.get(f"{p}_sot"),
            "sot_against": df.get(f"{o}_sot"),
            "corners": df.get(f"{p}_corners"),
        })
        gd = out["goals_for"] - out["goals_against"]
        out["points"] = np.select([gd > 0, gd == 0], [3.0, 1.0], default=0.0)
        out["won"] = (gd > 0).astype(float)
        out.loc[out["goals_for"].isna(), ["points", "won"]] = np.nan
        return out

    return pd.concat([side(True), side(False)], ignore_index=True)


def rolling_form(long_played: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-match pre-kickoff form, current form per team).

    The first is shifted so a match never sees its own result. The second is
    the unshifted state after each team's most recent match, which is what an
    upcoming fixture should be given.
    """
    long_played = long_played.sort_values(["team", "date"]).copy()
    g = long_played.groupby("team", sort=False)

    per_match = long_played[["match_id", "team", "is_home"]].copy()
    current = {}

    for w in ROLL_WINDOWS:
        for col in STAT_COLS:
            # shift(1) first: the window covers only strictly earlier matches.
            per_match[f"{col}_r{w}"] = g[col].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=2).mean()
            )
            current[f"{col}_r{w}"] = g[col].apply(
                lambda s, w=w: s.rolling(w, min_periods=2).mean().iloc[-1]
            )

    # Venue-specific form: how the side does at home / away specifically.
    for venue, label in ((1, "home"), (0, "away")):
        sub = long_played[long_played["is_home"] == venue]
        gv = sub.groupby("team", sort=False)
        ppg = gv["points"].transform(
            lambda s: s.shift(1).rolling(10, min_periods=3).mean()
        )
        per_match.loc[sub.index, f"venue_ppg_{label}"] = ppg
        current[f"venue_ppg_{label}"] = gv["points"].apply(
            lambda s: s.rolling(10, min_periods=3).mean().iloc[-1]
        )

    current_df = pd.DataFrame(current)
    current_df.index.name = "team"
    return per_match, current_df


def compute_elo(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Walk matches in date order, recording pre-match ratings then updating.

    Returns per-match (home_elo, away_elo) as they stood before kickoff, plus
    the final rating of every team for use on upcoming fixtures.
    """
    matches = matches.sort_values("date")
    first_season = matches["season"].min()
    ratings: dict[str, float] = {}
    prev_season = None
    rows = []

    for idx, m in matches.iterrows():
        season, home, away = m["season"], m["home"], m["away"]

        if prev_season is not None and season != prev_season:
            # Between seasons, regress everyone towards the mean.
            for t in ratings:
                ratings[t] = ELO_START + ELO_REGRESS * (ratings[t] - ELO_START)
        prev_season = season

        for t in (home, away):
            if t not in ratings:
                # Everyone starts level in the first season we have data for;
                # a club appearing later has just been promoted.
                ratings[t] = ELO_START if season == first_season else ELO_PROMOTED

        rh, ra = ratings[home], ratings[away]
        rows.append({"match_id": idx, "home_elo": rh, "away_elo": ra})

        hg, ag = m["home_goals"], m["away_goals"]
        if pd.isna(hg) or pd.isna(ag):
            continue

        expected_home = 1.0 / (1.0 + 10 ** ((ra - (rh + ELO_HFA)) / 400.0))
        score_home = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)

        # Scale K by margin of victory, so a 4-0 moves ratings more than a 1-0.
        margin = abs(hg - ag)
        k = ELO_K * (1.0 + np.log1p(max(margin - 1, 0)))

        delta = k * (score_home - expected_home)
        ratings[home] = rh + delta
        ratings[away] = ra - delta

    return pd.DataFrame(rows).set_index("match_id"), ratings


def add_rest_days(all_matches: pd.DataFrame) -> pd.DataFrame:
    """Days since each side's previous fixture. Uses the full schedule, so it
    is defined for upcoming fixtures too."""
    long_all = to_long(all_matches)[["match_id", "team", "date", "is_home"]]
    long_all = long_all.sort_values(["team", "date"])
    long_all["rest_days"] = (
        long_all.groupby("team", sort=False)["date"].diff().dt.days
    )
    home = long_all[long_all.is_home == 1].set_index("match_id")["rest_days"]
    away = long_all[long_all.is_home == 0].set_index("match_id")["rest_days"]
    out = pd.DataFrame(index=all_matches.index)
    out["home_rest_days"] = home.reindex(all_matches.index).clip(upper=30)
    out["away_rest_days"] = away.reindex(all_matches.index).clip(upper=30)
    return out


def head_to_head(all_matches: pd.DataFrame, n: int = 6) -> pd.Series:
    """Home side's average points in their last n meetings with this opponent."""
    # Each stored meeting is {team: points}, so the window counts meetings
    # rather than rows and the lookup works whichever side is at home.
    hist: dict[frozenset, list[dict]] = {}
    out = {}
    for idx, m in all_matches.sort_values("date").iterrows():
        key = frozenset((m["home"], m["away"]))
        prior = hist.get(key, [])
        pts = [meet[m["home"]] for meet in prior[-n:] if m["home"] in meet]
        out[idx] = float(np.mean(pts)) if pts else np.nan

        if pd.notna(m["home_goals"]):
            gd = m["home_goals"] - m["away_goals"]
            hp = 3.0 if gd > 0 else (1.0 if gd == 0 else 0.0)
            ap = 3.0 - hp if hp != 1.0 else 1.0
            hist.setdefault(key, []).append({m["home"]: hp, m["away"]: ap})
    return pd.Series(out, name="h2h_home_ppg").reindex(all_matches.index)


def build(matches: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Assemble the full feature frame for played matches and fixtures alike."""
    matches = matches.copy()
    fixtures = fixtures.copy()
    matches["is_fixture"] = 0
    fixtures["is_fixture"] = 1
    for c in ("home_goals", "away_goals"):
        fixtures[c] = np.nan

    allm = pd.concat([matches, fixtures], ignore_index=True)
    allm["date"] = pd.to_datetime(allm["date"])
    allm = allm.sort_values("date").reset_index(drop=True)

    played = allm[allm["home_goals"].notna()]

    # --- Elo
    elo, final_elo = compute_elo(allm)
    allm["home_elo"] = elo["home_elo"]
    allm["away_elo"] = elo["away_elo"]
    # Fixtures all sit after the last played match, so every team's rating at
    # that point is its final rating.
    fut = allm["is_fixture"] == 1
    allm.loc[fut, "home_elo"] = allm.loc[fut, "home"].map(final_elo)
    allm.loc[fut, "away_elo"] = allm.loc[fut, "away"].map(final_elo)
    allm["elo_diff"] = allm["home_elo"] - allm["away_elo"]
    allm["elo_expected_home"] = 1.0 / (
        1.0 + 10 ** ((allm["away_elo"] - (allm["home_elo"] + ELO_HFA)) / 400.0)
    )

    # --- rolling form
    long_played = to_long(played)
    per_match, current = rolling_form(long_played)
    form_cols = [c for c in per_match.columns if c not in ("match_id", "team", "is_home")]

    for side, flag in (("home", 1), ("away", 0)):
        side_form = per_match[per_match["is_home"] == flag].set_index("match_id")
        for c in form_cols:
            allm[f"{side}_{c}"] = side_form[c].reindex(allm.index)
        # Upcoming fixtures take each team's current state.
        for c in form_cols:
            allm.loc[fut, f"{side}_{c}"] = allm.loc[fut, side].map(current[c])

    # --- differentials: the model should see the gap, not just two levels
    for c in form_cols:
        if c.startswith("venue_ppg"):
            continue
        allm[f"diff_{c}"] = allm[f"home_{c}"] - allm[f"away_{c}"]
    allm["venue_ppg_diff"] = allm["home_venue_ppg_home"] - allm["away_venue_ppg_away"]

    # --- schedule context
    allm = allm.join(add_rest_days(allm))
    allm["rest_diff"] = allm["home_rest_days"] - allm["away_rest_days"]
    allm["h2h_home_ppg"] = head_to_head(allm)
    allm["hour"] = pd.to_datetime(
        allm["time"], format="%H:%M", errors="coerce"
    ).dt.hour.fillna(15)
    allm["dow"] = allm["date"].dt.dayofweek
    allm["matchweek"] = pd.to_numeric(allm["matchweek"], errors="coerce")

    # Newly promoted: in this season's division but not in the previous one.
    # Teams in the earliest season we hold are not flagged -- we simply have
    # no prior season to compare against.
    appearances = pd.concat([
        allm[["season", "home"]].rename(columns={"home": "team"}),
        allm[["season", "away"]].rename(columns={"away": "team"}),
    ])
    by_season = appearances.groupby("season")["team"].apply(set).to_dict()
    ordered = sorted(by_season)
    promoted = {
        s: by_season[s] - by_season[ordered[i - 1]] if i else set()
        for i, s in enumerate(ordered)
    }
    for side in ("home", "away"):
        allm[f"{side}_is_new"] = [
            int(t in promoted.get(s, set()))
            for t, s in zip(allm[side], allm["season"])
        ]

    return allm


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The columns the model trains on. Deliberately excludes odds."""
    cols = [
        "elo_diff", "elo_expected_home", "home_elo", "away_elo",
        "venue_ppg_diff", "home_venue_ppg_home", "away_venue_ppg_away",
        "home_rest_days", "away_rest_days", "rest_diff",
        "h2h_home_ppg", "matchweek", "hour", "dow",
        "home_is_new", "away_is_new",
    ]
    cols += [c for c in df.columns if c.startswith("diff_")]
    cols += [c for c in df.columns if c.startswith(("home_goals_for_r", "away_goals_for_r",
                                                    "home_points_r", "away_points_r"))]
    return [c for c in cols if c in df.columns]
