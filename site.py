#!/usr/bin/env python3
"""Render the dashboard at docs/index.html.

A static page generated from the same files the pipeline writes, so the
weekly refresh updates the site as a side effect of updating the data.

Layout follows the FotMob pattern: a centre column of match rows grouped
under collapsible date headers, a rail either side, and pills for moving
between matchweeks. Past matchweeks show the score and whether the call was
right; future ones show the call and its probability. One list, one control.

Match rows are shipped as JSON and rendered on the client. Rendering 380
fixtures plus 380 results as HTML grew the page by roughly 700 bytes per
match; as data it is about a fifth of that.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime

import pandas as pd

import config
from poisson_model import PoissonGoals

warnings.filterwarnings("ignore")

SITE = config.ROOT / "docs"
SITE.mkdir(exist_ok=True)

# Three-letter codes for the monogram badges. Real crests need licensed
# assets; these give the row the same shape without them.
TEAM_ABBR = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE",
    "Brighton": "BHA", "Burnley": "BUR", "Cardiff City": "CAR", "Chelsea": "CHE",
    "Coventry City": "COV", "Crystal Palace": "CRY", "Everton": "EVE", "Fulham": "FUL",
    "Huddersfield": "HUD", "Hull City": "HUL", "Ipswich Town": "IPS", "Leeds United": "LEE",
    "Leicester City": "LEI", "Liverpool": "LIV", "Luton Town": "LUT",
    "Manchester City": "MCI", "Manchester United": "MUN", "Middlesbrough": "MID",
    "Newcastle United": "NEW", "Norwich City": "NOR", "Nottingham Forest": "NFO",
    "Sheffield United": "SHU", "Southampton": "SOU", "Stoke City": "STK",
    "Sunderland": "SUN", "Swansea City": "SWA", "Tottenham": "TOT", "Watford": "WAT",
    "West Brom": "WBA", "West Ham": "WHU", "Wolves": "WOL",
}


# Short forms for rows and rails, the way FotMob and the BBC set them.
SHORT_NAME = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nottm Forest",
    "Ipswich Town": "Ipswich", "Leeds United": "Leeds", "Coventry City": "Coventry",
    "Hull City": "Hull", "Sheffield United": "Sheffield Utd", "Leicester City": "Leicester",
    "Cardiff City": "Cardiff", "Luton Town": "Luton", "Norwich City": "Norwich",
    "Stoke City": "Stoke", "Swansea City": "Swansea",
}


def short(team: str) -> str:
    return SHORT_NAME.get(team, team)


def hue(name: str) -> int:
    h = 0
    for ch in name:
        h = (h * 31 + ord(ch)) % 360
    return h


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def crest(team: str, small: bool = False) -> str:
    code = TEAM_ABBR.get(team, team[:3].upper())
    return (f'<span class="crest{" sm" if small else ""}" style="--h:{hue(team)}" '
            f'aria-hidden="true">{code}</span>')


# --------------------------------------------------------------- data prep
def league_table(cur: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in sorted(set(cur.home) | set(cur.away)):
        h, a = cur[cur.home == t], cur[cur.away == t]
        w = int((h.home_goals > h.away_goals).sum() + (a.away_goals > a.home_goals).sum())
        d = int((h.home_goals == h.away_goals).sum() + (a.away_goals == a.home_goals).sum())
        gf = int(h.home_goals.sum() + a.away_goals.sum())
        ga = int(h.away_goals.sum() + a.home_goals.sum())
        pl = len(h) + len(a)
        rows.append(dict(team=t, played=pl, won=w, drew=d, lost=pl - w - d,
                         gf=gf, ga=ga, gd=gf - ga, pts=3 * w + d))
    tab = pd.DataFrame(rows).sort_values(["pts", "gd", "gf"], ascending=False)
    tab.insert(0, "pos", range(1, len(tab) + 1))
    return tab


def recent_form(cur: pd.DataFrame, team: str, n: int = 5) -> list[str]:
    played = cur[(cur.home == team) | (cur.away == team)].sort_values("date")
    out = []
    for _, m in played.tail(n).iterrows():
        gf, ga = (m.home_goals, m.away_goals) if m.home == team else (m.away_goals, m.home_goals)
        out.append("W" if gf > ga else ("D" if gf == ga else "L"))
    return out


# ------------------------------------------------------------------ build
def build() -> str:
    matches_df = pd.read_csv(config.MATCHES_CSV, parse_dates=["date"])
    preds = pd.read_csv(
        config.DATA / f"predictions_{config.CURRENT_SEASON.replace('-', '_')}.csv",
        parse_dates=["date"],
    )
    summary = json.loads((config.DATA / "summary.json").read_text())

    season = config.CURRENT_SEASON
    season_txt = season.replace("-", "/")
    cur = matches_df[matches_df.season == season]
    table = league_table(cur)

    pm = PoissonGoals(xi=0.002, ridge=1.0).fit(matches_df)
    ratings = pm.ratings()
    elo = preds.groupby("home")["home_elo"].first()

    # --- every match of the season as one list: played ones carry the score
    #     and verdict, upcoming ones carry the call.
    kickoff = {
        (r.date.strftime("%Y-%m-%d"), r.home, r.away): (r.time if pd.notna(r.time) else None)
        for _, r in cur.iterrows()
    }
    matches: list[dict] = []
    for _, r in preds.iterrows():
        matches.append({
            "mw": int(r.matchweek), "date": r.date.strftime("%Y-%m-%d"),
            "time": r.time if pd.notna(r.time) else None,
            "h": r.home, "a": r.away,
            "pH": round(float(r.p_H), 3), "pD": round(float(r.p_D), 3), "pA": round(float(r.p_A), 3),
            "pred": r.prediction, "conf": r.confidence,
            "scores": r.likely_scores if pd.notna(r.likely_scores) else None,
            "eh": int(round(r.home_elo)), "ea": int(round(r.away_elo)),
            "played": False,
        })
    for res in summary["results"]:
        matches.append({
            "mw": int(res["matchweek"]), "date": res["date"],
            "time": kickoff.get((res["date"], res["home"], res["away"])),
            "h": res["home"], "a": res["away"],
            "pH": res["pH"], "pD": res["pD"], "pA": res["pA"],
            "pred": res["prediction"], "played": True,
            "hg": res["hg"], "ag": res["ag"],
            "result": res["result"], "correct": res["correct"],
        })

    # --- per-matchweek summary for the pill strip and nav header
    mws: dict[int, dict] = {}
    for mw in range(1, 39):
        ms = [m for m in matches if m["mw"] == mw]
        if not ms:
            continue
        ds = sorted(m["date"] for m in ms)
        d0, d1 = datetime.strptime(ds[0], "%Y-%m-%d"), datetime.strptime(ds[-1], "%Y-%m-%d")
        label = f"{d0:%d %b}" if d0 == d1 else f"{d0:%d %b} to {d1:%d %b}"
        mws[mw] = {"label": label, "n": len(ms), "played": sum(m["played"] for m in ms)}

    next_mw = int(preds.matchweek.min()) if len(preds) else max(mws)

    pills = "\n".join(
        f'<button class="pill mwp{" done" if info["played"] == info["n"] else ""}" '
        f'data-mw="{mw}" type="button">MW {mw}</button>'
        for mw, info in mws.items()
    )

    # --- table rail
    table_rows = []
    for _, r in table.iterrows():
        form = "".join(f'<i class="f{f}" title="{f}"></i>' for f in recent_form(cur, r.team))
        table_rows.append(
            f'<tr><td class="pos">{r.pos}</td>'
            f'<td class="tm">{crest(r.team, True)}<span>{esc(short(r.team))}</span></td>'
            f'<td class="n">{r.played}</td><td class="n">{r.gd:+d}</td>'
            f'<td class="n pts">{r.pts}</td><td class="form">{form}</td></tr>'
        )

    # --- strength rail
    power = []
    for t in sorted(set(preds.home) | set(cur.home)):
        rt = ratings.loc[t] if t in ratings.index else None
        power.append(dict(team=t, elo=float(elo.get(t, 1500)),
                          atk=float(rt["attack"]) if rt is not None else 1.0,
                          dfn=float(rt["defence"]) if rt is not None else 1.0))
    power.sort(key=lambda x: -x["elo"])
    emax, emin = max(p["elo"] for p in power), min(p["elo"] for p in power)
    power_rows = []
    for p in power:
        w = 8 + 92 * (p["elo"] - emin) / max(emax - emin, 1)
        power_rows.append(
            f'<tr><td class="tm">{crest(p["team"], True)}<span>{esc(short(p["team"]))}</span></td>'
            f'<td class="elo"><span class="track"><span class="fill" style="width:{w:.0f}%"></span></span>'
            f'<span class="n">{p["elo"]:.0f}</span></td>'
            f'<td class="n dim">{p["atk"]:.2f}</td><td class="n dim">{p["dfn"]:.2f}</td></tr>'
        )

    bt = summary["backtest"]
    hits = sum(1 for r in summary["results"] if r["correct"])
    mkt = summary.get("season_market_accuracy")

    data = {
        "season": season_txt, "nextMw": next_mw, "mws": mws,
        "abbr": TEAM_ABBR, "short": SHORT_NAME, "matches": matches,
    }
    data_json = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")

    gen = datetime.strptime(summary["generated"], "%Y-%m-%d %H:%M")
    subs = {
        "SEASON": season_txt,
        "GENERATED": f"{gen:%d %b, %H:%M}",
        "PLAYED": str(summary["played"]), "UPCOMING": str(summary["upcoming"]),
        "TRAINED": f'{summary["trained_on"]:,}',
        "NEXTMW": str(next_mw), "PILLS": pills,
        "TABLEROWS": "\n".join(table_rows), "POWERROWS": "\n".join(power_rows),
        "ACC": f'{summary["season_accuracy"]:.0%}',
        "ACCN": f'{hits} of {summary["played"]} correct',
        "MKTACC": f"{mkt:.0%}" if mkt else "n/a", "MKTN": str(summary["season_market_n"]),
        "ALWAYSHOME": f'{summary["season_always_home"]:.0%}',
        "BTACC": f'{bt["model_accuracy"]:.0%}', "BTN": f'{bt["matches"]:,}',
        "BTSEASONS": bt["seasons"].replace("-", " to "),
        "BTMACC": f'{bt["market_accuracy"]:.0%}', "BTBACC": f'{bt["baseline_accuracy"]:.0%}',
        "DATA": data_json,
    }
    html = TEMPLATE
    for k, v in subs.items():
        html = html.replace("__%s__" % k, v)
    return html


TEMPLATE = r"""<title>Premier Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&display=swap">
<style>
:root{
  --bg:#171717; --card:#262626; --card2:#2d2d2d; --pill:#363636;
  --pill-on:#ededed; --pill-on-text:#171717;
  --text:#f3f3f3; --muted:#9d9d9d; --faint:#6e6e6e; --line:#333333;
  --accent:#43d17a;
  --home:#E0A050; --draw:#9AA8B0; --away:#5BB2C6;
  --home-fill:#D9922F; --draw-fill:#6E7E87; --away-fill:#2E8AA1;
  --good:#63B183; --bad:#D4715F;
  --crest-s:36%; --crest-l:30%;
  --shadow:0 1px 2px rgba(0,0,0,.35), 0 8px 24px rgba(0,0,0,.22);
}
@media (prefers-color-scheme:light){
  :root:not([data-theme="dark"]){
    --bg:#f1f2f4; --card:#ffffff; --card2:#f6f7f9; --pill:#e7e9ed;
    --pill-on:#1b1b1b; --pill-on-text:#ffffff;
    --text:#141414; --muted:#6b7280; --faint:#9aa0a8; --line:#e8eaee;
    --accent:#15803d;
    --home:#A8630F; --draw:#5F6E77; --away:#166274;
    --good:#2F6B45; --bad:#9C3B2E;
    --crest-s:42%; --crest-l:40%;
    --shadow:0 1px 2px rgba(0,0,0,.05), 0 8px 24px rgba(0,0,0,.06);
  }
}
:root[data-theme="light"]{
  --bg:#f1f2f4; --card:#ffffff; --card2:#f6f7f9; --pill:#e7e9ed;
  --pill-on:#1b1b1b; --pill-on-text:#ffffff;
  --text:#141414; --muted:#6b7280; --faint:#9aa0a8; --line:#e8eaee;
  --accent:#15803d;
  --home:#A8630F; --draw:#5F6E77; --away:#166274;
  --good:#2F6B45; --bad:#9C3B2E;
  --crest-s:42%; --crest-l:40%;
  --shadow:0 1px 2px rgba(0,0,0,.05), 0 8px 24px rgba(0,0,0,.06);
}

*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font:500 14px/1.5 Manrope,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased}
button{font:inherit;color:inherit}
h1,h2{margin:0;text-wrap:balance}
.n{font-variant-numeric:tabular-nums}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:6px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}

/* ---------- top bar ---------- */
.top{position:sticky;top:0;z-index:10;background:var(--bg);border-bottom:1px solid var(--line)}
.top .in{max-width:1380px;margin:0 auto;padding:12px 20px;display:flex;align-items:center;gap:18px}
.brand{display:flex;align-items:center;gap:11px;font-weight:800;font-size:17px;letter-spacing:-.01em;
  text-decoration:none;color:var(--text)}
.brand img{width:34px;height:34px;border-radius:9px;display:block}
.brand small{display:block;font-weight:600;font-size:11px;color:var(--muted);letter-spacing:.02em}
.topnav{display:flex;gap:6px;margin-left:auto}
.topnav a{text-decoration:none}
.meta{font-size:12px;color:var(--muted);white-space:nowrap}
@media (max-width:720px){.meta{display:none}}

/* ---------- pills (the bubbles) ---------- */
.pill{display:inline-flex;align-items:center;gap:7px;padding:8px 14px;border-radius:999px;
  background:var(--pill);color:var(--text);border:0;cursor:pointer;font-weight:700;font-size:13px;
  white-space:nowrap;line-height:1;transition:background .12s}
.pill:hover{background:var(--card2)}
.pill.on{background:var(--pill-on);color:var(--pill-on-text)}
.pill .dot{width:7px;height:7px;border-radius:50%;background:var(--accent)}
.pill.done{color:var(--muted)}
.pill.done.on{color:var(--pill-on-text)}

/* ---------- layout ---------- */
.layout{max-width:1380px;margin:0 auto;padding:18px 20px 70px;display:grid;
  grid-template-columns:300px minmax(0,1fr) 320px;gap:18px;align-items:start}
.rail{display:flex;flex-direction:column;gap:18px;position:sticky;top:76px}
.center{display:flex;flex-direction:column;gap:14px;min-width:0}
@media (max-width:1180px){
  .layout{grid-template-columns:minmax(0,1fr) 320px}
  .rail.left{grid-column:1/-1;position:static;display:grid;grid-template-columns:1fr 1fr;align-items:start}
}
@media (max-width:860px){
  .layout{grid-template-columns:1fr;padding:12px 12px 60px}
  .rail,.rail.left{position:static;display:flex;flex-direction:column}
}

/* ---------- cards ---------- */
.card{background:var(--card);border-radius:18px;box-shadow:var(--shadow);overflow:hidden}
.hd{display:flex;align-items:center;justify-content:space-between;gap:12px;
  padding:15px 18px;font-weight:800;font-size:15px}
.hd .sub{font-weight:600;font-size:12px;color:var(--muted)}
details.grp>summary{list-style:none;cursor:pointer;user-select:none}
details.grp>summary::-webkit-details-marker{display:none}
.chev{width:9px;height:9px;border-right:2px solid var(--muted);border-bottom:2px solid var(--muted);
  transform:rotate(45deg);transition:transform .15s;margin-right:4px}
details.grp:not([open]) .chev{transform:rotate(-135deg)}

/* ---------- matchweek nav ---------- */
.mwnav{display:flex;align-items:center;justify-content:space-between;padding:14px 14px 8px}
.mwnav .arrow{width:34px;height:34px;border-radius:50%;background:var(--pill);border:0;cursor:pointer;
  display:grid;place-items:center}
.mwnav .arrow:disabled{opacity:.3;cursor:default}
.mwnav .arrow::before{content:"";width:8px;height:8px;border-left:2px solid var(--text);
  border-bottom:2px solid var(--text);transform:rotate(45deg);margin-left:3px}
.mwnav .arrow.next::before{transform:rotate(-135deg);margin-left:-3px}
.mwnav .ttl{text-align:center}
.mwnav .ttl strong{display:block;font-size:16px;font-weight:800}
.mwnav .ttl span{font-size:12px;color:var(--muted)}
.strip{display:flex;gap:8px;overflow-x:auto;padding:6px 14px 14px;scrollbar-width:none}
.strip::-webkit-scrollbar{display:none}

/* ---------- match rows ---------- */
.rows{border-top:1px solid var(--line)}
.match{display:grid;grid-template-columns:60px minmax(0,1fr) 118px minmax(0,1fr) 104px;
  align-items:center;gap:10px;padding:12px 16px;cursor:pointer;border-top:1px solid var(--line)}
.match:first-child{border-top:0}
.match:hover{background:var(--card2)}
.status{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.team{display:flex;align-items:center;gap:10px;min-width:0;font-weight:700;font-size:14px}
.team .name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.team.home{justify-content:flex-end;text-align:right}
.crest{width:30px;height:30px;border-radius:50%;flex:0 0 auto;display:grid;place-items:center;
  background:hsl(var(--h) var(--crest-s) var(--crest-l));color:#fff;font-weight:800;font-size:10px;
  letter-spacing:.03em}
.crest.sm{width:22px;height:22px;font-size:8px}
.mid{text-align:center}
.mid .big{font-weight:800;font-size:19px;letter-spacing:-.01em;line-height:1.1;font-variant-numeric:tabular-nums}
.mid .lbl{font-size:10px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-top:2px}
.bar{display:flex;height:5px;border-radius:3px;overflow:hidden;background:var(--line);margin:7px auto 0;width:104px}
.seg{display:block;height:100%}
.seg.home{background:var(--home-fill)} .seg.draw{background:var(--draw-fill)} .seg.away{background:var(--away-fill)}
.call{display:flex;flex-direction:column;align-items:flex-end;gap:4px;text-align:right}
.call small{font-size:10.5px;color:var(--faint);font-weight:600}
.tag{display:inline-block;padding:4px 9px;border-radius:999px;font-weight:800;font-size:10px;
  letter-spacing:.06em;text-transform:uppercase;line-height:1;white-space:nowrap}
.tag.ft{background:var(--pill);color:var(--muted)}
.tag.home{background:color-mix(in srgb,var(--home-fill) 22%,transparent);color:var(--home)}
.tag.draw{background:color-mix(in srgb,var(--draw-fill) 26%,transparent);color:var(--draw)}
.tag.away{background:color-mix(in srgb,var(--away-fill) 22%,transparent);color:var(--away)}
.tag.high{background:color-mix(in srgb,var(--accent) 20%,transparent);color:var(--accent)}
.tag.medium{background:var(--pill);color:var(--text)}
.tag.low{background:var(--pill);color:var(--muted)}
.tag.hit{background:color-mix(in srgb,var(--good) 22%,transparent);color:var(--good)}
.tag.miss{background:color-mix(in srgb,var(--bad) 22%,transparent);color:var(--bad)}

.detail{display:none;padding:2px 16px 14px;background:var(--card2);border-top:1px solid var(--line)}
.match.open+.detail{display:block}
.probs{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 8px}
.probs .p{padding:6px 11px;border-radius:999px;font-weight:700;font-size:12px;background:var(--pill)}
.probs .p.home{color:var(--home)} .probs .p.draw{color:var(--draw)} .probs .p.away{color:var(--away)}
.detail .ln{font-size:12.5px;color:var(--muted);margin-top:4px}
.detail .ln b{color:var(--text);font-weight:700}
.empty{padding:28px 18px;color:var(--muted);text-align:center}

@media (max-width:640px){
  .match{grid-template-columns:minmax(0,1fr) 104px minmax(0,1fr);
    grid-template-areas:"status status call" "home mid away";row-gap:8px;padding:12px 12px}
  .status{grid-area:status}.call{grid-area:call;flex-direction:row;align-items:center;gap:8px}
  .team.home{grid-area:home}.mid{grid-area:mid}.team.away{grid-area:away}
  .team{font-size:13px;flex-direction:column;gap:6px;text-align:center}
  .team.home{justify-content:flex-start;text-align:center}
  .team .name{white-space:normal;line-height:1.2}
  .crest{order:-1}
  .detail{padding:2px 12px 12px}
}

/* ---------- rails ---------- */
table{width:100%;border-collapse:collapse;font-size:13px}
th{font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);
  text-align:right;padding:4px 10px 8px;white-space:nowrap}
th:first-child,th.l{text-align:left}
td{padding:7px 8px;border-top:1px solid var(--line);text-align:right;white-space:nowrap}
td.tm{text-align:left;font-weight:700;display:flex;align-items:center;gap:8px;max-width:160px}
td.tm span:last-child{overflow:hidden;text-overflow:ellipsis}
td.pos{color:var(--faint);width:26px;text-align:left}
td.pts{font-weight:800}
td.dim{color:var(--muted)}
.form{display:flex;gap:3px;justify-content:flex-end}
.form i{width:8px;height:8px;border-radius:50%;display:block}
.fW{background:var(--good)} .fD{background:var(--draw-fill)} .fL{background:var(--bad)}
td.elo{display:flex;align-items:center;gap:8px;text-align:left}
.track{flex:1;height:6px;border-radius:3px;background:var(--line);overflow:hidden;min-width:50px}
.fill{display:block;height:100%;background:var(--accent);opacity:.75;border-radius:3px}
td.elo .n{color:var(--muted);font-size:12px;width:32px;text-align:right}

.stat{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 18px;
  border-top:1px solid var(--line)}
.stat .k{font-weight:700}
.stat .s{font-size:12px;color:var(--muted)}
.stat .v{font-weight:800;font-size:22px;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.stat.lead .v{color:var(--accent)}
.note{padding:12px 18px 16px;font-size:12.5px;color:var(--muted);line-height:1.5;border-top:1px solid var(--line)}
.note b{color:var(--text);font-weight:700}
.legend{display:flex;gap:14px;padding:0 18px 14px;font-size:12px;color:var(--muted);flex-wrap:wrap}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
footer{max-width:1380px;margin:0 auto;padding:0 20px 40px;font-size:12px;color:var(--faint)}
</style>

<header class="top"><div class="in">
  <a class="brand" href="#">
    <img src="logo.svg" alt="">
    <span>Premier Predictor<small>__SEASON__</small></span>
  </a>
  <nav class="topnav">
    <a class="pill" href="#matches">Matches</a>
    <a class="pill" href="#table">Table</a>
    <a class="pill" href="#model">Model</a>
  </nav>
  <span class="meta">Updated __GENERATED__</span>
</div></header>

<div class="layout">

  <aside class="rail left">
    <section class="card" id="table">
      <div class="hd">Table <span class="sub">__PLAYED__ played</span></div>
      <table>
        <thead><tr><th></th><th class="l">Team</th><th>P</th><th>GD</th><th>Pts</th><th>Form</th></tr></thead>
        <tbody>__TABLEROWS__</tbody>
      </table>
    </section>
    <section class="card">
      <div class="hd">Strength <span class="sub">Elo, attack, defence</span></div>
      <table>
        <thead><tr><th class="l">Team</th><th class="l">Elo</th><th>Att</th><th>Def</th></tr></thead>
        <tbody>__POWERROWS__</tbody>
      </table>
      <div class="note">Attack is goals scored against an average side and defence is goals
      conceded, so for defence lower is better. 1.00 is league average.</div>
    </section>
  </aside>

  <main class="center" id="matches">
    <section class="card">
      <div class="mwnav">
        <button class="arrow prev" id="prev" type="button" aria-label="Previous matchweek"></button>
        <div class="ttl"><strong id="mwttl">Matchweek __NEXTMW__</strong><span id="mwrng"></span></div>
        <button class="arrow next" id="next" type="button" aria-label="Next matchweek"></button>
      </div>
      <div class="strip" id="strip">__PILLS__</div>
    </section>
    <div id="list"></div>
  </main>

  <aside class="rail right">
    <section class="card" id="model">
      <div class="hd">How the model is doing <span class="sub">this season</span></div>
      <div class="stat lead"><div><div class="k">Model</div><div class="s">__ACCN__</div></div><div class="v">__ACC__</div></div>
      <div class="stat"><div><div class="k">Bookmaker favourite</div><div class="s">Bet365, __MKTN__ matches with odds</div></div><div class="v">__MKTACC__</div></div>
      <div class="stat"><div><div class="k">Always pick home</div><div class="s">no model at all</div></div><div class="v">__ALWAYSHOME__</div></div>
      <div class="stat"><div><div class="k">Backtest</div><div class="s">__BTN__ matches, __BTSEASONS__</div></div><div class="v">__BTACC__</div></div>
      <div class="note"><b>A few months of matches is a small sample.</b> The backtest is the better
      measure: over __BTN__ matches the model scored __BTACC__, the bookmaker __BTMACC__ and always
      picking the home side __BTBACC__. Odds are never used as an input.</div>
    </section>
    <section class="card">
      <div class="hd">Reading the rows</div>
      <div class="legend">
        <span><i style="background:var(--home-fill)"></i>Home win</span>
        <span><i style="background:var(--draw-fill)"></i>Draw</span>
        <span><i style="background:var(--away-fill)"></i>Away win</span>
      </div>
      <div class="note" style="border-top:0;padding-top:0">The big number is the model's probability
      for its pick. The bar underneath splits home, draw and away. Tap a row for the full breakdown
      and the most likely scorelines. Draw is rarely the single most likely result, so the pick is
      almost always home or away even when a draw is well priced.</div>
    </section>
  </aside>
</div>

<footer>Trained on __TRAINED__ matches. Refreshed every Tuesday from fbref and football-data.co.uk.
Predictions are model output, not advice.</footer>

<script id="d" type="application/json">__DATA__</script>
<script>
(function(){
  const D = JSON.parse(document.getElementById('d').textContent);
  const L = {H:'Home win', D:'Draw', A:'Away win'};
  const K = {H:'home', D:'draw', A:'away'};
  const list = document.getElementById('list');
  const strip = document.getElementById('strip');
  const prev = document.getElementById('prev'), next = document.getElementById('next');
  const mwKeys = Object.keys(D.mws).map(Number).sort((a,b)=>a-b);
  let cur = D.nextMw;

  const hue = s => { let h = 0; for (const c of s) h = (h*31 + c.charCodeAt(0)) % 360; return h; };
  const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const ab  = n => D.abbr[n] || n.slice(0,3).toUpperCase();
  const nm  = n => D.short[n] || n;
  const crest = n => `<span class="crest" style="--h:${hue(n)}" aria-hidden="true">${ab(n)}</span>`;
  const pct = x => Math.round(x*100) + '%';
  const bar = m => `<div class="bar" role="img" aria-label="Home ${pct(m.pH)}, draw ${pct(m.pD)}, away ${pct(m.pA)}">` +
    `<span class="seg home" style="width:${m.pH*100}%"></span>` +
    `<span class="seg draw" style="width:${m.pD*100}%"></span>` +
    `<span class="seg away" style="width:${m.pA*100}%"></span></div>`;
  const longDate = s => new Date(s + 'T12:00:00').toLocaleDateString('en-GB', {weekday:'long', day:'numeric', month:'long'});

  function row(m){
    const top = Math.max(m.pH, m.pD, m.pA);
    let status, mid, call;
    if (m.played){
      status = `<span class="tag ft">FT</span>`;
      mid = `<div class="big">${m.hg} - ${m.ag}</div>${bar(m)}`;
      call = `<span class="tag ${m.correct ? 'hit' : 'miss'}">${m.correct ? 'Correct' : 'Missed'}</span>` +
             `<small>called ${L[m.pred]} ${pct(top)}</small>`;
    } else {
      status = `<span>${m.time || 'TBC'}</span>`;
      mid = `<div class="big">${pct(top)}</div><div class="lbl">${L[m.pred]}</div>${bar(m)}`;
      call = `<span class="tag ${m.conf.toLowerCase()}">${m.conf}</span><small>confidence</small>`;
    }
    const detail =
      `<div class="probs">` +
        `<span class="p home">Home ${pct(m.pH)}</span>` +
        `<span class="p draw">Draw ${pct(m.pD)}</span>` +
        `<span class="p away">Away ${pct(m.pA)}</span></div>` +
      (m.scores ? `<div class="ln">Most likely scores: <b>${esc(m.scores)}</b></div>` : '') +
      (m.eh ? `<div class="ln">Elo going in: <b>${m.eh}</b> v <b>${m.ea}</b></div>` : '') +
      (m.played ? `<div class="ln">Result was <b>${L[m.result]}</b>. The model called ${L[m.pred]} at ${pct(top)}, so it was <b>${m.correct ? 'right' : 'wrong'}</b>.</div>` : '');
    return `<div class="match" role="button" tabindex="0" aria-expanded="false">` +
      `<div class="status">${status}</div>` +
      `<div class="team home"><span class="name">${esc(nm(m.h))}</span>${crest(m.h)}</div>` +
      `<div class="mid">${mid}</div>` +
      `<div class="team away">${crest(m.a)}<span class="name">${esc(nm(m.a))}</span></div>` +
      `<div class="call">${call}</div></div><div class="detail">${detail}</div>`;
  }

  function render(mw){
    cur = mw;
    history.replaceState(null, '', '#mw' + mw);
    const ms = D.matches.filter(m => m.mw === mw)
      .sort((a,b) => (a.date + (a.time||'')) < (b.date + (b.time||'')) ? -1 : 1);
    const groups = {};
    for (const m of ms) (groups[m.date] = groups[m.date] || []).push(m);
    const info = D.mws[mw];
    document.getElementById('mwttl').textContent = 'Matchweek ' + mw;
    document.getElementById('mwrng').textContent = info ? info.label + (info.played === info.n ? ' · played' : info.played ? ` · ${info.played} of ${info.n} played` : '') : '';
    list.innerHTML = Object.keys(groups).sort().map(d =>
      `<details class="card grp" open><summary class="hd"><span>${longDate(d)}</span><span class="chev"></span></summary>` +
      `<div class="rows">${groups[d].map(row).join('')}</div></details>`).join('')
      || `<div class="card empty">No fixtures scheduled for this matchweek yet.</div>`;
    strip.querySelectorAll('.mwp').forEach(p => {
      const on = +p.dataset.mw === mw;
      p.classList.toggle('on', on);
      p.setAttribute('aria-current', on ? 'true' : 'false');
      if (on) strip.scrollTo({left: p.offsetLeft - strip.clientWidth/2 + p.offsetWidth/2, behavior:'smooth'});
    });
    prev.disabled = mw <= mwKeys[0];
    next.disabled = mw >= mwKeys[mwKeys.length-1];
  }

  strip.addEventListener('click', e => { const p = e.target.closest('.mwp'); if (p) render(+p.dataset.mw); });
  prev.addEventListener('click', () => render(mwKeys[Math.max(0, mwKeys.indexOf(cur)-1)]));
  next.addEventListener('click', () => render(mwKeys[Math.min(mwKeys.length-1, mwKeys.indexOf(cur)+1)]));
  const toggle = el => { const o = el.classList.toggle('open'); el.setAttribute('aria-expanded', o); };
  list.addEventListener('click', e => { const r = e.target.closest('.match'); if (r) toggle(r); });
  list.addEventListener('keydown', e => {
    const r = e.target.closest('.match');
    if (r && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); toggle(r); }
  });
  document.addEventListener('keydown', e => {
    if (e.target.closest('input,textarea')) return;
    if (e.key === 'ArrowLeft') prev.click(); else if (e.key === 'ArrowRight') next.click();
  });

  const h = /^#mw(\d+)$/.exec(location.hash);
  render(h && D.mws[h[1]] ? +h[1] : D.nextMw);
})();
</script>
"""

DOC = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Three-way Premier League match predictions, refreshed weekly.">
<meta name="color-scheme" content="dark light">
<link rel="icon" type="image/svg+xml" href="favicon.svg">
<link rel="apple-touch-icon" href="logo.svg">
%s
</head>
<body>%s</body>
</html>
"""


if __name__ == "__main__":
    fragment = build()
    (SITE / "index.html").write_text(DOC % ("", fragment))
    (SITE / "fragment.html").write_text(fragment)
    print(f"Wrote {SITE/'index.html'} ({len((SITE/'index.html').read_text()):,} bytes)")
