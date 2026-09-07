#!/usr/bin/env python3
"""Render the dashboard at docs/index.html.

A static page generated from the same files the pipeline writes, so the
weekly refresh updates the site as a side effect of updating the data.
Everything is inlined -- no fetches, no build step, no server.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import config
from poisson_model import PoissonGoals

warnings.filterwarnings("ignore")

SITE = config.ROOT / "docs"
SITE.mkdir(exist_ok=True)

OUTCOME = {"H": ("home", "Home"), "D": ("draw", "Draw"), "A": ("away", "Away")}


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


# ------------------------------------------------------------- components
def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def prob_bar(ph: float, pd_: float, pa: float) -> str:
    label = f"Home {ph:.0%}, draw {pd_:.0%}, away {pa:.0%}"
    return (
        f'<div class="bar" role="img" aria-label="{label}">'
        f'<span class="seg home" style="width:{ph*100:.1f}%"></span>'
        f'<span class="seg draw" style="width:{pd_*100:.1f}%"></span>'
        f'<span class="seg away" style="width:{pa*100:.1f}%"></span></div>'
    )


def fixture_row(m, show_scores: bool = True) -> str:
    cls, _ = OUTCOME[m["prediction"]]
    when = f'{m["date"]:%a %d %b}' + (f' &middot; {m["time"]}' if pd.notna(m["time"]) else "")
    scores = ""
    if show_scores and pd.notna(m.get("likely_scores")):
        scores = f'<div class="scores">{esc(m["likely_scores"])}</div>'
    return f"""
      <li class="fixture">
        <div class="when">{when}</div>
        <div class="teams">
          <span class="side home-side">{esc(m["home"])}</span>
          <span class="vs">v</span>
          <span class="side away-side">{esc(m["away"])}</span>
        </div>
        <div class="odds">
          {prob_bar(m["p_H"], m["p_D"], m["p_A"])}
          <div class="pcts">
            <span class="pct home">{m["p_H"]:.0%}</span>
            <span class="pct draw">{m["p_D"]:.0%}</span>
            <span class="pct away">{m["p_A"]:.0%}</span>
          </div>
        </div>
        <div class="callcol">
          <span class="call {cls}">{OUTCOME[m["prediction"]][1]}</span>
          <span class="conf">{esc(m["confidence"])}</span>
        </div>
        {scores}
      </li>"""


def result_row(r: dict) -> str:
    hit = "hit" if r["correct"] else "miss"
    mark = "correct" if r["correct"] else "wrong"
    return f"""
      <li class="result {hit}">
        <div class="when">MW{r['matchweek']} &middot; {datetime.strptime(r['date'],'%Y-%m-%d'):%d %b}</div>
        <div class="teams">
          <span class="side home-side">{esc(r['home'])}</span>
          <span class="score">{r['hg']}&ndash;{r['ag']}</span>
          <span class="side away-side">{esc(r['away'])}</span>
        </div>
        <div class="odds">
          {prob_bar(r['pH'], r['pD'], r['pA'])}
          <div class="pcts">
            <span class="pct home">{r['pH']:.0%}</span>
            <span class="pct draw">{r['pD']:.0%}</span>
            <span class="pct away">{r['pA']:.0%}</span>
          </div>
        </div>
        <div class="callcol">
          <span class="call {OUTCOME[r['prediction']][0]}">{OUTCOME[r['prediction']][1]}</span>
          <span class="verdict {hit}">{mark}</span>
        </div>
      </li>"""


# ------------------------------------------------------------------ build
def build() -> str:
    matches = pd.read_csv(config.MATCHES_CSV, parse_dates=["date"])
    preds = pd.read_csv(config.DATA / f"predictions_{config.CURRENT_SEASON.replace('-','_')}.csv",
                        parse_dates=["date"])
    summary = json.loads((config.DATA / "summary.json").read_text())

    season = config.CURRENT_SEASON
    season_txt = season.replace("-", "/")
    cur = matches[matches.season == season]
    table = league_table(cur)

    pm = PoissonGoals(xi=0.002, ridge=1.0).fit(matches)
    ratings = pm.ratings()
    elo = preds.groupby("home")["home_elo"].first()

    bt = summary["backtest"]
    next_mw = int(preds.matchweek.min())
    next_fixtures = preds[preds.matchweek == next_mw].sort_values("date")

    # --- power ratings, ordered by Elo
    power = []
    for t in sorted(set(preds.home)):
        r = ratings.loc[t] if t in ratings.index else None
        power.append(dict(team=t, elo=float(elo.get(t, 1500)),
                          atk=float(r["attack"]) if r is not None else 1.0,
                          dfn=float(r["defence"]) if r is not None else 1.0))
    power.sort(key=lambda x: -x["elo"])
    emax = max(p["elo"] for p in power)
    emin = min(p["elo"] for p in power)

    # --- section: next matchweek
    next_html = "\n".join(fixture_row(m) for _, m in next_fixtures.iterrows())

    # --- section: later matchweeks
    later = preds[preds.matchweek > next_mw]
    later_html = []
    for mw, grp in later.groupby("matchweek"):
        rows = "\n".join(fixture_row(m, show_scores=False) for _, m in grp.sort_values("date").iterrows())
        later_html.append(
            f'<details class="mw"><summary><span class="mwno">Matchweek {int(mw)}</span>'
            f'<span class="mwdate">{grp.date.min():%d %b}</span>'
            f'<span class="mwn">{len(grp)} matches</span></summary>'
            f'<ul class="fixtures">{rows}</ul></details>')
    later_html = "\n".join(later_html)

    # --- section: results
    results_html = "\n".join(result_row(r) for r in reversed(summary["results"]))
    hits = sum(1 for r in summary["results"] if r["correct"])

    # --- section: table rows
    table_rows = []
    for _, r in table.iterrows():
        form = "".join(f'<i class="f{f}" title="{f}">{f}</i>' for f in recent_form(cur, r.team))
        table_rows.append(f"""<tr>
          <td class="num pos">{r.pos}</td><td class="tm">{esc(r.team)}</td>
          <td class="num">{r.played}</td><td class="num">{r.won}</td>
          <td class="num">{r.drew}</td><td class="num">{r.lost}</td>
          <td class="num dim">{r.gf}</td><td class="num dim">{r.ga}</td>
          <td class="num">{r.gd:+d}</td><td class="num pts">{r.pts}</td>
          <td class="form">{form}</td></tr>""")
    table_rows = "\n".join(table_rows)

    # --- section: power rows
    power_rows = []
    for p in power:
        w = 6 + 94 * (p["elo"] - emin) / max(emax - emin, 1)
        power_rows.append(f"""<tr>
          <td class="tm">{esc(p['team'])}</td>
          <td class="elowrap"><span class="elobar" style="width:{w:.1f}%"></span>
              <span class="eloval">{p['elo']:.0f}</span></td>
          <td class="num">{p['atk']:.2f}</td>
          <td class="num">{p['dfn']:.2f}</td></tr>""")
    power_rows = "\n".join(power_rows)

    acc = summary["season_accuracy"]
    mkt = summary["season_market_accuracy"]

    subs = {
        "SEASON": season_txt, "GENERATED": summary["generated"],
        "PLAYED": str(summary["played"]), "UPCOMING": str(summary["upcoming"]),
        "TRAINED": f'{summary["trained_on"]:,}',
        "NEXTMW": str(next_mw), "NEXTFIX": next_html, "LATERFIX": later_html,
        "RESULTS": results_html, "TABLEROWS": table_rows, "POWERROWS": power_rows,
        "ACC": f"{acc:.0%}", "ACCN": f'{hits}/{summary["played"]}',
        "SEASONLL": f'{summary["season_log_loss"]:.3f}',
        "ALWAYSHOME": f'{summary["season_always_home"]:.0%}',
        "MKTACC": f"{mkt:.0%}" if mkt else "n/a", "MKTN": str(summary["season_market_n"]),
        "BTACC": f'{bt["model_accuracy"]:.1%}', "BTLL": f'{bt["model_log_loss"]:.3f}',
        "BTMACC": f'{bt["market_accuracy"]:.1%}', "BTMLL": f'{bt["market_log_loss"]:.3f}',
        "BTBACC": f'{bt["baseline_accuracy"]:.1%}', "BTBLL": f'{bt["baseline_log_loss"]:.3f}',
        "BTN": f'{bt["matches"]:,}', "BTSEASONS": bt["seasons"],
    }
    html = TEMPLATE
    for k, v in subs.items():
        html = html.replace("__%s__" % k, v)
    return html


TEMPLATE = r"""<title>Premier League Model</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{
  --ground:#F4F2EC; --surface:#FCFBF7; --raised:#FFFFFF;
  --ink:#141F1A; --muted:#5D6B64; --faint:#8A968F;
  --line:#E0DCD0; --line-soft:#EDEAE0;
  --home:#A8630F; --draw:#5F6E77; --away:#166274;
  --home-fill:#D9922F; --draw-fill:#94A2AA; --away-fill:#2E8AA1;
  --good:#2F6B45; --bad:#9C3B2E;
  --accent:#1B3A2F;
  --shadow:0 1px 2px rgba(20,31,26,.06), 0 4px 16px rgba(20,31,26,.05);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0E1613; --surface:#151F1B; --raised:#1B2723;
    --ink:#E9E7DE; --muted:#94A29A; --faint:#6E7C74;
    --line:#26332D; --line-soft:#1E2A25;
    --home:#E0A050; --draw:#93A2AB; --away:#5BB2C6;
    --home-fill:#D9922F; --draw-fill:#6E7E87; --away-fill:#2E8AA1;
    --good:#63B183; --bad:#D4715F;
    --accent:#8FCBAE;
    --shadow:0 1px 2px rgba(0,0,0,.3), 0 4px 18px rgba(0,0,0,.22);
  }
}
:root[data-theme="dark"]{
  --ground:#0E1613; --surface:#151F1B; --raised:#1B2723;
  --ink:#E9E7DE; --muted:#94A29A; --faint:#6E7C74;
  --line:#26332D; --line-soft:#1E2A25;
  --home:#E0A050; --draw:#93A2AB; --away:#5BB2C6;
  --home-fill:#D9922F; --draw-fill:#6E7E87; --away-fill:#2E8AA1;
  --good:#63B183; --bad:#D4715F;
  --accent:#8FCBAE;
  --shadow:0 1px 2px rgba(0,0,0,.3), 0 4px 18px rgba(0,0,0,.22);
}

*{box-sizing:border-box}
body{
  background:var(--ground); color:var(--ink);
  font:400 15px/1.55 "IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  margin:0; padding:0 20px 72px;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1080px;margin:0 auto}
h1,h2,h3{font-family:Archivo,"Helvetica Neue",Arial,sans-serif;text-wrap:balance;margin:0}
.num,.mono,td.num,.pct,.score,.eloval{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--away-fill);outline-offset:2px;border-radius:3px}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}

/* ---------- masthead ---------- */
header{padding:44px 0 26px;border-bottom:2px solid var(--ink)}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--muted);margin-bottom:12px}
h1{font-size:clamp(30px,5.4vw,50px);font-weight:800;letter-spacing:-.024em;line-height:1.02}
h1 .sub{display:block;font-weight:500;font-size:clamp(15px,2vw,19px);color:var(--muted);
  letter-spacing:-.005em;margin-top:10px;max-width:56ch;line-height:1.45}

.status{display:flex;flex-wrap:wrap;gap:0;margin:26px 0 0;border:1px solid var(--line);
  border-radius:3px;background:var(--surface);overflow:hidden}
.status div{flex:1 1 130px;padding:13px 16px;border-right:1px solid var(--line-soft)}
.status div:last-child{border-right:0}
.status dt{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.13em;
  text-transform:uppercase;color:var(--faint);margin:0 0 5px}
.status dd{margin:0;font-family:"IBM Plex Mono",monospace;font-size:17px;font-weight:500;
  font-variant-numeric:tabular-nums}

/* ---------- sections ---------- */
section{margin-top:52px}
.shead{display:flex;align-items:baseline;justify-content:space-between;gap:16px;
  margin-bottom:16px;padding-bottom:9px;border-bottom:1px solid var(--line)}
h2{font-size:20px;font-weight:700;letter-spacing:-.014em}
.shead .note{font-size:12.5px;color:var(--muted);text-align:right}
.lede{color:var(--muted);font-size:13.5px;margin:-6px 0 16px;max-width:68ch}

/* ---------- fixtures ---------- */
ul.fixtures{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:1px;
  background:var(--line-soft);border:1px solid var(--line);border-radius:3px;overflow:hidden}
.fixture,.result{display:grid;background:var(--surface);padding:13px 16px;
  grid-template-columns:118px minmax(210px,1.15fr) minmax(150px,1fr) 96px;
  gap:16px;align-items:center}
.fixture .scores{grid-column:2/-1;font-family:"IBM Plex Mono",monospace;font-size:11.5px;
  color:var(--faint);margin-top:-4px}
.when{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--muted);
  letter-spacing:.02em}
.teams{display:grid;grid-template-columns:1fr auto 1fr;gap:10px;align-items:center;
  font-weight:500;font-size:14.5px}
.home-side{text-align:right}
.away-side{text-align:left}
.vs,.score{font-family:"IBM Plex Mono",monospace;color:var(--faint);font-size:12px}
.score{color:var(--ink);font-weight:600;font-size:14px;letter-spacing:.02em}

.bar{display:flex;height:7px;border-radius:2px;overflow:hidden;background:var(--line-soft)}
.seg{display:block;height:100%}
.seg.home{background:var(--home-fill)}
.seg.draw{background:var(--draw-fill)}
.seg.away{background:var(--away-fill)}
.pcts{display:flex;justify-content:space-between;margin-top:5px;font-size:11px;font-weight:500}
.pct.home{color:var(--home)} .pct.draw{color:var(--draw)} .pct.away{color:var(--away)}

.callcol{display:flex;flex-direction:column;align-items:flex-end;gap:3px}
.call{font-family:Archivo,sans-serif;font-size:11px;font-weight:700;letter-spacing:.05em;
  text-transform:uppercase;padding:3px 7px;border-radius:2px;white-space:nowrap}
.call.home{color:var(--home);background:color-mix(in srgb,var(--home-fill) 15%,transparent)}
.call.draw{color:var(--draw);background:color-mix(in srgb,var(--draw-fill) 18%,transparent)}
.call.away{color:var(--away);background:color-mix(in srgb,var(--away-fill) 15%,transparent)}
.conf,.verdict{font-family:"IBM Plex Mono",monospace;font-size:10px;color:var(--faint);
  letter-spacing:.06em;text-transform:uppercase}
.verdict.hit{color:var(--good)} .verdict.miss{color:var(--bad)}
.result.hit{border-left:2px solid var(--good)}
.result.miss{border-left:2px solid var(--bad)}

/* ---------- later matchweeks ---------- */
details.mw{border:1px solid var(--line);border-radius:3px;background:var(--surface);
  margin-bottom:8px}
details.mw summary{cursor:pointer;padding:11px 16px;display:flex;align-items:baseline;gap:14px;
  font-family:"IBM Plex Mono",monospace;font-size:12.5px;list-style:none}
details.mw summary::-webkit-details-marker{display:none}
details.mw summary::before{content:"+";color:var(--faint);font-weight:600;width:11px}
details.mw[open] summary::before{content:"\2212"}
.mwno{font-weight:600;letter-spacing:.02em}
.mwdate{color:var(--muted)}
.mwn{margin-left:auto;color:var(--faint);font-size:11px}
details.mw ul.fixtures{border:0;border-top:1px solid var(--line);border-radius:0}

/* ---------- tables ---------- */
.cols{display:grid;grid-template-columns:1.35fr 1fr;gap:28px;align-items:start}
.panel{border:1px solid var(--line);border-radius:3px;background:var(--surface);overflow:hidden}
.tscroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px}
th{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--faint);font-weight:500;text-align:right;padding:10px 8px;border-bottom:1px solid var(--line);
  white-space:nowrap}
th:nth-child(2),th:first-child{text-align:left}
td{padding:7px 8px;border-bottom:1px solid var(--line-soft);text-align:right;white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
td.tm{text-align:left;font-weight:500}
td.num{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
td.pos{color:var(--faint);width:28px}
td.dim{color:var(--faint)}
td.pts{font-weight:600}
.form{display:flex;gap:2px;justify-content:flex-end;padding-right:12px}
.form i{font-style:normal;font-family:"IBM Plex Mono",monospace;font-size:9px;font-weight:600;
  width:14px;height:14px;line-height:14px;text-align:center;border-radius:2px;color:var(--surface)}
.fW{background:var(--good)} .fD{background:var(--draw-fill)} .fL{background:var(--bad)}
.elowrap{position:relative;width:100%;min-width:120px;text-align:left!important}
.elobar{display:inline-block;height:6px;border-radius:2px;background:var(--accent);opacity:.45;
  vertical-align:middle}
.eloval{margin-left:8px;font-size:11.5px;color:var(--muted)}

/* ---------- scorecard ---------- */
.score-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:1px;
  background:var(--line-soft);border:1px solid var(--line);border-radius:3px;overflow:hidden}
.tile{background:var(--surface);padding:17px 18px}
.tile .k{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.12em;
  text-transform:uppercase;color:var(--faint);margin-bottom:7px}
.tile .v{font-family:Archivo,sans-serif;font-size:29px;font-weight:700;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums;line-height:1}
.tile .s{font-size:11.5px;color:var(--muted);margin-top:6px;line-height:1.4}
.tile.lead .v{color:var(--accent)}

.caveat{margin-top:14px;padding:13px 16px;border-left:2px solid var(--home-fill);
  background:var(--surface);font-size:13px;color:var(--muted);border-radius:0 3px 3px 0}
.caveat b{color:var(--ink);font-weight:600}

footer{margin-top:60px;padding-top:20px;border-top:1px solid var(--line);
  font-size:12px;color:var(--faint);display:flex;justify-content:space-between;
  flex-wrap:wrap;gap:10px}

@media (max-width:860px){
  .cols{grid-template-columns:1fr}
  .fixture,.result{grid-template-columns:1fr 1fr;row-gap:10px}
  .when{order:1} .callcol{order:2;align-items:flex-end}
  .teams{order:3;grid-column:1/-1} .odds{order:4;grid-column:1/-1}
  .fixture .scores{order:5}
}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Premier League &middot; __SEASON__</div>
  <h1>Match Predictions
    <span class="sub">A three-way outcome model over Elo, rolling form and a Dixon&ndash;Coles
    goals model. Bookmaker odds are used to keep it honest, never as an input.</span>
  </h1>
  <dl class="status">
    <div><dt>Updated</dt><dd>__GENERATED__</dd></div>
    <div><dt>Played</dt><dd>__PLAYED__</dd></div>
    <div><dt>Remaining</dt><dd>__UPCOMING__</dd></div>
    <div><dt>Trained on</dt><dd>__TRAINED__</dd></div>
  </dl>
</header>

<section>
  <div class="shead"><h2>Matchweek __NEXTMW__</h2>
    <div class="note">Bars read home &middot; draw &middot; away</div></div>
  <ul class="fixtures">__NEXTFIX__</ul>
</section>

<section>
  <div class="shead"><h2>How it is doing</h2>
    <div class="note">Season figures are out-of-sample</div></div>
  <p class="lede">Matches already played this season were predicted by a model trained only on
  earlier seasons, so these are not marks awarded for homework the model had already seen.</p>
  <div class="score-grid">
    <div class="tile lead"><div class="k">This season</div><div class="v">__ACC__</div>
      <div class="s">__ACCN__ correct &middot; log loss __SEASONLL__</div></div>
    <div class="tile"><div class="k">Bet365 favourite</div><div class="v">__MKTACC__</div>
      <div class="s">over the __MKTN__ of those with published odds</div></div>
    <div class="tile"><div class="k">Always pick home</div><div class="v">__ALWAYSHOME__</div>
      <div class="s">what guessing without a model would score</div></div>
    <div class="tile"><div class="k">Backtest</div><div class="v">__BTACC__</div>
      <div class="s">__BTN__ matches, __BTSEASONS__ &middot; log loss __BTLL__</div></div>
  </div>
  <div class="caveat"><b>A few dozen matches settles nothing.</b> The season figure above swings
  wildly on this little evidence. The backtest is the real measure: across __BTN__ matches the
  model scores __BTACC__ at __BTLL__ log loss, against __BTMACC__ / __BTMLL__ for the bookmaker
  and __BTBACC__ / __BTBLL__ for always picking the home side &mdash; so it closes roughly three
  quarters of the distance between knowing nothing and the market price.</div>
</section>

<div class="cols">
  <section style="margin-top:52px">
    <div class="shead"><h2>Table</h2><div class="note">form, most recent last</div></div>
    <div class="panel tscroll"><table>
      <thead><tr><th></th><th>Team</th><th>P</th><th>W</th><th>D</th><th>L</th>
        <th>GF</th><th>GA</th><th>GD</th><th>Pts</th><th style="text-align:right">Form</th></tr></thead>
      <tbody>__TABLEROWS__</tbody></table></div>
  </section>

  <section style="margin-top:52px">
    <div class="shead"><h2>Strength</h2><div class="note">1.00 is average</div></div>
    <div class="panel tscroll"><table>
      <thead><tr><th>Team</th><th style="text-align:left">Elo</th><th>Att</th><th>Def</th></tr></thead>
      <tbody>__POWERROWS__</tbody></table></div>
    <p class="lede" style="margin-top:12px">Attack is goals scored against an average side;
    defence is goals conceded, so <em>lower is better</em>. Both come from the Poisson model.</p>
  </section>
</div>

<section>
  <div class="shead"><h2>Results so far</h2>
    <div class="note">most recent first</div></div>
  <ul class="fixtures">__RESULTS__</ul>
</section>

<section>
  <div class="shead"><h2>The rest of the season</h2>
    <div class="note">__UPCOMING__ fixtures</div></div>
  __LATERFIX__
</section>

<footer>
  <span>Regenerated every Tuesday from fbref and football-data.co.uk.</span>
  <span>Predictions are model output, not advice.</span>
</footer>
</div>
"""

DOC = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Three-way Premier League match predictions, refreshed weekly.">
<meta name="color-scheme" content="light dark">
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
