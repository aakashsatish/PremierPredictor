"""Dixon-Coles style Poisson goals model.

Rather than classifying the outcome, this models the thing that actually
happens on the pitch -- goals -- and derives the outcome from it.

Each club carries an attack and a defence rating. For a fixture:

    home expected goals = exp(base + attack[home] - defence[away] + home_adv)
    away expected goals = exp(base + attack[away] - defence[home])

Those two numbers give a Poisson distribution over each side's goals, whose
outer product is a grid of every plausible scoreline. Summing the grid below,
on, and above its diagonal yields home win, draw and away win. The draw is a
real quantity here, not a leftover -- which is the whole reason for building
this alongside the classifier.

Three departures from a textbook Poisson fit, each addressing something the
data actually does:

  * Dixon-Coles low-score correction. Plain Poisson understates 0-0 and 1-1;
    measured on this dataset, 26.7% of team-innings are goalless against a
    Poisson prediction of 24.2%. The tau term below adjusts the four lowest
    scorelines and rho is fitted rather than assumed.

  * Exponential time decay. A match from 2016 says little about a club now,
    so each match is weighted by exp(-xi * days ago).

  * Ridge shrinkage toward league average. Newly promoted clubs arrive with a
    handful of matches, and without a brake their ratings chase noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10          # grid size; P(11+ goals) is negligible
DEFAULT_XI = 0.0018     # time decay per day (~1 year half-life)
DEFAULT_RIDGE = 8.0     # shrinkage strength toward league average


def _tau(hg, ag, lh, la, rho):
    """Dixon-Coles correction to the four lowest scorelines.

    Plain Poisson treats the two teams' goal counts as independent. At 0-0,
    1-0, 0-1 and 1-1 that assumption is measurably wrong, and this is the
    standard fix.
    """
    out = np.ones_like(lh, dtype=float)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    out[m00] = 1.0 - lh[m00] * la[m00] * rho
    out[m01] = 1.0 + lh[m01] * rho
    out[m10] = 1.0 + la[m10] * rho
    out[m11] = 1.0 - rho
    return np.clip(out, 1e-9, None)


class PoissonGoals:
    def __init__(self, xi: float = DEFAULT_XI, ridge: float = DEFAULT_RIDGE):
        self.xi = xi
        self.ridge = ridge
        self.teams: list[str] = []

    # --- fitting ---------------------------------------------------------
    def fit(self, matches: pd.DataFrame, ref_date=None) -> "PoissonGoals":
        df = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        self.teams = sorted(set(df["home"]) | set(df["away"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        hi = df["home"].map(idx).to_numpy()
        ai = df["away"].map(idx).to_numpy()
        hg = df["home_goals"].to_numpy().astype(int)
        ag = df["away_goals"].to_numpy().astype(int)

        # Recent matches matter more.
        ref = pd.Timestamp(ref_date) if ref_date is not None else df["date"].max()
        days = (ref - pd.to_datetime(df["date"])).dt.days.to_numpy()
        w = np.exp(-self.xi * np.clip(days, 0, None))

        def unpack(p):
            return p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1], p[2 * n + 2]

        def neg_ll(p):
            atk, dfn, home_adv, base, rho = unpack(p)
            lh = np.exp(base + atk[hi] - dfn[ai] + home_adv)
            la = np.exp(base + atk[ai] - dfn[hi])
            lh = np.clip(lh, 1e-6, 12.0)
            la = np.clip(la, 1e-6, 12.0)

            ll = poisson.logpmf(hg, lh) + poisson.logpmf(ag, la)
            ll = ll + np.log(_tau(hg, ag, lh, la, rho))
            # Ridge pulls ratings toward zero (i.e. league average).
            penalty = self.ridge * (np.sum(atk ** 2) + np.sum(dfn ** 2))
            return -np.sum(w * ll) + penalty

        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [np.log(1.35)], [-0.05]])
        bounds = ([(-1.5, 1.5)] * n + [(-1.5, 1.5)] * n
                  + [(-0.5, 1.0), (-1.0, 1.5), (-0.2, 0.2)])
        res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 800})

        self.attack_, self.defence_, self.home_adv_, self.base_, self.rho_ = unpack(res.x)
        self.converged_ = bool(res.success)
        self._idx = idx
        return self

    # --- prediction ------------------------------------------------------
    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        """Expected goals for each side. Unknown clubs fall back to average."""
        i, j = self._idx.get(home), self._idx.get(away)
        atk_h = self.attack_[i] if i is not None else 0.0
        def_h = self.defence_[i] if i is not None else 0.0
        atk_a = self.attack_[j] if j is not None else 0.0
        def_a = self.defence_[j] if j is not None else 0.0
        lh = np.exp(self.base_ + atk_h - def_a + self.home_adv_)
        la = np.exp(self.base_ + atk_a - def_h)
        return float(np.clip(lh, 0.05, 12)), float(np.clip(la, 0.05, 12))

    def score_grid(self, home: str, away: str) -> np.ndarray:
        """Probability of every scoreline from 0-0 up to MAX_GOALS."""
        lh, la = self.expected_goals(home, away)
        g = np.outer(poisson.pmf(np.arange(MAX_GOALS + 1), lh),
                     poisson.pmf(np.arange(MAX_GOALS + 1), la))
        # Apply the same low-score correction used when fitting.
        g[0, 0] *= 1.0 - lh * la * self.rho_
        g[0, 1] *= 1.0 + lh * self.rho_
        g[1, 0] *= 1.0 + la * self.rho_
        g[1, 1] *= 1.0 - self.rho_
        return g / g.sum()

    def predict_proba(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Return A / D / H probabilities, in sorted class order."""
        rows = []
        for _, m in fixtures.iterrows():
            g = self.score_grid(m["home"], m["away"])
            rows.append({
                "A": float(np.triu(g, 1).sum()),
                "D": float(np.trace(g)),
                "H": float(np.tril(g, -1).sum()),
            })
        return pd.DataFrame(rows, index=fixtures.index)[["A", "D", "H"]]

    def ratings(self) -> pd.DataFrame:
        """Fitted strengths, on a readable multiplicative scale."""
        return pd.DataFrame({
            "attack": np.exp(self.attack_),
            "defence": np.exp(-self.defence_),
        }, index=self.teams).sort_values("attack", ascending=False)
