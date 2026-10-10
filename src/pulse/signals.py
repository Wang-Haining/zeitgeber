"""Shared estimators and paths.

Paper-level model with week fixed effects absorbed (Frisch-Waugh-Lovell):

    y = a_week + b * exposed + c * exposed * in_window + e

`c` is how much more the exposed group's outcome moves inside a schedule's pre-deadline window than the control
group's, holding everything common to that week fixed. SE clustered by week.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pulse.parse_jats import INTERIM

RESULTS = INTERIM.parents[1] / "results"
RNG = np.random.default_rng(20261004)


def _demean(x: np.ndarray, codes: np.ndarray, n: int) -> np.ndarray:
    s = np.bincount(codes, weights=x, minlength=n)
    c = np.bincount(codes, minlength=n)
    return x - (s / np.maximum(c, 1))[codes]


def _fit(y, X, codes, n):
    yd = _demean(y, codes, n)
    Xd = np.column_stack([_demean(X[:, j], codes, n) for j in range(X.shape[1])])
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    beta = XtX_inv @ Xd.T @ yd
    return beta, yd - Xd @ beta, Xd, XtX_inv


@dataclass
class Contrast:
    coef: float
    se: float
    p_cluster: float
    p_perm: float
    n: int
    n_exposed_in_window: int


def phase_contrast(
    df: pd.DataFrame, y: str, exposed: str, window: str, fe: str = "week", n_perm: int = 500
) -> Contrast | None:
    d = df[[y, exposed, window, fe]].dropna()
    if d[exposed].nunique() < 2 or len(d) < 200:
        return None
    codes, uniq = pd.factorize(d[fe])
    n = len(uniq)
    yv = d[y].to_numpy(float)
    g = d[exposed].to_numpy(float)
    w = d[window].to_numpy(float)
    X = np.column_stack([g, g * w])
    beta, resid, Xd, XtX_inv = _fit(yv, X, codes, n)

    # cluster-robust (week) sandwich
    scores = Xd * resid[:, None]
    S = np.vstack([np.bincount(codes, weights=scores[:, j], minlength=n) for j in range(2)]).T
    meat = S.T @ S
    V = XtX_inv @ meat @ XtX_inv * n / max(n - 1, 1)
    se = float(np.sqrt(V[1, 1]))
    from scipy import stats  # local import: scipy comes with statsmodels

    p_cl = float(2 * stats.t.sf(abs(beta[1] / se), df=max(n - 1, 1))) if se > 0 else np.nan

    # permutation: shuffle exposure within week
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    blocks = np.split(order, bounds)
    null = np.empty(n_perm)
    for b in range(n_perm):
        gp = g.copy()
        for blk in blocks:
            if len(blk) > 1:
                gp[blk] = RNG.permutation(g[blk])
        bp, *_ = _fit(yv, np.column_stack([gp, gp * w]), codes, n)
        null[b] = bp[1]
    p_perm = float((1 + np.sum(np.abs(null) >= abs(beta[1]))) / (n_perm + 1))
    return Contrast(float(beta[1]), se, p_cl, p_perm, len(d), int(((g == 1) & (w == 1)).sum()))


def window_effect(df: pd.DataFrame, y: str, window: str, fe: str = "year") -> tuple[float, float, int]:
    """Linear-probability window effect on y with `fe` fixed effects, SE clustered by week."""
    d = df[[y, window, fe]].dropna()
    codes, uniq = pd.factorize(d[fe])
    n = len(uniq)
    beta, resid, Xd, XtX_inv = _fit(d[y].to_numpy(float), d[[window]].to_numpy(float), codes, n)
    wk = pd.factorize(df.loc[d.index, "week"])[0]
    S = np.bincount(wk, weights=Xd[:, 0] * resid)
    se = float(np.sqrt(XtX_inv[0, 0] ** 2 * (S @ S)))
    return float(beta[0]), se, len(d)


def prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["us"] = (d["country"] == "US").astype(float).where(d["country"].notna())
    d["nih_us"] = d["nih"].astype(float).where((d["country"] == "US") & d["has_funding_text"])
    d["hype_pk"] = 1000 * d["ab_hype_nosig"] / d["ab_n_words"].replace(0, np.nan)
    d["hype_pk_all"] = 1000 * d["ab_hype"] / d["ab_n_words"].replace(0, np.nan)
    d["abs_head_z"] = d["ab_head_z"].abs()
    d["log_n"] = np.log10(d["ab_n_sample"])
    d["caliper_any_below"] = (
        (d["ab_n_p_caliper_below"] > 0).astype(float).where((d["ab_n_p_caliper_below"] + d["ab_n_p_caliper_above"]) > 0)
    )
    d["share_sig"] = d["ab_share_p_sig"]
    d["one"] = 1.0
    d["rs_median_abs_z"] = d.get("rs_median_abs_z")
    return d


WINDOWS = ["win_R_new", "win_F", "win_AIDS", "win_R_resub"]


def periods(server: str) -> dict[str, tuple[int, int]]:
    if server == "medrxiv":
        return {"discovery": (2019, 2022), "confirm": (2023, 2026)}
    return {"discovery": (2017, 2021), "confirm": (2022, 2026)}
