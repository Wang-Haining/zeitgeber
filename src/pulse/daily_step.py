"""Is there a step at the R01 due date itself? A seasonal component is smooth; a deadline is a sawtooth.

Step statistic: on Poisson day counts of US and non-US first versions (day FE, group x cycle FE, group x weekday
FE), the US log-ratio in the 7 days before an anchor date minus the 7 days from the anchor on. True anchors are
5 February, 5 June and 5 October; the null is the same statistic with every anchor shifted by s days,
14 <= |s| <= 60. Pooled and per cycle family. Also the daily event-time profile (-21..+20 days) and the R01-window
share effect for 21-, 28- and 29-day windows (cycle-clustered SE).

    python -m pulse.daily_step  ->  results/daily_step.csv, daily_step_profile.csv, window_length.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pulse import cases, signals
from pulse.counts import day_panel
from pulse.parse_jats import INTERIM

R = signals.RESULTS
SHIFTS = [s for s in range(-60, 61) if abs(s) >= 14]


def anchors(months: tuple[int, ...], shift: int, years=range(2013, 2028)) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(y, m, 5) + pd.Timedelta(days=shift) for y in years for m in months])


def rel_days(dates: pd.Series, anc: pd.DatetimeIndex) -> np.ndarray:
    """Signed days from the nearest anchor (negative = before)."""
    d = dates.to_numpy("datetime64[D]")
    a = np.sort(anc.to_numpy("datetime64[D]"))
    i = np.searchsorted(a, d)
    nxt = a[np.clip(i, 0, len(a) - 1)]
    prv = a[np.clip(i - 1, 0, len(a) - 1)]
    dn = (d - nxt).astype(int)
    dp = (d - prv).astype(int)
    return np.where(np.abs(dn) <= np.abs(dp), dn, dp)


def step(p: pd.DataFrame, us: np.ndarray, anc: pd.DatetimeIndex) -> tuple[float, float]:
    import pyfixest as pf

    q = p.copy()
    r = rel_days(q["date"], anc)
    q["xpre"] = us * ((r >= -7) & (r <= -1))
    q["xpost"] = us * ((r >= 0) & (r <= 6))
    m = pf.fepois("n ~ xpre + xpost | day + gc + gd", data=q, vcov={"CRV1": "cycle"})
    co = m.coef()
    V = pd.DataFrame(m._vcov, index=co.index, columns=co.index)
    b = float(co["xpre"] - co["xpost"])
    se = float(np.sqrt(V.loc["xpre", "xpre"] + V.loc["xpost", "xpost"] - 2 * V.loc["xpre", "xpost"]))
    return b, se


def profile(p: pd.DataFrame, us: np.ndarray, anc: pd.DatetimeIndex, server: str) -> list[dict]:
    import pyfixest as pf

    q = p.copy()
    r = rel_days(q["date"], anc)
    cols = []
    for k in range(-21, 21):
        nm = f"d{k + 21}"
        q[nm] = us * (r == k)
        cols.append(nm)
    m = pf.fepois(f"n ~ {' + '.join(cols)} | day + gc + gd", data=q, vcov={"CRV1": "cycle"})
    co, se = m.coef(), m.se()
    return [{"server": server, "rel_day": int(c[1:]) - 21, "coef": float(co[c]), "se": float(se[c])} for c in cols]


def window_length() -> pd.DataFrame:
    rows = []
    for s in ("biorxiv", "medrxiv"):
        v = signals.prep(pd.read_parquet(INTERIM / f"{s}_v1.parquet"))
        v = v[~v["covid"] & v["us"].notna()].copy()
        v["cycle"] = (v["date"] + pd.to_timedelta(v["phi"], unit="D")).dt.strftime("%Y-%m")
        for lab, hi in (("21 days (0-20)", 20), ("28 days (0-27)", 27), ("29 days (0-28, as reported)", 28)):
            x = (v["du_R_new"] <= hi).to_numpy(float)[:, None]
            codes, uniq = pd.factorize(v["cycle"])
            beta, resid, Xd, XtX_inv = signals._fit(v["us"].to_numpy(float), x, codes, len(uniq))
            S = np.bincount(codes, weights=Xd[:, 0] * resid)
            se = float(np.sqrt(XtX_inv[0, 0] ** 2 * (S @ S) * len(uniq) / (len(uniq) - 1)))
            rows.append({"server": s, "window": lab, "pp": 100 * beta[0], "se_pp": 100 * se})
    return pd.DataFrame(rows)


def run() -> None:
    wl = window_length()
    wl.to_csv(R / "window_length.csv", index=False)
    print(wl.round(3).to_string(index=False), flush=True)
    rows, prof = [], []
    for server in ("biorxiv", "medrxiv"):
        d = cases.load(server)
        g = np.where(d["us"] == 1, "US", "nonUS")
        p = day_panel(d, pd.Series(g), ["nonUS", "US"], d["date"].min(), d["date"].max())
        us = (p["g"] == "US").to_numpy().astype(int)
        prof += profile(p, us, anchors((2, 6, 10), 0), server)
        for lab, months in (("pooled", (2, 6, 10)), ("Feb", (2,)), ("Jun", (6,)), ("Oct", (10,))):
            b, se = step(p, us, anchors(months, 0))
            null = np.array([step(p, us, anchors(months, s))[0] for s in SHIFTS])
            rows.append(
                {
                    "server": server,
                    "cycles": lab,
                    "step": b,
                    "se": se,
                    "null_mean": null.mean(),
                    "null_sd": null.std(),
                    "null_max": null.max(),
                    "rank_p": (1 + (null >= b).sum()) / (len(null) + 1),
                    "n_shifts": len(null),
                }
            )
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(R / "daily_step.csv", index=False)
    pd.DataFrame(prof).to_csv(R / "daily_step_profile.csv", index=False)


if __name__ == "__main__":
    run()
