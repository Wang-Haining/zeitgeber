"""Does the rhythm break at the due date, and does it follow the date NIH actually uses?

1. Local linear discontinuity at the R01 due date: on Poisson day counts of US and non-US first versions (day,
   group-by-cycle and group-by-weekday fixed effects, cycles centred on the due dates, SE clustered by cycle),
   restricted to days within h of a due date,
   the US log ratio is fitted with its own line on each side; the drop is the US x after-due-date coefficient, sign
   reversed. Bandwidths 7, 14 and 21 days; nominal and business-day dates; a shifted-date null for h = 14.
2. Phase test: when 5 February, 5 June or 5 October falls on a weekend, NIH accepts applications on the next business
   day. The weekend days between the nominal and the actual due date are then still before the deadline. US x those days
   is compared with US x the 7 days before the nominal date and the 7 days after the actual date.

    python -m pulse.rd_phase  ->  results/rd_phase.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pulse import cases, signals
from pulse.counts import day_panel
from pulse.daily_step import SHIFTS, centre_cycles, rel_days

R = signals.RESULTS


def dues(months=(2, 6, 10), shift=0, business=False, years=range(2013, 2028)) -> pd.DatetimeIndex:
    out = []
    for y in years:
        for m in months:
            d = pd.Timestamp(y, m, 5)
            if business:
                d += pd.Timedelta(days={5: 2, 6: 1}.get(d.dayofweek, 0))
            out.append(d + pd.Timedelta(days=shift))
    return pd.DatetimeIndex(out)


def rd(p: pd.DataFrame, us: np.ndarray, anc: pd.DatetimeIndex, h: int) -> tuple[float, float, int]:
    import pyfixest as pf

    q = centre_cycles(p, anc)
    q["r"] = rel_days(q["date"], anc)
    keep = (q["r"] >= -h) & (q["r"] <= h - 1)
    q, u = q[keep].copy(), us[keep.to_numpy()]
    post = (q["r"] >= 0).astype(int).to_numpy()
    q["xpost"] = u * post
    q["xr"] = u * q["r"].to_numpy()
    q["xrpost"] = u * q["r"].to_numpy() * post
    m = pf.fepois("n ~ xpost + xr + xrpost | day + gc + gd", data=q, vcov={"CRV1": "cycle"})
    return -float(m.coef()["xpost"]), float(m.se()["xpost"]), int(q["cycle"].nunique())


def weekend_test(p: pd.DataFrame, us: np.ndarray, months=(2, 6, 10)) -> dict:
    import pyfixest as pf

    nom, act = dues(months), dues(months, business=True)
    gap = set()
    for a, b in zip(nom, act, strict=False):
        gap |= set(pd.date_range(a, b - pd.Timedelta(days=1)))
    q = centre_cycles(p, nom)
    lo, hi = q["date"].min(), q["date"].max()
    rn, ra = rel_days(q["date"], nom), rel_days(q["date"], act)
    q["xpre"] = us * ((rn >= -7) & (rn <= -1))
    q["xgap"] = us * q["date"].isin(gap).to_numpy()
    q["xpost"] = us * ((ra >= 0) & (ra <= 6))
    m = pf.fepois("n ~ xpre + xgap + xpost | day + gc + gd", data=q, vcov={"CRV1": "cycle"})
    co, se = m.coef(), m.se()
    return {
        "pre_nominal": float(co["xpre"]),
        "pre_se": float(se["xpre"]),
        "gap": float(co["xgap"]),
        "gap_se": float(se["xgap"]),
        "post_actual": float(co["xpost"]),
        "post_se": float(se["xpost"]),
        "gap_days": sum(lo <= d <= hi for d in gap),
        "shifted_dates": int(((nom != act) & (nom >= lo) & (nom <= hi)).sum()),
    }


def run() -> None:
    rows = []
    for server in ("biorxiv", "medrxiv"):
        d = cases.load(server)
        g = np.where(d["us"] == 1, "US", "nonUS")
        p = day_panel(d, pd.Series(g), ["nonUS", "US"], d["date"].min(), d["date"].max())
        us = (p["g"] == "US").to_numpy().astype(int)
        for business in (False, True):
            for h in (7, 14, 21):
                b, se, nc = rd(p, us, dues(business=business), h)
                row = {
                    "server": server,
                    "test": "discontinuity",
                    "dates": "business day" if business else "nominal",
                    "bandwidth": h,
                    "drop": b,
                    "se": se,
                    "cycles": nc,
                }
                if h == 14 and not business:
                    null = np.array([rd(p, us, dues(shift=s), h)[0] for s in SHIFTS])
                    row.update(
                        {
                            "null_mean": null.mean(),
                            "null_sd": null.std(),
                            "null_max": null.max(),
                            "rank_p": (1 + (null >= b).sum()) / (len(null) + 1),
                        }
                    )
                rows.append(row)
                print(row, flush=True)
        w = weekend_test(p, us)
        rows.append({"server": server, "test": "weekend phase", **w})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(R / "rd_phase.csv", index=False)


if __name__ == "__main__":
    run()
