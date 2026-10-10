"""Resubmission due dates: individual profiles, the aggregate break at the date, and the kernel.

NIH receives R01 resubmissions and renewals on 5 March, 5 July and 5 November, 28 to 31 days after new applications.

  investigator  event studies of linked investigators' preprint counts in 14-day bins around their own due date,
                separately for funded first submissions (A0, new-application date) and funded resubmissions (A1,
                resubmission date); same model as Fig. 2C
  aggregate     US vs non-US daily counts: daily profile around the resubmission date; local linear discontinuity
                (US-specific slopes on each side) at the R01 date, the resubmission date (business day) and a placebo
                45 days after the R01 date, bandwidths 7 and 10 days, pooled and by month; weekend test at the
                resubmission date
  kernel        the investigator kernel placed before new-application and A1 resubmission schedules with separate
                non-negative amplitudes (renewals dropped), fitted to the aggregate weekly profile

    python -m pulse.resub_checks <investigator|aggregate|kernel>
        ->  results/resub_events.csv, resub_aggregate.csv, resub_profile.csv, resub_kernel.csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from pulse import cases, conv_model, pi_design
from pulse.counts import day_panel
from pulse.daily_step import anchors, profile
from pulse.rd_phase import dues, rd, weekend_test
from pulse.signals import RESULTS as R

NEW, RESUB = (2, 6, 10), (3, 7, 11)


def investigator() -> None:
    lk, a, c, posts = pi_design.base()
    base_ = c[(c["type"] == "1") & ~c["rfa"] & ~c["ambiguous"]]
    out = []
    for lab, e in (("A0, new-application date", base_[~base_["a1"]]), ("A1, resubmission date", base_[base_["a1"]])):
        es = pi_design.event_study(e, posts).assign(design=lab, events=len(e[["pi", "due"]].drop_duplicates()))
        out.append(es)
        print(lab, es[es["k"].between(-4, 4)].round(3).to_string(index=False), flush=True)
    pd.concat(out).to_csv(R / "resub_events.csv", index=False)


def aggregate() -> None:
    rows, prof = [], []
    for server in ("biorxiv", "medrxiv"):
        d = cases.load(server)
        g = np.where(d["us"] == 1, "US", "nonUS")
        p = day_panel(d, pd.Series(g), ["nonUS", "US"], d["date"].min(), d["date"].max())
        us = (p["g"] == "US").to_numpy().astype(int)
        prof += [dict(r, anchor="resubmission") for r in profile(p, us, anchors(RESUB, 0), server)]
        sets = {
            "R01 date": [("pooled", dues(NEW, business=True))],
            "resubmission date": [("pooled", dues(RESUB, business=True))]
            + [(m, dues((k,), business=True)) for m, k in (("March", 3), ("July", 7), ("November", 11))],
            "placebo, 45 days after R01 date": [("pooled", dues(NEW, shift=45))],
        }
        for anchor, specs in sets.items():
            for cyc, anc in specs:
                for h in (7, 10):
                    b, se, nc = rd(p, us, anc, h)
                    rows.append(
                        {"server": server, "anchor": anchor, "cycles": cyc, "bandwidth": h, "drop": b, "se": se}
                    )
                    print(rows[-1], flush=True)
        w = weekend_test(p, us, RESUB)
        rows.append({"server": server, "anchor": "resubmission date", "cycles": "weekend test", **w})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(R / "resub_aggregate.csv", index=False)
    pd.DataFrame(prof).to_csv(R / "resub_profile.csv", index=False)


def kernel() -> None:
    from scipy.optimize import nnls

    K = conv_model.kernel()
    dd = pd.read_csv(R / "pi_duedist.csv")
    n_new = dd[dd["schedule"].str.startswith("NEW")].groupby("schedule")["n"].sum()
    n_a1 = dd[dd["schedule"].str.startswith("RESUB") & (dd["type"] == 1)].groupby("schedule")["n"].sum()
    P_new = conv_model.predict(K, n_new, lambda s: True)
    P_a1 = conv_model.predict(K, n_a1, lambda s: True)
    t = pd.read_csv(R / "phase_scan.csv")
    rows = []
    for srv in ("biorxiv", "medrxiv"):
        o = t[t["scan"] == f"{srv} aggregate"].sort_values("bin")
        y, w = o["coef"].to_numpy(), 1 / o["se"].to_numpy() ** 2
        sw = np.sqrt(w)
        for lab, X in (
            ("new applications", P_new[:, None]),
            ("new applications + A1, separate amplitudes", np.column_stack([P_new, P_a1])),
        ):
            amp, _ = nnls(X * sw[:, None], y * sw)
            res = y - X @ amp
            r2 = 1 - (w * res**2).sum() / (w * (y - np.average(y, weights=w)) ** 2).sum()
            rows.append(
                {
                    "server": srv,
                    "model": lab,
                    "amp_new": amp[0],
                    "amp_a1": amp[1] if len(amp) > 1 else np.nan,
                    "a1_to_new": amp[1] / amp[0] if len(amp) > 1 and amp[0] > 0 else np.nan,
                    "weighted_r2": r2,
                    "chi2": float((w * res**2).sum()),
                    "funded_new": int(n_new.sum()),
                    "funded_a1": int(n_a1.sum()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(R / "resub_kernel.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    {"investigator": investigator, "aggregate": aggregate, "kernel": kernel}[sys.argv[1]]()
