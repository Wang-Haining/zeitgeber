"""Is the step at the R01 due date a Lunar New Year or Golden Week artifact of the comparison group?

Lunar New Year (late January to mid-February) and China's National Day week (1-7 October) lower posting in East Asia
near the February and October R01 due dates, which raises the US share. The step and the discontinuity are
re-estimated (a) with East Asian authors (CN, HK, MO, TW, KR, SG, VN) removed from the comparison group and (b) with
the Lunar New Year holiday weeks and 1-7 October removed from the panel, pooled and for the February cycle.

    python -m pulse.holiday_check  ->  results/holiday_check.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pulse import funder_pulse, signals
from pulse.counts import day_panel
from pulse.daily_step import SHIFTS, anchors, step
from pulse.rd_phase import dues, rd

R = signals.RESULTS
EAST_ASIA = {"CN", "HK", "MO", "TW", "KR", "SG", "VN"}
LNY = [
    "2017-01-28",
    "2018-02-16",
    "2019-02-05",
    "2020-01-25",
    "2021-02-12",
    "2022-02-01",
    "2023-01-22",
    "2024-02-10",
    "2025-01-29",
    "2026-02-17",
]  # first day of the lunar year


def holiday_days() -> set:
    out = set()
    for d in LNY:
        out |= set(pd.date_range(pd.Timestamp(d) - pd.Timedelta(days=1), pd.Timestamp(d) + pd.Timedelta(days=6)))
    for y in range(2017, 2027):
        out |= set(pd.date_range(f"{y}-10-01", f"{y}-10-07"))
    return out


def run() -> None:
    rows = []
    hol = holiday_days()
    for server in ("biorxiv", "medrxiv"):
        d = funder_pulse.load(server)
        d["cycle"] = (d["date"] + pd.to_timedelta(d["phi"], unit="D")).dt.strftime("%Y-%m")
        variants = {
            "all non-US (reference)": d,
            "non-US without East Asia": d[~d["iso2"].isin(EAST_ASIA)],
        }
        for lab, x in variants.items():
            g = np.where(x["iso2"] == "US", "US", "nonUS")
            p = day_panel(x, pd.Series(g), ["nonUS", "US"], d["date"].min(), d["date"].max())
            for drop_hol in (False, True):
                q = p[~p["date"].isin(hol)].reset_index(drop=True) if drop_hol else p
                us = (q["g"] == "US").to_numpy().astype(int)
                for cy, months in (("pooled", (2, 6, 10)), ("Feb", (2,))):
                    b, se = step(q, us, anchors(months, 0))
                    null = np.array([step(q, us, anchors(months, s))[0] for s in SHIFTS])
                    r_, rse, _ = rd(q, us, dues(months=months), 14)
                    rows.append(
                        {
                            "server": server,
                            "comparison": lab,
                            "holidays_removed": drop_hol,
                            "cycles": cy,
                            "step": b,
                            "se": se,
                            "null_sd": null.std(),
                            "rank_p": (1 + (null >= b).sum()) / (len(null) + 1),
                            "rd14": r_,
                            "rd14_se": rse,
                        }
                    )
                    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(R / "holiday_check.csv", index=False)


if __name__ == "__main__":
    run()
