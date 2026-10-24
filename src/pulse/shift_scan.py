"""Step statistic at every assumed due date from 60 days before to 60 days after the true R01 dates (for Fig. 1D).

python -m pulse.shift_scan  ->  results/shift_scan.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pulse import cases, signals
from pulse.counts import day_panel
from pulse.daily_step import anchors, step

R = signals.RESULTS


def run() -> None:
    rows = []
    for server in ("biorxiv", "medrxiv"):
        d = cases.load(server)
        g = np.where(d["us"] == 1, "US", "nonUS")
        p = day_panel(d, pd.Series(g), ["nonUS", "US"], d["date"].min(), d["date"].max())
        us = (p["g"] == "US").to_numpy().astype(int)
        for s in range(-60, 61):
            b, se = step(p, us, anchors((2, 6, 10), s))
            rows.append({"server": server, "shift": s, "step": b, "se": se})
        print(server, "done", flush=True)
    pd.DataFrame(rows).to_csv(R / "shift_scan.csv", index=False)


if __name__ == "__main__":
    run()
