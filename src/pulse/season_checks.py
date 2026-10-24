"""Is the field gradient an annual season in disguise (summer fieldwork in ecology and plant biology)?

Per bioRxiv category, the R01- and F-window coefficients of log(US / non-US) daily counts are re-estimated with
group x calendar-month-of-year fixed effects added to the day, group x cycle and group x weekday fixed effects, so
each field's own seasonal profile in each group is absorbed. The window effect is then identified from days inside
the same calendar month. The inverse-variance weighted slope on NIH dependence is compared with the baseline.

    python -m pulse.season_checks  ->  results/season_field_counts.csv, results/season_field_slope.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pulse import cases, field_split, signals
from pulse.counts import day_panel
from pulse.phase_scan import wls_slope

R = signals.RESULTS


def coef(p: pd.DataFrame, ind: np.ndarray, w: str, month_fe: bool) -> tuple[float, float]:
    import pyfixest as pf

    q = p.copy()
    q["x"] = ind * q[w].to_numpy()
    q["gm"] = q["g"] + "_" + q["date"].dt.month.astype(str)
    fe = "day + gc + gd" + (" + gm" if month_fe else "")
    m = pf.fepois(f"n ~ x | {fe}", data=q, vcov={"CRV1": "cycle"})
    return float(m.coef()["x"]), float(m.se()["x"])


def run() -> None:
    d = cases.load("biorxiv")
    dep = field_split.per_category(field_split.load()).set_index("category")["nih_share_us_funded"]
    dep.index = dep.index.str.strip().str.lower()
    rows = []
    g = np.where(d["us"] == 1, "US", "nonUS")
    p = day_panel(d, pd.Series(g), ["nonUS", "US"], d["date"].min(), d["date"].max())
    ind = (p["g"] == "US").to_numpy().astype(int)
    for w in ("win_R_new", "win_F"):
        for mfe in (False, True):
            b, se = coef(p, ind, w, mfe)
            rows.append(
                {"category": "all fields", "dep": np.nan, "window": w[4:], "month_fe": mfe, "coef": b, "se": se}
            )
    for cat, s in d.groupby("category"):
        if cat not in dep.index:
            continue
        g = np.where(s["us"] == 1, "US", "nonUS")
        p = day_panel(s, pd.Series(g), ["nonUS", "US"], d["date"].min(), d["date"].max())
        ind = (p["g"] == "US").to_numpy().astype(int)
        for w in ("win_R_new", "win_F"):
            for mfe in (False, True):
                b, se = coef(p, ind, w, mfe)
                rows.append(
                    {
                        "category": cat,
                        "dep": dep[cat],
                        "window": w[4:],
                        "month_fe": mfe,
                        "coef": b,
                        "se": se,
                        "n": len(s),
                    }
                )
        print(cat, flush=True)
    f = pd.DataFrame(rows)
    f.to_csv(R / "season_field_counts.csv", index=False)
    srows = []
    for (w, mfe), s in f[f["category"] != "all fields"].groupby(["window", "month_fe"]):
        b, se, a0 = wls_slope(s["coef"], s["se"], s["dep"])
        srows.append({"window": w, "month_fe": mfe, "slope": b, "se": se, "intercept": a0, "k": len(s)})
    sl = pd.DataFrame(srows)
    sl.to_csv(R / "season_field_slope.csv", index=False)
    with pd.option_context("display.width", 200):
        print(f[f["category"] == "all fields"].round(4).to_string(index=False))
        print(sl.round(4).to_string(index=False))


if __name__ == "__main__":
    run()
