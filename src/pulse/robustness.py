"""Robustness of the field gradient and of the aggregate estimate.

1. Field dependence (field_split): leave-one-category-out slope, unweighted slope, per-period slopes, and the slope
   after scaling each effect by p(1-p).
2. The 2025 shutdown: due dates of 5 October and 5 November 2025 moved to 8 December 2025 (NOT-OD-26-012).
3. The aggregate R01-window estimate with SE clustered by cycle instead of week.

    python -m pulse.robustness  ->  results/field_split_robustness.csv, results/d0_cycle_clustered.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pulse import field_split, signals
from pulse.parse_jats import INTERIM


def field_robustness(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    t = field_split.per_category(d)
    a, b, se, _ = field_split.slope(t)
    rows.append({"spec": "weighted, all categories", "slope": b, "se": se, "intercept": a, "k": len(t)})
    # unweighted OLS
    X = np.column_stack([np.ones(len(t)), t["nih_share_us_funded"]])
    beta, *_ = np.linalg.lstsq(X, t["pulse_R_new_pp"], rcond=None)
    res = t["pulse_R_new_pp"] - X @ beta
    V = np.linalg.inv(X.T @ X) @ (X.T @ np.diag(res**2) @ X) @ np.linalg.inv(X.T @ X)
    rows.append({"spec": "unweighted", "slope": beta[1], "se": np.sqrt(V[1, 1]), "intercept": beta[0], "k": len(t)})
    # scaled by p(1-p)
    ts = t.assign(
        pulse_scaled=t["pulse_R_new_pp"] / (t["us_share"] * (1 - t["us_share"])),
        se_scaled=t["se_pp"] / (t["us_share"] * (1 - t["us_share"])),
    )
    a2, b2, se2, _ = field_split.slope(ts, "pulse_scaled", "se_scaled")
    rows.append(
        {
            "spec": "pulse / p(1-p) (relative US-count change, x100)",
            "slope": b2,
            "se": se2,
            "intercept": a2,
            "k": len(t),
        }
    )
    # leave one out
    loo = []
    for c in t["category"]:
        aa, bb, ss, _ = field_split.slope(t[t["category"] != c])
        loo.append((c, bb, ss))
    lo = min(loo, key=lambda x: x[1])
    hi = max(loo, key=lambda x: x[1])
    rows.append({"spec": f"leave-one-out min (drop {lo[0]})", "slope": lo[1], "se": lo[2], "k": len(t) - 1})
    rows.append({"spec": f"leave-one-out max (drop {hi[0]})", "slope": hi[1], "se": hi[2], "k": len(t) - 1})
    # periods
    for name, (y0, y1) in signals.periods("biorxiv").items():
        tp = field_split.per_category(d[(d.year >= y0) & (d.year <= y1)], min_n=1500)
        aa, bb, ss, _ = field_split.slope(tp)
        rows.append({"spec": f"period {name}", "slope": bb, "se": ss, "intercept": aa, "k": len(tp)})
    return pd.DataFrame(rows)


def d0_cycle_clustered() -> pd.DataFrame:
    rows = []
    for s in ("biorxiv", "medrxiv"):
        v = signals.prep(pd.read_parquet(INTERIM / f"{s}_v1.parquet"))
        v = v[~v["covid"] & v["us"].notna()].copy()
        v["cycle"] = (v["date"] + pd.to_timedelta(v["phi"], unit="D")).dt.strftime("%Y-%m")
        for w in ("win_R_new", "win_AIDS", "win_F", "win_R_resub"):
            codes, uniq = pd.factorize(v["cycle"])
            beta, resid, Xd, XtX_inv = signals._fit(v["us"].to_numpy(float), v[[w]].to_numpy(float), codes, len(uniq))
            S = np.bincount(codes, weights=Xd[:, 0] * resid)
            se = float(np.sqrt(XtX_inv[0, 0] ** 2 * (S @ S) * len(uniq) / (len(uniq) - 1)))
            rows.append(
                {
                    "server": s,
                    "window": w[4:],
                    "pp": 100 * beta[0],
                    "se_pp_cycle_cluster": 100 * se,
                    "n_cycles": len(uniq),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    fr = field_robustness(field_split.load())
    fr.to_csv(signals.RESULTS / "field_split_robustness.csv", index=False)
    cc = d0_cycle_clustered()
    cc.to_csv(signals.RESULTS / "d0_cycle_clustered.csv", index=False)
    with pd.option_context("display.width", 200):
        print(fr.round(3).to_string(index=False))
        print(cc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
