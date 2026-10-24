"""Does the investigator-level kernel add up to the aggregate rhythm? (Text S2 model, tested.)

Kernel K: pooled excess share (application vs other cycles, no fixed effects) in the 0-3 weeks before the own due date
(pi_phase_profile.csv, negative values set to 0). Due-date mix: funded non-RFA applications by schedule
(pi_duedist.csv). Prediction: n_s * K placed by calendar day before every due date of each schedule, summed, averaged
over the days in each 7-day phi bin (binned as in phase_scan), centred; one free amplitude fitted to the aggregate
US-vs-non-US profile (phase_scan.csv) by inverse-variance weighted least squares.

    python -m pulse.conv_model   ->  results/conv_model.csv, results/conv_pred.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pulse import deadlines
from pulse.pi_design import schedule_index
from pulse.signals import RESULTS as R

NB = 18
DAYS = pd.date_range("2017-02-06", "2026-10-05")  # whole R01 cycles of the sample period
PHI = deadlines.phi(DAYS)
BIN = np.minimum(PHI // 7, NB - 1)  # as phase_scan: bin 17 holds the 1-4 days left of a 120-123-day cycle
MODELS = {
    "new applications": lambda s: s.startswith("NEW"),
    "new applications + AIDS": lambda s: s.startswith("NEW") or s.startswith("AIDS"),
    "all funded schedules (equal weight)": lambda s: True,
}
# benchmarks that ignore the kernel's shape: a flat bump over its four-week support
FLAT = {
    "flat four-week bump, new applications": lambda s: s.startswith("NEW"),
    "flat four-week bump, R01 dates only": lambda s: s == "NEW-5",
}


def kernel() -> np.ndarray:
    p = pd.read_csv(R / "pi_phase_profile.csv").sort_values("bin")
    return np.clip(p["diff"].to_numpy()[:4], 0, None)


def predict(K, n, keep, shift: int = 0) -> np.ndarray:
    """Kernel week j covers days 7j..7j+6 before each due date (shift > 0 moves it that many weeks earlier)."""
    Kx = np.r_[K, 0.0]
    t = DAYS + pd.Timedelta(weeks=shift)
    x = np.zeros(len(DAYS))
    for s in n.index:
        if keep(s):
            idx = schedule_index(s)
            wk = (idx[idx.searchsorted(t, side="left")] - t).days.to_numpy() // 7
            x += n[s] * Kx[np.minimum(wk, len(K))]
    P = np.bincount(BIN, weights=x, minlength=NB) / np.bincount(BIN, minlength=NB)
    return P - P.mean()


def main() -> None:
    K = kernel()
    n = pd.read_csv(R / "pi_duedist.csv").groupby("schedule")["n"].sum()
    t = pd.read_csv(R / "phase_scan.csv")
    rows, preds = [], []
    for srv in ("biorxiv", "medrxiv"):
        o = t[t["scan"] == f"{srv} aggregate"].sort_values("bin")
        y, se = o["coef"].to_numpy(), o["se"].to_numpy()
        w = 1 / se**2
        specs = [(lab, K, keep) for lab, keep in MODELS.items()]
        specs += [(lab, np.ones(len(K)), keep) for lab, keep in FLAT.items()]
        for lab, KK, keep in specs:
            P = predict(KK, n, keep)
            a = (w * P * y).sum() / (w * P * P).sum()
            res = y - a * P
            r2 = 1 - (w * res**2).sum() / (w * (y - np.average(y, weights=w)) ** 2).sum()
            chi2 = float((w * res**2).sum())
            rows.append(
                {
                    "server": srv,
                    "model": lab,
                    "corr": float(np.corrcoef(P, y)[0, 1]),
                    "weighted_r2": r2,
                    "amplitude": a,
                    "chi2": chi2,
                    "dof": NB - 1,
                }
            )
            preds.append(
                pd.DataFrame(
                    {
                        "server": srv,
                        "model": lab,
                        "bin": o["bin"].to_numpy(),
                        "phi_mid": o["phi_mid"].to_numpy(),
                        "pred": a * P,
                        "obs": y,
                        "se": se,
                    }
                )
            )
    out = pd.DataFrame(rows)
    out.to_csv(R / "conv_model.csv", index=False)
    pd.concat(preds).to_csv(R / "conv_pred.csv", index=False)
    print("kernel", K.round(4))
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
