"""Does the investigator-level kernel add up to the aggregate rhythm? (Text S2 model, tested.)

Kernel K: within-investigator excess share in the 0-3 weeks before an investigator's own due date
(pi_phase_profile.csv, negative values set to 0). Due-date mix: funded non-RFA applications by schedule
(pi_duedist.csv). Prediction for each 7-day bin of the NIH cycle: sum over schedules of n_s * K placed before the
schedule's due date, centred; one free amplitude fitted to the aggregate US-vs-non-US profile (phase_scan.csv) by
inverse-variance weighted least squares.

    python -m pulse.conv_model   ->  results/conv_model.csv, results/conv_pred.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pulse.signals import RESULTS as R

NB = 18
PHI_DUE = {
    "NEW-5": 0,
    "NEW-12": 115,
    "NEW-16": 111,
    "NEW-25": 102,
    "AIDS-7": 29,
    "RESUB-5": 92,
    "RESUB-12": 85,
    "RESUB-16": 81,
}  # days until the next R01 date at each schedule's due day
MODELS = {
    "new applications": lambda s: s.startswith("NEW"),
    "new applications + AIDS": lambda s: s.startswith("NEW") or s.startswith("AIDS"),
    "all funded schedules (equal weight)": lambda s: True,
}


def kernel() -> np.ndarray:
    p = pd.read_csv(R / "pi_phase_profile.csv").sort_values("bin")
    return np.clip(p["diff"].to_numpy()[:4], 0, None)


def predict(K, n, keep) -> np.ndarray:
    P = np.zeros(NB)
    for s, phi in PHI_DUE.items():
        if s in n.index and keep(s):
            for j, k in enumerate(K):
                P[int(((phi + 7 * j) % (7 * NB)) // 7)] += n[s] * k
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
        for lab, keep in MODELS.items():
            P = predict(K, n, keep)
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
