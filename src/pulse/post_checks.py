"""Checks computed from saved results (no raw data needed).

python -m pulse.post_checks  ->  results/field_intercept_profile.csv   per-bin slope and intercept on NIH dependence
                                 results/field_slope_by_cycles.csv     dependence slope from Feb/Oct, Jun, all cycles
                                 results/field_drop_two.csv            dependence slope without ecology, plant biology
                                 results/conv_falsification.csv        kernel shifted -8..+8 weeks; free-phase sinusoid
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pulse.conv_model import BIN, NB, PHI, kernel, predict
from pulse.signals import RESULTS as R


def wls(y, se, x):
    """Inverse-variance weighted line with HC1 standard errors: (coef [a, b], se [a, b])."""
    y, se, x = map(np.asarray, (y, se, x))
    w = 1 / se**2
    X = np.column_stack([np.ones(len(x)), x])
    A = np.linalg.inv(X.T @ (X * w[:, None]))
    b = A @ X.T @ (w * y)
    r = y - X @ b
    V = A @ (X.T @ (X * (w**2 * r**2)[:, None])) @ A * len(x) / (len(x) - 2)
    return b, np.sqrt(np.diag(V))


def intercept_profile() -> None:
    t = pd.read_csv(R / "phase_scan.csv")
    rows = []
    for b_, s in t[t["scan"] == "field"].groupby("bin"):
        b, se = wls(s["coef"], s["se"], s["dep"])
        rows.append(
            {
                "bin": b_,
                "phi_mid": s["phi_mid"].iloc[0],
                "intercept": b[0],
                "intercept_se": se[0],
                "slope": b[1],
                "slope_se": se[1],
            }
        )
    pd.DataFrame(rows).to_csv(R / "field_intercept_profile.csv", index=False)


def slope_by_cycles() -> None:
    d = pd.read_csv(R / "design_checks_fields.csv")
    rows = []
    for lab, ms in (("Feb and Oct cycles", [2, 10]), ("all cycles", [2, 6, 10]), ("Jun cycle", [6])):
        s = d[d["cycle_month"].isin(ms)].assign(w=lambda x: 1 / x["se"] ** 2)
        g = (
            s.groupby(["cat", "dep"])
            .apply(lambda x: pd.Series({"b": np.average(x["b"], weights=x["w"]), "se": 1 / np.sqrt(x["w"].sum())}))
            .reset_index()
        )
        b, se = wls(g["b"], g["se"], g["dep"])
        rows.append({"cycles": lab, "slope": b[1], "se": se[1], "intercept": b[0], "intercept_se": se[0], "k": len(g)})
    pd.DataFrame(rows).to_csv(R / "field_slope_by_cycles.csv", index=False)


def field_drop_two() -> None:
    """Slope of the R01-window effect on NIH dependence with all 21 fields and without the two least dependent
    (ecology, plant biology), on the share (pp) and log-count scales. Standard errors: weighted least squares (known
    variance, as printed in the main text for shares), HC0 (table S fields) and HC1 (counts)."""
    sh = pd.read_csv(R / "field_split_categories.csv").rename(
        columns={"category": "cat", "nih_share_us_funded": "dep", "pulse_R_new_pp": "b", "se_pp": "se"}
    )
    ct = pd.read_csv(R / "field_counts.csv")
    ct = ct[ct["window"] == "R_new"].rename(columns={"category": "cat", "coef": "b"})
    rows = []
    for measure, f in (("share, pp", sh), ("log counts", ct)):
        rest = f[~f["cat"].isin(["ecology", "plant biology"])]
        for lab, g in (("all", f), ("without ecology and plant biology", rest)):
            y, se, x = g["b"].to_numpy(), g["se"].to_numpy(), g["dep"].to_numpy()
            n, w = len(x), 1 / se**2
            X = np.column_stack([np.ones(n), x])
            A = np.linalg.inv(X.T @ (X * w[:, None]))
            b = A @ X.T @ (w * y)
            r = y - X @ b
            V = A @ (X.T @ (X * (w**2 * r**2)[:, None])) @ A
            rows.append(
                {
                    "measure": measure,
                    "fields": lab,
                    "k": n,
                    "slope": b[1],
                    "se_wls": np.sqrt(A[1, 1]),
                    "se_hc0": np.sqrt(V[1, 1]),
                    "se_hc1": np.sqrt(V[1, 1] * n / (n - 2)),
                    "intercept": b[0],
                }
            )
    pd.DataFrame(rows).to_csv(R / "field_drop_two.csv", index=False)


def conv_falsification() -> None:
    K = kernel()
    n = pd.read_csv(R / "pi_duedist.csv").groupby("schedule")["n"].sum()
    t = pd.read_csv(R / "phase_scan.csv")

    def pred(shift):
        return predict(K, n, lambda s: s.startswith("NEW"), shift)

    def r2(y, w, fit):
        return 1 - (w * (y - fit) ** 2).sum() / (w * (y - np.average(y, weights=w)) ** 2).sum()

    rows = []
    for srv in ("biorxiv", "medrxiv"):
        o = t[t["scan"] == f"{srv} aggregate"].sort_values("bin")
        y, se = o["coef"].to_numpy(), o["se"].to_numpy()
        w = 1 / se**2
        for k in range(-8, 9):
            P = pred(k)
            a = (w * P * y).sum() / (w * P * P).sum()
            rows.append(
                {
                    "server": srv,
                    "model": f"kernel shifted {k:+d} wk",
                    "shift_weeks": k,
                    "weighted_r2": r2(y, w, a * P),
                    "amp": a,
                }
            )
        cyc = 365.25 / 3  # mean days between R01 due dates
        ph = np.bincount(BIN, weights=PHI + 0.5) / np.bincount(BIN) / cyc * 2 * np.pi  # mean phase of each bin's days
        X = np.column_stack([np.ones(NB), np.cos(ph), np.sin(ph)])
        b = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * y))
        peak = (np.degrees(np.arctan2(b[2], b[1])) % 360) / 360 * cyc
        rows.append(
            {
                "server": srv,
                "model": "best-phase sinusoid (3 parameters: level, amplitude, phase)",
                "weighted_r2": r2(y, w, X @ b),
                "peak_days_before_R01": peak,
            }
        )
    pd.DataFrame(rows).to_csv(R / "conv_falsification.csv", index=False)


def main() -> None:
    intercept_profile()
    slope_by_cycles()
    field_drop_two()
    conv_falsification()


if __name__ == "__main__":
    main()
