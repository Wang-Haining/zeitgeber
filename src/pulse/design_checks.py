"""Cycle-by-cycle estimates and the HIV phase contrast.

1. Cycle by cycle. An annual confound (academic year, holidays, conferences) shares the 4-month period only as
   the third harmonic and should not give three equal peaks; a deadline mechanism should. For the February,
   June and October cycles separately: US x R01-window and US x F-window on Poisson daily counts (day FE), both
   servers; and the bioRxiv field gradient (21 categories) per cycle.
2. HIV phase shift as one contrast: in one model with both windows, (US x HIV x AIDS) - (US x HIV x R01).

    python -m pulse.design_checks
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pulse import cases, field_split, signals
from pulse.counts import day_panel


def wls_fit(y, se, x):
    """Inverse-variance weighted line with HC1 covariance: (slope, slope_se, intercept, intercept_se)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    w = 1 / np.asarray(se, float) ** 2
    X = np.column_stack([np.ones(len(x)), x])
    A = np.linalg.inv(X.T @ (X * w[:, None]))
    b = A @ X.T @ (w * y)
    r = y - X @ b
    V = A @ (X.T @ (X * (w**2 * r**2)[:, None])) @ A * len(x) / (len(x) - 2)
    return b[1], float(np.sqrt(V[1, 1])), b[0], float(np.sqrt(V[0, 0]))


R = signals.RESULTS


def us_window(p: pd.DataFrame, windows=("win_R_new", "win_F")) -> dict:
    import pyfixest as pf

    q = p.copy()
    ind = (q["g"] == "US").to_numpy().astype(int)
    out = {}
    for w in windows:
        q["x"] = ind * q[w].to_numpy()
        m = pf.fepois("n ~ x | day + gc + gd", data=q, vcov={"CRV1": "cycle"})
        out[w[4:]] = (float(m.coef()["x"]), float(m.se()["x"]))
    return out


def run() -> None:
    import pyfixest as pf

    rows, fields = [], []
    dep = field_split.per_category(field_split.load()).set_index("category")["nih_share_us_funded"]
    dep.index = dep.index.str.strip().str.lower()
    for server in ("biorxiv", "medrxiv"):
        d = cases.load(server)
        for m in (2, 6, 10):
            s = d[d["cyc_month"] == m]
            p = day_panel(
                s, pd.Series(np.where(s["us"] == 1, "US", "nonUS")), ["nonUS", "US"], d["date"].min(), d["date"].max()
            )
            p = p[p["cycle"].str[5:7].astype(int) == m]
            for w, (b, se) in us_window(p).items():
                rows.append(
                    {
                        "check": "per cycle",
                        "server": server,
                        "cycle_month": m,
                        "what": f"US x {w}",
                        "coef": b,
                        "se": se,
                        "n": len(s),
                    }
                )
            if server == "biorxiv":
                fr = []
                for cat, c in s.groupby("category"):
                    if cat not in dep.index:
                        continue
                    pc = day_panel(
                        c,
                        pd.Series(np.where(c["us"] == 1, "US", "nonUS")),
                        ["nonUS", "US"],
                        d["date"].min(),
                        d["date"].max(),
                    )
                    pc = pc[pc["cycle"].str[5:7].astype(int) == m]
                    b, se = us_window(pc, ("win_R_new",))["R_new"]
                    fr.append((cat, dep[cat], b, se))
                f = pd.DataFrame(fr, columns=["cat", "dep", "b", "se"])
                f = f[np.isfinite(f["se"]) & (f["se"] > 0)]
                sl, sse, a0, ase = wls_fit(f["b"], f["se"], f["dep"])
                rows.append(
                    {
                        "check": "per cycle",
                        "server": server,
                        "cycle_month": m,
                        "what": "field gradient slope (R01 window, counts)",
                        "coef": sl,
                        "se": sse,
                        "intercept": a0,
                        "intercept_se": ase,
                        "n": len(f),
                    }
                )
                fields.append(f.assign(cycle_month=m))
            print(server, m, flush=True)
        # HIV phase-shift contrast in one model
        h = d[d["date"] <= cases.AIDS_END]
        gh = np.where(h["us"] == 1, "US", "nonUS") + np.where(h["hiv"], "-HIV", "-other")
        p = day_panel(
            h, pd.Series(gh), ["US-HIV", "US-other", "nonUS-HIV", "nonUS-other"], h["date"].min(), h["date"].max()
        )
        us = p["g"].str.startswith("US").to_numpy().astype(int)
        hv = p["g"].str.endswith("HIV").to_numpy().astype(int)
        names = []
        for w in ("win_AIDS", "win_R_new"):
            for t, ind in (("US", us), ("HIV", hv), ("UH", us * hv)):
                nm = f"{t}_{w[4:]}"
                p[nm] = ind * p[w].to_numpy()
                names.append(nm)
        mdl = pf.fepois(f"n ~ {' + '.join(names)} | day + gc + gd", data=p, vcov={"CRV1": "cycle"})
        co = mdl.coef()
        V = pd.DataFrame(mdl._vcov, index=co.index, columns=co.index)
        c = co["UH_AIDS"] - co["UH_R_new"]
        se = float(
            np.sqrt(V.loc["UH_AIDS", "UH_AIDS"] + V.loc["UH_R_new", "UH_R_new"] - 2 * V.loc["UH_AIDS", "UH_R_new"])
        )
        for nm in ("UH_AIDS", "UH_R_new"):
            rows.append(
                {
                    "check": "HIV joint model",
                    "server": server,
                    "what": nm,
                    "coef": float(co[nm]),
                    "se": float(mdl.se()[nm]),
                }
            )
        rows.append(
            {
                "check": "HIV joint model",
                "server": server,
                "what": "phase shift: US x HIV x (AIDS - R01)",
                "coef": float(c),
                "se": se,
            }
        )
    out = pd.DataFrame(rows)
    out["z"] = out["coef"] / out["se"]
    out.to_csv(R / "design_checks.csv", index=False)
    pd.concat(fields).to_csv(R / "design_checks_fields.csv", index=False)
    with pd.option_context("display.width", 200):
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    run()
