"""Phase scans on Poisson daily counts: where in the NIH cycle does each contrast move?

Every scan is a 7-day-bin profile of a log count ratio over phi (days to the next R01 due date, 0-122), from a
Poisson model on day x group counts with day FE, group x cycle FE and group x weekday FE. Bin coefficients are
centred so the bins average zero; SEs from the centred contrast of the clustered (by cycle) covariance.

  aggregate   log(US / non-US), both servers
  hiv         US-HIV vs US-other, non-US-HIV vs non-US-other, and the difference (triple), bioRxiv and medRxiv
  intramural  log(US intramural / US extramural), both servers
  field       per bioRxiv category log(US / non-US) profile; per bin, inverse-variance weighted slope on NIH
              dependence. Also the 28-day R01-window coefficient per category -> count-based gradient.

    python -m pulse.phase_scan
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pulse import cases, field_split, signals
from pulse.counts import day_panel

R = signals.RESULTS
NB = 18  # 7-day bins, phi 0..125


def _bins(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    p["bin"] = (p["phi"] // 7).clip(upper=NB - 1).astype(int)
    return p


def profile(p: pd.DataFrame, terms: dict[str, np.ndarray], label: str) -> list[dict]:
    """terms: name -> 0/1 indicator over rows; one centred bin profile per term (all estimated jointly)."""
    import pyfixest as pf

    q = _bins(p)
    names = {}
    for t, ind in terms.items():
        cols = []
        for b in range(1, NB):  # bin 0 is the reference; recentred below
            nm = f"{t}_b{b}"
            q[nm] = ind * (q["bin"].to_numpy() == b)
            cols.append(nm)
        names[t] = cols
    allc = [c for v in names.values() for c in v]
    m = pf.fepois(f"n ~ {' + '.join(allc)} | day + gc + gd", data=q, vcov={"CRV1": "cycle"})
    co, V = m.coef(), pd.DataFrame(m._vcov, index=m.coef().index, columns=m.coef().index)
    rows = []
    for t, cols in names.items():
        b = np.r_[0.0, co[cols].to_numpy()]
        S = np.zeros((NB, NB))
        S[1:, 1:] = V.loc[cols, cols].to_numpy()
        C = np.eye(NB) - 1.0 / NB
        bc, Vc = C @ b, C @ S @ C.T
        for i in range(NB):
            rows.append(
                {
                    "scan": label,
                    "term": t,
                    "bin": i,
                    "phi_mid": 7 * i + 3.5,
                    "coef": bc[i],
                    "se": float(np.sqrt(max(Vc[i, i], 0))),
                }
            )
    return rows


def window_coef(p: pd.DataFrame, ind: np.ndarray, w: str = "win_R_new") -> tuple[float, float]:
    import pyfixest as pf

    q = p.copy()
    q["x"] = ind * q[w].to_numpy()
    m = pf.fepois("n ~ x | day + gc + gd", data=q, vcov={"CRV1": "cycle"})
    return float(m.coef()["x"]), float(m.se()["x"])


def wls_slope(y, se, x):
    w = 1 / np.asarray(se) ** 2
    X = np.column_stack([np.ones(len(x)), x])
    XtWX = X.T @ (X * w[:, None])
    b = np.linalg.solve(XtWX, X.T @ (w * np.asarray(y)))
    r = np.asarray(y) - X @ b
    # heteroskedasticity-robust (HC1) SE for the slope
    meat = X.T @ (X * (w**2 * r**2)[:, None])
    Vb = np.linalg.inv(XtWX) @ meat @ np.linalg.inv(XtWX) * len(x) / (len(x) - 2)
    return b[1], float(np.sqrt(Vb[1, 1])), b[0]


def run() -> None:
    rows, frows = [], []
    for server in ("biorxiv", "medrxiv"):
        d = cases.load(server)
        us = (d["us"] == 1).to_numpy()
        start, end = d["date"].min(), d["date"].max()
        # aggregate
        g = np.where(us, "US", "nonUS")
        p = day_panel(d, pd.Series(g), ["nonUS", "US"], start, end)
        rows += profile(p, {"US": (p["g"] == "US").to_numpy().astype(int)}, f"{server} aggregate")
        # HIV
        h = d[d["date"] <= cases.AIDS_END]
        gh = np.where(h["us"] == 1, "US", "nonUS") + np.where(h["hiv"], "-HIV", "-other")
        p = day_panel(
            h, pd.Series(gh), ["US-HIV", "US-other", "nonUS-HIV", "nonUS-other"], h["date"].min(), h["date"].max()
        )
        is_us = p["g"].str.startswith("US").to_numpy().astype(int)
        is_hiv = p["g"].str.endswith("HIV").to_numpy().astype(int)
        rows += profile(p, {"US": is_us, "HIV": is_hiv, "US_x_HIV": is_us * is_hiv}, f"{server} hiv")
        # intramural vs extramural (US only)
        u = d[us]
        email = u["corresp_email"].fillna("").str.lower()
        inst = u["author_corresponding_institution"].fillna("")
        intra = (email.str.endswith("nih.gov") | inst.str.contains(cases.INTRAMURAL_INST)).to_numpy()
        gi = np.where(intra, "intra", "extra")
        p = day_panel(u, pd.Series(gi), ["extra", "intra"], start, end)
        rows += profile(p, {"intra": (p["g"] == "intra").to_numpy().astype(int)}, f"{server} intramural")
        print(server, "scans done", flush=True)
    # field gradient on counts (bioRxiv)
    d = cases.load("biorxiv")
    fs = field_split.load()
    dep = field_split.per_category(fs).set_index("category")["nih_share_us_funded"]
    dep.index = dep.index.str.strip().str.lower()
    for cat, s in d.groupby("category"):
        if cat not in dep.index:
            continue
        g = np.where(s["us"] == 1, "US", "nonUS")
        p = day_panel(s, pd.Series(g), ["nonUS", "US"], d["date"].min(), d["date"].max())
        ind = (p["g"] == "US").to_numpy().astype(int)
        for w in ("win_R_new", "win_F"):
            b, se = window_coef(p, ind, w)
            frows.append(
                {
                    "category": cat,
                    "dep": dep[cat],
                    "window": w[4:],
                    "coef": b,
                    "se": se,
                    "n": len(s),
                    "n_us": int((s["us"] == 1).sum()),
                }
            )
        rows += [dict(r, category=cat, dep=dep[cat]) for r in profile(p, {"US": ind}, "field")]
        print(cat, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(R / "phase_scan.csv", index=False)
    f = pd.DataFrame(frows)
    f.to_csv(R / "field_counts.csv", index=False)
    srows = []
    for w, s in f.groupby("window"):
        b, se, a0 = wls_slope(s["coef"], s["se"], s["dep"])
        srows.append({"what": f"{w} window coef on dependence", "slope": b, "se": se, "intercept": a0, "k": len(s)})
        for drop in s["category"]:
            t = s[s["category"] != drop]
            bb, _, _ = wls_slope(t["coef"], t["se"], t["dep"])
            srows.append({"what": f"{w} LOO drop {drop}", "slope": bb, "k": len(t)})
    fl = out[out["scan"] == "field"]
    for b_, s in fl.groupby("bin"):
        sl, se, a0 = wls_slope(s["coef"], s["se"], s["dep"])
        srows.append(
            {"what": f"bin {b_} profile slope", "bin": b_, "slope": sl, "se": se, "intercept": a0, "k": len(s)}
        )
    sl = pd.DataFrame(srows)
    sl.to_csv(R / "field_counts_slope.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(sl.round(4).to_string(index=False))
        print(out[out["scan"] != "field"].round(3).to_string(index=False))


if __name__ == "__main__":
    run()
