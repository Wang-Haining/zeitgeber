"""Group rates, year-by-year estimates and funder splits on Poisson daily counts.

groups     per-group Poisson rate ratios for US and non-US daily counts, and the R01 - F contrast in one model;
           wild cluster bootstrap (Rademacher, by cycle) for the US-share R01 and R01 - F estimates
years      US x R01 and US x F on daily counts, year by year (2017/2019 .. 2025), both servers; and a raw
           year x 7-day-bin log(US/non-US) matrix for the heatmap
funders    bioRxiv US first versions split by OpenAlex funders (NIH acknowledged; NSF but not NIH; other funder
           only; none) vs non-US on the same day; also within high-NIH fields

    python -m pulse.group_rates  ->  results/group_rates.csv, results/year_heatmap.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pulse import cases, field_split, signals
from pulse.counts import day_panel

R = signals.RESULTS
WINS = ("win_R_new", "win_F")


def fe_pois(q, terms):
    import pyfixest as pf

    m = pf.fepois(f"n ~ {' + '.join(terms)} | day + gc + gd", data=q, vcov={"CRV1": "cycle"})
    return m


def wild_cluster(y, X, codes, reps=999, seed=1):
    """OLS on cycle-demeaned data; Rademacher wild cluster bootstrap-t for each coefficient (H0: beta=0)."""
    n = codes.max() + 1

    def demean(v):
        s = np.bincount(codes, weights=v, minlength=n) / np.bincount(codes, minlength=n)
        return v - s[codes]

    yd = demean(y)
    Xd = np.column_stack([demean(X[:, j]) for j in range(X.shape[1])])
    XtXi = np.linalg.inv(Xd.T @ Xd)
    b = XtXi @ Xd.T @ yd

    def cr_se(Xd, e):
        G = np.zeros((n, Xd.shape[1]))
        for j in range(Xd.shape[1]):
            G[:, j] = np.bincount(codes, weights=Xd[:, j] * e, minlength=n)
        V = XtXi @ (G.T @ G) @ XtXi * n / (n - 1)
        return np.sqrt(np.diag(V))

    e = yd - Xd @ b
    se = cr_se(Xd, e)
    t_obs = b / se
    rng = np.random.default_rng(seed)
    out = []
    for k in range(Xd.shape[1]):
        # restricted residuals under H0: beta_k = 0
        keep = [j for j in range(Xd.shape[1]) if j != k]
        Xr = Xd[:, keep]
        br = np.linalg.lstsq(Xr, yd, rcond=None)[0]
        er = yd - Xr @ br
        fit = Xr @ br
        ts = []
        for _ in range(reps):
            w = rng.choice([-1.0, 1.0], n)[codes]
            ys = fit + er * w
            bs = XtXi @ Xd.T @ ys
            es = ys - Xd @ bs
            ts.append(bs[k] / cr_se(Xd, es)[k])
        out.append(float((np.abs(np.array(ts)) >= abs(t_obs[k])).mean()))
    return b, se, np.array(out)


def run() -> None:
    rows, heat = [], []
    for server in ("biorxiv", "medrxiv"):
        d = cases.load(server)
        g = np.where(d["us"] == 1, "US", "nonUS")
        p = day_panel(d, pd.Series(g), ["nonUS", "US"], d["date"].min(), d["date"].max())
        us = (p["g"] == "US").to_numpy().astype(int)
        # per-group rate ratios (cycle + weekday FE)
        import pyfixest as pf

        for grp, s in p.groupby("g"):
            for w in WINS:
                m = pf.fepois(f"n ~ {w} | cycle + dow", data=s, vcov={"CRV1": "cycle"})
                rows.append(
                    {
                        "analysis": "per-group rate ratio",
                        "server": server,
                        "group": grp,
                        "term": w[4:],
                        "coef": float(m.coef()[w]),
                        "se": float(m.se()[w]),
                    }
                )
        # R01 - F contrast in one model (US x window, day FE)
        q = p.copy()
        q["xR"] = us * q["win_R_new"].to_numpy()
        q["xF"] = us * q["win_F"].to_numpy()
        m = fe_pois(q, ["xR", "xF"])
        V = pd.DataFrame(m._vcov, index=m.coef().index, columns=m.coef().index)
        diff = m.coef()["xR"] - m.coef()["xF"]
        sed = float(np.sqrt(V.loc["xR", "xR"] + V.loc["xF", "xF"] - 2 * V.loc["xR", "xF"]))
        for k in ("xR", "xF"):
            rows.append(
                {
                    "analysis": "joint US x window (day FE)",
                    "server": server,
                    "group": "US",
                    "term": k,
                    "coef": float(m.coef()[k]),
                    "se": float(m.se()[k]),
                }
            )
        rows.append(
            {
                "analysis": "joint US x window (day FE)",
                "server": server,
                "group": "US",
                "term": "R01 - F",
                "coef": float(diff),
                "se": sed,
            }
        )
        # wild cluster bootstrap on the share model: us ~ R + F | cycle
        codes = pd.factorize(d["cycle"])[0]
        X = d[["win_R_new", "win_F"]].to_numpy(float)
        b, se, pw = wild_cluster(d["us"].to_numpy(float), X, codes)
        Xc = np.column_stack([X[:, 0] - X[:, 1], X[:, 0] + X[:, 1]])  # contrast parameterisation
        bc, sec, pwc = wild_cluster(d["us"].to_numpy(float), Xc / 2, codes)
        for name, bb, ss, pp in (
            ("share R01", b[0], se[0], pw[0]),
            ("share F", b[1], se[1], pw[1]),
            ("share R01 - F", bc[0], sec[0], pwc[0]),
        ):
            rows.append(
                {
                    "analysis": "share, cycle FE, wild cluster bootstrap",
                    "server": server,
                    "group": "US",
                    "term": name,
                    "coef": 100 * float(bb),
                    "se": 100 * float(ss),
                    "p_wild": float(pp),
                    "clusters": int(codes.max() + 1),
                }
            )
        # year by year
        p["year"] = p["date"].dt.year
        for yr, s in p.groupby("year"):
            if len(s) < 300 * 2:
                continue
            s = s.copy()
            u = (s["g"] == "US").to_numpy().astype(int)
            s["xR"] = u * s["win_R_new"].to_numpy()
            s["xF"] = u * s["win_F"].to_numpy()
            try:
                m = fe_pois(s, ["xR", "xF"])
                Vy = pd.DataFrame(m._vcov, index=m.coef().index, columns=m.coef().index)
                dy = m.coef()["xR"] - m.coef()["xF"]
                sdy = float(np.sqrt(Vy.loc["xR", "xR"] + Vy.loc["xF", "xF"] - 2 * Vy.loc["xR", "xF"]))
                for k, lab in (("xR", "R01"), ("xF", "F")):
                    rows.append(
                        {
                            "analysis": "year",
                            "server": server,
                            "group": str(yr),
                            "term": lab,
                            "coef": float(m.coef()[k]),
                            "se": float(m.se()[k]),
                        }
                    )
                rows.append(
                    {
                        "analysis": "year",
                        "server": server,
                        "group": str(yr),
                        "term": "R01 - F",
                        "coef": float(dy),
                        "se": sdy,
                    }
                )
            except Exception as e:  # noqa: BLE001
                print("year", yr, e, flush=True)
        # heatmap: raw log(US/nonUS) by year x 7-day bin, minus the year mean
        p["bin"] = (p["phi"] // 7).clip(upper=17)
        h = p.pivot_table(index=["year", "bin"], columns="g", values="n", aggfunc="sum").reset_index()
        h["lr"] = np.log((h["US"] + 0.5) / (h["nonUS"] + 0.5))
        h["lr"] = h["lr"] - h.groupby("year")["lr"].transform("mean")
        heat.append(h.assign(server=server))
        print(server, "done", flush=True)

    # funders (bioRxiv)
    import pyfixest as pf

    d = field_split.load()
    names = d["funder_names"].apply(
        lambda v: " | ".join(map(str, v)) if isinstance(v, (list, np.ndarray)) else (v if isinstance(v, str) else "")
    )
    nsf = names.str.contains(r"(?:^|\| )National Science Foundation(?: \||$)", regex=True)
    nih = d["nih_oa"] == 1
    funded = d["n_funders"].fillna(0) > 0
    grp = np.select(
        [d["us"] == 0, nih, nsf & ~nih, funded & ~nih & ~nsf],
        ["nonUS", "US NIH", "US NSF not NIH", "US other funder"],
        default="US unfunded",
    )
    dep = field_split.per_category(d).set_index("category")["nih_share_us_funded"]
    for lab, mask in (
        ("all fields", np.ones(len(d), bool)),
        ("high NIH dependence (>= 0.7)", d["category"].map(dep).fillna(0).to_numpy() >= 0.7),
    ):
        s = d[mask]
        gs = grp[mask]
        levels = ["nonUS", "US NIH", "US NSF not NIH", "US other funder", "US unfunded"]
        p = day_panel(s, pd.Series(gs), levels, d["date"].min(), d["date"].max())
        for w in WINS:
            q = p.copy()
            terms = []
            for lv in levels[1:]:
                nm = "g_" + lv.replace(" ", "_")
                q[nm] = (q["g"] == lv).to_numpy().astype(int) * q[w].to_numpy()
                terms.append(nm)
            m = fe_pois(q, terms)
            for lv, nm in zip(levels[1:], terms, strict=False):
                rows.append(
                    {
                        "analysis": f"funder group vs non-US ({lab})",
                        "server": "biorxiv",
                        "group": lv,
                        "term": w[4:],
                        "coef": float(m.coef()[nm]),
                        "se": float(m.se()[nm]),
                        "n_group": int((gs == lv).sum()),
                    }
                )
    out = pd.DataFrame(rows)
    out["z"] = out["coef"] / out["se"]
    out.to_csv(R / "group_rates.csv", index=False)
    pd.concat(heat).to_csv(R / "year_heatmap.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 300):
        print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    run()
