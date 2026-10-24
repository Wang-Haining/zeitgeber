"""Does the rise before R01 due dates track NIH dependence across fields?

If the rise follows NIH, it should concentrate in NIH-dependent fields (neuroscience, cancer, immunology) and be near
zero in fields NIH barely funds (plant biology, ecology, evolution, paleontology). A shared academic calendar would
not scale with NIH dependence.

NIH dependence of a field = NIH share among US, funded (OpenAlex funders), non-COVID bioRxiv preprints in that
category. Slope test: inverse-variance weighted regression of the per-category R01-window effect on that share.
The pooled share change is also decomposed into log US and log non-US daily counts (cycle FE).

    python -m pulse.field_split
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pulse import funding_tests, signals
from pulse.parse_jats import INTERIM

LOW_NIH = {
    "plant biology",
    "ecology",
    "evolutionary biology",
    "paleontology",
    "zoology",
    "animal behavior and cognition",
}
HIGH_NIH = {"neuroscience", "cancer biology", "immunology", "cell biology", "molecular biology"}


def load() -> pd.DataFrame:
    v = pd.read_parquet(INTERIM / "biorxiv_v1.parquet")
    v["doi"] = v["doi"].str.lower()
    m = pd.read_parquet(INTERIM / "oa_match.parquet")
    m = m[m.server == "biorxiv"][["doi", "n_funders", "funder_names", "awards"]]
    d = signals.prep(v.merge(m, on="doi", how="left"))
    d = d[~d["covid"] & d["us"].notna()].copy()
    d["category"] = d["category"].str.strip().str.lower()
    d["anchor"] = d["date"] + pd.to_timedelta(d["phi"], unit="D")
    d["cycle"] = d["anchor"].dt.strftime("%Y-%m")
    funded = d["n_funders"].fillna(0) > 0
    d["nih_oa"] = np.nan
    d.loc[funded, "nih_oa"] = funding_tests.oa_nih(d[funded]).astype(float).values
    return d


def per_category(d: pd.DataFrame, min_n: int = 3000) -> pd.DataFrame:
    rows = []
    for cat, s in d.groupby("category"):
        if len(s) < min_n:
            continue
        b, se, n = signals.window_effect(s, "us", "win_R_new", fe="cycle")
        bf, sef, _ = signals.window_effect(s, "us", "win_F", fe="cycle")
        us_f = s[(s["us"] == 1)]
        rows.append(
            {
                "category": cat,
                "n": n,
                "us_share": s["us"].mean(),
                "nih_share_us_funded": us_f["nih_oa"].mean(),
                "n_us_funded": int(us_f["nih_oa"].notna().sum()),
                "pulse_R_new_pp": 100 * b,
                "se_pp": 100 * se,
                "pulse_F_pp": 100 * bf,
                "se_F_pp": 100 * sef,
                "group": "low_NIH" if cat in LOW_NIH else "high_NIH" if cat in HIGH_NIH else "other",
            }
        )
    return pd.DataFrame(rows).sort_values("nih_share_us_funded")


def slope(t: pd.DataFrame, y: str = "pulse_R_new_pp", se: str = "se_pp") -> tuple[float, float, float, float]:
    w = 1 / t[se] ** 2
    X = np.column_stack([np.ones(len(t)), t["nih_share_us_funded"]])
    W = np.diag(w)
    XtWX_inv = np.linalg.inv(X.T @ W @ X)
    beta = XtWX_inv @ X.T @ W @ t[y].to_numpy()
    resid = t[y].to_numpy() - X @ beta
    # sandwich SE (robust to over-dispersion across categories)
    meat = X.T @ np.diag((w * resid) ** 2) @ X
    V = XtWX_inv @ meat @ XtWX_inv
    return float(beta[0]), float(beta[1]), float(np.sqrt(V[1, 1])), float(np.sqrt(V[0, 0]))


def pooled_groups(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g, cats in (("low_NIH", LOW_NIH), ("high_NIH", HIGH_NIH)):
        s = d[d["category"].isin(cats)]
        for w in ("win_R_new", "win_F"):
            b, se, n = signals.window_effect(s, "us", w, fe="cycle")
            rows.append(
                {
                    "group": g,
                    "window": w[4:],
                    "pp": 100 * b,
                    "se_pp": 100 * se,
                    "n": n,
                    "nih_share_us_funded": s.loc[s["us"] == 1, "nih_oa"].mean(),
                }
            )
    lo = rows[0]
    hi = rows[2]
    diff = hi["pp"] - lo["pp"]
    rows.append(
        {
            "group": "high_minus_low",
            "window": "R_new",
            "pp": diff,
            "se_pp": float(np.hypot(hi["se_pp"], lo["se_pp"])),
            "n": hi["n"] + lo["n"],
        }
    )
    return pd.DataFrame(rows)


def count_decomposition(server: str) -> pd.DataFrame:
    """log daily counts of US and non-US preprints, window vs rest of cycle (cycle FE + weekday FE via demeaning)."""
    v = signals.prep(pd.read_parquet(INTERIM / f"{server}_v1.parquet"))
    v = v[~v["covid"] & v["us"].notna()]
    days = pd.DataFrame({"date": pd.date_range(v["date"].min(), v["date"].max())})
    from pulse import deadlines

    days = deadlines.annotate(days)
    days["cycle"] = (days["date"] + pd.to_timedelta(days["phi"], unit="D")).dt.strftime("%Y-%m")
    days["week"] = days["date"].dt.to_period("W").astype(str)
    days["dow"] = days["date"].dt.dayofweek
    rows = []
    for grp, val in (("US", 1.0), ("nonUS", 0.0)):
        c = v[v["us"] == val].groupby("date").size()
        dd = days.assign(y=np.log1p(days["date"].map(c).fillna(0)))
        dd["y"] = dd["y"] - dd.groupby("dow")["y"].transform("mean")
        for w in ("win_R_new", "win_F"):
            b, se, n = signals.window_effect(dd, "y", w, fe="cycle")
            rows.append({"server": server, "group": grp, "window": w[4:], "log_count_effect": b, "se": se})
    return pd.DataFrame(rows)


def main() -> None:
    d = load()
    t = per_category(d)
    t.to_csv(signals.RESULTS / "field_split_categories.csv", index=False)
    a, b, se_b, _ = slope(t)
    _, bf, se_bf, _ = slope(t, "pulse_F_pp", "se_F_pp")
    g = pooled_groups(d)
    g.to_csv(signals.RESULTS / "field_split_groups.csv", index=False)
    cd = pd.concat([count_decomposition("biorxiv"), count_decomposition("medrxiv")])
    cd.to_csv(signals.RESULTS / "count_decomposition.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(t.round(3).to_string(index=False))
        print(f"\nslope of R_new pulse (pp) on NIH share: {b:.2f} (SE {se_b:.2f}); intercept {a:.2f}")
        print(f"slope of F pulse (pp) on NIH share: {bf:.2f} (SE {se_bf:.2f})")
        print(g.round(3).to_string(index=False))
        print(cd.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
