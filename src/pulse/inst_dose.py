"""Within-field, across-institution dose-response (separates NIH dependence from field).

Institution NIH dependence = leave-one-out NIH share among the institution's funded
preprints (OpenAlex funders), last-author institution, US institutions with >= 30
funded preprints. Test: among US preprints, within field x cycle, is the mean
institutional NIH dependence higher for preprints posted in the R01-new window
(and lower in the F window)? Also by institution type (education vs healthcare/
facility) to separate medical schools, and a tertile profile.

    python -m pulse.inst_dose
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pulse import funding_tests, signals
from pulse.parse_jats import INTERIM


def load(server: str) -> pd.DataFrame:
    v = pd.read_parquet(INTERIM / f"{server}_v1.parquet")
    v["doi"] = v["doi"].str.lower()
    m = pd.read_parquet(INTERIM / "oa_match.parquet").drop_duplicates("doi")
    m = m[m.server == server][
        [
            "doi",
            "n_funders",
            "funder_names",
            "awards",
            "last_inst_id",
            "last_inst_name",
            "last_inst_country",
            "last_inst_type",
        ]
    ]
    d = signals.prep(v.merge(m, on="doi", how="inner"))
    d = d[~d["covid"] & (d["last_inst_country"] == "US") & d["last_inst_id"].notna()].copy()
    d["category"] = d["category"].str.strip().str.lower()
    d["cycle"] = (d["date"] + pd.to_timedelta(d["phi"], unit="D")).dt.strftime("%Y-%m")
    funded = d["n_funders"].fillna(0) > 0
    d["nih_oa"] = np.nan
    d.loc[funded, "nih_oa"] = funding_tests.oa_nih(d[funded]).astype(float).values
    g = d.groupby("last_inst_id")["nih_oa"]
    s, n = g.transform("sum"), g.transform("count")
    own = d["nih_oa"].fillna(0)
    nn = n - d["nih_oa"].notna().astype(int)
    d["inst_dep"] = ((s - own) / nn).where(nn >= 30)
    d["fe"] = d["category"] + "|" + d["cycle"]
    return d[d["inst_dep"].notna()]


def run() -> pd.DataFrame:
    rows = []
    for server in ("biorxiv", "medrxiv"):
        d = load(server)
        print(
            server,
            "US preprints with institution dependence:",
            len(d),
            "institutions:",
            d["last_inst_id"].nunique(),
            flush=True,
        )
        subsets = {
            "all": d,
            "education": d[d["last_inst_type"] == "education"],
            "healthcare/facility": d[d["last_inst_type"].isin(["healthcare", "facility"])],
        }
        for name, s in subsets.items():
            for w in ("win_R_new", "win_F"):
                b, se, n = signals.window_effect(s, "inst_dep", w, fe="fe")
                rows.append(
                    {
                        "server": server,
                        "subset": name,
                        "window": w[4:],
                        "coef": b,
                        "se": se,
                        "n": n,
                        "z": b / se if se else np.nan,
                        "mean_inst_dep": s["inst_dep"].mean(),
                        "sd_inst_dep": s["inst_dep"].std(),
                        "n_inst": s["last_inst_id"].nunique(),
                    }
                )
        # tertile composition: share of top-tertile-dependence institutions among US preprints
        q = d.groupby("category")["inst_dep"].transform(lambda x: x.rank(pct=True))
        d["top"] = (q > 2 / 3).astype(float)
        d["bottom"] = (q <= 1 / 3).astype(float)
        for y in ("top", "bottom"):
            for w in ("win_R_new", "win_F"):
                b, se, n = signals.window_effect(d, y, w, fe="fe")
                rows.append(
                    {
                        "server": server,
                        "subset": f"share of {y}-tertile institutions (within field)",
                        "window": w[4:],
                        "coef": b,
                        "se": se,
                        "n": n,
                        "z": b / se if se else np.nan,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(signals.RESULTS / "inst_dose.csv", index=False)
    return out


if __name__ == "__main__":
    with pd.option_context("display.width", 200):
        print(run().round(4).to_string(index=False))
