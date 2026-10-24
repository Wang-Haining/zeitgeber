"""Do pre-deadline preprints shrink more on the way to the journal?

Same paper, two abstracts: the preprint (v1) and the journal version (OpenAlex abstract).
Headline ratio estimate extracted from each with pulse.extract; shrinkage = preprint |log ratio|
minus journal |log ratio| (positive = the effect got smaller). Hype change = journal minus preprint
hype per 1000 words.

  A. PI level: pulse preprints (0-13 d before the PI's own due date) vs the same PI's other
     preprints; PI FE + year-month FE, SE by PI.
  B. Aggregate: all rxiv preprints with both abstracts; US vs non-US gap in the R01-new window,
     week FE (signals.phase_contrast).

    python -m pulse.shrink
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pulse import extract, signals
from pulse.parse_jats import INTERIM

R = signals.RESULTS


def rebuild(aii: str) -> str:
    try:
        idx = json.loads(aii)
    except (TypeError, ValueError):
        return ""
    pos = {}
    for w, ps in idx.items():
        for p in ps:
            pos[p] = w
    return " ".join(pos[k] for k in sorted(pos))


def journal_table() -> pd.DataFrame:
    j = pd.read_parquet(INTERIM / "oa_pub_abstracts.parquet").drop_duplicates("doi")
    j["jabs"] = j["aii"].map(rebuild)
    s = pd.DataFrame([extract.summarize(t) for t in j["jabs"]], index=j.index)
    j["j_head_abs_log"] = s["head_abs_log"]
    j["j_head_z"] = s["head_z"]
    j["j_n_ratios"] = s["n_ratios"]
    j["j_hype_pk"] = 1000 * s["hype_nosig"] / s["n_words"].replace(0, np.nan)
    return j.rename(columns={"doi": "published_doi"})[
        ["published_doi", "j_head_abs_log", "j_head_z", "j_n_ratios", "j_hype_pk"]
    ]


def pi_level(j: pd.DataFrame) -> pd.DataFrame:
    import pyfixest as pf

    t = pd.read_parquet(INTERIM / "pulse_content_posts.parquet")
    t = t.merge(j, on="published_doi", how="inner")
    t["shrink"] = t["ab_head_abs_log"] - t["j_head_abs_log"]
    t["hype_change"] = t["j_hype_pk"] - t["hype_pk"]
    t["pulse_f"] = t["pulse"].astype(float)
    t["ym"] = pd.to_datetime(t["date"]).dt.strftime("%Y-%m")
    rows = []
    for y in ("shrink", "hype_change", "j_head_abs_log", "j_hype_pk"):
        s = t[t[y].notna()]
        n_p = int(s["pulse_f"].sum())
        row = {
            "level": "PI (same PI, other preprints)",
            "outcome": y,
            "n": len(s),
            "n_pulse": n_p,
            "mean_pulse": float(s.loc[s["pulse"], y].mean()) if n_p else np.nan,
            "mean_other": float(s.loc[~s["pulse"], y].mean()),
        }
        if n_p >= 15:
            m = pf.feols(f"{y} ~ pulse_f | pi + ym", data=s, vcov={"CRV1": "pi"})
            row.update({"coef": float(m.coef()["pulse_f"]), "se": float(m.se()["pulse_f"])})
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(j: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for server in ("biorxiv", "medrxiv"):
        d = signals.prep(pd.read_parquet(INTERIM / f"{server}_v1.parquet"))
        d = d[~d["covid"]].copy()
        d["published_doi"] = d["published"].fillna("NA").str.lower().where(lambda x: x != "na")
        d = d.merge(j, on="published_doi", how="inner")
        d["shrink"] = d["ab_head_abs_log"] - d["j_head_abs_log"]
        d["hype_change"] = d["j_hype_pk"] - d["hype_pk"]
        for y in ("shrink", "hype_change"):
            s = d[d[y].notna() & d["us"].notna()]
            c = signals.phase_contrast(s, y, "us", "win_R_new", n_perm=200)
            if c is None:
                continue
            rows.append(
                {
                    "level": f"aggregate {server} (US vs non-US, R01 window, week FE)",
                    "outcome": y,
                    "n": c.n,
                    "n_pulse": c.n_exposed_in_window,
                    "coef": c.coef,
                    "se": c.se,
                    "p_perm": c.p_perm,
                    "mean_other": float(s[y].mean()),
                }
            )
        rows.append(
            {
                "level": f"descriptive {server}",
                "outcome": "mean shrink (all with both ratios)",
                "n": int(d["shrink"].notna().sum()),
                "mean_other": float(d["shrink"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    j = journal_table()
    print("journal abstracts:", len(j), "with a headline ratio:", int(j["j_head_abs_log"].notna().sum()), flush=True)
    out = pd.concat([pi_level(j), aggregate(j)], ignore_index=True)
    out["z"] = out["coef"] / out["se"]
    out.to_csv(R / "shrink.csv", index=False)
    with pd.option_context("display.width", 220):
        print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
