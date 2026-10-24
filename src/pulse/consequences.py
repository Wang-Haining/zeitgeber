"""Consequences of deadline-timed preprinting.

    python -m pulse.consequences scale        # how many US preprints a year are deadline-timed
    python -m pulse.consequences congestion   # do crowded deadline weeks cost citations?
    python -m pulse.consequences ackaward     # are pulse preprints later tied to the upcoming award?

scale: daily US and non-US counts (non-COVID, both servers). Counterfactual US count in the
  R01-new windows = non-US count x the US/non-US ratio outside the windows of the same year;
  excess = observed - counterfactual, summed per year. Also the deficit in F windows.
congestion: bioRxiv; log(1 + citations of preprint + journal version), FE field x year and
  calendar week; regressor = R01-new window x field NIH dependence (fields that crowd more at
  deadlines), for all authors and for non-US authors only (who do not time to NIH, so any loss
  is a crowding externality, not a property of deadline-timed work).
ackaward: linked PIs' preprints with a funded award due within 365 days; outcome = the preprint
  or its journal version acknowledges that award's project number (OpenAlex awards, JATS
  funding text); pulse (0-13 d) vs the same PI's other preprints, controlling log gap.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from pulse import deadlines, field_split, pi_checks, pi_design, signals
from pulse.parse_jats import INTERIM

R = signals.RESULTS


def step_scale():
    rows = []
    frames = []
    for s in ("biorxiv", "medrxiv"):
        v = signals.prep(pd.read_parquet(INTERIM / f"{s}_v1.parquet"))
        frames.append(v[~v["covid"] & v["us"].notna()][["date", "us"]].assign(server=s))
    d = pd.concat(frames)
    days = deadlines.annotate(pd.DataFrame({"date": pd.date_range("2017-01-01", "2026-06-30")}))
    cu = d[d["us"] == 1].groupby("date").size()
    cn = d[d["us"] == 0].groupby("date").size()
    days["us"] = days["date"].map(cu).fillna(0)
    days["nonus"] = days["date"].map(cn).fillna(0)
    days["year"] = days["date"].dt.year
    for (yr), g in days.groupby("year"):
        out = g[(g["win_R_new"] == 0) & (g["win_F"] == 0)]
        ratio = out["us"].sum() / max(out["nonus"].sum(), 1)
        w = g[g["win_R_new"] == 1]
        f = g[g["win_F"] == 1]
        rows.append(
            {
                "year": yr,
                "us_total": g["us"].sum(),
                "ratio_outside": ratio,
                "rnew_observed": w["us"].sum(),
                "rnew_expected": ratio * w["nonus"].sum(),
                "f_observed": f["us"].sum(),
                "f_expected": ratio * f["nonus"].sum(),
            }
        )
    t = pd.DataFrame(rows)
    t["rnew_excess"] = t["rnew_observed"] - t["rnew_expected"]
    t["f_deficit"] = t["f_observed"] - t["f_expected"]
    t["net"] = t["rnew_excess"] + t["f_deficit"]
    t["excess_share_of_us_total"] = t["rnew_excess"] / t["us_total"]
    t.to_csv(R / "consequence_scale.csv", index=False)
    with pd.option_context("display.width", 200):
        print(t.round(3).to_string(index=False))
        full = t[t["year"].between(2018, 2025)]
        print(
            "2018-2025 mean per year: excess in R01 windows",
            round(full["rnew_excess"].mean(), 1),
            "| F deficit",
            round(full["f_deficit"].mean(), 1),
            "| US preprints",
            round(full["us_total"].mean(), 1),
        )


def step_congestion():
    import pyfixest as pf

    d = field_split.load()  # bioRxiv non-COVID, us flag, category, cycle
    dep = field_split.per_category(d).set_index("category")["nih_share_us_funded"]
    pub = pd.read_parquet(INTERIM / "oa_pubinfo.parquet")
    pp = pub[pub["kind"] == "preprint"].drop_duplicates("doi").set_index("doi")["cited_by_count"]
    pj = pub[pub["kind"] == "published"].drop_duplicates("doi").set_index("doi")["cited_by_count"]
    d["doi_l"] = d["doi"].str.lower()
    pdoi = d["published"].fillna("NA").str.lower()
    d["cites"] = d["doi_l"].map(pp).fillna(0) + pdoi.map(pj).fillna(0)
    d["log_cites"] = np.log1p(d["cites"])
    d["dep"] = d["category"].map(dep)
    d = d[d["dep"].notna() & (d["date"] <= pd.Timestamp("2025-06-30"))]
    d["fy"] = d["category"] + "_" + d["year"].astype(str)
    d["wk"] = d["date"].dt.to_period("W").astype(str)
    d["win_dep"] = d["win_R_new"] * d["dep"]
    d["log_crowd"] = np.log(d.groupby(["category", "wk"])["doi"].transform("size").astype(float))
    rows = []
    for name, s in {"all authors": d, "non-US authors": d[d["us"] == 0], "US authors": d[d["us"] == 1]}.items():
        m = pf.feols("log_cites ~ win_dep | fy + wk", data=s, vcov={"CRV1": "wk"})
        m2 = pf.feols("log_cites ~ log_crowd | fy + wk", data=s, vcov={"CRV1": "wk"})
        rows.append(
            {
                "sample": name,
                "term": "R01 window x field NIH share",
                "coef": float(m.coef()["win_dep"]),
                "se": float(m.se()["win_dep"]),
                "n": len(s),
            }
        )
        rows.append(
            {
                "sample": name,
                "term": "log same-field same-week preprints",
                "coef": float(m2.coef()["log_crowd"]),
                "se": float(m2.se()["log_crowd"]),
                "n": len(s),
            }
        )
    out = pd.DataFrame(rows)
    out["z"] = out["coef"] / out["se"]
    out.to_csv(R / "consequence_congestion.csv", index=False)
    with pd.option_context("display.width", 200):
        print(out.round(4).to_string(index=False))


def step_ackaward():
    import pyfixest as pf

    t = pd.read_parquet(INTERIM / "pulse_content_posts.parquet")
    t = t[t["appl_id"].notna()].copy()
    a = pd.read_parquet(INTERIM / "reporter_awards.parquet").drop_duplicates("appl_id")
    a["core"] = a["project_num"].map(pi_design.core_of)
    core = a.set_index("appl_id")["core"]
    t["core"] = t["appl_id"].astype("int64").map(core)
    # acknowledgments: preprint level (OpenAlex awards + JATS funding text) and journal version (OpenAlex)
    m = pd.read_parquet(INTERIM / "oa_match.parquet")[["doi", "awards"]].drop_duplicates("doi")
    fund = []
    for s in ("biorxiv", "medrxiv"):
        v = pd.read_parquet(INTERIM / f"{s}_v1.parquet", columns=["doi", "funding"])
        v["doi"] = v["doi"].str.lower()
        fund.append(v)
    fund = pd.concat(fund).drop_duplicates("doi")
    t = t.merge(m, on="doi", how="left").merge(fund, on="doi", how="left")
    t["pre_cores"] = [pi_checks.cores(f"{x or ''} {y or ''}") for x, y in zip(t["awards"], t["funding"], strict=False)]
    pub = pd.read_parquet(INTERIM / "oa_pubinfo.parquet")
    wid = pub[pub["kind"] == "published"].drop_duplicates("doi").set_index("doi")["work_id"]
    t["j_work"] = t["published_doi"].map(wid)
    ack = pd.read_parquet(INTERIM / "oa_nih_acks.parquet", columns=["work_id", "award_ids"])
    ack = ack[ack["work_id"].isin(set(t["j_work"].dropna()))]
    jc = {
        w: {pi_design.core_of(x) for x in (ids if ids is not None else [])} - {None}
        for w, ids in zip(ack["work_id"], ack["award_ids"], strict=False)
    }
    t["j_cores"] = t["j_work"].map(lambda w: jc.get(w, set()) if isinstance(w, str) else set())
    t["ack_pre"] = [float(c in s) for c, s in zip(t["core"], t["pre_cores"], strict=False)]
    t["ack_journal"] = [float(c in s) for c, s in zip(t["core"], t["j_cores"], strict=False)]
    t["ack_any"] = np.maximum(t["ack_pre"], t["ack_journal"])
    t["pulse_f"] = t["pulse"].astype(float)
    t["lgap"] = np.log1p(t["gap"])
    t["ym"] = pd.to_datetime(t["date"]).dt.strftime("%Y-%m")
    t["has_journal"] = t["j_work"].notna()
    rows = []
    for y in ("ack_pre", "ack_journal", "ack_any"):
        for name, s, ctrl in (
            ("all, control log gap", t, " + lgap"),
            ("gap <= 90 days", t[t["gap"] <= 90], ""),
            ("journal version exists, control log gap", t[t["has_journal"]], " + lgap"),
        ):
            if y == "ack_journal" and not name.startswith("journal"):
                continue
            mdl = pf.feols(f"{y} ~ pulse_f{ctrl} | pi + ym", data=s, vcov={"CRV1": "pi"})
            rows.append(
                {
                    "outcome": y,
                    "spec": name,
                    "coef": float(mdl.coef()["pulse_f"]),
                    "se": float(mdl.se()["pulse_f"]),
                    "n": len(s),
                    "n_pulse": int(s["pulse_f"].sum()),
                    "mean_pulse": float(s.loc[s["pulse"], y].mean()),
                    "mean_other": float(s.loc[~s["pulse"], y].mean()),
                }
            )
    out = pd.DataFrame(rows)
    out["z"] = out["coef"] / out["se"]
    out.to_csv(R / "consequence_ackaward.csv", index=False)
    with pd.option_context("display.width", 220):
        print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    {"scale": step_scale, "congestion": step_congestion, "ackaward": step_ackaward}[sys.argv[1]]()
