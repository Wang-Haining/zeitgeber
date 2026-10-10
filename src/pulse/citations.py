"""Are the preprints the deadline pulls forward better cited than other work in their field?

Citation percentile: citations to a first version and its journal version in the calendar year of posting and the two
following years (OpenAlex counts by year, snapshot of 26 June 2026), ranked within server x subject category x posting
month (0-100, ties averaged). First versions posted through December 2023, so every paper has three complete years.

  A. Within investigator: linked investigators' pre-deadline preprints (0-13 days before the own due date) against the
     same investigator's other preprints; investigator and year-month fixed effects, SE clustered by investigator.
     Outcomes: percentile, top-10% indicator; and the interaction with similarity to the forthcoming award.
  B. Aggregate: US x R01 window on the percentile of all country-coded first versions, with category x month and
     US fixed effects, SE clustered by cycle; divided by the excess share of US window posts to give the implied
     difference for the posts the deadline adds.

    python -m pulse.citations       ->  results/hype_lexicon.csv, citations_within.csv, citations_aggregate.csv
    python -m pulse.citations hype  ->  results/hype_lexicon.csv only
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from pulse import cases, signals
from pulse.parse_jats import INTERIM

R = signals.RESULTS
EXCESS = {
    "biorxiv": 1 - np.exp(-(0.150 - 0.056)),  # share of US R01-window posts the deadline adds: per-group
    "medrxiv": 1 - np.exp(-(0.170 - 0.053)),
}  # R01-window rates, US minus non-US (results/group_rates.csv)


def window_cites(cby, y0: int) -> float:
    if cby is None or (isinstance(cby, float) and np.isnan(cby)):
        return 0.0
    return float(sum(x["n"] for x in cby if y0 <= x["year"] <= y0 + 2))


def percentile_table() -> pd.DataFrame:
    pub = pd.read_parquet(INTERIM / "oa_pubinfo.parquet", columns=["doi", "kind", "cites_by_year"])
    pub["doi"] = pub["doi"].str.lower()
    pre = pub[pub["kind"] == "preprint"].drop_duplicates("doi").set_index("doi")["cites_by_year"]
    jou = pub[pub["kind"] == "published"].drop_duplicates("doi").set_index("doi")["cites_by_year"]
    rows = []
    for s in ("biorxiv", "medrxiv"):
        v = pd.read_parquet(INTERIM / f"{s}_v1.parquet", columns=["doi", "date", "category", "published", "covid"])
        v["server"] = s
        rows.append(v)
    d = pd.concat(rows, ignore_index=True)
    d["doi"] = d["doi"].str.lower()
    d["date"] = pd.to_datetime(d["date"])
    d = d[(d["date"] <= "2023-12-31") & ~d["covid"].fillna(False)].copy()
    y0 = d["date"].dt.year.to_numpy()
    pdoi = d["published"].fillna("").str.lower()
    d["cites3"] = [
        window_cites(pre.get(a), y) + (window_cites(jou.get(b), y) if b and b != "na" else 0.0)
        for a, b, y in zip(d["doi"], pdoi, y0, strict=False)
    ]
    d["ym"] = d["date"].dt.strftime("%Y-%m")
    d["category"] = d["category"].str.strip().str.lower()
    d["cell"] = d["server"] + "|" + d["category"] + "|" + d["ym"]
    d["cite_pct"] = 100 * d.groupby("cell")["cites3"].rank(pct=True, method="average")
    d["top10"] = (d["cite_pct"] > 90).astype(float)
    return d[["doi", "server", "category", "ym", "cell", "cites3", "cite_pct", "top10"]]


def within(pt: pd.DataFrame) -> pd.DataFrame:
    import pyfixest as pf

    t = pd.read_parquet(INTERIM / "pulse_content_posts.parquet")
    t["doi"] = t["doi"].str.lower()
    t = t.merge(pt[["doi", "cite_pct", "top10"]], on="doi", how="inner")
    t["pulse_f"] = t["pulse"].astype(float)
    med = t["sim_next_award"].median()
    t["high_sim"] = (t["sim_next_award"] > med).astype(float)
    rows = []
    for y in ("cite_pct", "top10"):
        m = pf.feols(f"{y} ~ pulse_f | pi + ym", data=t, vcov={"CRV1": "pi"})
        rows.append(
            {
                "outcome": y,
                "term": "pre-deadline",
                "coef": float(m.coef()["pulse_f"]),
                "se": float(m.se()["pulse_f"]),
                "n": len(t),
                "n_pulse": int(t["pulse_f"].sum()),
                "mean_other": float(t.loc[~t["pulse"], y].mean()),
            }
        )
    s = t[t["sim_next_award"].notna()]
    m = pf.feols("cite_pct ~ pulse_f + pulse_f:high_sim + high_sim | pi + ym", data=s, vcov={"CRV1": "pi"})
    co, se = m.coef(), m.se()
    k = [i for i in co.index if i.startswith("pulse_f:")][0]
    rows.append(
        {
            "outcome": "cite_pct",
            "term": "pre-deadline, low similarity to the award",
            "coef": float(co["pulse_f"]),
            "se": float(se["pulse_f"]),
            "n": len(s),
            "n_pulse": int(s.loc[s["high_sim"] == 0, "pulse_f"].sum()),
        }
    )
    rows.append(
        {
            "outcome": "cite_pct",
            "term": "pre-deadline x high similarity to the award",
            "coef": float(co[k]),
            "se": float(se[k]),
            "n": len(s),
            "n_pulse": int(s["pulse_f"].sum()),
        }
    )
    V = m._vcov
    names = list(co.index)
    i, j = names.index("pulse_f"), names.index(k)
    rows.append(
        {
            "outcome": "cite_pct",
            "term": "pre-deadline, high similarity to the award",
            "coef": float(co["pulse_f"] + co[k]),
            "se": float(np.sqrt(V[i, i] + V[j, j] + 2 * V[i, j])),
            "n": len(s),
            "n_pulse": int(s.loc[s["high_sim"] == 1, "pulse_f"].sum()),
        }
    )
    return pd.DataFrame(rows)


def aggregate(pt: pd.DataFrame) -> pd.DataFrame:
    import pyfixest as pf

    rows = []
    for server in ("biorxiv", "medrxiv"):
        d = cases.load(server)
        d["doi"] = d["doi"].str.lower()
        d = d.merge(pt[["doi", "cite_pct", "top10", "cell"]], on="doi", how="inner")
        d["us_f"] = d["us"].astype(float)
        d["x"] = d["us_f"] * d["win_R_new"]
        for y in ("cite_pct", "top10"):
            m = pf.feols(f"{y} ~ x + us_f + win_R_new | cell", data=d, vcov={"CRV1": "cycle"})
            b, se = float(m.coef()["x"]), float(m.se()["x"])
            rows.append(
                {
                    "server": server,
                    "outcome": y,
                    "us_x_window": b,
                    "se": se,
                    "n": len(d),
                    "mean_us_outside": float(d.loc[(d["us_f"] == 1) & (d["win_R_new"] == 0), y].mean()),
                    "excess_share": EXCESS[server],
                    "implied_excess_diff": b / EXCESS[server],
                    "implied_excess_se": se / EXCESS[server],
                }
            )
    return pd.DataFrame(rows)


def hype_full() -> pd.DataFrame:
    """Promotional adjectives with the published 139-word list applied unchanged (Millar et al. 2022, 2023), and
    with "significant" excluded, which the list's authors note mostly marks statistical significance in abstracts."""
    import pyfixest as pf

    t = pd.read_parquet(INTERIM / "pulse_content_posts.parquet")
    t["doi"] = t["doi"].str.lower()
    v = pd.concat(
        [
            pd.read_parquet(INTERIM / f"{s}_v1.parquet", columns=["doi", "ab_hype", "ab_hype_nosig", "ab_n_words"])
            for s in ("biorxiv", "medrxiv")
        ]
    )
    v["doi"] = v["doi"].str.lower()
    t = t.drop(columns=[c for c in ("ab_hype", "ab_hype_nosig", "ab_n_words") if c in t.columns]).merge(v, on="doi")
    w = t["ab_n_words"].replace(0, np.nan)
    t["hype_all_pk"], t["hype_nosig_pk"] = 1000 * t["ab_hype"] / w, 1000 * t["ab_hype_nosig"] / w
    t["pulse_f"] = t["pulse"].astype(float)
    rows = []
    for y, lab in (("hype_all_pk", "full published list"), ("hype_nosig_pk", "list without 'significant'")):
        s = t[t[y].notna()]
        m = pf.feols(f"{y} ~ pulse_f | pi + ym", data=s, vcov={"CRV1": "pi"})
        rows.append(
            {
                "lexicon": lab,
                "coef": float(m.coef()["pulse_f"]),
                "se": float(m.se()["pulse_f"]),
                "n": len(s),
                "n_pulse": int(s["pulse_f"].sum()),
                "mean_other": float(s.loc[~s["pulse"], y].mean()),
            }
        )
    return pd.DataFrame(rows)


def publication(pt: pd.DataFrame) -> pd.DataFrame:
    """Did pre-deadline preprints reach a journal by June 2026, and how were published and unpublished ones cited?"""
    import pyfixest as pf

    t = pd.read_parquet(INTERIM / "pulse_content_posts.parquet")
    t["doi"] = t["doi"].str.lower()
    t = t.merge(pt[["doi", "cite_pct"]], on="doi", how="inner")  # first versions posted through 2023
    t["published_ever"] = (
        t["published_doi"].notna().astype(float)
        if "published_doi" in t
        else (t["published"].fillna("NA").str.upper() != "NA").astype(float)
    )
    t["pulse_f"] = t["pulse"].astype(float)
    rows = []
    m = pf.feols("published_ever ~ pulse_f | pi + ym", data=t, vcov={"CRV1": "pi"})
    rows.append(
        {
            "outcome": "published by June 2026",
            "sample": "all",
            "coef": float(m.coef()["pulse_f"]),
            "se": float(m.se()["pulse_f"]),
            "n": len(t),
            "n_pulse": int(t["pulse_f"].sum()),
            "mean_pulse": float(t.loc[t["pulse"], "published_ever"].mean()),
            "mean_other": float(t.loc[~t["pulse"], "published_ever"].mean()),
        }
    )
    for lab, s in (("published", t[t["published_ever"] == 1]), ("not published", t[t["published_ever"] == 0])):
        m = pf.feols("cite_pct ~ pulse_f | pi + ym", data=s, vcov={"CRV1": "pi"})
        rows.append(
            {
                "outcome": "citation percentile",
                "sample": lab,
                "coef": float(m.coef()["pulse_f"]),
                "se": float(m.se()["pulse_f"]),
                "n": len(s),
                "n_pulse": int(s["pulse_f"].sum()),
                "mean_pulse": float(s.loc[s["pulse"], "cite_pct"].mean()),
                "mean_other": float(s.loc[~s["pulse"], "cite_pct"].mean()),
            }
        )
    return pd.DataFrame(rows)


def run() -> None:
    h = hype_full()
    h.to_csv(R / "hype_lexicon.csv", index=False)
    print(h.round(3).to_string(index=False), flush=True)
    if sys.argv[1:2] == ["hype"]:
        return
    pt = percentile_table()
    print("papers with percentile:", len(pt), flush=True)
    pb = publication(pt)
    pb.to_csv(R / "citations_publication.csv", index=False)
    print(pb.round(3).to_string(index=False), flush=True)
    w = within(pt)
    w.to_csv(R / "citations_within.csv", index=False)
    print(w.round(3).to_string(index=False), flush=True)
    a = aggregate(pt)
    a.to_csv(R / "citations_aggregate.csv", index=False)
    print(a.round(3).to_string(index=False))


if __name__ == "__main__":
    run()
