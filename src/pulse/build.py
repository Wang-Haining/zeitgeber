"""Assemble the analysis table: one row per preprint (version 1), and the DOI lists for the OpenAlex extracts.

    python -m pulse.build <medrxiv|biorxiv>  ->  data/interim/{server}_v1.parquet
    python -m pulse.build dois               ->  data/interim/rxiv_dois.csv, data/interim/pub_dois.parquet

Merges JATS-derived fields (funding, affiliations, results text) when
data/interim/{server}_jats.parquet exists.
"""

from __future__ import annotations

import re
import sys

import pandas as pd

from pulse import deadlines, extract, funding, geo, ror
from pulse.fetch_rxiv import RAW
from pulse.parse_jats import INTERIM

COVID = re.compile(r"covid|sars-?cov-?2|coronavirus|2019-?ncov|\bpandemic\b|lockdown", re.I)
HIV = re.compile(r"\bHIV\b|\bAIDS\b|antiretroviral|\bPrEP\b", re.I)


def load_v1(server: str) -> pd.DataFrame:
    df = pd.concat(pd.read_parquet(f) for f in sorted((RAW / server).glob("*.parquet")))
    df["version"] = df["version"].astype(int)
    n_versions = df.groupby("doi")["version"].max().rename("n_versions")
    v1 = df[df["version"] == 1].drop_duplicates("doi").merge(n_versions, on="doi", how="left")
    v1["date"] = pd.to_datetime(v1["date"])
    return v1


def build(server: str) -> pd.DataFrame:
    v1 = load_v1(server)
    jats_path = INTERIM / f"{server}_jats.parquet"
    if jats_path.exists():
        j = pd.read_parquet(jats_path).drop(columns=["doi"], errors="ignore")
        v1["doi_suffix"] = v1["doi"].str.split("/").str[-1]
        v1 = v1.merge(j, on="doi_suffix", how="left").drop(columns=["doi_suffix"])
    for c in ("corresp_email", "corresp", "first_aff_country", "funding", "results_text"):
        if c not in v1:
            v1[c] = None

    v1["country"] = [
        geo.corresponding_country(e, c, i, f)
        for e, c, i, f in zip(
            v1.corresp_email, v1.corresp, v1.author_corresponding_institution, v1.first_aff_country, strict=False
        )
    ]
    ror_cache = ror.load_cache()
    from_ror = v1["author_corresponding_institution"].map(
        lambda s: {"": None, "US": "US"}.get(ror_cache.get(s, ""), "nonUS") if s in ror_cache else None
    )
    v1["country_source"] = v1["country"].notna().map({True: "text", False: None})
    v1.loc[v1["country"].isna() & from_ror.notna(), "country_source"] = "ror"
    v1["country"] = v1["country"].fillna(from_ror)

    text = v1["title"].fillna("") + " " + v1["abstract"].fillna("")
    v1["covid"] = text.str.contains(COVID)
    v1["hiv"] = text.str.contains(HIV) | v1["category"].str.contains("hiv", case=False, na=False)

    fund = pd.DataFrame([funding.classify(t) for t in v1["funding"].fillna("")], index=v1.index)
    v1 = pd.concat([v1, fund], axis=1)
    v1["has_funding_text"] = v1["funding"].fillna("").str.len() > 0

    ab = pd.DataFrame([extract.summarize(t) for t in v1["abstract"].fillna("")], index=v1.index)
    v1 = pd.concat([v1, ab.add_prefix("ab_")], axis=1)
    rs = pd.DataFrame([extract.summarize(t) for t in v1["results_text"].fillna("")], index=v1.index)
    v1 = pd.concat(
        [
            v1,
            rs[
                [
                    "n_ratios",
                    "head_abs_log",
                    "max_abs_log",
                    "median_abs_z",
                    "n_p_exact",
                    "n_p_caliper_below",
                    "n_p_caliper_above",
                    "n_sample",
                ]
            ].add_prefix("rs_"),
        ],
        axis=1,
    )

    v1 = deadlines.annotate(v1, "date")
    v1["year"] = v1["date"].dt.year
    v1["week"] = v1["date"].dt.to_period("W").astype(str)
    v1["published_flag"] = v1["published"].fillna("NA").ne("NA")
    keep = [c for c in v1.columns if c not in ("results_text", "abstract_jats", "jatsxml", "corresp")]
    return v1[keep]


def doi_lists() -> None:
    """Preprint DOIs with their server, and preprint plus journal-version DOIs, for oa_dump."""
    v = {s: pd.read_parquet(INTERIM / f"{s}_v1.parquet", columns=["doi", "published"]) for s in ("medrxiv", "biorxiv")}
    d = pd.concat(x[["doi"]].assign(server=s) for s, x in v.items())
    d["doi"] = d["doi"].str.lower()
    d.drop_duplicates("doi").to_csv(INTERIM / "rxiv_dois.csv", index=False)
    rows = []
    for x in v.values():
        rows.append(pd.DataFrame({"doi": x["doi"].str.lower(), "kind": "preprint"}))
        p = x["published"].fillna("NA")
        rows.append(pd.DataFrame({"doi": p[p.ne("NA")].str.lower(), "kind": "published"}))
    p = pd.concat(rows).drop_duplicates("doi")
    p = p[p["doi"].notna()].assign(doi=lambda t: t["doi"].str.strip())
    p.to_parquet(INTERIM / "pub_dois.parquet", index=False)


def main(server: str) -> None:
    out = build(server)
    INTERIM.mkdir(parents=True, exist_ok=True)
    out.to_parquet(INTERIM / f"{server}_v1.parquet", index=False)
    print(f"{server}: {len(out)} v1 preprints; country {out.country.value_counts(dropna=False).to_dict()}")


if __name__ == "__main__":
    doi_lists() if sys.argv[1] == "dois" else main(sys.argv[1])
