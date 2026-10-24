"""What the pre-deadline preprints are: similarity to the forthcoming award and other content measures.

Pre-deadline preprint: a linked investigator's last-author US preprint posted 0-13 days before that investigator's
own due date (type-1 or type-2 award, non-RFA, unambiguous notice month). Comparison: the same investigator's other
preprints (investigator and year-month FE), SE clustered by investigator. Window contrasts are also rescaled by the
excess share in the 14-day window, 1 - exp(-0.283), to the implied difference for the preprints the deadline adds.

    python -m pulse.pulse_content
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pulse import pi_design, signals
from pulse.parse_jats import INTERIM

R = signals.RESULTS
EXCESS = 1 - np.exp(-0.283)


def preprint_table() -> pd.DataFrame:
    cols = [
        "doi",
        "date",
        "server",
        "category",
        "title",
        "abstract",
        "ab_hype_nosig",
        "ab_n_words",
        "ab_head_abs_log",
        "rs_head_abs_log",
        "rs_median_abs_z",
        "ab_n_sample",
        "ab_share_p_sig",
        "n_versions",
        "published",
    ]
    rows = []
    for s in ("biorxiv", "medrxiv"):
        v = pd.read_parquet(INTERIM / f"{s}_v1.parquet")
        v = v[[c for c in cols if c in v.columns]].assign(server=s)
        rows.append(v)
    d = pd.concat(rows, ignore_index=True)
    d["doi"] = d["doi"].str.lower()
    d["date"] = pd.to_datetime(d["date"])
    d["hype_pk"] = 1000 * d["ab_hype_nosig"] / d["ab_n_words"].replace(0, np.nan)
    d["log_n"] = np.log10(d["ab_n_sample"])
    d["published_doi"] = d["published"].fillna("NA").str.lower().where(lambda x: x != "na")
    pub = pd.read_parquet(INTERIM / "oa_pubinfo.parquet")
    pp = pub[pub["kind"] == "preprint"].drop_duplicates("doi").set_index("doi")
    pj = pub[pub["kind"] == "published"].drop_duplicates("doi").set_index("doi")
    d["preprint_cites"] = d["doi"].map(pp["cited_by_count"])
    d["journal_date"] = pd.to_datetime(d["published_doi"].map(pj["publication_date"]), errors="coerce")
    d["journal_cites"] = d["published_doi"].map(pj["cited_by_count"])
    d["days_to_journal"] = (d["journal_date"] - d["date"]).dt.days
    d["days_to_journal"] = d["days_to_journal"].where(d["days_to_journal"] >= 0)
    follow = d["date"] <= pd.Timestamp("2024-06-26")  # 24 months before the snapshot
    d["published_24m"] = ((d["days_to_journal"] <= 730).astype(float)).where(follow)
    d["log_days_to_journal"] = np.log1p(d["days_to_journal"])
    d["log_cites"] = np.log1p(d["journal_cites"].fillna(0) + d["preprint_cites"].fillna(0))
    return d


def events_and_posts():
    lk, a, c, posts = pi_design.base()
    e = c[(~c["rfa"]) & (~c["ambiguous"])][["pi", "due", "appl_id", "type", "a1"]].drop_duplicates(
        ["pi", "due", "appl_id"]
    )
    return lk, a, e, posts


def tag_pulse(posts: pd.DataFrame, e: pd.DataFrame) -> pd.DataFrame:
    m = posts.merge(e[["pi", "due", "appl_id"]], on="pi", how="left")
    m["gap"] = (m["due"] - m["date"]).dt.days
    m["pulse"] = m["gap"].between(0, 13)
    m["upcoming"] = m["gap"].between(0, 365)
    best = m[m["upcoming"]].sort_values("gap").drop_duplicates(["pi", "doi"])[["pi", "doi", "appl_id", "gap"]]
    pul = m.groupby(["pi", "doi"])["pulse"].any().rename("pulse").reset_index()
    out = posts.merge(pul, on=["pi", "doi"]).merge(best, on=["pi", "doi"], how="left")
    return out


def similarity(t: pd.DataFrame, a: pd.DataFrame) -> pd.DataFrame:
    """TF-IDF cosine between each preprint (title+abstract) and (i) the PI's upcoming award abstract,
    (ii) the mean of the PI's other award abstracts (placebo)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    aw = a.drop_duplicates("appl_id")[["appl_id", "title", "abstract"]]
    aw["text"] = aw["title"].fillna("") + " " + aw["abstract"].fillna("")
    t = t.copy()
    t["text"] = t["title"].fillna("") + " " + t["abstract"].fillna("")
    vec = TfidfVectorizer(min_df=3, max_df=0.5, stop_words="english", sublinear_tf=True)
    vec.fit(pd.concat([aw["text"], t["text"]]))
    A = vec.transform(aw["text"])
    P = vec.transform(t["text"])
    aidx = dict(zip(aw["appl_id"], range(len(aw)), strict=False))
    pi_awards = a.explode("all_pi_ids").groupby("all_pi_ids")["appl_id"].agg(lambda s: list(set(s)))
    sim_next, sim_other = [], []
    for i, r in enumerate(t.itertuples(index=False)):
        own = int(r.appl_id) if not pd.isna(r.appl_id) else None
        if own is None or own not in aidx:
            sim_next.append(np.nan)
            sim_other.append(np.nan)
            continue
        p = P[i]
        sim_next.append(float(p.multiply(A[aidx[own]]).sum()))
        others = [aidx[x] for x in pi_awards.get(r.pi, []) if x != own and x in aidx]
        sim_other.append(float(np.mean([p.multiply(A[j]).sum() for j in others])) if others else np.nan)
    t["sim_next_award"] = sim_next
    t["sim_other_awards"] = sim_other
    t["sim_diff"] = t["sim_next_award"] - t["sim_other_awards"]
    return t


def step_content():
    import pyfixest as pf

    lk, a, e, posts = events_and_posts()
    t = tag_pulse(posts, e)
    d = preprint_table()
    t = t.merge(d.drop(columns=["date"]), on="doi", how="left")
    t["ym"] = t["date"].dt.strftime("%Y-%m")
    t = similarity(t, a)
    t.drop(columns=["text"]).to_parquet(INTERIM / "pulse_content_posts.parquet", index=False)
    outcomes = {
        "rs_head_abs_log": "Results first |log ratio|",
        "ab_head_abs_log": "abstract headline |log ratio|",
        "hype_pk": "hype per 1000 words",
        "log_n": "log10 sample size",
        "ab_share_p_sig": "share P<.05",
        "n_versions": "versions posted",
        "published_24m": "in a journal within 24 months",
        "log_days_to_journal": "log days to journal (published only)",
        "log_cites": "log(1+citations)",
        "sim_next_award": "similarity to the upcoming award",
        "sim_other_awards": "similarity to PI's other awards",
        "sim_diff": "upcoming minus other awards",
    }
    rows = []
    t["pulse_f"] = t["pulse"].astype(float)
    for y, lab in outcomes.items():
        s = t[t[y].notna()]
        if s["pulse_f"].sum() < 30:
            continue
        m = pf.feols(f"{y} ~ pulse_f | pi + ym", data=s, vcov={"CRV1": "pi"})
        b, se = float(m.coef()["pulse_f"]), float(m.se()["pulse_f"])
        rows.append(
            {
                "outcome": lab,
                "var": y,
                "coef": b,
                "se": se,
                "z": b / se,
                "n": len(s),
                "n_pulse": int(s["pulse_f"].sum()),
                "mean_nonpulse": float(s.loc[~s["pulse"], y].mean()),
                "implied_excess_diff": b / EXCESS,
                "implied_excess_se": se / EXCESS,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(R / "pulse_content.csv", index=False)
    with pd.option_context("display.width", 220):
        print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    step_content()
