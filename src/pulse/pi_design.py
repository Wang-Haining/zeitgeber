"""Within-investigator design: linked NIH investigators and their own due dates.

    python -m pulse.pi_design link                                     # investigator links -> pi_links.parquet
    python -m pulse.pi_design within <main|failed_a0|renewal|itt> 200  # estimates and clean placebo per design
    python -m pulse.pi_design phase_profile 200                        # weekly profile before the own due date
    python -m pulse.pi_design events                                   # event study and 12-month totals
    python -m pulse.pi_design norxiv 100                               # links rebuilt without preprints
    python -m pulse.pi_design activity 100                             # R01 vs R21/R03
    python -m pulse.pi_design family 100                               # investigator x cycle-family FE
    python -m pulse.pi_design <main|sim|bymonth|totals|decomp|bycycle|adjacent|duedist|moderators>

Inputs (data/interim): reporter_awards.parquet, oa_nih_acks.parquet, oa_last_author_names.parquet,
oa_match.parquet, oa_author_firstyear.parquet, {server}_v1.parquet.
"""

from __future__ import annotations

import re
import sys
import unicodedata

import numpy as np
import pandas as pd

from pulse import pi_checks, signals
from pulse.parse_jats import INTERIM

R = signals.RESULTS
RNG = np.random.default_rng(7)
CORE_RE = re.compile(
    r"(?<![a-z])(ai|hl|mh|ca|da|ag|dk|hd|ns|es|gm|hg|aa|ar|eb|dc|de|ey|md|nr|tr|at|lm|tw|od)"
    r"\s?-?\s?0?(\d{5,6})(?!\d)"
)
DUE_DAY = {"R01": 5, "R37": 5, "R35": 5, "R15": 25, "R21": 16, "R03": 16, "R34": 16, "R61": 16, "UG3": 16, "DP2": 5}
STOP = {
    "university",
    "of",
    "the",
    "at",
    "and",
    "school",
    "medicine",
    "college",
    "institute",
    "center",
    "centre",
    "medical",
    "health",
    "sciences",
    "science",
    "inc",
    "foundation",
    "hospital",
    "research",
    "system",
    "for",
    "in",
    "dept",
    "department",
    "campus",
    "llc",
    "corporation",
    "corp",
}  # "state" kept: Penn != Penn State


# ---------------------------------------------------------------- helpers
def norm(s) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", " ", s)


def toks(s) -> set[str]:
    return {t for t in norm(s).split() if t not in STOP and len(t) > 1}


def name_key(first, last) -> str:
    f, l_ = norm(first).split(), norm(last).split()
    return f"{l_[-1]}|{f[0][0]}" if f and l_ else ""


def core_of(project_num: str) -> str | None:
    m = CORE_RE.search((project_num or "").lower())
    return m.group(1) + m.group(2).zfill(6) if m else None


# ---------------------------------------------------------------- awards and cycles
def load_awards() -> pd.DataFrame:
    a = pd.read_parquet(INTERIM / "reporter_awards.parquet")
    pn = a["project_num"].fillna("")
    a["type"] = pn.str[0]
    a = a[(a["type"].isin(["1", "2"]))].copy()
    a["support_year"] = pn.str.extract(r"-(\d{2})")[0]
    a["a1"] = pn.str.contains(r"-\d{2}A\d$")
    a = a[(a["type"] == "2") | (a["support_year"] == "01")]  # first year of each competing segment
    a["core"] = a["project_num"].map(core_of)
    a["fy"] = a["fy"].astype(int)
    a["group"] = np.where(a["activity"].str.startswith("K"), "K", "R")
    a["day"] = a["activity"].map(DUE_DAY).fillna(12).astype(int)  # K awards: 12th
    a["aids"] = a["spending"].fillna("").str.contains("HIV/AIDS", case=False)
    a["rfa"] = a["foa"].fillna("").str.upper().str.startswith("RFA")
    a["som"] = a["org_type"].fillna("").str.contains("SCHOOLS OF MEDICINE")
    a["multi_pi"] = a["n_pis"].fillna(1) > 1
    n = pd.to_datetime(a["notice"])
    a["notice_month"] = n.dt.month
    a["ambiguous"] = n.dt.month.isin([1, 2, 3, 6, 9])
    return a


def cycles(a: pd.DataFrame) -> pd.DataFrame:
    """One row per (award, candidate receipt cycle). cyc in {2,6,10} = Feb/Jun/Oct council family."""
    n = pd.to_datetime(a["notice"])
    st = pd.to_datetime(a["start"])
    m, y, sm = n.dt.month, n.dt.year, st.dt.month
    rows = []

    def add(mask, cyc, yy, cand):
        s = a[mask].copy()
        s["cyc"], s["cyc_year"], s["cand"] = cyc, yy[mask].values, cand
        rows.append(s)

    add(m.isin([10, 11, 12]), 2, y, "primary")
    q1 = m.isin([1, 2, 3])
    add(q1 & sm.between(4, 9), 6, y - 1, "primary")
    add(q1 & ~sm.between(4, 9), 2, y - 1, "primary")
    add(m.isin([4, 5]), 6, y - 1, "primary")
    add((m == 6) & (sm >= 7), 10, y - 1, "primary")
    add((m == 6) & (sm < 7), 6, y - 1, "primary")
    add(m.isin([7, 8, 9]), 10, y - 1, "primary")
    add(m == 9, 2, y, "alt")
    c = pd.concat(rows)
    resub = (c["a1"] | (c["type"] == "2")) & (c["activity"] != "R15")  # R15: one schedule for all application types
    month = np.where(c["aids"], c["cyc"].map({2: 1, 6: 5, 10: 9}), c["cyc"] + resub.astype(int))
    day = np.where(c["aids"], 7, c["day"])
    c["due"] = pd.to_datetime(dict(year=c["cyc_year"].astype(int), month=month, day=day), errors="coerce")
    c["schedule"] = np.where(c["aids"], "AIDS-7", np.where(resub, "RESUB-", "NEW-") + c["day"].astype(str))
    return c.dropna(subset=["due"])


def schedule_index(schedule: str) -> pd.DatetimeIndex:
    if schedule.startswith("AIDS"):
        months, day = (1, 5, 9), 7
    else:
        kind, day = schedule.split("-")
        months = (2, 6, 10) if kind == "NEW" else (3, 7, 11)
        day = int(day)
    return pd.DatetimeIndex([pd.Timestamp(y, m_, day) for y in range(2013, 2028) for m_ in months])


# ---------------------------------------------------------------- linkage
def build_links() -> pd.DataFrame:
    a = load_awards()
    pis = []
    for r in a[["core", "all_pi_ids", "all_pi_names", "org"]].itertuples(index=False):
        ids = list(r.all_pi_ids) if r.all_pi_ids is not None else []
        names = list(r.all_pi_names) if r.all_pi_names is not None else []
        for pid, nm in zip(ids, names, strict=False):
            f, l_ = (nm.split("|") + [""])[:2]
            pis.append((pid, r.core, f, l_, r.org))
    pis = pd.DataFrame(pis, columns=["pi_id", "core", "first", "last", "org"]).dropna(subset=["pi_id"])
    pis["key"] = [name_key(f, l_) for f, l_ in zip(pis["first"], pis["last"], strict=False)]
    print("PI-award rows:", len(pis), flush=True)
    pis = pis[pis["key"] != ""]

    # anchored: OpenAlex authors of works acknowledging the PI's own award, same name key
    ack = pd.read_parquet(INTERIM / "oa_nih_acks.parquet", columns=["award_ids", "authors"])
    ack["cores"] = ack["award_ids"].map(lambda xs: {core_of(x) for x in (xs if xs is not None else [])} - {None})
    ack = ack[ack["cores"].str.len() > 0].explode("cores")
    ack = ack[ack["cores"].isin(set(pis["core"]))].explode("authors")
    ack = ack.dropna(subset=["authors"])
    ack["oa_id"] = [d.get("id") for d in ack["authors"]]
    ack["oa_name"] = [d.get("name") for d in ack["authors"]]
    ack = ack.drop(columns=["authors", "award_ids"])
    uniq = pd.Series(ack["oa_name"].dropna().unique())
    keys = {n: name_key(n, (norm(n).split() or [""])[-1]) for n in uniq}
    ack["key"] = ack["oa_name"].map(keys)
    ack = ack[ack["key"].isin(set(pis["key"]))]
    print("ack author rows after key filter:", len(ack), flush=True)
    anch = pis[["pi_id", "core", "key"]].merge(
        ack[["cores", "oa_id", "key"]].rename(columns={"cores": "core"}), on=["core", "key"]
    )
    anch = anch.groupby(["pi_id", "oa_id"]).size().rename("n_ack").reset_index()
    multi = anch.groupby("oa_id")["pi_id"].nunique()
    anch = anch[anch["oa_id"].isin(multi[multi == 1].index)].assign(source="anchored")

    # name + institution (rxiv last authors)
    names = pd.read_parquet(INTERIM / "oa_last_author_names.parquet")
    nn = names["display_name"].map(norm).str.split()
    names = names[nn.str.len() >= 2]
    nn = names["display_name"].map(norm).str.split()
    names["key"] = nn.str[-1] + "|" + nn.str[0].str[0]
    m = pd.read_parquet(INTERIM / "oa_match.parquet")[["last_id", "last_inst_name"]].dropna()
    inst = m.groupby("last_id")["last_inst_name"].agg(lambda s: set().union(*[toks(x) for x in s.unique()]))
    po = pis.groupby(["pi_id", "key"])["org"].agg(lambda s: set().union(*[toks(x) for x in s.dropna().unique()]))
    po = po.rename("orgs").reset_index()
    cand = po.merge(names[["id", "key"]], on="key")
    cand["inst"] = cand["id"].map(inst)

    def ok(r):
        if not isinstance(r["inst"], set) or not r["orgs"]:
            return False
        inter = r["orgs"] & r["inst"]
        return len(inter) >= 1 and len(inter) / min(len(r["orgs"]), len(r["inst"])) >= 0.5

    print("name candidates:", len(cand), flush=True)
    cand = cand[[ok(r) for r in cand.to_dict("records")]]
    multi = cand.groupby("id")["pi_id"].nunique()
    ni = cand[cand["id"].isin(multi[multi == 1].index)].rename(columns={"id": "oa_id"})[["pi_id", "oa_id"]]
    ni = ni.assign(source="name_inst")

    # audit: precision of name+inst links among PIs that also have anchored links
    surname_freq = pis.groupby("key")["pi_id"].nunique()
    both = ni[ni["pi_id"].isin(anch["pi_id"])].copy()
    anchored_pairs = set(zip(anch["pi_id"], anch["oa_id"], strict=False))
    both["confirmed"] = [(p, o) in anchored_pairs for p, o in zip(both["pi_id"], both["oa_id"], strict=False)]
    keymap = pis.drop_duplicates("pi_id").set_index("pi_id")["key"]
    both["surname_common"] = both["pi_id"].map(keymap).map(surname_freq).fillna(1) > 1
    both["ids_per_pi"] = both.groupby("pi_id")["oa_id"].transform("nunique")
    strata = both.groupby(["surname_common", both["ids_per_pi"] > 1])["confirmed"].agg(["mean", "size"])
    sample = both.groupby(["surname_common", both["ids_per_pi"] > 1], group_keys=False).apply(
        lambda g: g.sample(min(len(g), 50), random_state=1)
    )
    audit = {
        "name_inst_links": len(ni),
        "anchored_links": len(anch),
        "pis_name_inst": ni["pi_id"].nunique(),
        "pis_anchored": anch["pi_id"].nunique(),
        "overlap_links": len(both),
        "precision_proxy_all": both["confirmed"].mean(),
        "precision_proxy_sample200": sample["confirmed"].mean(),
    }
    print(audit)
    print(strata)
    pd.DataFrame([audit]).to_csv(R / "pi_link_audit.csv", index=False)
    strata.reset_index().to_csv(R / "pi_link_audit_strata.csv", index=False)
    sample.to_csv(R / "pi_link_audit_sample.csv", index=False)
    links = pd.concat([anch[["pi_id", "oa_id", "source"]], ni])
    links.to_parquet(INTERIM / "pi_links.parquet", index=False)
    return links


def build_links_norxiv() -> pd.DataFrame:
    """Anchored links using only works that are neither a bioRxiv/medRxiv preprint nor the journal version of one,
    so a PI's own preprints (the outcome) cannot make the PI linkable."""
    a = load_awards()
    pis = []
    for r in a[["core", "all_pi_ids", "all_pi_names"]].itertuples(index=False):
        ids = list(r.all_pi_ids) if r.all_pi_ids is not None else []
        names = list(r.all_pi_names) if r.all_pi_names is not None else []
        for pid, nm in zip(ids, names, strict=False):
            f, l_ = (nm.split("|") + [""])[:2]
            pis.append((pid, r.core, name_key(f, l_)))
    pis = pd.DataFrame(pis, columns=["pi_id", "core", "key"]).dropna(subset=["pi_id"])
    pis = pis[pis["key"] != ""]
    rx = set(pd.read_parquet(INTERIM / "oa_match.parquet", columns=["work_id"])["work_id"])
    rx |= set(pd.read_parquet(INTERIM / "oa_pubinfo.parquet", columns=["work_id"])["work_id"])
    ack = pd.read_parquet(INTERIM / "oa_nih_acks.parquet", columns=["work_id", "award_ids", "authors"])
    print("ack works:", len(ack), "rxiv-related dropped:", int(ack["work_id"].isin(rx).sum()), flush=True)
    ack = ack[~ack["work_id"].isin(rx)]
    ack["cores"] = ack["award_ids"].map(lambda xs: {core_of(x) for x in (xs if xs is not None else [])} - {None})
    ack = ack[ack["cores"].str.len() > 0].explode("cores")
    ack = ack[ack["cores"].isin(set(pis["core"]))].explode("authors").dropna(subset=["authors"])
    ack["oa_id"] = [d.get("id") for d in ack["authors"]]
    ack["oa_name"] = [d.get("name") for d in ack["authors"]]
    uniq = pd.Series(ack["oa_name"].dropna().unique())
    keys = {n: name_key(n, (norm(n).split() or [""])[-1]) for n in uniq}
    ack["key"] = ack["oa_name"].map(keys)
    ack = ack[ack["key"].isin(set(pis["key"]))]
    anch = pis.merge(ack[["cores", "oa_id", "key"]].rename(columns={"cores": "core"}), on=["core", "key"])
    anch = anch.groupby(["pi_id", "oa_id"]).size().rename("n_ack").reset_index()
    multi = anch.groupby("oa_id")["pi_id"].nunique()
    anch = anch[anch["oa_id"].isin(multi[multi == 1].index)].assign(source="anchored")
    anch[["pi_id", "oa_id", "source"]].to_parquet(INTERIM / "pi_links_norxiv.parquet", index=False)
    print("PIs anchored without rxiv works:", anch["pi_id"].nunique(), flush=True)
    return anch


def links(which: str = "auto") -> pd.DataFrame:
    if which == "norxiv":
        return pd.read_parquet(INTERIM / "pi_links_norxiv.parquet").drop_duplicates(["pi_id", "oa_id"])
    lk = pd.read_parquet(INTERIM / "pi_links.parquet")
    if which == "auto":
        aud = pd.read_csv(R / "pi_link_audit.csv").iloc[0]
        which = "anchored" if aud["precision_proxy_sample200"] < 0.95 else "both"
    if which == "anchored":
        lk = lk[lk["source"] == "anchored"]
    return lk.drop_duplicates(["pi_id", "oa_id"])


def posts_for(lk: pd.DataFrame) -> pd.DataFrame:
    d = pi_checks.preprints()
    p = d.merge(lk, left_on="last_id", right_on="oa_id")[["pi_id", "date", "doi"]].drop_duplicates(["pi_id", "doi"])
    return p.rename(columns={"pi_id": "pi"})


# ---------------------------------------------------------------- estimation
def stacked_panel(ev: pd.DataFrame, posts: pd.DataFrame, win_days: int = 28) -> pd.DataFrame:
    """Stack one cycle panel per schedule (each with its own due-date index)."""
    out = []
    for sch, e in ev.groupby("schedule"):
        p = posts[posts["pi"].isin(set(e["pi"]))]
        if p.empty:
            continue
        cnt = pi_checks.cycle_panel(e[["pi", "due"]], p, schedule_index(sch), "pi", win_days)
        out.append(cnt.assign(panel=sch, unit=cnt["pi"].astype(str) + "|" + sch))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def fepois_x(c: pd.DataFrame, k: str, extra: str | None = None, within: bool = False) -> dict:
    """Across investigators (default): Poisson, FE unit x cycle + calendar-cycle x window; the treated unit-cycle's
    pre-deadline split is compared with the calendar-cycle average across units.
    within=True: weighted linear share model at the unit-cycle level, y = window posts / cycle posts, FE unit +
    calendar cycle, weights = cycle posts; each unit's own other cycles are the reference. log_rr is reported as
    log(1 + coef / mean share) so both specs sit on a relative scale; coef_share keeps the share-point estimate."""
    import pyfixest as pf

    c = c.copy()
    if within:
        g = (
            c.groupby(["unit", "cyc_due"])
            .agg(panel=("panel", "first"), pi=("pi", "first"), x=(k, "max"), N=("n", "sum"))
            .reset_index()
        )
        n1 = c[c["w"] == 1].set_index(["unit", "cyc_due"])["n"]
        g["y"] = n1.reindex(pd.MultiIndex.from_frame(g[["unit", "cyc_due"]])).fillna(0).to_numpy() / g["N"]
        g["pc"] = g["panel"] + "_" + g["cyc_due"].astype(str)
        g = g[g["N"] > 0]
        m = pf.feols("y ~ x | unit + pc", data=g, weights="N", vcov={"CRV1": "pi"})
        b, se = float(m.coef()["x"]), float(m.se()["x"])
        ybar = float(np.average(g["y"], weights=g["N"]))
        return {
            "log_rr": float(np.log1p(b / ybar)),
            "se": se / ybar,
            "coef_share": b,
            "se_share": se,
            "mean_share": ybar,
            "pis": int(g["pi"].nunique()),
            "treated_pre_posts": int(c.loc[(c[k] == 1) & (c["w"] == 1), "n"].sum()),
        }
    c["uc"] = c["unit"].astype(str) + "_" + c["cyc_due"].astype(str)
    c["cw"] = c["panel"] + "_" + c["cyc_due"].astype(str) + "_" + c["w"].astype(str)
    c["x"] = c[k] * c["w"]
    f = "n ~ x" + (f" + x:{extra}" if extra else "") + " | uc + cw"
    m = pf.fepois(f, data=c, vcov={"CRV1": "pi"})
    co, se = m.coef(), m.se()
    out = {
        "log_rr": float(co["x"]),
        "se": float(se["x"]),
        "pis": int(c["pi"].nunique()),
        "treated_pre_posts": int(c.loc[(c[k] == 1) & (c["w"] == 1), "n"].sum()),
    }
    if extra:
        nm = [i for i in co.index if i.startswith("x:")][0]
        out.update({"interaction": float(co[nm]), "interaction_se": float(se[nm])})
    return out


def report(rows: list[dict], name: str) -> pd.DataFrame:
    r = pd.DataFrame(rows)
    if "log_rr" in r:
        r["rr"] = np.exp(r["log_rr"])
        r["z"] = r["log_rr"] / r["se"]
    if "interaction" in r:
        r["interaction_z"] = r["interaction"] / r["interaction_se"]
    r.to_csv(R / f"{name}.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(r.round(3).to_string(index=False))
    return r


def base(which="auto"):
    lk = links(which)
    a = load_awards()
    a = a[a["fy"] >= 2015]
    pi_awards = a.explode("all_pi_ids").rename(columns={"all_pi_ids": "pi"})
    pi_awards = pi_awards[pi_awards["pi"].isin(set(lk["pi_id"]))]
    c = cycles(pi_awards)
    posts = posts_for(lk)
    return lk, a, c, posts


def step_main():
    lk, a, c, posts = base()
    print("linked PIs:", lk["pi_id"].nunique(), "posts:", len(posts), flush=True)
    new = c[(c["type"] == "1")]
    specs = {
        "main: non-RFA, unambiguous notice month": new[~new["rfa"] & ~new["ambiguous"]],
        "non-RFA, all notice months": new[~new["rfa"]],
        "all incl. RFA": new,
        "A0 only (non-RFA, unambiguous)": new[~new["rfa"] & ~new["ambiguous"] & ~new["a1"]],
        "A1 only (non-RFA, unambiguous)": new[~new["rfa"] & ~new["ambiguous"] & new["a1"]],
        "AIDS-designated (non-RFA)": new[~new["rfa"] & new["aids"]],
    }
    rows = []
    for name, e in specs.items():
        cnt = stacked_panel(e, posts)
        for k in ("treated", "treated_prev", "treated_next", "treated_or_prev"):
            rows.append({"spec": name, "treatment": k, **fepois_x(cnt, k)})
        print(name, "done", flush=True)
    report(rows, "pi_design_main")


def event_study(e: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    """Raw counts around application events (pi, due): 14-day bins of event time, event FE + calendar-bin FE."""
    import pyfixest as pf

    e = e[["pi", "due"]].drop_duplicates().reset_index(drop=True).assign(ev=lambda x: np.arange(len(x)))
    bins = np.arange(-35, 35)
    grid = e.loc[e.index.repeat(len(bins))].assign(k=np.tile(bins, len(e)))
    grid["start"] = grid["due"] + pd.to_timedelta(grid["k"] * 14, unit="D")
    p = posts.merge(e, on="pi")
    p["k"] = ((p["date"] - p["due"]).dt.days // 14).astype(int)
    p = p[p["k"].between(bins.min(), bins.max())]
    cnt = p.groupby(["ev", "k"]).size().rename("n").reset_index()
    g = grid.merge(cnt, on=["ev", "k"], how="left").fillna({"n": 0})
    g["calbin"] = ((g["start"] - pd.Timestamp("2013-01-01")).dt.days // 14).astype(int)
    tot = g.groupby("ev")["n"].transform("sum")
    g = g[tot > 0]
    g["kk"] = g["k"].astype(str)
    refs = {"-20", "20"}  # two reference bins: event FE + calendar FE leave a linear trend in k unidentified
    for kk in g["kk"].unique():
        if kk not in refs:
            g[f"k_{kk.replace('-', 'm')}"] = (g["kk"] == kk).astype(int)
    regs = [col for col in g.columns if col.startswith("k_")]
    m = pf.fepois(f"n ~ {' + '.join(regs)} | ev + calbin", data=g, vcov={"CRV1": "pi"})
    co, se = m.coef(), m.se()
    es = pd.DataFrame(
        {"k": [int(r[2:].replace("m", "-")) for r in regs], "coef": co[regs].values, "se": se[regs].values}
    ).sort_values("k")
    return es


def step_events():
    """Raw counts: stacked events, 14-day bins of event time, event FE + calendar-bin FE."""
    import pyfixest as pf

    lk, a, c, posts = base()
    e = c[(c["type"] == "1") & ~c["rfa"] & ~c["ambiguous"]][["pi", "due"]].drop_duplicates()
    e = e.reset_index(drop=True).assign(ev=lambda x: np.arange(len(x)))
    es = event_study(e, posts)
    es.to_csv(R / "pi_event_study.csv", index=False)
    # cumulative: last 2 bins before due vs the 8 bins after
    pre = es[es["k"].isin([-2, -1])]["coef"].sum()
    post = es[es["k"].between(0, 7)]["coef"].sum()
    print(es.round(3).to_string(index=False))
    print(f"sum pre (k=-2,-1): {pre:.3f}; sum post (k=0..7): {post:.3f}")
    # 12-month totals: PI x year, application year vs other years
    yy = posts.assign(year=posts["date"].dt.year).groupby(["pi", "year"]).size().rename("n").reset_index()
    pis = yy["pi"].unique()
    full = pd.MultiIndex.from_product([pis, range(2015, 2026)], names=["pi", "year"]).to_frame(index=False)
    yy = full.merge(yy, how="left").fillna({"n": 0})
    app = e.assign(year=e["due"].dt.year)[["pi", "year"]].drop_duplicates().assign(app=1)
    yy = yy.merge(app, how="left").fillna({"app": 0})
    yy = yy[yy.groupby("pi")["n"].transform("sum") > 0]
    m2 = pf.fepois("n ~ app | pi + year", data=yy, vcov={"CRV1": "pi"})
    tot_row = {
        "log_rr_app_year": float(m2.coef()["app"]),
        "se": float(m2.se()["app"]),
        "pi_years": len(yy),
        "pre_sum": pre,
        "post_sum": post,
    }
    print(tot_row)
    pd.DataFrame([tot_row]).to_csv(R / "pi_event_totals.csv", index=False)


def bound_events(lk, a, c) -> dict:
    """Application events (pi, due, schedule) for the reference design and the three alternatives."""
    out = {
        "main": ("type-1 new (non-RFA, unambiguous)", c[(c["type"] == "1") & ~c["rfa"] & ~c["ambiguous"]]),
        "renewal": ("type-2 renewals (non-RFA)", c[(c["type"] == "2") & ~c["rfa"]]),
    }
    # renewal-clock ITT: every R01 of a linked PI, renewal due from project end
    r01 = a[a["activity"].isin(["R01", "R37"])].explode("all_pi_ids").rename(columns={"all_pi_ids": "pi"})
    r01 = r01[r01["pi"].isin(set(lk["pi_id"]))].dropna(subset=["end"])
    endd = pd.to_datetime(r01["end"]) + pd.Timedelta(days=1)
    r01 = r01.assign(due=pi_checks.due_from_start(endd, pi_checks.RESUB_DUE).values)
    r01["schedule"] = np.where(r01["aids"], "AIDS-7", "RESUB-5")
    out["itt"] = ("renewal-clock ITT (all R01 ends)", r01.dropna(subset=["due"]))
    # A0 of funded A1: the failed first submission, 2 or 3 new-schedule cycles before the A1 due date
    a1 = c[(c["type"] == "1") & c["a1"] & ~c["rfa"] & ~c["aids"]]
    a0 = []
    for sh in (2, 3):
        x = a1.copy()
        idx = schedule_index("NEW-5")
        pos = idx.searchsorted(x["due"].values) - sh
        x = x[(pos >= 0)]
        x["due"] = idx[pos[pos >= 0]]
        x["schedule"] = "NEW-" + x["day"].astype(str)
        a0.append(x)
    out["failed_a0"] = ("A0 behind funded A1 (failed submission)", pd.concat(a0))
    return out


def bound_panels(lk, a, c, posts) -> dict:
    """Stacked panels for the reference design and the three alternatives, keyed by short name."""
    return {k: (name, stacked_panel(ev, posts)) for k, (name, ev) in bound_events(lk, a, c).items()}


def step_moderators():
    lk, a, c, posts = base()
    e = c[(c["type"] == "1") & ~c["rfa"] & ~c["ambiguous"]].copy()
    # ESI: no earlier R01/R37/R21/K award for this PI, and career age <= 10 (OpenAlex first year >= 2 works)
    allaw = load_awards().explode("all_pi_ids").rename(columns={"all_pi_ids": "pi"})
    prior = allaw[allaw["activity"].isin(["R01", "R37", "R21"]) | allaw["activity"].str.startswith("K")]
    firstfy = prior.groupby("pi")["fy"].min()
    fy = pd.read_parquet(INTERIM / "oa_author_firstyear.parquet").set_index("id")["first_year_2"]
    start = lk.assign(fy2=lk["oa_id"].map(fy)).groupby("pi_id")["fy2"].min()
    e["esi"] = ((e["fy"] <= e["pi"].map(firstfy)) & (e["fy"] - e["pi"].map(start) <= 10)).astype(float)
    e["intensity2"] = (e.groupby(["pi", "fy"])["appl_id"].transform("nunique") >= 2).astype(float)
    rows = []
    for mod in ("som", "multi_pi", "esi", "intensity2"):
        cnt = stacked_panel(e, posts)
        mm = e.groupby("pi")[mod].max().astype(float)
        cnt[mod] = cnt["pi"].map(mm).fillna(0)
        rows.append({"moderator": mod, "share_pis": float(mm.mean()), **fepois_x(cnt, "treated", extra=mod)})
        print(mod, "done", flush=True)
    # prior-cycle diagnosis: drop control cycles that were the A0 submission of a later funded A1
    a1 = e[e["a1"]]
    idx = schedule_index("NEW-5")
    a0c = set()
    for sh in (2, 3):
        pos = idx.searchsorted(a1["due"].values) - sh
        ok = pos >= 0
        a0c |= set(zip(a1["pi"].values[ok], idx[pos[ok]], strict=False))
    cnt = stacked_panel(e, posts)
    drop = [(p, d) in a0c for p, d in zip(cnt["pi"], cnt["cyc_due"], strict=False)]
    cnt2 = cnt[~np.array(drop)]
    for k in ("treated", "treated_prev"):
        rows.append({"moderator": f"prior-cycle diagnosis: drop A0-of-A1 cycles ({k})", **fepois_x(cnt2, k)})
        rows.append({"moderator": f"prior-cycle diagnosis: all cycles ({k})", **fepois_x(cnt, k)})
    report(rows, "pi_design_moderators")


# ---------------------------------------------------------------- within-investigator estimand, clean placebo
def clean_placebo(cnt: pd.DataFrame, reps: int, within: bool, seed0: int = 5000) -> np.ndarray:
    """Remove true application cycles, then assign placebo cycles among each unit's remaining cycles."""
    nt = cnt[cnt["treated"] == 1].groupby("unit")["cyc_due"].nunique()
    base_ = cnt[cnt["treated"] == 0].drop(columns=["treated"])
    uc = base_[["unit", "cyc_due"]].drop_duplicates()
    r = uc.groupby("unit").size()
    m = np.minimum(nt.reindex(r.index).fillna(0), (r - 1).clip(lower=0))
    null = []
    for i in range(reps):
        u = uc.assign(rnd=np.random.default_rng(seed0 + i).random(len(uc)))
        u["rk"] = u.groupby("unit")["rnd"].rank(method="first")
        pick = u[u["rk"] <= u["unit"].map(m)][["unit", "cyc_due"]].assign(treated=1.0)
        cc = base_.merge(pick, on=["unit", "cyc_due"], how="left").fillna({"treated": 0.0})
        null.append(fepois_x(cc, "treated", within=within)["log_rr"])
        if (i + 1) % 50 == 0:
            print("placebo", within, i + 1, flush=True)
    return np.array(null)


def step_within(key: str, reps: int = 200):
    """For one design: cross-unit and within-unit estimates, adjacent cycles and the clean placebo."""
    lk, a, c, posts = base()
    name, cnt = bound_panels(lk, a, c, posts)[key]
    rows = []
    for within in (False, True):
        spec = "within investigator (unit x window FE)" if within else "across investigators (calendar-cycle FE)"
        for k in ("treated", "treated_prev", "treated_next"):
            if cnt[k].sum() == 0:
                continue
            rows.append({"design": key, "spec": spec, "treatment": k, **fepois_x(cnt, k, within=within)})
        null = clean_placebo(cnt, reps, within)
        pd.DataFrame({"null": null}).to_csv(
            R / f"pi_clean_null_{key}_{'within' if within else 'across'}.csv", index=False
        )
        obs = [r_ for r_ in rows if r_["spec"] == spec and r_["treatment"] == "treated"][0]["log_rr"]
        rows.append(
            {
                "design": key,
                "spec": spec,
                "treatment": "clean placebo",
                "log_rr": float(null.mean()),
                "se": float(null.std()),
                "null_p975": float(np.quantile(null, 0.975)),
                "p_one_sided": float((1 + (null >= obs).sum()) / (len(null) + 1)),
                "reps": reps,
            }
        )
    report(rows, f"pi_within_{key}")


def step_sim(reps: int = 200):
    """Known-truth check on the main panel: unit-cycle totals fixed, window count ~ Binomial(N, share);
    share = calendar-cycle share (non-application cycles) with unit heterogeneity in log-odds; beta in {0, 0.18}."""
    lk, a, c, posts = base()
    _, cnt = bound_panels(lk, a, c, posts)["main"]
    w1 = cnt[cnt["w"] == 1].set_index(["unit", "cyc_due"])
    tot = cnt.groupby(["unit", "cyc_due"])["n"].sum()
    g = (
        cnt.groupby(["unit", "cyc_due"])
        .agg(panel=("panel", "first"), treated=("treated", "max"), pi=("pi", "first"))
        .join(tot.rename("N"))
        .reset_index()
    )
    g["n1"] = w1["n"].reindex(pd.MultiIndex.from_frame(g[["unit", "cyc_due"]])).values
    ctl = g[g["treated"] == 0].groupby(["panel", "cyc_due"])[["n1", "N"]].sum()
    pi_c = (ctl["n1"] / ctl["N"]).clip(0.01, 0.99)
    g["lo"] = np.log(pi_c / (1 - pi_c)).reindex(pd.MultiIndex.from_frame(g[["panel", "cyc_due"]])).values
    g = g[g["lo"].notna()].reset_index(drop=True)
    units = g["unit"].unique()
    rows = []
    for beta in (0.0, 0.18):
        for i in range(reps):
            rng = np.random.default_rng(9000 + i)
            ue = pd.Series(rng.normal(0, 0.5, len(units)), index=units)
            # rate ratio exp(beta) on the window count, holding the rest of the cycle fixed
            lo = g["lo"].to_numpy() + ue.reindex(g["unit"]).to_numpy() + beta * g["treated"].to_numpy()
            n1 = rng.binomial(g["N"].astype(int).to_numpy(), 1 / (1 + np.exp(-lo)))
            sim = pd.concat([g.assign(w=1, n=n1), g.assign(w=0, n=g["N"].to_numpy() - n1)])[
                ["unit", "cyc_due", "panel", "pi", "w", "n", "treated"]
            ]
            for within in (False, True):
                f = fepois_x(sim, "treated", within=within)
                rows.append({"beta": beta, "rep": i, "within": within, "log_rr": f["log_rr"], "se": f["se"]})
            if (i + 1) % 25 == 0:
                print("sim", beta, i + 1, flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(R / "pi_sim_draws.csv", index=False)
    s_ = r.groupby(["beta", "within"]).apply(
        lambda x: pd.Series(
            {
                "mean_est": x["log_rr"].mean(),
                "sd_est": x["log_rr"].std(),
                "mean_se": x["se"].mean(),
                "coverage95": float((np.abs(x["log_rr"] - x.name[0]) <= 1.96 * x["se"]).mean()),
            }
        ),
        include_groups=False,
    )
    s_ = s_.reset_index()
    s_.to_csv(R / "pi_sim_summary.csv", index=False)
    print(s_.round(4).to_string(index=False))


def step_bymonth():
    """Mapping check: treated / prev / next by award notice month (unambiguous months, new awards)."""
    lk, a, c, posts = base()
    e = c[(c["type"] == "1") & ~c["rfa"] & ~c["ambiguous"]]
    rows = []
    for m in sorted(e["notice_month"].unique()):
        cnt = stacked_panel(e[e["notice_month"] == m], posts)
        for within in (False, True):
            for k in ("treated", "treated_prev", "treated_next"):
                rows.append(
                    {"notice_month": int(m), "within": within, "treatment": k, **fepois_x(cnt, k, within=within)}
                )
    report(rows, "pi_bymonth")


def step_totals():
    """Moves or adds? Total preprints per unit x cycle (zero cycles kept, Feb 2017 - Jun 2026), Poisson with unit FE
    and schedule x calendar-cycle FE; exposure = own due cycle, plus the cycles before and after. Units = PIs with
    at least one preprint; treated flags come from the events, so application cycles with no output are kept."""
    import pyfixest as pf

    lk, a, c, posts = base()
    active = set(posts["pi"])
    rows = []
    for key, (name, ev) in bound_events(lk, a, c).items():
        ev = ev[ev["pi"].isin(active)][["pi", "due", "schedule"]].dropna().drop_duplicates()
        out = []
        for sch, e in ev.groupby("schedule"):
            idx = schedule_index(sch)
            span = idx[(idx > pd.Timestamp("2017-01-31")) & (idx <= pd.Timestamp("2026-06-30"))]
            g = pd.DataFrame({"pi": e["pi"].unique()}).merge(pd.DataFrame({"cyc_due": span}), how="cross")
            g["panel"] = sch
            g["unit"] = g["pi"].astype(str) + "|" + sch
            p = posts[posts["pi"].isin(set(e["pi"]))].copy()
            j = idx.searchsorted(p["date"].values, side="left")
            ok = j < len(idx)
            p = p[ok].assign(cyc_due=idx[j[ok]])
            g = g.merge(
                p.groupby(["pi", "cyc_due"]).size().rename("n").reset_index(), on=["pi", "cyc_due"], how="left"
            ).fillna({"n": 0})
            for k, sh in (("treated", 0), ("treated_prev", -1), ("treated_next", 1)):
                pos = idx.searchsorted(e["due"].values) + sh
                good = (pos >= 0) & (pos < len(idx))
                flag = pd.DataFrame({"pi": e["pi"].values[good], "cyc_due": idx[pos[good]]}).drop_duplicates()
                g = g.merge(flag.assign(**{k: 1.0}), on=["pi", "cyc_due"], how="left").fillna({k: 0.0})
            out.append(g)
        g = pd.concat(out, ignore_index=True)
        g["pc"] = g["panel"] + "_" + g["cyc_due"].astype(str)
        ks = ["treated", "treated_prev", "treated_next"]
        m = pf.fepois("n ~ " + " + ".join(ks) + " | unit + pc", data=g, vcov={"CRV1": "pi"})
        co, V = m.coef(), m._vcov
        names = list(co.index)
        for k in ks:
            rows.append(
                {
                    "design": key,
                    "spec": name,
                    "term": k,
                    "log_rr": float(co[k]),
                    "se": float(m.se()[k]),
                    "unit_cycles": len(g),
                    "pis": int(g["pi"].nunique()),
                }
            )
        ii = [names.index(k) for k in ks]
        rows.append(
            {
                "design": key,
                "spec": name,
                "term": "sum prev+own+next",
                "log_rr": float(co[ks].sum()),
                "se": float(np.sqrt(V[np.ix_(ii, ii)].sum())),
                "unit_cycles": len(g),
                "pis": int(g["pi"].nunique()),
            }
        )
        print(key, "done", flush=True)
    report(rows, "pi_totals")


def _share(fml, d, ybar):
    import pyfixest as pf

    m = pf.feols(fml, data=d, weights="N", vcov={"CRV1": "pi"})
    return float(np.log1p(m.coef()["x"] / ybar)), float(m.se()["x"] / ybar)


def _pois(fml, d):
    import pyfixest as pf

    try:
        m = pf.fepois(fml, data=d, offset="lN", vcov={"CRV1": "pi"}, fixef_maxiter=100000, fixef_tol=1e-6)
        return float(m.coef()["x"]), float(m.se()["x"])
    except ValueError as e:
        print("failed:", fml, e, flush=True)
        return np.nan, np.nan


def step_decomp():
    """Across- vs within-investigator estimates on the same panel, several estimators, 28- and 14-day windows."""
    lk, a, c, posts = base()
    _, ev = bound_events(lk, a, c)["main"]
    rows = []
    for wd in (28, 14):
        cnt = stacked_panel(ev, posts, win_days=wd)
        g = (
            cnt.groupby(["unit", "cyc_due"])
            .agg(panel=("panel", "first"), pi=("pi", "first"), x=("treated", "max"), N=("n", "sum"))
            .reset_index()
        )
        g["n1"] = (
            cnt[cnt["w"] == 1]
            .set_index(["unit", "cyc_due"])["n"]
            .reindex(pd.MultiIndex.from_frame(g[["unit", "cyc_due"]]))
            .fillna(0)
            .to_numpy()
        )
        g["y"] = g["n1"] / g["N"]
        g["pc"] = g["panel"] + "_" + g["cyc_due"].astype(str)
        g["lN"] = np.log(g["N"])
        k = g.groupby("unit")["cyc_due"].transform("nunique")
        multi = set(g.loc[k >= 2, "unit"])
        ybar = float(np.average(g["y"], weights=g["N"]))

        r_all = fepois_x(cnt, "treated")
        r_multi = fepois_x(cnt[cnt["unit"].isin(multi)], "treated")
        gm = g[g["unit"].isin(multi)]
        specs = {
            "A across Poisson (uc + cw), all units": (r_all["log_rr"], r_all["se"]),
            "B across Poisson (uc + cw), units with >= 2 cycles": (r_multi["log_rr"], r_multi["se"]),
            "C share OLS, calendar-cycle FE only, all units": _share("y ~ x | pc", g, ybar),
            "D share OLS, calendar-cycle FE only, units with >= 2 cycles": _share("y ~ x | pc", gm, ybar),
            "E share OLS, unit + calendar-cycle FE (within)": _share("y ~ x | unit + pc", g, ybar),
            "F Poisson window count, offset log N, calendar-cycle FE only": _pois("n1 ~ x | pc", g),
            "G Poisson window count, offset log N, unit + calendar-cycle FE (within)": _pois("n1 ~ x | unit + pc", g),
        }
        for nm, (b, se) in specs.items():
            rows.append(
                {
                    "window_days": wd,
                    "spec": nm,
                    "log_rr": b,
                    "se": se,
                    "mean_share": ybar,
                    "units": g["unit"].nunique(),
                    "units_multi": len(multi),
                }
            )
            print(wd, nm, round(b, 4), round(se, 4), flush=True)
    report(rows, "pi_decomp")


def step_phase_profile(reps: int = 200):
    """Within-PI phase profile: share of a unit-cycle's posts in each 7-day bin before its due date, own application
    cycles vs the same units' other cycles; PI-cluster bootstrap for the difference."""
    lk, a, c, posts = base()
    _, ev = bound_events(lk, a, c)["main"]
    cnt = stacked_panel(ev, posts)
    own = set(zip(cnt.loc[cnt["treated"] == 1, "unit"], cnt.loc[cnt["treated"] == 1, "cyc_due"], strict=False))
    out = []
    for sch, e in ev.groupby("schedule"):
        idx = schedule_index(sch)
        p = posts[posts["pi"].isin(set(e["pi"]))].copy()
        j = idx.searchsorted(p["date"].values, side="left")
        ok = j < len(idx)
        p = p[ok].assign(cyc_due=idx[j[ok]])
        p["days"] = (p["cyc_due"] - p["date"]).dt.days
        p["unit"] = p["pi"].astype(str) + "|" + sch
        out.append(p)
    p = pd.concat(out)
    p = p[p["unit"].isin(set(cnt["unit"]))]
    p["bin"] = (p["days"] // 7).clip(upper=17)
    p["own"] = [(u, d) in own for u, d in zip(p["unit"], p["cyc_due"], strict=False)]
    # restrict to units that have both own and other cycles with posts
    has = p.groupby("unit")["own"].agg(["any", "all"])
    units = has.index[has["any"] & ~has["all"]]
    p = p[p["unit"].isin(units)]

    def prof(q):
        t = q.groupby(["own", "bin"]).size().unstack(fill_value=0)
        sh = t.div(t.sum(axis=1), axis=0)
        return sh.loc[True] - sh.loc[False], sh

    diff, sh = prof(p)
    pis = p["pi"].unique()
    boots = []
    by = {k: v for k, v in p.groupby("pi")}
    for r in range(reps):
        pick = np.random.default_rng(r).choice(pis, len(pis))
        q = pd.concat([by[x] for x in pick])
        boots.append(prof(q)[0])
    B = pd.concat(boots, axis=1)
    res = pd.DataFrame(
        {
            "bin": diff.index,
            "own_share": sh.loc[True].values,
            "other_share": sh.loc[False].values,
            "diff": diff.values,
            "se": B.std(axis=1).values,
        }
    )
    res["units"] = len(units)
    res["own_posts"] = int(p["own"].sum())
    res.to_csv(R / "pi_phase_profile.csv", index=False)
    print(res.round(4).to_string(index=False))

    # two-week bins with a placebo band: each unit's application cycle replaced by one of its other cycles
    p["bin"] = (p["days"] // 14).clip(upper=8)
    diff2, _ = prof(p)
    by = {k: v for k, v in p.groupby("pi")}
    boots = [prof(pd.concat([by[x] for x in np.random.default_rng(r).choice(pis, len(pis))]))[0] for r in range(reps)]
    q = p[~p["own"]]
    cyc = q[["unit", "cyc_due"]].drop_duplicates()
    cyc = cyc[cyc.groupby("unit")["cyc_due"].transform("size") >= 2]
    q = q[q["unit"].isin(set(cyc["unit"]))]
    null = []
    for r in range(reps):
        pick = cyc.sample(frac=1, random_state=r).drop_duplicates("unit")
        fake = set(zip(pick["unit"], pick["cyc_due"], strict=False))
        null.append(prof(q.assign(own=[(u, d) in fake for u, d in zip(q["unit"], q["cyc_due"], strict=False)]))[0])
    B, N = pd.concat(boots, axis=1), pd.concat(null, axis=1)
    res2 = pd.DataFrame(
        {
            "bin": diff2.index,
            "diff": diff2.values,
            "se": B.std(axis=1).values,
            "placebo_lo": N.quantile(0.025, axis=1).values,
            "placebo_hi": N.quantile(0.975, axis=1).values,
        }
    )
    res2["units"] = len(units)
    res2.to_csv(R / "pi_phase_profile_2w.csv", index=False)
    print(res2.round(4).to_string(index=False))


def step_bycycle():
    """Within-PI own-cycle effect separately for the February, June and October cycle families (own cycle vs
    the same PI's other cycles of the same family)."""
    lk, a, c, posts = base()
    _, ev = bound_events(lk, a, c)["main"]
    fam = {1: 2, 2: 2, 3: 2, 5: 6, 6: 6, 7: 6, 9: 10, 10: 10, 11: 10}
    cnt = stacked_panel(ev, posts)
    cnt["fam"] = cnt["cyc_due"].dt.month.map(fam)
    rows = []
    for f_, s in cnt.groupby("fam"):
        for within in (False, True):
            rows.append({"cycle_family": int(f_), "within": within, **fepois_x(s, "treated", within=within)})
    report(rows, "pi_bycycle")


def step_adjacent():
    """Within-investigator effect with the cycles just before and after each application dropped from the reference;
    and the new-vs-renewal difference."""
    lk, a, c, posts = base()
    rows = []
    for key in ("main", "renewal", "failed_a0", "itt"):
        name, cnt = bound_panels(lk, a, c, posts)[key]
        full = fepois_x(cnt, "treated", within=True)
        drop = cnt[(cnt["treated"] == 1) | ((cnt["treated_prev"] == 0) & (cnt["treated_next"] == 0))]
        f2 = fepois_x(drop, "treated", within=True)
        rows.append({"design": key, "reference": "all other cycles", **full})
        rows.append({"design": key, "reference": "excluding cycles adjacent to an application", **f2})
    report(rows, "pi_adjacent")


def step_activity(reps: int = 200):
    """Main design split by activity code. R21 and R03 do not require preliminary data; R01 does. Both estimands,
    clean placebo for each, and the across-investigator R21/R03 x treatment interaction on the pooled R panel."""
    lk, a, c, posts = base()
    e = c[(c["type"] == "1") & ~c["rfa"] & ~c["ambiguous"]].copy()
    e["act"] = np.select(
        [e["activity"].isin(["R01", "R37"]), e["activity"].isin(["R21", "R03"]), e["group"] == "K"],
        ["R01", "R21/R03", "K"],
        "other",
    )
    rows = []
    for act in ("R01", "R21/R03"):
        cnt = stacked_panel(e[e["act"] == act], posts)
        for within in (False, True):
            spec = "within investigator" if within else "across investigators"
            obs = fepois_x(cnt, "treated", within=within)
            null = clean_placebo(cnt, reps, within)
            rows.append(
                {
                    "activity": act,
                    "spec": spec,
                    **obs,
                    "null_mean": float(null.mean()),
                    "null_p975": float(np.quantile(null, 0.975)),
                    "p_one_sided": float((1 + (null >= obs["log_rr"]).sum()) / (len(null) + 1)),
                }
            )
        print(act, "done", flush=True)
    pooled = stacked_panel(e[e["act"].isin(["R01", "R21/R03"])], posts)
    pooled["r21"] = pooled["panel"].str.endswith("-16").astype(int)
    rows.append(
        {
            "activity": "R21/R03 minus R01 (pooled)",
            "spec": "across investigators",
            **fepois_x(pooled, "treated", extra="r21"),
        }
    )
    report(rows, "pi_activity")


def step_norxiv(reps: int = 100):
    """Main design re-estimated on links anchored without any preprint or journal version of a preprint."""
    build_links_norxiv()
    lk, a, c, posts = base("norxiv")
    print("linked PIs:", lk["pi_id"].nunique(), "posts:", len(posts), flush=True)
    rows = []
    for key in ("main", "failed_a0"):
        name, cnt = bound_panels(lk, a, c, posts)[key]
        for within in (False, True):
            spec = "within investigator" if within else "across investigators"
            obs = fepois_x(cnt, "treated", within=within)
            null = clean_placebo(cnt, reps, within)
            rows.append(
                {
                    "design": key,
                    "spec": spec,
                    **obs,
                    "null_mean": float(null.mean()),
                    "null_p975": float(np.quantile(null, 0.975)),
                    "p_one_sided": float((1 + (null >= obs["log_rr"]).sum()) / (len(null) + 1)),
                }
            )
        print(key, "done", flush=True)
    report(rows, "pi_norxiv")


def step_family(reps: int = 100):
    """Main design with investigator-by-cycle-family fixed effects: each investigator's application cycles are compared
    only with that investigator's other cycles of the same annual family (February, June or October schedules), so an
    investigator-specific season that coincides with the investigator's usual application season is absorbed."""
    lk, a, c, posts = base()
    name, cnt = bound_panels(lk, a, c, posts)["main"]
    fam = cnt["cyc_due"].dt.month.map({1: 1, 2: 1, 3: 1, 5: 2, 6: 2, 7: 2, 9: 3, 10: 3, 11: 3}).astype(str)
    rows = [{"spec": "investigator FE", **fepois_x(cnt, "treated", within=True)}]
    cf = cnt.assign(unit=cnt["unit"].astype(str) + "|f" + fam)
    obs = fepois_x(cf, "treated", within=True)
    null = clean_placebo(cf, reps, True)
    rows.append(
        {
            "spec": "investigator x cycle-family FE",
            **obs,
            "null_mean": float(null.mean()),
            "null_p975": float(np.quantile(null, 0.975)),
            "p_one_sided": float((1 + (null >= obs["log_rr"]).sum()) / (len(null) + 1)),
        }
    )
    report(rows, "pi_family")


def step_duedist():
    """Distribution of inferred due dates by schedule (funded applications, non-RFA): weights for the convolution."""
    lk, a, c, posts = base()
    e = c[~c["rfa"]]
    t = e.groupby(["schedule", "type"]).size().rename("n").reset_index()
    t.to_csv(R / "pi_duedist.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    {
        "link": build_links,
        "main": step_main,
        "events": step_events,
        "moderators": step_moderators,
        "within": lambda: step_within(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 200),
        "sim": lambda: step_sim(int(sys.argv[2]) if len(sys.argv) > 2 else 200),
        "bymonth": step_bymonth,
        "totals": step_totals,
        "decomp": step_decomp,
        "phase_profile": step_phase_profile,
        "bycycle": step_bycycle,
        "adjacent": step_adjacent,
        "duedist": step_duedist,
        "family": lambda: step_family(int(sys.argv[2]) if len(sys.argv) > 2 else 100),
        "norxiv": lambda: step_norxiv(int(sys.argv[2]) if len(sys.argv) > 2 else 100),
        "activity": lambda: step_activity(int(sys.argv[2]) if len(sys.argv) > 2 else 200),
    }[sys.argv[1]]()
