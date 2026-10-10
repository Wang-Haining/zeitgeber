"""Before and after NOT-OD-17-050 (NIH applications may cite preprints).

bioRxiv only (medRxiv starts 2019). Pre-2017 metadata go to a separate raw directory. Country is assigned by one rule
in both periods (no JATS e-mail before 2017): ROR match of the corresponding-institution string, then the CN/JP/HK
text patterns. Window effects: US share, cycle FE, SE clustered by cycle; also US x window on Poisson daily counts
(day FE).

    python -m pulse.pre2017 fetch              # 2013-11 .. 2016-12 metadata -> data/raw_pre2017/biorxiv
    python -m pulse.pre2017 ror <i> <n>        # institution countries, shard i of n
    python -m pulse.pre2017 <analyze|byfield>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pulse import country, deadlines, fetch_rxiv, ror, signals
from pulse.counts import day_panel

R = signals.RESULTS
RAW_PRE = Path(__file__).resolve().parents[2] / "data" / "raw_pre2017"
POLICY = pd.Timestamp("2017-03-24")


def fetch() -> None:
    fetch_rxiv.RAW = RAW_PRE
    fetch_rxiv.main("biorxiv", "2013-11", "2016-12")


def first_versions() -> pd.DataFrame:
    files = sorted((RAW_PRE / "biorxiv").glob("*.parquet")) + sorted(
        f for f in (fetch_rxiv.RAW / "biorxiv").glob("*.parquet") if f.stem <= "2019-12"
    )
    df = pd.concat(pd.read_parquet(f) for f in files)
    df["version"] = pd.to_numeric(df["version"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    v1 = df.sort_values(["doi", "version", "date"]).drop_duplicates("doi")
    v1 = v1[(v1["version"] == 1) & (v1["date"] <= pd.Timestamp("2019-12-31"))]
    return v1


def assign_country(v1: pd.DataFrame) -> pd.Series:
    cache = ror.load_cache()
    inst = v1["author_corresponding_institution"].fillna("").str.strip()
    missing = sorted(set(inst) - set(cache) - {""})
    print("institution strings not in ROR cache:", len(missing), flush=True)
    return pd.Series([country.iso2(institution=s, ror_code=cache.get(s) or None) for s in inst], index=v1.index)


def analyze() -> None:
    import pyfixest as pf

    v1 = first_versions()
    v1["country"] = assign_country(v1)
    d = deadlines.annotate(v1[["doi", "date", "country", "category"]].copy())
    d = d[d["country"].notna()].copy()
    d["us"] = (d["country"] == "US").astype(float)
    d["cycle"] = (d["date"] + pd.to_timedelta(d["phi"], unit="D")).dt.strftime("%Y-%m")
    d["post"] = d["date"] > POLICY
    rows = []
    periods = {"pre (2013-11 to 2017-03-24)": d[~d["post"]], "post (2017-03-25 to 2019-12)": d[d["post"]]}
    for name, s in periods.items():
        codes, uniq = pd.factorize(s["cycle"])
        for w in ("win_R_new", "win_F", "win_AIDS", "win_R_resub"):
            beta, resid, Xd, XtX_inv = signals._fit(s["us"].to_numpy(float), s[[w]].to_numpy(float), codes, len(uniq))
            S = np.bincount(codes, weights=Xd[:, 0] * resid)
            se = float(np.sqrt(XtX_inv[0, 0] ** 2 * (S @ S) * len(uniq) / max(len(uniq) - 1, 1)))
            rows.append(
                {
                    "period": name,
                    "measure": "US share (pp), cycle FE, cycle-clustered",
                    "window": w[4:],
                    "coef": 100 * beta[0],
                    "se": 100 * se,
                    "n": len(s),
                    "n_cycles": len(uniq),
                    "us_share": s["us"].mean(),
                }
            )
        g = np.where(s["us"] == 1, "US", "nonUS")
        p = day_panel(s, pd.Series(g), ["nonUS", "US"], s["date"].min(), s["date"].max())
        for w in ("win_R_new", "win_F"):
            q = p.copy()
            q["x"] = (q["g"] == "US").astype(int) * q[w]
            m = pf.fepois("n ~ x | day + gc + gd", data=q, vcov={"CRV1": "cycle"})
            rows.append(
                {
                    "period": name,
                    "measure": "log(US/nonUS) counts, Poisson day FE",
                    "window": w[4:],
                    "coef": float(m.coef()["x"]),
                    "se": float(m.se()["x"]),
                    "n": len(s),
                }
            )
    # interaction in one model: US share window effect x post, cycle FE
    s = d[d["date"] <= pd.Timestamp("2019-12-31")].copy()
    s["post_f"] = s["post"].astype(float)
    for w in ("win_R_new", "win_F"):
        s["wp"] = s[w] * s["post_f"]
        m = pf.feols(f"us ~ {w} + wp | cycle", data=s, vcov={"CRV1": "cycle"})
        rows.append(
            {
                "period": "difference post - pre",
                "measure": "US share (pp), cycle FE",
                "window": w[4:],
                "coef": 100 * float(m.coef()["wp"]),
                "se": 100 * float(m.se()["wp"]),
                "n": len(s),
            }
        )
    out = pd.DataFrame(rows)
    out["z"] = out["coef"] / out["se"]
    out.to_csv(R / "pre2017.csv", index=False)
    yr = d.groupby(d["date"].dt.year).agg(n=("doi", "size"), us_share=("us", "mean"))
    yr.to_csv(R / "pre2017_years.csv")
    with pd.option_context("display.width", 200):
        print(yr.to_string())
        print(out.round(3).to_string(index=False))


def ror_shard(i: int, n: int) -> None:
    """Look up institution strings not yet in the ROR cache, shard i of n -> ror_cache_pre2017_{i}.json."""
    import json
    import time

    import httpx

    v1 = first_versions()
    cache = ror.load_cache()
    inst = v1["author_corresponding_institution"].fillna("").str.strip()
    missing = sorted(set(inst) - set(cache) - {""})[i::n]
    print("shard", i, "strings", len(missing), flush=True)
    new = {}
    with httpx.Client(headers={"User-Agent": "pulse-research/0.1"}) as client:
        for k, s in enumerate(missing):
            new[s] = ror.lookup(client, s)
            time.sleep(ror.PAUSE)
            if (k + 1) % 200 == 0:
                (ror.INTERIM / f"ror_cache_pre2017_{i}.json").write_text(json.dumps(new))
                print("ror", k + 1, flush=True)
    (ror.INTERIM / f"ror_cache_pre2017_{i}.json").write_text(json.dumps(new))


def byfield() -> None:
    """Pre/post contrast restricted to high- and to low-NIH-dependence categories (dependence from 2017-2026)."""
    import pyfixest as pf

    from pulse import field_split

    v1 = first_versions()
    v1["country"] = assign_country(v1)
    d = deadlines.annotate(v1[["doi", "date", "country", "category"]].copy())
    d = d[d["country"].notna()].copy()
    d["us"] = (d["country"] == "US").astype(float)
    d["cycle"] = (d["date"] + pd.to_timedelta(d["phi"], unit="D")).dt.strftime("%Y-%m")
    d["post_f"] = (d["date"] > POLICY).astype(float)
    d["category"] = d["category"].fillna("").str.strip().str.lower()
    dep = field_split.per_category(field_split.load()).set_index("category")["nih_share_us_funded"]
    dep.index = dep.index.str.strip().str.lower()
    d["dep"] = d["category"].map(dep)
    rows = []
    for name, s in {
        "high NIH dependence (>= 0.7)": d[d["dep"] >= 0.7],
        "low NIH dependence (< 0.7)": d[d["dep"] < 0.7],
    }.items():
        for per, q in (("pre", s[s["post_f"] == 0]), ("post", s[s["post_f"] == 1])):
            for w in ("win_R_new", "win_F"):
                m = pf.feols(f"us ~ {w} | cycle", data=q, vcov={"CRV1": "cycle"})
                rows.append(
                    {
                        "fields": name,
                        "period": per,
                        "window": w[4:],
                        "coef": 100 * float(m.coef()[w]),
                        "se": 100 * float(m.se()[w]),
                        "n": len(q),
                        "us_share": q["us"].mean(),
                    }
                )
        for w in ("win_R_new", "win_F"):
            q = s.copy()
            q["wp"] = q[w] * q["post_f"]
            m = pf.feols(f"us ~ {w} + wp | cycle", data=q, vcov={"CRV1": "cycle"})
            rows.append(
                {
                    "fields": name,
                    "period": "post - pre",
                    "window": w[4:],
                    "coef": 100 * float(m.coef()["wp"]),
                    "se": 100 * float(m.se()["wp"]),
                    "n": len(q),
                }
            )
    out = pd.DataFrame(rows)
    out["z"] = out["coef"] / out["se"]
    out.to_csv(R / "pre2017_byfield.csv", index=False)
    print(d.groupby([d["post_f"], d["dep"] >= 0.7]).size())
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "ror":
        ror_shard(int(sys.argv[2]), int(sys.argv[3]))
    else:
        {"fetch": fetch, "analyze": analyze, "byfield": byfield}[sys.argv[1]]()
