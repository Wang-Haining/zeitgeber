"""Poisson daily-count tests for the HIV, intramural and linked-investigator comparisons.

  hiv        groups US-HIV, US-other, nonUS-HIV, nonUS-other (to the last AIDS due date)
  intramural groups US-extramural (reference), US-intramural, nonUS
  linked     groups US-other (reference), US with a linked NIH investigator as last author, nonUS

For each window (one at a time vs the rest of its cycle):
  A. per-group rate ratio: n ~ window | cycle + weekday, Poisson, SE clustered by cycle
  B. relative to the reference group on the same day: day FE + group x cycle FE + group x weekday FE;
     coefficients on group x window (and, for HIV, US x window, HIV x window, US x HIV x window).

    python -m pulse.counts
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pulse import cases, deadlines, signals

R = signals.RESULTS
WINDOWS = ("win_R_new", "win_AIDS", "win_F", "win_R_resub")


def day_panel(d: pd.DataFrame, group: pd.Series, levels: list[str], start, end) -> pd.DataFrame:
    days = deadlines.annotate(pd.DataFrame({"date": pd.date_range(start, end)}))
    days["cycle"] = (days["date"] + pd.to_timedelta(days["phi"], unit="D")).dt.strftime("%Y-%m")
    days["dow"] = days["date"].dt.dayofweek.astype(str)
    c = d.assign(g=group.values).groupby(["date", "g"]).size().rename("n").reset_index()
    full = days.merge(pd.DataFrame({"g": levels}), how="cross")
    p = full.merge(c, on=["date", "g"], how="left").fillna({"n": 0})
    p["day"] = p["date"].dt.strftime("%Y-%m-%d")
    p["gc"] = p["g"] + "_" + p["cycle"]
    p["gd"] = p["g"] + "_" + p["dow"]
    return p


def per_group(p: pd.DataFrame, label: str) -> list[dict]:
    import pyfixest as pf

    rows = []
    for g, s in p.groupby("g"):
        for w in WINDOWS:
            m = pf.fepois(f"n ~ {w} | cycle + dow", data=s, vcov={"CRV1": "cycle"})
            rows.append(
                {
                    "test": label,
                    "model": "A per-group rate ratio",
                    "group": g,
                    "term": w[4:],
                    "coef": float(m.coef()[w]),
                    "se": float(m.se()[w]),
                    "mean_daily": float(s["n"].mean()),
                    "n_total": int(s["n"].sum()),
                }
            )
    return rows


def relative(p: pd.DataFrame, label: str, terms: dict[str, pd.Series]) -> list[dict]:
    """terms: name -> 0/1 indicator over rows of p, each multiplied by the window."""
    import pyfixest as pf

    rows = []
    for w in WINDOWS:
        q = p.copy()
        names = []
        for t, ind in terms.items():
            nm = f"{t}_x_{w[4:]}".replace("-", "_")
            q[nm] = ind.to_numpy() * q[w].to_numpy()
            names.append(nm)
        m = pf.fepois(f"n ~ {' + '.join(names)} | day + gc + gd", data=q, vcov={"CRV1": "cycle"})
        for t, nm in zip(terms, names, strict=False):
            rows.append(
                {
                    "test": label,
                    "model": "B relative, day FE",
                    "group": t,
                    "term": w[4:],
                    "coef": float(m.coef()[nm]),
                    "se": float(m.se()[nm]),
                }
            )
    return rows


def run() -> None:
    from pulse import pi_design

    rows = []
    lk = pi_design.links()
    linked_ids = set(lk["oa_id"])
    m = pd.read_parquet(cases.INTERIM / "oa_match.parquet").drop_duplicates("doi")
    for server in ("biorxiv", "medrxiv"):
        d = cases.load(server)
        d["doi_l"] = d["doi"].str.lower()
        us = d["us"] == 1
        # HIV, to the last AIDS due date
        h = d[d["date"] <= cases.AIDS_END]
        gh = np.where(h["us"] == 1, "US", "nonUS") + np.where(h["hiv"], "-HIV", "-other")
        levels = ["US-HIV", "US-other", "nonUS-HIV", "nonUS-other"]
        for lab, mask in (("all days", np.ones(len(h), bool)), ("conferences removed", ~cases.conf_mask(h["date"]))):
            p = day_panel(h[mask], pd.Series(gh[mask]), levels, h["date"].min(), h["date"].max())
            if lab == "conferences removed":
                p = p[~cases.conf_mask(p["date"])]
            label = f"{server} HIV ({lab})"
            if lab == "all days":
                rows += per_group(p, label)
            is_us = p["g"].str.startswith("US").astype(int)
            is_hiv = p["g"].str.endswith("HIV").astype(int)
            rows += relative(p, label, {"US": is_us, "HIV": is_hiv, "US_x_HIV": is_us * is_hiv})
        # NIH intramural
        email = d["corresp_email"].fillna("").str.lower()
        inst = d["author_corresponding_institution"].fillna("")
        intra = us & (email.str.endswith("nih.gov") | inst.str.contains(cases.INTRAMURAL_INST))
        gi = np.where(~us, "nonUS", np.where(intra, "US-intramural", "US-extramural"))
        p = day_panel(d, pd.Series(gi), ["US-extramural", "US-intramural", "nonUS"], d["date"].min(), d["date"].max())
        rows += per_group(p, f"{server} intramural")
        rows += relative(
            p,
            f"{server} intramural",
            {"US-intramural": (p["g"] == "US-intramural").astype(int), "nonUS": (p["g"] == "nonUS").astype(int)},
        )
        # linked NIH investigators as last author
        last = d["doi_l"].map(m[m["server"] == server].set_index("doi")["last_id"])
        lnk = us & last.isin(linked_ids)
        gl = np.where(~us, "nonUS", np.where(lnk, "US-linked-PI", "US-other"))
        p = day_panel(d, pd.Series(gl), ["US-other", "US-linked-PI", "nonUS"], d["date"].min(), d["date"].max())
        rows += per_group(p, f"{server} linked")
        rows += relative(
            p,
            f"{server} linked",
            {"US-linked-PI": (p["g"] == "US-linked-PI").astype(int), "nonUS": (p["g"] == "nonUS").astype(int)},
        )
        print(server, "done", flush=True)
    out = pd.DataFrame(rows)
    out["z"] = out["coef"] / out["se"]
    out.to_csv(R / "counts_poisson.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 400):
        print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    run()
