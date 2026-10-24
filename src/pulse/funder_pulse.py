"""Do other funders' calendars produce the same rise?

For funder F (country C, annual due date D): the share of country-C preprints in the 28 days before D, and the
cross-field slope of that rise on the field's dependence on F (F's share among country-C funded preprints in the
field, OpenAlex funders). The slope is robust to holidays that affect a whole country at once (e.g. Lunar New Year).
Also the pooled rise with Lunar New Year -7 .. +21 days removed, an event-time profile, and placebo due dates
shifted by +-3 and 6 months.

Due dates
  NSFC (CN): 20 March every year (NSFC notices for 2019, 2020, 2022, 2023, 2025, 2026; 2017, 2018 and 2021 assumed
             identical).
  KAKENHI (JP, Kiban B/C, Wakate): 2022-10-05, 2023-10-05, 2024-09-18, 2025-09-17 (JSPS calls).

    python -m pulse.funder_pulse [nonus]
"""

from __future__ import annotations

import sys
from datetime import date

import numpy as np
import pandas as pd

from pulse import country, field_split, ror, signals
from pulse.parse_jats import INTERIM

CNY = [
    date(2017, 1, 28),
    date(2018, 2, 16),
    date(2019, 2, 5),
    date(2020, 1, 25),
    date(2021, 2, 12),
    date(2022, 2, 1),
    date(2023, 1, 22),
    date(2024, 2, 10),
    date(2025, 1, 29),
    date(2026, 2, 17),
]

FUNDERS = {
    "NSFC": {
        "country": "CN",
        "years": range(2017, 2027),
        "due": lambda y: date(y, 3, 20),
        "funder_regex": r"National Natural Science Foundation of China",
        "drop_cny": True,
    },
    "KAKENHI": {
        "country": "JP",
        "years": [2022, 2023, 2024, 2025],
        "due": {2022: date(2022, 10, 5), 2023: date(2023, 10, 5), 2024: date(2024, 9, 18), 2025: date(2025, 9, 17)}.get,
        "funder_regex": r"Japan Society for the Promotion of Science|KAKENHI|"
        r"Ministry of Education, Culture, Sports, Science and Technology",
        "drop_cny": False,
    },
}
WINDOW = 28


def load(server: str) -> pd.DataFrame:
    v = pd.read_parquet(INTERIM / f"{server}_v1.parquet")
    v["doi"] = v["doi"].str.lower()
    m = pd.read_parquet(INTERIM / "oa_match.parquet")
    m = m[m.server == server][["doi", "n_funders", "funder_names"]]
    v = v.merge(m, on="doi", how="left")
    d = signals.prep(v)
    d = d[~d["covid"]].copy()
    cache = ror.load_cache()
    d["iso2"] = [
        country.iso2(e, i, f, cache.get(i) if isinstance(i, str) else None)
        for e, i, f in zip(
            d["corresp_email"], d["author_corresponding_institution"], d["first_aff_country"], strict=False
        )
    ]
    d = d[d["iso2"].notna()]
    d["category"] = d["category"].str.strip().str.lower()
    return d


def annotate(d: pd.DataFrame, cfg: dict, shift_days: int = 0) -> pd.DataFrame:
    """days until the next (possibly shifted) due date; 'fy' = the due-date year each day belongs to."""
    dues = pd.DatetimeIndex(
        sorted(pd.Timestamp(cfg["due"](y)) + pd.Timedelta(days=shift_days) for y in cfg["years"] if cfg["due"](y))
    )
    t = d["date"].values.astype("datetime64[ns]")
    idx = np.searchsorted(dues.values, t, side="left")
    ok = idx < len(dues)
    out = d[ok].copy()
    nxt = dues[idx[ok]]
    out["du"] = (nxt - pd.DatetimeIndex(out["date"])).days
    out["fy"] = nxt.year
    out = out[out["du"] <= 364]  # one cycle back from each due date
    out["win"] = (out["du"] < WINDOW).astype(float)
    if cfg["drop_cny"]:
        dd = pd.DatetimeIndex(out["date"])
        drop = np.zeros(len(out), bool)
        for c in CNY:
            c = pd.Timestamp(c)
            drop |= (dd >= c - pd.Timedelta(days=7)) & (dd <= c + pd.Timedelta(days=21))
        out = out[~drop]
    return out


def pulse(d: pd.DataFrame, c: str) -> tuple[float, float, int]:
    x = d.assign(y=(d["iso2"] == c).astype(float))
    b, se, n = signals.window_effect(x, "y", "win", fe="fy")
    return 100 * b, 100 * se, n


def dependence(d: pd.DataFrame, c: str, regex: str) -> pd.Series:
    x = d[(d["iso2"] == c) & (d["n_funders"].fillna(0) > 0)]
    return x.groupby("category")["funder_names"].apply(lambda s: s.fillna("").str.contains(regex).mean())


def run_funder(name: str, d: pd.DataFrame, server: str, min_country_n: int = 300) -> dict:
    cfg = FUNDERS[name]
    c = cfg["country"]
    a = annotate(d, cfg)
    out = {"funder": name, "server": server, "country": c, "n": len(a), "country_share": float((a["iso2"] == c).mean())}
    out["pulse_pp"], out["se_pp"], _ = pulse(a, c)
    # placebo due dates
    plac = {s: pulse(annotate(d, cfg, s), c)[0] for s in (-182, -91, 91, 182)}
    out.update({f"placebo_{s:+d}d_pp": v for s, v in plac.items()})
    # placebo countries at the same dates
    for pc in ("US", "DE", "GB", "KR", "JP" if c != "JP" else "CN"):
        out[f"placebo_country_{pc}_pp"] = pulse(a, pc)[0]
    # cross-field dose-response
    dep = dependence(d, c, cfg["funder_regex"])
    rows = []
    for cat, s in a.groupby("category"):
        if (s["iso2"] == c).sum() < min_country_n or cat not in dep.index:
            continue
        p, se, n = pulse(s, c)
        rows.append(
            {
                "funder": name,
                "server": server,
                "category": cat,
                "n": n,
                "n_country": int((s["iso2"] == c).sum()),
                "dependence": dep[cat],
                "pulse_pR_new_pp": p,
                "se_pp": se,
            }
        )
    t = pd.DataFrame(rows)
    if len(t) >= 5:
        t2 = t.rename(columns={"dependence": "nih_share_us_funded", "pulse_pR_new_pp": "pulse_R_new_pp"})
        a0, b, se_b, _ = field_split.slope(t2)
        out.update({"field_slope": b, "field_slope_se": se_b, "field_intercept": a0, "k_fields": len(t)})
    return out, t


def profile(d: pd.DataFrame, name: str, bin_days: int = 14) -> pd.DataFrame:
    cfg = FUNDERS[name]
    a = annotate(d, cfg)
    a["y"] = (a["iso2"] == cfg["country"]).astype(float)
    a["dev"] = a["y"] - a.groupby("fy")["y"].transform("mean")
    a["bin"] = (a["du"] // bin_days) * bin_days
    g = a.groupby("bin")["dev"].agg(["mean", "count"])
    g["se"] = a.groupby("bin")["y"].std() / np.sqrt(g["count"])
    return g.assign(funder=name).reset_index()


def non_us_check(d: pd.DataFrame, server: str) -> pd.DataFrame:
    """Shares sum to one, so a US dip lifts every other country. Re-estimate among
    non-US preprints only, for the funder country and placebo countries, at real and
    shifted due dates; report relative change (pp / base share)."""
    x = d[d["iso2"] != "US"]
    rows = []
    for name, cfg in FUNDERS.items():
        for c in (cfg["country"], "DE", "GB", "FR", "KR", "CA", "CN" if cfg["country"] != "CN" else "JP"):
            for shift in (0, -182, -91, 91, 182):
                a = annotate(x, cfg, shift)
                pp, se, n = pulse(a, c)
                base = 100 * (a["iso2"] == c).mean()
                rows.append(
                    {
                        "server": server,
                        "funder": name,
                        "country": c,
                        "shift_days": shift,
                        "pp": pp,
                        "se_pp": se,
                        "base_pp": base,
                        "rel": pp / base if base else np.nan,
                        "n": n,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    if sys.argv[1:2] == ["nonus"]:
        out = pd.concat([non_us_check(load(s), s) for s in ("biorxiv", "medrxiv")])
        out.to_csv(signals.RESULTS / "funder_pulse_nonUS.csv", index=False)
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print(out[out.shift_days == 0].round(3).to_string(index=False))
            print(out.pivot_table(index=["server", "funder", "country"], columns="shift_days", values="rel").round(3))
        return
    res, fields, profs = [], [], []
    for server in ("biorxiv", "medrxiv"):
        d = load(server)
        print(server, "country-coded non-COVID v1:", len(d), d["iso2"].value_counts().head(8).to_dict(), flush=True)
        for name in FUNDERS:
            r, t = run_funder(name, d, server)
            res.append(r)
            fields.append(t)
            profs.append(profile(d, name).assign(server=server))
    R = signals.RESULTS
    pd.DataFrame(res).to_csv(R / "funder_pulse.csv", index=False)
    pd.concat(fields).to_csv(R / "funder_pulse_fields.csv", index=False)
    pd.concat(profs).to_csv(R / "funder_pulse_profile.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(pd.DataFrame(res).round(3).T.to_string())
        print(pd.concat(fields).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
