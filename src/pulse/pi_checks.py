"""Shared pieces of the investigator-level design.

Award start date -> receipt date, project-number cores, the preprint table of linked investigators, and the
unit x 4-month cycle x {28 days before the cycle's due date, rest of cycle} panel. On that panel a Poisson model
with unit x cycle and cycle x window fixed effects estimates how much a unit concentrates output in the weeks before
its own due date, relative to its other cycles and to other units in the same cycle.
"""

from __future__ import annotations

import re

import pandas as pd

from pulse import signals
from pulse.parse_jats import INTERIM

R = signals.RESULTS
CORE = re.compile(
    r"(?<![A-Za-z])(AI|HL|MH|CA|DA|AG|DK|HD|NS|ES|GM|HG|AA|AR|EB|DC|DE|EY|MD|NR|TR|AT|LM|TW|OD)"
    r"\s?-?\s?0?(\d{5,6})(?!\d)",
    re.I,
)
NEW_DUE = {
    9: (2, 0),
    10: (2, 0),
    11: (2, 0),
    12: (2, 0),
    1: (2, -1),
    2: (2, -1),
    3: (2, -1),
    4: (6, -1),
    5: (6, -1),
    6: (6, -1),
    7: (10, -1),
    8: (10, -1),
}
RESUB_DUE = {m: (mm + 1, dy) for m, (mm, dy) in NEW_DUE.items()}  # Mar/Jul/Nov 5


def due_from_start(start: pd.Series, table: dict) -> pd.Series:
    s = pd.to_datetime(start)
    mm = s.dt.month.map(lambda m: table[m][0])
    yy = s.dt.year + s.dt.month.map(lambda m: table[m][1])
    return pd.to_datetime(dict(year=yy, month=mm, day=5), errors="coerce")


def cores(text: str) -> set[str]:
    return {m.group(1).lower() + m.group(2).zfill(6) for m in CORE.finditer(text or "")}


def preprints() -> pd.DataFrame:
    m = pd.read_parquet(INTERIM / "oa_match.parquet").drop_duplicates("doi")
    rows = []
    for s in ("biorxiv", "medrxiv"):
        v = pd.read_parquet(
            INTERIM / f"{s}_v1.parquet", columns=["doi", "date", "country", "title", "abstract", "funding"]
        )
        v["doi"] = v["doi"].str.lower()
        v = v.merge(m[m.server == s][["doi", "last_id", "awards", "funder_names"]], on="doi", how="left")
        rows.append(v.assign(server=s))
    d = pd.concat(rows, ignore_index=True)
    text = d["title"].fillna("") + " " + d["abstract"].fillna("")
    d = d[(d["country"] == "US") & ~text.str.contains(r"covid|sars-?cov-?2|coronavirus", case=False)]
    d["date"] = pd.to_datetime(d["date"])
    d["cores"] = [cores(f"{a} {b}") for a, b in zip(d["awards"].fillna(""), d["funding"].fillna(""), strict=False)]
    return d


def cycle_panel(
    events: pd.DataFrame, posts: pd.DataFrame, dues: pd.DatetimeIndex, unit: str, win_days: int = 28
) -> pd.DataFrame:
    """events: unit, due (treated cycles). posts: unit, date. dues: the schedule's due dates."""
    p = posts.copy()
    idx = dues.searchsorted(p["date"].values, side="left")
    ok = idx < len(dues)
    p = p[ok]
    p["cyc_due"] = dues[idx[ok]]
    p["w"] = ((p["cyc_due"] - p["date"]).dt.days < win_days).astype(int)
    cnt = p.groupby([unit, "cyc_due", "w"]).size().rename("n").reset_index()
    units = cnt[unit].unique()
    span = pd.DatetimeIndex(sorted(cnt["cyc_due"].unique()))
    full = pd.MultiIndex.from_product([units, span, [0, 1]], names=[unit, "cyc_due", "w"]).to_frame(index=False)
    cnt = full.merge(cnt, how="left").fillna({"n": 0})
    tot = cnt.groupby([unit, "cyc_due"])["n"].transform("sum")
    cnt = cnt[tot > 0]  # unit-cycles with any output (others carry no information)
    ev = events[[unit, "due"]].dropna().drop_duplicates()
    ev = ev.assign(t=1)
    for k, sh in (("treated", 0), ("treated_prev", -1), ("treated_next", 1)):
        e = ev.copy()
        pos = dues.searchsorted(e["due"].values)
        pos = pos + sh
        good = (pos >= 0) & (pos < len(dues))
        e = e[good]
        e["cyc_due"] = dues[pos[good]]
        cnt = cnt.merge(e[[unit, "cyc_due"]].drop_duplicates().assign(**{k: 1}), on=[unit, "cyc_due"], how="left")
        cnt[k] = cnt[k].fillna(0)
    cnt["treated_or_prev"] = ((cnt["treated"] + cnt["treated_prev"]) > 0).astype(float)
    return cnt
