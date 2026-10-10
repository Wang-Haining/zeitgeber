"""Shared panels for the HIV and NIH-intramural comparisons.

HIV preprints are identified by category and keywords; HIV conference windows (CROI, IAS/AIDS) can be masked.
NIH intramural authors (corresponding e-mail @nih.gov or an NIH institute as institution) do not apply on extramural
due dates and serve as a negative case.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from pulse import signals
from pulse.parse_jats import INTERIM

AIDS_END = pd.Timestamp("2026-05-07")
CONFERENCES = [  # (start, end) — CROI and IAS/AIDS
    ("2017-02-13", "2017-02-16"),
    ("2018-03-04", "2018-03-07"),
    ("2019-03-04", "2019-03-07"),
    ("2020-03-08", "2020-03-11"),
    ("2021-03-07", "2021-03-10"),
    ("2022-02-12", "2022-02-16"),
    ("2023-02-19", "2023-02-22"),
    ("2024-03-03", "2024-03-06"),
    ("2025-03-09", "2025-03-12"),
    ("2026-02-22", "2026-02-25"),
    ("2017-07-23", "2017-07-26"),
    ("2018-07-23", "2018-07-27"),
    ("2019-07-21", "2019-07-24"),
    ("2020-07-06", "2020-07-10"),
    ("2021-07-18", "2021-07-21"),
    ("2022-07-29", "2022-08-03"),
    ("2023-07-23", "2023-07-26"),
    ("2024-07-20", "2024-07-26"),
    ("2025-07-13", "2025-07-17"),
]
INTRAMURAL_INST = re.compile(
    r"National Institutes? of Health|\bNIH\b|National Cancer Institute|National Human Genome Research Institute|"
    r"National Institute (?:of|on) [A-Z]|National Heart, Lung|National Eye Institute|National Library of Medicine|"
    r"\bN(?:IAID|HLBI|IMH|IDDK|HGRI|INDS|ICHD|IEHS|IDA|IA|IAAA|IAMS|IBIB|IDCD|IDCR|EI|IMHD|INR|CATS|CI)\b"
)


def load(server: str) -> pd.DataFrame:
    d = signals.prep(pd.read_parquet(INTERIM / f"{server}_v1.parquet"))
    d = d[~d["covid"] & d["us"].notna()].copy()
    d["cycle"] = (d["date"] + pd.to_timedelta(d["phi"], unit="D")).dt.strftime("%Y-%m")
    d["cyc_month"] = (d["date"] + pd.to_timedelta(d["phi"], unit="D")).dt.month
    d["category"] = d["category"].str.strip().str.lower()
    d["hiv_f"] = d["hiv"].astype(float)
    return d


def conf_mask(dates: pd.Series) -> np.ndarray:
    m = np.zeros(len(dates), bool)
    for a, b in CONFERENCES:
        m |= (
            (dates >= pd.Timestamp(a) - pd.Timedelta(days=21)) & (dates <= pd.Timestamp(b) + pd.Timedelta(days=7))
        ).to_numpy()
    return m
