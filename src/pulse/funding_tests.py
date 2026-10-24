"""NIH-acknowledging vs other US preprints: helpers for the funding split.

Funding source:
  jats      medRxiv JATS funding statement
  openalex  OpenAlex funders and awards, both servers (NIH-funded vs other-funded)
"""

from __future__ import annotations

import pandas as pd

from pulse import funding


def oa_nih(m: pd.DataFrame) -> pd.Series:
    awards = m["awards"] if "awards" in m else pd.Series("", index=m.index)
    text = m["funder_names"].fillna("") + " | " + awards.fillna("")
    return pd.Series([funding.classify(t)["nih"] for t in text], index=m.index)
