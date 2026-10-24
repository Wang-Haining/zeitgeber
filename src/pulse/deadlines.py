"""NIH due-date calendar and per-date deadline phase.

All NIH standard schedules sit at fixed offsets inside one ~4-month cycle,
so a single coordinate -- days until the next R01-new anchor (Feb/Jun/Oct 5),
called `phi` -- places every schedule:

    schedule   due days            phi of due date
    R_new      5th/12th/16th       0 (R01), ~7 (K), ~11 (R21/R03)
    AIDS       Jan/May/Sep 7       ~29   (dedicated AIDS dates, 1988-2026)
    F          Apr/Aug/Dec 8       ~58
    R_resub    Mar/Jul/Nov 5-16    ~81-92

Source: grants.nih.gov standard due dates; AIDS dates removed by NOT-OD-26-029
(effective January 2027 Council). Deadline shocks:
  NOT-OD-20-091  due dates 2020-03-09..2020-05-01 accepted through 2020-05-01
  NOT-OD-26-012  due dates 2025-10-01..2025-12-05 accepted through 2025-12-08
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

# (month, day) of the first due date in each schedule's cluster, per cycle
SCHEDULES: dict[str, list[tuple[int, int]]] = {
    "R_new": [(2, 5), (6, 5), (10, 5)],
    "AIDS": [(1, 7), (5, 7), (9, 7)],
    "F": [(4, 8), (8, 8), (12, 8)],
    "R_resub": [(3, 5), (7, 5), (11, 5)],
}

# days before a due date counted as that schedule's pre-deadline window
WINDOW = 28

# nominal due date -> effective due date, for applications affected by a shock
SHOCKS: dict[str, dict[date, date]] = {
    "covid2020": {date(2020, 3, 16): date(2020, 5, 1), date(2020, 4, 8): date(2020, 5, 1)},
    "shutdown2025": {
        date(2025, 10, 5): date(2025, 12, 8),
        date(2025, 11, 5): date(2025, 12, 8),
    },
}


def due_dates(
    schedule: str, first_year: int = 2012, last_year: int = 2028, shocks: tuple[str, ...] = ()
) -> pd.DatetimeIndex:
    days = [date(y, m, d) for y in range(first_year, last_year + 1) for m, d in SCHEDULES[schedule]]
    for s in shocks:
        days = [SHOCKS[s].get(d, d) for d in days]
    return pd.DatetimeIndex(sorted(set(days)))


def days_until(dates, schedule: str, shocks: tuple[str, ...] = ()) -> np.ndarray:
    """Days from each date until the next due date (0 on the due date itself)."""
    t = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    dd = due_dates(schedule, t.year.min() - 1, t.year.max() + 1, shocks)
    idx = dd.searchsorted(t, side="left")
    return (dd[idx] - t).days.to_numpy()


def phi(dates, shocks: tuple[str, ...] = ()) -> np.ndarray:
    """Cycle coordinate: days until the next R01-new anchor (0..~122)."""
    return days_until(dates, "R_new", shocks)


def annotate(df: pd.DataFrame, date_col: str = "date", shocks: tuple[str, ...] = ()) -> pd.DataFrame:
    """Add phi, days-until per schedule, and 0/1 pre-deadline window flags."""
    out = df.copy()
    d = pd.to_datetime(out[date_col])
    out["phi"] = phi(d, shocks)
    for s in SCHEDULES:
        du = days_until(d, s, shocks)
        out[f"du_{s}"] = du
        out[f"win_{s}"] = (du <= WINDOW).astype(int)
    return out
