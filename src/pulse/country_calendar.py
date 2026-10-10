"""Same calendar, different funder: does each country's share follow the NIH 4-month rhythm?

Canada shares the North American academic year (September start, semesters) but not
NIH funding; the UK, Germany, Australia have different academic calendars. If the US
R_new/F rhythm is the academic calendar, Canada should show it too; if it is NIH, Canada
should not (beyond its own NIH-funded share). Shares are computed among all
country-coded non-COVID preprints, cycle FE, SE clustered by week.

    python -m pulse.country_calendar
"""

from __future__ import annotations

import pandas as pd

from pulse import funder_pulse, signals

COUNTRIES = ["US", "CA", "GB", "DE", "FR", "AU", "CN", "JP"]


def run() -> pd.DataFrame:
    rows = []
    for server in ("biorxiv", "medrxiv"):
        d = funder_pulse.load(server)
        d["cycle"] = (d["date"] + pd.to_timedelta(d["phi"], unit="D")).dt.strftime("%Y-%m")
        for c in COUNTRIES:
            x = d.assign(y=(d["iso2"] == c).astype(float))
            base = x["y"].mean()
            for w in signals.WINDOWS:
                b, se, n = signals.window_effect(x, "y", w, fe="cycle")
                rows.append(
                    {
                        "server": server,
                        "country": c,
                        "window": w[4:],
                        "pp": 100 * b,
                        "se_pp": 100 * se,
                        "base_pp": 100 * base,
                        "rel": b / base,
                        "z": b / se,
                    }
                )
            # monthly profile (share minus country mean), for the Canada-September question
            m = x.groupby(x["date"].dt.month)["y"].mean() - base
            rows += [
                {
                    "server": server,
                    "country": c,
                    "window": f"month_{k:02d}",
                    "pp": 100 * v,
                    "base_pp": 100 * base,
                    "rel": v / base,
                }
                for k, v in m.items()
            ]
    out = pd.DataFrame(rows)
    out.to_csv(signals.RESULTS / "country_calendar.csv", index=False)
    return out


if __name__ == "__main__":
    t = run()
    with pd.option_context("display.width", 220, "display.max_rows", 300):
        w = t[t.window.isin(["R_new", "F", "AIDS", "R_resub"])]
        print(w.pivot_table(index=["server", "country"], columns="window", values="rel").round(3))
        print(w.pivot_table(index=["server", "country"], columns="window", values="z").round(2))
        m = t[t.window.str.startswith("month")]
        print(m.pivot_table(index=["server", "country"], columns="window", values="rel").round(3))
