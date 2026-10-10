"""Is the similarity of pulse preprints to the upcoming award just time proximity?
Compare pulse (0-13 d before due) with the same PI's preprints at other gaps, controlling
log gap, and within short gaps only.

    python -m pulse.sim_check
"""

import numpy as np
import pandas as pd
import pyfixest as pf

from pulse import signals
from pulse.parse_jats import INTERIM


def main() -> None:
    t = pd.read_parquet(INTERIM / "pulse_content_posts.parquet")
    t = t[t["sim_next_award"].notna()].copy()
    t["pulse_f"] = t["pulse"].astype(float)
    t["ym"] = pd.to_datetime(t["date"]).dt.strftime("%Y-%m")
    t["lgap"] = np.log1p(t["gap"])
    rows = []
    for lab, s, f_extra in (
        ("all, control log gap", t, " + lgap"),
        ("gap <= 90 days", t[t["gap"] <= 90], ""),
        ("gap <= 45 days", t[t["gap"] <= 45], ""),
    ):
        for y in ("sim_next_award", "sim_diff", "hype_pk", "rs_head_abs_log"):
            ss = s[s[y].notna()]
            if ss["pulse_f"].sum() < 20:
                continue
            m = pf.feols(f"{y} ~ pulse_f{f_extra} | pi + ym", data=ss, vcov={"CRV1": "pi"})
            b, se = float(m.coef()["pulse_f"]), float(m.se()["pulse_f"])
            rows.append(
                {
                    "spec": lab,
                    "outcome": y,
                    "coef": b,
                    "se": se,
                    "z": b / se,
                    "n": len(ss),
                    "n_pulse": int(ss["pulse_f"].sum()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(signals.RESULTS / "pulse_sim_check.csv", index=False)
    prof = (
        t.assign(gb=pd.cut(t["gap"], [-1, 13, 27, 45, 90, 180, 365]))
        .groupby("gb", observed=True)["sim_next_award"]
        .agg(["mean", "count"])
    )
    prof.to_csv(signals.RESULTS / "pulse_sim_by_gap.csv")
    print(out.round(4).to_string(index=False))
    print(prof.round(4))
    # within-PI gap profile: similarity by gap bin, PI + year-month FE, reference = 180-365 days; SE by PI
    t["gb"] = pd.cut(
        t["gap"], [-1, 13, 27, 45, 90, 180, 365], labels=["0-13", "14-27", "28-45", "46-90", "91-180", "181-365"]
    ).astype(str)
    for g in ["0-13", "14-27", "28-45", "46-90", "91-180"]:
        t[f"g_{g.replace('-', '_')}"] = (t["gb"] == g).astype(float)
    cols = [c for c in t.columns if c.startswith("g_")]
    m = pf.feols(f"sim_next_award ~ {' + '.join(cols)} | pi + ym", data=t[t["gb"] != "nan"], vcov={"CRV1": "pi"})
    fe = pd.DataFrame(
        {"gap": [c[2:].replace("_", "-") for c in cols], "coef": m.coef()[cols].values, "se": m.se()[cols].values}
    )
    raw = t.groupby("gb")["sim_next_award"].agg(["mean", "std", "count"]).reset_index().rename(columns={"gb": "gap"})
    raw["se_raw"] = raw["std"] / np.sqrt(raw["count"])
    fe = raw.merge(fe, on="gap", how="left").fillna({"coef": 0.0, "se": 0.0})
    fe.to_csv(signals.RESULTS / "pulse_sim_by_gap_fe.csv", index=False)
    print(fe.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
