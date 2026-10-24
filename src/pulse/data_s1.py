"""Data S1: the numbers behind every figure panel, one sheet per panel, with 95% intervals and two-sided P values.

python -m pulse.data_s1  ->  manuscript/data_S1.xlsx
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pulse.signals import RESULTS as R

OUT = Path(__file__).resolve().parents[2] / "manuscript" / "data_S1.xlsx"


def ci(d: pd.DataFrame, b: str, se: str) -> pd.DataFrame:
    d = d.copy()
    d["ci_low"], d["ci_high"] = d[b] - 1.96 * d[se], d[b] + 1.96 * d[se]
    d["p_two_sided"] = 2 * stats.norm.sf(np.abs(d[b] / d[se]))
    return d


def sheets() -> dict[str, pd.DataFrame]:
    s = {}
    h = pd.read_csv(R / "year_heatmap.csv")
    s["Fig1b"] = h[(h["server"] == "biorxiv") & h["year"].between(2017, 2025)]
    s["Fig1c"] = ci(pd.read_csv(R / "daily_step_profile.csv"), "coef", "se")
    s["Fig1d"] = ci(pd.read_csv(R / "shift_scan.csv"), "step", "se")
    sa = pd.read_csv(R / "season_field_counts.csv")
    s["Fig1e"] = ci(sa[sa["category"] == "all fields"], "coef", "se")
    s["Fig2b"] = pd.read_csv(R / "pi_phase_profile_2w.csv")
    s["Fig2c"] = ci(pd.read_csv(R / "resub_events.csv"), "coef", "se")
    t = pd.read_csv(R / "phase_scan.csv")
    cp = pd.read_csv(R / "conv_pred.csv")
    s["Fig3b observed"] = ci(t[t["scan"].str.endswith("aggregate")], "coef", "se")
    s["Fig3b individual"] = cp[cp["model"] == "new applications"]
    fb = pd.read_csv(R / "conv_falsification.csv")
    fb["plotted"] = fb["amp"].isna() | (fb["amp"] > 0)  # a negative amplitude inverts the model
    s["Fig3c"] = fb
    s["Fig4b"] = pd.read_csv(R / "pulse_sim_by_gap_fe.csv")
    pc = pd.read_csv(R / "pulse_content.csv")
    pc = pc[pc["var"] != "rs_head_abs_log"]
    s["Fig4c"] = pc
    s["Fig4c citations"] = pd.read_csv(R / "citations_within.csv")
    s["figS1 scans"] = ci(t, "coef", "se")
    s["figS1e fields"] = ci(pd.read_csv(R / "field_counts.csv"), "coef", "se")
    s["figS2"] = pd.read_csv(R / "pre2017_byfield.csv")
    s["figS3a"] = pd.read_csv(R / "field_split_categories.csv")
    s["figS3b"] = pd.read_csv(R / "field_intercept_profile.csv")
    s["figS3c"] = pc
    return s


POISSON = (
    "Poisson model of daily first versions by group (US, non-US) with day, group-by-cycle and group-by-weekday "
    "fixed effects, SE clustered by four-month cycle"
)
WITHIN = (
    "weighted share OLS per investigator-cycle (window posts / cycle posts) with investigator and calendar-cycle "
    "fixed effects, weights = cycle posts, SE clustered by investigator, reported as log(1 + b / mean share)"
)
README = [
    (
        "Fig1b",
        "Log ratio of US to non-US first versions, bioRxiv, by year and 7-day bin of days to the next R01 due date",
        "results/year_heatmap.csv",
        "python -m pulse.group_rates",
        "log((US + 0.5) / (non-US + 0.5)) per year x bin, minus the year mean",
    ),
    (
        "Fig1c",
        "Daily US vs non-US log ratio from 21 days before to 20 days after the R01 due date, bioRxiv",
        "results/daily_step_profile.csv",
        "python -m pulse.daily_step",
        POISSON + "; one US x day indicator per day relative to 5 February, 5 June and 5 October; cycles centred on "
        "these dates",
    ),
    (
        "Fig1d",
        "Drop at an assumed due date moved -60 to +60 days from the true R01 dates, both servers",
        "results/shift_scan.csv",
        "python -m pulse.shift_scan",
        POISSON
        + "; step = US x (7 days before) minus US x (7 days from) the assumed date, cycles centred on the assumed "
        "dates; SE from the model covariance",
    ),
    (
        "Fig1e",
        "R01- and F-window US vs non-US coefficients without and with calendar-month fixed effects, bioRxiv",
        "results/season_field_counts.csv (category = 'all fields')",
        "python -m pulse.season_checks",
        POISSON + "; second model adds group-by-calendar-month fixed effects",
    ),
    (
        "Fig2b",
        "Two-week share of preprints, application cycles minus other cycles, pooled, with placebo band",
        "results/pi_phase_profile_2w.csv",
        "python -m pulse.pi_design phase_profile 200",
        "shares pooled over the preprints of investigators who posted in both kinds of cycle, no fixed effects; "
        "column pis counts investigators, units counts investigator-schedule units; "
        "SE from 200 bootstrap resamples of investigators; "
        "placebo band, 2.5th to 97.5th percentiles over 200 draws replacing each application cycle with another cycle",
    ),
    (
        "Fig2c",
        "Event studies of preprint counts in 14-day bins around the own due date: first submissions at the "
        "new-application date, funded resubmissions at the resubmission date",
        "results/resub_events.csv",
        "python -m pulse.resub_checks investigator",
        "Poisson with event and calendar-bin fixed effects, SE clustered by investigator; reference bins 280 days "
        "before and after the due date; resubmission curve drawn 30 days later",
    ),
    (
        "Fig3b observed",
        "Week-by-week US vs non-US log ratio around the cycle, both servers",
        "results/phase_scan.csv (scan = '<server> aggregate')",
        "python -m pulse.phase_scan",
        POISSON + "; US x 7-day-bin terms, centred to average zero",
    ),
    (
        "Fig3b individual",
        "Individual surges (pooled profile, weeks 0-3) placed before every new-application due date (dashed curve)",
        "results/conv_pred.csv (model = 'new applications')",
        "python -m pulse.conv_model",
        "pooled excess in weeks 0-3 before the own due date (results/pi_phase_profile.csv, negatives set "
        "to 0), "
        "placed before each new-application due date (5th, 12th, 16th, 25th), weighted by funded applications "
        "(results/pi_duedist.csv), one amplitude fitted by inverse-variance weighted least squares",
    ),
    (
        "Fig3c",
        "Fit to the national profile when the individual surges are moved -8 to +8 weeks; free-phase sinusoid",
        "results/conv_falsification.csv",
        "python -m pulse.post_checks",
        "surges as in Fig3b individual, one amplitude per shift; fits with a negative amplitude (plotted = False) are "
        "not shown because they would mean fewer US preprints before the due dates; sinusoid "
        "with the mean cycle (365.25/3 days) as its period and a free phase",
    ),
    (
        "Fig4b",
        "Similarity of a preprint to the following award abstract, by days before the due date",
        "results/pulse_sim_by_gap_fe.csv",
        "python -m pulse.sim_check",
        "TF-IDF cosine similarity regressed on distance bins with investigator and year-month fixed effects; reference "
        "181-365 days",
    ),
    (
        "Fig4c",
        "Pre-deadline preprints (0-13 days) vs the same investigator's other preprints",
        "results/pulse_content.csv",
        "python -m pulse.pulse_content",
        "linear models with investigator and year-month fixed effects, SE clustered by investigator; plotted as % of "
        "the comparison mean (100 x log difference for logged outcomes)",
    ),
    (
        "Fig4c citations",
        "Citation percentile of pre-deadline preprints vs the same investigator's other preprints",
        "results/citations_within.csv",
        "python -m pulse.citations",
        "citations in the posting year and two following years ranked within server x category x month; investigator "
        "and "
        "year-month fixed effects, SE clustered by investigator; plotted as % of the comparison mean",
    ),
    (
        "figS1 scans",
        "Week-by-week scans: aggregate, HIV, NIH intramural (panels A-D)",
        "results/phase_scan.csv",
        "python -m pulse.phase_scan",
        POISSON + "; terms as named in the 'scan' and 'term' columns",
    ),
    (
        "figS1e fields",
        "R01-window coefficient on daily counts by field",
        "results/field_counts.csv",
        "python -m pulse.phase_scan",
        POISSON + ", fitted per bioRxiv category",
    ),
    (
        "figS2",
        "US share in R01 and F windows before and after 24 March 2017, bioRxiv",
        "results/pre2017_byfield.csv",
        "python -m pulse.pre2017 byfield",
        "US share on the window indicator with cycle fixed effects, SE clustered by cycle; country by the same rule "
        "in both periods",
    ),
    (
        "figS3a",
        "R01-window rise in US share (pp) against field NIH dependence, 21 bioRxiv categories",
        "results/field_split_categories.csv",
        "python -m pulse.field_split",
        "per category: US share on the R01-window indicator with cycle fixed effects, SE clustered by week; "
        "dependence = NIH share among US funded first versions; line = inverse-variance weighted least squares",
    ),
    (
        "figS3b",
        "Slope and intercept of week-by-week field coefficients on NIH dependence",
        "results/field_intercept_profile.csv",
        "python -m pulse.post_checks",
        "from results/phase_scan.csv (scan = 'field'): per 7-day bin, inverse-variance WLS of field coefficients on "
        "dependence, HC1 SE",
    ),
    (
        "figS3c",
        "All content measures of pre-deadline preprints",
        "results/pulse_content.csv, results/citations_within.csv",
        "python -m pulse.pulse_content; python -m pulse.citations",
        "as Fig4c",
    ),
]


def main() -> None:
    sh = sheets()
    readme = pd.DataFrame(README, columns=["sheet", "what is plotted", "source file", "command", "how it is computed"])
    assert list(readme["sheet"]) == list(sh)
    readme["rows"] = [len(v) for v in sh.values()]
    head = pd.DataFrame(
        {
            "sheet": ["", "", ""],
            "what is plotted": [
                "Data S1 for 'Funding deadlines act as a zeitgeber for science'. One sheet per figure panel.",
                "Commands run from the root of github.com/Wang-Haining/zeitgeber after the data pipeline in its "
                "README; "
                "each writes the source file named here.",
                "ci_low, ci_high: estimate +/- 1.96 SE. p_two_sided: normal approximation to estimate / SE. Log scale "
                "unless a column says pp.",
            ],
        }
    )
    readme = pd.concat([head, readme], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT) as w:
        readme.to_excel(w, sheet_name="README", index=False)
        ws = w.sheets["README"]
        from openpyxl.styles import Alignment, Font

        for col, width in zip("ABCDEF", (14, 48, 40, 44, 70, 7), strict=False):
            ws.column_dimensions[col].width = width
        for row in ws.iter_rows():
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        for c in ws[1]:
            c.font = Font(bold=True)
        for k, v in sh.items():
            v.to_excel(w, sheet_name=k[:31], index=False)
    print("wrote", OUT, len(sh), "sheets")


if __name__ == "__main__":
    main()
