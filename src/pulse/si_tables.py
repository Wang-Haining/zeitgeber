"""Supplementary tables written straight from results/*.csv (no hand transcription).

python -m pulse.si_tables   ->  manuscript/si_tables.tex
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from pulse.signals import RESULTS as R

OUT = Path(__file__).resolve().parents[2] / "manuscript" / "si_tables.tex"
SRV = {"biorxiv": "bioRxiv", "medrxiv": "medRxiv"}
WIN = {"R_new": "R01", "F": "F", "AIDS": "AIDS", "R_resub": "Resubmission"}


def f(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$")


def es(b, se, d=3):
    return f"{f(b, d)} ({f(se, d)})"


def table(label, caption, cols, rows, align=None, note=None):
    head, _, rest = caption.partition(". ")
    caption = rf"\textbf{{{head}.}} {rest}" if rest else rf"\textbf{{{head.rstrip('.')}.}}"
    align = align or "l" + "r" * (len(cols) - 1)
    align = align.replace("p{", r">{\raggedright\arraybackslash}p{")
    if len(rows) > 40:  # long tables break across pages and repeat the header
        head = " & ".join(cols) + r" \\"
        out = [
            r"\clearpage",  # flush pending float tables so numbering and page order agree
            r"\begingroup\footnotesize\renewcommand{\arraystretch}{1.1}",
            rf"\begin{{longtable}}{{{align}}}",
            rf"\caption{{{caption}}}\label{{{label}}}\\",
            r"\toprule",
            head,
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            head,
            r"\midrule",
            r"\endhead",
            r"\bottomrule",
            r"\endlastfoot",
        ]
        out += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
        out += [r"\end{longtable}", r"\endgroup"]
        return "\n".join(out)
    out = [
        r"\begin{table}[htbp]",
        r"\centering\footnotesize",
        r"\renewcommand{\arraystretch}{1.1}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{adjustbox}{max width=\textwidth}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    out += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    if note:
        out.append(rf"\par\smallskip\begin{{minipage}}{{0.92\textwidth}}\footnotesize {note}\end{{minipage}}")
    out.append(r"\end{table}")
    return "\n".join(out)


def t_aggregate():
    d = pd.read_csv(R / "d0_cycle_clustered.csv")
    rows = []
    for srv in ("biorxiv", "medrxiv"):
        for w in ("R_new", "AIDS", "F", "R_resub"):
            r = d[(d["server"] == srv) & (d["window"] == w)].iloc[0]
            rows.append([SRV[srv], WIN[w], es(r["pp"], r["se_pp_cycle_cluster"], 2), int(r["n_cycles"])])
    return table(
        "tab:aggregate",
        "Change in the US share of first versions in each window, estimated one window at a time "
        "against the rest of its four-month cycle (linear probability model, cycle fixed effects, SE clustered "
        "by cycle). These are the estimates cited in the main text; table~\\ref{tab:contrast} gives the R01 and F "
        "windows "
        "estimated jointly and on daily counts.",
        ["Server", "Window", "US share, pp (SE)", "Cycles"],
        rows,
        "llrr",
    )


def t_cycles():
    d = pd.read_csv(R / "design_checks.csv")
    d = d[d["check"] == "per cycle"]
    rows = []
    for srv in ("biorxiv", "medrxiv"):
        for m, lab in ((2, "February"), (6, "June"), (10, "October")):
            s = d[(d["server"] == srv) & (d["cycle_month"] == m)].set_index("what")
            g = (
                s.loc["field gradient slope (R01 window, counts)"]
                if "field gradient slope (R01 window, counts)" in s.index
                else None
            )
            rows.append(
                [
                    SRV[srv],
                    lab,
                    es(s.loc["US x R_new", "coef"], s.loc["US x R_new", "se"]),
                    es(s.loc["US x F", "coef"], s.loc["US x F", "se"]),
                    es(g["coef"], g["se"]) if g is not None else "--",
                    es(g["intercept"], g["intercept_se"]) if g is not None else "--",
                ]
            )
    return table(
        "tab:cycles",
        "Rhythm by annual cycle. US $\\times$ window coefficients on log daily counts (Poisson, day "
        "fixed effects, SE clustered by cycle), and the bioRxiv field gradient of the R01-window coefficient on "
        "NIH dependence (21 categories, inverse-variance weighted, HC1 SE).",
        ["Server", "Cycle", "US $\\times$ R01", "US $\\times$ F", "Gradient slope", "Gradient intercept"],
        rows,
        "llrrrr",
    )


def t_fields():
    s = pd.read_csv(R / "field_split_categories.csv")
    c = pd.read_csv(R / "field_counts.csv")
    c = c[c["window"] == "R_new"].set_index("category")
    s = s.sort_values("nih_share_us_funded")
    rows = []
    for _, r in s.iterrows():
        cc = c.loc[r["category"]] if r["category"] in c.index else None
        rows.append(
            [
                r["category"].capitalize(),
                f"{int(r['n']):,}",
                f(r["nih_share_us_funded"], 2),
                es(r["pulse_R_new_pp"], r["se_pp"], 2),
                es(r["pulse_F_pp"], r["se_F_pp"], 2),
                es(cc["coef"], cc["se"]) if cc is not None else "--",
            ]
        )
    rb = pd.read_csv(R / "field_split_robustness.csv")
    lab = {
        "weighted, all categories": "weighted, all categories (heteroskedasticity-robust SE)",
        "leave-one-out min (drop plant biology)": "leave-one-out minimum (plant biology dropped)",
        "leave-one-out max (drop neuroscience)": "leave-one-out maximum (neuroscience dropped)",
        "period discovery": "first half of the period",
        "period confirm": "second half of the period",
    }
    rb = rb[rb["spec"].isin(lab)]
    note = ". ".join(
        f"{lab[r['spec']].capitalize()}: slope {f(r['slope'], 2)}"
        + (f" (SE {f(r['se'], 2)})" if pd.notna(r["se"]) else "")
        for _, r in rb.iterrows()
    )
    fc = pd.read_csv(R / "field_cluster.csv")
    fc = fc[fc["measure"] == "share, pp"].set_index("group")
    lo, hi, dd = fc.loc["dependence < 0.6"], fc.loc["dependence >= 0.6"], fc.loc["high minus low"]
    note += (
        f". Inverse-variance means: dependence below 0.6 ({int(lo['k'])} categories) {es(lo['mean'], lo['se'], 2)} pp, "
        f"0.6 or more ({int(hi['k'])} categories) {es(hi['mean'], hi['se'], 2)} pp, difference "
        f"{es(dd['mean'], dd['se'], 2)} pp with categories treated as independent; Spearman correlation "
        f"with dependence within the second group {f(hi['spearman'], 2)} ($P = {f(hi['spearman_p'], 2)}$)"
    )
    return table(
        "tab:fields",
        "bioRxiv subject categories ordered by NIH dependence. Change in US share in the R01 and F "
        "windows (pp, cycle fixed effects) and the US $\\times$ R01-window coefficient on log daily counts.",
        ["Category", "First versions", "NIH dependence", "R01 window, pp", "F window, pp", "R01, log counts"],
        rows,
        "lrrrrr",
        note="Slopes on dependence (share-based): " + note.replace("_", r"\_") + ".",
    )


def t_2017():
    p = pd.read_csv(R / "pre2017.csv")
    b = pd.read_csv(R / "pre2017_byfield.csv")
    rows = []
    for _, r in p.iterrows():
        rows.append(
            [
                "All categories",
                r["period"].split(" (")[0],
                r["measure"].split(",")[0],
                WIN[r["window"]],
                es(r["coef"], r["se"], 3 if "log" in r["measure"] else 2),
                f"{int(r['n']):,}" if pd.notna(r["n"]) else "--",
            ]
        )
    for _, r in b.iterrows():
        rows.append(
            [
                r["fields"].split(" (")[0].capitalize(),
                r["period"],
                "US share (pp)",
                WIN[r["window"]],
                es(r["coef"], r["se"], 2),
                f"{int(r['n']):,}",
            ]
        )
    return table(
        "tab:2017",
        "Before and after NIH allowed applications to cite preprints (NOT-OD-17-050, 24 March 2017), "
        "bioRxiv. Country assigned in both periods from "
        "the ROR match of the corresponding institution.",
        ["Fields", "Period", "Measure", "Window", "Estimate (SE)", "N"],
        rows,
        "p{2.6cm}p{2.2cm}p{4.2cm}lrr",
    )


def t_designs():
    rows = []
    names = {
        "main": "Funded new applications",
        "failed_a0": "Failed first submission behind a funded A1",
        "renewal": "Funded renewals",
        "itt": "Renewal clock, all R01s",
    }
    for k, lab in names.items():
        r = pd.read_csv(R / f"pi_within_{k}.csv")
        for spec, sl in (("across", "Across investigators (log OR)"), ("within", "Within investigator")):
            s = r[r["spec"].str.startswith(spec)].set_index("treatment")
            pl = s.loc["clean placebo"]
            rows.append(
                [
                    lab if spec == "across" else "",
                    sl,
                    es(s.loc["treated", "log_rr"], s.loc["treated", "se"]),
                    es(s.loc["treated_prev", "log_rr"], s.loc["treated_prev", "se"]),
                    es(s.loc["treated_next", "log_rr"], s.loc["treated_next", "se"]),
                    f"{f(pl['log_rr'])} [{f(pl['p_one_sided'])}]",
                    f"{int(s.loc['treated', 'pis']):,}",
                ]
            )
    return table(
        "tab:designs",
        "Investigator-level estimates for four designs. Own application cycle, the cycle before and "
        "the cycle after (28-day window). Placebo: mean of 200 estimates with application cycles removed "
        "and placebo cycles drawn among the remaining cycles [one-sided $P$ of the observed estimate].",
        ["Design", "Estimand", "Own cycle", "Cycle before", "Cycle after", "Placebo mean [$P$]", "Investigators"],
        rows,
        "p{3.6cm}p{3.4cm}rrrrr",
    )


def t_decomp():
    d = pd.read_csv(R / "pi_decomp.csv")

    def spell(t):
        return (
            t.replace("(uc + cw)", "(investigator-by-cycle and cycle-by-window FE)")
            .replace("unit + calendar-cycle FE", "investigator and calendar-cycle FE")
            .replace(">= 2 cycles", "at least 2 cycles")
            .replace("OLS", "regression")
        )

    rows = [[int(r["window_days"]), spell(r["spec"][2:]), es(r["log_rr"], r["se"])] for _, r in d.iterrows()]
    return table(
        "tab:decomp",
        "From the across-investigator estimate to the within-investigator estimate (funded new "
        "applications). The Poisson model with investigator-by-cycle fixed effects estimates a log odds ratio of "
        "the window share; share models report $\\log(1+b/\\bar y)$.",
        ["Window (days)", "Specification", "Estimate (SE)"],
        rows,
        "rp{10.5cm}r",
    )


def t_sim():
    d = pd.read_csv(R / "pi_sim_draws.csv")
    p = 0.2516  # weighted mean window share in the main panel
    lo = np.log(p / (1 - p))
    share_truth = np.log(1 / (1 + np.exp(-(lo + 0.18))) / p)
    rows = []
    for (b, w), g in d.groupby(["beta", "within"]):
        w = str(w).lower() == "true"
        truth = share_truth if (w and b > 0) else b
        rows.append(
            [
                f(b, 2),
                "within investigator (share scale)" if w else "across investigators (log odds)",
                f(truth),
                f(g["log_rr"].mean()),
                f(g["log_rr"].std()),
                f(g["se"].mean()),
                f(float((np.abs(g["log_rr"] - truth) <= 1.96 * g["se"]).mean()), 2),
            ]
        )
    return table(
        "tab:sim",
        "Simulation on the observed panel (100 replicates). Investigator-cycle totals fixed; window "
        "counts binomial with the calendar-cycle share plus investigator noise (SD 0.5 on the log-odds scale) and "
        "a known log-odds effect $\\beta$ on application cycles. Each estimator is evaluated against the truth "
        "on its own scale.",
        ["$\\beta$", "Estimator", "Truth", "Mean estimate", "SD", "Mean SE", "Coverage"],
        rows,
        "lp{4.6cm}rrrrr",
    )


def t_month():
    b = pd.read_csv(R / "pi_bymonth.csv")
    b = b[b["within"].astype(str).str.lower() == "true"]
    rows = []
    for m in sorted(b["notice_month"].unique()):
        s = b[b["notice_month"] == m].set_index("treatment")
        rows.append(
            [
                f"Notice month {m}",
                es(s.loc["treated_prev", "log_rr"], s.loc["treated_prev", "se"]),
                es(s.loc["treated", "log_rr"], s.loc["treated", "se"]),
                es(s.loc["treated_next", "log_rr"], s.loc["treated_next", "se"]),
                int(s.loc["treated", "treated_pre_posts"]),
            ]
        )
    MONTHS = {2: "February", 6: "June", 10: "October"}
    c = pd.read_csv(R / "pi_bycycle.csv")
    c = c[c["within"].astype(str).str.lower() == "true"]
    for _, r in c.iterrows():
        rows.append(
            [
                f"Cycle family {MONTHS[int(r['cycle_family'])]}",
                "--",
                es(r["log_rr"], r["se"]),
                "--",
                int(r["treated_pre_posts"]),
            ]
        )
    return table(
        "tab:month",
        "Within-investigator estimates by award notice month (checks the inferred due date: a mapping "
        "wrong by one cycle would move the effect to the cycle before or after) and by annual cycle.",
        ["Subset", "Cycle before", "Own cycle", "Cycle after", "Pre-deadline posts, own cycle"],
        rows,
        "p{3.4cm}rrrp{2.4cm}",
    )


def t_totals():
    t = pd.read_csv(R / "pi_totals.csv")
    names = {
        "main": "Funded new applications",
        "failed_a0": "Failed first submission",
        "renewal": "Funded renewals",
        "itt": "Renewal clock, all R01s",
    }
    rows = []
    for k, lab in names.items():
        s = t[t["design"] == k].set_index("term")
        rows.append(
            [lab]
            + [
                es(s.loc[x, "log_rr"], s.loc[x, "se"])
                for x in ("treated_prev", "treated", "treated_next", "sum prev+own+next")
            ]
        )
    return table(
        "tab:totals",
        "Total preprints per investigator-cycle around applications (Poisson, investigator and "
        "calendar-cycle fixed effects, cycles with no preprints kept). Output is higher in the cycles around an "
        "application.",
        ["Design", "Cycle before", "Own cycle", "Cycle after", "Sum"],
        rows,
        "lrrrr",
    )


def t_content():
    c = pd.read_csv(R / "pulse_content.csv")
    rows = [
        [
            r["outcome"]
            .replace("hype per 1000 words", "promotional adjectives per 1,000 words (without ``significant'')")
            .replace("|", "$|$")
            .replace("<", "$<$"),
            es(r["coef"], r["se"]),
            f(r["mean_nonpulse"]),
            f"{int(r['n']):,}",
            int(r["n_pulse"]),
        ]
        for _, r in c.iterrows()
    ]
    s = pd.read_csv(R / "pulse_sim_check.csv")
    for _, r in s.iterrows():
        nm = {
            "sim_next_award": "similarity to the upcoming award",
            "sim_diff": "upcoming minus other awards",
            "hype_pk": "promotional adjectives per 1,000 words (without ``significant'')",
            "rs_head_abs_log": "Results $|$log ratio$|$",
        }
        sp = {
            "all, control log gap": "adjusting for log days to due date",
            "gap <= 90 days": "within 90 days of due date",
            "gap <= 45 days": "within 45 days of due date",
        }
        rows.append(
            [
                f"{nm.get(r['outcome'], r['outcome'])}, {sp.get(r['spec'], r['spec'])}",
                es(r["coef"], r["se"]),
                "--",
                f"{int(r['n']):,}",
                int(r["n_pulse"]),
            ]
        )
    return table(
        "tab:content",
        "Preprints posted 0--13 days before the investigator's own due date compared with the same "
        "investigator's other preprints (investigator and year-month fixed effects, SE clustered by investigator).",
        ["Outcome", "Difference (SE)", "Comparison mean", "N", "Pre-deadline"],
        rows,
        "p{7.5cm}rrrr",
    )


def t_scale():
    t = pd.read_csv(R / "consequence_scale.csv")
    rows = [
        [
            int(r["year"]),
            f"{int(r['us_total']):,}",
            f"{r['rnew_excess']:.0f}",
            f"{r['f_deficit']:.0f}",
            f"{100 * r['excess_share_of_us_total']:.1f}",
        ]
        for _, r in t.iterrows()
    ]
    return table(
        "tab:scale",
        "US first versions in R01 windows above, and in F windows below, the number expected from "
        "non-US counts and the US-to-non-US ratio outside the R01 and F windows in the same year (2026: "
        "January--June).",
        ["Year", "US first versions", "R01-window excess", "F-window deficit", "Excess, \\% of US"],
        rows,
        "rrrrr",
    )


def t_consequences():
    c = pd.read_csv(R / "consequence_congestion.csv")
    a = pd.read_csv(R / "consequence_ackaward.csv")
    rows = [[r["sample"], r["term"], es(r["coef"], r["se"]), f"{int(r['n']):,}"] for _, r in c.iterrows()]
    ack = {
        "ack_pre": "preprint acknowledges the award",
        "ack_journal": "journal version acknowledges the award",
        "ack_any": "either acknowledges the award",
    }
    spec = {
        "all, control log gap": "adjusting for log days to due date",
        "gap <= 90 days": "within 90 days of due date",
        "journal version exists, control log gap": "journal version exists, adjusting for log days",
    }
    rows += [
        [
            "Award acknowledgment",
            f"{ack[r['outcome']]} ({spec.get(r['spec'], r['spec'])})",
            es(r["coef"], r["se"]),
            f"{int(r['n']):,}",
        ]
        for _, r in a.iterrows()
    ]
    return table(
        "tab:consequences",
        "Citation cost of deadline crowding (log(1 + citations), field-by-year and calendar-week "
        "fixed effects, SE clustered by week) and acknowledgment of the forthcoming award (linear probability, "
        "investigator and year-month fixed effects).",
        ["Sample", "Term", "Estimate (SE)", "N"],
        rows,
        "p{3.2cm}p{7.6cm}rr",
    )


def t_checks():
    p = pd.read_csv(R / "counts_poisson.csv")
    rows = []
    for test in (
        "biorxiv HIV (all days)",
        "biorxiv HIV (conferences removed)",
        "medrxiv HIV (all days)",
        "biorxiv intramural",
        "medrxiv intramural",
        "biorxiv linked",
        "medrxiv linked",
    ):
        s = p[(p["test"] == test) & p["term"].isin(["R_new", "AIDS", "F"])]
        for _, r in s.iterrows():
            lab = (
                test.replace("biorxiv", "bioRxiv")
                .replace("medrxiv", "medRxiv")
                .replace("linked", "linked investigators")
                .replace("intramural", "NIH intramural")
            )
            rows.append(
                [
                    lab,
                    r["model"][0],
                    r["group"].replace("_x_", " $\\times$ ").replace("_", " "),
                    WIN[r["term"]],
                    es(r["coef"], r["se"]),
                ]
            )
    d = pd.read_csv(R / "design_checks.csv")
    for _, r in d[d["check"] == "HIV joint model"].iterrows():
        rows.append(
            [
                f"{SRV[r['server']]} HIV, joint model",
                "B",
                r["what"].replace("UH_", "US $\\times$ HIV $\\times$ ").replace("_", " ").replace(" x ", " $\\times$ "),
                "--",
                es(r["coef"], r["se"]),
            ]
        )
    return table(
        "tab:checks",
        "Consistency checks on daily counts (Poisson). Model A: per-group rate ratio, cycle and "
        "weekday fixed effects. Model B: relative to the reference group on the same day, with day, "
        "group-by-cycle and group-by-weekday fixed effects. SE clustered by cycle.",
        ["Comparison", "Model", "Group or term", "Window", "Estimate (SE)"],
        rows,
        "p{4.2cm}lp{4.2cm}lr",
    )


def t_other():
    cc = pd.read_csv(R / "country_calendar.csv")
    cc = cc[cc["country"].isin(["US", "CA"]) & cc["window"].isin(["R_new", "F"])]
    rows = [
        [SRV[r["server"]], r["country"], WIN[r["window"]], es(r["pp"], r["se_pp"], 2), f(100 * r["rel"], 1)]
        for _, r in cc.iterrows()
    ]
    fp = pd.read_csv(R / "funder_pulse_nonUS.csv")
    fp = fp[
        ((fp["funder"] == "NSFC") & (fp["country"] == "CN")) | ((fp["funder"] == "KAKENHI") & (fp["country"] == "JP"))
    ]
    for _, r in fp.iterrows():
        rows.append(
            [
                SRV[r["server"]],
                f"{r['country']} ({r['funder']})",
                "own due date" if r["shift_days"] == 0 else f"placebo {int(r['shift_days']):+d} d",
                es(r["pp"], r["se_pp"], 2),
                f(100 * r["rel"], 1),
            ]
        )
    return table(
        "tab:other",
        "Other countries and funders. Country shares are among all country-coded first versions, so the "
        "US rows differ slightly from table~\\ref{tab:aggregate}, which compares US with all non-US authors; SE "
        "clustered by week. "
        "Canada shares the North American academic year and may apply for R01 grants but not NIH fellowships. NSFC and "
        "KAKENHI: share of Chinese or Japanese corresponding authors "
        "among non-US first versions in the 28 days before the funder's due date and at placebo dates.",
        ["Server", "Country", "Window", "Change, pp (SE)", "Relative, \\%"],
        rows,
        "lp{2.8cm}p{3cm}rr",
    )


def t_link():
    a = pd.read_csv(R / "pi_link_audit.csv").iloc[0]
    s = pd.read_csv(R / "pi_link_audit_strata.csv")
    rows = [
        ["Anchored links (investigators)", f"{int(a['anchored_links']):,} ({int(a['pis_anchored']):,})"],
        ["Name-and-institution links (investigators)", f"{int(a['name_inst_links']):,} ({int(a['pis_name_inst']):,})"],
        ["Name-and-institution links confirmed by anchor, all", f(a["precision_proxy_all"], 2)],
        ["Confirmed, stratified sample of 200", f(a["precision_proxy_sample200"], 2)],
    ]
    for _, r in s.iterrows():
        rows.append(
            [
                f"Confirmed: {'common' if r['surname_common'] else 'uncommon'} surname, "
                f"{'several' if r['ids_per_pi'] else 'one'} OpenAlex ID(s)",
                f"{f(r['mean'], 2)} (n = {int(r['size']):,})",
            ]
        )
    return table(
        "tab:link",
        "Linking investigators to OpenAlex authors. Only anchored links (an author named on a paper "
        "acknowledging the investigator's own award) were used.",
        ["", "Value"],
        rows,
        "p{10cm}r",
    )


def t_rev2():
    r = pd.read_csv(R / "group_rates.csv")
    rows = []
    g = r[r["analysis"] == "per-group rate ratio"]
    for _, x in g.iterrows():
        rows.append(
            [
                SRV[x["server"]],
                "rate ratio, " + ("US" if x["group"] == "US" else "non-US"),
                WIN[x["term"]],
                es(x["coef"], x["se"]),
                "--",
            ]
        )
    j = r[r["analysis"] == "joint US x window (day FE)"]
    for _, x in j.iterrows():
        rows.append(
            [
                SRV[x["server"]],
                "US vs non-US, day FE",
                {"xR": "R01", "xF": "F"}.get(x["term"], "R01 $-$ F"),
                es(x["coef"], x["se"]),
                "--",
            ]
        )
    w = r[r["analysis"] == "share, cycle FE, wild cluster bootstrap"]
    for _, x in w.iterrows():
        rows.append(
            [
                SRV[x["server"]],
                f"US share (pp), {int(x['clusters'])} clusters",
                x["term"].replace("share ", "").replace(" - ", " $-$ "),
                es(x["coef"], x["se"], 2),
                f(x["p_wild"], 3),
            ]
        )
    t1 = table(
        "tab:contrast",
        "Aggregate rhythm on daily counts and the R01 $-$ F contrast. Rate-ratio rows are log rate "
        "ratios from Poisson models of each group's daily counts (cycle and weekday fixed effects). US-versus-non-US "
        "rows come from one Poisson model with both window indicators and day fixed effects. Share rows come from "
        "one linear model with both window indicators, which is why the R01 share differs from the one-window "
        "estimate in table~\\ref{tab:aggregate}. Wild cluster bootstrap: Rademacher weights by cycle, 999 draws, $P$ "
        "for the null of "
        "zero.",
        ["Server", "Measure", "Window", "Estimate (SE)", "Wild bootstrap $P$"],
        rows,
        "lp{4.6cm}lrr",
    )
    y = r[r["analysis"] == "year"]
    rows = []
    for (srv, yr), s_ in y.groupby(["server", "group"]):
        s_ = s_.set_index("term")
        rows.append(
            [
                SRV[srv],
                yr,
                es(s_.loc["R01", "coef"], s_.loc["R01", "se"]),
                es(s_.loc["F", "coef"], s_.loc["F", "se"]),
                es(s_.loc["R01 - F", "coef"], s_.loc["R01 - F", "se"]),
            ]
        )
    t2 = table(
        "tab:years",
        "Year by year: US $\\times$ window on daily counts (Poisson, day fixed effects, SE by cycle).",
        ["Server", "Year", "R01 window", "F window", "R01 $-$ F"],
        rows,
        "llrrr",
    )
    fu = r[r["analysis"].str.startswith("funder group")]

    def flab(a):
        return "All fields" if "all fields" in a else "Fields with NIH dependence at least 0.7"

    rows = [
        [flab(x["analysis"]), x["group"], WIN[x["term"]], es(x["coef"], x["se"]), f"{int(x['n_group']):,}"]
        for _, x in fu.iterrows()
    ]
    t3 = table(
        "tab:funders",
        "US bioRxiv first versions by OpenAlex funder group, each relative to non-US first versions "
        "on the same day (Poisson, day, group-by-cycle and group-by-weekday fixed effects). All US groups rise in the "
        "R01 window, by amounts that cannot be distinguished.",
        ["Fields", "US group", "Window", "Estimate (SE)", "N"],
        rows,
        "p{3.6cm}p{3.2cm}lrr",
    )
    return "\n\n".join([t1, t2, t3])


def t_rev_pi():
    a = pd.read_csv(R / "pi_adjacent.csv")
    names = {
        "main": "Funded new applications",
        "failed_a0": "Failed first submission",
        "renewal": "Funded renewals",
        "itt": "Renewal clock, all R01s",
    }
    rows = [
        [names[x["design"]], x["reference"], es(x["log_rr"], x["se"]), f"{int(x['pis']):,}"] for _, x in a.iterrows()
    ]
    t1 = table(
        "tab:adjacent",
        "Within-investigator estimates with the cycles immediately before and after each application "
        "removed from the reference.",
        ["Design", "Reference cycles", "Estimate (SE)", "Investigators"],
        rows,
        "p{4.2cm}p{6cm}rr",
    )
    c = pd.read_csv(R / "conv_model.csv")
    rows = [
        [
            SRV[x["server"]],
            x["model"],
            f(x["corr"]),
            f(x["weighted_r2"]),
            f"{1e4 * x['amplitude']:.2f}",
            f(x["chi2"], 1),
            int(x["dof"]),
        ]
        for _, x in c.iterrows()
    ]
    t2 = table(
        "tab:conv",
        "The individual surge (investigator kernel) fitted to the national (aggregate) rhythm. "
        "The excess in the 0--3 weeks before "
        "an investigator's own due date (Fig.~2B) is placed before the due date of each schedule, weighted by the "
        "number of funded applications on that schedule, and fitted to the aggregate week-by-week profile with one "
        "amplitude (in units of $10^{-4}$ on the log ratio per weighted application).",
        ["Server", "Schedules", "Correlation", "Weighted $R^2$", "Amplitude", "$\\chi^2$", "df"],
        rows,
        "lp{5.2cm}rrrrr",
    )
    return "\n\n".join([t1, t2])


def t_round4():
    """Tables S22 onward: the step at the due date, window length, season, intercept profile, linkage without
    preprints, activity codes, falsification of the kernel fit."""
    CY = {"pooled": "February, June and October", "Feb": "February", "Jun": "June", "Oct": "October"}
    d = pd.read_csv(R / "daily_step.csv")
    rows = [
        [
            SRV[x["server"]],
            CY[x["cycles"]],
            es(x["step"], x["se"]),
            f(x["null_mean"]),
            f(x["null_sd"]),
            f(x["null_max"]),
            f(x["rank_p"]),
        ]
        for _, x in d.iterrows()
    ]
    for srv, g in d.groupby("server", sort=False):  # June and October without February, inverse-variance pooled
        g = g[g["cycles"].isin(["Jun", "Oct"])]
        w = 1 / g["se"] ** 2
        b, se = float((g["step"] * w).sum() / w.sum()), float(1 / np.sqrt(w.sum()))
        i = max(k for k, r in enumerate(rows) if r[0] == SRV[srv])
        rows.insert(i + 1, [SRV[srv], "June and October (inverse-variance pooled)", es(b, se), "--", "--", "--", "--"])
    t1 = table(
        "tab:step",
        "The drop at the R01 due date. Log ratio of US to non-US daily first versions in the 7 days "
        "before the due date minus the 7 days from the due date on (Poisson, day, group-by-cycle and "
        "group-by-weekday fixed effects, SE clustered by cycle). The null distribution is the same statistic with "
        "every due date shifted by 14 to 60 days in either direction (94 shifts); rank $P$ is the share of shifts "
        "with a drop at least as large, counting the true date.",
        ["Server", "Cycles", "Drop (SE)", "Null mean", "Null SD", "Null max", "Rank $P$"],
        rows,
        "lp{3.6cm}rrrrr",
    )
    w = pd.read_csv(R / "window_length.csv")
    rows = [
        [
            SRV[x["server"]],
            x["window"].replace("0-", "0--").replace(", as reported", ", main text"),
            es(x["pp"], x["se_pp"], 2),
        ]
        for _, x in w.iterrows()
    ]
    t2 = table(
        "tab:winlen",
        "R01-window change in the US share of first versions (pp, cycle fixed effects, SE clustered "
        "by cycle) for windows of different length, counted in days before the due date.",
        ["Server", "Window", "US share, pp (SE)"],
        rows,
        "llr",
    )
    a = pd.read_csv(R / "season_field_counts.csv")
    a = a[a["category"] == "all fields"]
    sl = pd.read_csv(R / "season_field_slope.csv")
    rows = []
    for wnd in ("R_new", "F"):
        for mfe in (False, True):
            x = a[(a["window"] == wnd) & (a["month_fe"] == mfe)].iloc[0]
            y = sl[(sl["window"] == wnd) & (sl["month_fe"] == mfe)].iloc[0]
            rows.append(
                [
                    WIN[wnd],
                    "yes" if mfe else "no",
                    es(x["coef"], x["se"]),
                    es(y["slope"], y["se"]),
                    es(y["intercept"], np.nan),
                ]
            )
    bc = pd.read_csv(R / "field_slope_by_cycles.csv")
    for _, y in bc.iterrows():
        rows.append(
            [
                f"R01, {y['cycles'].replace('Feb and Oct', 'February and October').replace('Jun', 'June')}",
                "no",
                "--",
                es(y["slope"], y["se"]),
                es(y["intercept"], y["intercept_se"]),
            ]
        )
    t3 = table(
        "tab:season",
        "Annual season. Upper rows: US $\\times$ window on bioRxiv daily counts, and the slope of the "
        "per-field window coefficient on NIH dependence (21 fields, inverse-variance weights, HC1 SE), without and "
        "with group-by-calendar-month fixed effects. Lower rows: the R01-window slope from the per-cycle field "
        "estimates, pooled by inverse variance within field over the cycles named.",
        ["Window", "Month FE", "All fields (SE)", "Slope on dependence (SE)", "Intercept (SE)"],
        rows,
        "p{4.2cm}lrrr",
    )
    ip = pd.read_csv(R / "field_intercept_profile.csv")
    rows = [
        [
            int(x["bin"]),
            f"{int(x['phi_mid'] - 3.5)}--{int(x['phi_mid'] + 2.5)}",
            es(x["intercept"], x["intercept_se"]),
            es(x["slope"], x["slope_se"]),
        ]
        for _, x in ip.iterrows()
    ]
    t4 = table(
        "tab:intercept",
        "Week-by-week field gradient on bioRxiv daily counts. For each 7-day bin of days to the "
        "next R01 due date, the per-field US $\\times$ bin coefficient regressed on NIH dependence (21 fields, "
        "inverse-variance weights, HC1 SE). The intercept is the coefficient expected for a field with no NIH "
        "dependence.",
        ["Bin", "Days to R01 due date", "Intercept (SE)", "Slope (SE)"],
        rows,
        "rlrr",
    )
    n = pd.read_csv(R / "pi_norxiv.csv")
    names = {"main": "Funded new applications", "failed_a0": "Failed first submission"}
    rows = [
        [
            names[x["design"]],
            x["spec"].capitalize(),
            es(x["log_rr"], x["se"]),
            f(x["null_p975"]),
            f(x["p_one_sided"]),
            f"{int(x['pis']):,}",
            f"{int(x['treated_pre_posts']):,}",
        ]
        for _, x in n.iterrows()
    ]
    t5 = table(
        "tab:norxiv",
        "Investigator-level estimates with links anchored only on works that are neither a bioRxiv "
        "or medRxiv preprint nor the journal version of one (141,438 of 1,918,958 acknowledging works removed; "
        "66,364 investigators linked). Placebo: 100 draws among non-application cycles.",
        [
            "Design",
            "Estimand",
            "Estimate (SE)",
            "Placebo 97.5\\%",
            "Placebo $P$",
            "Investigators",
            "Pre-deadline preprints",
        ],
        rows,
        "p{3.4cm}p{3.2cm}rrrrr",
    )
    ac = pd.read_csv(R / "pi_activity.csv")
    rows = []
    for _, x in ac.iterrows():
        if x["activity"].startswith("R21/R03 minus"):
            continue
        else:
            rows.append(
                [
                    x["activity"],
                    x["spec"].capitalize(),
                    es(x["log_rr"], x["se"]),
                    f(x["null_p975"]),
                    f(x["p_one_sided"]),
                    f"{int(x['pis']):,}",
                    f"{int(x['treated_pre_posts']):,}",
                ]
            )
    t6 = table(
        "tab:activity",
        "Funded new applications by activity code. R01 applications are reviewed on preliminary "
        "data; R21 and R03 applications do not require them. Investigators who hold both kinds of award keep the "
        "cycles of one kind in the reference of the other, which biases each estimate toward zero. Placebo: 100 "
        "draws among non-application cycles.",
        [
            "Activity",
            "Estimand",
            "Estimate (SE)",
            "Placebo 97.5\\%",
            "Placebo $P$",
            "Investigators",
            "Pre-deadline preprints",
        ],
        rows,
        "p{3.4cm}p{3.2cm}rrrrr",
    )
    cf = pd.read_csv(R / "conv_falsification.csv")
    rows = []
    for srv in ("biorxiv", "medrxiv"):
        g = cf[(cf["server"] == srv) & cf["shift_weeks"].notna()]
        for _, x in g.iterrows():
            ok = x["amp"] > 0
            rows.append(
                [
                    SRV[srv],
                    f"{int(x['shift_weeks']):+d}",
                    f(x["weighted_r2"]) if ok else "--",
                    "" if ok else "negative amplitude",
                ]
            )
        sx = cf[(cf["server"] == srv) & cf["shift_weeks"].isna()].iloc[0]
        rows.append([SRV[srv], "free-phase sinusoid", f(sx["weighted_r2"]), "three parameters"])
    t7 = table(
        "tab:convshift",
        "Moving the individual surges away from the due dates (falsification of the kernel fit). "
        "The investigator kernel is moved by the stated number of "
        "weeks (positive, earlier in the cycle) before each new-application due date and fitted at each shift with one "
        "amplitude; fits with a negative amplitude invert the model and are not admissible. The last row of each "
        "server is a sinusoid with the period of the cycle and a free phase.",
        ["Server", "Kernel shift (weeks)", "Weighted $R^2$", "Note"],
        rows,
        "llrl",
    )
    return "\n\n".join([t1, t2, t3, t4, t5, t6, t7])


def t_round5():
    d = pd.read_csv(R / "rd_phase.csv")
    rows = []
    for _, x in d[d["test"] == "discontinuity"].iterrows():
        rows.append(
            [
                SRV[x["server"]],
                x["dates"].capitalize(),
                int(x["bandwidth"]),
                es(x["drop"], x["se"]),
                f(x["rank_p"]) if pd.notna(x.get("rank_p")) else "--",
            ]
        )
    t1 = table(
        "tab:rd",
        "Discontinuity at the R01 due date. Poisson day counts of US and non-US first versions (day, "
        "group-by-cycle and group-by-weekday fixed effects, SE clustered by cycle) within the bandwidth of a due "
        "date, with a separate US linear trend on each side; the drop is the fall in the US log ratio at the due "
        "date. Business-day dates move a due date that falls on a weekend to the following Monday. Rank $P$ "
        "compares the drop with the same estimate at due dates shifted by 14 to 60 days (94 shifts).",
        ["Server", "Due dates", "Bandwidth (days)", "Drop (SE)", "Rank $P$"],
        rows,
        "llrrr",
    )
    rows = []
    for _, x in d[d["test"] == "weekend phase"].iterrows():
        rows.append(
            [
                SRV[x["server"]],
                es(x["pre_nominal"], x["pre_se"]),
                es(x["gap"], x["gap_se"]),
                es(x["post_actual"], x["post_se"]),
                int(x["shifted_dates"]),
                int(x["gap_days"]),
            ]
        )
    t2 = table(
        "tab:weekend",
        "When the due date falls on a weekend. US $\\times$ indicator coefficients on daily counts for "
        "the 7 days before the published date, the weekend days between the published date and the business day "
        "on which NIH accepts applications, and the 7 days from that business day on (same model as "
        "table~\\ref{tab:rd}). "
        "If posting follows the date NIH uses, the weekend days resemble the days before the deadline.",
        [
            "Server",
            "7 days before published date",
            "Weekend days before business day",
            "7 days from business day",
            "Shifted dates",
            "Weekend days",
        ],
        rows,
        "lrrrrr",
    )
    fm = pd.read_csv(R / "pi_family.csv")
    rows = [
        [
            {
                "investigator FE": "Investigator",
                "investigator x cycle-family FE": "Investigator $\\times$ cycle family",
            }[x["spec"]],
            es(x["log_rr"], x["se"]),
            es(100 * x["coef_share"], 100 * x["se_share"], 2),
            f(x["p_one_sided"]) if pd.notna(x["p_one_sided"]) else "--",
            f"{int(x['pis']):,}",
        ]
        for _, x in fm.iterrows()
    ]
    t3 = table(
        "tab:family",
        "Within-investigator estimate for funded new applications with investigator fixed effects and "
        "with investigator-by-cycle-family fixed effects (February, June and October schedules), which compare an "
        "application cycle only with the investigator's other cycles of the same annual family. Placebo: 100 draws "
        "among non-application cycles.",
        ["Fixed effects", "Log ratio (SE)", "Share, pp (SE)", "Placebo $P$", "Investigators"],
        rows,
        "p{5cm}rrrr",
    )
    return "\n\n".join([t1, t2, t3])


def t_falsify():
    """One table of every comparison that could count against the account, with its predicted sign and power."""
    rows = []

    def add(name, pred, b, se, note=""):
        rows.append([name, pred, es(b, se), f(2.8 * se), note])

    pre = pd.read_csv(R / "pre2017.csv")
    x = pre[
        pre["period"].str.startswith("pre") & pre["measure"].str.startswith("US share") & (pre["window"] == "R_new")
    ].iloc[0]
    add("bioRxiv before the 2017 citation rule, R01 window (pp)", "0 if the rule is necessary", x["coef"], x["se"])
    dc = pd.read_csv(R / "design_checks.csv")
    for srv in ("biorxiv", "medrxiv"):
        x = dc[
            (dc["check"] == "HIV joint model") & (dc["server"] == srv) & dc["what"].str.startswith("phase shift")
        ].iloc[0]
        add(f"{SRV[srv]} HIV researchers, AIDS minus R01 window", "positive", x["coef"], x["se"])
    cp = pd.read_csv(R / "counts_poisson.csv")
    for srv in ("biorxiv", "medrxiv"):
        x = cp[
            (cp["test"] == f"{srv} intramural")
            & (cp["model"] == "B relative, day FE")
            & (cp["group"] == "US-intramural")
            & (cp["term"] == "R_new")
        ].iloc[0]
        add(
            f"{SRV[srv]} NIH intramural minus extramural, R01 window",
            "negative",
            x["coef"],
            x["se"],
            "clearance delays smooth the peak",
        )
        x = cp[
            (cp["test"] == f"{srv} linked")
            & (cp["model"] == "B relative, day FE")
            & (cp["group"] == "US-linked-PI")
            & (cp["term"] == "R_new")
        ].iloc[0]
        add(
            f"{SRV[srv]} linked investigators minus other US last authors",
            "none (applicants in both)",
            x["coef"],
            x["se"],
            "most applicants are unlinked",
        )
    idp = pd.read_csv(R / "inst_dose.csv")
    for srv in ("biorxiv", "medrxiv"):
        x = idp[(idp["server"] == srv) & (idp["subset"] == "all") & (idp["window"] == "R_new")].iloc[0]
        add(
            f"{SRV[srv]} R01-window change in the NIH dependence of the posting institution (within field)",
            "positive",
            x["coef"],
            x["se"],
            "applicants at every institution",
        )
    r2 = pd.read_csv(R / "group_rates.csv")
    fu = r2[(r2["analysis"] == "funder group vs non-US (all fields)") & (r2["term"] == "R_new")].set_index("group")
    b = fu.loc["US NIH", "coef"] - fu.loc["US NSF not NIH", "coef"]
    se = float(np.hypot(fu.loc["US NIH", "se"], fu.loc["US NSF not NIH", "se"]))
    add(
        "bioRxiv US acknowledging NIH minus NSF but not NIH",
        "positive",
        b,
        se,
        "acknowledgments record past funding; SE approximate",
    )
    ac = pd.read_csv(R / "pi_activity.csv")
    a1 = ac[(ac["activity"] == "R01") & (ac["spec"] == "within investigator")].iloc[0]
    a2 = ac[(ac["activity"] == "R21/R03") & (ac["spec"] == "within investigator")].iloc[0]
    add(
        "Within investigator, R21/R03 minus R01",
        "negative",
        a2["log_rr"] - a1["log_rr"],
        float(np.hypot(a1["se"], a2["se"])),
        "shared references bias toward 0; SE approximate",
    )
    m = pd.read_csv(R / "pi_within_main.csv")
    rn = pd.read_csv(R / "pi_within_renewal.csv")

    def pick(d):
        return d[d["spec"].str.startswith("within") & (d["treatment"] == "treated")].iloc[0]

    x, y = pick(m), pick(rn)
    add(
        "Within investigator, renewals minus new applications",
        "negative",
        y["log_rr"] - x["log_rr"],
        float(np.hypot(x["se"], y["se"])),
        "SE approximate",
    )
    return table(
        "tab:falsify",
        "Comparisons that could count against the account. Prediction: the sign expected if the NIH "
        "calendar times the rhythm through applicants. MDE: minimum detectable effect at 80\\% power and a "
        "two-sided 5\\% level (2.8 SE). Estimates on the log scale unless stated.",
        ["Comparison", "Prediction", "Estimate (SE)", "MDE", "Note"],
        rows,
        "p{5.4cm}p{2.6cm}rrp{3.6cm}",
    )


def t_holiday():
    d = pd.read_csv(R / "holiday_check.csv")
    CY = {"pooled": "all", "Feb": "February"}
    rows = [
        [
            SRV[x["server"]],
            x["comparison"].replace("non-US", "Non-US").replace(" (reference)", ""),
            "removed" if x["holidays_removed"] else "kept",
            CY[x["cycles"]],
            es(x["step"], x["se"]),
            f(x["rank_p"]),
            es(x["rd14"], x["rd14_se"]),
        ]
        for _, x in d.iterrows()
    ]
    return table(
        "tab:holiday",
        "The drop at the R01 due date without East Asian authors in the comparison group (CN, HK, MO, "
        "TW, KR, SG, VN) and without the Lunar New Year holiday weeks and 1--7 October, which lower posting in East "
        "Asia near the February and October due dates. Country-coded first versions; drop and rank $P$ as in "
        "table~\\ref{tab:step}; discontinuity with a 14-day bandwidth as in table~\\ref{tab:rd}.",
        ["Server", "Comparison group", "Holidays", "Cycles", "Drop (SE)", "Rank $P$", "Discontinuity (SE)"],
        rows,
        "lp{3.4cm}llrrr",
    )


def t_citations():
    w = pd.read_csv(R / "citations_within.csv")
    lab = {
        ("cite_pct", "pre-deadline"): "Citation percentile, all pre-deadline preprints",
        ("top10", "pre-deadline"): "Top 10\\% of field and month (share)",
        (
            "cite_pct",
            "pre-deadline, high similarity to the award",
        ): "Citation percentile, similarity to the award above the median",
        (
            "cite_pct",
            "pre-deadline, low similarity to the award",
        ): "Citation percentile, similarity to the award at or below the median",
        ("cite_pct", "pre-deadline x high similarity to the award"): "Difference, high minus low similarity",
    }
    rows = [
        [lab[(x["outcome"], x["term"])], es(x["coef"], x["se"], 2), f"{int(x['n']):,}", f"{int(x['n_pulse']):,}"]
        for _, x in w.iterrows()
        if (x["outcome"], x["term"]) in lab
    ]
    pb = pd.read_csv(R / "citations_publication.csv")
    for _, x in pb.iterrows():
        nm = (
            "Published by June 2026 (share)"
            if x["outcome"].startswith("published")
            else f"Citation percentile, {x['sample']} preprints"
        )
        rows.append(
            [
                nm,
                es(x["coef"], x["se"], 3 if x["outcome"].startswith("published") else 2),
                f"{int(x['n']):,}",
                f"{int(x['n_pulse']):,}",
            ]
        )
    hy = pd.read_csv(R / "hype_lexicon.csv")
    for _, x in hy.iterrows():
        rows.append(
            [
                f"Promotional adjectives per 1,000 words, {x['lexicon']}",
                es(x["coef"], x["se"], 2),
                f"{int(x['n']):,}",
                f"{int(x['n_pulse']):,}",
            ]
        )
    a = pd.read_csv(R / "citations_aggregate.csv")
    for _, x in a[a["outcome"] == "cite_pct"].iterrows():
        rows.append(
            [
                f"{SRV[x['server']]}: US $\\times$ R01 window, all first versions",
                es(x["us_x_window"], x["se"], 2),
                f"{int(x['n']):,}",
                "--",
            ]
        )
        rows.append(
            [
                f"{SRV[x['server']]}: implied difference for the posts the deadline adds",
                es(x["implied_excess_diff"], x["implied_excess_se"], 2),
                "--",
                "--",
            ]
        )
    return table(
        "tab:citations",
        "Citations, publication and promotional language of pre-deadline preprints. Citation percentile: citations to "
        "a first version and "
        "its journal version in the calendar year of posting and the two following years, ranked within server, "
        "subject category and posting month (0 to 100; first versions posted through 2023). Upper rows: linked "
        "investigators' preprints posted 0--13 days before their own due date against the same investigator's "
        "other preprints (investigator and year-month fixed effects, SE clustered by investigator); similarity is "
        "split at the median similarity to the forthcoming award among all compared preprints; publication is a "
        "journal version linked by the "
        "server by "
        "June 2026; promotional adjectives use the 139-word list of Millar and colleagues, unchanged and without "
        "``significant''. Last rows: US $\\times$ R01 window on all "
        "country-coded first versions (category-by-month fixed effects, SE clustered by cycle), divided by the "
        "share of US window posts the deadline adds (0.09 bioRxiv, 0.11 medRxiv).",
        ["Comparison", "Estimate (SE)", "N", "Pre-deadline preprints"],
        rows,
        "p{7.5cm}rrr",
    )


def t_resub():
    ev = pd.read_csv(R / "resub_events.csv")
    rows = []
    bins = {-2: "28 to 15 days before", -1: "14 to 1 days before", 0: "0 to 13 days after", 1: "14 to 27 days after"}
    for lab, name in (("A0, new-application date", "First submissions"), ("A1, resubmission date", "Resubmissions")):
        e = ev[ev["design"] == lab].set_index("k")
        rows.append([name, f"{int(e['events'].iloc[0]):,}"] + [es(e.loc[k, "coef"], e.loc[k, "se"]) for k in bins])
    t1 = table(
        "tab:resub",
        "Linked investigators' preprint counts around their own due date: funded first submissions at the "
        "new-application date and funded resubmissions at the resubmission date one month later. Poisson model of "
        "counts in 14-day bins with event and calendar-bin fixed effects, SE clustered by investigator (as Fig.~2C).",
        ["Applications", "Events"] + list(bins.values()),
        rows,
        "lrrrrr",
    )
    a = pd.read_csv(R / "resub_aggregate.csv")
    a = a[a["cycles"] != "weekend test"]
    rows = [
        [
            SRV[x["server"]],
            x["anchor"].capitalize(),
            x["cycles"].capitalize(),
            int(x["bandwidth"]),
            es(x["drop"], x["se"]),
        ]
        for _, x in a.iterrows()
    ]
    k = pd.read_csv(R / "resub_kernel.csv")
    k = k[k["model"].str.contains("separate")]
    note = "; ".join(
        f"{SRV[x['server']]}: per funded resubmission, {f(x['a1_to_new'], 2)} of the amplitude per funded new "
        f"application, weighted $R^2$ {f(x['weighted_r2'], 3)}"
        for _, x in k.iterrows()
    )
    t2 = table(
        "tab:resubagg",
        "National drop at the R01 date, the resubmission date and a placebo date 45 days after the R01 date. Local "
        "linear discontinuity on US vs non-US daily counts with US-specific slopes on each side "
        "(as table~\\ref{tab:rd}); resubmission dates on the business day.",
        ["Server", "Date", "Cycles", "Bandwidth (days)", "Drop (SE)"],
        rows,
        "lp{4.2cm}lrr",
        note="Individual surges placed before new-application and resubmission due dates with separate amplitudes: "
        + note
        + ".",
    )
    return "\n\n".join([t1, t2])


def main():
    parts = [
        t_aggregate(),
        t_cycles(),
        t_fields(),
        t_2017(),
        t_designs(),
        t_decomp(),
        t_sim(),
        t_month(),
        t_totals(),
        t_content(),
        t_scale(),
        t_consequences(),
        t_checks(),
        t_other(),
        t_link(),
        t_rev2(),
        t_rev_pi(),
        t_round4(),
        t_round5(),
        t_falsify(),
        t_holiday(),
        t_citations(),
        t_resub(),
    ]
    blocks = [b for part in parts for b in part.split("\n\n") if "\\label{tab:" in b]

    def lab(b):
        return re.search(r"\\label\{(tab:[a-z0-9]+)\}", b).group(1)

    man = Path(OUT).parent
    man.mkdir(parents=True, exist_ok=True)
    if (man / "main.tex").exists():  # number tables by first citation in the manuscript; else keep generator order
        main_tex = (man / "main.tex").read_text()
        body = main_tex[main_tex.index("\\begin{abstract}") : main_tex.index("\\bibliography{")]
        text = body + (man / "supplement_body.tex").read_text() + "".join(blocks)
    else:
        text = "".join(blocks)
    first = {}
    for m in re.finditer(r"\\ref\{(tab:[a-z0-9]+)\}", text):
        first.setdefault(m.group(1), m.start())
    blocks.sort(key=lambda b: first.get(lab(b), 10**9))
    OUT.write_text("% generated by python -m pulse.si_tables; do not edit by hand\n" + "\n\n".join(blocks) + "\n")
    print("wrote", OUT, len(parts), "tables")


if __name__ == "__main__":
    main()
