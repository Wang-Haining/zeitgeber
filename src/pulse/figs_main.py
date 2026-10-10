"""Main and supplementary figures, written to manuscript/figures/ as PDF and PNG.

Colours, fixed across figures (Paul Tol muted palette):
  rose #CC6677        R01 window / the investigator's own due date
  light blue #88CCEE  F (fellowship) window
  heatmap             indigo #332288 -> white -> wine #882255
  black / grey        bioRxiv / medRxiv; dashed grey = null or reference

    python -m pulse.figs_main [fig1 fig2 fig3 fig4 figS1 figS2 figS3]
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from pulse.nature_style import apply_nature_style, save_nature_figure
from pulse.signals import RESULTS as R

OUT = Path(__file__).resolve().parents[2] / "manuscript" / "figures"
MM = 1 / 25.4
DEAD, FEL, INK, MID, LIGHT = "#CC6677", "#88CCEE", "#1a1a1a", "0.55", "0.82"
DEAD_T, FEL_T = "#F3DCE1", "#DDEFF8"  # tints for window shading
HEAT = LinearSegmentedColormap.from_list(
    "tol_div", ["#332288", "#8E88C4", "#F7F7F7", "#E3A9B5", "#882255"]
)  # Paul Tol muted: indigo -> white -> rose -> wine
CYCLE = 126.0  # 18 seven-day bins around the ring
DUE = {"R01": 0, "AIDS": 29, "F": 58, "Resub": 92}


def label(ax, s, x=None, y=None):
    """Defer: panel letters go to the top-left of each axes' tight bounding box (see place_labels)."""
    ax._panel_letter = s


def place_labels(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    items = []
    for ax in fig.axes:
        s = getattr(ax, "_panel_letter", None)
        if s:
            bb = inv.transform(ax.get_tightbbox(r))
            pos = ax.get_position()
            items.append(
                {"s": s, "x": max(bb[0][0] - 0.018, 0.003), "y": min(bb[1][1], 0.995), "top": pos.y1, "left": pos.x0}
            )

    def share(key, val, tol, pick):
        # panels whose axes start in the same row (column) get one letter height (position)
        for it in items:
            grp = [o[val] for o in items if abs(o[key] - it[key]) < tol]
            it[val + "_shared"] = pick(grp)

    share("top", "y", 0.08, max)
    share("left", "x", 0.05, min)
    for it in items:
        fig.text(it["x_shared"], it["y_shared"], it["s"], fontsize=10, fontweight="bold", ha="left", va="top")


def comply(fig):
    """Science figure rules: type at least 7 pt as drawn (6.3 pt at text width), plain type, lines at least 0.5 pt."""
    import matplotlib.colors as mc
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    colored = {mc.to_hex(DEAD), mc.to_hex(FEL)}
    keep = ("bioRxiv", "medRxiv", "med-", "n =", "x ")
    for t in fig.findobj(Text):
        s_ = t.get_text()
        if not s_:
            continue
        if s_[0].islower() and not s_.startswith(keep):
            t.set_text(s_[0].upper() + s_[1:])
        if "-0." in s_ or " -" in s_:
            t.set_text(t.get_text().replace("-0.", "\u22120.").replace(" -", " \u2212"))
        if t.get_fontsize() < 7:
            t.set_fontsize(7)
        if mc.to_hex(t.get_color()) in colored:
            t.set_color(INK)
    for ln in fig.findobj(Line2D):
        if 0 < ln.get_linewidth() < 0.5:
            ln.set_linewidth(0.5)
    for ax in fig.axes:
        for sp in ax.spines.values():
            sp.set_linewidth(max(sp.get_linewidth(), 0.5))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    comply(fig)
    place_labels(fig)
    save_nature_figure(fig, str(OUT / name))


def windows(ax, labels=True):
    ax.axvspan(0, 28, color=DEAD_T, lw=0, zorder=0)
    ax.axvspan(58, 86, color=FEL_T, lw=0, zorder=0)
    if labels:
        for name, lo, hi, col in (
            ("R01", 0, 28, DEAD),
            ("AIDS", 28, 56, MID),
            ("F", 58, 86, FEL),
            ("Resubmission", 86, 114, MID),
        ):
            ax.text(
                104 if name == "Resubmission" else (lo + hi) / 2,  # wide label shifted into the free margin
                1.0,
                name,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=5.5,
                color=col,
                fontweight="bold" if col != MID else "normal",
            )
    ax.axhline(0, color=MID, lw=0.5, ls="--", zorder=1)
    ax.set_xlim(124, -1)
    ax.set_xticks([0, 28, 56, 84, 112])


def ribbon(ax, x, y, se, col=INK, lw=1.1, alpha=0.15, z=3):
    x, y, se = map(np.asarray, (x, y, se))
    ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=col, alpha=alpha, lw=0, zorder=z)
    ax.plot(x, y, color=col, lw=lw, zorder=z + 1)


# ----------------------------------------------------------------------------------------------- Figure 1


def phase_clock(ax, t):
    """Polar plot: the four-month NIH cycle as a clock; R01 due date at 12 o'clock; time runs clockwise."""
    r0 = 0.32

    def th(phi):
        return 2 * np.pi * (CYCLE - np.asarray(phi, float)) / CYCLE

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    for lo, hi, col in ((0, 28, DEAD), (58, 86, FEL), (28, 56, LIGHT), (86, 114, LIGHT)):
        tt = np.linspace(th(hi), th(lo), 60)
        ax.fill_between(tt, 0.535, 0.56, color=col, lw=0)
    tt = np.linspace(0, 2 * np.pi, 200)
    ax.plot(tt, np.full_like(tt, r0), color=MID, lw=0.5, ls="--")
    for srv, col, lw, al in (("biorxiv", INK, 1.3, 0.13),):  # medRxiv is in panel B
        s = t[t["scan"] == f"{srv} aggregate"].sort_values("phi_mid", ascending=False)
        ang = np.r_[th(s["phi_mid"]), th(s["phi_mid"].iloc[0]) + 2 * np.pi]
        y = np.r_[s["coef"], s["coef"].iloc[0]]
        se = np.r_[s["se"], s["se"].iloc[0]]
        ax.fill_between(ang, r0 + y - 1.96 * se, r0 + y + 1.96 * se, color=col, alpha=al, lw=0)
        ax.plot(ang, r0 + y, color=col, lw=lw, label="bioRxiv" if srv == "biorxiv" else "medRxiv")
    ax.set_ylim(0, 0.56)
    ax.set_yticks([])
    ax.set_xticks([th(v) % (2 * np.pi) for v in DUE.values()])
    ax.set_xticklabels([("Resubmission\ndue" if k == "Resub" else f"{k} due") for k in DUE], fontsize=5.8)
    for tick, k in zip(ax.get_xticklabels(), DUE, strict=False):
        tick.set_color(DEAD if k == "R01" else FEL if k == "F" else MID)
        tick.set_fontweight("bold" if k in ("R01", "F") else "normal")
    ax.tick_params(axis="x", pad=1)
    ax.spines["polar"].set_visible(False)
    ax.grid(False)
    for dv in (-0.1, 0.1):
        ax.plot(tt, np.full_like(tt, r0 + dv), color=LIGHT, lw=0.4, ls=":")
        ax.text(th(44), r0 + dv, f"{dv:+.1f}", fontsize=4.6, color=MID, ha="center", va="center")
    ax.annotate(
        "more US\npreprints",
        xy=(th(3), r0 + 0.14),
        xytext=(th(116), 0.63),
        annotation_clip=False,
        fontsize=5.3,
        color=DEAD,
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="-", color=DEAD, lw=0.5),
    )
    ax.annotate(
        "fewer",
        xy=(th(73), r0 - 0.12),
        xytext=(th(73), 0.08),
        fontsize=5.3,
        color=FEL,
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="-", color=FEL, lw=0.5),
    )
    ax.text(0, 0, "", ha="center")


def heatmap(ax, server="biorxiv"):
    h = pd.read_csv(R / "year_heatmap.csv")
    h = h[(h["server"] == server) & h["year"].between(2017, 2025)]
    M = h.pivot(index="year", columns="bin", values="lr").sort_index()
    v = 0.3
    im = ax.imshow(
        M.to_numpy()[:, ::-1],
        aspect="auto",
        cmap=HEAT,
        vmin=-v,
        vmax=v,
        extent=[126, 0, M.index.max() + 0.5, M.index.min() - 0.5],
        interpolation="nearest",
    )
    for lo, hi, col in ((0, 28, DEAD), (58, 86, FEL)):
        ax.plot([hi, lo], [M.index.min() - 0.85] * 2, color=col, lw=2.5, clip_on=False, solid_capstyle="butt")
    ax.set_xlim(126, 0)
    ax.set_xticks([0, 28, 56, 84, 112])
    ax.set_yticks(M.index)
    ax.set_xlabel("Days before R01 due date")
    ax.set_ylabel("Year")
    ax.text(14, M.index.min() - 1.25, "R01", color=DEAD, fontsize=5.5, ha="center", fontweight="bold", va="bottom")
    ax.text(72, M.index.min() - 1.25, "F", color=FEL, fontsize=5.5, ha="center", fontweight="bold", va="bottom")
    cax = ax.inset_axes([0.45, -0.2, 0.55, 0.022])
    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("US vs non-US, log ratio (relative to year average)", fontsize=5, labelpad=1)
    cb.ax.tick_params(labelsize=4.5, length=1.5)
    cb.ax.tick_params(labelsize=5)
    cb.outline.set_linewidth(0.4)


# ----------------------------------------------------------------------------------------------- panels


def clock_panel(b):
    t = pd.read_csv(R / "phase_scan.csv")
    cp = pd.read_csv(R / "conv_pred.csv")
    phase_clock(b, t)
    p = cp[(cp["server"] == "biorxiv") & (cp["model"] == "new applications")].sort_values("phi_mid", ascending=False)

    def th(phi):
        return 2 * np.pi * (CYCLE - np.asarray(phi, float)) / CYCLE

    ang = np.r_[th(p["phi_mid"]), th(p["phi_mid"].iloc[0]) + 2 * np.pi]
    yy = np.r_[p["pred"], p["pred"].iloc[0]]
    b.plot(ang, 0.32 + yy, color=DEAD, lw=1.0, ls="--", label="Individual surges placed\nbefore every due date")


def daily_panel(c):
    dp = pd.read_csv(R / "daily_step_profile.csv")
    dp = dp[dp["server"] == "biorxiv"]
    c.axvspan(-7.5, -0.5, color=DEAD_T, lw=0, zorder=0)
    c.axhline(0, color=MID, lw=0.5, ls="--")
    ribbon(c, dp["rel_day"], dp["coef"], dp["se"], col=INK, lw=1.0)
    c.axvline(-0.5, color=DEAD, lw=1.0)
    c.annotate(
        "Due date",
        (-0.5, 0.345),
        xytext=(3, 0.345),
        fontsize=6.5,
        va="center",
        arrowprops=dict(arrowstyle="-", color=DEAD, lw=0.5),
    )
    for xx, lab, dx, ha in ((7, "Career", -0.4, "right"), (11, "R21, R03", 0.4, "left")):
        c.axvline(xx - 0.5, ymax=0.9, color=MID, lw=0.6, ls=(0, (3, 2)), zorder=1)
        c.text(xx - 0.5 + dx, 0.315, lab, fontsize=6, color=MID, ha=ha, va="center")
    c.text(-4, -0.12, "7 days\nbefore", fontsize=6, ha="center", va="bottom")
    c.set_xlim(-21.5, 20.5)
    c.set_ylim(-0.15, 0.37)
    c.set_xticks([-21, -14, -7, 0, 7, 14])
    c.set_xlabel("Days from R01 due date")
    c.set_ylabel("US vs non-US first versions\n(log ratio)")


def shift_panel(d):
    sc = pd.read_csv(R / "shift_scan.csv")
    for srv, col, lw_ in (("medrxiv", MID, 0.8), ("biorxiv", INK, 1.1)):
        g = sc[sc["server"] == srv].sort_values("shift")
        if srv == "biorxiv":
            d.fill_between(
                g["shift"], g["step"] - 1.96 * g["se"], g["step"] + 1.96 * g["se"], color=INK, alpha=0.1, lw=0
            )
        d.plot(g["shift"], g["step"], color=col, lw=lw_, label="bioRxiv" if srv == "biorxiv" else "medRxiv")
    d.axhline(0, color=MID, lw=0.5, ls="--")
    g0 = sc[(sc["server"] == "biorxiv") & (sc["shift"] == 0)].iloc[0]
    d.plot(0, g0["step"], "o", color=DEAD, ms=5, zorder=6)
    d.axvline(0, ymax=0.92, color=DEAD, lw=0.8, ls=(0, (3, 2)), zorder=1)
    d.text(0, 0.25, "R01", fontsize=6, color=INK, ha="center", va="bottom", fontweight="bold")
    for x0, lab in ((-58, "F"), (-29, "AIDS"), (28, "Resubmission")):
        d.axvline(x0, ymax=0.92, color=MID, lw=0.6, ls=(0, (3, 2)), zorder=1)
        d.text(x0, 0.25, lab, fontsize=6, color=MID, ha="center", va="bottom")
    d.set_xlim(-61, 61)
    d.set_ylim(-0.23, 0.29)
    d.set_xticks([-60, -30, 0, 30, 60])
    d.set_xlabel("Assumed due date, days from true R01 due date")
    d.set_ylabel("Drop at assumed due date\n(7 days before minus after)")
    d.legend(loc="lower left", fontsize=6, handlelength=1.4)


def season_panel(d):
    sa = pd.read_csv(R / "season_field_counts.csv")
    sa = sa[sa["category"] == "all fields"]
    for wnd, col, lab in (("R_new", DEAD, "R01 window"), ("F", FEL, "F window")):
        r = [sa[(sa["window"] == wnd) & (sa["month_fe"] == m)].iloc[0] for m in (False, True)]
        y = [x["coef"] for x in r]
        e = [1.96 * x["se"] for x in r]
        d.plot([0, 1], y, color=col, lw=1.2, zorder=2)
        d.errorbar([0, 1], y, yerr=e, fmt="none", ecolor=col, elinewidth=0.9, capsize=0, zorder=3)
        d.plot(0, y[0], "o", color=col, ms=5, zorder=4)
        d.plot(1, y[1], "o", color=col, mfc="white", mew=1.2, ms=5, zorder=4)
        d.text(1.08, y[1], lab, fontsize=6.5, va="center", ha="left")
    d.axhline(0, color=MID, lw=0.5, ls="--")
    d.set_xticks([0, 1], ["Adjusted for cycle\nand weekday", "+ calendar\nmonth"])
    d.set_xlim(-0.3, 1.75)
    d.set_ylim(-0.12, 0.13)
    d.set_ylabel("US vs non-US, bioRxiv\n(log ratio)")
    d.set_title("Season explains the F dip, not the R01 rise", fontsize=6.5, loc="left", pad=2)


def field_panel(b):
    fs = pd.read_csv(R / "field_split_categories.csv")
    size = 5 + 55 * fs["n"] / fs["n"].max()
    w = 1 / fs["se_pp"] ** 2
    X = np.column_stack([np.ones(len(fs)), fs["nih_share_us_funded"]])
    A = np.linalg.inv(X.T @ (X * w.values[:, None]))
    bb = A @ X.T @ (w.values * fs["pulse_R_new_pp"].values)
    xs = np.linspace(0, 0.9, 50)
    Xs = np.column_stack([np.ones(len(xs)), xs])
    resid = fs["pulse_R_new_pp"].values - X @ bb
    s2 = (w.values * resid**2).sum() / (len(fs) - 2)
    sef = np.sqrt(np.einsum("ij,jk,ik->i", Xs, A * s2, Xs))
    b.fill_between(xs, Xs @ bb - 1.96 * sef, Xs @ bb + 1.96 * sef, color=DEAD, alpha=0.12, lw=0)
    b.plot(xs, Xs @ bb, color=DEAD, lw=1.1, zorder=2, label="weighted fit across fields")
    mean_rise = np.average(fs["pulse_R_new_pp"], weights=w)
    b.axhline(mean_rise, color=MID, lw=0.8, ls="--", label="same rise in every field")
    b.errorbar(
        fs["nih_share_us_funded"],
        fs["pulse_R_new_pp"],
        yerr=1.96 * fs["se_pp"],
        fmt="none",
        ecolor=LIGHT,
        elinewidth=0.5,
        zorder=1,
    )
    b.scatter(
        fs["nih_share_us_funded"], fs["pulse_R_new_pp"], s=size, color=DEAD, alpha=0.8, lw=0.4, ec="white", zorder=3
    )
    b.axhline(0, color=MID, lw=0.4)
    offs = {
        "ecology": (-14, 6),
        "plant biology": (-40, -11),
        "animal behavior and cognition": (-60, -2),
        "genetics": (-30, 6),
        "neuroscience": (-50, -26),
    }
    for name, (dx, dy) in offs.items():
        r = fs[fs["category"] == name]
        b.annotate(
            name.replace(" and cognition", "").capitalize(),
            (r["nih_share_us_funded"].iloc[0], r["pulse_R_new_pp"].iloc[0]),
            fontsize=5,
            xytext=(dx, dy),
            textcoords="offset points",
            color=INK,
            arrowprops=dict(arrowstyle="-", color=MID, lw=0.5) if name == "neuroscience" else None,
        )
    b.set_xlim(0, 0.92)
    b.set_ylim(-2.5, 7.5)
    b.set_xlabel("Field NIH dependence")
    b.set_ylabel("R01-window rise in US share\n(percentage points)")
    b.legend(loc="upper left", fontsize=4.9, handlelength=1.6)
    b.set_title("21 bioRxiv fields (size = preprints)", fontsize=6.5, loc="left", pad=2)


def field_week_panel(c):
    ip = pd.read_csv(R / "field_intercept_profile.csv")
    windows(c)
    ribbon(c, ip["phi_mid"], ip["intercept"], ip["intercept_se"], col=MID, lw=0.9, alpha=0.12)
    ribbon(c, ip["phi_mid"], ip["slope"], ip["slope_se"], col=INK)
    c.plot(3.5, ip[ip["bin"] == 0]["slope"].iloc[0], "o", color=DEAD, ms=3.5, zorder=6)
    c.plot([], [], color=INK, label="slope (rise per unit dependence)")
    c.plot([], [], color=MID, label="intercept (no dependence)")
    c.legend(loc="lower left", fontsize=4.9, handlelength=1.2)
    c.set_xlabel("Days before R01 due date")
    c.set_ylabel("Rise per unit of NIH\ndependence (log ratio)")


def kernel_shift_panel(a):
    cf = pd.read_csv(R / "conv_falsification.csv")
    for srv, col, mk in (("biorxiv", INK, "o"), ("medrxiv", MID, "s")):
        g = cf[(cf["server"] == srv) & cf["shift_weeks"].notna()].sort_values("shift_weeks")
        r2 = np.where(g["amp"] > 0, g["weighted_r2"], np.nan)
        a.plot(
            g["shift_weeks"],
            r2,
            mk + "-",
            color=col,
            ms=2.8,
            lw=0.8,
            mfc=col if srv == "biorxiv" else "white",
            label="bioRxiv" if srv == "biorxiv" else "medRxiv",
        )
    a.axvline(0, color=DEAD, lw=0.6)
    a.text(0.2, 0.05, "true\ndue dates", fontsize=6, color=MID, ha="left")
    a.set_xlabel("Surges moved from the due dates\n(weeks; positive = earlier)")
    a.set_ylabel("Fit to the national profile\n(weighted $R^2$)")
    a.set_ylim(0, 0.85)
    a.set_xticks([-2, 0, 2, 4])
    a.legend(loc="upper left", fontsize=5.5)


def content_forest(b, keep):
    c = pd.read_csv(R / "pulse_content.csv").set_index("outcome")
    cw = pd.read_csv(R / "citations_within.csv")
    cw = cw[(cw["outcome"] == "cite_pct") & (cw["term"] == "pre-deadline")].iloc[0]
    c.loc["citation percentile"] = {"coef": cw["coef"], "se": cw["se"], "mean_nonpulse": cw["mean_other"]}
    for j, (k, _lab, col) in enumerate(keep):
        r = c.loc[k]
        islog = k.startswith("log")
        rel = 100 * (r["coef"] if islog else r["coef"] / abs(r["mean_nonpulse"]))
        rse = 100 * (r["se"] if islog else r["se"] / abs(r["mean_nonpulse"]))
        b.errorbar(rel, j, xerr=1.96 * rse, fmt="o", color=col, ms=3.8 if col != MID else 3, elinewidth=0.8, capsize=0)
    b.axvline(0, color=MID, lw=0.5, ls="--")
    b.set_yticks(range(len(keep)), [x[1] for x in keep])
    b.set_ylim(len(keep) - 0.5, -0.5)
    b.set_xlabel("Pre-deadline vs the same investigator's\nother preprints (%)")


def schematic_within(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0.6, 3.0)
    ax.axis("off")
    rng = np.random.default_rng(3)
    for k in range(4):
        x0 = 0.25 + 2.45 * k
        own = k == 2
        ax.add_patch(mpatches.Rectangle((x0 + 1.5, 1.15), 0.55, 0.85, color=DEAD_T if own else "0.92", lw=0))
        ax.plot([x0, x0 + 2.05], [1.15, 1.15], color=INK, lw=0.8)
        ax.plot([x0 + 2.05] * 2, [1.0, 2.15], color=DEAD if own else MID, lw=1.2 if own else 0.6)
        xs = np.sort(rng.uniform(x0 + 0.1, x0 + 1.95, 4))
        if own:
            xs = np.r_[xs[:2], [x0 + 1.62, x0 + 1.78, x0 + 1.94]]
        for xx in xs:
            ax.plot(xx, 1.45, marker="o", ms=3.6, color=DEAD if (own and xx > x0 + 1.5) else INK)
        ax.text(
            x0 + 2.05,
            2.25,
            "own application\ndue date" if own else "due date",
            ha="center",
            fontsize=7,
            color=DEAD if own else MID,
            fontweight="bold" if own else "normal",
            va="bottom",
        )
        ax.text(x0 + 1.0, 0.85, f"cycle {k + 1}", ha="center", fontsize=7, color=MID)
    ax.text(0.6, 3.55, "Within-investigator design", fontsize=8, va="top")


# ----------------------------------------------------------------------------------------------- Figure 1


def fig1():
    fig = plt.figure(figsize=(183 * MM, 140 * MM))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1, 1],
        height_ratios=[1.15, 1],
        left=0.09,
        right=0.985,
        bottom=0.09,
        top=0.94,
        wspace=0.3,
        hspace=0.55,
    )
    a = fig.add_subplot(gs[0, 0])
    heatmap(a)
    a.set_title("bioRxiv, each year", fontsize=7, loc="left", pad=16)
    label(a, "A")
    b = fig.add_subplot(gs[0, 1])
    daily_panel(b)
    label(b, "B")
    c = fig.add_subplot(gs[1, 0])
    shift_panel(c)
    label(c, "C")
    d = fig.add_subplot(gs[1, 1])
    season_panel(d)
    label(d, "D")
    save(fig, "fig1")


# ----------------------------------------------------------------------------------------------- Figure 2


def resub_panels(top, bot):
    """First submissions and resubmissions, each around its own due date, on one calendar axis (shared x)."""
    ev = pd.read_csv(R / "resub_events.csv")
    gap = 30  # resubmission due dates fall 28-31 days after new-application due dates
    for ax, lab, off, name in (
        (top, "A0, new-application date", 0, "First submissions"),
        (bot, "A1, resubmission date", gap, "Resubmissions"),
    ):
        e = ev[(ev["design"] == lab) & ev["k"].between(-4, 3)].sort_values("k")
        x = e["k"] * 14 + 7 + off
        ax.axvspan(off - 14, off, color=DEAD_T, lw=0, zorder=0)
        ax.axvline(off, color=DEAD, lw=0.8)
        ax.axhline(0, color=MID, lw=0.5, ls="--")
        ribbon(ax, x, e["coef"], e["se"], col=INK, lw=1.0)
        pk = e[e["k"] == -1].iloc[0]
        ax.plot(off - 7, pk["coef"], "o", color=DEAD, ms=3.8, zorder=6)
        ax.text(0.02, 0.92, name, transform=ax.transAxes, fontsize=6.5, va="top", fontweight="bold")
        ax.set_xlim(-63, 81)
        ax.set_ylim(-0.42, 0.62)
        ax.set_yticks([-0.2, 0, 0.2, 0.4])
    top.text(1, 0.6, "new-application\ndue date", fontsize=5.5, color=MID, ha="left", va="top")
    bot.text(gap + 1, 0.6, "resubmission due date,\none month later", fontsize=5.5, color=MID, ha="left", va="top")
    top.tick_params(labelbottom=False)
    bot.set_xticks([-56, -28, 0, 28, 56])
    bot.set_xlabel("Days from the new-application due date")
    top.text(
        -0.16,
        -0.05,
        "Preprints per 14 days (log ratio)",
        transform=top.transAxes,
        rotation=90,
        ha="center",
        va="center",
    )


def fig2():
    fig = plt.figure(figsize=(183 * MM, 96 * MM))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.3, 1.15],
        width_ratios=[1, 1.15],
        left=0.105,
        right=0.985,
        bottom=0.12,
        top=0.95,
        wspace=0.32,
        hspace=0.22,
    )
    a = fig.add_subplot(gs[0, :])
    schematic_within(a)
    label(a, "A")

    p = pd.read_csv(R / "pi_phase_profile_2w.csv")
    b = fig.add_subplot(gs[1, 0])
    b.axvspan(0, 14, color=DEAD_T, lw=0, zorder=0)
    x, y, se = 14 * p["bin"] + 7, 100 * p["diff"], 100 * p["se"]
    b.fill_between(x, 100 * p["placebo_lo"], 100 * p["placebo_hi"], color=LIGHT, lw=0, alpha=0.6, zorder=1)
    b.axhline(0, color=MID, lw=0.5, ls="--", zorder=2)
    last = p["bin"] == 0
    for m, col in ((~last, INK), (last, DEAD)):
        b.errorbar(x[m], y[m], yerr=1.96 * se[m], fmt="o", color=col, ms=3.4, lw=0.8, capsize=0, zorder=5)
    b.plot(x, y, color=INK, lw=0.6, zorder=4)
    b.set_xlim(126, -2)
    b.set_xticks([0, 28, 56, 84, 112])
    b.set_xlabel("Days before the investigator's own due date")
    b.set_ylabel("Application cycle minus other cycles\n(percentage points per two weeks)")
    b.text(
        118, 100 * p["placebo_hi"].max() + 0.15, "chance range (placebo cycles)", fontsize=5.3, color=MID, va="bottom"
    )
    b.annotate(
        "final two weeks",
        xy=(7, y[last].iloc[0]),
        xytext=(52, y[last].iloc[0] + 0.3),
        fontsize=5.3,
        color=DEAD,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=DEAD, lw=0.5),
    )
    b.text(0.02, 0.03, f"{int(p['units'].iloc[0]):,} investigators", transform=b.transAxes, fontsize=5.3, color=MID)
    label(b, "B")

    sub = gs[1, 1].subgridspec(2, 1, hspace=0.12)
    c1 = fig.add_subplot(sub[0])
    c2 = fig.add_subplot(sub[1], sharex=c1)
    resub_panels(c1, c2)
    label(c1, "C")
    save(fig, "fig2")


# ----------------------------------------------------------------------------------------------- Figure 3


def fig3():
    fig = plt.figure(figsize=(183 * MM, 70 * MM))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.15, 0.85, 1.05], left=0.05, right=0.99, bottom=0.22, top=0.86, wspace=0.58
    )
    a = fig.add_subplot(gs[0], projection="polar")
    clock_panel(a)
    label(a, "A")
    b = fig.add_subplot(gs[1])
    kernel_shift_panel(b)
    label(b, "B")
    c = fig.add_subplot(gs[2])
    field_panel(c)
    label(c, "C")
    save(fig, "fig3")


# ----------------------------------------------------------------------------------------------- Figure 4


def fig4():
    fig = plt.figure(figsize=(183 * MM, 58 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.1], left=0.09, right=0.99, bottom=0.22, top=0.87, wspace=0.75)
    g = pd.read_csv(R / "pulse_sim_by_gap_fe.csv")
    order = ["181-365", "91-180", "46-90", "28-45", "14-27", "0-13"]
    g = g.set_index("gap").loc[order].reset_index()
    a = fig.add_subplot(gs[0])
    xs = np.arange(len(g))
    a.axvspan(4.5, 5.5, color=DEAD_T, lw=0, zorder=0)
    a.axhline(0, color=MID, lw=0.5, ls="--")
    a.plot(xs, g["coef"], color=INK, lw=1.0, zorder=3)
    for i, r in g.iterrows():
        col = DEAD if r["gap"] == "0-13" else INK
        a.errorbar(
            i,
            r["coef"],
            yerr=1.96 * r["se"],
            fmt="o",
            color=col,
            ms=3.8 if col == DEAD else 2.8,
            elinewidth=0.8,
            capsize=0,
            zorder=4,
        )
    a.set_xticks(xs, g["gap"])
    a.set_xlabel("Days before the investigator's due date")
    a.set_ylabel("Similarity to the award\n(vs 6–12 months before)")
    a.annotate(
        "only the final two weeks\nresemble the application",
        xy=(4.9, 0.039),
        xytext=(0.6, 0.044),
        fontsize=5.3,
        color=DEAD,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=DEAD, lw=0.5),
    )
    a.set_ylim(-0.012, 0.058)
    a.set_title("What is released", fontsize=6.5, loc="left", pad=2)
    label(a, "A")
    b = fig.add_subplot(gs[1])
    content_forest(
        b,
        [
            ("similarity to the upcoming award", "Similarity to the upcoming award", DEAD),
            ("similarity to PI's other awards", "Similarity to the investigator's\nother awards", MID),
            ("hype per 1000 words", "Promotional adjectives", INK),
            ("citation percentile", "Citation percentile in field", MID),
        ],
    )
    b.set_title("Content, language and citations", fontsize=6.5, loc="left", pad=2)
    label(b, "B")
    save(fig, "fig4")


def figS2():
    pre = pd.read_csv(R / "pre2017.csv")
    pf_ = pd.read_csv(R / "pre2017_byfield.csv")
    fig, d = plt.subplots(figsize=(89 * MM, 66 * MM))
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.15, top=0.92)
    vals = {}
    for per in ("pre", "post"):
        for wnd in ("R_new", "F"):
            vals[("all", per, wnd)] = (
                pre[
                    pre["period"].str.startswith(per)
                    & pre["measure"].str.startswith("US share")
                    & (pre["window"] == wnd)
                ]
                .iloc[0][["coef", "se"]]
                .to_numpy(float)
            )
            vals[("high", per, wnd)] = (
                pf_[(pf_["fields"] == "high NIH dependence (>= 0.7)") & (pf_["period"] == per) & (pf_["window"] == wnd)]
                .iloc[0][["coef", "se"]]
                .to_numpy(float)
            )
    for wnd, col in (("R_new", DEAD), ("F", FEL)):
        for grp, ls, off, mfc in (("all", "-", -0.07, col), ("high", ":", 0.07, "white")):
            yv = [vals[(grp, "pre", wnd)][0], vals[(grp, "post", wnd)][0]]
            e = [vals[(grp, "pre", wnd)][1], vals[(grp, "post", wnd)][1]]
            x = np.array([0, 1]) + off
            d.plot(x, yv, color=col, lw=1, ls=ls)
            d.errorbar(x, yv, yerr=1.96 * np.array(e), fmt="o", color=col, mfc=mfc, ms=3, elinewidth=0.6, capsize=0)
    d.axhline(0, color=MID, lw=0.5, ls="--")
    d.set_xticks([0, 1], ["Nov 2013 - Mar 2017", "Apr 2017 - 2019"])
    d.set_xlim(-0.4, 1.4)
    d.set_ylim(-6, 5)
    d.set_xlim(-0.4, 1.65)
    d.set_ylabel("Change in US share, bioRxiv\n(percentage points)")
    d.text(1.15, vals[("high", "post", "R_new")][0], "R01 window", fontsize=6.5, va="center")
    d.text(1.15, vals[("high", "post", "F")][0], "F window", fontsize=6.5, va="center")
    d.text(-0.38, -5.8, "Filled, all fields; open, NIH dependence of at least 0.7", fontsize=6, color=MID, va="bottom")
    save(fig, "figS2")


# ----------------------------------------------------------------------------------------------- Figure S1
def figS1():
    t = pd.read_csv(R / "phase_scan.csv")
    sl = pd.read_csv(R / "field_counts_slope.csv")
    fc = pd.read_csv(R / "field_counts.csv")
    fig = plt.figure(figsize=(183 * MM, 118 * MM))
    gs = fig.add_gridspec(2, 6, left=0.095, right=0.985, bottom=0.09, top=0.93, wspace=1.6, hspace=0.6)
    ax = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 0:3]),
        fig.add_subplot(gs[1, 3:6]),
    ]
    for srv, col, lab in (("medrxiv", MID, "medRxiv"), ("biorxiv", INK, "bioRxiv")):
        s = t[t["scan"] == f"{srv} aggregate"]
        ribbon(ax[0], s["phi_mid"], s["coef"], s["se"], col=col)
        ax[0].plot([], [], color=col, label=lab)
    windows(ax[0])
    ax[0].set_ylabel("US vs non-US (log ratio)")
    ax[0].legend(loc="lower left", fontsize=6)
    for i, srv in ((1, "biorxiv"), (2, "medrxiv")):
        s = t[t["scan"] == f"{srv} hiv"]
        h, u = s[s["term"] == "HIV"], s[s["term"] == "US_x_HIV"]
        windows(ax[i])
        ax[i].axvspan(28, 56, color="0.88", lw=0, zorder=0)
        ax[i].plot(h["phi_mid"], h["coef"], color=MID, lw=0.8, zorder=3)
        ribbon(ax[i], u["phi_mid"], u["coef"], u["se"], col=INK)
        ax[i].set_ylabel("Log ratio")
        ax[i].plot([], [], color=MID, label="HIV vs other, non-US")
        ax[i].plot([], [], color=INK, label="Extra for US HIV researchers")
        ax[i].legend(loc="lower left", fontsize=6)
        ax[i].set_title("bioRxiv" if srv == "biorxiv" else "medRxiv", fontsize=7, loc="left", pad=12)
    for srv, col, lab in (("medrxiv", MID, "medRxiv"), ("biorxiv", INK, "bioRxiv")):
        s = t[t["scan"] == f"{srv} intramural"]
        ribbon(ax[3], s["phi_mid"], s["coef"], s["se"], col=col)
        ax[3].plot([], [], color=col, label=lab)
    windows(ax[3])
    ax[3].set_ylabel("Intramural vs extramural\n(log ratio)")
    ax[3].legend(loc="lower left", fontsize=6)
    f = fc[fc["window"] == "R_new"]
    row = sl[sl["what"] == "R_new window coef on dependence"].iloc[0]
    ax[4].errorbar(
        f["dep"], f["coef"], yerr=1.96 * f["se"], fmt="o", color=DEAD, ecolor=LIGHT, ms=3.5, elinewidth=0.6, capsize=0
    )
    xs = np.linspace(0, 0.9, 10)
    ax[4].plot(xs, row["intercept"] + row["slope"] * xs, color=DEAD, lw=1)
    ax[4].axhline(0, color=MID, lw=0.5, ls="--")
    ax[4].set_xlabel("Field NIH dependence")
    ax[4].set_ylabel("US vs non-US in R01 window\n(log ratio, daily counts)")
    for a in ax[:4]:
        a.set_xlabel("Days before R01 due date")
    for a, l in zip(ax, "ABCDE", strict=False):
        label(a, l)
    save(fig, "figS1")


def figS3():
    fig, axs = plt.subplots(1, 2, figsize=(160 * MM, 58 * MM), gridspec_kw={"width_ratios": [1, 1.1]})
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.22, top=0.9, wspace=0.95)
    field_week_panel(axs[0])
    label(axs[0], "A")
    content_forest(
        axs[1],
        [
            ("similarity to the upcoming award", "Similarity to the upcoming award", DEAD),
            ("similarity to PI's other awards", "Similarity to the investigator's\nother awards", MID),
            ("hype per 1000 words", "Promotional adjectives", INK),
            ("versions posted", "Versions posted", MID),
            ("in a journal within 24 months", "Published within 24 months", MID),
            ("log days to journal (published only)", "Days to journal", MID),
            ("citation percentile", "Citation percentile in field", MID),
        ],
    )
    axs[1].set_title("All content measures", fontsize=6.5, loc="left", pad=2)
    label(axs[1], "B")
    save(fig, "figS3")


if __name__ == "__main__":
    import sys

    import matplotlib as mpl

    apply_nature_style()
    # figures are 183 mm wide and placed at the 165-mm text width, so type is set 1 pt above Science's 6-pt floor
    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )
    mpl.rcParams["figure.constrained_layout.use"] = False
    mpl.rcParams["savefig.bbox"] = "standard"
    for f in (fig1, fig2, fig3, fig4, figS1, figS2, figS3):
        if len(sys.argv) > 1 and f.__name__ not in sys.argv[1:]:
            continue
        f()
