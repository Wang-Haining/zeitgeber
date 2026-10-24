"""Matplotlib settings and export helper for journal figures (sans-serif type, 0.5 pt axes, TrueType fonts)."""

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Wong / Okabe-Ito palette  (Nature Methods 8, 441, 2011)
# The gold-standard colorblind-safe categorical palette, explicitly
# recommended by Nature Portfolio for all scientific figures.
# ---------------------------------------------------------------------------
WONG = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

# Convenient two-group subset (blue vs vermillion) -- most common in clinical

# Convenient three-group subset

# ---------------------------------------------------------------------------
# Figure width helpers (mm -> inches)
# ---------------------------------------------------------------------------
_WIDTHS_MM = {
    "single": 89,
    "one-half": 120,
    "1.5": 120,
    "double": 183,
    "full": 183,
}
_MAX_HEIGHT_MM = 247
_MM_TO_IN = 1 / 25.4


# ---------------------------------------------------------------------------
# Apply Nature style to matplotlib rcParams
# ---------------------------------------------------------------------------
def apply_nature_style(palette="wong"):
    """Set all rcParams to Nature Portfolio defaults.

    Call this once at the top of your script before creating any figures.
    """
    # Font embedding: TrueType (type 42) so text stays editable in
    # Illustrator and Nature's production pipeline can modify it.
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    # Font: sans-serif, Arial preferred. Nature rejects serif fonts in figures.
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    mpl.rcParams["font.size"] = 7  # Nature max for body text is 7 pt
    mpl.rcParams["axes.labelsize"] = 7
    mpl.rcParams["axes.titlesize"] = 7
    mpl.rcParams["xtick.labelsize"] = 6
    mpl.rcParams["ytick.labelsize"] = 6
    mpl.rcParams["legend.fontsize"] = 6
    mpl.rcParams["legend.title_fontsize"] = 7

    # Axes and ticks
    mpl.rcParams["axes.linewidth"] = 0.5
    mpl.rcParams["xtick.major.width"] = 0.5
    mpl.rcParams["ytick.major.width"] = 0.5
    mpl.rcParams["xtick.major.size"] = 3
    mpl.rcParams["ytick.major.size"] = 3
    mpl.rcParams["xtick.minor.width"] = 0.3
    mpl.rcParams["ytick.minor.width"] = 0.3
    mpl.rcParams["xtick.minor.size"] = 1.5
    mpl.rcParams["ytick.minor.size"] = 1.5
    mpl.rcParams["xtick.direction"] = "out"
    mpl.rcParams["ytick.direction"] = "out"

    # Lines
    mpl.rcParams["lines.linewidth"] = 1.0
    mpl.rcParams["lines.markersize"] = 4

    # Legend: Nature says no frames, no colored text
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["legend.borderpad"] = 0.3

    # Grid: Nature explicitly prohibits background gridlines
    mpl.rcParams["axes.grid"] = False

    # Spines: left + bottom only (cleaner, widely used in Nature papers)
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False

    # Figure defaults
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["savefig.facecolor"] = "white"
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.pad_inches"] = 0.02
    mpl.rcParams["figure.constrained_layout.use"] = True

    # Color cycle
    if palette == "wong":
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=WONG)
    elif palette == "tableau-colorblind10":
        plt.style.use("tableau-colorblind10")


# ---------------------------------------------------------------------------
# Panel labels
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Significance brackets (for bar charts)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------
def save_nature_figure(fig, filename, formats=("pdf", "png"), dpi=300, ghost_text=False):
    """Save a figure in Nature-compliant formats.

    By default exports both PDF (vector, for Nature submission) and PNG
    (raster, for Word embedding / co-author sharing / preprints).

    Parameters
    ----------
    fig : matplotlib Figure
    filename : str
        Base filename without extension.
    formats : tuple of str
        Any of 'pdf', 'eps', 'tiff', 'png', 'svg'.
        Default is ('pdf', 'png').
    dpi : int
        Resolution for raster formats (default 300).
    ghost_text : bool
        If True, also save a "*_ghost" variant where all text is set to
        white (invisible on white background) while preserving identical
        layout/spacing. Useful for post-hoc editing in PowerPoint or
        Illustrator: overlay your own text knowing the bounding boxes
        match the labeled version exactly.
    """
    fig.align_labels()

    # --- Standard export (labeled) ---
    for fmt in formats:
        path = f"{filename}.{fmt}"
        kwargs = {"format": fmt, "dpi": dpi}
        if fmt == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        print(f"Saved: {path}")

    # --- Ghost-text export (invisible labels, same spacing) ---
    if ghost_text:
        _save_ghost_variant(fig, filename, formats, dpi)


def _collect_texts(fig):
    """Recursively collect all Text objects in a figure."""
    texts = []
    texts.extend(fig.texts)  # suptitle, fig-level text
    for ax in fig.get_axes():
        texts.append(ax.title)  # ax.set_title()
        texts.append(ax.xaxis.label)  # xlabel
        texts.append(ax.yaxis.label)  # ylabel
        texts.extend(ax.get_xticklabels())  # xtick labels
        texts.extend(ax.get_yticklabels())  # ytick labels
        texts.extend(ax.texts)  # ax.text() annotations
        leg = ax.get_legend()
        if leg is not None:
            texts.extend(leg.get_texts())  # legend entries
            title = leg.get_title()
            if title:
                texts.append(title)
    return texts


def _save_ghost_variant(fig, filename, formats, dpi):
    """Save a version with all text set to white (invisible).

    The text objects remain in place with the same font and size so the
    layout (axis tick spacing, label bounding boxes, legend width) is
    identical. This makes it safe to overlay custom text in PowerPoint
    or Illustrator without the figure shifting.
    """
    texts = _collect_texts(fig)

    # Store originals
    originals = [(t, t.get_color()) for t in texts if t.get_text()]

    # Set all text to white
    for t, _ in originals:
        t.set_color("white")

    for fmt in formats:
        path = f"{filename}_ghost.{fmt}"
        kwargs = {"format": fmt, "dpi": dpi}
        if fmt == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        print(f"Saved (ghost): {path}")

    # Restore originals so the figure object stays usable
    for t, original_color in originals:
        t.set_color(original_color)


# ---------------------------------------------------------------------------
# Seaborn integration
# ---------------------------------------------------------------------------
