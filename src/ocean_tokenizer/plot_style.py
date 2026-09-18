"""One matplotlib style for every figure the audit and the training runs write.

Values are the validated reference palette (categorical slots in fixed order,
never cycled; hairline solid grid; 2 px lines; >= 8 px markers with a surface
ring).  Figures are paired with the table of the same numbers in the report,
which is what licenses the two low-contrast slots (aqua, yellow).
"""
from __future__ import annotations

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
#: fixed slot order — assign by series identity, never cycle
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948")
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a",
          "critical": "#d03b3b"}


def use_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "text.color": INK, "axes.labelcolor": INK_2,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
        "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "lines.markersize": 5,          # r >= 4 px
        "legend.frameon": False, "legend.fontsize": 9,
        "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.dpi": 140,
    })
    return plt


def ring(line_or_scatter):
    """2 px surface ring so markers stay legible where they overlap."""
    try:
        line_or_scatter.set_markeredgecolor(SURFACE)
        line_or_scatter.set_markeredgewidth(2.0)
    except AttributeError:
        line_or_scatter.set_edgecolor(SURFACE)
        line_or_scatter.set_linewidth(2.0)
    return line_or_scatter
