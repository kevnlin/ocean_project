"""The architecture diagram for the next update: what the pipeline does, and where
the audit found it loses the signal.

Two rows.  The top row is the registered pipeline as it runs today, each stage
annotated with the number the audit measured there.  The bottom row is the
translation-equivariant alternative being trialled (SetConv -> U-Net -> query),
drawn to the same scale so the two are comparable at a glance.

  .venv/bin/python experiments/real_data/64_architecture_figure.py
"""
from __future__ import annotations

import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from ocean_tokenizer.plot_style import use_style, SERIES, INK, INK_2, MUTED, STATUS

plt = use_style()
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "reports", "real_data")

BLUE, ORANGE, AQUA = SERIES[0], SERIES[1], SERIES[2]
CRIT, WARN = STATUS["critical"], STATUS["serious"]

fig, ax = plt.subplots(figsize=(13.6, 8.6))
ax.set_xlim(0, 136); ax.set_ylim(0, 86); ax.axis("off")
ax.grid(False)


def box(x, y, w, h, title, lines, color=BLUE, fill="#ffffff", lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.4",
                                linewidth=lw, edgecolor=color, facecolor=fill))
    ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top", fontsize=9.5,
            color=INK, fontweight="bold")
    top = y + h - 6.2
    step = min(3.0, max(2.4, (top - y - 1.4) / max(len(lines), 1)))
    for i, t in enumerate(lines):
        ax.text(x + w / 2, top - i * step, t, ha="center", va="top",
                fontsize=8.2, color=INK_2)


def arrow(x1, y1, x2, y2, color=MUTED, label=None, dy=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=1.6, color=color,
                                 shrinkA=0, shrinkB=0))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, ha="center", va="bottom",
                fontsize=7.8, color=MUTED)


def finding(x, y, text, color=CRIT):
    ax.text(x, y, text, ha="center", va="top", fontsize=7.8, color=color,
            style="italic", linespacing=1.35)


# ---------------------------------------------------------------- title
ax.text(1, 84.5, "Sparse-profile ocean reconstruction — the pipeline as it runs, "
                 "and where the signal is lost", fontsize=13, color=INK,
        fontweight="bold", va="top")
ax.text(1, 80.6, "Gulf Stream / N. Pacific gyre · WOA23 monthly anomaly target · "
                 "per-level z-score · held-out (WMO-disjoint) Argo floats as truth",
        fontsize=9, color=INK_2, va="top")

# ---------------------------------------------------------------- top row
Y = 60.0
H = 17.0
box(1, Y, 20, H, "Real Argo profiles",
    ["QC'd GDAC cohort", "23 levels, 5–1400 m", "~250–470 / month"], color=AQUA)
box(24.5, Y, 21, H, "Anomaly target",
    ["minus WOA23 monthly", "at the region CELL", "centre (0.66°×1.96°)"], color=AQUA)
box(49, Y, 19, H, "Per-level z-score",
    ["train-years mean/σ", "σ = 1.6–2.1 °C (GS)", "plain, not robust"], color=AQUA)
box(71.5, Y, 21, H, "Profile tokeniser",
    ["5 depth-band tokens", "mean-pooled levels", "at band midpoints"], color=BLUE)
box(96, Y, 18, H, "DFS evidence",
    ["ridge leverage τ", "per token", "CV 0.08–0.17"], color=BLUE)
box(117.5, Y, 17, H, "Resampler",
    ["32 latent slots", "d = 64", "mass-conserving"], color=BLUE)

for x1, x2 in ((21, 24.5), (45.5, 49), (68, 71.5), (92.5, 96), (114, 117.5)):
    arrow(x1, Y + H / 2, x2, Y + H / 2)

# second line of the top row: latent -> decoder -> prediction
Y2 = 32.0
box(96, Y2, 38.5, 15.0, "D4RT query decoder",
    ["independent-query cross-attention into the 32 latents",
     "+ query-local refiner over tokens  + T/S expert heads"], color=BLUE)
box(71.5, Y2, 21, 15.0, "Prediction",
    ["T, S anomaly at", "(lat, lon, depth, month)", "of a held-out float"], color=BLUE)
ax.add_patch(FancyArrowPatch((126, Y), (126, Y2 + 15.0), arrowstyle="-|>",
                             mutation_scale=13, linewidth=1.6, color=MUTED))
arrow(96, Y2 + 7.5, 92.5, Y2 + 7.5)

# ---------------------------------------------------------------- findings
finding(11, Y - 1.0, "cap 128/month → 32 % of\nthe floats that reported;\n"
                     "nearest input 148 km (vs 113)")
finding(35, Y - 1.0, "profile sits up to 100 km\nfrom that centre → 0.33 °C\n"
                     "spurious anomaly (4.5 % var)", color=WARN)
finding(58.5, Y - 1.0, "outliers set the scale:\none float at |z| = 719\n"
                       "inflates deep σ 6×")
finding(82, Y - 1.0, "vertical detail pooled away;\ntoken depth = band midpoint\n"
                     "(300 m query ← 222–409 m)", color=WARN)
finding(105, Y - 1.0, "little redundancy to correct\nat this density → DFS ≈ Uniform")
finding(126, Y - 1.0, "640 tokens → 32 slots", color=WARN)
finding(115, Y2 - 1.4, "refiner Gaussian starts at 3 500 × 5 600 km — wider than the region box —\n"
                       "and gated at 0.05; coordinate features resolve ≥ 400 km, while the\n"
                       "anomaly decorrelates in 50–150 km")

# ---------------------------------------------------------------- bottom row
Y3 = 3.0
ax.text(1, 21.0, "Alternative backbone under trial — locality and translation "
                 "equivariance built in, size-matched (409 K vs 405 K)",
        fontsize=10.5, color=INK, fontweight="bold", va="top")
box(1, Y3, 26, 14.0, "SetConv encoder",
    ["profiles smeared onto a 0.5° grid", "value + density channel per level",
     "Gaussian width learned (init 75 km)"], color=ORANGE)
box(31, Y3, 24, 14.0, "U-Net processor",
    ["translation equivariant", "receptive field grows with depth",
     "no token bottleneck"], color=ORANGE)
box(59, Y3, 24, 14.0, "Bilinear query readout",
    ["whole field written out", "sampled at the float's", "own position and level"],
    color=ORANGE)
box(87, Y3, 24, 14.0, "Kriging OI (reference)",
    ["covariance fitted on train years", "zero trained parameters",
     "the number to beat"], color=MUTED, lw=1.4)
for x1, x2 in ((27, 31), (55, 59)):
    arrow(x1, Y3 + 7.0, x2, Y3 + 7.0)
ax.text(85, Y3 + 7.0, "vs", ha="center", va="center", fontsize=9, color=MUTED)

fig.savefig(os.path.join(OUT, "fig_architecture.png"), bbox_inches="tight", dpi=170)
fig.savefig(os.path.join(OUT, "fig_architecture.svg"), bbox_inches="tight")
print("wrote reports/real_data/fig_architecture.{png,svg}")
