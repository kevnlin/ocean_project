"""The architecture diagram for the next update: what the GLOBAL reconstruction
pipeline does, and where the audit found it loses the signal.

Two rows.  The top row is the registered pipeline as it runs today, each stage
annotated with the number the audit measured there.  The bottom row is the
PhCA-style (Latent Neural Operator) fuse stage trialled in place of the
Perceiver-IO resampler and latent.

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


# ---------------------------------------------------------------- numbers
# every annotation is read from the global audit, so the figure is regenerated,
# never edited, when the pipeline or the data change
import json
AUD = json.load(open(os.path.join(ROOT, "outputs", "cache", "pipeline_audit.json")))
R = AUD["regions"]["global"]
D, S, O, T = R["input_density"], R["scale"], R["outliers"], R["tokenisation"]
C, LR = R["climatology_position"], T["local_refiner_init"]
per_month = D["available_per_month"]["median"]
n_tok = T["profile_tokens"]["tokens_per_profile"]
nlev = len(R["levels"])
sig_up = S["TEMP"]["anom_std_by_band"]["0-100m"]
bl = T["profile_tokens"]["band_mean_loses"]["TEMP"]["within_band_variance_fraction"]
salt_rms, salt_clean = O["SALT"]["dev_input_z_rms"], O["SALT"]["dev_input_z_rms_clean"]
near_all = D["nearest_input_km_all_profiles"]["median"]
bands = R.get("covariance_fit")
if bands:
    import numpy as np
    L1 = [np.median([lv["bands"][b]["L1_km"] for lv in bands["TEMP"]])
          for b in range(len(bands["TEMP"][0]["bands"]))]
    decor = f"{min(L1):.0f}-{max(L1):.0f} km by latitude band"
else:
    decor = "~50-250 km"

# ---------------------------------------------------------------- title
ax.text(1, 84.5, "Global ocean state reconstruction — the pipeline as it runs, "
                 "and where the signal is lost", fontsize=13, color=INK,
        fontweight="bold", va="top")
ax.text(1, 80.6, "whole ocean, one domain · every Argo profile of the month · WOA23 "
                 "monthly anomaly · per-level z-score · held-out (WMO-disjoint) "
                 "floats as truth", fontsize=9, color=INK_2, va="top")

# ---------------------------------------------------------------- top row
Y = 60.0
H = 17.0
box(1, Y, 20, H, "Real Argo profiles",
    ["QC'd GDAC, global", f"{nlev} levels, 5–{max(R['levels']):.0f} m",
     f"~{per_month:,.0f} / test month"], color=AQUA)
box(24.5, Y, 21, H, "Anomaly target",
    ["minus WOA23 monthly", "at the 1° CELL centre", "not at the profile"], color=AQUA)
box(49, Y, 19, H, "Per-level z-score",
    ["train-years mean/σ", f"σ = {sig_up:.2f} °C (0-100 m)", "one σ for all oceans"],
    color=AQUA)
box(71.5, Y, 21, H, "Profile tokeniser",
    [f"{n_tok} depth-band tokens", "mean-pooled levels", "at band midpoints"], color=BLUE)
box(96, Y, 18, H, "DFS evidence",
    ["ridge leverage τ", "kNN over every", "token of the month"], color=BLUE)
box(117.5, Y, 17, H, "Resampler",
    ["32 latent slots", "d = 64", "mass-conserving"], color=BLUE)
for x1, x2 in ((21, 24.5), (45.5, 49), (68, 71.5), (92.5, 96), (114, 117.5)):
    arrow(x1, Y + H / 2, x2, Y + H / 2)

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
finding(11, Y - 1.0, f"{O['SALT']['profiles_over_10_sigma']:,} salinity profiles\n"
                     f"beyond 10σ pass Argo QC;\nmax |z| = {O['SALT']['max_abs_z']:.0f}")
finding(35, Y - 1.0, f"profile up to {C['offset_km']['lat_max']:.0f} km from the\n"
                     f"centre → {C['TEMP']['std_difference_by_band']['0-100m']:.2f} °C "
                     f"spurious\nanomaly near the surface", color=WARN)
finding(58.5, Y - 1.0, f"test-year input salinity\nz-RMS {salt_rms:.2f} "
                       f"({salt_clean:.2f} without\nthe outliers)")
finding(82, Y - 1.0, f"{100 * bl:.0f} % of a profile's\nvertical variance pooled\n"
                     f"away; token at band mid", color=WARN)
finding(105, Y - 1.0, f"~{per_month * n_tok * 0.7 / 1000:.0f} k tokens per\n"
                      "training month: the\ncost driver", color=WARN)
finding(126, Y - 1.0, "all of it → 32 slots", color=WARN)
finding(115, Y2 - 1.4,
        f"refiner Gaussian starts at {LR['ell_lat_km']:,.0f} × "
        f"{LR['ell_lon_km_at_region']:,.0f} km and is gated at 0.05;\n"
        f"coordinate features resolve ≥ "
        f"{T['coordinate_features']['finest_sphere_wavelength_km']:.0f} km, while the anomaly "
        f"decorrelates in\n{decor}; median target → nearest input {near_all:.0f} km")

# ---------------------------------------------------------------- bottom row
Y3 = 3.0
ax.text(1, 21.0, "Backbone replacement — PhCA-style (LNO) fuse in place of the Perceiver-IO "
                 "resampler + latent; D4RT decoder and DFS unchanged (405 543 vs 405 063)",
        fontsize=10.5, color=INK, fontweight="bold", va="top")
box(1, Y3, 26, 14.0, "Position-only weights",
    ["MLP on each token's coordinates", "→ logits over 32 latent slots",
     "values never enter the weights"], color=ORANGE)
box(31, Y3, 24, 14.0, "Softmax over SLOTS",
    ["each token spreads its DFS τ", "slot mass Σ = total evidence",
     "(Slot-Attention normalisation)"], color=ORANGE)
box(59, Y3, 24, 14.0, "Slots vs reference",
    ["each slot: own content @ log m", "vs null / reference @ log λ_bg",
     "→ latent blocks → D4RT"], color=ORANGE)
for x1, x2 in ((27, 31), (55, 59)):
    arrow(x1, Y3 + 7.0, x2, Y3 + 7.0)

fig.savefig(os.path.join(OUT, "fig_architecture.png"), bbox_inches="tight", dpi=170)
fig.savefig(os.path.join(OUT, "fig_architecture.svg"), bbox_inches="tight")
print("wrote reports/real_data/fig_architecture.{png,svg}")
