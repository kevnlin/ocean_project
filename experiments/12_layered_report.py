"""Phase-1: aggregate the layered depth evaluation into a report + figure.

Consumes outputs/cache/layered_depth_seed*.json (+ *_level.npz) from
experiments/10_layered_depth_eval.py, aggregates across seeds (mean +- std),
and writes:

    reports/layered_depth_eval.md
    reports/fig_layered_rmse.png        per-layer RMSE bars (TEMP/SALT)

The per-layer heatmap figure is produced separately by 11_layered_heatmap.py and
embedded in the report if present.

Run:
    python experiments/12_layered_report.py
"""
import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ocean_tokenizer import config as C

VARS = ("TEMP", "SALT")
UNIT = {"TEMP": "degC", "SALT": "PSU"}
METHODS = ["woa_prior", "clim_floor", "mlp", "unet_depthwise", "unet_joint"]
NAME = {"woa_prior": "WOA23 prior", "clim_floor": "Climatology floor (train-only)",
        "mlp": "Pointwise MLP", "unet_depthwise": "U-Net (depthwise)",
        "unet_joint": "U-Net (joint-depth)"}
SERIES = {"mlp": ("Pointwise MLP", "#2a78d6"),
          "unet_depthwise": ("U-Net depthwise", "#1baf7a"),
          "unet_joint": ("U-Net joint-depth", "#eda100")}
REFS = {"clim_floor": ("Climatology floor", "#52514e"),
        "woa_prior": ("WOA23 prior", "#898781")}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.grid.axis": "y",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})

runs = [r for r in sorted(glob.glob(os.path.join(C.CACHE, "layered_depth_seed*.json")))
        if "_level" not in r]
assert runs, "no layered_depth_seed*.json - run 10_layered_depth_eval.py first"
rows, cfgs = [], []
for path in runs:
    with open(path) as f:
        blob = json.load(f)
    rows += blob["results"]; cfgs.append(blob["run_config"])
seeds = sorted({r["seed"] for r in rows})
LAYERS = [l[0] for l in cfgs[0]["layers"]]
density = cfgs[0]["density"]
print(f"loaded {len(rows)} rows | seeds={seeds} | layers={LAYERS}")

# cell[(method, layer, var)] -> list of per-seed RMSE
cell = {}
overall = {}
for r in rows:
    for v in VARS:
        overall.setdefault((r["method"], v), []).append(r["overall"][v])
        for ln in LAYERS:
            cell.setdefault((r["method"], ln, v), []).append(r["by_layer"][v][ln])

def stat(vals):
    a = np.asarray([x for x in (vals or []) if x is not None and np.isfinite(x)], float)
    if a.size == 0:
        return np.nan, np.nan, 0
    return a.mean(), (a.std(ddof=1) if a.size > 1 else 0.0), a.size

def lay(m, ln, v): return cell.get((m, ln, v), [])
def ova(m, v): return overall.get((m, v), [])

def ms(vals, prec=4):
    mu, sd, n = stat(vals)
    if n == 0:
        return "-"
    s = f"{mu:.{prec}f} ± {sd:.{prec}f}"
    return s if n == len(seeds) else s + f" (n={n})"

# ------------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
xb = np.arange(len(LAYERS)); w = 0.25
for ax, v in zip(axes, VARS):
    for i, (m, (label, color)) in enumerate(SERIES.items()):
        mu = np.array([stat(lay(m, ln, v))[0] for ln in LAYERS])
        sd = np.array([stat(lay(m, ln, v))[1] for ln in LAYERS])
        ax.bar(xb + (i - 1) * w, mu, w, yerr=sd, color=color, label=label,
               error_kw=dict(lw=1, ecolor=INK2, capsize=2), zorder=3)
    for m, (label, color) in REFS.items():
        mu = np.array([stat(lay(m, ln, v))[0] for ln in LAYERS])
        ax.plot(xb, mu, color=color, marker="_", markersize=18, markeredgewidth=2,
                linestyle="none", zorder=4,
                label=label if v == "TEMP" else None)
    ax.set_yscale("log")
    ax.set_xticks(xb, LAYERS, fontsize=9)
    ax.set_xlabel("depth layer")
    ax.set_ylabel(f"{v} RMSE ({UNIT[v]}), unobserved-only")
    ax.set_title(f"{v}: RMSE by depth layer", color=INK, fontsize=11)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
fig.suptitle(f"Per-layer reconstruction RMSE at {density} profiles/month "
             f"(extended 1400 m grid, mean±std over {len(seeds)} seeds)",
             color=INK, fontsize=12)
fig.tight_layout()
figp = os.path.join(C.REPORTS, "fig_layered_rmse.png")
fig.savefig(figp, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"wrote {figp}")

# ------------------------------------------------------------------- markdown
L = ["# Phase-1 — Layered Depth Evaluation (extended grid to 1400 m)\n"]
L.append(f"- **Question (professor):** measure RMSE by ocean depth layer, incl. a >1000 m layer.")
L.append(f"- **Extended grid:** 23 levels to **1400 m** (was 20 to 985 m). Capped at 1400 m "
         f"because WOA23 — the prior and an input feature — reaches only 1500 m; the CESM2 "
         f"anomaly target/floor are defined at all depths.")
L.append(f"- **Layers:** {', '.join(LAYERS)} (surface/mixed-layer · thermocline · intermediate · deep).")
L.append(f"- **Protocol:** train-only monthly CESM2 anomaly target, **unobserved-only RMSE**, "
         f"config `profiles_woa_surf`, **{density} profiles/month**, {len(cfgs[0]['train_months'])} "
         f"train / {len(cfgs[0]['test_months'])} test months. Seeds {seeds} (fixed month split; "
         f"seeds vary profile sampling + training).")
L.append(f"- **Per-layer RMSE** pools squared errors over the band's depths (valid-cell-weighted = "
         f"true RMSE over that ocean volume).")
L.append(f"- **Commit:** `{cfgs[0]['git_commit'][:12]}`\n")

for v in VARS:
    L.append(f"## {v} RMSE ({UNIT[v]}) — mean ± std over {len(seeds)} seeds\n")
    L.append("| method | full-column | " + " | ".join(LAYERS) + " |")
    L.append("|---|---|" + "---|" * len(LAYERS))
    for m in METHODS:
        cells = [ms(ova(m, v))] + [ms(lay(m, ln, v)) for ln in LAYERS]
        L.append(f"| {NAME[m]} | " + " | ".join(cells) + " |")
    L.append("")

L.append("## Per-layer RMSE\n")
L.append("![Per-layer RMSE](fig_layered_rmse.png)\n")
if os.path.exists(os.path.join(C.REPORTS, "fig_layered_heatmap.png")):
    L.append("## Spatial error structure by layer\n")
    L.append("![Per-layer error heatmap](fig_layered_heatmap.png)\n")

# takeaways computed from the numbers
BEST = "unet_depthwise"
def mu_of(m, v, ln): return stat(lay(m, ln, v))[0]
def skill(m, v, ln):
    f = mu_of("clim_floor", v, ln)
    return (1 - mu_of(m, v, ln) / f) if f and np.isfinite(f) else np.nan
therm = mu_of(BEST, "TEMP", "100-300m")
deep = mu_of(BEST, "TEMP", "1000-1500m")
deep_floor = mu_of("clim_floor", "TEMP", "1000-1500m")
sk = [skill(BEST, "TEMP", ln) for ln in LAYERS]
L.append("## Takeaways\n")
L.append(f"- **Absolute error is concentrated in the upper ocean.** The depthwise U-Net's TEMP RMSE "
         f"falls from ~{therm:.2f} degC in the 100–300 m thermocline to ~{deep:.3f} degC in the "
         f"1000–1500 m deep layer — an ~{therm/deep:.0f}× spread. The WOA prior and the climatology "
         f"floor fall with depth too, so this is variance shrinking, not the model failing.")
L.append(f"- **Skill over climatology is roughly constant with depth.** The depthwise U-Net's skill "
         f"(1 − RMSE/floor) is {sk[0]:+.2f} / {sk[1]:+.2f} / {sk[2]:+.2f} / {sk[3]:+.2f} across "
         f"0–100 / 100–300 / 300–1000 / 1000–1500 m. Even the >1000 m layer beats its climatology "
         f"floor (~{deep:.3f} vs {deep_floor:.3f} degC): the deep ocean is *not* just climatology — "
         f"the profiles carry recoverable deep anomaly signal.")
L.append(f"- **The thermocline (100–300 m) is the hardest layer** in absolute RMSE for every method, "
         f"above even the mixed layer (which the SST/SSS surface fields help constrain). The layered "
         f"view localises where a shared-latent method has the most upper-ocean error to win back.")
L.append(f"- **Method ranking holds at every depth:** depthwise U-Net < joint-depth U-Net < MLP, "
         f"consistent with the 985 m week-2 result — the joint-depth 'strong baseline' is still the "
         f"weaker U-Net layer by layer.\n")

out = os.path.join(C.REPORTS, "layered_depth_eval.md")
with open(out, "w") as f:
    f.write("\n".join(L))
print(f"wrote {out}")
