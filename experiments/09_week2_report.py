"""Week-2 report: aggregate the multi-seed density ablation into tables + figures.

Consumes outputs/cache/density_ablation_seed*.json (+ *_depth.npz) written by
experiments/08_density_ablation.py, aggregates across seeds (mean +- std,
min-max bands), and writes:

    reports/week2_density_ablation.md
    reports/fig_week2_density_rmse.png     RMSE vs profile density (TEMP/SALT)
    reports/fig_week2_depth_rmse.png       RMSE by depth at the standard density

Tolerates partial sweeps: cells are aggregated over whichever seeds have
finished them, and the per-cell seed count is reported.

Run:
    python experiments/09_week2_report.py
"""
import sys, os, json, glob, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ocean_tokenizer import config as C

VARS = ("TEMP", "SALT")
UNIT = {"TEMP": "degC", "SALT": "PSU"}

ap = argparse.ArgumentParser()
ap.add_argument("--std-density", type=int, default=1500,
                help="headline operating point for table 1 / figure 2")
args = ap.parse_args()
STD_DENSITY = args.std_density

# ---- palette (validated reference palette; refs are neutral ink) ----
SERIES = {                               # categorical slots, fixed order
    "mlp":            ("Pointwise MLP",    "#2a78d6", "o"),
    "unet_depthwise": ("U-Net depthwise",  "#1baf7a", "s"),
    "unet_joint":     ("U-Net joint-depth", "#eda100", "D"),
}
REFS = {                                 # reference floors: neutral grays
    "clim_floor": ("CESM2 train climatology (floor)", "#52514e", (0, (4, 2))),
    "woa_prior":  ("WOA23 prior",                     "#898781", (0, (1, 2))),
}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

# ---------------------------------------------------------------- load sweeps
runs = sorted(glob.glob(os.path.join(C.CACHE, "density_ablation_seed*.json")))
runs = [r for r in runs if "_depth" not in r]
assert runs, "no density_ablation_seed*.json found - run 08_density_ablation.py"

rows, run_cfgs = [], []
for path in runs:
    with open(path) as f:
        blob = json.load(f)
    rows += blob["results"]
    run_cfgs.append(blob["run_config"])
seeds = sorted({r["seed"] for r in rows})
densities = sorted({r["density"] for r in rows})
methods = ["woa_prior", "clim_floor", "mlp", "unet_depthwise", "unet_joint"]
print(f"loaded {len(rows)} rows | seeds={seeds} | densities={densities}")

# cell[(method, density)][var] -> list of per-seed RMSE (unobserved-only)
cell = {}
for r in rows:
    for v in VARS:
        cell.setdefault((r["method"], r["density"]), {}).setdefault(v, []).append(
            r[f"{v}_unobs"])

def stat(method, density, var):
    vals = cell.get((method, density), {}).get(var, [])
    if not vals:
        return np.nan, np.nan, np.nan, np.nan, 0
    a = np.asarray(vals, float)
    return a.mean(), a.std(ddof=1) if a.size > 1 else 0.0, a.min(), a.max(), a.size

# ---------------------------------------------------- depth tables (npz mean)
depth_acc = {}                           # (method, density, var) -> list of (D,)
depths = None
for path in sorted(glob.glob(os.path.join(C.CACHE, "density_ablation_seed*_depth.npz"))):
    z = np.load(path)
    depths = z["depths"]
    for k in z.files:
        if k == "depths":
            continue
        m, d, v = k.split("__")
        depth_acc.setdefault((m, int(d), v), []).append(z[k])

def depth_mean(method, density, var):
    key = (method, density, var)
    return (np.nanmean(np.stack(depth_acc[key]), 0) if key in depth_acc else None)

# ------------------------------------------------------------------- figure 1
def last_finite(xs, ys):
    ok = np.where(np.isfinite(ys))[0]
    return (xs[ok[-1]], ys[ok[-1]]) if ok.size else (None, None)

def place_labels(ax, anchors, fontsize=8.5):
    """Direct labels at line ends, nudged apart in log space if they collide."""
    anchors = [a for a in anchors if a[1] is not None]
    anchors.sort(key=lambda a: a[2])
    logy = [np.log10(a[2]) for a in anchors]
    min_gap = 0.05                                  # ~12% apart in value space
    for i in range(1, len(logy)):
        if logy[i] - logy[i - 1] < min_gap:
            logy[i] = logy[i - 1] + min_gap
    for (label, xa, _), ly in zip(anchors, logy):
        ax.annotate(label, (xa, 10 ** ly), xytext=(6, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=fontsize, color=INK2)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
x = np.arange(len(densities))
for ax, v in zip(axes, VARS):
    anchors = []
    for m, (label, color, ls) in REFS.items():
        mu = np.array([stat(m, d, v)[0] for d in densities])
        ax.plot(x, mu, color=color, linestyle=ls, linewidth=1.6, zorder=2)
        xa, ya = last_finite(x, mu)
        anchors.append((label, xa, ya))
    for m, (label, color, marker) in SERIES.items():
        mu = np.array([stat(m, d, v)[0] for d in densities])
        lo = np.array([stat(m, d, v)[2] for d in densities])
        hi = np.array([stat(m, d, v)[3] for d in densities])
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=3)
        ax.plot(x, mu, color=color, marker=marker, markersize=5.5,
                linewidth=2, zorder=4)
        xa, ya = last_finite(x, mu)
        anchors.append((label, xa, ya))
    ax.set_yscale("log")
    place_labels(ax, anchors)
    ax.set_xticks(x, [str(d) for d in densities])
    ax.set_xlabel("synthetic Argo profiles per month (ordinal spacing)")
    ax.set_ylabel(f"{v} RMSE ({UNIT[v]}), unobserved columns only")
    ax.set_title(f"{v}: skill vs profile density", color=INK, fontsize=11)
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(-0.3, len(densities) - 1 + 2.4)   # room for direct labels
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.suptitle(f"Profile-density ablation - anomaly target, unobserved-only RMSE "
             f"(band = min-max over {len(seeds)} seeds)",
             color=INK, fontsize=12, y=1.0)
fig.tight_layout()
fig1_path = os.path.join(C.REPORTS, "fig_week2_density_rmse.png")
fig.savefig(fig1_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"wrote {fig1_path}")

# ------------------------------------------------------------------- figure 2
fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.2), sharey=True)
for ax, v in zip(axes, VARS):
    for m, (label, color, ls) in REFS.items():
        rd = depth_mean(m, STD_DENSITY, v)
        if rd is not None:
            ax.plot(rd, depths, color=color, linestyle=ls, linewidth=1.6)
    for m, (label, color, marker) in SERIES.items():
        rd = depth_mean(m, STD_DENSITY, v)
        if rd is not None:
            ax.plot(rd, depths, color=color, marker=marker, markersize=4.5,
                    linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel(f"{v} RMSE ({UNIT[v]})")
    ax.set_title(v, color=INK, fontsize=11)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("depth (m)")
axes[0].invert_yaxis()
handles = ([plt.Line2D([], [], color=c, marker=mk, linewidth=2, markersize=5)
            for _, c, mk in SERIES.values()]
           + [plt.Line2D([], [], color=c, linestyle=ls, linewidth=1.6)
              for _, c, ls in REFS.values()])
labels = [l for l, _, _ in SERIES.values()] + [l for l, _, _ in REFS.values()]
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.13), fontsize=9)
fig.suptitle(f"RMSE by depth at {STD_DENSITY} profiles/month "
             f"(seed mean, unobserved-only)", color=INK, fontsize=12)
fig.tight_layout()
fig2_path = os.path.join(C.REPORTS, "fig_week2_depth_rmse.png")
fig.savefig(fig2_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"wrote {fig2_path}")

# ------------------------------------------------------------------- markdown
NAME = {"woa_prior": "WOA23 prior", "clim_floor": "Climatology floor (train-only)",
        "mlp": "Pointwise MLP", "unet_depthwise": "U-Net (depthwise)",
        "unet_joint": "U-Net (joint-depth)"}

def ms(method, density, var, prec=4):
    mu, sd, lo, hi, n = stat(method, density, var)
    if n == 0:
        return "-"
    s = f"{mu:.{prec}f} ± {sd:.{prec}f}"
    return s if n == len(seeds) else s + f" (n={n})"

L = ["# Week-2 — Profile-Density Ablation (multi-seed) + Shared-Latent Interface\n"]
L.append(f"- **Protocol:** identical to the week-1 audit — train-only monthly CESM2 "
         f"anomaly target, **unobserved-only RMSE** (observed profile columns excluded), "
         f"input config `profiles_woa_surf` for the learned methods.")
L.append(f"- **Seeds:** {seeds} — the train/test month split is FIXED (seed {C.SEED}); "
         f"seeds vary the profile sampling, MLP point subsampling, and torch "
         f"init/training. Spread = observation-sampling + training variance.")
L.append(f"- **Split:** {run_cfgs[0]['train_months'].__len__()} train / "
         f"{run_cfgs[0]['test_months'].__len__()} test months.")
L.append(f"- **Densities:** {densities} profiles/month; every learned method is "
         f"**retrained per density** (the retrain-required contrast the shared-latent "
         f"model is meant to beat with a single model).")
L.append(f"- **Commit:** `{run_cfgs[0]['git_commit'][:12]}`\n")

L.append(f"## Headline — {STD_DENSITY} profiles/month (mean ± std over {len(seeds)} seeds)\n")
L.append("| method | TEMP RMSE (degC) | SALT RMSE (PSU) | TEMP skill | SALT skill |")
L.append("|---|---|---|---|---|")
floor_mu = {v: stat("clim_floor", STD_DENSITY, v)[0] for v in VARS}
for m in methods:
    sk = {}
    for v in VARS:
        mu = stat(m, STD_DENSITY, v)[0]
        sk[v] = "—" if m in REFS else f"{1 - mu / floor_mu[v]:+.3f}"
    L.append(f"| {NAME[m]} | {ms(m, STD_DENSITY, 'TEMP')} | {ms(m, STD_DENSITY, 'SALT')} "
             f"| {sk['TEMP']} | {sk['SALT']} |")
L.append("\nSkill = 1 − RMSE / floor-RMSE at the same density (positive = beats the "
         "train-only climatology).\n")

L.append("## RMSE vs profile density\n")
L.append("![RMSE vs profile density](fig_week2_density_rmse.png)\n")
for v in VARS:
    L.append(f"### {v} ({UNIT[v]})\n")
    L.append("| method | " + " | ".join(str(d) for d in densities) + " |")
    L.append("|---|" + "---|" * len(densities))
    for m in methods:
        L.append(f"| {NAME[m]} | " + " | ".join(ms(m, d, v) for d in densities) + " |")
    L.append("")

L.append("## RMSE by depth at the standard density\n")
L.append("![RMSE by depth](fig_week2_depth_rmse.png)\n")

L.append("## Per-seed detail (unobserved-only RMSE, TEMP / SALT)\n")
L.append("| seed | density | " + " | ".join(NAME[m] for m in methods) + " |")
L.append("|---|---|" + "---|" * len(methods))
by_seed = {}
for r in rows:
    by_seed.setdefault((r["seed"], r["density"]), {})[r["method"]] = r
for (s, d) in sorted(by_seed):
    cells = []
    for m in methods:
        r = by_seed[(s, d)].get(m)
        cells.append("-" if r is None else f"{r['TEMP_unobs']:.3f} / {r['SALT_unobs']:.4f}")
    L.append(f"| {s} | {d} | " + " | ".join(cells) + " |")
L.append("")

L.append("## Shared-latent interface status (Week-2 deliverable)\n")
L.append("- `src/ocean_tokenizer/token_api.py` — unified `TokenBatch` observation-token "
         "schema; `GridPatchEncoder` (surface + volume), `ProfileEncoder` (profile → "
         "multiple depth-segment tokens), `PointEncoder`; shared `coord_features` "
         "guaranteeing encoder/query coordinate consistency; abstract "
         "`SharedLatentModel` (encode → fuse → decode) contract.")
L.append("- `SharedLatentStub` — masked mean-pool + MLP query head. Interface plumbing "
         "only (permutation/padding-invariant by construction); **not** the method and "
         "not benchmarked as such.")
L.append("- `tests/test_token_api.py` — 19 tests covering variable profile counts "
         "(incl. 0), missing modalities (all subsets), coordinate consistency, "
         "mask/padding/permutation invariance, NaN handling, gradient flow, and the "
         "`prepare_month` bridge.")
L.append("- **Deliberately deferred to Weeks 3-4:** the Perceiver-style fusion core and "
         "the dense-grid-vs-sparse-profile token-imbalance handling.\n")

out = os.path.join(C.REPORTS, "week2_density_ablation.md")
with open(out, "w") as f:
    f.write("\n".join(L))
print(f"wrote {out}")
