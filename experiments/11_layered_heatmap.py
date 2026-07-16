"""Phase-1: per-layer spatial reconstruction-error heatmaps (extended depth).

Generalises the single-panel week-1 error map into one spatial RMSE map per
oceanographic layer, so the *structure* of the error is visible by depth (e.g.
thermocline error concentrating in western-boundary currents vs. a flat, small
deep field).  Model = joint-depth U-Net (profiles + WOA + SST/SSS, anomaly
target, unobserved-only loss) — the whole-column baseline — on the extended
1400 m grid.

Per-(lat,lon) layer RMSE pools squared errors over the band's depths and the
held-out test months, scored on UNOBSERVED columns only (profile columns
excluded, consistent with the headline metric).

Output: reports/fig_layered_heatmap.png  (rows = TEMP/SALT, cols = 4 layers)

Run:
    CUDA_VISIBLE_DEVICES=6 python experiments/11_layered_heatmap.py [--smoke]
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from ocean_tokenizer import config as C
EXTENDED_DEPTH_INDICES = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 21, 24,
                          27, 30, 33, 36, 39, 40, 41, 42]
LAYERS = [("0-100 m", 0.0, 100.0), ("100-300 m", 100.0, 300.0),
          ("300-1000 m", 300.0, 1000.0), ("1000-1500 m", 1000.0, 1500.0)]

from ocean_tokenizer import data, baselines as B
from ocean_tokenizer.anomaly import Climatology, AnomNorm

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--density", type=int, default=1500)
ap.add_argument("--seed", type=int, default=C.SEED)
args = ap.parse_args()

C.DEPTH_INDICES = EXTENDED_DEPTH_INDICES
if args.smoke:
    C.N_TRAIN_MONTHS, C.N_TEST_MONTHS, C.UNET_EPOCHS = 6, 3, 3

t0 = time.time()
device = C.DEVICE if torch.cuda.is_available() else "cpu"
grid = data.CommonGrid()
print(grid, "device=", device, flush=True)

split_rng = np.random.default_rng(C.SEED)
tr_pool = data.select_month_indices(C.GT_SOURCE, C.TRAIN_YEARS)
te_pool = data.select_month_indices(C.GT_SOURCE, C.TEST_YEARS)
tr_idx = np.sort(split_rng.choice(tr_pool, size=min(C.N_TRAIN_MONTHS, tr_pool.size), replace=False))
te_idx = np.sort(split_rng.choice(te_pool, size=min(C.N_TEST_MONTHS, te_pool.size), replace=False))

print("loading fields ...", flush=True); ts = time.time()
ftrain = data.load_gt_fields(tr_idx, grid)
ftest = data.load_gt_fields(te_idx, grid)
woa = data.woa_prior(grid)
print(f"  {time.time()-ts:.1f}s", flush=True)

surf_train = {v: ftrain[v] for v in C.VARS_SURF if v in ftrain}
clim = Climatology(ftrain, surf_train)
norm = AnomNorm(clim, ftrain, surf_train)

rng = np.random.default_rng([args.seed, args.density])
torch.manual_seed(args.seed * 100_003 + args.density)
train_samples = [B.prepare_month(ftrain, ftrain, woa, grid, t, rng, args.density)
                 for t in range(len(ftrain["months"]))]
test_samples = [B.prepare_month(ftest, ftest, woa, grid, t, rng, args.density)
                for t in range(len(ftest["months"]))]

print("training joint-depth U-Net ...", flush=True); ts = time.time()
preds = B.train_predict_unet_joint(train_samples, test_samples, grid, norm,
                                   ("profiles", "woa", "surf"), device, unobs_loss=True)
print(f"  {time.time()-ts:.1f}s", flush=True)

TRUE = {v: np.stack([s["gt"][v] for s in test_samples], 0) for v in B.VARS}   # (N,D,H,W)
PRED = {v: np.stack([p[v] for p in preds], 0) for v in B.VARS}
UNOBS = np.stack([s["unobs_mask"] for s in test_samples], 0)                  # (N,H,W)
depths = grid.depth


def layer_rmse_map(v, lo, hi):
    """Per-(lat,lon) RMSE pooled over the layer's depths & test months,
    scored on unobserved columns only."""
    sel = (depths > lo) & (depths <= hi)
    if lo <= depths.min():
        sel = sel | np.isclose(depths, depths.min())
    di = np.where(sel)[0]
    d = (PRED[v][:, di] - TRUE[v][:, di]) ** 2          # (N,k,H,W)
    keep = np.broadcast_to(UNOBS[:, None, :, :], d.shape)
    d = np.where(keep, d, np.nan)
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.nanmean(d, axis=(0, 1)))       # (H,W)


UNIT = {"TEMP": "degC", "SALT": "PSU"}
cmap = plt.cm.hot.copy(); cmap.set_bad("0.75")
extent = [float(grid.lon.min()), float(grid.lon.max()),
          float(grid.lat.min()), float(grid.lat.max())]

fig, axes = plt.subplots(len(B.VARS), len(LAYERS), figsize=(4.1 * len(LAYERS), 6.4),
                         constrained_layout=True)
print("\nlayer RMSE (median over ocean):", flush=True)
for r, v in enumerate(B.VARS):
    # shared colour scale per variable across layers -> layers are comparable
    maps = [layer_rmse_map(v, lo, hi) for _, lo, hi in LAYERS]
    vmax = float(np.nanpercentile(np.concatenate([m[np.isfinite(m)] for m in maps]), 99))
    for c, ((name, lo, hi), fld) in enumerate(zip(LAYERS, maps)):
        ax = axes[r, c]
        im = ax.imshow(np.ma.masked_invalid(fld), origin="lower", extent=extent,
                       aspect="auto", cmap=cmap, norm=Normalize(0, vmax))
        if r == 0:
            ax.set_title(name, fontsize=11)
        if c == 0:
            ax.set_ylabel(f"{v} ({UNIT[v]})\nlatitude", fontsize=10)
        ax.set_xlabel("longitude" if r == len(B.VARS) - 1 else "")
        ax.tick_params(labelsize=8)
        print(f"  {v} {name:12s} median={np.nanmedian(fld):.4f} p99={vmax:.4f}", flush=True)
    fig.colorbar(im, ax=axes[r, :], shrink=0.7, location="right",
                 label=f"{v} RMSE ({UNIT[v]}) — lighter = worse")

fig.suptitle("Reconstruction error by depth layer — joint-depth U-Net "
             "(prof+WOA+SST/SSS), unobserved-only, anomaly target", fontsize=13)
out = os.path.join(C.REPORTS, "fig_layered_heatmap.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nDONE in {time.time()-t0:.1f}s -> {out}", flush=True)
