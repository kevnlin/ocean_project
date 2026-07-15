"""Spatial reconstruction-error heatmap for the best Week-1 model.

Retrains the strongest baseline (depthwise U-Net, profiles+WOA+SST/SSS, anomaly
target, unobserved-only loss) and maps its per-location reconstruction error,
averaged over the held-out test months.

Two rows:
  * full-column RMSE  : sqrt(mean over depth & test months of (pred-truth)^2)
  * thermocline layer : same but at the ~105 m level where error peaks

Colour: 'hot' — **lighter = more error**, darker = better reconstruction.
Land is drawn grey (no data).  Output: reports/error_map.png

Run:
    python experiments/07_error_map.py [--smoke]
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

from ocean_tokenizer import data, baselines as B, config as C
from ocean_tokenizer.anomaly import Climatology, AnomNorm

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--depth-m", type=float, default=105.0,
                help="layer (metres) for the single-level panel")
args = ap.parse_args()
if args.smoke:
    C.N_TRAIN_MONTHS, C.N_TEST_MONTHS, C.N_PROFILES, C.UNET_EPOCHS = 6, 3, 600, 3

t0 = time.time()
rng = np.random.default_rng(C.SEED)
device = C.DEVICE if torch.cuda.is_available() else "cpu"
grid = data.CommonGrid()
print(grid, "device=", device)

tr_pool = data.select_month_indices(C.GT_SOURCE, C.TRAIN_YEARS)
te_pool = data.select_month_indices(C.GT_SOURCE, C.TEST_YEARS)
tr_idx = np.sort(rng.choice(tr_pool, size=min(C.N_TRAIN_MONTHS, tr_pool.size), replace=False))
te_idx = np.sort(rng.choice(te_pool, size=min(C.N_TEST_MONTHS, te_pool.size), replace=False))

print("loading fields ..."); ts = time.time()
ftrain = data.load_gt_fields(tr_idx, grid)
ftest = data.load_gt_fields(te_idx, grid)
woa = data.woa_prior(grid)
print(f"  {time.time()-ts:.1f}s")

surf_train = {v: ftrain[v] for v in C.VARS_SURF if v in ftrain}
clim = Climatology(ftrain, surf_train)
norm = AnomNorm(clim, ftrain, surf_train)

def make_samples(fields):
    return [B.prepare_month(fields, fields, woa, grid, t, rng, C.N_PROFILES)
            for t in range(len(fields["months"]))]
train_samples = make_samples(ftrain)
test_samples = make_samples(ftest)

# ---- best model ----
cfg = ("profiles", "woa", "surf")
print("training best model (unet_depthwise + surf) ..."); ts = time.time()
preds = B.train_predict_unet(train_samples, test_samples, grid, norm, cfg, device,
                             unobs_loss=True)
print(f"  {time.time()-ts:.1f}s")

TRUE = {v: np.stack([s["gt"][v] for s in test_samples], 0) for v in B.VARS}  # (N,D,H,W)
PRED = {v: np.stack([p[v] for p in preds], 0) for v in B.VARS}

kz = int(np.argmin(np.abs(grid.depth - args.depth_m)))
z_used = float(grid.depth[kz])
print(f"single-level panel at {z_used:.0f} m (index {kz})")

def rmse_col(v):      # per-(lat,lon) full-column RMSE over depth & months
    d = PRED[v] - TRUE[v]                          # (N,D,H,W)
    return np.sqrt(np.nanmean(d * d, axis=(0, 1)))  # (H,W)

def rmse_lvl(v):      # per-(lat,lon) RMSE at one depth level
    d = PRED[v][:, kz] - TRUE[v][:, kz]            # (N,H,W)
    return np.sqrt(np.nanmean(d * d, axis=0))       # (H,W)

panels = [
    ("TEMP", "full-column RMSE (degC)", rmse_col("TEMP")),
    ("SALT", "full-column RMSE (PSU)",  rmse_col("SALT")),
    ("TEMP", f"{z_used:.0f} m RMSE (degC)", rmse_lvl("TEMP")),
    ("SALT", f"{z_used:.0f} m RMSE (PSU)",  rmse_lvl("SALT")),
]

# ---- plot ----
cmap = plt.cm.hot.copy()
cmap.set_bad("0.75")               # land / no-data -> grey
extent = [float(grid.lon.min()), float(grid.lon.max()),
          float(grid.lat.min()), float(grid.lat.max())]

fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
for ax, (var, label, field) in zip(axes.ravel(), panels):
    m = np.ma.masked_invalid(field)
    vmax = float(np.nanpercentile(field, 99))       # robust upper limit
    im = ax.imshow(m, origin="lower", extent=extent, aspect="auto",
                   cmap=cmap, norm=Normalize(0, vmax))
    ax.set_title(f"{var} — {label}", fontsize=11)
    ax.set_xlabel("longitude (deg E)"); ax.set_ylabel("latitude (deg)")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("RMSE  (lighter = more error)", fontsize=9)

fig.suptitle("Reconstruction error — depthwise U-Net (profiles + WOA + SST/SSS), "
             f"anomaly target, {te_idx.size} held-out months",
             fontsize=13)
out = os.path.join(C.REPORTS, "error_map.png")
fig.savefig(out, dpi=140)
print(f"\nDONE in {time.time()-t0:.1f}s -> {out}")
print("  field RMSE ranges:")
for var, label, field in panels:
    print(f"    {var:4s} {label:24s}  min={np.nanmin(field):.3f}  "
          f"median={np.nanmedian(field):.3f}  p99={np.nanpercentile(field,99):.3f}")
