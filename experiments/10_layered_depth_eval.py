"""Phase-1: layered depth evaluation on an extended (to ~1400 m) grid.

Answers the professor's evaluation ask — "split into layers to measure RMSE at
different ocean depths (<1000 m, >1000 m, ...)".  Two changes vs the week-1/2
runs, everything else identical (train-only monthly CESM2 anomaly target,
unobserved-only RMSE, config profiles_woa_surf):

  1. EXTENDED DEPTH GRID — 23 levels to 1400 m (was 20 levels to 985 m).  The
     three deeper native CESM2-LE levels (1106/1245/1400 m) give a genuine
     >1000 m layer.  Capped at 1400 m because WOA23 (the prior + input feature)
     only reaches 1500 m; below that the prior would be undefined.

  2. PER-LAYER RMSE — reported over four oceanographic layers:
        0-100 m    surface / mixed layer
        100-300 m  thermocline (where error peaks)
        300-1000 m intermediate
        1000-1500 m deep  (the >1000 m layer)
     Layer RMSE pools squared errors over the band's depths (valid-cell-weighted
     = the true RMSE over that ocean volume).

Focused at the headline density (1500 profiles/month); the density story is
already characterised at 985 m in week 2.  One process per seed (parallel GPUs).
Outputs are namespaced (layered_depth_*) so the 985 m week-2 results are intact.

Run (one seed per GPU):
    CUDA_VISIBLE_DEVICES=6 python experiments/10_layered_depth_eval.py --seed 1234
    CUDA_VISIBLE_DEVICES=7 python experiments/10_layered_depth_eval.py --seed 1235
"""
import sys, os, json, time, argparse, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

# ---- extend the depth grid BEFORE CommonGrid reads it ----
from ocean_tokenizer import config as C
EXTENDED_DEPTH_INDICES = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 21, 24,
                          27, 30, 33, 36, 39, 40, 41, 42]   # -> ~5..1400 m
LAYERS = [("0-100m", 0.0, 100.0), ("100-300m", 100.0, 300.0),
          ("300-1000m", 300.0, 1000.0), ("1000-1500m", 1000.0, 1500.0)]

from ocean_tokenizer import data, baselines as B, metrics
from ocean_tokenizer.anomaly import Climatology, AnomNorm

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True,
                help="sweep seed (profile sampling + torch); month split stays C.SEED")
ap.add_argument("--density", type=int, default=1500)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

C.DEPTH_INDICES = EXTENDED_DEPTH_INDICES
if args.smoke:
    C.N_TRAIN_MONTHS, C.N_TEST_MONTHS = 6, 3
    C.MLP_EPOCHS, C.UNET_EPOCHS = 4, 3
    C.UNET_JOINT_EPOCHS = 8
    C.MLP_POINTS_PER_MONTH = 30_000

CFG = ("profiles", "woa", "surf")
CFG_NAME = "profiles_woa_surf"

t0 = time.time()
device = C.DEVICE if torch.cuda.is_available() else "cpu"
print(f"seed={args.seed} density={args.density} device={device} smoke={args.smoke}",
      flush=True)

grid = data.CommonGrid()
print(grid, flush=True)
assert grid.ndepth == len(EXTENDED_DEPTH_INDICES)
print(f"depths(m): {[round(float(d)) for d in grid.depth]}", flush=True)

# ---- month split: ALWAYS C.SEED, identical to week-1/2 ----
split_rng = np.random.default_rng(C.SEED)
tr_pool = data.select_month_indices(C.GT_SOURCE, C.TRAIN_YEARS)
te_pool = data.select_month_indices(C.GT_SOURCE, C.TEST_YEARS)
tr_idx = np.sort(split_rng.choice(tr_pool, size=min(C.N_TRAIN_MONTHS, tr_pool.size), replace=False))
te_idx = np.sort(split_rng.choice(te_pool, size=min(C.N_TEST_MONTHS, te_pool.size), replace=False))
print(f"train months={tr_idx.size} test months={te_idx.size}", flush=True)

print("loading GT fields (extended depth) ...", flush=True); ts = time.time()
ftrain = data.load_gt_fields(tr_idx, grid)
ftest = data.load_gt_fields(te_idx, grid)
woa = data.woa_prior(grid)
print(f"  loaded in {time.time()-ts:.1f}s", flush=True)

surf_train = {v: ftrain[v] for v in C.VARS_SURF if v in ftrain}
clim = Climatology(ftrain, surf_train)
norm = AnomNorm(clim, ftrain, surf_train)

TRUE = {v: ftest[v] for v in B.VARS}

rng = np.random.default_rng([args.seed, args.density])
torch.manual_seed(args.seed * 100_003 + args.density)

print("assembling samples ...", flush=True); ts = time.time()
train_samples = [B.prepare_month(ftrain, ftrain, woa, grid, t, rng, args.density)
                 for t in range(len(ftrain["months"]))]
test_samples = [B.prepare_month(ftest, ftest, woa, grid, t, rng, args.density)
                for t in range(len(ftest["months"]))]
unobs = np.stack([s["unobs_mask"] for s in test_samples], 0)
obs_frac = 1.0 - unobs.sum() / (grid.ocean.sum() * len(test_samples))
print(f"  assembled in {time.time()-ts:.1f}s | obs-column frac excluded: {obs_frac:.3%}",
      flush=True)

results = []
level_tables = {}

def git_commit():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"

def record(method, preds):
    P = {v: np.stack([p[v] for p in preds], 0) for v in B.VARS}
    ev_un = metrics.evaluate_masked(P, TRUE, unobs, grid.depth)
    ev_ly = metrics.evaluate_layers(P, TRUE, unobs, grid.depth, LAYERS)
    row = {"seed": args.seed, "density": args.density, "method": method, "config": CFG_NAME,
           "overall": {v: ev_un["overall"][v] for v in B.VARS},
           "by_layer": {v: ev_ly["by_layer"][v] for v in B.VARS}}
    results.append(row)
    level_tables[method] = ev_un["by_depth"]
    lstr = "  ".join(f"{ln}:{ev_ly['by_layer']['TEMP'][ln]:.3f}" for ln, _, _ in LAYERS)
    print(f"  [{method:14s}] TEMP overall={row['overall']['TEMP']:.4f} | {lstr}", flush=True)
    flush_outputs()

def flush_outputs():
    run_cfg = {
        "git_commit": git_commit(), "smoke": args.smoke, "sweep_seed": args.seed,
        "split_seed": C.SEED, "density": args.density, "config": CFG_NAME,
        "depth_indices": EXTENDED_DEPTH_INDICES,
        "depths_m": [float(d) for d in grid.depth], "layers": LAYERS,
        "device": device, "train_months": tr_idx.tolist(), "test_months": te_idx.tolist(),
        "target": "train-only monthly CESM2 anomaly",
        "headline_metric": "unobserved-only RMSE (profile columns excluded)"}
    with open(os.path.join(C.CACHE, f"layered_depth_seed{args.seed}.json"), "w") as f:
        json.dump({"results": results, "run_config": run_cfg}, f, indent=2)
    np.savez(os.path.join(C.CACHE, f"layered_depth_seed{args.seed}_level.npz"),
             depths=grid.depth,
             **{f"{m}__{v}": lt[v] for m, lt in level_tables.items() for v in B.VARS})

print(f"\n== reference floors (seed {args.seed}) ==", flush=True)
record("woa_prior", [B.predict_climatology(s) for s in test_samples])
record("clim_floor", [B.predict_clim_floor(s, clim, grid) for s in test_samples])

print(f"\n== learned methods (seed {args.seed}) ==", flush=True)
record("mlp", B.train_predict_mlp(train_samples, test_samples, grid, norm, CFG, rng, device))
record("unet_depthwise", B.train_predict_unet(train_samples, test_samples, grid, norm, CFG,
                                              device, unobs_loss=True))
record("unet_joint", B.train_predict_unet_joint(train_samples, test_samples, grid, norm, CFG,
                                                device, unobs_loss=True))

print(f"\nDONE seed={args.seed} in {time.time()-t0:.1f}s", flush=True)
print(f"  -> outputs/cache/layered_depth_seed{args.seed}.json", flush=True)
