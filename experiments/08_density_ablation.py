"""Week-2: profile-density ablation with multiple random seeds.

Sweeps the number of synthetic Argo profiles per month (0 -> 3000) and retrains
each learned baseline at every density, under the corrected week-1 protocol
(train-only monthly anomaly target, unobserved-only loss and metric).  One
process handles one seed so seeds can run on separate GPUs.

Design choices (kept fixed so cells are comparable):

* The train/test month split is ALWAYS drawn with C.SEED (identical to the
  week-1 audit) — the sweep seed varies only the profile sampling, the MLP
  point subsampling, and the torch init/training stochasticity.  Seed spread
  therefore measures observation-sampling + training variance, not
  evaluation-set variance.
* Input config is fixed at profiles+WOA+SST/SSS (the headline config).  At
  density 0 the profile channels are present but empty — exactly what a fixed
  architecture sees when no profiles arrive.
* Training and test profile density match within a cell: every baseline is
  RETRAINED per density.  (The shared-latent method will later be one model
  evaluated across all densities — this sweep is its retrain-required
  contrast.)

Run (one seed per GPU):
    CUDA_VISIBLE_DEVICES=6 python experiments/08_density_ablation.py --seed 1234
    CUDA_VISIBLE_DEVICES=7 python experiments/08_density_ablation.py --seed 1235
"""
import sys, os, json, time, argparse, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
from ocean_tokenizer import data, baselines as B, metrics, config as C
from ocean_tokenizer.anomaly import Climatology, AnomNorm

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True,
                help="sweep seed (profile sampling + torch); month split stays C.SEED")
ap.add_argument("--densities", type=str, default="0,100,300,750,1500,3000")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
DENSITIES = [int(x) for x in args.densities.split(",")]

if args.smoke:
    C.N_TRAIN_MONTHS, C.N_TEST_MONTHS = 6, 3
    C.MLP_EPOCHS, C.UNET_EPOCHS = 4, 3
    C.UNET_JOINT_EPOCHS = 8
    C.MLP_POINTS_PER_MONTH = 30_000

CFG = ("profiles", "woa", "surf")          # headline input configuration
CFG_NAME = "profiles_woa_surf"

t0 = time.time()
device = C.DEVICE if torch.cuda.is_available() else "cpu"
print(f"seed={args.seed} device={device} densities={DENSITIES} smoke={args.smoke}",
      flush=True)

grid = data.CommonGrid()
print(grid, flush=True)

# ---- month split: ALWAYS C.SEED, identical to the week-1 audit ----
split_rng = np.random.default_rng(C.SEED)
tr_pool = data.select_month_indices(C.GT_SOURCE, C.TRAIN_YEARS)
te_pool = data.select_month_indices(C.GT_SOURCE, C.TEST_YEARS)
tr_idx = np.sort(split_rng.choice(tr_pool, size=min(C.N_TRAIN_MONTHS, tr_pool.size), replace=False))
te_idx = np.sort(split_rng.choice(te_pool, size=min(C.N_TEST_MONTHS, te_pool.size), replace=False))
print(f"train months={tr_idx.size} test months={te_idx.size}", flush=True)

print("loading GT fields ...", flush=True); ts = time.time()
ftrain = data.load_gt_fields(tr_idx, grid)
ftest = data.load_gt_fields(te_idx, grid)
woa = data.woa_prior(grid)
print(f"  loaded in {time.time()-ts:.1f}s", flush=True)

# ---- train-only climatology + anomaly normaliser (density/seed independent) ----
surf_train = {v: ftrain[v] for v in C.VARS_SURF if v in ftrain}
clim = Climatology(ftrain, surf_train)
norm = AnomNorm(clim, ftrain, surf_train)

TRUE = {v: ftest[v] for v in B.VARS}                       # (N,D,H,W)

def git_commit():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"

results = []
depth_tables = {}
out_json = os.path.join(C.CACHE, f"density_ablation_seed{args.seed}.json")
out_npz = os.path.join(C.CACHE, f"density_ablation_seed{args.seed}_depth.npz")

def flush_outputs():
    run_cfg = {
        "git_commit": git_commit(), "smoke": args.smoke,
        "sweep_seed": args.seed, "split_seed": C.SEED,
        "config": CFG_NAME, "densities": DENSITIES, "device": device,
        "train_months": tr_idx.tolist(), "test_months": te_idx.tolist(),
        "target": "train-only monthly CESM2 anomaly",
        "headline_metric": "unobserved-only RMSE (profile columns excluded)",
    }
    with open(out_json, "w") as f:
        json.dump({"results": results, "run_config": run_cfg}, f, indent=2)
    np.savez(out_npz, depths=grid.depth,
             **{f"{m}__{d}__{v}": dt[v]
                for (m, d), dt in depth_tables.items() for v in B.VARS})

def record(method, density, preds, unobs_masks):
    P = {v: np.stack([p[v] for p in preds], 0) for v in B.VARS}
    ev_all = metrics.evaluate(P, TRUE, grid.depth)
    ev_un = metrics.evaluate_masked(P, TRUE, unobs_masks, grid.depth)
    row = {"seed": args.seed, "density": density, "method": method, "config": CFG_NAME,
           "TEMP_unobs": ev_un["overall"]["TEMP"], "SALT_unobs": ev_un["overall"]["SALT"],
           "TEMP_all": ev_all["overall"]["TEMP"], "SALT_all": ev_all["overall"]["SALT"]}
    results.append(row)
    depth_tables[(method, density)] = ev_un["by_depth"]
    print(f"  [n={density:5d} | {method:14s}] unobs TEMP={row['TEMP_unobs']:.4f} "
          f"SALT={row['SALT_unobs']:.4f}", flush=True)
    flush_outputs()

for density in DENSITIES:
    tcell = time.time()
    # independent, reproducible randomness per (seed, density) cell
    rng = np.random.default_rng([args.seed, density])
    torch.manual_seed(args.seed * 100_003 + density)

    print(f"\n== density {density} (seed {args.seed}) ==", flush=True)
    train_samples = [B.prepare_month(ftrain, ftrain, woa, grid, t, rng, density)
                     for t in range(len(ftrain["months"]))]
    test_samples = [B.prepare_month(ftest, ftest, woa, grid, t, rng, density)
                    for t in range(len(ftest["months"]))]
    unobs = np.stack([s["unobs_mask"] for s in test_samples], 0)   # (N,H,W)
    obs_frac = 1.0 - unobs.sum() / (grid.ocean.sum() * len(test_samples))
    print(f"  observed-column fraction excluded: {obs_frac:.3%}", flush=True)

    # floors: no training, but re-scored on this cell's unobserved mask
    record("woa_prior", density, [B.predict_climatology(s) for s in test_samples], unobs)
    record("clim_floor", density, [B.predict_clim_floor(s, clim, grid) for s in test_samples], unobs)

    preds = B.train_predict_mlp(train_samples, test_samples, grid, norm, CFG, rng, device)
    record("mlp", density, preds, unobs)

    preds = B.train_predict_unet(train_samples, test_samples, grid, norm, CFG, device,
                                 unobs_loss=True)
    record("unet_depthwise", density, preds, unobs)

    preds = B.train_predict_unet_joint(train_samples, test_samples, grid, norm, CFG, device,
                                       unobs_loss=True)
    record("unet_joint", density, preds, unobs)

    del train_samples, test_samples, preds
    torch.cuda.empty_cache()
    print(f"  cell done in {time.time()-tcell:.1f}s", flush=True)

print(f"\nDONE seed={args.seed} in {time.time()-t0:.1f}s", flush=True)
print(f"  -> {out_json}\n  -> {out_npz}", flush=True)
