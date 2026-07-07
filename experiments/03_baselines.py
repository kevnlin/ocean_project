"""Baseline sweep: methods x configs -> RMSE by variable and by depth.

Methods : climatology, nearest, mlp, unet
Configs : profiles_only, woa_only, profiles_woa, profiles_woa_surf
GT      : CESM2-LE full simulation (held-out test months)
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
from ocean_tokenizer import data, baselines as B, metrics, config as C

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

if args.smoke:
    C.N_TRAIN_MONTHS, C.N_TEST_MONTHS = 6, 3
    C.N_PROFILES = 600
    C.MLP_EPOCHS, C.UNET_EPOCHS = 4, 3
    C.MLP_POINTS_PER_MONTH = 30_000

t0 = time.time()
rng = np.random.default_rng(C.SEED)
device = C.DEVICE if torch.cuda.is_available() else "cpu"
print(f"device={device} smoke={args.smoke}")

grid = data.CommonGrid()
print(grid)

# ---- pick train/test months ----
tr_pool = data.select_month_indices(C.GT_SOURCE, C.TRAIN_YEARS)
te_pool = data.select_month_indices(C.GT_SOURCE, C.TEST_YEARS)
tr_idx = np.sort(rng.choice(tr_pool, size=min(C.N_TRAIN_MONTHS, tr_pool.size), replace=False))
te_idx = np.sort(rng.choice(te_pool, size=min(C.N_TEST_MONTHS, te_pool.size), replace=False))
print(f"train months={tr_idx.size} test months={te_idx.size}")

# ---- load fields ----
print("loading GT fields ..."); ts = time.time()
ftrain = data.load_gt_fields(tr_idx, grid)
ftest = data.load_gt_fields(te_idx, grid)
woa = data.woa_prior(grid)
print(f"  loaded in {time.time()-ts:.1f}s")

norm = B.Norm(ftrain, ftrain)

# ---- assemble per-month samples ----
def make_samples(fields):
    out = []
    for t in range(len(fields["months"])):
        out.append(B.prepare_month(fields, fields, woa, grid, t, rng, C.N_PROFILES))
    return out

print("assembling samples ..."); ts = time.time()
train_samples = make_samples(ftrain)
test_samples = make_samples(ftest)
print(f"  assembled in {time.time()-ts:.1f}s")

# ground-truth test stacks
TRUE = {v: np.stack([s["gt"][v] for s in test_samples], 0) for v in B.VARS}  # (N,D,H,W)

def stack_pred(preds):
    return {v: np.stack([p[v] for p in preds], 0) for v in B.VARS}

results = []      # rows: method, config, RMSE_TEMP, RMSE_SALT
depth_tables = {} # key (method,config) -> {var: by_depth array}

def record(method, cfg, preds):
    P = stack_pred(preds)
    ev = metrics.evaluate(P, TRUE, grid.depth)
    row = {"method": method, "config": cfg,
           "RMSE_TEMP": ev["overall"]["TEMP"], "RMSE_SALT": ev["overall"]["SALT"]}
    results.append(row)
    depth_tables[(method, cfg)] = ev["by_depth"]
    print(f"  [{method:11s} | {cfg:18s}] TEMP={row['RMSE_TEMP']:.4f}  SALT={row['RMSE_SALT']:.4f}")

CONFIGS = {
    "profiles_only":     ("profiles",),
    "woa_only":          ("woa",),
    "profiles_woa":      ("profiles", "woa"),
    "profiles_woa_surf": ("profiles", "woa", "surf"),
}

# ============================ run methods ============================
print("\n== climatology ==")
record("climatology", "woa_only", [B.predict_climatology(s) for s in test_samples])

print("\n== nearest ==")
record("nearest", "profiles_only", [B.predict_nearest(s, use_woa=False) for s in test_samples])
record("nearest", "profiles_woa", [B.predict_nearest(s, use_woa=True) for s in test_samples])

print("\n== mlp ==")
for name, cfg in CONFIGS.items():
    preds = B.train_predict_mlp(train_samples, test_samples, grid, norm, cfg, rng, device)
    record("mlp", name, preds)

print("\n== unet ==")
for name, cfg in CONFIGS.items():
    preds = B.train_predict_unet(train_samples, test_samples, grid, norm, cfg, device)
    record("unet", name, preds)

# ============================ save ============================
os.makedirs(C.CACHE, exist_ok=True)
np.savez(os.path.join(C.CACHE, "baseline_depth_tables.npz"),
         depths=grid.depth,
         **{f"{m}__{c}__{v}": dt[v]
            for (m, c), dt in depth_tables.items() for v in B.VARS})
with open(os.path.join(C.CACHE, "baseline_results.json"), "w") as f:
    json.dump({"results": results, "depths": grid.depth.tolist(),
               "n_train": int(tr_idx.size), "n_test": int(te_idx.size),
               "n_profiles": int(C.N_PROFILES)}, f, indent=2)
print(f"\nDONE in {time.time()-t0:.1f}s -> outputs/cache/baseline_results.json")
