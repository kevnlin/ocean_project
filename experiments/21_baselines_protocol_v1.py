"""Task 9a — protocol_v1 non-neural baseline rows for the headline table.

The Week-4 report compares the fusion variants against the climatology floor
and the two *certified* U-Nets (audit_depthwise_e40 / audit_joint_e400_cos),
but the multi-seed headline the plan calls for also lists the **pointwise MLP**
(and the nearest-profile fill).  Those were only ever computed under the
pre-protocol week-1 setup (312 random train months, no val split); this script
recomputes them under protocol_v1 exactly:

  * 276 contiguous train months (1985-2007), the 12 pinned test months;
  * train-only climatology + anomaly z-stats; unobserved-only anomaly RMSE;
  * config ``profiles_woa_surf`` (the fusion variants' inputs), 1500 profiles;
  * seeds {1234, 1235, 1236} -> mean +/- std, per variable + per band.

The MLP has no validation-based model selection (fixed 30-epoch schedule), so
the 36 val months are simply unused here — no protocol contact.

Run:  CUDA_VISIBLE_DEVICES=6 python experiments/21_baselines_protocol_v1.py
"""
import sys, os, json, time, argparse, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from ocean_tokenizer import data, baselines as B, metrics, config as C
from ocean_tokenizer.anomaly import Climatology, AnomNorm

TRAIN_YEARS = (1985, 2007)          # 276 months (protocol_v1)
TEST_TIME_INDICES = [1933, 1935, 1936, 1938, 1942, 1946, 1952, 1953,
                     1956, 1965, 1967, 1976]
BANDS = [("0-100m", 0.0, 100.0), ("100-300m", 100.0, 300.0),
         ("300-max", 300.0, 1e9)]
CFG = ("profiles", "woa", "surf")
CFG_NAME = "profiles_woa_surf"
VARS = B.VARS

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="1234,1235,1236")
ap.add_argument("--methods", default="clim_floor,nearest,mlp",
                help="subset of clim_floor,nearest,mlp")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(",")]
METHODS = args.methods.split(",")
if args.smoke:
    C.MLP_EPOCHS = 3
    C.MLP_POINTS_PER_MONTH = 30_000

t0 = time.time()
device = C.DEVICE if torch.cuda.is_available() else "cpu"
grid = data.CommonGrid()
print(f"device={device} grid={grid}", flush=True)

tr_idx = data.select_month_indices(C.GT_SOURCE, TRAIN_YEARS)
te_idx = np.asarray(TEST_TIME_INDICES)
if args.smoke:
    tr_idx = tr_idx[:8]
print(f"train={tr_idx.size} test={te_idx.size}", flush=True)

print("loading fields ...", flush=True); ts = time.time()
ftrain = data.load_gt_fields(tr_idx, grid)
ftest = data.load_gt_fields(te_idx, grid)
woa = data.woa_prior(grid)
print(f"  loaded in {time.time()-ts:.1f}s", flush=True)

surf_train = {v: ftrain[v] for v in C.VARS_SURF if v in ftrain}
clim = Climatology(ftrain, surf_train)
norm = AnomNorm(clim, ftrain, surf_train)


def band_eval(P, TRUE, UNOBS):
    ev = metrics.evaluate_masked(P, TRUE, UNOBS, grid.depth)
    evb = metrics.evaluate_layers(P, TRUE, UNOBS, grid.depth, BANDS)
    return ({v: ev["overall"][v] for v in VARS},
            {v: evb["by_layer"][v] for v in VARS})


per_seed = {m: [] for m in METHODS}
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    tr_samples = [B.prepare_month(ftrain, ftrain, woa, grid, t, rng, C.N_PROFILES)
                  for t in range(len(ftrain["months"]))]
    te_samples = [B.prepare_month(ftest, ftest, woa, grid, t, rng, C.N_PROFILES)
                  for t in range(len(ftest["months"]))]
    TRUE = {v: np.stack([s["gt"][v] for s in te_samples], 0) for v in VARS}
    UNOBS = np.stack([s["unobs_mask"] for s in te_samples], 0)

    def stack(preds):
        return {v: np.stack([p[v] for p in preds], 0) for v in VARS}

    for m in METHODS:
        tm = time.time()
        if m == "clim_floor":
            P = stack([B.predict_clim_floor(s, clim, grid) for s in te_samples])
        elif m == "nearest":
            P = stack([B.predict_nearest(s, use_woa=True) for s in te_samples])
        elif m == "mlp":
            P = stack(B.train_predict_mlp(tr_samples, te_samples, grid, norm,
                                          CFG, rng, device))
        else:
            raise ValueError(m)
        full, bands = band_eval(P, TRUE, UNOBS)
        per_seed[m].append({"seed": seed, "full": full, "by_band": bands})
        print(f"  seed {seed} [{m:10s}] TEMP={full['TEMP']:.4f} "
              f"SALT={full['SALT']:.4f} ({time.time()-tm:.0f}s)", flush=True)


def agg(m, var):
    vals = [r["full"][var] for r in per_seed[m]]
    return float(np.mean(vals)), float(np.std(vals))


def agg_band(m, var, band):
    vals = [r["by_band"][var][band] for r in per_seed[m]]
    return float(np.mean(vals))


summary = {}
for m in METHODS:
    summary[m] = {
        "TEMP_mean": agg(m, "TEMP")[0], "TEMP_std": agg(m, "TEMP")[1],
        "SALT_mean": agg(m, "SALT")[0], "SALT_std": agg(m, "SALT")[1],
        "by_band": {v: {b[0]: agg_band(m, v, b[0]) for b in BANDS}
                    for v in VARS}}


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


out = {"task": "baselines_protocol_v1", "protocol": "protocol_v1",
       "config": CFG_NAME, "git_commit": git_commit(), "smoke": args.smoke,
       "seeds": SEEDS, "n_profiles": C.N_PROFILES,
       "train_months": int(tr_idx.size), "test_months": te_idx.tolist(),
       "per_seed": per_seed, "summary": summary,
       "gpu_hours": round((time.time() - t0) / 3600, 3)}
path = os.path.join(C.CACHE, "baselines_protocol_v1.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nDONE in {(time.time()-t0)/60:.1f} min -> {path}", flush=True)
for m in METHODS:
    s = summary[m]
    print(f"  {m:10s}: TEMP={s['TEMP_mean']:.4f}+/-{s['TEMP_std']:.4f} "
          f"SALT={s['SALT_mean']:.4f}+/-{s['SALT_std']:.4f}", flush=True)
