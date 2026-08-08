"""Phase-1 / Task 1.3 — tune the optimal-interpolation baseline (L, gamma, k).

Protocol
--------
Hyperparameter selection is **model selection**, so under protocol_v1 it uses
the *validation* months (2008-2010) — not the training months and never the 12
pinned test months.  (The intern plan said "training months only"; protocol_v1
is the stricter rule and the repo's frozen protocol wins.  ``--split train``
reruns the identical sweep on training months as a stability check: if the
optimum moves between the two, say so in the report.)

Sweep
-----
Stage A : L_km x gamma at fixed k, pooled over ``--months`` snapshots.
Stage B : k at the stage-A optimum (per variable — TEMP and SALT need not agree).

Both stages pool squared errors over every scored cell of every month before
taking the root, which is exactly what ``metrics.evaluate_masked`` does over a
stacked array, so these numbers are directly comparable to the headline table.

Efficiency: the kd-tree query and the great-circle distance matrices are ~75 %
of an analysis and do not depend on (L, gamma), so ``oi.LevelSweep`` builds
them once per (month, variable, depth) and every (L, gamma) reuses them.
Stage B builds the largest k once and slices the smaller ones out of it.

Run:
    python experiments/26_oi_tuning.py                     # full sweep, ~1 h CPU
    python experiments/26_oi_tuning.py --smoke             # 1 month, 2x2 grid
    python experiments/26_oi_tuning.py --split train       # stability check
"""
import argparse
import json
import os
import subprocess
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np

from ocean_tokenizer import baselines as B, config as C, data
from ocean_tokenizer.anomaly import AnomNorm, Climatology
from ocean_tokenizer.oi import LevelSweep

TRAIN_YEARS = (1985, 2007)          # 276 months (protocol_v1)
VAL_YEARS = (2008, 2010)            # 36 months  (protocol_v1 model selection)
VARS = B.VARS

# Six snapshots spread over the calendar year and over distinct years, so the
# chosen (L, gamma) is not tuned to one season's mixed-layer depth.
VAL_PICKS = [0, 4, 8, 14, 18, 22]     # 2008-01, -05, -09, 2009-03, -07, -11
TRAIN_PICKS = [252, 256, 260, 266, 270, 274]   # 2006-01, -05, -09, 2007-03, -07, -11

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val", choices=("val", "train"),
                help="months the sweep is scored on (protocol_v1: val)")
ap.add_argument("--seed", type=int, default=1234, help="profile-sampling seed")
ap.add_argument("--months", type=int, default=6)
ap.add_argument("--n-profiles", type=int, default=C.N_PROFILES)
ap.add_argument("--L", default="300,500,800,1200")
ap.add_argument("--gamma", default="0.03,0.1,0.3")
ap.add_argument("--k-stage-a", type=int, default=20)
ap.add_argument("--k-sweep", default="10,20,40")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

L_GRID = [float(x) for x in args.L.split(",")]
G_GRID = [float(x) for x in args.gamma.split(",")]
K_GRID = [int(x) for x in args.k_sweep.split(",")]
if args.smoke:
    L_GRID, G_GRID, K_GRID = [500.0, 800.0], [0.1, 0.3], [10, 20]
    args.months = 1

t0 = time.time()
grid = data.CommonGrid()
print(f"grid={grid}", flush=True)

# The climatology and the anomaly z-scores must come from the 276 training
# months regardless of which split we score on — that is what makes "predict
# zero anomaly" the reported floor.
tr_idx = data.select_month_indices(C.GT_SOURCE, TRAIN_YEARS)
print(f"loading {tr_idx.size} train months for climatology/z-stats ...", flush=True)
ts = time.time()
ftrain = data.load_gt_fields(tr_idx, grid)
surf_train = {v: ftrain[v] for v in C.VARS_SURF if v in ftrain}
clim = Climatology(ftrain, surf_train)
anorm = AnomNorm(clim, ftrain, surf_train)
woa = data.woa_prior(grid)
print(f"  climatology built in {time.time()-ts:.0f}s", flush=True)

if args.split == "val":
    sweep_idx = data.select_month_indices(C.GT_SOURCE, VAL_YEARS)
    picks = VAL_PICKS[:args.months]
    fsweep = data.load_gt_fields(sweep_idx, grid)
else:
    picks = TRAIN_PICKS[:args.months]
    fsweep = ftrain
del ftrain

rng = np.random.default_rng(args.seed)
samples = [B.prepare_month(fsweep, fsweep, woa, grid, t, rng, args.n_profiles)
           for t in picks]
print(f"split={args.split} months={[fsweep['time'][t] for t in picks]}", flush=True)

lat2d, lon2d = np.meshgrid(grid.lat, grid.lon, indexing="ij")


def sweep(combos, k_of):
    """Pooled unobserved-only RMSE for each combo.

    ``combos``  : list of hashable keys
    ``k_of``    : key -> k, so one geometry can serve several combos
    Returns {var: {key: rmse}}.
    """
    kmax = max(k_of(c) for c in combos)
    sse = {v: {c: 0.0 for c in combos} for v in VARS}
    cnt = {v: {c: 0 for c in combos} for v in VARS}
    for si, s in enumerate(samples):
        mo = s["month"]
        score = s["unobs_mask"]                          # (H,W) unobserved ocean
        for v in VARS:
            obs_z = anorm.z3d(v, s["obs"][v], mo)        # (D,H,W)
            gt = s["gt"][v]
            zacc = {c: np.zeros((grid.ndepth, grid.nlat, grid.nlon), "float64")
                    for c in combos}
            for d in range(grid.ndepth):
                m = np.isfinite(obs_z[d])
                geom = LevelSweep(lat2d[m], lon2d[m], lat2d, lon2d,
                                  grid.ocean, k=kmax)
                oval = obs_z[d][m]
                cache = {}
                for c in combos:
                    kc = k_of(c)
                    if kc not in cache:
                        cache[kc] = geom.sub_k(kc)
                    zacc[c][d] = cache[kc].analyse(oval, c[0], c[1])
            for c in combos:
                pred = anorm.unz3d(v, zacc[c].astype("float32"), mo)
                err = pred - gt                                   # (D,H,W)
                keep = np.broadcast_to(score[None], err.shape) & np.isfinite(err)
                e = err[keep]
                sse[v][c] += float(np.dot(e, e))
                cnt[v][c] += e.size
        print(f"  [{args.split}] month {si+1}/{len(samples)} "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)
    return {v: {c: float(np.sqrt(sse[v][c] / cnt[v][c])) for c in combos}
            for v in VARS}


# ---------------- Stage A: L x gamma at fixed k ----------------
print(f"\nStage A: L={L_GRID} x gamma={G_GRID} at k={args.k_stage_a}", flush=True)
combos_a = [(L, g) for L in L_GRID for g in G_GRID]
res_a = sweep(combos_a, k_of=lambda c: args.k_stage_a)

best = {}
for v in VARS:
    bc = min(res_a[v], key=res_a[v].get)
    best[v] = {"L_km": bc[0], "gamma": bc[1], "rmse": res_a[v][bc]}
    print(f"  best {v}: L={bc[0]:.0f} gamma={bc[1]} -> {res_a[v][bc]:.4f}", flush=True)

# ---------------- Stage B: k at the stage-A optimum ----------------
print(f"\nStage B: k={K_GRID} at the stage-A optimum", flush=True)
combos_b = sorted({(best[v]["L_km"], best[v]["gamma"], k)
                   for v in VARS for k in K_GRID})
res_b_raw = sweep(combos_b, k_of=lambda c: c[2])
res_b = {v: {c: r for c, r in res_b_raw[v].items()
             if c[0] == best[v]["L_km"] and c[1] == best[v]["gamma"]}
         for v in VARS}
for v in VARS:
    bc = min(res_b[v], key=res_b[v].get)
    best[v]["k"] = bc[2]
    best[v]["rmse"] = res_b[v][bc]
    print(f"  best {v}: L={bc[0]:.0f} gamma={bc[1]} k={bc[2]} -> {res_b[v][bc]:.4f}",
          flush=True)


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


out = {
    "task": "oi_tuning", "protocol": "protocol_v1",
    "selection_split": args.split, "smoke": args.smoke,
    "git_commit": git_commit(), "seed": args.seed,
    "n_profiles": args.n_profiles,
    "months": [str(fsweep["time"][t]) for t in picks],
    "k_stage_a": args.k_stage_a,
    "stage_a": {v: [{"L_km": c[0], "gamma": c[1], "rmse": r}
                    for c, r in res_a[v].items()] for v in VARS},
    "stage_b": {v: [{"L_km": c[0], "gamma": c[1], "k": c[2], "rmse": r}
                    for c, r in res_b[v].items()] for v in VARS},
    "best": best,
    "cpu_hours": round((time.time() - t0) / 3600, 3),
}
suffix = "_smoke" if args.smoke else ""
path = os.path.join(C.CACHE, f"oi_tuning_{args.split}{suffix}.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nDONE in {(time.time()-t0)/60:.1f} min -> {path}", flush=True)
print("FROZEN HYPERPARAMETERS:", json.dumps(best), flush=True)
