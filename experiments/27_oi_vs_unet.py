"""Phase-1 / Task 1.4 [Milestone M1] — OI vs the learned reconstructors.

Produces every number the M1 report needs, under protocol_v1, with **one
sample set shared by every method**: the 12 pinned test months, 1500 synthetic
profiles each.

Identical samples, not merely identical settings
------------------------------------------------
``13_joint_audit.py`` (which trained the certified U-Nets) draws profiles from
``default_rng(seed)`` in a fixed order: 276 train months, then 36 validation
months, then the 12 test months.  ``prepare_month`` touches the generator
exactly once per month, inside ``argo.sample_profiles``, via
``rng.choice(n_ocean, 1500, replace=False)`` — a draw that depends only on the
ocean mask, not on the field values.  Replaying 312 such draws therefore
reproduces the certified run's test-month profile positions *exactly*, so OI
and the U-Net are compared on the same observations rather than on two
independent draws.  ``--verify-unet`` asserts this by checking the checkpoint
reproduces its cached test RMSE.

Methods
-------
clim_floor · woa_prior · nearest · **oi** · depthwise U-Net (certified ckpt,
``profiles_woa_surf``).  The plan also wants a U-Net trained on
``profiles_only`` — the like-for-like information comparison against OI, which
sees profiles only.  That needs training (~14 GB of GPU tensors), so it lives
behind ``--train-profiles-only`` and is skipped when no GPU is free.

Metrics
-------
Unobserved-only anomaly RMSE, globally and over the North Atlantic box
(20-55 N, 275-335 E — Gulf Stream + subtropical gyre), full column and per
protocol_v1 depth band.  Per-cell error maps are cached for the figure script.

Run:
    python experiments/27_oi_vs_unet.py                       # CPU, ~10 min
    python experiments/27_oi_vs_unet.py --verify-unet         # + ckpt check
    CUDA_VISIBLE_DEVICES=N python experiments/27_oi_vs_unet.py \
        --train-profiles-only                                 # needs a free GPU
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
import torch

from ocean_tokenizer import baselines as B, config as C, data, metrics
from ocean_tokenizer.anomaly import AnomNorm, Climatology
from ocean_tokenizer.oi import predict_oi
from ocean_tokenizer.unet import UNet2D

TRAIN_YEARS = (1985, 2007)
VAL_YEARS = (2008, 2010)
TEST_TIME_INDICES = [1933, 1935, 1936, 1938, 1942, 1946, 1952, 1953,
                     1956, 1965, 1967, 1976]
BANDS = [("0-100m", 0.0, 100.0), ("100-300m", 100.0, 300.0),
         ("300-max", 300.0, 1e9)]
CFG_PWS = ("profiles", "woa", "surf")
CFG_PONLY = ("profiles",)
VARS = B.VARS
CERT_CKPT = "audit_depthwise_e40"          # certified depthwise U-Net (week-3)

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--n-profiles", type=int, default=C.N_PROFILES)
ap.add_argument("--oi-tuning", default=None,
                help="tuning json (default: outputs/cache/oi_tuning_val.json)")
ap.add_argument("--L", type=float, default=None, help="override tuned L_km")
ap.add_argument("--gamma", type=float, default=None, help="override tuned gamma")
ap.add_argument("--k", type=int, default=None, help="override tuned k")
ap.add_argument("--verify-unet", action="store_true",
                help="assert the ckpt reproduces its cached test RMSE")
ap.add_argument("--train-profiles-only", action="store_true",
                help="also train a profiles_only U-Net (needs a free GPU)")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

t0 = time.time()
grid = data.CommonGrid()
print(f"grid={grid}", flush=True)

# ---------------- frozen OI hyperparameters ----------------
tune_path = args.oi_tuning or os.path.join(C.CACHE, "oi_tuning_val.json")
if os.path.exists(tune_path):
    tuned = json.load(open(tune_path))["best"]
    OI_HP = {v: {"L_km": tuned[v]["L_km"], "gamma": tuned[v]["gamma"],
                 "k": tuned[v]["k"]} for v in VARS}
    oi_src = os.path.basename(tune_path)
else:
    OI_HP = {v: {"L_km": 500.0, "gamma": 0.1, "k": 20} for v in VARS}
    oi_src = "defaults (tuning json absent)"
for v in VARS:                                   # CLI overrides
    if args.L is not None:
        OI_HP[v]["L_km"] = args.L
    if args.gamma is not None:
        OI_HP[v]["gamma"] = args.gamma
    if args.k is not None:
        OI_HP[v]["k"] = args.k
print(f"OI hyperparameters from {oi_src}: {json.dumps(OI_HP)}", flush=True)

# ---------------- fields ----------------
tr_idx = data.select_month_indices(C.GT_SOURCE, TRAIN_YEARS)
va_idx = data.select_month_indices(C.GT_SOURCE, VAL_YEARS)
te_idx = np.asarray(TEST_TIME_INDICES)
if args.smoke:
    te_idx = te_idx[:2]
print(f"loading fields (train={tr_idx.size} test={te_idx.size}) ...", flush=True)
ts = time.time()
ftrain = data.load_gt_fields(tr_idx, grid)
ftest = data.load_gt_fields(te_idx, grid)
woa = data.woa_prior(grid)
surf_train = {v: ftrain[v] for v in C.VARS_SURF if v in ftrain}
clim = Climatology(ftrain, surf_train)
norm = AnomNorm(clim, ftrain, surf_train)
print(f"  loaded + climatology in {time.time()-ts:.0f}s", flush=True)

# ---------------- replay the certified run's RNG, then build test samples ----
rng = np.random.default_rng(args.seed)
n_ocean = int(grid.ocean.sum())
n_skip = tr_idx.size + va_idx.size                    # 276 + 36
for _ in range(n_skip):                               # identical draws, discarded
    rng.choice(n_ocean, size=min(args.n_profiles, n_ocean), replace=False)
print(f"replayed {n_skip} profile draws to reach the certified test-month state",
      flush=True)
te_samples = [B.prepare_month(ftest, ftest, woa, grid, t, rng, args.n_profiles)
              for t in range(len(ftest["months"]))]
print(f"test months: {list(ftest['time'])}", flush=True)

TRUE = {v: np.stack([s["gt"][v] for s in te_samples], 0) for v in VARS}
UNOBS = np.stack([s["unobs_mask"] for s in te_samples], 0)          # (N,H,W)

# North Atlantic box (Gulf Stream + subtropical gyre)
lat2d, lon2d = np.meshgrid(grid.lat, grid.lon, indexing="ij")
NA_BOX = (lat2d >= 20) & (lat2d <= 55) & (lon2d >= 275) & (lon2d <= 335)
UNOBS_NA = UNOBS & NA_BOX[None]
print(f"North Atlantic box: {int(NA_BOX.sum())} cells, "
      f"{int(UNOBS_NA.sum()/len(te_samples))} scored per month", flush=True)


def stack(preds):
    return {v: np.stack([p[v] for p in preds], 0) for v in VARS}


# ---------------- the certified depthwise U-Net, inference only --------------
def unet_certified():
    ck_path = os.path.join(C.CKPT, f"{CERT_CKPT}.pt")
    ck = torch.load(ck_path, map_location="cpu")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X0 = B._unet_channels(te_samples[0], grid, norm, CFG_PWS)
    model = UNet2D(X0.shape[1], len(VARS), base=C.UNET_BASE).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"  loaded {CERT_CKPT} (epoch {ck['epoch']}, c_in={X0.shape[1]}) on {dev}",
          flush=True)
    preds = []
    with torch.no_grad():
        for s in te_samples:
            X = torch.from_numpy(B._unet_channels(s, grid, norm, CFG_PWS)).to(dev)
            out = model(X).cpu().numpy()                    # (D,2,H,W)
            preds.append({v: np.where(grid.ocean[None],
                                      norm.unz3d(v, out[:, k], s["month"]),
                                      np.nan).astype("float32")
                          for k, v in enumerate(VARS)})
    return preds


def unet_profiles_only():
    """Train the like-for-like row: a U-Net that, like OI, sees only profiles."""
    dev = C.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"  training profiles_only depthwise U-Net on {dev} "
          f"({C.UNET_EPOCHS} epochs) ...", flush=True)
    tr_rng = np.random.default_rng(args.seed)
    tr_samples = [B.prepare_month(ftrain, ftrain, woa, grid, t, tr_rng,
                                  args.n_profiles)
                  for t in range(len(ftrain["months"]))]
    torch.manual_seed(args.seed)
    return B.train_predict_unet(tr_samples, te_samples, grid, norm, CFG_PONLY,
                                dev, unobs_loss=True)


# ---------------- run every method ----------------
P = {}
tm = time.time()
P["clim_floor"] = stack([B.predict_clim_floor(s, clim, grid) for s in te_samples])
P["woa_prior"] = stack([B.predict_climatology(s) for s in te_samples])
P["nearest"] = stack([B.predict_nearest(s, use_woa=True) for s in te_samples])
print(f"analytic baselines in {time.time()-tm:.0f}s", flush=True)

tm = time.time()
P["oi"] = stack([predict_oi(s, norm, grid,
                            L_km={v: OI_HP[v]["L_km"] for v in VARS},
                            gamma={v: OI_HP[v]["gamma"] for v in VARS},
                            k={v: OI_HP[v]["k"] for v in VARS})
                 for s in te_samples])
print(f"OI in {time.time()-tm:.0f}s", flush=True)

tm = time.time()
P["unet_depthwise_pws"] = stack(unet_certified())
print(f"certified U-Net inference in {time.time()-tm:.0f}s", flush=True)

if args.train_profiles_only:
    tm = time.time()
    P["unet_depthwise_profiles_only"] = stack(unet_profiles_only())
    print(f"profiles_only U-Net in {time.time()-tm:.0f}s", flush=True)

# ---------------- verify the checkpoint reproduces its certified number ------
CERT = json.load(open(os.path.join(C.CACHE, f"{CERT_CKPT}.json")))["test"]
ev_chk = metrics.evaluate_masked(P["unet_depthwise_pws"], TRUE, UNOBS, grid.depth)
delta = {v: ev_chk["overall"][v] - CERT[v] for v in VARS}
print(f"\ncertified-checkpoint reproduction: "
      f"TEMP {ev_chk['overall']['TEMP']:.5f} vs {CERT['TEMP']:.5f} "
      f"(d={delta['TEMP']:+.1e}), "
      f"SALT {ev_chk['overall']['SALT']:.5f} vs {CERT['SALT']:.5f} "
      f"(d={delta['SALT']:+.1e})", flush=True)
if args.verify_unet and not args.smoke:
    for v in VARS:
        assert abs(delta[v]) < 1e-4, (
            f"{v} deviates from the certified number by {delta[v]:.2e}: the "
            f"RNG replay or the inference path does not match 13_joint_audit.py")
    print("  VERIFIED: identical samples + identical inference path", flush=True)


# ---------------- metrics ----------------
def region_metrics(mask):
    out = {}
    for m, pred in P.items():
        ev = metrics.evaluate_masked(pred, TRUE, mask, grid.depth)
        evb = metrics.evaluate_layers(pred, TRUE, mask, grid.depth, BANDS)
        out[m] = {"full": {v: ev["overall"][v] for v in VARS},
                  "by_band": {v: evb["by_layer"][v] for v in VARS},
                  "by_depth": {v: ev["by_depth"][v].tolist() for v in VARS}}
    return out


results = {"global": region_metrics(UNOBS), "north_atlantic": region_metrics(UNOBS_NA)}

# per-cell RMSE maps (for the error-map figure)
maps = {}
for m, pred in P.items():
    for v in VARS:
        err = pred[v] - TRUE[v]                                  # (N,D,H,W)
        keep = np.broadcast_to(UNOBS[:, None], err.shape) & np.isfinite(err)
        se = np.where(keep, err ** 2, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            maps[f"{m}_{v}"] = np.sqrt(np.nanmean(se, axis=(0, 1))).astype("float32")


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


out = {
    "task": "oi_vs_unet", "protocol": "protocol_v1", "smoke": args.smoke,
    "git_commit": git_commit(), "seed": args.seed,
    "n_profiles": args.n_profiles,
    "oi_hyperparams": OI_HP, "oi_tuning_source": oi_src,
    "train_months": int(tr_idx.size),
    "test_months": [str(x) for x in ftest["time"]],
    "rng_draws_skipped": n_skip,
    "certified_ckpt": CERT_CKPT,
    "certified_reproduction": {v: {"got": ev_chk["overall"][v],
                                   "cached": CERT[v], "delta": delta[v]}
                               for v in VARS},
    "na_box": {"lat": [20, 55], "lon": [275, 335],
               "cells": int(NA_BOX.sum())},
    "results": results,
    "cpu_hours": round((time.time() - t0) / 3600, 3),
}
suffix = "_smoke" if args.smoke else ""
path = os.path.join(C.CACHE, f"oi_vs_unet_seed{args.seed}{suffix}.json")
json.dump(out, open(path, "w"), indent=2)
np.savez_compressed(
    os.path.join(C.CACHE, f"oi_vs_unet_seed{args.seed}{suffix}_maps.npz"), **maps)

print(f"\nDONE in {(time.time()-t0)/60:.1f} min -> {path}", flush=True)
print(f"\n{'method':32s} {'TEMP':>8s} {'SALT':>8s} | {'NA TEMP':>8s} {'NA SALT':>8s}")
for m in P:
    g, na = results["global"][m]["full"], results["north_atlantic"][m]["full"]
    print(f"{m:32s} {g['TEMP']:8.4f} {g['SALT']:8.4f} | "
          f"{na['TEMP']:8.4f} {na['SALT']:8.4f}")
