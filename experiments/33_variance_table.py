"""RMSE + target variance per level/band — the numbers an external group needs
to normalise our skill against theirs.

Motivation: a collaborator asked for "RMSE and target variance for SST and SSH".
Our system does not predict either -- it predicts TEMP/SALT as 3-D anomaly
fields, while SST/SSS and the pseudo-SSH are *inputs*.  What is well defined,
and what a normalised comparison actually needs, is:

  * RMSE of the reconstruction,
  * the variance of the target it is trying to explain,
  * explained variance = 1 - MSE/Var,

reported per depth level (so the surface level, 5 m, is the SST analogue), per
band, and full column.

Two variances are reported because "target variance" is ambiguous:

  var0  = mean(a^2)          -- second moment about ZERO.  This is the one that
                                matters here: the protocol's floor is "predict
                                zero anomaly", so RMSE_floor = sqrt(var0) exactly,
                                and 1 - MSE/var0 is the skill we report.
  varm  = mean((a-abar)^2)   -- variance about the target's own mean.  Differs
                                from var0 only by the squared spatial-mean
                                anomaly, which is small but not zero.

All statistics are over the protocol_v1 scoring set: the 12 pinned test months,
unobserved ocean cells only (profile columns excluded at every level).

Run:  python experiments/33_variance_table.py
"""
import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np
import torch

from ocean_tokenizer import baselines as B, config as C, data
from ocean_tokenizer.anomaly import AnomNorm, Climatology
from ocean_tokenizer.oi import predict_oi
from ocean_tokenizer.ssh import SSHAnom, load_ssh_cache, ssh_for_indices
from ocean_tokenizer.unet import UNet2D

TRAIN_YEARS, VAL_YEARS = (1985, 2007), (2008, 2010)
TEST_TIME_INDICES = [1933, 1935, 1936, 1938, 1942, 1946, 1952, 1953,
                     1956, 1965, 1967, 1976]
CFG_PWS = ("profiles", "woa", "surf")
CFG_SSH = ("profiles", "woa", "surf", "ssh")
BANDS = [("0-100m", 0.0, 100.0), ("100-300m", 100.0, 300.0),
         ("300-max", 300.0, 1e9)]

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--device", default="cpu")
args = ap.parse_args()

grid = data.CommonGrid()
tr_idx = data.select_month_indices(C.GT_SOURCE, TRAIN_YEARS)
va_idx = data.select_month_indices(C.GT_SOURCE, VAL_YEARS)
te_idx = np.asarray(TEST_TIME_INDICES)
print(f"loading fields ({tr_idx.size} train) ...", flush=True)
ftrain = data.load_gt_fields(tr_idx, grid)
ftest = data.load_gt_fields(te_idx, grid)
woa = data.woa_prior(grid)
surf_train = {v: ftrain[v] for v in C.VARS_SURF if v in ftrain}
clim = Climatology(ftrain, surf_train)
norm = AnomNorm(clim, ftrain, surf_train)

rng = np.random.default_rng(args.seed)
n_ocean = int(grid.ocean.sum())
for _ in range(tr_idx.size + va_idx.size):
    rng.choice(n_ocean, size=min(C.N_PROFILES, n_ocean), replace=False)
te = [B.prepare_month(ftest, ftest, woa, grid, t, rng, C.N_PROFILES)
      for t in range(len(ftest["months"]))]

ssh_all, ssh_idx = load_ssh_cache(os.path.join(C.CACHE, "ssh_dyn.npz"))
sshnorm = SSHAnom(ssh_for_indices(ssh_all, ssh_idx, tr_idx), ftrain["months"])
ssh_te = ssh_for_indices(ssh_all, ssh_idx, te_idx)
for k, s in enumerate(te):
    s["ssh_z"] = sshnorm.z(ssh_te[k], s["month"])


def load_unet(tag, cfg):
    ck = torch.load(os.path.join(C.CKPT, f"{tag}.pt"), map_location="cpu",
                    weights_only=False)
    c_in = B._unet_channels(te[0], grid, norm, cfg).shape[1]
    m = UNet2D(c_in, len(B.VARS), base=C.UNET_BASE).to(args.device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


@torch.no_grad()
def unet_pred(model, cfg):
    out = {v: [] for v in B.VARS}
    for s in te:
        X = torch.from_numpy(B._unet_channels(s, grid, norm, cfg)).to(args.device)
        o = model(X).cpu().numpy()
        for k, v in enumerate(B.VARS):
            out[v].append(np.where(grid.ocean[None],
                                   norm.unz3d(v, o[:, k], s["month"]), np.nan))
    return {v: np.stack(out[v], 0).astype("float32") for v in B.VARS}


hp = json.load(open(os.path.join(C.CACHE, "oi_tuning_val.json")))["best"]
MODELS = {}
print("running models ...", flush=True)
MODELS["unet_ssh"] = unet_pred(load_unet(f"ssh_treat_pws_ssh_s{args.seed}", CFG_SSH),
                               CFG_SSH)
MODELS["unet_certified"] = unet_pred(load_unet("audit_depthwise_e40", CFG_PWS),
                                     CFG_PWS)
oi = [predict_oi(s, norm, grid,
                 L_km={v: hp[v]["L_km"] for v in B.VARS},
                 gamma={v: hp[v]["gamma"] for v in B.VARS},
                 k={v: hp[v]["k"] for v in B.VARS}) for s in te]
MODELS["oi"] = {v: np.stack([p[v] for p in oi], 0) for v in B.VARS}

# ---- targets: the ANOMALY, which is what protocol_v1 asks the model to predict
TRUE = {v: np.stack([s["gt"][v] for s in te], 0) for v in B.VARS}
ANOM = {v: np.stack([s["gt"][v] - clim.clim3d(v, s["month"]) for s in te], 0)
        for v in B.VARS}
UNOBS = np.stack([s["unobs_mask"] for s in te], 0)                # (N,H,W)
KEEP = np.broadcast_to(UNOBS[:, None], ANOM["TEMP"].shape)        # (N,D,H,W)


def stats(v, level=None, band=None):
    """(rmse per model, var0, varm, n) over the scoring set."""
    sel = slice(None) if (level is None and band is None) else None
    if level is not None:
        idx = [level]
    elif band is not None:
        lo, hi = band
        idx = np.where((grid.depth > lo) & (grid.depth <= hi))[0]
        if lo <= grid.depth.min():
            idx = np.union1d(idx, np.where(np.isclose(grid.depth,
                                                      grid.depth.min()))[0])
    else:
        idx = np.arange(grid.ndepth)
    a = ANOM[v][:, idx]
    m = KEEP[:, idx] & np.isfinite(a)
    av = a[m]
    var0 = float(np.mean(av ** 2))
    varm = float(np.var(av))
    out = {}
    for name, P in MODELS.items():
        e = (P[v][:, idx] - TRUE[v][:, idx])
        mm = m & np.isfinite(e)
        ev = e[mm]
        out[name] = float(np.sqrt(np.mean(ev ** 2)))
    return out, var0, varm, int(av.size)


UNIT = {"TEMP": "degC", "SALT": "PSU"}
rows = []
print()
for v in B.VARS:
    print(f"===== {v} ({UNIT[v]}) =====")
    print(f"{'level / band':>14} {'depth_m':>9} {'target_std':>11} {'target_var':>11} "
          f"{'RMSE_ssh':>9} {'RMSE_cert':>10} {'RMSE_OI':>9} {'EV_ssh':>7} {'EV_cert':>8}")
    for d in range(grid.ndepth):
        r, var0, varm, n = stats(v, level=d)
        rows.append({"var": v, "kind": "level", "level": d,
                     "depth_m": float(grid.depth[d]), "target_var0": var0,
                     "target_varm": varm, "target_std": float(np.sqrt(var0)),
                     "n_cells": n, "rmse": r,
                     "explained_var": {k: 1 - (x * x) / var0 for k, x in r.items()}})
        tag = f"L{d:02d}" + (" (SST-analogue)" if d == 0 else "")
        print(f"{tag:>14} {grid.depth[d]:9.1f} {np.sqrt(var0):11.4f} {var0:11.5f} "
              f"{r['unet_ssh']:9.4f} {r['unet_certified']:10.4f} {r['oi']:9.4f} "
              f"{1-r['unet_ssh']**2/var0:7.3f} {1-r['unet_certified']**2/var0:8.3f}")
    for name, lo, hi in BANDS:
        r, var0, varm, n = stats(v, band=(lo, hi))
        rows.append({"var": v, "kind": "band", "band": name, "target_var0": var0,
                     "target_varm": varm, "target_std": float(np.sqrt(var0)),
                     "n_cells": n, "rmse": r,
                     "explained_var": {k: 1 - (x * x) / var0 for k, x in r.items()}})
        print(f"{name:>14} {'':>9} {np.sqrt(var0):11.4f} {var0:11.5f} "
              f"{r['unet_ssh']:9.4f} {r['unet_certified']:10.4f} {r['oi']:9.4f} "
              f"{1-r['unet_ssh']**2/var0:7.3f} {1-r['unet_certified']**2/var0:8.3f}")
    r, var0, varm, n = stats(v)
    rows.append({"var": v, "kind": "full", "target_var0": var0, "target_varm": varm,
                 "target_std": float(np.sqrt(var0)), "n_cells": n, "rmse": r,
                 "explained_var": {k: 1 - (x * x) / var0 for k, x in r.items()}})
    print(f"{'FULL COLUMN':>14} {'':>9} {np.sqrt(var0):11.4f} {var0:11.5f} "
          f"{r['unet_ssh']:9.4f} {r['unet_certified']:10.4f} {r['oi']:9.4f} "
          f"{1-r['unet_ssh']**2/var0:7.3f} {1-r['unet_certified']**2/var0:8.3f}")
    print()

# ---- SSH and the surface inputs: variance only, they are INPUTS not targets ----
ssh_anom = np.stack([ssh_te[k] - sshnorm.clim[te[k]["month"] - 1]
                     for k in range(len(te))], 0)
m = UNOBS & np.isfinite(ssh_anom)
ssh_var0 = float(np.mean(ssh_anom[m] ** 2))
inputs = {"SSH_pseudo_m": {"target_var0": ssh_var0,
                           "target_std": float(np.sqrt(ssh_var0)),
                           "std_cm": float(np.sqrt(ssh_var0) * 100),
                           "role": "INPUT (derived steric height), never predicted"}}
for sv in C.VARS_SURF:
    a = np.stack([ftest[sv][k] - clim.clim_surf3d(sv, te[k]["month"])
                  for k in range(len(te))], 0)
    mm = UNOBS & np.isfinite(a)
    v0 = float(np.mean(a[mm] ** 2))
    inputs[sv] = {"target_var0": v0, "target_std": float(np.sqrt(v0)),
                  "role": "INPUT (dense satellite-analogue field), never predicted"}
print("===== fields that are INPUTS, not prediction targets =====")
print("(variance of their anomaly over the same scoring set, for reference)")
for k, d in inputs.items():
    extra = f"  = {d['std_cm']:.2f} cm" if "std_cm" in d else ""
    print(f"  {k:14s} std {d['target_std']:.4f}{extra}   var {d['target_var0']:.5f}"
          f"   [{d['role']}]")

out = {"protocol": "protocol_v1", "seed": args.seed,
       "scoring_set": "12 pinned test months, unobserved ocean cells only",
       "target_definition": "anomaly = field - train-only monthly climatology",
       "var0_note": "second moment about zero; sqrt(var0) == the climatology-floor RMSE",
       "models": {"unet_ssh": "depthwise U-Net + pseudo-SSH (best)",
                  "unet_certified": "depthwise U-Net, protocol_v1 certified",
                  "oi": "optimal interpolation, L=500km gamma=0.1 k=10"},
       "rows": rows, "inputs_not_targets": inputs}
p = os.path.join(C.CACHE, "variance_table.json")
json.dump(out, open(p, "w"), indent=2)
print(f"\nwrote {p}", flush=True)
