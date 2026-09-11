"""2-D reconstruction error maps on REAL held-out Argo floats.

The global CESM2 heat maps (`36_d4rt_recon_heatmap.py`) put their brightest band
straight along the equator: the cold tongue and its sharp, shallow thermocline
are the hardest water in the reconstruction. This script asks the same question
of real observations, in a real region, and it has to answer it differently.

Why this cannot be the same plot
--------------------------------
The CESM2 map scores a DENSE field against a known truth, so every grid cell has
an error. Here the truth is a set of **point measurements from floats the model
has never seen**, so error exists only where a held-out float actually surfaced.
A cell's value is therefore the pooled RMSE of every held-out observation that
fell in it, and cells no float visited are blank — not zero, and not
interpolated. Painting them would invent skill in exactly the places we have no
evidence about.

`--min-obs` sets how many scored values a cell needs before it is drawn at all.
A cell holding three values has an RMSE with an enormous standard error, and at
1° resolution there are a lot of those; drawing them produces a speckle of
extreme values that reads as structure and is noise. The count map is written
alongside so the coverage behind every panel is visible.

Panels follow the global figure: rows are TEMP and SALT, columns are the depth
bands plus the pooled full column, dark = accurate, grey = unscored.

  .venv/bin/python experiments/52_argo_recon_map.py --region eq_pacific \
      --checkpoint outputs/argo_P0_eq_pacific_pall/dfs_expertlocal_cbottle_s1234.pt
  .venv/bin/python experiments/52_argo_recon_map.py --region eq_pacific --compare
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ocean_tokenizer import protocol as P
from ocean_tokenizer.argo_obs import (ArgoCohort, ArgoNorm, ArgoObsConfig,
                                      build_argo_sample)
from ocean_tokenizer.clustered_ci import (accumulate, ci_rmse_across_seeds,
                                          ci_difference_across_seeds)
from ocean_tokenizer.godas_model import build_row

LAYERS = [("0-100 m", 0.0, 100.0), ("100-300 m", 100.0, 300.0),
          ("300-985 m", 300.0, 1000.0)]
VARS = ("TEMP", "SALT")
UNITS = {"TEMP": "degC", "SALT": "PSU"}

ap = argparse.ArgumentParser()
ap.add_argument("--region", default="eq_pacific", choices=list(P.REGIONS))
ap.add_argument("--checkpoint", default=None,
                help="one row's .pt; omit with --compare to sweep the arms")
ap.add_argument("--row", default="dfs_expertlocal_cbottle")
ap.add_argument("--arms", default="p24,pall",
                help="output-dir suffixes to compare, in order")
ap.add_argument("--seeds", default="1234,1235,1236")
ap.add_argument("--n-profiles", type=int, default=None,
                help="input profiles at eval; defaults to the arm's own budget")
ap.add_argument("--eval-split", default="development")
ap.add_argument("--min-obs", type=int, default=8)
ap.add_argument("--region-kernel", action="store_true", default=True)
ap.add_argument("--compare", action="store_true")
ap.add_argument("--reports", default=None)
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORTS = args.reports or os.path.join(ROOT, "reports", "real_data")
os.makedirs(REPORTS, exist_ok=True)
seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()

c = ArgoCohort.load(os.path.join(ROOT, "data", "argo_cohort", f"{args.region}.nc"))
norm = ArgoNorm.fit(c, "train")
NY, NX = c.grid
LEV = c.levels
box = P.REGIONS[args.region]
extent = [box["lon"][0], box["lon"][1], box["lat"][0], box["lat"][1]]

layer_of_level = np.full(LEV.size, len(LAYERS), dtype=int)
for li, (_, lo, hi) in enumerate(LAYERS):
    sel = (LEV > lo) & (LEV <= hi)
    if lo <= LEV.min():
        sel |= np.isclose(LEV, LEV.min())
    layer_of_level[sel] = li
assert (layer_of_level < len(LAYERS)).all(), "a level fell outside LAYERS"

months = [int(m) for m in c.months_in(args.eval_split)
          if c.month(m, float_split="cohort_float").size
          and c.month(m, float_split="heldout_float").size]


def score_maps(ckpt: str, n_profiles: int, seed: int):
    """Per-cell (se, n) by variable and depth panel, plus per-float stats."""
    model = build_row(args.row,
                      region=(args.region if args.region_kernel else None)).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()
    npanel = len(LAYERS) + 1
    se = {v: np.zeros((npanel, NY, NX)) for v in VARS}
    cnt = {v: np.zeros((npanel, NY, NX)) for v in VARS}
    labs = {v: [] for v in VARS}
    errs = {v: [] for v in VARS}
    # every held-out value is scored: no query subsampling, or the map would be
    # a sample of the coverage rather than the coverage
    cfg = ArgoObsConfig(n_profiles=n_profiles, n_queries=0, train=False)
    for m in months:
        rng = np.random.default_rng([seed, m, 0])
        s = build_argo_sample(c, norm, m, cfg=cfg, rng=rng, lead=0)
        if s is None or s["target"].shape[0] == 0:
            continue
        sd = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        with torch.no_grad():
            pred = model(sd)
        e = ((pred - sd["target"]) ** 2).cpu().numpy()
        msk = sd["target_mask"].cpu().numpy()
        wmo = np.asarray(sd["target_wmo"])
        q = sd["query"].cpu().numpy()
        gx = np.rint(q[:, 0] * max(NX - 1, 1)).astype(int).clip(0, NX - 1)
        gy = np.rint(q[:, 1] * max(NY - 1, 1)).astype(int).clip(0, NY - 1)
        zi = np.rint(q[:, 2] * max(LEV.size - 1, 1)).astype(int).clip(0, LEV.size - 1)
        for j, v in enumerate(VARS):
            ok = msk[:, j] & np.isfinite(e[:, j])
            if not ok.any():
                continue
            # physical units for the map; z-units for the CI, which must stay
            # comparable with every other table in the study
            phys = e[ok, j] * (norm.std[v][zi[ok]] ** 2)
            for panel in (layer_of_level[zi[ok]], np.full(ok.sum(), len(LAYERS))):
                np.add.at(se[v], (panel, gy[ok], gx[ok]), phys)
                np.add.at(cnt[v], (panel, gy[ok], gx[ok]), 1.0)
            labs[v].append(wmo[ok]); errs[v].append(e[ok, j])
    del model
    torch.cuda.empty_cache()
    rmse = {v: np.where(cnt[v] >= args.min_obs,
                        np.sqrt(np.divide(se[v], np.maximum(cnt[v], 1))), np.nan)
            for v in VARS}
    stats = {v: (accumulate(np.concatenate(labs[v]), np.concatenate(errs[v]),
                            cluster_unit="wmo", channel=v, method=args.row)
                 if labs[v] else None) for v in VARS}
    return rmse, cnt, stats


def find_ckpt(arm: str, seed: int) -> str | None:
    p = os.path.join(ROOT, "outputs", f"argo_P0_{args.region}_{arm}",
                     f"{args.row}_s{seed}.pt")
    return p if os.path.exists(p) else None


ARM_NPROF = {"p24": 24, "pall": 0}
results = {}
if args.compare:
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
else:
    arms = ["single"]

for arm in arms:
    maps, counts, per_seed = [], None, {v: [] for v in VARS}
    for sd in seeds:
        ck = args.checkpoint if arm == "single" else find_ckpt(arm, sd)
        if ck is None or not os.path.exists(ck):
            continue
        npro = (args.n_profiles if args.n_profiles is not None
                else ARM_NPROF.get(arm, 24))
        r, n, st = score_maps(ck, npro, sd)
        maps.append(r); counts = n
        for v in VARS:
            per_seed[v].append(st[v])
        print(f"  {arm:6s} seed {sd}: "
              + "  ".join(f"{v} {st[v].rmse():.4f}" for v in VARS if st[v]),
              flush=True)
    if not maps:
        print(f"  {arm}: no checkpoints found"); continue
    mean_map = {v: np.nanmean(np.stack([m[v] for m in maps]), axis=0) for v in VARS}
    ci = {}
    for v in VARS:
        good = [x for x in per_seed[v] if x is not None]
        if good:
            iv = ci_rmse_across_seeds(good, n_boot=2000, seed=20260905)
            ci[v] = iv.to_dict()
    results[arm] = {"map": mean_map, "counts": counts, "ci": ci,
                    "stats": per_seed, "n_seeds": len(maps),
                    "n_profiles": ARM_NPROF.get(arm, args.n_profiles or 24)}

if not results:
    raise SystemExit("nothing scored — check --checkpoint / --arms")

# ------------------------------------------------------------------ figure
PANELS = [nm for nm, _, _ in LAYERS] + ["full column"]
cmap = plt.cm.hot.copy(); cmap.set_bad("0.75")
# One colour scale across every arm. Letting each panel pick its own 98th
# percentile makes two arms look identical when one is genuinely better -- the
# improvement goes into the colour bar instead of into the image.
VMAX = {v: float(np.nanpercentile(
    np.concatenate([R["map"][v].ravel() for R in results.values()]), 98))
    for v in VARS}

for arm, R in results.items():
    fig, axes = plt.subplots(len(VARS), len(PANELS),
                             figsize=(4.1 * len(PANELS), 6.6),
                             constrained_layout=True)
    for r, v in enumerate(VARS):
        vmax = VMAX[v]
        for cix, nm in enumerate(PANELS):
            ax = axes[r, cix]
            im = ax.imshow(np.ma.masked_invalid(R["map"][v][cix]), origin="lower",
                           extent=extent, aspect="auto", cmap=cmap,
                           vmin=0.0, vmax=vmax)
            if r == 0:
                ax.set_title(nm)
            if cix == 0:
                ax.set_ylabel(f"{v} ({UNITS[v]})\nlatitude")
            if r == len(VARS) - 1:
                ax.set_xlabel("longitude")
            ax.axhline(0.0, color="0.35", lw=0.6, ls=":")   # the equator itself
        fig.colorbar(im, ax=axes[r, :], fraction=0.02,
                     label=f"{v} RMSE ({UNITS[v]})")
    nprof = "all available" if R["n_profiles"] == 0 else str(R["n_profiles"])
    fig.suptitle(
        f"Real-Argo 2-D reconstruction error — {args.region}, "
        f"{args.row} ({R['n_seeds']} seeds)\n"
        f"{len(months)} held-out {args.eval_split} months, input {nprof} "
        f"profiles/month; scored ONLY at WMO-disjoint held-out float positions\n"
        f"dark = accurate, bright = more error, grey = no held-out float "
        f"reported there (>= {args.min_obs} values per cell)", fontsize=10)
    out = os.path.join(REPORTS, f"fig_argo_recon_{args.region}_{arm}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {os.path.relpath(out, ROOT)}")

# where the extra observations actually changed the reconstruction
if len(results) == 2:
    (a_name, A), (b_name, B) = list(results.items())
    fig, axes = plt.subplots(len(VARS), len(PANELS),
                             figsize=(4.1 * len(PANELS), 6.6),
                             constrained_layout=True)
    dcm = plt.cm.RdBu_r.copy(); dcm.set_bad("0.75")
    for r, v in enumerate(VARS):
        d = A["map"][v] - B["map"][v]          # >0 : the second arm is better
        lim = float(np.nanpercentile(np.abs(d), 98))
        for cix, nm in enumerate(PANELS):
            ax = axes[r, cix]
            im = ax.imshow(np.ma.masked_invalid(d[cix]), origin="lower",
                           extent=extent, aspect="auto", cmap=dcm,
                           vmin=-lim, vmax=lim)
            if r == 0:
                ax.set_title(nm)
            if cix == 0:
                ax.set_ylabel(f"{v} ({UNITS[v]})\nlatitude")
            if r == len(VARS) - 1:
                ax.set_xlabel("longitude")
            ax.axhline(0.0, color="0.25", lw=0.6, ls=":")
        fig.colorbar(im, ax=axes[r, :], fraction=0.02,
                     label=f"RMSE({a_name}) - RMSE({b_name})  [{UNITS[v]}]")
    fig.suptitle(
        f"Where the extra Argo profiles helped — {args.region}, {args.row}\n"
        f"RMSE({a_name}) minus RMSE({b_name}); "
        f"**red = {b_name} is better**, blue = worse, grey = unscored\n"
        f"paired over {len(seeds)} seeds and the same held-out floats",
        fontsize=10)
    out = os.path.join(REPORTS, f"fig_argo_recon_{args.region}_improvement.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {os.path.relpath(out, ROOT)}")

# coverage, so the blank cells in the maps above are accountable
fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.2), constrained_layout=True)
cm2 = plt.cm.viridis.copy(); cm2.set_bad("0.75")
first = next(iter(results.values()))
im = ax.imshow(np.ma.masked_where(first["counts"]["TEMP"][-1] == 0,
                                  first["counts"]["TEMP"][-1]),
               origin="lower", extent=extent, aspect="auto", cmap=cm2)
ax.set_title(f"held-out float values per cell — {args.region}")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.axhline(0.0, color="w", lw=0.6, ls=":")
fig.colorbar(im, ax=ax, label="scored values")
cov = os.path.join(REPORTS, f"fig_argo_recon_{args.region}_coverage.png")
fig.savefig(cov, dpi=130); plt.close(fig)
print(f"  wrote {os.path.relpath(cov, ROOT)}")

# Paired arm comparison. Both arms are scored on the SAME held-out floats and
# the same seeds, so an unpaired look at two overlapping CIs badly understates
# the evidence -- the arms' errors are strongly correlated through the ocean
# state they are both trying to reconstruct.
paired = {}
if len(results) == 2:
    (a_name, A), (b_name, B) = list(results.items())
    for v in VARS:
        sa = [x for x in A["stats"][v] if x is not None]
        sb = [x for x in B["stats"][v] if x is not None]
        if len(sa) != len(sb) or not sa:
            continue
        for x in sa: x.method = a_name
        for x in sb: x.method = b_name
        iv = ci_difference_across_seeds(sb, sa, n_boot=4000, seed=20260905)
        per = [float(y.rmse() - x.rmse()) for x, y in zip(sa, sb)]
        paired[v] = {**iv.to_dict(), "per_seed": per,
                     "label": f"{b_name} - {a_name} (negative favours {b_name})"}
        print(f"  {v}: {b_name} - {a_name} = {iv}  "
              f"excludes_zero={iv.excludes_zero}  per-seed "
              f"{['%+.4f' % z for z in per]}", flush=True)

summary = {arm: {"ci": R["ci"], "n_seeds": R["n_seeds"],
                 "n_profiles": R["n_profiles"],
                 "cells_scored": int(np.isfinite(R["map"]["TEMP"][-1]).sum()),
                 "cells_total": int(NY * NX)} for arm, R in results.items()}
with open(os.path.join(REPORTS, f"argo_recon_{args.region}.json"), "w") as f:
    json.dump({"arms": summary, "paired": paired}, f, indent=1)
print(json.dumps(summary, indent=1))
print(f"total {time.time()-t0:.0f}s")
