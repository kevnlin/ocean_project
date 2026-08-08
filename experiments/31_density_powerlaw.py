"""Phase-2 / Task 2.2 — power-law fit of the profile-density curve.

Answers the advisor's question directly: **at how many profiles per month does
the depthwise U-Net's TEMP RMSE cross 0.1 degC?**

Fits ``log RMSE = log a - alpha * log N`` over the densities present in the
week-2 cache, reports alpha, the implied crossing, and the fit residuals (so a
reader can judge whether extrapolating is defensible at all).  Re-run after
`08_density_ablation.py --densities 4000,6000 --out-suffix _ext` and
`merge_density_json.py` to replace the extrapolation with measurements.

Density 0 is excluded from the fit: log(0) is undefined, and the zero-profile
point is the no-observation limit, which no power law in N should describe.

Run:  python experiments/31_density_powerlaw.py
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ocean_tokenizer import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="1234,1235,1236")
ap.add_argument("--method", default="unet_depthwise")
ap.add_argument("--targets", default="0.1", help="TEMP thresholds, degC")
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(",")]
TARGETS = [float(x) for x in args.targets.split(",")]
VARS = ("TEMP", "SALT")
UNITS = {"TEMP": "degC", "SALT": "PSU"}

agg = {v: collections.defaultdict(list) for v in VARS}
floor = {v: collections.defaultdict(list) for v in VARS}
cfg = {}
for s in SEEDS:
    p = os.path.join(C.CACHE, f"density_ablation_seed{s}.json")
    if not os.path.exists(p):
        print(f"seed {s}: missing {os.path.basename(p)} — skipped")
        continue
    d = json.load(open(p))
    cfg = d.get("run_config", cfg)
    for r in d["results"]:
        for v in VARS:
            key = f"{v}_unobs"
            if r.get("method") == args.method and r.get(key) is not None:
                agg[v][r["density"]].append(r[key])
            if r.get("method") == "clim_floor" and r.get(key) is not None:
                floor[v][r["density"]].append(r[key])

D = sorted(agg["TEMP"])
if not D:
    raise SystemExit(f"no rows for method={args.method}")
# Seed count can differ per density: an extension run (4000/6000) may exist for
# only one seed while the original curve has three.  Say so rather than letting
# a "+/- 0.0000" masquerade as a tight 3-seed error bar.
n_by_density = {int(n): len(agg["TEMP"][n]) for n in D}
print(f"method={args.method}  densities={D}")
print(f"seeds per density: {n_by_density}")
if len(set(n_by_density.values())) > 1:
    single = [n for n, c in n_by_density.items() if c == 1]
    print(f"  NOTE: {single} are single-seed; their '+/- 0.0000' is not an "
          f"error bar, it is one measurement.")
print()

summary = {"method": args.method, "seeds": SEEDS, "densities": D,
           "n_seeds_by_density": n_by_density, "by_var": {}}
fits = {}
for v in VARS:
    mean = np.array([np.mean(agg[v][n]) for n in D])
    std = np.array([np.std(agg[v][n]) for n in D])
    pos = np.array(D) > 0
    lx, ly = np.log(np.array(D)[pos]), np.log(mean[pos])
    slope, intercept = np.polyfit(lx, ly, 1)
    alpha, a = -slope, float(np.exp(intercept))
    pred = a * np.array(D)[pos] ** (-alpha)
    resid = mean[pos] - pred
    rel = np.abs(resid) / mean[pos]
    r2 = 1 - np.sum((ly - (slope * lx + intercept)) ** 2) / np.sum((ly - ly.mean()) ** 2)
    fits[v] = (a, alpha)
    print(f"{v}: RMSE = {a:.4f} * N^(-{alpha:.4f})   R2(log-log) = {r2:.5f}   "
          f"max |rel resid| = {rel.max()*100:.1f}%")
    for n, m, sd, p_ in zip(np.array(D)[pos], mean[pos], std[pos], pred):
        print(f"   N={n:5d}  measured {m:.4f}+/-{sd:.4f}  fit {p_:.4f}  "
              f"({100*(m-p_)/m:+.1f}%)")
    summary["by_var"][v] = {
        "a": a, "alpha": float(alpha), "r2_loglog": float(r2),
        "measured": {int(n): {"mean": float(np.mean(agg[v][n])),
                              "std": float(np.std(agg[v][n]))} for n in D},
        "max_rel_resid": float(rel.max())}
    print()

# ---- local slopes: is a single power law even the right model? ----
# The global fit's residuals are systematic (over-predicting in the middle,
# under-predicting at the ends), which is the signature of a curve whose slope
# is changing.  Segment-wise slopes make that explicit instead of hiding it in
# an R^2 that looks reassuring.
print("Local (segment-wise) TEMP slopes -- a single power law would give a "
      "constant column:")
Dp = [n for n in D if n > 0]
local = []
for n1, n2 in zip(Dp[:-1], Dp[1:]):
    m1, m2 = np.mean(agg["TEMP"][n1]), np.mean(agg["TEMP"][n2])
    al = -np.log(m2 / m1) / np.log(n2 / n1)
    local.append({"from": int(n1), "to": int(n2), "alpha": float(al)})
    print(f"   {n1:5d} -> {n2:5d}:  alpha = {al:.3f}")
summary["local_slopes_TEMP"] = local
alpha_tail = local[-1]["alpha"]
a_tail = np.mean(agg["TEMP"][Dp[-1]]) * Dp[-1] ** alpha_tail
print(f"   -> tail slope {alpha_tail:.3f} vs global fit {fits['TEMP'][1]:.3f}: "
      f"the curve is steepening, so the two extrapolations disagree.\n")

# ---- crossings, under BOTH models ----
a, alpha = fits["TEMP"]
summary["crossings"] = {}
print("TEMP crossings -- reported under both models, because they disagree:")
for tgt in TARGETS:
    n_global = (a / tgt) ** (1.0 / alpha)
    n_tail = (a_tail / tgt) ** (1.0 / alpha_tail)
    summary["crossings"][str(tgt)] = {
        "n_profiles_global_fit": float(n_global),
        "n_profiles_tail_slope": float(n_tail),
        "extrapolated": bool(min(n_global, n_tail) > max(D))}
    print(f"   {tgt} degC:  global fit N = {n_global:.0f}   |   "
          f"tail slope N = {n_tail:.0f}"
          f"{'   [both EXTRAPOLATED beyond N=' + str(max(D)) + ']' if min(n_global, n_tail) > max(D) else ''}")
for n in (4000, 6000):
    pg, pt = a * n ** (-alpha), a_tail * n ** (-alpha_tail)
    print(f"   predicted at N={n}: global fit {pg:.4f} | tail slope {pt:.4f} degC")
    summary.setdefault("predictions", {})[str(n)] = {
        "global_fit": float(pg), "tail_slope": float(pt)}

path = os.path.join(C.CACHE, "density_powerlaw.json")
json.dump(summary, open(path, "w"), indent=2)

# ---- figure ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, v in zip(axes, VARS):
    mean = np.array([np.mean(agg[v][n]) for n in D])
    std = np.array([np.std(agg[v][n]) for n in D])
    pos = np.array(D) > 0
    a_, al_ = fits[v]
    grid_n = np.logspace(np.log10(80), np.log10(8000), 100)
    ax.plot(grid_n, a_ * grid_n ** (-al_), "k--", lw=1,
            label=f"fit  N^(-{al_:.2f})")
    ax.errorbar(np.array(D)[pos], mean[pos], yerr=std[pos], marker="o", lw=1.6,
                capsize=3, label=f"{args.method} (measured)")
    if floor[v]:
        fl = np.mean([np.mean(floor[v][n]) for n in D if floor[v][n]])
        ax.axhline(fl, ls=":", c="gray", label="climatology floor")
    if v == "TEMP":
        for tgt in TARGETS:
            ax.axhline(tgt, c="#d62728", lw=1, alpha=.7,
                       label=f"target {tgt} degC")
            ns = (fits["TEMP"][0] / tgt) ** (1 / fits["TEMP"][1])
            ax.axvline(ns, c="#d62728", ls="--", lw=1, alpha=.7)
            ax.annotate(f"N≈{ns:.0f}", (ns, tgt), textcoords="offset points",
                        xytext=(6, 6), color="#d62728", fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("profiles per month"); ax.set_ylabel(f"unobserved-only RMSE ({UNITS[v]})")
    ax.set_title(f"{v} vs profile density")
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(C.REPORTS, "fig_density_powerlaw.png"), dpi=140)
print(f"\nwrote {path} and reports/fig_density_powerlaw.png")
