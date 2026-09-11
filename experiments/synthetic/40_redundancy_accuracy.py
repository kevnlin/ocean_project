"""Phase 2 — does redundancy handling change ACCURACY when redundancy exists?

This is the experiment the whole line turns on.

Phase 0 (corrected) established the mechanism is operative: exact duplicates
collapse, dual-stream copies collapse along ``n_eff``, clustered sampling
carries ~31% less evidence.  Yet three independent evaluations reported ~1%
end-to-end effects.  The work plan's explanation is that *the evaluation
contains none of the disease the method cures* — profiles drawn uniformly from
a gridded reanalysis, no duplicate stream, no overlapping product.

So: take the SAME trained models and score them twice — once on the clean
evaluation, once on an evaluation whose INPUT carries redundancy — and see
whether the DFS-vs-uniform gap opens.

    clean        the standard protocol: uniformly drawn profile columns
    duplicated   a fraction of columns re-ingested k times (no new information)
    dual_stream  those copies instead share a per-level provenance group,
                 i.e. real-time + delayed-mode of one float
    clustered    the same column COUNT confined to a small box

Only the observations change.  Targets, query points and the scored mask are
identical across regimes and across rows, so any difference is the fusion rule
responding to the input pathology and nothing else.

Rows are the THIN STACK (``*_expertlocal_cbottle``, no frozen OI residual),
which is Phase 1c: the OI background already merges duplicate supports into
superobservations, so comparing on top of it asks whether redundancy handling
helps something that has already de-duplicated the observations.

Run (after experiments/real_data/14_godas_dfs_d4rt.py has trained the rows):
    .venv/bin/python experiments/synthetic/40_redundancy_accuracy.py \
        --checkpoints outputs/godas_thinstack --data data/godas_gulfstream
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ocean_tokenizer import config as C
from ocean_tokenizer.godas import load_godas, GodasNorm
from ocean_tokenizer.godas_obs import (build_sample, ObsConfig, MOD_PROFILE,
                                        CONTEXT_MONTHS)
from ocean_tokenizer.godas_model import build_row

CHANNELS = ("TEMP", "SALT")
DEPTHS = 16
SPLITS = {"train": (2000, 2018), "validation": (2019, 2021),
          "development": (2022, 2024), "holdout": (2025, 2025)}

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoints", default="outputs/godas_thinstack")
ap.add_argument("--data", default="data/godas_gulfstream")
ap.add_argument("--rows", default="dfs_expertlocal_cbottle,"
                                  "uniform_expertlocal_cbottle,"
                                  "count_expertlocal_cbottle")
ap.add_argument("--split", default="development",
                help="development is the diagnostic split; holdout stays "
                     "sealed for the final locked evaluation")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--queries", type=int, default=1024)
ap.add_argument("--dup-k", type=int, default=4)
ap.add_argument("--dup-frac", type=float, default=0.5,
                help="fraction of profile columns that arrive duplicated")
ap.add_argument("--rho", type=float, default=0.0,
                help="provenance correlation AT EVAL.  Default 0.0 to match "
                     "training: 14_godas_dfs_d4rt.py trains with rho=0, so a "
                     "nonzero value here shifts the mass distribution the model "
                     "was fitted to and the comparison stops being clean.  A "
                     "fair rho>0 test needs the rows retrained with it, which "
                     "is a Phase 1 follow-up; --rho 0.9 shows the ceiling.")
ap.add_argument("--tag", default="redundancy_accuracy")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
ROWS = [r.strip() for r in args.rows.split(",") if r.strip()]

# ===================================================================== data
raw = load_godas(args.data)
years = np.array([int(str(m)[:4]) for m in raw["months"]])
tr = np.flatnonzero((years >= SPLITS["train"][0]) & (years <= SPLITS["train"][1]))
norm = GodasNorm(raw, tr)
FIELDS = {v: norm.z(v, raw[v], raw["months"]) for v in ("TEMP", "SALT", "SSH")}
FIELDS["depth"] = raw["depth"]

lo, hi = SPLITS[args.split]
sel = np.flatnonzero((years >= lo) & (years <= hi))
# a source month is eligible only if its whole window stays inside the split
elig = [int(t) for t in sel if t - (CONTEXT_MONTHS - 1) >= sel[0]]
print(f"[{args.tag}] split={args.split} eligible source months={len(elig)} "
      f"device={dev}", flush=True)

PER_TOKEN = ("coord", "value", "value_mask", "mask", "support_mask", "modality",
             "variable_group", "noise_density", "support_area", "support_dz",
             "provenance")


def to_dev(s):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}


# ================================================================== regimes
def inject(s, regime, rng):
    """Return a token dict whose OBSERVATIONS carry the regime's pathology.

    Targets, queries and the scored mask are never touched, so every regime and
    every row is scored on exactly the same quantity.
    """
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in s.items()}
    if regime == "clean":
        return out
    prof_cols = int((out["modality"] == MOD_PROFILE).sum()) // DEPTHS
    if prof_cols == 0:
        return out
    if regime == "clustered":
        # same COUNT, confined: redundancy by geometry rather than by copying
        prof = out["modality"] == MOD_PROFILE
        cx, cy = rng.uniform(0.25, 0.75, 2)
        n = int(prof.sum())
        xs = np.repeat(rng.normal(cx, 0.02, prof_cols), DEPTHS)[:n]
        ys = np.repeat(rng.normal(cy, 0.02, prof_cols), DEPTHS)[:n]
        out["coord"][prof, 0] = torch.as_tensor(np.clip(xs, 0, 1),
                                                dtype=out["coord"].dtype)
        out["coord"][prof, 1] = torch.as_tensor(np.clip(ys, 0, 1),
                                                dtype=out["coord"].dtype)
        return out

    # duplicated / dual_stream: re-ingest a fraction of the columns k times
    n_dup = max(1, int(round(args.dup_frac * prof_cols)))
    pick = rng.choice(prof_cols, size=n_dup, replace=False)
    blocks = []
    for c_i, c in enumerate(pick):
        blk = {k: out[k][c * DEPTHS:(c + 1) * DEPTHS].clone() for k in PER_TOKEN}
        if regime == "dual_stream":
            # per LEVEL pairing: one float, two reprocessings of each level
            pid = torch.arange(DEPTHS) + 50_000 + 100 * c_i
            out["provenance"][c * DEPTHS:(c + 1) * DEPTHS] = pid
            blk["provenance"] = pid.clone()
        else:                       # plain duplicate: a distinct platform id
            blk["provenance"] = blk["provenance"] + 9_000 + c_i
        blocks.append(blk)
    for rep in range(args.dup_k - 1):
        for blk in blocks:
            b = {k: v.clone() for k, v in blk.items()}
            if regime == "duplicated":
                b["provenance"] = b["provenance"] + 1000 * (rep + 1)
            for k in PER_TOKEN:
                out[k] = torch.cat([out[k], b[k]], dim=0)
    return out


REGIMES = ("clean", "duplicated", "dual_stream", "clustered")


# ================================================================== evaluate
@torch.no_grad()
def rmse_of(model, regime):
    """Unobserved-cell RMSE per channel, z space — same protocol as 14."""
    rng = np.random.default_rng([args.seed, 7])
    inj = np.random.default_rng([args.seed, 11])     # regime draws, row-independent
    se = torch.zeros(len(CHANNELS), dtype=torch.float64, device=dev)
    n = torch.zeros(len(CHANNELS), dtype=torch.float64, device=dev)
    model.eval()
    for t in elig:
        s = build_sample(FIELDS, t, ObsConfig(train=False,
                                              n_queries=args.queries),
                         rng, lead=0)
        s = to_dev(inject(s, regime, inj))
        out = model(s)
        m = s["target_mask"]
        se += ((out - s["target"]) ** 2 * m).sum(dim=0).double()
        n += m.sum(dim=0).double()
    r = (se / n.clamp(min=1)).sqrt()
    return {c: float(r[i]) for i, c in enumerate(CHANNELS)}


results, missing = {}, []
for row in ROWS:
    ck = os.path.join(args.checkpoints, f"{row}_seed{args.seed}.pt")
    if not os.path.exists(ck):
        print(f"  SKIP {row}: no checkpoint at {ck}", flush=True)
        missing.append(row)
        continue
    rho = args.rho if row.startswith("dfs") else 0.0
    model = build_row(row, provenance_rho=rho).to(dev)
    model.load_state_dict(torch.load(ck, map_location=dev,
                                     weights_only=False)["state_dict"])
    results[row] = {}
    print(f"\n=== {row} (rho={rho}) ===", flush=True)
    for reg in REGIMES:
        r = rmse_of(model, reg)
        results[row][reg] = r
        print(f"  {reg:<12} TEMP {r['TEMP']:.4f}  SALT {r['SALT']:.4f}",
              flush=True)
    del model
    torch.cuda.empty_cache()

if not results:
    raise SystemExit("no checkpoints found — run experiments/real_data/14_godas_dfs_d4rt.py first")

# ============================================================== the contrast
print("\n" + "=" * 68, flush=True)
print("THE CONTRAST — DFS advantage over uniform, by regime", flush=True)
print("(positive = DFS better; the Phase 2 hypothesis says the gap should be", flush=True)
print(" ~0 on 'clean' and OPEN once the input carries redundancy)", flush=True)
print("=" * 68, flush=True)
adv = {}
if "dfs_expertlocal_cbottle" in results and "uniform_expertlocal_cbottle" in results:
    d, u = (results["dfs_expertlocal_cbottle"],
            results["uniform_expertlocal_cbottle"])
    print(f"  {'regime':<12}" + "  ".join(f"{c:>24}" for c in CHANNELS), flush=True)
    for reg in REGIMES:
        cells = []
        adv[reg] = {}
        for c in CHANNELS:
            gain = 100.0 * (u[reg][c] - d[reg][c]) / u[reg][c]
            adv[reg][c] = gain
            cells.append(f"{d[reg][c]:.4f} vs {u[reg][c]:.4f} {gain:+5.1f}%")
        print(f"  {reg:<12}" + "  ".join(f"{x:>24}" for x in cells), flush=True)
    clean = float(np.mean([adv["clean"][c] for c in CHANNELS]))
    for reg in ("duplicated", "dual_stream", "clustered"):
        got = float(np.mean([adv[reg][c] for c in CHANNELS]))
        print(f"\n  {reg}: DFS advantage {got:+.2f}% vs {clean:+.2f}% on clean "
              f"-> opened by {got - clean:+.2f} points", flush=True)

# ==================================================================== figure
if adv:
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    xs = np.arange(len(REGIMES))
    for i, c in enumerate(CHANNELS):
        ax.bar(xs + (i - 0.5) * 0.38, [adv[r][c] for r in REGIMES],
               width=0.38, label=c)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(REGIMES)
    ax.set_ylabel("DFS advantage over uniform (% RMSE)")
    ax.set_title("Does redundancy handling pay off when redundancy is present?\n"
                 f"thin stack, {args.split} split, {len(elig)} months, "
                 f"k={args.dup_k} copies of {100*args.dup_frac:.0f}% of columns",
                 fontsize=11)
    ax.legend(); ax.grid(alpha=.3, axis="y")
    fig_path = os.path.join(C.REPORTS_SYNTHETIC, "fig_redundancy_accuracy.png")
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    print(f"\nwrote {fig_path}", flush=True)
else:
    fig_path = None


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


rec = {"task": "phase2_redundancy_accuracy", "git_commit": git_commit(),
       "split": args.split, "months": len(elig), "seed": args.seed,
       "dup_k": args.dup_k, "dup_frac": args.dup_frac, "rho": args.rho,
       "rows": ROWS, "missing_rows": missing, "results": results,
       "dfs_advantage_pct": adv, "figure": fig_path}
os.makedirs(C.CACHE, exist_ok=True)
with open(os.path.join(C.CACHE, f"{args.tag}.json"), "w") as f:
    json.dump(rec, f, indent=2)
print(f"wrote {os.path.join(C.CACHE, args.tag + '.json')}", flush=True)
print(f"TOTAL {time.time()-t0:.1f}s", flush=True)
