"""Phase 2 — put the pathology into the evaluation.

work_plan.md Phase 2: *"The evaluation currently contains none of the disease
the method cures — profiles are drawn uniformly from a gridded reanalysis, with
no duplicate stream, no overlapping product, no clustered sampling. That is why
DFS and uniform agree."*

Phase 0 (corrected) showed the mechanism is operative: exact duplicates already
collapse to ~3x at k=8 (71% discount). Phase 1 added vertical quadrature and
provenance noise correlation. This script builds the regimes in which those
mechanisms are supposed to matter, and measures the evidence response of each.

The regimes
-----------
``exact``        k bit-exact copies of one column — no new information
``jittered``     k copies displaced by a small distance — near-duplicates
``dual_stream``  2 copies sharing a PROVENANCE group (real-time + delayed-mode
                 of one float), which is the paper's opening example and is
                 reachable only with Phase 1b
``separated``    **the positive control** — k genuinely new columns, far apart.
                 Without it, "suppresses redundancy" cannot be told apart from
                 "learned to ignore the profile stream": both give a flat curve.
``clustered``    the same token COUNT confined to a small box, versus uniform
``split_patch``  the same field tiled at 4x4 versus 2x2 — 4x the tokens, one
                 field.  Token-count invariance: total evidence must track the
                 SUPPORT, not the token count.

Everything here is geometry: ``omega`` depends on support, noise, provenance and
variable group alone, so no training and no GODAS fields are required. The
accuracy half of Phase 2 (representation shift, the control ladder, RMSE
triples) needs a trained model and is not attempted here.

Run:
    .venv/bin/python experiments/39_redundancy_regimes.py
"""
from __future__ import annotations

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ocean_tokenizer import config as C
from ocean_tokenizer import batched_dfs as B
from ocean_tokenizer.batched_dfs import (RandomFourierBasis, integrate_support,
                                         dfs_omega, variable_group_coords,
                                         to_physical, vertical_quadrature,
                                         n_eff)
from ocean_tokenizer.godas_obs import (ObsConfig, build_sample,
                                        N_VARIABLE_GROUPS, MOD_PROFILE)

DEPTHS = 16
KS = [1, 2, 4, 8]

ap = argparse.ArgumentParser()
ap.add_argument("--months", type=int, default=6)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--rho", type=float, default=0.9,
                help="provenance noise correlation for the dual-stream regime")
ap.add_argument("--jitter-km", type=float, default=25.0)
ap.add_argument("--n-vertical-nodes", type=int, default=3)
ap.add_argument("--tag", default="redundancy_regimes")
args = ap.parse_args()

t0 = time.time()
rng = np.random.default_rng(args.seed)
T, Z, Y, X = args.months + 2, DEPTHS, B.GRID_NY, B.GRID_NX
FIELDS = {"TEMP": rng.standard_normal((T, Z, Y, X)),
          "SALT": rng.standard_normal((T, Z, Y, X)),
          "SSH": rng.standard_normal((T, Y, X))}
CFG = ObsConfig(train=False)
SAMPLES = [build_sample(FIELDS, t, CFG, rng, 0) for t in range(args.months)]
BASIS = RandomFourierBasis(
    B.N_FEATURES,
    B.LENGTH_SCALES_KM + variable_group_coords.scales(N_VARIABLE_GROUPS),
    B.BASIS_SEED)
print(f"[{args.tag}] {len(SAMPLES)} samples, "
      f"{int(SAMPLES[0]['mask'].sum())} tokens each, F={B.N_FEATURES}, "
      f"rho={args.rho}", flush=True)


# ===================================================================== engine
def omega_of(s: dict, rho: float = 0.0):
    """Physical-units omega for a token dict (Phase 0 + Phase 1 estimator)."""
    coord = to_physical(s["coord"])
    area = s["support_area"].to(torch.float64)
    if args.n_vertical_nodes > 1:
        zq, wq = vertical_quadrature(coord[:, 2],
                                     s["support_dz"].to(torch.float64),
                                     args.n_vertical_nodes)
        Q = zq.shape[1]
        nodes = coord[:, None, :].repeat(1, Q, 1).clone()
        nodes[:, :, 2] = zq
        weight = wq * area[:, None]
    else:
        nodes = coord[:, None, :]
        weight = area[:, None]
    grp = variable_group_coords(s["variable_group"], N_VARIABLE_GROUPS)
    g = grp[:, None, :].expand(-1, nodes.shape[1], -1)
    psi, lam = integrate_support(BASIS, torch.cat([nodes, g], -1), weight,
                                 s["noise_density"])
    live = s["mask"] & s["support_mask"]
    return dfs_omega(psi, lam, live, s.get("provenance"), rho), live


def clone(s):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in s.items()}


PER_TOKEN = ("coord", "value", "value_mask", "mask", "support_mask", "modality",
             "variable_group", "noise_density", "support_area", "support_dz",
             "provenance")


def append_block(s, block):
    out = clone(s)
    for k in PER_TOKEN:
        out[k] = torch.cat([out[k], block[k]], dim=0)
    return out


def head_block(s, n=DEPTHS):
    return {k: s[k][:n].clone() for k in PER_TOKEN}


# =================================================================== regimes
def regime_copies(s, k, kind):
    """k copies of the leading column: exact / jittered / dual_stream."""
    out = clone(s)
    if kind == "dual_stream":
        # the ORIGINAL column must carry the same per-level ids its copies get,
        # or the pairing never forms
        out["provenance"] = out["provenance"].clone()
        out["provenance"][:DEPTHS] = torch.arange(DEPTHS) + 10_000
    n0 = int(s["mask"].shape[0])
    idx = [torch.arange(DEPTHS)]
    for c in range(1, k):
        blk = head_block(s)
        if kind == "jittered":
            # displace in x by a real distance: a near-duplicate, not a copy
            dx = args.jitter_km / B.BOX_EXTENT[0]
            blk["coord"][:, 0] = (blk["coord"][:, 0] + c * dx).clamp(0, 1)
            blk["provenance"] = blk["provenance"] + 1000 + c   # distinct platform
        elif kind == "exact":
            blk["provenance"] = blk["provenance"] + 1000 + c   # distinct platform
        elif kind == "dual_stream":
            # Pair PER LEVEL, not per column.  Real-time and delayed-mode are
            # two reprocessings of the SAME measurement, so level j of stream A
            # is correlated with level j of stream B — while different levels
            # of one float are distinct measurements and must stay independent.
            # Grouping the whole column instead makes the 16 differing levels
            # the dominant term, and the (1-rho)^-1/2 amplification of their
            # deviation swamps the collapse being measured.
            blk["provenance"] = torch.arange(DEPTHS) + 10_000
        else:
            raise ValueError(kind)
        idx.append(torch.arange(n0 + DEPTHS * (c - 1), n0 + DEPTHS * c))
        out = append_block(out, blk)
    return out, torch.cat(idx)


def regime_separated(s, k):
    """POSITIVE CONTROL: k genuinely new columns, far apart.

    Evidence must grow with k here.  A mechanism that merely ignores the
    profile stream gives a flat curve for duplicates AND for this, so only the
    contrast between the two identifies redundancy suppression.
    """
    out = clone(s)
    n0 = int(s["mask"].shape[0])
    idx = [torch.arange(DEPTHS)]
    for c in range(1, k):
        blk = head_block(s)
        # spread across the box, far beyond the 1544 km x length scale
        blk["coord"][:, 0] = float((c * 0.37) % 1.0)
        blk["coord"][:, 1] = float((c * 0.61) % 1.0)
        blk["provenance"] = blk["provenance"] + 2000 + c
        idx.append(torch.arange(n0 + DEPTHS * (c - 1), n0 + DEPTHS * c))
        out = append_block(out, blk)
    return out, torch.cat(idx)


def regime_clustered(s, span=0.05):
    """Same profile COUNT, confined to a small box instead of spread out."""
    out = clone(s)
    prof = (out["modality"] == MOD_PROFILE)
    n = int(prof.sum())
    r = np.random.default_rng(args.seed)
    cols = n // DEPTHS
    cx, cy = r.uniform(0.2, 0.8, 2)
    xs = np.repeat(r.uniform(cx - span, cx + span, cols), DEPTHS)
    ys = np.repeat(r.uniform(cy - span, cy + span, cols), DEPTHS)
    out["coord"][prof, 0] = torch.as_tensor(np.clip(xs, 0, 1)[:n],
                                            dtype=out["coord"].dtype)
    out["coord"][prof, 1] = torch.as_tensor(np.clip(ys, 0, 1)[:n],
                                            dtype=out["coord"].dtype)
    return out


def regime_split_patch(s, factor=2):
    """Tile the SAME surface field at a finer patch size.

    ``factor`` x more patch tokens, each with 1/factor^2 the area, so the total
    support is unchanged.  Total evidence must track the support, not the count.
    """
    out = clone(s)
    grid = (out["modality"] != MOD_PROFILE)
    keep = {k: out[k][~grid] for k in PER_TOKEN}          # profiles untouched
    blk = {k: out[k][grid] for k in PER_TOKEN}
    n = int(grid.sum())
    f2 = factor * factor
    sub = {k: blk[k].repeat_interleave(f2, dim=0) for k in PER_TOKEN}
    sub["support_area"] = sub["support_area"] / f2
    # offset the sub-patches inside their parent so they tile it rather than
    # sitting on top of each other (which would be a duplicate, not a split)
    cell_x = 1.0 / max(B.GRID_NX - 1, 1)
    cell_y = 1.0 / max(B.GRID_NY - 1, 1)
    off = torch.tensor([[(i % factor - (factor - 1) / 2) * cell_x * 2,
                         (i // factor - (factor - 1) / 2) * cell_y * 2]
                        for i in range(f2)], dtype=sub["coord"].dtype)
    sub["coord"][:, 0] = (sub["coord"][:, 0] + off[:, 0].repeat(n)).clamp(0, 1)
    sub["coord"][:, 1] = (sub["coord"][:, 1] + off[:, 1].repeat(n)).clamp(0, 1)
    for k in PER_TOKEN:
        out[k] = torch.cat([keep[k], sub[k]], dim=0)
    return out


# =================================================================== measure
def group_mass(s, idx, rho=0.0):
    w, live = omega_of(s, rho)
    sel = torch.zeros_like(live)
    sel[idx] = True
    return float(w[sel & live].double().sum())


def total_mass(s, rho=0.0):
    w, live = omega_of(s, rho)
    return float(w[live].double().sum())


results = {}
print("\n[1] duplication families — group mass of the attacked column", flush=True)
print(f"    {'regime':<14}" + "".join(f"{'k='+str(k):>10}" for k in KS)
      + f"{'k8/k1':>9}{'discount':>10}", flush=True)
for kind, rho in (("exact", 0.0), ("jittered", 0.0),
                  ("dual_stream", args.rho), ("separated", 0.0)):
    row = {}
    for k in KS:
        gm = []
        for s in SAMPLES:
            if kind == "separated":
                a, idx = regime_separated(s, k)
            else:
                a, idx = regime_copies(s, k, kind)
            gm.append(group_mass(a, idx, rho))
        row[k] = float(np.mean(gm))
    ratio = row[8] / row[1]
    disc = 1.0 - (ratio - 1.0) / 7.0
    results[kind] = {"mass": {str(k): row[k] for k in KS},
                     "k8_over_k1": ratio, "discount": disc, "rho": rho}
    print(f"    {kind:<14}" + "".join(f"{row[k]:>10.4f}" for k in KS)
          + f"{ratio:>9.3f}{disc:>10.3f}", flush=True)

print("\n    positive control check: 'separated' must NOT collapse.", flush=True)
sep, exa = results["separated"]["k8_over_k1"], results["exact"]["k8_over_k1"]
print(f"    separated k8/k1 = {sep:.3f} (want ~8 = independent), "
      f"exact = {exa:.3f} (want ~1 = collapsed)", flush=True)
print(f"    SEPARATION = {sep/exa:.2f}x  -> the estimator distinguishes new "
      f"evidence from re-ingested evidence" if sep > 1.5 * exa else
      f"    WARNING: separation only {sep/exa:.2f}x — cannot tell suppression "
      f"from ignoring the stream", flush=True)

print("\n[2] dual-stream vs n_eff theory (Phase 1b)", flush=True)
for rho in (0.0, 0.5, 0.9, 0.99):
    gm = []
    for s in SAMPLES:
        a, idx = regime_copies(s, 2, "dual_stream")
        gm.append(group_mass(a, idx, rho))
    solo = float(np.mean([group_mass(*regime_copies(s, 1, "dual_stream"), rho)
                          for s in SAMPLES]))
    print(f"    rho={rho:<5} 2-stream mass {np.mean(gm):.4f}  solo {solo:.4f}  "
          f"ratio {np.mean(gm)/solo:.3f}   n_eff(2,rho)={n_eff(2, rho):.3f}",
          flush=True)
results["dual_stream_rho"] = {
    str(r): float(np.mean([group_mass(*regime_copies(s, 2, "dual_stream"), r)
                           for s in SAMPLES])) for r in (0.0, 0.5, 0.9, 0.99)}

print("\n[3] clustered vs uniform sampling at equal token count", flush=True)
uni = float(np.mean([total_mass(s) for s in SAMPLES]))
clu = float(np.mean([total_mass(regime_clustered(s)) for s in SAMPLES]))
print(f"    uniform   total evidence {uni:.3f}", flush=True)
print(f"    clustered total evidence {clu:.3f}   ({100*(clu-uni)/uni:+.1f}%)",
      flush=True)
results["clustered"] = {"uniform": uni, "clustered": clu,
                        "rel_change": (clu - uni) / uni}

print("\n[4] token-count invariance — same field, finer patches", flush=True)
base = float(np.mean([total_mass(s) for s in SAMPLES]))
n_base = float(np.mean([float(s["mask"].sum()) for s in SAMPLES]))
sp = [regime_split_patch(s, 2) for s in SAMPLES]
split = float(np.mean([total_mass(a) for a in sp]))
n_split = float(np.mean([float(a["mask"].sum()) for a in sp]))
print(f"    4x4 patches: {n_base:.0f} tokens, evidence {base:.3f}", flush=True)
print(f"    2x2 patches: {n_split:.0f} tokens ({n_split/n_base:.1f}x), "
      f"evidence {split:.3f} ({split/base:.2f}x)", flush=True)
print(f"    -> evidence grew {split/base:.2f}x for a {n_split/n_base:.1f}x "
      f"token count on the SAME field and the SAME total support", flush=True)
results["token_count_invariance"] = {
    "n_tokens_base": n_base, "n_tokens_split": n_split,
    "evidence_base": base, "evidence_split": split,
    "token_ratio": n_split / n_base, "evidence_ratio": split / base}

# ==================================================================== figure
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
for kind, style in (("separated", "o-"), ("jittered", "s-"),
                    ("exact", "^-"), ("dual_stream", "d-")):
    ax[0].plot(KS, [results[kind]["mass"][str(k)] / results[kind]["mass"]["1"]
                    for k in KS], style, label=kind)
ax[0].plot(KS, KS, "k:", lw=1.2, label="independent votes (no collapse)")
ax[0].axhline(1.0, color="grey", ls="--", lw=1, label="perfect collapse")
ax[0].set_xlabel("k copies"); ax[0].set_ylabel("group mass / k=1")
ax[0].set_title("Duplication families"); ax[0].legend(fontsize=8)
ax[0].grid(alpha=.3); ax[0].set_xscale("log", base=2)

rhos = [0.0, 0.5, 0.9, 0.99]
meas = [results["dual_stream_rho"][str(r)] /
        results["dual_stream_rho"]["0.0"] for r in rhos]
ax[1].plot(rhos, meas, "o-", label="measured (2 streams)")
ax[1].plot(rhos, [n_eff(2, r) / n_eff(2, 0.0) for r in rhos], "k--",
           label="n_eff(2, rho) / n_eff(2, 0)")
ax[1].set_xlabel("provenance noise correlation rho")
ax[1].set_ylabel("relative group mass")
ax[1].set_title("Dual-stream collapse vs n_eff theory")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.suptitle("Phase 2 — redundancy regimes (geometry only, frozen estimator)",
             fontsize=12)
fig_path = os.path.join(C.REPORTS, "fig_redundancy_regimes.png")
fig.savefig(fig_path, dpi=140, bbox_inches="tight")
print(f"\nwrote {fig_path}", flush=True)


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


record = {"task": "phase2_redundancy_regimes", "git_commit": git_commit(),
          "seed": args.seed, "months": args.months, "rho": args.rho,
          "jitter_km": args.jitter_km,
          "n_vertical_nodes": args.n_vertical_nodes,
          "n_features": B.N_FEATURES, "results": results, "figure": fig_path}
os.makedirs(C.CACHE, exist_ok=True)
with open(os.path.join(C.CACHE, f"{args.tag}.json"), "w") as f:
    json.dump(record, f, indent=2)
print(f"wrote {os.path.join(C.CACHE, args.tag + '.json')}", flush=True)
print(f"TOTAL {time.time()-t0:.1f}s", flush=True)
