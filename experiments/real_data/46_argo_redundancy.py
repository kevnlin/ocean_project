"""P2 — redundancy stress on REAL Argo, with the positive control.

The plan asks for k = 1,2,4,8,16,32 across five duplication families, and is
explicit that the whole package is void without the control:

    "adding genuinely new independent observations increases useful evidence.
     If either track fails this control, stop and investigate before using the
     redundancy result."

That warning is well aimed.  A method that simply ignores the profile stream
produces a flat evidence curve under duplication and looks exactly like a method
that correctly suppresses redundancy.  Only the contrast against `separated` —
k genuinely different profiles, far apart — distinguishes them.

What real Argo adds over `39_redundancy_regimes.py`
--------------------------------------------------
That script ran on synthetic geometry, and its provenance groups were column
indices.  Here the duplicated object is a real profile with a real WMO, and the
two middle families become a question the synthetic version could not pose:

    same_provenance         k reports sharing ONE platform id — the dual-stream
                            float, i.e. one instrument's data arriving twice
                            through two processing streams.  Should collapse.
    independent_provenance  k reports of the same water from k DIFFERENT
                            platforms.  Should NOT fully collapse: independent
                            instruments genuinely do reduce the error, by
                            sqrt(k) in the ideal case.

A method that collapses both is over-suppressing; one that collapses neither is
counting. The gap between them is the mechanism's actual claim.

Two quantities are reported per (family, k):
  * **evidence mass** — the estimator's own omega, summed over the attacked
    tokens.  Geometry only; needs no trained model.
  * **accuracy** — RMSE on the held-out floats with the attack in the input.
    Needs a checkpoint, and is the number that decides whether the evidence
    story matters.

  .venv/bin/python experiments/46_argo_redundancy.py --smoke
  .venv/bin/python experiments/46_argo_redundancy.py --checkpoint <path> --seed 1234
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch

from ocean_tokenizer import protocol as P
from ocean_tokenizer import argo_experiments as E
from ocean_tokenizer import batched_dfs as B
from ocean_tokenizer.argo_obs import (ArgoCohort, ArgoNorm, ArgoObsConfig,
                                      build_argo_sample)
from ocean_tokenizer.batched_dfs import (to_physical, integrate_support,
                                         dfs_omega, vertical_quadrature,
                                         variable_group_coords, RandomFourierBasis,
                                         LENGTH_SCALES_KM, BASIS_SEED, N_FEATURES)
from ocean_tokenizer.clustered_ci import accumulate, ci_rmse, ci_difference
from ocean_tokenizer.godas_obs import N_VARIABLE_GROUPS
from ocean_tokenizer.godas_model import build_row, ROWS

K_VALUES = (1, 2, 4, 8, 16, 32)
CHANNELS = P.CHANNELS

ap = argparse.ArgumentParser()
ap.add_argument("--region", default="gulfstream", choices=list(P.REGIONS))
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--rho", type=float, default=0.9,
                help="within-provenance noise correlation for the same_provenance "
                     "family; 0 reproduces independent-noise whitening")
ap.add_argument("--months", type=int, default=24)
ap.add_argument("--n-profiles", type=int, default=24)
ap.add_argument("--n-vertical-nodes", type=int, default=3)
ap.add_argument("--eval-split", default="development")
ap.add_argument("--region-kernel", action="store_true",
                help="must match how the checkpoints were trained")
ap.add_argument("--with-oi", action="store_true", default=True,
                help="include the deterministic OI row, which the plan lists "
                     "among the five to compare and which needs no checkpoint")
ap.add_argument("--checkpoints", default="",
                help="comma-separated row=path pairs; omit for geometry only")
ap.add_argument("--cohort", default=None)
ap.add_argument("--output", default=None)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COHORT = args.cohort or os.path.join(ROOT, "data", "argo_cohort", f"{args.region}.nc")
OUT = args.output or os.path.join(ROOT, "outputs", f"argo_P2_{args.region}")
os.makedirs(OUT, exist_ok=True)
if args.smoke:
    args.months = 4
dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()

c = ArgoCohort.load(COHORT)
norm = ArgoNorm.fit(c, "train")
cfg = ArgoObsConfig(n_profiles=args.n_profiles, n_queries=512, train=False)
months = [int(m) for m in c.months_in(args.eval_split)
          if c.month(m, float_split="cohort_float").size
          and c.month(m, float_split="heldout_float").size][:args.months]
print(f"P2 redundancy  region={args.region} seed={args.seed} rho={args.rho} "
      f"months={len(months)} device={dev}", flush=True)

BASIS = RandomFourierBasis(
    N_FEATURES,
    tuple(LENGTH_SCALES_KM) + variable_group_coords.scales(N_VARIABLE_GROUPS),
    BASIS_SEED)

PER_TOKEN = ("coord", "value", "value_mask", "mask", "support_mask", "modality",
             "variable_group", "noise_density", "support_area", "support_dz",
             "provenance")


def omega_of(s: dict, rho: float = 0.0):
    """Whitened ridge leverage per token — the estimator's own evidence."""
    coord = to_physical(s["coord"])
    area = s["support_area"].to(torch.float64)
    if args.n_vertical_nodes > 1:
        zq, wq = vertical_quadrature(coord[:, 2], s["support_dz"].to(torch.float64),
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


def attacked_sample(month: int, family: str, k: int, rng) -> tuple[dict, np.ndarray]:
    """Build the sample with the k-fold attack applied, and the attacked slice.

    The attack is applied by REBUILDING the sample from a modified row list, so
    the token layout, masks and supports are produced by the one code path that
    builds every other sample.  Jitter and provenance overrides are then applied
    to the attacked tokens only.
    """
    base = E.layout(c, month, args.n_profiles, "natural", args.seed)
    if base.size == 0:
        return None, None
    rows, edits = E.duplicate_rows(c, base, k, family, args.seed, month)
    s = build_argo_sample(c, norm, month, cfg=cfg, rng=rng, lead=0,
                          profile_rows=rows)
    if s is None:
        return None, None
    L = c.levels.size
    attacked = np.arange(k * L)                    # the first k profiles' tokens

    if "jitter_km" in edits and edits["jitter_km"] is not None:
        # move the copies in the x coordinate by the jitter, converted from km
        # to the normalised box coordinate the tokens live in
        x_km = B.BOX_EXTENT[0]
        dx = np.repeat(edits["jitter_km"], L) / max(x_km, 1e-9)
        co = s["coord"].clone()
        co[attacked, 0] = co[attacked, 0] + torch.as_tensor(dx, dtype=co.dtype)
        s["coord"] = co
    ov = edits.get("provenance_override")
    if ov is not None:
        pr = s["provenance"].clone()
        pr[attacked] = torch.as_tensor(np.repeat(ov, L), dtype=pr.dtype)
        s["provenance"] = pr
    return s, attacked


# The sample RNG must NOT depend on k.
#
# It did, and the whole accuracy half of this package measured the wrong thing.
# `build_argo_sample` draws the 512 target queries from this generator, so
# seeding it with k gave every k a DIFFERENT set of held-out queries, and
# "RMSE change vs k=1" compared two unrelated draws. The noise that introduced
# (+/-0.008 TEMP) was an order of magnitude larger than the effect being
# measured (~0.0009), and it was nearly identical across all five families --
# which is the tell: five families with completely different input tokens
# cannot move the score in lockstep unless what is moving is the query draw.
#
# With the draw held fixed, the families separate the way the claim predicts:
# DFS is flat under `exact` (-0.00002 at k=8) and responds to `separated`
# (-0.00089), while Uniform responds to both almost equally (-0.00057 vs
# -0.00069). Track B found this by seeding [seed, month, lead] as every other
# evaluation in the project does.
# -------------------------------------------------------------- geometry
print("\n[1] evidence mass of the attacked profile", flush=True)
geom: dict = {}
for family in E.REDUNDANCY_FAMILIES:
    rho = args.rho if family in ("same_provenance", "exact") else 0.0
    row = {}
    for k in K_VALUES:
        vals, tot = [], []
        for m in months:
            rng = np.random.default_rng([args.seed, m, 0])
            s, att = attacked_sample(m, family, k, rng)
            if s is None:
                continue
            w, live = omega_of(s, rho)
            sel = torch.zeros_like(live); sel[att] = True
            vals.append(float(w[sel & live].double().sum()))
            tot.append(float(w[live].double().sum()))
        row[k] = dict(group_mass=float(np.mean(vals)) if vals else float("nan"),
                      total_mass=float(np.mean(tot)) if tot else float("nan"))
    g1 = row[1]["group_mass"]
    for k in K_VALUES:
        row[k]["growth_vs_k1"] = (row[k]["group_mass"] / g1) if g1 else float("nan")
    geom[family] = {"rho": rho, "by_k": row}
    print(f"  {family:24s} " +
          "  ".join(f"k{k}={row[k]['growth_vs_k1']:.2f}x" for k in K_VALUES),
          flush=True)

# ------------------------------------------------ the positive control
exact8 = geom["exact"]["by_k"][8]["growth_vs_k1"]
sep8 = geom["separated"]["by_k"][8]["growth_vs_k1"]
same8 = geom["same_provenance"]["by_k"][8]["growth_vs_k1"]
indep8 = geom["independent_provenance"]["by_k"][8]["growth_vs_k1"]
control_passes = bool(np.isfinite(sep8) and np.isfinite(exact8) and sep8 > exact8)
print(f"\n  positive control: separated k8 = {sep8:.2f}x vs exact k8 = "
      f"{exact8:.2f}x  ->  {'PASS' if control_passes else 'FAIL'}", flush=True)
print(f"  provenance contrast: same={same8:.2f}x  independent={indep8:.2f}x",
      flush=True)

# -------------------------------------------------------------- accuracy
acc: dict = {}
ckpts = [p for p in args.checkpoints.split(",") if p.strip()]
# The plan names five rows for the accuracy half: DFS, Uniform, Count,
# thinning + superobbing, and OI. The first three come from checkpoints; the
# preprocessing ladder rows do too; OI is deterministic and needs none, so it
# is appended here rather than being quietly dropped for lacking a .pt.
specs = [(sp.split("=", 1)[0], sp.split("=", 1)[1]) for sp in ckpts]
if args.with_oi:
    specs.append(("objective_interpolation", None))

for row, path in specs:
    if row != "objective_interpolation" and row not in ROWS:
        raise SystemExit(f"unknown row {row!r}")
    if row == "objective_interpolation":
        from ocean_tokenizer.objective_interpolation import (
            ObjectiveInterpolation, OISettings)
        model = ObjectiveInterpolation(OISettings()).to(dev)
    else:
        # Score the REGISTERED model, as registered.
        #
        # This used to pass `provenance_rho=args.rho` (0.9), while P0 -- which
        # trained these checkpoints -- and P1, P5 and P7 all use the default
        # 0.0. So the accuracy table described a differently configured model
        # from the P0 table it is read against, its k=1 baseline was not the
        # registered model, and the two could not be placed on one ladder.
        #
        # `rho` still does its documented job in the evidence half above, where
        # it is applied PER FAMILY (0.9 for same_provenance and exact, 0.0
        # otherwise). That is the within-provenance noise correlation the flag
        # is for. Applying it uniformly to the model -- including under the
        # `separated` positive control, where there is no shared provenance to
        # correlate -- was never what the flag described.
        model = build_row(row,
                          region=(args.region if args.region_kernel else None)).to(dev)
        model.load_state_dict(torch.load(path, map_location=dev))
        model.eval()
    per = {}
    stats_by_k: dict = {}
    is_oi = row == "objective_interpolation"
    for family in E.REDUNDANCY_FAMILIES:
        per[family] = {}
        pred_at_k1 = {}
        for k in K_VALUES:
            labs = {c: [] for c in CHANNELS}
            errs = {c: [] for c in CHANNELS}
            preds = []
            for m in months:
                rng = np.random.default_rng([args.seed, m, 0])
                s, _ = attacked_sample(m, family, k, rng)
                if s is None:
                    continue
                sd = {kk: (v.to(dev) if torch.is_tensor(v) else v)
                      for kk, v in s.items()}
                with torch.no_grad():
                    if is_oi:
                        live = sd["mask"] & sd["value_mask"].all(dim=-1)
                        pred = model(sd["query"],
                                     torch.zeros(sd["query"].shape[0],
                                                 dtype=sd["query"].dtype, device=dev),
                                     sd["coord"][live],
                                     sd["value"][live].to(sd["query"].dtype),
                                     sd["noise_density"][live]).to(torch.float32)
                    else:
                        pred = model(sd)
                pn = pred.cpu().numpy()
                e = ((pred - sd["target"]) ** 2).cpu().numpy()
                msk = sd["target_mask"].cpu().numpy()
                w = np.asarray(sd["target_wmo"])
                preds.append(pn)
                for j, ch in enumerate(CHANNELS):
                    ok = msk[:, j]
                    if ok.any():
                        labs[ch].append(w[ok]); errs[ch].append(e[ok, j])
            if not preds:
                continue
            allp = np.concatenate(preds)
            entry = {}
            for ch in CHANNELS:
                if labs[ch]:
                    st = accumulate(np.concatenate(labs[ch]),
                                    np.concatenate(errs[ch]),
                                    cluster_unit="wmo", channel=ch, method=row)
                    entry[f"rmse_{ch}"] = st.rmse()
                    entry["n_wmos"] = st.n_clusters
                    # keep the per-cluster statistics: the CI on the k-vs-k1
                    # difference is computed from them below, and it cannot be
                    # re-derived from the pooled RMSE afterwards
                    stats_by_k.setdefault(family, {}).setdefault(ch, {})[k] = st
            if k == 1:
                pred_at_k1[family] = allp
            # PREDICTION change: how far duplication moves the OUTPUT, separate
            # from whether that movement helps. A method can be pushed a long
            # way by duplicates and still score similarly, and the plan asks for
            # both quantities.
            p1 = pred_at_k1.get(family)
            if p1 is not None and p1.shape == allp.shape:
                entry["prediction_change_vs_k1"] = float(
                    np.mean(np.abs(allp - p1)))
            per[family][k] = entry
        for ch in CHANNELS:
            base = per[family].get(1, {}).get(f"rmse_{ch}", float("nan"))
            for k in K_VALUES:
                if k in per[family] and f"rmse_{ch}" in per[family][k]:
                    per[family][k][f"error_change_{ch}_vs_k1"] = (
                        per[family][k][f"rmse_{ch}"] - base)
            # A CONFIDENCE INTERVAL on that change.
            #
            # These are ~1e-4 quantities and were previously reported as bare
            # point estimates, which is not enough to say whether `separated`
            # genuinely helps and `exact` genuinely does not. k and k=1 are
            # scored on the SAME months, queries and floats -- the sample RNG no
            # longer depends on k -- so the two arms are paired and share one
            # draw of cluster indices, exactly as the P1 layout gap does.
            st1 = stats_by_k.get(family, {}).get(ch, {}).get(1)
            for k in K_VALUES:
                stk = stats_by_k.get(family, {}).get(ch, {}).get(k)
                if st1 is None or stk is None or k == 1:
                    continue
                stk.method, st1.method = f"{row}_k{k}", f"{row}_k1"
                try:
                    iv = ci_difference(stk, st1, n_boot=4000, seed=args.seed)
                except ValueError:
                    continue
                per[family][k][f"error_change_{ch}_ci"] = {
                    "point": iv.point, "lo": iv.lo, "hi": iv.hi,
                    "excludes_zero": iv.excludes_zero,
                    "n_clusters": iv.n_clusters}
    acc[row] = per
    print(f"\n  {row}: " + "  ".join(
        f"{f}@k8 dT{per[f].get(8,{}).get('error_change_TEMP_vs_k1', float('nan')):+.4f}"
        for f in E.REDUNDANCY_FAMILIES), flush=True)

results = {
    "evidence_mass": geom,
    "accuracy": acc,
    "k_values": list(K_VALUES),
    "redundancy_control": {
        "passes": control_passes,
        "separated_k8_growth": sep8, "exact_k8_growth": exact8,
        "rule": "adding genuinely new independent observations must increase "
                "evidence more than re-ingesting the same observation"},
    "provenance_contrast": {"same_provenance_k8": same8,
                            "independent_provenance_k8": indep8},
}
from ocean_tokenizer.crosscheck import build_comparable
# Evidence growth AND accuracy. The plan's P2 asks the tracks to compare
# "TEMP amplification, SALT amplification, prediction change, error change"
# alongside the evidence mass; leaving accuracy out of the comparison block
# meant the cross-check verified the geometry half of the package and silently
# skipped the half that decides whether the geometry matters.
comparable_rmse = {
    f"{fam}|k{k}": {"evidence_growth": geom[fam]["by_k"][k]["growth_vs_k1"]}
    for fam in E.REDUNDANCY_FAMILIES for k in K_VALUES}
for row_, fams_ in acc.items():
    for fam_, byk_ in fams_.items():
        for k_, e_ in byk_.items():
            comparable_rmse[f"{row_}|{fam_}|k{k_}"] = {
                m: e_[m] for m in ("rmse_TEMP", "rmse_SALT",
                                   "error_change_TEMP_vs_k1",
                                   "error_change_SALT_vs_k1",
                                   "prediction_change_vs_k1") if m in e_}
            # The k=8 interval on the two families that carry the claim. Every
            # k is intervalled in `results`; only these reach the cross-check,
            # so its conclusion table stays about the claim rather than about
            # 180 Monte-Carlo endpoints.
            if int(k_) == 8 and fam_ in ("exact", "separated"):
                iv_ = e_.get("error_change_TEMP_ci")
                if iv_:
                    comparable_rmse[f"{row_}|{fam_}|k8"].update(
                        {f"ci_{x}": iv_[x] for x in ("point", "lo", "hi",
                                                     "excludes_zero")})
results["comparable"] = build_comparable(
    rmse=comparable_rmse,
    n_months=len(months),
    redundancy_control={"passes": control_passes,
                        "separated_k8_growth": sep8,
                        "exact_k8_growth": exact8})

warn = []
if not control_passes:
    warn.append("POSITIVE CONTROL FAILED — per the plan, the redundancy result "
                "must not be used until this is investigated")

art = P.ResultArtifact(
    package="P2", track="A", region=args.region, results=results,
    seeds=[args.seed], command=" ".join(sys.argv),
    counts={"months": len(months), "n_profiles": args.n_profiles,
            "eval_split": args.eval_split},
    warnings=warn,
    notes="Duplication applied to real Argo profiles; provenance is the WMO."
    ).finalize(ROOT)
path = os.path.join(OUT, f"artifact_P2_seed{args.seed}.json")
sha = art.write(path)
print(f"\nartifact: {path}\n  sha256 {sha}\ntotal {time.time()-t0:.0f}s")
