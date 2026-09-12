"""Phase 0 gate — measure and calibrate the DFS operating point.

Registered rule: `doc/phase0_registration.md` (committed BEFORE this script was
run; its commit hash is recorded below as ``registration_commit``).

Why this is a gate, not a warm-up (work_plan.md): at ``s << 1`` the estimator
provably degenerates to per-token area weighting — the NEO/Berner corner — and
the set-level coupling that makes DFS distinct is ``O(s)``. Every mechanism
effect is then capped at about 1% by construction. Redundancy experiments,
robustness sweeps and the Argo confirmation would all be run in the regime
where the method is designed to do nothing.

What the operating point is
---------------------------
``batched_dfs`` whitens each token by its noise and takes ridge leverage in a
joint solve::

    psi~_i  = psi_i / sqrt(lambda_i)          lambda_i = noise_i * |support_i|
    A       = Psi~^T Psi~ + I
    omega_i = psi~_i^T A^-1 psi~_i

Before Phase 0 the quadrature weight was a dimensionless ``1`` and the noise a
pilot constant, so the operating point was a side effect of a normalisation.
Phase 0 makes the weight a physical area (km²) and the noise an error-variance
density in the same units, so

    s_token = |psi~_i|^2 = |support_i| / noise_i

reads as "support area divided by error-variance density", and ``--sweep``
moves it deliberately.

Two ceilings are distinguished, because they call for opposite fixes:

* ``sum omega = F - trace(A^-1) <= F`` — the FEATURE ceiling. If ``sum omega``
  sits at ``F``, evidence is capped by the basis, and raising ``p`` is the fix.
* ``s_token`` small — the NOISE ceiling. Each token is individually negligible,
  ``A -> I``, and the joint solve stops coupling. Lowering the noise is the fix.

Reading either as the other is how the previous cycle was lost, so both are
reported side by side.

Sections
--------
``--operating-point``  the headline diagnostic, legacy config vs Phase 0
``--sweep``            s-family over the noise scale c (geometry only, free)
``--duplicates``       duplicate-collapse family, k = 1..8, per c
``--convergence``      RFF kernel error vs feature count p

The geometry half needs no training and no fitted weights: ``omega`` depends on
support geometry, noise density and variable group alone. That is the half the
work plan calls free, and it is what runs here. Accuracy ``J`` — and therefore
the registered ``s*`` SELECTION — needs GODAS fields and training; when they are
absent this script reports the calibration curve and refuses to pick ``s*``,
per the registration's fallback clause.

Run:
    .venv/bin/python experiments/synthetic/29_dfs_operating_point.py            # all
    .venv/bin/python experiments/synthetic/29_dfs_operating_point.py --convergence
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

from ocean_tokenizer import config as C
from ocean_tokenizer import batched_dfs as B
from ocean_tokenizer.batched_dfs import (RandomFourierBasis, integrate_support,
                                         dfs_omega, variable_group_coords,
                                         to_physical)
from ocean_tokenizer.godas_obs import (ObsConfig, build_sample,
                                        duplicate_profile_attack,
                                        N_VARIABLE_GROUPS, MOD_PROFILE,
                                        MOD_SURF, MOD_SSH)

MOD_NAME = {MOD_PROFILE: "profile", MOD_SURF: "surf", MOD_SSH: "ssh"}
DUP_DEPTHS = 16                      # one profile column = 16 depth tokens


def dup_group_index(n_orig: int, k: int, depths: int = DUP_DEPTHS):
    """Indices of the k copies of the attacked column after the attack.

    ``duplicate_profile_attack`` returns ``original(n_orig) ++ block*(k-1)``,
    so the group is the leading block plus the appended tail — two disjoint
    ranges, not one contiguous prefix.
    """
    return torch.cat([torch.arange(depths),
                      torch.arange(n_orig, n_orig + depths * (k - 1))])
#: registered in doc/phase0_registration.md before any number was read
SWEEP_C = [1e-3, 1e-2, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
DUP_K = [1, 2, 4, 8]

ap = argparse.ArgumentParser()
ap.add_argument("--godas-dir", default="",
                help="real GODAS subset; without it the token GEOMETRY is "
                     "rebuilt on synthetic all-ocean fields (values never "
                     "enter omega, but the land mask does — see the caveat)")
ap.add_argument("--months", type=int, default=12, help="source months sampled")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--operating-point", action="store_true")
ap.add_argument("--sweep", action="store_true")
ap.add_argument("--duplicates", action="store_true")
ap.add_argument("--convergence", action="store_true")
ap.add_argument("--tag", default="dfs_operating_point")
args = ap.parse_args()
if not (args.operating_point or args.sweep or args.duplicates or args.convergence):
    args.operating_point = args.sweep = args.duplicates = args.convergence = True

t0 = time.time()
torch.manual_seed(args.seed)


def git_commit(ref="HEAD"):
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", ref],
                                       text=True).strip()
    except Exception:
        return "unknown"


def registration_commit():
    """Hash of the commit that registered the rule — proves rule-before-number."""
    try:
        return subprocess.check_output(
            ["git", "-C", C.ROOT, "log", "-1", "--format=%H", "--",
             "doc/phase0_registration.md"], text=True).strip()
    except Exception:
        return "unknown"


# ======================================================================= data
def load_fields():
    """GODAS fields if available, else synthetic all-ocean fields.

    ``omega`` is a function of token GEOMETRY (coords, support, noise,
    variable group) and of the liveness mask.  Field *values* never enter it.
    Synthetic fields therefore reproduce the operating point exactly for an
    all-ocean box; what they cannot reproduce is land, which removes tokens and
    would lower the token count.  The synthetic case is thus the dense/upper
    bound on competition, and is labelled as such in the output.
    """
    if args.godas_dir:
        from ocean_tokenizer.godas import load_godas
        f = load_godas(args.godas_dir)
        return f, "godas"
    rng = np.random.default_rng(args.seed)
    T, Z, Y, X = max(args.months + 2, 4), 16, B.GRID_NY, B.GRID_NX
    fields = {"TEMP": rng.standard_normal((T, Z, Y, X)),
              "SALT": rng.standard_normal((T, Z, Y, X)),
              "SSH": rng.standard_normal((T, Y, X))}
    return fields, "synthetic-all-ocean"


FIELDS, SOURCE = load_fields()
N_SRC = min(args.months, FIELDS["TEMP"].shape[0] - 1)
print(f"[{args.tag}] source={SOURCE} months={N_SRC} "
      f"grid={FIELDS['TEMP'].shape[1:]} device=cpu", flush=True)


def samples():
    cfg = ObsConfig(train=False)
    rng = np.random.default_rng(args.seed)
    for t in range(N_SRC):
        yield build_sample(FIELDS, t, cfg, rng, lead=0)


SAMPLES = list(samples())
print(f"  {len(SAMPLES)} samples, {int(SAMPLES[0]['mask'].sum())} live tokens each",
      flush=True)


# ================================================================== estimator
def omega_of(s: dict, *, n_features: int, physical: bool, noise_scale: float,
             length_scales=None):
    """omega and its inputs for one sample under a given configuration."""
    grp = variable_group_coords(s["variable_group"], N_VARIABLE_GROUPS)
    if physical:
        coord = to_physical(s["coord"])
        base = length_scales or B.LENGTH_SCALES_KM
        weight = s["support_area"].to(torch.float64)[:, None]
    else:
        coord = s["coord"]
        base = length_scales or B.LENGTH_SCALES
        weight = torch.ones(coord.shape[0], 1, dtype=torch.float64)
    basis = RandomFourierBasis(
        n_features, tuple(base) + variable_group_coords.scales(N_VARIABLE_GROUPS),
        B.BASIS_SEED)
    full = torch.cat([coord, grp], dim=-1)
    psi, lam = integrate_support(basis, full[:, None, :], weight,
                                 s["noise_density"] * noise_scale)
    live = s["mask"] & s["support_mask"]
    w = dfs_omega(psi, lam, live)
    # per-token operating point BEFORE the joint solve
    s_tok = (psi.to(torch.float64) ** 2).sum(-1) / lam.to(torch.float64)
    return w, s_tok, live, psi, lam


def summarise(w, s_tok, live, n_features, modality=None):
    wl = w[live].double()
    mean_w = float(wl.mean()) if wl.numel() else float("nan")
    out = {
        "n_live": int(live.sum()),
        "sum_omega": float(wl.sum()),
        "feature_ceiling": int(n_features),
        "sum_omega_over_F": float(wl.sum()) / n_features,
        "mean_omega": mean_w,
        "median_omega": float(wl.median()) if wl.numel() else float("nan"),
        "max_omega": float(wl.max()) if wl.numel() else float("nan"),
        "s_measured": mean_w / max(1.0 - mean_w, 1e-12),
        "s_token_mean": float(s_tok[live].double().mean()),
        "s_token_median": float(s_tok[live].double().median()),
    }
    if modality is not None:
        out["by_modality"] = {}
        for m, name in MOD_NAME.items():
            sel = live & (modality == m)
            if bool(sel.any()):
                out["by_modality"][name] = {
                    "n": int(sel.sum()),
                    "mean_omega": float(w[sel].double().mean()),
                    "sum_omega": float(w[sel].double().sum()),
                    "s_token_mean": float(s_tok[sel].double().mean()),
                }
    return out


def config_stats(*, n_features, physical, noise_scale):
    """Average the diagnostic over the sampled months."""
    acc, per = None, []
    for s in SAMPLES:
        w, st, live, _, _ = omega_of(s, n_features=n_features,
                                     physical=physical, noise_scale=noise_scale)
        per.append(summarise(w, st, live, n_features, s["modality"]))
    keys = ("sum_omega", "sum_omega_over_F", "mean_omega", "median_omega",
            "max_omega", "s_measured", "s_token_mean", "s_token_median",
            "n_live")
    acc = {k: float(np.mean([p[k] for p in per])) for k in keys}
    acc["feature_ceiling"] = n_features
    mods = set().union(*[set(p.get("by_modality", {})) for p in per])
    acc["by_modality"] = {
        m: {k: float(np.mean([p["by_modality"][m][k] for p in per
                              if m in p["by_modality"]]))
            for k in ("n", "mean_omega", "sum_omega", "s_token_mean")}
        for m in sorted(mods)}
    return acc


record = {
    "task": "phase0_dfs_operating_point",
    "registration": "doc/phase0_registration.md",
    "registration_commit": registration_commit(),
    "git_commit": git_commit(),
    "source": SOURCE, "months": N_SRC, "seed": args.seed,
    "geometry": {
        "box_extent_x_km_y_km_z_m_t_months": list(B.BOX_EXTENT),
        "length_scales_normalised": list(B.LENGTH_SCALES),
        "length_scales_physical_km_km_m_months": list(B.LENGTH_SCALES_KM),
        "cell_area_km2": B.cell_area_km2(),
        "profile_support_km2": B.profile_support_area_km2(),
        "patch_4x4_support_km2": B.patch_support_area_km2(4, 4),
        "noise_area_point_km2": B.NOISE_AREA_POINT_KM2,
        "noise_area_patch_km2": B.NOISE_AREA_PATCH_KM2,
    },
}

# ========================================================= 1. operating point
if args.operating_point:
    print("\n[1] operating point — legacy (pre-Phase-0) vs Phase 0", flush=True)
    legacy = config_stats(n_features=B.N_FEATURES_LEGACY, physical=False,
                          noise_scale=1.0)
    # the legacy row must use the legacy dimensionless noise, not the km² one
    legacy_noise = {}
    for s in SAMPLES:
        s["_keep_noise"] = s["noise_density"].clone()
    for s in SAMPLES:
        nd = torch.where(s["modality"] == MOD_PROFILE,
                         torch.full_like(s["noise_density"],
                                         B.NOISE_DENSITY_POINT_LEGACY),
                         torch.full_like(s["noise_density"],
                                         B.NOISE_DENSITY_PATCH_LEGACY))
        s["noise_density"] = nd
    legacy = config_stats(n_features=B.N_FEATURES_LEGACY, physical=False,
                          noise_scale=1.0)
    for s in SAMPLES:
        s["noise_density"] = s.pop("_keep_noise")
    phase0 = config_stats(n_features=B.N_FEATURES, physical=True,
                          noise_scale=1.0)
    record["operating_point"] = {"legacy": legacy, "phase0": phase0}
    for name, r in (("legacy (F=32, unit support)", legacy),
                    ("phase0 (F=256, km² support)", phase0)):
        print(f"  {name}", flush=True)
        print(f"    s_token   mean {r['s_token_mean']:.4g}  "
              f"median {r['s_token_median']:.4g}", flush=True)
        print(f"    omega     mean {r['mean_omega']:.4g}  "
              f"max {r['max_omega']:.4g}", flush=True)
        print(f"    sum omega {r['sum_omega']:.2f} / F={r['feature_ceiling']}"
              f"  ({100*r['sum_omega_over_F']:.1f}% of the feature ceiling)",
              flush=True)
        print(f"    s_measured = mean_omega/(1-mean_omega) = "
              f"{r['s_measured']:.4g}", flush=True)
        for m, v in r["by_modality"].items():
            print(f"      {m:8s} n={v['n']:5.0f}  s_token {v['s_token_mean']:.4g}"
                  f"  mean omega {v['mean_omega']:.4g}", flush=True)

# =================================================================== 2. sweep
if args.sweep:
    print("\n[2] s-family — sweeping the noise scale c (geometry only)", flush=True)
    rows = []
    for c in SWEEP_C:
        r = config_stats(n_features=B.N_FEATURES, physical=True, noise_scale=c)
        rows.append({"c": c, **{k: r[k] for k in
                                ("s_token_mean", "mean_omega", "sum_omega",
                                 "sum_omega_over_F", "s_measured")}})
        print(f"  c={c:<7g} s_token {r['s_token_mean']:>10.4g}  "
              f"mean omega {r['mean_omega']:.4g}  sum omega "
              f"{r['sum_omega']:8.2f} ({100*r['sum_omega_over_F']:5.1f}% of F)"
              f"  s_meas {r['s_measured']:.4g}", flush=True)
    record["sweep"] = rows

# ============================================================== 3. duplicates
if args.duplicates:
    print("\n[3] duplicate collapse — k copies of one profile column", flush=True)
    print("    ideal: group mass flat in k (copies carry no new evidence);", flush=True)
    print("    independent-vote failure: group mass grows ~linearly in k", flush=True)
    dup = []
    # swept over the SAME registered c grid as the s-family, so "does the
    # discount respond to s?" is answered across the whole family and not only
    # at the high-noise end where it cannot
    for c in SWEEP_C:
        row = {"c": c, "k": {}}
        for k in DUP_K:
            g = []
            for s in SAMPLES:
                try:
                    a = duplicate_profile_attack(s, k)
                except ValueError:
                    continue
                w, st, live, _, _ = omega_of(a, n_features=B.N_FEATURES,
                                             physical=True, noise_scale=c)
                # duplicate_profile_attack keeps the original column at [0:D]
                # and APPENDS its k-1 copies at the tail, so the attacked group
                # is those two ranges — NOT w[:D*k], which is the first k
                # DISTINCT columns of the original set and grows ~k for reasons
                # that have nothing to do with duplication.
                idx = dup_group_index(int(s["mask"].shape[0]), k, DUP_DEPTHS)
                assert idx.numel() == DUP_DEPTHS * k
                assert torch.allclose(a["coord"][idx].reshape(k, DUP_DEPTHS, -1),
                                      a["coord"][:DUP_DEPTHS][None]), \
                    "attacked group is not k copies of one column"
                g.append(float(w[idx][live[idx]].double().sum()))
            row["k"][str(k)] = float(np.mean(g)) if g else float("nan")
        base = row["k"]["1"]
        row["ratio_k8_over_k1"] = row["k"]["8"] / base if base else float("nan")
        # 1.0 = perfect collapse (k copies carry one copy's evidence);
        # 8.0 = no collapse at all
        row["discount"] = 1.0 - (row["ratio_k8_over_k1"] - 1.0) / 7.0
        dup.append(row)
        ks = "  ".join(f"k={k}: {row['k'][str(k)]:.4f}" for k in DUP_K)
        print(f"  c={c:<6g} {ks}   k8/k1={row['ratio_k8_over_k1']:.3f}  "
              f"discount={row['discount']:.3f}", flush=True)
    record["duplicates"] = dup

# ============================================================= 4. convergence
if args.convergence:
    print("\n[4] RFF convergence — kernel error vs feature count", flush=True)
    rng = np.random.default_rng(args.seed)
    ell = torch.as_tensor(B.LENGTH_SCALES_KM, dtype=torch.float64)
    x = torch.as_tensor(rng.uniform(0, 1, (400, 4)), dtype=torch.float64)
    x = x * torch.as_tensor(B.BOX_EXTENT, dtype=torch.float64)
    du = (x[:, None, :] - x[None, :, :]) / ell
    k_exact = torch.exp(-0.5 * (du ** 2).sum(-1))
    conv = []
    for p in (32, 64, 128, 256, 512, 1024):
        errs = []
        for sd in range(5):                      # basis seed variance matters
            basis = RandomFourierBasis(p, B.LENGTH_SCALES_KM, seed=sd)
            phi = basis(x)
            k_rff = phi @ phi.T
            errs.append(float(torch.sqrt(((k_rff - k_exact) ** 2).mean())))
        conv.append({"p": p, "rmse_mean": float(np.mean(errs)),
                     "rmse_std": float(np.std(errs)),
                     "rel_pct": 100 * float(np.mean(errs))})
        print(f"  p={p:<5d} kernel RMSE {np.mean(errs):.4f} "
              f"+/- {np.std(errs):.4f}   ({100*np.mean(errs):.2f}% of k(x,x)=1)",
              flush=True)
    record["convergence"] = conv

# ================================================================== s* + exit
mean_s = (record.get("operating_point", {}).get("phase0", {}) or {}).get("s_measured")
record["s_star"] = None
record["s_star_status"] = (
    "NOT SELECTED — the registered rule selects s* on validation accuracy J "
    "(one-standard-error rule, three seeds). J requires GODAS fields and "
    "training, which are unavailable here, so the registration's fallback "
    "clause applies: report the geometry calibration curve and defer s* rather "
    "than substituting a geometry-only criterion.")

os.makedirs(C.CACHE, exist_ok=True)
out = os.path.join(C.CACHE, f"{args.tag}.json")
with open(out, "w") as f:
    json.dump(record, f, indent=2)
print(f"\nwrote {out}", flush=True)
print(f"registration commit {record['registration_commit'][:12]} "
      f"(rule registered before this run)", flush=True)
phase0 = (record.get("operating_point", {}) or {}).get("phase0", {})
s_tok = phase0.get("s_token_mean")
if s_tok is not None:
    # Read the operating point on s_token, NOT on omega/(1-omega).
    #
    # omega = s/(1+s) is the LONE-token identity, so inverting it as
    # s_hat = omega/(1-omega) is only valid for a set of one.  With N tokens
    # competing for a bounded evidence pool every omega_i is pushed well below
    # s_i/(1+s_i), so the inversion is biased LOW by roughly the competition
    # factor -- here 0.95 -> 0.070, a factor of ~14.  Reporting s_hat as "the
    # operating point" is what makes a working mechanism look degenerate.
    verdict = ("s_token = O(1): the estimator is OUT of the degenerate corner"
               if s_tok >= 0.1 else
               "s_token << 1: at the NEO/Berner corner")
    print(f"EXIT CRITERION 1: s_token = {s_tok:.4g} -> {verdict}", flush=True)
    print(f"  (for reference, the lone-token inversion omega/(1-omega) reads "
          f"{mean_s:.4g}; it is biased low by competition and is NOT the "
          f"operating point of a multi-token set)", flush=True)
dups = record.get("duplicates") or []
prior = next((d for d in dups if d["c"] == 1.0), None)
if prior:
    print(f"MECHANISM: duplicate discount at the prior (c=1) = "
          f"{prior['discount']:.3f}  (1.0 = perfect collapse, 0.0 = independent "
          f"votes)", flush=True)
print(f"TOTAL {time.time()-t0:.1f}s", flush=True)
