"""P6 — post-hoc predictive uncertainty on frozen real-Argo checkpoints.

The plan approves uncertainty as a **post-hoc module only**: frozen base
checkpoints, and *"the uncertainty module may not change the registered mean
prediction."* It then names three tests, all of which are asserted here against
the REAL checkpoints rather than against a fixture:

    base checkpoint hash unchanged
    zero gradient into the base model
    bit-identical mean prediction before / after attaching the head

`ocean_tokenizer.uncertainty` makes those structural rather than procedural: the
base is `requires_grad_(False)` in eval, the head consumes detached features, and
`forward` returns the base's own tensor for the mean. This script verifies that
the structure survives contact with a trained checkpoint and a real optimizer.

What is evaluated (plan P6)
---------------------------
CRPS, spread-skill, interval coverage, reliability (PIT), and **calibration
versus sparsity** — the last one matters most here. A calibrator fitted at one
observation density is not entitled to that calibration at another, and the
sparsity sweep is what shows whether the head widens its intervals when the
input thins or keeps quoting the density it was trained at.

Scores are clustered by held-out WMO like every other number in the study.

  .venv/bin/python experiments/real_data/54_argo_uncertainty.py --region gulfstream
  .venv/bin/python experiments/real_data/54_argo_uncertainty.py --smoke
"""
from __future__ import annotations

import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch

from ocean_tokenizer import protocol as P
from ocean_tokenizer import argo_experiments as E
from ocean_tokenizer.argo_obs import (ArgoCohort, ArgoNorm, ArgoObsConfig,
                                      build_argo_sample, training_rows)
from ocean_tokenizer.clustered_ci import accumulate, ci_rmse_across_seeds
from ocean_tokenizer.godas_model import build_row
from ocean_tokenizer.uncertainty import (CalibratedModel, UncertaintyHead,
                                         state_dict_hash, gaussian_nll,
                                         crps_gaussian, spread_skill,
                                         interval_coverage, reliability)

CHANNELS = P.CHANNELS

ap = argparse.ArgumentParser()
ap.add_argument("--region", default="gulfstream", choices=list(P.REGIONS))
ap.add_argument("--row", default="dfs_expertlocal_cbottle")
ap.add_argument("--seeds", default="1234,1235,1236")
ap.add_argument("--checkpoint-dir", default=None)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--n-profiles", type=int, default=24)
ap.add_argument("--queries", type=int, default=512)
ap.add_argument("--eval-split", default="development")
ap.add_argument("--output", default=None)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CKDIR = args.checkpoint_dir or os.path.join(ROOT, "outputs", f"argo_P0_{args.region}")
OUT = args.output or os.path.join(ROOT, "outputs", f"argo_P6_{args.region}")
os.makedirs(OUT, exist_ok=True)
seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
if args.smoke:
    args.steps, seeds = 60, seeds[:1]
dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()

c = ArgoCohort.load(os.path.join(ROOT, "data", "argo_cohort", f"{args.region}.nc"))
norm = ArgoNorm.fit(c, "train")
cfg_tr = ArgoObsConfig(n_profiles=args.n_profiles, n_queries=args.queries, train=True)
cfg_ev = ArgoObsConfig(n_profiles=args.n_profiles, n_queries=args.queries, train=False)


def eligible(split):
    return [int(m) for m in c.months_in(split)
            if c.month(m, float_split="cohort_float").size
            and c.month(m, float_split="heldout_float").size]


TRM, EVM = eligible("train"), eligible(args.eval_split)
print(f"P6 uncertainty  region={args.region} row={args.row} seeds={seeds} "
      f"device={dev}\n  {len(TRM)} train / {len(EVM)} {args.eval_split} months",
      flush=True)

results = {"per_seed": {}, "guarantees": {}, "sparsity": {}}
warn = []

for sd in seeds:
    ck = os.path.join(CKDIR, f"{args.row}_s{sd}.pt")
    if not os.path.exists(ck):
        warn.append(f"missing checkpoint {ck}"); continue
    base = build_row(args.row).to(dev)
    base.load_state_dict(torch.load(ck, map_location=dev))

    # ---- guarantee 1 + 3, measured BEFORE the head exists -------------
    file_hash = P.sha256_file(ck)
    sd_hash_before = state_dict_hash(base.state_dict())
    base.eval()
    probe_rng = np.random.default_rng([sd, 99])
    probe = build_argo_sample(c, norm, EVM[0], cfg=cfg_ev, rng=probe_rng, lead=0)
    probe = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in probe.items()}
    with torch.no_grad():
        mean_before = base(probe).clone()

    model = CalibratedModel(base, UncertaintyHead(
        n_features=len(CHANNELS) + probe["query"].shape[1] + 1).to(dev)).to(dev)
    mean_after, _ = model(probe)
    bit_identical = bool(torch.equal(mean_before, mean_after))

    # ---- train the head only ------------------------------------------
    opt = torch.optim.AdamW(model.head.parameters(), lr=args.lr)
    rng = np.random.default_rng([sd, 5])
    model.train()
    step = 0
    while step < args.steps:
        m = int(rng.choice(TRM))
        pr, tr = training_rows(c, m, 0, args.n_profiles, rng)
        if pr.size == 0 or tr.size == 0:
            continue
        s = build_argo_sample(c, norm, m, cfg=cfg_tr, rng=rng, lead=0,
                              profile_rows=pr, target_rows=tr)
        if s is None or s["target"].shape[0] == 0:
            continue
        s = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        mean, logvar = model(s)
        loss = gaussian_nll(mean, logvar, s["target"], s["target_mask"])
        opt.zero_grad(set_to_none=True); loss.backward()
        # ---- guarantee 2, checked on a real backward pass -------------
        if step == 0:
            leaked = sum(float(p.grad.abs().sum()) for p in model.base.parameters()
                         if p.grad is not None)
            results["guarantees"][f"seed{sd}_grad_into_base"] = leaked
        opt.step(); step += 1

    sd_hash_after = state_dict_hash(model.base.state_dict())
    results["guarantees"][f"seed{sd}"] = {
        "checkpoint_sha256": file_hash,
        "base_state_hash_before": sd_hash_before,
        "base_state_hash_after": sd_hash_after,
        "base_unchanged": sd_hash_before == sd_hash_after,
        "mean_bit_identical": bit_identical,
        "grad_into_base": results["guarantees"].pop(f"seed{sd}_grad_into_base", None)}
    g = results["guarantees"][f"seed{sd}"]
    print(f"  seed {sd} guarantees: base_unchanged={g['base_unchanged']} "
          f"mean_bit_identical={g['mean_bit_identical']} "
          f"grad_into_base={g['grad_into_base']}", flush=True)
    if not (g["base_unchanged"] and g["mean_bit_identical"]
            and (g["grad_into_base"] in (0.0, None))):
        warn.append(f"seed {sd}: a P6 guarantee FAILED — the uncertainty head "
                    f"has touched the registered mean model")

    # ---- calibration, on held-out floats, per sparsity level ----------
    @torch.no_grad()
    def collect(frac: float):
        model.eval()
        mu, sg, yy, wm = [], [], [], []
        r2 = np.random.default_rng([sd, 7])
        for m in EVM:
            base_rows = E.layout(c, m, args.n_profiles, "natural", sd)
            rows = E.thin(c, base_rows, frac, sd, m) if frac < 1.0 else base_rows
            s = build_argo_sample(c, norm, m, cfg=cfg_ev, rng=r2, lead=0,
                                  profile_rows=rows)
            if s is None or s["target"].shape[0] == 0:
                continue
            s = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
            mean, logvar = model(s)
            msk = s["target_mask"].cpu().numpy()
            mu.append(mean.cpu().numpy()); sg.append((0.5 * logvar).exp().cpu().numpy())
            yy.append(s["target"].cpu().numpy()); wm.append(np.asarray(s["target_wmo"]))
        model.train()
        if not mu:
            return None
        return (np.concatenate(mu), np.concatenate(sg), np.concatenate(yy),
                np.concatenate(wm), np.concatenate([m for m in [msk]]))

    for frac in E.SPARSITY_FRACTIONS:
        got = collect(frac)
        if got is None:
            continue
        mu, sg, yy, wm, _ = got
        if frac == 1.0:
            # Persist (mu, sigma, y, wmo) so Track B can recompute CRPS,
            # coverage and reliability from the SAME predictions with its own
            # metric code. The plan asks Track B to recompute the calibration
            # metrics; without the predictions it would have to retrain the
            # head, which would test a different object.
            np.savez_compressed(
                os.path.join(OUT, f"predictions_seed{sd}.npz"),
                mu=mu.astype("float32"), sigma=sg.astype("float32"),
                y=yy.astype("float32"), wmo=wm.astype(str))
        per_ch = {}
        for j, ch in enumerate(CHANNELS):
            ok = np.isfinite(mu[:, j]) & np.isfinite(sg[:, j]) & np.isfinite(yy[:, j])
            if not ok.any():
                continue
            err = yy[ok, j] - mu[ok, j]
            per_ch[ch] = {
                "crps": float(np.mean(crps_gaussian(mu[ok, j], sg[ok, j], yy[ok, j]))),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "mean_sigma": float(np.mean(sg[ok, j])),
                "spread_skill": spread_skill(sg[ok, j], err),
                "coverage": interval_coverage(mu[ok, j], sg[ok, j], yy[ok, j]),
                "reliability": reliability(mu[ok, j], sg[ok, j], yy[ok, j]),
                "n_wmos": int(np.unique(wm[ok]).size)}
        key = f"{int(frac*100)}pct"
        results["sparsity"].setdefault(key, {})[str(sd)] = per_ch
        if frac == 1.0:
            results["per_seed"][str(sd)] = per_ch
            t = per_ch.get("TEMP", {})
            print(f"    100%: TEMP CRPS {t.get('crps', float('nan')):.4f}  "
                  f"cov90 {t.get('coverage', {}).get('0.90', float('nan')):.3f}  "
                  f"spread-skill slope {t.get('spread_skill', {}).get('slope', float('nan')):.2f}",
                  flush=True)
    del model, base
    torch.cuda.empty_cache()

# ---- calibration vs sparsity summary ---------------------------------
for key, per in sorted(results["sparsity"].items(),
                       key=lambda kv: -int(kv[0].rstrip("pct"))):
    cov = [v["TEMP"]["coverage"]["0.90"] for v in per.values() if "TEMP" in v]
    slp = [v["TEMP"]["spread_skill"]["slope"] for v in per.values() if "TEMP" in v]
    if cov:
        print(f"  {key:>6s}  TEMP 90% coverage {np.mean(cov):.3f}  "
              f"spread-skill slope {np.mean(slp):.2f}", flush=True)

art = P.ResultArtifact(
    package="P6", track="A", region=args.region, results=results,
    seeds=seeds, command=" ".join(sys.argv),
    counts={"eval_split": args.eval_split, "n_profiles": args.n_profiles,
            "months": len(EVM), "row": args.row},
    warnings=warn,
    notes="Post-hoc uncertainty on frozen real-Argo checkpoints. The head "
          "consumes stop-gradient features only; the registered mean is "
          "returned unchanged, asserted per seed."
    ).finalize(ROOT)
path = os.path.join(OUT, f"artifact_P6_seed{seeds[0]}.json")
print(f"\nartifact: {path}\n  sha256 {art.write(path)}")
if warn:
    print("\nWARNINGS:"); [print("  -", w) for w in warn]
print(f"total {time.time()-t0:.0f}s")
