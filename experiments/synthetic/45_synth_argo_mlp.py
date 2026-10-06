"""Pointwise MLP baseline on the synthetic Argo cohort.

The gridded line's pointwise MLP (`ocean_tokenizer.baselines.MLP` with its
settings in `config.py`), in its profiles-only form, scored on the cells the
trained models and the fixed baselines of `44_synth_argo_oi.py` are scored on:
all 6,080 input profiles of the month, the 1,520 query profiles, the
at-position target, the train-year normalisation.

Kept from the original: the network (256-256-256, SiLU), Adam, the learning
rate, 30 epochs, the batch size, 120,000 training points a month, last-epoch
weights (no selection), and the nine inputs of `baselines._point_features`
with the `profiles` input alone -- position, calendar month, the nearest input
profile's TEMP and SALT at the cell's level, and the distance to it.

What has to differ: the original trained on the full CESM2 field at random
grid points. A method on this cohort sees Argo profiles only, so the targets
are profiles, drawn as the trained model draws them (`62_sanity_train.py`):
in a training month 30 % of the input profiles are targets and the other 70 %
are the inputs the features are computed from. Query profiles are never
trained on.

The input-parity assertion and the zero-prediction identity check of
`ocean_tokenizer.synth_argo_eval` run before training.

Writes outputs/audit/synthetic/pointwise_mlp/summary_seed<seed>.json (and the
weights, model_seed<seed>.pt).

  .venv/bin/python experiments/synthetic/45_synth_argo_mlp.py --seed 1234
  .venv/bin/python experiments/synthetic/45_synth_argo_mlp.py --smoke   # prints only
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch

from ocean_tokenizer import config as C
from ocean_tokenizer import synth_argo_eval
from ocean_tokenizer.audit_tools import cohort_path
from ocean_tokenizer.baselines import MLP
from ocean_tokenizer.point_baselines import CH, MLP_FEATURES, Scores, mlp_point_features
from ocean_tokenizer.synth_argo_eval import (EVAL_CELLS, N_INPUT, N_QUERY, REFERENCE,
                                             SPLITS, identity_check)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TARGET_FRAC = 0.3   # share of a training month's input profiles drawn as targets (62)
DRAWS = 4           # target draws per training month, before the points-per-month cap

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=1234,
                help="target draws, point subsample, initial weights and batch order")
ap.add_argument("--out", default=None,
                help="default: outputs/audit/synthetic/pointwise_mlp")
ap.add_argument("--smoke", action="store_true",
                help="2 training months, 2 epochs, 1 validation month, no identity "
                     "check; prints and writes nothing")
args = ap.parse_args()
OUT = args.out or os.path.join(ROOT, "outputs", "audit", "synthetic", "pointwise_mlp")
dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()

# ------------------------------------------------------------------ data
c, norm, OBS = synth_argo_eval.load(ROOT)
LEV = c.levels


def features(src, tgt, m):
    """(R * L, 9) inputs of the query profiles `tgt`, given input profiles `src`."""
    return mlp_point_features(c.lat[src], c.lon[src], {ch: OBS[ch][src] for ch in CH},
                              c.lat[tgt], c.lon[tgt], LEV, month=int(m) % 12 + 1)


val = synth_argo_eval.eval_set(c, OBS, "validation", EVAL_CELLS,
                               n_months=1 if args.smoke else None)
test = [] if args.smoke else synth_argo_eval.eval_set(c, OBS, "development")
print(f"pointwise_mlp seed={args.seed} device={dev}: {len(val)} validation months, "
      f"{len(test)} test months, {N_INPUT} inputs and {N_QUERY} queries a month "
      f"({time.time() - t0:.0f}s)", flush=True)


def score(evs, model=None):
    """Pooled scores on the months' cells; ``model=None`` scores a zero prediction.
    Every month's features come from ALL of its input profiles."""
    s = Scores(LEV, norm.std)
    for ev in evs:
        if model is None:
            pred = np.zeros((ev["lev"].size, len(CH)))
        else:
            X = torch.from_numpy(features(ev["src"], ev["tgt"], ev["month"])).to(dev)
            with torch.no_grad():
                y = model(X).cpu().numpy().astype(np.float64)   # (R * L, 2) profile-major
            pred = y[ev["prof"] * LEV.size + ev["lev"]]
        for k, ch in enumerate(CH):
            s.add(ch, pred[:, k], ev["target"][ch], ev["lev"])
    return s.result()


ident = {}
if not args.smoke:
    ident = {split: identity_check(ROOT, split, score(evs))
             for split, evs in (("validation", val), ("development", test))}
    print(f"identity check passed: zero prediction matches {REFERENCE} on both splits",
          flush=True)

# ------------------------------------------------------------------ training set
rng = np.random.default_rng(args.seed)
train_months = [int(m) for m in c.months_in("train")]
if args.smoke:
    train_months = train_months[:2]
Xs, Ys = [], []
for m in train_months:
    pool = c.month(m, float_split="cohort_float")
    if pool.size != N_INPUT:
        raise SystemExit(f"input parity violated in training month {m}: "
                         f"{pool.size} inputs (expected {N_INPUT})")
    Xm, Ym = [], []
    for _ in range(DRAWS):
        tgt = np.sort(rng.choice(pool, int(round(TARGET_FRAC * pool.size)), replace=False))
        src = np.setdiff1d(pool, tgt)
        y = np.stack([OBS[ch][tgt].ravel() for ch in CH], axis=1)   # profile-major
        ok = np.isfinite(y).all(axis=1)
        Xm.append(features(src, tgt, m)[ok]); Ym.append(y[ok])
    Xm, Ym = np.concatenate(Xm), np.concatenate(Ym)
    keep = rng.choice(len(Xm), min(C.MLP_POINTS_PER_MONTH, len(Xm)), replace=False)
    Xs.append(Xm[keep]); Ys.append(Ym[keep].astype("float32"))
Xtr, Ytr = np.concatenate(Xs), np.concatenate(Ys)
print(f"training set: {len(train_months)} months, {len(Xtr):,} cells, "
      f"{Xtr.shape[1]} features ({time.time() - t0:.0f}s)", flush=True)

# ------------------------------------------------------------------ train (as baselines.train_predict_mlp)
torch.manual_seed(args.seed)
model = MLP(Xtr.shape[1], C.MLP_HIDDEN).to(dev)
n_par = sum(p.numel() for p in model.parameters())
opt = torch.optim.Adam(model.parameters(), lr=C.MLP_LR)
lossf = torch.nn.MSELoss()
Xt, Yt = torch.from_numpy(Xtr).to(dev), torch.from_numpy(Ytr).to(dev)
nb = int(np.ceil(len(Xt) / C.MLP_BATCH))
epochs = 2 if args.smoke else C.MLP_EPOCHS
for ep in range(epochs):
    perm = torch.randperm(len(Xt), device=dev)
    tot = 0.0
    for b in range(nb):
        sl = perm[b * C.MLP_BATCH:(b + 1) * C.MLP_BATCH]
        opt.zero_grad()
        loss = lossf(model(Xt[sl]), Yt[sl])
        loss.backward(); opt.step()
        tot += float(loss) * len(sl)
    train_mse = tot / len(Xt)
    if (ep + 1) % 5 == 0 or ep == 0 or ep + 1 == epochs:
        print(f"  epoch {ep + 1:2d}/{epochs}  train mse {train_mse:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
model.eval()

# ------------------------------------------------------------------ score
scores = {"validation": score(val, model)}
if test:
    scores["development"] = score(test, model)
for split, sc in scores.items():
    print(f"{split}: TEMP {sc['TEMP']['rmse_physical']:.4f} degC (J {sc['TEMP']['J']:.3f})  "
          f"SALT {sc['SALT']['rmse_physical']:.4f} PSU (J {sc['SALT']['J']:.3f})", flush=True)
if args.smoke:
    print(f"smoke ok ({time.time() - t0:.0f}s), nothing written")
    raise SystemExit(0)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


cpath = cohort_path(ROOT, "synthetic")
summary = {
    "tag": "pointwise_mlp", "region": "synthetic", "seed": args.seed,
    "target": "anomaly_exact", "splits": {k: list(v) for k, v in SPLITS.items()},
    "inputs_per_month": N_INPUT, "queries_per_month": N_QUERY, "eval_cells": EVAL_CELLS,
    "per_month": {name: {str(ev["month"]): {"inputs": int(ev["src"].size),
                                            "queries": int(ev["tgt"].size)}
                         for ev in evs}
                  for name, evs in (("validation", val), ("development", test))},
    "features": list(MLP_FEATURES), "hidden": list(C.MLP_HIDDEN), "params": n_par,
    "epochs": epochs, "batch": C.MLP_BATCH, "lr": C.MLP_LR,
    "points_per_month": C.MLP_POINTS_PER_MONTH, "target_fraction": TARGET_FRAC,
    "target_draws": DRAWS, "train_months": len(train_months),
    "train_cells": int(len(Xtr)), "final_train_mse": train_mse,
    "identity_check": {"reference": REFERENCE, **ident},
    "cohort": {"path": os.path.relpath(cpath, ROOT), "sha256": sha256(cpath)},
    "git_commit": git_commit(), "scores": scores, "runtime_s": time.time() - t0}
os.makedirs(OUT, exist_ok=True)
torch.save(model.state_dict(), os.path.join(OUT, f"model_seed{args.seed}.pt"))
with open(os.path.join(OUT, f"summary_seed{args.seed}.json"), "w") as f:
    json.dump(summary, f, indent=1, default=float)
print(f"wrote {OUT}/summary_seed{args.seed}.json ({time.time() - t0:.0f}s)")
