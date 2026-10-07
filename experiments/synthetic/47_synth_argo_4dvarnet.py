"""4DVarNet baseline on the synthetic Argo cohort, Argo-only or with surface fields.

4DVarNet (Fablet et al. 2021) learns a variational interpolation: a prior and a
gradient-based solver are trained together so that a few solver steps turn
gappy gridded observations into a complete field. The solver, the prior and
the gradient model are imported UNMODIFIED from the authors' `4dvarnet-starter`
(https://github.com/CIA-Oceanix/4dvarnet-starter, CeCILL-C), cloned at a pinned
commit into the git-ignored `external/` folder, with the settings of its
`config/xp/base.yaml`. What is ours is how the cohort is put in front of it:

  state          TEMP and SALT z-anomalies on the 20 levels = 40 channels on the
                 global 1-degree grid, one month at a time
  observations   the month's input profiles binned to their cell (mean), NaN
                 elsewhere; padded circularly in longitude
  --surface      adds SST, SSS and SLA (46_synth_surface_fields.py) through an
                 observation term modelled on the starter's
                 contrib/multimodal.MultiModalObsCost
  targets        Argo profiles ONLY: each time a training month is drawn, 30 %
                 of its input profiles are held out and gridded as the target,
                 the rule of 62_sanity_train.py. The starter trains on a dense
                 field and adds a Sobel-gradient loss; neither exists here, so
                 the loss is its 50 x MSE on the cells that have a target plus
                 its prior cost
  selection      after each epoch the validation year is scored as the test
                 year will be; the weights with the lowest macro z are kept
  solver output  the state after the solver's steps, in training and in scoring.
                 Out of training the starter's `GradSolver.forward` also passes
                 that state through the prior's auto-encoder; on this cohort the
                 projection, which the training loss never sees, returns a field
                 no better than climatology (validation J 1.00 against 0.41
                 without it, same weights). The score along the starter's own
                 evaluation path is recorded next to the one used
  scoring        all 6,080 input profiles of the month in, the gridded answer
                 sampled bilinearly at the 1,520 query profiles, pooled on the
                 evaluation sets of ocean_tokenizer.synth_argo_eval after the
                 input-parity assertion and the zero-prediction identity check

Writes outputs/audit/synthetic/fourdvarnet[_surface]/summary_seed<seed>.json and
the selected weights.

  .venv/bin/python experiments/synthetic/47_synth_argo_4dvarnet.py                 # Argo only
  .venv/bin/python experiments/synthetic/47_synth_argo_4dvarnet.py --surface
  .venv/bin/python experiments/synthetic/47_synth_argo_4dvarnet.py --smoke         # prints only
"""
from __future__ import annotations

import argparse, collections, copy, hashlib, json, os, subprocess, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXT = os.path.join(ROOT, "external", "4dvarnet-starter")
STARTER_COMMIT = "20f1b5f34b201342cde6dd21a30419d07541db54"
sys.path.insert(0, os.path.join(ROOT, "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr


def _starter_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=EXT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


if _starter_commit() != STARTER_COMMIT:
    raise SystemExit(
        f"expected 4dvarnet-starter at {STARTER_COMMIT[:7]} in external/4dvarnet-starter:\n"
        "  git clone https://github.com/CIA-Oceanix/4dvarnet-starter.git external/4dvarnet-starter\n"
        f"  git -C external/4dvarnet-starter checkout {STARTER_COMMIT}")
sys.path.insert(0, EXT)
from src.models import BaseObsCost, BilinAEPriorCost, ConvLstmGradModel, GradSolver  # noqa: E402

from ocean_tokenizer import synth_argo_eval  # noqa: E402
from ocean_tokenizer.audit_tools import cohort_path  # noqa: E402
from ocean_tokenizer.gridded_obs import bin_profiles, sample_bilinear  # noqa: E402
from ocean_tokenizer.point_baselines import CH, Scores  # noqa: E402
from ocean_tokenizer.synth_argo_eval import (EVAL_CELLS, N_INPUT, N_QUERY, REFERENCE,  # noqa: E402
                                             SPLITS, identity_check)

# ---- the starter's base configuration (config/xp/base.yaml, utils.cosanneal_lr_adam)
N_STEP, LR_GRAD = 10, 1e3
PRIOR_HIDDEN, PRIOR_DOWNSAMP, GRAD_HIDDEN = 32, 2, 48
EPOCHS, BATCH, LR, CLIP = 150, 4, 1e-3, 0.5
W_MSE, W_PRIOR = 50.0, 1.0          # Lit4dVarNet.step, without its Sobel-gradient term
SURF_HIDDEN = 5                     # config/params/multimodal.yaml
# ---- ours
NY, NX = 180, 360
PAD = 8                             # circular longitude padding, in cells
TARGET_FRAC = 0.3                   # input profiles held out as targets (62)
SURF_VARS = ("SST", "SSS", "SLA")
SURF_STORE = os.path.join("data", "synthetic_argo", "cesm2_surface_1deg.zarr")

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=1234,
                help="target draws, month order and initial weights")
ap.add_argument("--surface", action="store_true",
                help="also give SST, SSS and SLA of the month")
ap.add_argument("--epochs", type=int, default=EPOCHS)
ap.add_argument("--out", default=None,
                help="default: outputs/audit/synthetic/fourdvarnet[_surface]")
ap.add_argument("--smoke", action="store_true",
                help="4 training months, 2 epochs, 1 validation month, no identity "
                     "check; prints and writes nothing")
args = ap.parse_args()
TAG = "fourdvarnet_surface" if args.surface else "fourdvarnet"
OUT = args.out or os.path.join(ROOT, "outputs", "audit", "synthetic", TAG)
dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()

# ------------------------------------------------------------------ data
c, norm, OBS = synth_argo_eval.load(ROOT)
LEV = c.levels
L = LEV.size
C_STATE = len(CH) * L               # TEMP levels, then SALT levels
Item = collections.namedtuple("Item", ["input", "tgt", "surface"])


def grid_profiles(rows):
    """(2L, NY, NX) float32: the profiles `rows` binned to their cells, NaN elsewhere."""
    return np.concatenate([bin_profiles(c.grid_y[rows], c.grid_x[rows], OBS[ch][rows], (NY, NX))
                           for ch in CH]).astype("float32")


SURF = None
if args.surface:
    _o = xr.open_zarr(os.path.join(ROOT, SURF_STORE))
    _t = _o.time.values.astype("datetime64[M]")
    _mi = ((_t - np.datetime64("2000-01", "M")) / np.timedelta64(1, "M")).astype(int)
    _a = np.stack([_o[v].values for v in SURF_VARS], axis=1).astype("float32")   # (T, 3, NY, NX)
    _tr = np.array([SPLITS["train"][0] <= 2000 + m // 12 <= SPLITS["train"][1] for m in _mi])
    # z-scored per field on the train years, as 62 scales the satellite fields
    _mu = np.nanmean(_a[_tr], axis=(0, 2, 3), keepdims=True)
    _sd = np.nanstd(_a[_tr], axis=(0, 2, 3), keepdims=True)
    SURF = {int(m): (_a[i] - _mu[0]) / _sd[0] for i, m in enumerate(_mi)}        # NaN kept


def pad_lon(x):
    """Wrap PAD columns around in longitude (last dim), keeping NaN."""
    return torch.cat([x[..., -PAD:], x, x[..., :PAD]], dim=-1)


def crop_lon(x):
    return x[..., PAD:-PAD]


def batch_of(inputs, months, targets=None):
    """An Item on the device, longitude-padded. inputs/targets: lists of (2L, NY, NX)."""
    t = lambda a: pad_lon(torch.from_numpy(np.stack(a)).to(dev))
    return Item(input=t(inputs), tgt=None if targets is None else t(targets),
                surface=None if SURF is None else t([SURF[int(m)] for m in months]))


val = synth_argo_eval.eval_set(c, OBS, "validation", EVAL_CELLS,
                               n_months=1 if args.smoke else None)
test = [] if args.smoke else synth_argo_eval.eval_set(c, OBS, "development")
print(f"{TAG} seed={args.seed} device={dev}: {len(val)} validation months, {len(test)} "
      f"test months, {N_INPUT} inputs and {N_QUERY} queries a month, state {C_STATE} x "
      f"{NY} x {NX} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ model
class SurfaceObsCost(nn.Module):
    """The profile observation term plus one tying the state to the surface fields.

    After contrib/multimodal.MultiModalObsCost of the starter: the MSE between a
    learned bias-free 3 x 3 convolution of the state and one of the fields, the
    fields zero-filled where missing. Rewritten because the original builds both
    convolutions for one channel count; here the state has 2L channels and the
    fields three.
    """

    def __init__(self, dim_state, dim_surface, dim_hidden):
        super().__init__()
        self.base_cost = BaseObsCost()
        self.conv_state = nn.Conv2d(dim_state, dim_hidden, (3, 3), padding=1, bias=False)
        self.conv_surface = nn.Conv2d(dim_surface, dim_hidden, (3, 3), padding=1, bias=False)

    def forward(self, state, batch):
        return self.base_cost(state, batch) + F.mse_loss(
            self.conv_state(state), self.conv_surface(batch.surface.nan_to_num()))


torch.manual_seed(args.seed)
solver = GradSolver(
    prior_cost=BilinAEPriorCost(dim_in=C_STATE, dim_hidden=PRIOR_HIDDEN,
                                bilin_quad=False, downsamp=PRIOR_DOWNSAMP),
    obs_cost=(SurfaceObsCost(C_STATE, len(SURF_VARS), SURF_HIDDEN) if args.surface
              else BaseObsCost()),
    grad_mod=ConvLstmGradModel(dim_in=C_STATE, dim_hidden=GRAD_HIDDEN),
    n_step=N_STEP, lr_grad=LR_GRAD).to(dev)
n_par = sum(p.numel() for p in solver.parameters())
groups = [{"params": list(solver.grad_mod.parameters()), "lr": LR},
          {"params": list(solver.obs_cost.parameters()), "lr": LR},
          {"params": list(solver.prior_cost.parameters()), "lr": LR / 2}]
epochs = 2 if args.smoke else args.epochs
opt = torch.optim.Adam([g for g in groups if g["params"]])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
print(f"  {n_par:,} parameters; starter {STARTER_COMMIT[:7]}; {epochs} epochs", flush=True)


# ------------------------------------------------------------------ scoring
def solve(model, batch, upstream_eval=False):
    """The model's answer for a batch, out of training.

    ``upstream_eval=True`` is the starter's `GradSolver.forward` in eval mode:
    the solver steps, then the prior's auto-encoder applied to the final state.
    The default stops before that projection and returns the state itself, the
    quantity the training loss is computed on.
    """
    model.eval()
    if upstream_eval:
        return model(batch)
    with torch.set_grad_enabled(True):
        state = model.init_state(batch)
        model.grad_mod.reset_state(batch.input)
        for step in range(model.n_step):
            state = model.solver_step(state, batch, step=step).detach().requires_grad_(True)
    return state


def score(evs, model=None, upstream_eval=False):
    """Pooled scores on the months' cells; ``model=None`` scores a zero prediction.
    The observations of a scored month are ALL of its input profiles."""
    s = Scores(LEV, norm.std)
    for ev in evs:
        if model is None:
            pred = {ch: np.zeros(ev["lev"].size) for ch in CH}
        else:
            out = crop_lon(solve(model, batch_of([grid_profiles(ev["src"])], [ev["month"]]),
                                 upstream_eval))[0]
            q = sample_bilinear(out.detach().cpu().numpy().astype(np.float64),
                                c.lat[ev["tgt"]], c.lon[ev["tgt"]])          # (R, 2L)
            pred = {ch: q[:, k * L:(k + 1) * L][ev["prof"], ev["lev"]] for k, ch in enumerate(CH)}
        for ch in CH:
            s.add(ch, pred[ch], ev["target"][ch], ev["lev"])
    return s.result()


ident = {}
if not args.smoke:
    ident = {split: identity_check(ROOT, split, score(evs))
             for split, evs in (("validation", val), ("development", test))}
    print(f"identity check passed: zero prediction matches {REFERENCE} on both splits",
          flush=True)

# ------------------------------------------------------------------ train
rng = np.random.default_rng(args.seed)
train_months = [int(m) for m in c.months_in("train")]
if args.smoke:
    train_months = train_months[:4]
for m in train_months:
    if c.month(m, float_split="cohort_float").size != N_INPUT:
        raise SystemExit(f"input parity violated in training month {m}")


def train_pair(m):
    """Observations and target of one draw: 70 % / 30 % of the month's input profiles."""
    pool = c.month(m, float_split="cohort_float")
    tgt = np.sort(rng.choice(pool, int(round(TARGET_FRAC * pool.size)), replace=False))
    return grid_profiles(np.setdiff1d(pool, tgt)), grid_profiles(tgt)


best = {"macro_z": float("inf"), "epoch": -1, "state": None}
history = []
for ep in range(epochs):
    solver.train()
    order = [train_months[i] for i in rng.permutation(len(train_months))]
    tot, nb = 0.0, 0
    for b in range(0, len(order), BATCH):
        months = order[b:b + BATCH]
        pairs = [train_pair(m) for m in months]
        batch = batch_of([p[0] for p in pairs], months, [p[1] for p in pairs])
        out = solver(batch)
        m = batch.tgt.isfinite()
        mse = F.mse_loss(out[m], batch.tgt[m])
        loss = W_MSE * mse + W_PRIOR * solver.prior_cost(out)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(solver.parameters(), CLIP)
        opt.step()
        tot += float(mse); nb += 1
    sched.step()
    sc = score(val, solver)
    history.append({"epoch": ep + 1, "train_mse": tot / nb, "val_macro_z": sc["macro_z"],
                    "val_TEMP_z": sc["TEMP"]["rmse_z"], "val_SALT_z": sc["SALT"]["rmse_z"]})
    star = ""
    if sc["macro_z"] < best["macro_z"]:
        best = {"macro_z": sc["macro_z"], "epoch": ep + 1,
                "state": copy.deepcopy(solver.state_dict())}
        star = " *"
    if (ep + 1) % 10 == 0 or ep < 3 or ep + 1 == epochs:
        print(f"  epoch {ep + 1:3d}/{epochs}  train mse {tot / nb:.4f}  validation TEMP "
              f"{sc['TEMP']['rmse_z']:.3f}z SALT {sc['SALT']['rmse_z']:.3f}z{star}  "
              f"({time.time() - t0:.0f}s)", flush=True)

solver.load_state_dict(best["state"])
scores = {"validation": score(val, solver)}
scores_upstream = {"validation": score(val, solver, upstream_eval=True)}
if test:
    scores["development"] = score(test, solver)
    scores_upstream["development"] = score(test, solver, upstream_eval=True)
for split, sc in scores.items():
    print(f"{split}: TEMP {sc['TEMP']['rmse_physical']:.4f} degC (J {sc['TEMP']['J']:.3f})  "
          f"SALT {sc['SALT']['rmse_physical']:.4f} PSU (J {sc['SALT']['J']:.3f})  "
          f"[best epoch {best['epoch']}]", flush=True)
for split, sc in scores_upstream.items():
    print(f"  with the starter's final prior projection, {split}: TEMP J "
          f"{sc['TEMP']['J']:.3f}  SALT J {sc['SALT']['J']:.3f}", flush=True)
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
    "tag": TAG, "region": "synthetic", "seed": args.seed, "target": "anomaly_exact",
    "splits": {k: list(v) for k, v in SPLITS.items()},
    "inputs_per_month": N_INPUT, "queries_per_month": N_QUERY, "eval_cells": EVAL_CELLS,
    "per_month": {name: {str(ev["month"]): {"inputs": int(ev["src"].size),
                                            "queries": int(ev["tgt"].size)}
                         for ev in evs}
                  for name, evs in (("validation", val), ("development", test))},
    "surface": list(SURF_VARS) if args.surface else None,
    "surface_store": SURF_STORE if args.surface else None,
    "upstream": {"repo": "https://github.com/CIA-Oceanix/4dvarnet-starter",
                 "commit": STARTER_COMMIT, "license": "CeCILL-C",
                 "classes": ["GradSolver", "BilinAEPriorCost", "ConvLstmGradModel",
                             "BaseObsCost"]},
    "state_channels": C_STATE, "grid": [NY, NX], "lon_pad": PAD,
    "n_step": N_STEP, "lr_grad": LR_GRAD, "prior_hidden": PRIOR_HIDDEN,
    "prior_downsamp": PRIOR_DOWNSAMP, "grad_hidden": GRAD_HIDDEN,
    "surface_hidden": SURF_HIDDEN if args.surface else None,
    "epochs": epochs, "batch": BATCH, "lr": LR, "clip": CLIP,
    "loss": {"mse": W_MSE, "prior": W_PRIOR, "sobel_gradient": 0.0},
    "training_targets": "argo_profiles", "target_fraction": TARGET_FRAC,
    "train_months": len(train_months), "params": n_par,
    "best_epoch": best["epoch"], "history": history,
    "identity_check": {"reference": REFERENCE, **ident},
    "cohort": {"path": os.path.relpath(cpath, ROOT), "sha256": sha256(cpath)},
    "solver_output": "state after the solver steps, without the starter's eval-mode "
                     "prior projection",
    "git_commit": git_commit(), "scores": scores,
    "scores_upstream_eval_path": scores_upstream, "runtime_s": time.time() - t0}
os.makedirs(OUT, exist_ok=True)
torch.save(best["state"], os.path.join(OUT, f"model_seed{args.seed}.pt"))
with open(os.path.join(OUT, f"summary_seed{args.seed}.json"), "w") as f:
    json.dump(summary, f, indent=1, default=float)
print(f"wrote {OUT}/summary_seed{args.seed}.json ({time.time() - t0:.0f}s)")
