"""Task 8 (Week 4) — first full-scale training run of a shared-latent fusion variant.

Trains ONE (variant, seed) pair under protocol_v1 (configs/protocol_v1.yaml)
following reports/full_training_plan.md exactly:

  * 276 train / 36 val / 12 pinned test months; climatology, anomaly z-stats
    and the RMSE floor from the 276 TRAIN months only.
  * Per step: one random train month, profile-count augmentation K ~ U{0..3000}
    (flexibility axis in-distribution, decided before launch), 8192 random
    unobserved-cell queries, anomaly z-space MSE.
  * Adam 3e-4, cosine to 1e-5 over --steps; validation every --val-every steps
    on the full unobserved val pools (36 months, 1500 profiles each, fixed
    draws); best checkpoint on the val score mean(RMSE_v / floor_v); early
    stop after --patience non-improving evals.
  * Pinned test months scored ONCE from the best-val checkpoint.

Gates that had to be green before this run (both verified):
  * Task-6 invariance suite on MBCA — reports/invariance_test_summary.md
  * Task-7 tiny-overfit gate      — outputs/cache/poc_ocean.json

Queue (all 9 runs): python experiments/run_full_queue.py
Single run:         CUDA_VISIBLE_DEVICES=7 python experiments/18_full_train.py \
                        --variant mbca --seed 1234
"""
import sys, os, json, time, argparse, subprocess, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from ocean_tokenizer import data, config as C
from ocean_tokenizer.anomaly import Climatology, AnomNorm
from ocean_tokenizer.fusion import build_fusion_model
from ocean_tokenizer.fullrun import FullRunData, VARS

# ---------------- protocol_v1 (configs/protocol_v1.yaml) ----------------
TRAIN_YEARS = (1985, 2007)          # 276 months
VAL_YEARS = (2008, 2010)            # 36 months
TEST_TIME_INDICES = [1933, 1935, 1936, 1938, 1942, 1946, 1952, 1953,
                     1956, 1965, 1967, 1976]          # pinned 12 test months
CFG_NAME = "profiles_woa_surf"      # protocol_v1 primary inputs (no SSH/points)

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["perceiver", "resampler", "mbca"],
                required=True)
ap.add_argument("--seed", type=int, default=1234,
                help="init + profile-sampling seed (headline: 1234/1235/1236)")
ap.add_argument("--steps", type=int, default=100_000)
ap.add_argument("--val-every", type=int, default=1000)
ap.add_argument("--patience", type=int, default=10, help="evals without improvement")
ap.add_argument("--min-steps", type=int, default=30_000,
                help="no early stop before this step (the loss escapes its "
                     "initial plateau only after ~5-10k steps)")
ap.add_argument("--queries", type=int, default=8192, help="query points per step")
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--lr-min", type=float, default=1e-5)
ap.add_argument("--d-model", type=int, default=128)
ap.add_argument("--n-latent", type=int, default=128)
ap.add_argument("--n-heads", type=int, default=4)
ap.add_argument("--n-self-blocks", type=int, default=4)
ap.add_argument("--anchor-grid", default=None,
                help="'NLAT,NLON' -> geographically anchored latents (one per "
                     "coarse map cell); overrides --n-latent")
ap.add_argument("--aug-min", type=int, default=0)
ap.add_argument("--aug-max", type=int, default=3000)
ap.add_argument("--n-profiles-eval", type=int, default=C.N_PROFILES,
                help="fixed profile count for val/test months (protocol: 1500)")
ap.add_argument("--tag", default=None, help="default: full_<variant>_s<seed>")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--limit-train", type=int, default=None,
                help="use only the first N train months (diagnostics; NOT protocol)")
ap.add_argument("--limit-val", type=int, default=None)
ap.add_argument("--limit-test", type=int, default=None)
ap.add_argument("--probe-observed", action="store_true",
                help="DIAGNOSTIC ONLY: sample training queries AT observed "
                     "profile columns (answer present in the input tokens); "
                     "a wired-correctly model must fit this fast")
ap.add_argument("--obs-query-frac", type=float, default=0.0,
                help="fraction of TRAINING queries drawn at observed profile "
                     "columns (MAE-style reconstruction bootstrap for the "
                     "token-reading circuit).  Evaluation is always "
                     "unobserved-only; protocol_v1 prohibits scoring, not "
                     "training, on observed columns.")
ap.add_argument("--warmup", type=int, default=0,
                help="linear LR warmup steps before the cosine decay")
ap.add_argument("--input-noise", type=float, default=0.0,
                help="training-only additive N(0, sigma) z-space noise on "
                     "profile/surf/woa input values (anti-memorization "
                     "augmentation; evaluation inputs stay exact)")
ap.add_argument("--grid-drop", type=float, default=0.0,
                help="training-only MAE-style masking: drop each dense-grid "
                     "token (10x12 patch; per level for the WOA volume) with "
                     "this probability, so no step sees the month's full "
                     "dense-field fingerprint (anti-memorization).  "
                     "Evaluation inputs stay complete.")
ap.add_argument("--weight-decay", type=float, default=0.0,
                help=">0 switches the optimizer to AdamW")
ap.add_argument("--val-queries", type=int, default=0,
                help=">0: fixed random subsample of each val month's query "
                     "pool (cheap model selection; test always uses the "
                     "full pool)")
args = ap.parse_args()
tag = args.tag or f"full_{args.variant}_s{args.seed}"
if args.smoke:
    args.steps, args.val_every, args.patience = 300, 100, 3
    args.limit_train, args.limit_val, args.limit_test = 8, 3, 2

t0 = time.time()
dev = "cuda" if torch.cuda.is_available() else "cpu"
dev_name = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
# TF32 matmuls (A100): standard training practice, recorded in the run JSON.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print(f"full-train tag={tag} variant={args.variant} seed={args.seed} "
      f"steps={args.steps} device={dev_name} smoke={args.smoke}", flush=True)

# ======================================================================
# Data: fields, climatology/normaliser (train-only), z-space GPU tensors
# ======================================================================
grid = data.CommonGrid()
print(grid, flush=True)
tr_idx = data.select_month_indices(C.GT_SOURCE, TRAIN_YEARS)
va_idx = data.select_month_indices(C.GT_SOURCE, VAL_YEARS)
te_idx = np.asarray(TEST_TIME_INDICES)
if args.limit_train:
    tr_idx = tr_idx[:args.limit_train]
if args.limit_val:
    va_idx = va_idx[:args.limit_val]
if args.limit_test:
    te_idx = te_idx[:args.limit_test]
print(f"train={tr_idx.size} val={va_idx.size} test={te_idx.size}", flush=True)

print("loading fields ...", flush=True); ts = time.time()
ftrain = data.load_gt_fields(tr_idx, grid)
fval = data.load_gt_fields(va_idx, grid)
ftest = data.load_gt_fields(te_idx, grid)
woa = data.woa_prior(grid)
print(f"  loaded in {time.time()-ts:.1f}s", flush=True)

surf_train = {v: ftrain[v] for v in C.VARS_SURF if v in ftrain}
clim = Climatology(ftrain, surf_train)
norm = AnomNorm(clim, ftrain, surf_train)

rd = FullRunData(grid, norm, dev)
D, H, W, HW = rd.D, rd.H, rd.W, rd.HW
oi, oj, n_ocean = rd.oi, rd.oj, rd.n_ocean

print("building z-space tensors ...", flush=True); ts = time.time()
ZA_tr = torch.from_numpy(rd.z_volume(ftrain)).to(dev)           # (T,2,D,H,W)
ZAf_tr = ZA_tr.view(len(tr_idx), 2, D * HW)
surfZ_tr = torch.from_numpy(rd.z_surf(ftrain)).to(dev)          # (T,2,H,W)
tr_months = ftrain["months"].copy()
rd.load_woa(woa)

# per-train-month valid query pools (finite anomaly for BOTH vars)
valid_tr = []
for t in range(len(tr_idx)):
    fin = torch.isfinite(ZA_tr[t]).all(0).view(-1)              # (D*HW,)
    valid_tr.append(fin.nonzero(as_tuple=True)[0].cpu().numpy().astype("int64"))
del ftrain
print(f"  tensors ready in {time.time()-ts:.1f}s "
      f"(train ZA {ZA_tr.nbytes/1e9:.2f} GB on GPU)", flush=True)

print("building evaluation packs (fixed profile draws) ...", flush=True)
val_rng = np.random.default_rng([args.seed, 1])
test_rng = np.random.default_rng([args.seed, 2])
val_packs, val_n, val_se0 = rd.make_packs(fval, val_rng, args.n_profiles_eval,
                                          subsample=args.val_queries)
test_packs, test_n, test_se0 = rd.make_packs(ftest, test_rng,
                                             args.n_profiles_eval)
del fval, ftest
VAL_FLOOR = rd.physical_rmse(val_se0, val_n)["full"]
TEST_FLOOR = rd.physical_rmse(test_se0, test_n)["full"]
print(f"val clim floor:  TEMP={VAL_FLOOR['TEMP']:.4f} SALT={VAL_FLOOR['SALT']:.4f}",
      flush=True)
print(f"test clim floor: TEMP={TEST_FLOOR['TEMP']:.4f} SALT={TEST_FLOOR['SALT']:.4f}",
      flush=True)

# ======================================================================
# Model / optimizer  (identical trunk init across variants at equal seed)
# ======================================================================
anchor = (tuple(int(x) for x in args.anchor_grid.split(","))
          if args.anchor_grid else None)
model = build_fusion_model(args.variant, grid, d_model=args.d_model,
                           n_latent=args.n_latent, n_heads=args.n_heads,
                           n_self_blocks=args.n_self_blocks,
                           seed=args.seed, anchor_grid=anchor).to(dev)
n_params = sum(p.numel() for p in model.parameters())
opt = (torch.optim.AdamW(model.parameters(), lr=args.lr,
                         weight_decay=args.weight_decay)
       if args.weight_decay > 0
       else torch.optim.Adam(model.parameters(), lr=args.lr))


def _lr_lambda(step):
    if args.warmup and step < args.warmup:
        return (step + 1) / args.warmup
    p = (step - args.warmup) / max(args.steps - args.warmup, 1)
    p = min(max(p, 0.0), 1.0)
    lo = args.lr_min / args.lr
    return lo + 0.5 * (1 - lo) * (1 + math.cos(math.pi * p))


sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
print(f"model {args.variant}: params={n_params:,} d={args.d_model} "
      f"L={model.n_latent}{' anchored' if anchor else ''} "
      f"heads={args.n_heads} blocks={args.n_self_blocks}", flush=True)

os.makedirs(C.CKPT, exist_ok=True)
ckpt_path = os.path.join(C.CKPT, f"{tag}.pt")
json_path = os.path.join(C.CACHE, f"{tag}.json")


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


META = {
    "tag": tag, "task": "week4_full_train", "variant": args.variant,
    "config": CFG_NAME, "protocol": "protocol_v1", "git_commit": git_commit(),
    "smoke": args.smoke, "seed": args.seed, "device": dev_name, "tf32": True,
    "steps": args.steps, "val_every": args.val_every, "patience": args.patience,
    "queries_per_step": args.queries, "lr": args.lr, "lr_min": args.lr_min,
    "schedule": "cosine", "optimizer": "Adam", "n_params": n_params,
    "d_model": args.d_model, "n_latent": args.n_latent,
    "n_heads": args.n_heads, "n_self_blocks": args.n_self_blocks,
    "profile_count_augmentation": [args.aug_min, args.aug_max],
    "n_profiles_eval": args.n_profiles_eval,
    "obs_query_frac": args.obs_query_frac, "warmup": args.warmup,
    "input_noise": args.input_noise, "weight_decay": args.weight_decay,
    "grid_drop": args.grid_drop,
    "val_queries_subsample": args.val_queries,
    "coord_features": "fourier_v2",
    "train_months": int(tr_idx.size), "val_months": int(va_idx.size),
    "test_months": te_idx.tolist(),
    "val_floor": VAL_FLOOR, "test_floor": TEST_FLOOR, "ckpt": ckpt_path,
}

# ======================================================================
# Training loop
# ======================================================================
rng = np.random.default_rng([args.seed, 0])
curves = {"step": [], "train_loss": [], "lr": [],
          "val_TEMP": [], "val_SALT": [], "val_score": []}
best = {"score": float("inf"), "step": -1}
evals_since_best = 0
nan_streak = 0
loss_win = []
Ttr = len(tr_idx)
t_train = time.time()

for step in range(1, args.steps + 1):
    t = int(rng.integers(Ttr))
    mo = int(tr_months[t])
    K = int(rng.integers(args.aug_min, args.aug_max + 1))
    if K:
        pick = rng.choice(n_ocean, size=min(K, n_ocean), replace=False)
        ii_t = torch.from_numpy(oi[pick]).to(dev)
        jj_t = torch.from_numpy(oj[pick]).to(dev)
        col = torch.zeros(HW, dtype=torch.bool, device=dev)
        col[ii_t * W + jj_t] = True
    else:
        ii_t = torch.zeros(0, dtype=torch.long, device=dev)
        jj_t = torch.zeros(0, dtype=torch.long, device=dev)
        col = torch.zeros(HW, dtype=torch.bool, device=dev)

    # unobserved-cell queries: oversample, drop observed columns, trim
    pool = valid_tr[t]
    if args.probe_observed:
        over = min(pool.size, int(args.queries * n_ocean / max(K, 1) * 1.5))
    else:
        over = int(args.queries * (1.0 + 1.3 * K / max(n_ocean, 1)) + 64)
    cand = pool[rng.choice(pool.size, size=min(over, pool.size), replace=False)]
    idx_t = torch.from_numpy(cand).to(dev)
    keep = col[idx_t % HW] if args.probe_observed else ~col[idx_t % HW]
    idx_t = idx_t[keep][:args.queries]
    if args.obs_query_frac > 0 and K:
        # MAE-style bootstrap: replace a fraction of queries with cells at
        # observed profile columns (training only; never scored there)
        n_obs_q = int(args.queries * args.obs_query_frac)
        ocand = pool[rng.choice(pool.size,
                                size=min(pool.size, n_obs_q * max(n_ocean // K, 1) * 2),
                                replace=False)]
        oidx = torch.from_numpy(ocand).to(dev)
        oidx = oidx[col[oidx % HW]][:n_obs_q]
        if oidx.numel():
            idx_t = torch.cat([idx_t[:args.queries - oidx.numel()], oidx])
    if idx_t.numel() == 0:
        continue
    y = ZAf_tr[t][:, idx_t].T                                   # (q,2)
    q, _ = rd.q_from_flat(idx_t, mo)

    obs = rd.obs_dict(ZA_tr, surfZ_tr, t, mo, ii_t, jj_t)
    if args.grid_drop > 0:
        ph, pw = 10, 12                      # = the grid encoders' patch size
        ks = torch.rand(H // ph, W // pw, device=dev) >= args.grid_drop
        ms = ks.repeat_interleave(ph, 0).repeat_interleave(pw, 1)
        obs["surf"]["field"] = obs["surf"]["field"].masked_fill(
            ~ms[None, None], float("nan"))
        kw_ = torch.rand(D, H // ph, W // pw, device=dev) >= args.grid_drop
        mw = kw_.repeat_interleave(ph, 1).repeat_interleave(pw, 2)
        obs["woa"]["field"] = obs["woa"]["field"].masked_fill(
            ~mw[None, None], float("nan"))
    if args.input_noise > 0:
        # out-of-place adds: the cached month tensors are never mutated
        s = args.input_noise
        obs["profiles"]["prof"] = (obs["profiles"]["prof"]
                                   + s * torch.randn_like(obs["profiles"]["prof"]))
        obs["surf"]["field"] = (obs["surf"]["field"]
                                + s * torch.randn_like(obs["surf"]["field"]))
        obs["woa"]["field"] = (obs["woa"]["field"]
                               + s * torch.randn_like(obs["woa"]["field"]))
    out = model(obs, q)
    loss = ((out - y[None]) ** 2).mean()
    if not torch.isfinite(loss):
        nan_streak += 1
        opt.zero_grad()
        if nan_streak > 50:
            raise RuntimeError(f"{tag}: 50 consecutive non-finite losses")
        continue
    nan_streak = 0
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    loss_win.append(float(loss))
    if len(loss_win) > 200:
        loss_win.pop(0)

    if step % args.val_every == 0:
        se = rd.eval_packs(model, val_packs)
        vr = rd.physical_rmse(se, val_n)["full"]
        score = float(np.mean([vr[v] / VAL_FLOOR[v] for v in VARS]))
        curves["step"].append(step)
        curves["train_loss"].append(float(np.mean(loss_win)))
        curves["lr"].append(opt.param_groups[0]["lr"])
        curves["val_TEMP"].append(vr["TEMP"])
        curves["val_SALT"].append(vr["SALT"])
        curves["val_score"].append(score)
        if score < best["score"]:
            best = {"score": score, "step": step,
                    "val_TEMP": vr["TEMP"], "val_SALT": vr["SALT"]}
            evals_since_best = 0
            torch.save({"state_dict": model.state_dict(), "step": step,
                        "tag": tag, "variant": args.variant, "args": vars(args)},
                       ckpt_path)
        else:
            evals_since_best += 1
        sps = step / (time.time() - t_train)
        eta_h = (args.steps - step) / max(sps, 1e-9) / 3600
        print(f"  step {step:6d}/{args.steps}  loss={np.mean(loss_win):.4f}  "
              f"val TEMP={vr['TEMP']:.4f} SALT={vr['SALT']:.4f}  "
              f"score={score:.4f}  {sps:.1f} it/s  eta {eta_h:.1f}h"
              f"{'  *best*' if best['step'] == step else ''}", flush=True)
        with open(json_path, "w") as f:
            json.dump({**META, "status": "running", "best": best,
                       "gpu_hours": round((time.time() - t0) / 3600, 3),
                       "curves": curves}, f, indent=2)
        if evals_since_best >= args.patience and step >= args.min_steps:
            print(f"early stop at step {step} "
                  f"(no improvement in {args.patience} evals)", flush=True)
            break

train_secs = time.time() - t0

# ======================================================================
# Test ONCE from the best-validation checkpoint
# ======================================================================
print(f"\nbest step {best['step']} (val score {best['score']:.4f}); "
      f"scoring pinned test months ...", flush=True)
model.load_state_dict(torch.load(ckpt_path, map_location=dev)["state_dict"])
se = rd.eval_packs(model, test_packs)
test = rd.physical_rmse(se, test_n)
test_out = {"TEMP": test["full"]["TEMP"], "SALT": test["full"]["SALT"],
            "by_band": test["by_band"], "by_depth": test["by_depth"],
            "skill_vs_floor": {v: 1.0 - test["full"][v] / TEST_FLOOR[v]
                               for v in VARS}}
print(f"TEST unobs anomaly RMSE: TEMP={test_out['TEMP']:.4f} "
      f"SALT={test_out['SALT']:.4f}  "
      f"(floor {TEST_FLOOR['TEMP']:.4f}/{TEST_FLOOR['SALT']:.4f})", flush=True)

with open(json_path, "w") as f:
    json.dump({**META, "status": "done", "best": best, "test": test_out,
               "optimizer_steps": curves["step"][-1] if curves["step"] else 0,
               "gpu_hours": round(train_secs / 3600, 3),
               "curves": curves}, f, indent=2)
print(f"\nDONE {tag} in {train_secs/3600:.2f} h -> {json_path}", flush=True)
