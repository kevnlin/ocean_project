"""Task 1 — joint-depth U-Net convergence audit under protocol_v1.

Trains ONE U-Net variant with the protocol_v1 split (276 train / 36 val /
12 pinned test months), logging per-epoch training loss and validation anomaly
RMSE, selecting the best checkpoint on validation, and scoring the test months
once from that checkpoint.  The goal is NOT to beat the depthwise model — it is
to certify the joint-depth baseline as fairly trained (plateaued validation
curve, matched optimizer-step budget, documented LR schedule).

Runs (one per GPU, sequential per GPU is fine):
    CUDA_VISIBLE_DEVICES=6 python experiments/13_joint_audit.py --model depthwise --epochs 40  --schedule const  --tag depthwise_e40
    CUDA_VISIBLE_DEVICES=7 python experiments/13_joint_audit.py --model joint     --epochs 200 --schedule const  --tag joint_e200_const
    CUDA_VISIBLE_DEVICES=6 python experiments/13_joint_audit.py --model joint     --epochs 200 --schedule cosine --tag joint_e200_cos
    CUDA_VISIBLE_DEVICES=7 python experiments/13_joint_audit.py --model joint     --epochs 400 --schedule cosine --tag joint_e400_cos

Then:  python experiments/14_joint_audit_report.py
"""
import sys, os, json, time, argparse, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from ocean_tokenizer import data, baselines as B, metrics, config as C
from ocean_tokenizer.anomaly import Climatology, AnomNorm
from ocean_tokenizer.unet import UNet2D

# ---------------- protocol_v1 split (configs/protocol_v1.yaml) ----------------
TRAIN_YEARS = (1985, 2007)          # 276 months
VAL_YEARS = (2008, 2010)            # 36 months
TEST_TIME_INDICES = [1933, 1935, 1936, 1938, 1942, 1946, 1952, 1953,
                     1956, 1965, 1967, 1976]          # pinned 12 test months
BANDS = [("0-100m", 0.0, 100.0), ("100-300m", 100.0, 300.0),
         ("300-max", 300.0, 1e9)]
CFG = ("profiles", "woa", "surf")
CFG_NAME = "profiles_woa_surf"

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["joint", "depthwise"], required=True)
ap.add_argument("--epochs", type=int, required=True)
ap.add_argument("--schedule", choices=["const", "cosine"], default="const")
ap.add_argument("--lr", type=float, default=None, help="default: config value per model")
ap.add_argument("--tag", required=True, help="run name -> outputs/cache/audit_<tag>.json")
ap.add_argument("--seed", type=int, default=1234, help="profile sampling + torch seed")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

if args.smoke:
    C.N_PROFILES = 300

t0 = time.time()
device = C.DEVICE if torch.cuda.is_available() else "cpu"
dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
print(f"audit tag={args.tag} model={args.model} epochs={args.epochs} "
      f"schedule={args.schedule} device={dev_name} smoke={args.smoke}", flush=True)

grid = data.CommonGrid()
print(grid, flush=True)

# ---- months ----
tr_idx = data.select_month_indices(C.GT_SOURCE, TRAIN_YEARS)
va_idx = data.select_month_indices(C.GT_SOURCE, VAL_YEARS)
te_idx = np.asarray(TEST_TIME_INDICES)
if args.smoke:
    tr_idx, va_idx, te_idx = tr_idx[:6], va_idx[:2], te_idx[:2]
print(f"train={tr_idx.size} val={va_idx.size} test={te_idx.size}", flush=True)

print("loading fields ...", flush=True); ts = time.time()
ftrain = data.load_gt_fields(tr_idx, grid)
fval = data.load_gt_fields(va_idx, grid)
ftest = data.load_gt_fields(te_idx, grid)
woa = data.woa_prior(grid)
print(f"  loaded in {time.time()-ts:.1f}s", flush=True)

# ---- climatology / normaliser from the 276 TRAIN months ONLY ----
surf_train = {v: ftrain[v] for v in C.VARS_SURF if v in ftrain}
clim = Climatology(ftrain, surf_train)
norm = AnomNorm(clim, ftrain, surf_train)

rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)

print("assembling samples ...", flush=True); ts = time.time()
def make_samples(fields):
    return [B.prepare_month(fields, fields, woa, grid, t, rng, C.N_PROFILES)
            for t in range(len(fields["months"]))]
train_samples = make_samples(ftrain)
val_samples = make_samples(fval)
test_samples = make_samples(ftest)
print(f"  assembled in {time.time()-ts:.1f}s", flush=True)

D, H, W = grid.ndepth, grid.nlat, grid.nlon
VARS = B.VARS

# =========================================================================
# Tensor assembly (joint: month rows; depthwise: depth-slice rows)
# =========================================================================
def joint_xy(samples):
    X = np.stack([B._unet_channels_joint(s, grid, norm, CFG) for s in samples], 0)
    Y = np.stack([np.stack([np.nan_to_num(norm.z3d(v, s["gt"][v], s["month"]), nan=0.0)
                            for v in VARS], 0).reshape(len(VARS) * D, H, W)
                  for s in samples], 0)
    Wm = np.stack([s["unobs_mask"].astype("float32") for s in samples], 0)
    return X, Y, Wm

def depthwise_xy(samples):
    X = np.concatenate([B._unet_channels(s, grid, norm, CFG) for s in samples], 0)
    Y = np.concatenate([np.stack([np.nan_to_num(norm.z3d(v, s["gt"][v], s["month"]), nan=0.0)
                                  for v in VARS], 1) for s in samples], 0)
    Wm = np.concatenate([np.repeat(s["unobs_mask"].astype("float32")[None], D, 0)
                         for s in samples], 0)
    return X, Y, Wm

print("building tensors ...", flush=True); ts = time.time()
build = joint_xy if args.model == "joint" else depthwise_xy
Xtr, Ytr, Wtr = build(train_samples)
Xva, Yva, Wva = build(val_samples)
print(f"  Xtr {Xtr.shape} ({Xtr.nbytes/1e9:.1f} GB) in {time.time()-ts:.1f}s", flush=True)

Xtr_t = torch.from_numpy(Xtr).to(device); del Xtr
Ytr_t = torch.from_numpy(Ytr).to(device); del Ytr
Wtr_t = torch.from_numpy(Wtr).to(device)
Xva_t = torch.from_numpy(Xva).to(device); del Xva

if args.model == "joint":
    c_out, base, batch = len(VARS) * D, C.UNET_JOINT_BASE, C.UNET_JOINT_BATCH
    lr = args.lr or C.UNET_JOINT_LR
else:
    c_out, base, batch = len(VARS), C.UNET_BASE, C.UNET_BATCH
    lr = args.lr or C.UNET_LR
model = UNet2D(Xtr_t.shape[1], c_out, base=base).to(device)
n_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.Adam(model.parameters(), lr=lr)
sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
         if args.schedule == "cosine" else None)
print(f"model: c_in={Xtr_t.shape[1]} c_out={c_out} base={base} params={n_params:,} "
      f"lr={lr} batch={batch}", flush=True)

# =========================================================================
# Validation: physical anomaly RMSE on unobserved val cells (batched fwd)
# =========================================================================
val_months = [s["month"] for s in val_samples]
VAL_TRUE = {v: np.stack([s["gt"][v] for s in val_samples], 0) for v in VARS}
VAL_UNOBS = np.stack([s["unobs_mask"] for s in val_samples], 0)

@torch.no_grad()
def predict_months(X_t, samples):
    """Forward -> {var: (N,D,H,W) absolute fields}."""
    model.eval()
    outs = []
    fb = 8 if args.model == "joint" else 64
    for i in range(0, X_t.shape[0], fb):
        outs.append(model(X_t[i:i + fb]).float().cpu().numpy())
    out = np.concatenate(outs, 0)
    model.train()
    N = len(samples)
    if args.model == "joint":
        out = out.reshape(N, len(VARS), D, H, W)
    else:
        out = out.reshape(N, D, len(VARS), H, W).transpose(0, 2, 1, 3, 4)
    pred = {}
    for k, v in enumerate(VARS):
        arr = np.stack([norm.unz3d(v, out[n, k], samples[n]["month"])
                        for n in range(N)], 0)
        pred[v] = np.where(grid.ocean[None, None], arr, np.nan).astype("float32")
    return pred

def val_rmse():
    pred = predict_months(Xva_t, val_samples)
    ev = metrics.evaluate_masked(pred, VAL_TRUE, VAL_UNOBS, grid.depth)
    return {v: ev["overall"][v] for v in VARS}

# ---- val-selection scalar: mean of per-var RMSE / clim-floor RMSE ----
floor_pred = {v: np.stack([B.predict_clim_floor(s, clim, grid)[v]
                           for s in val_samples], 0) for v in VARS}
ev_floor = metrics.evaluate_masked(floor_pred, VAL_TRUE, VAL_UNOBS, grid.depth)
VAL_FLOOR = {v: ev_floor["overall"][v] for v in VARS}
print(f"val clim floor: TEMP={VAL_FLOOR['TEMP']:.4f} SALT={VAL_FLOOR['SALT']:.4f}",
      flush=True)

# =========================================================================
# Training loop
# =========================================================================
os.makedirs(C.CKPT, exist_ok=True)
ckpt_path = os.path.join(C.CKPT, f"audit_{args.tag}.pt")
N = Xtr_t.shape[0]
nb = int(np.ceil(N / batch))
curves = {"epoch": [], "train_loss": [], "lr": [],
          "val_TEMP": [], "val_SALT": [], "val_score": []}
best = {"score": float("inf"), "epoch": -1}
steps = 0
for ep in range(1, args.epochs + 1):
    perm = torch.randperm(N, device=device)
    ep_loss, nb_seen = 0.0, 0
    for b in range(nb):
        sl = perm[b * batch:(b + 1) * batch]
        opt.zero_grad()
        out = model(Xtr_t[sl])
        w = Wtr_t[sl][:, None]
        loss = (((out - Ytr_t[sl]) ** 2) * w).sum() / (w.sum() * out.shape[1] + 1e-8)
        loss.backward(); opt.step()
        ep_loss += float(loss); nb_seen += 1; steps += 1
    if sched is not None:
        sched.step()
    vr = val_rmse()
    score = float(np.mean([vr[v] / VAL_FLOOR[v] for v in VARS]))
    curves["epoch"].append(ep)
    curves["train_loss"].append(ep_loss / max(nb_seen, 1))
    curves["lr"].append(opt.param_groups[0]["lr"])
    curves["val_TEMP"].append(vr["TEMP"])
    curves["val_SALT"].append(vr["SALT"])
    curves["val_score"].append(score)
    if score < best["score"]:
        best = {"score": score, "epoch": ep,
                "val_TEMP": vr["TEMP"], "val_SALT": vr["SALT"]}
        torch.save({"state_dict": model.state_dict(), "epoch": ep,
                    "tag": args.tag, "model": args.model}, ckpt_path)
    if ep % max(1, args.epochs // 40) == 0 or ep == 1:
        print(f"  ep {ep:4d}/{args.epochs}  loss={curves['train_loss'][-1]:.5f}  "
              f"val TEMP={vr['TEMP']:.4f} SALT={vr['SALT']:.4f}  "
              f"score={score:.4f}  lr={curves['lr'][-1]:.2e}"
              f"{'  *best*' if best['epoch'] == ep else ''}", flush=True)

train_secs = time.time() - t0

# =========================================================================
# Test ONCE from the best-validation checkpoint
# =========================================================================
print(f"\nbest epoch {best['epoch']} (val score {best['score']:.4f}); "
      f"scoring pinned test months ...", flush=True)
model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
Xte, _, _ = build(test_samples)
Xte_t = torch.from_numpy(Xte).to(device); del Xte
TEST_TRUE = {v: np.stack([s["gt"][v] for s in test_samples], 0) for v in VARS}
TEST_UNOBS = np.stack([s["unobs_mask"] for s in test_samples], 0)
pred = predict_months(Xte_t, test_samples)
ev = metrics.evaluate_masked(pred, TEST_TRUE, TEST_UNOBS, grid.depth)
ev_band = metrics.evaluate_layers(pred, TEST_TRUE, TEST_UNOBS, grid.depth, BANDS)
test = {"TEMP": ev["overall"]["TEMP"], "SALT": ev["overall"]["SALT"],
        "by_band": {v: ev_band["by_layer"][v] for v in VARS}}
print(f"TEST unobs anomaly RMSE: TEMP={test['TEMP']:.4f} SALT={test['SALT']:.4f}",
      flush=True)

def git_commit():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"

out = {
    "tag": args.tag, "model": args.model, "config": CFG_NAME,
    "protocol": "protocol_v1", "git_commit": git_commit(), "smoke": args.smoke,
    "seed": args.seed, "device": dev_name,
    "epochs": args.epochs, "batch": batch, "lr": lr, "schedule": args.schedule,
    "optimizer": "Adam", "n_params": n_params, "optimizer_steps": steps,
    "gpu_hours": round(train_secs / 3600.0, 3),
    "train_months": int(tr_idx.size), "val_months": int(va_idx.size),
    "test_months": te_idx.tolist(),
    "val_floor": VAL_FLOOR, "best": best, "test": test,
    "curves": curves, "ckpt": ckpt_path,
}
path = os.path.join(C.CACHE, f"audit_{args.tag}.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nDONE {args.tag} in {train_secs/3600:.2f} h -> {path}", flush=True)
