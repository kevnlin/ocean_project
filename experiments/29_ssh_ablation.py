"""Phase-4 / Task 4.2 — does the pseudo-SSH channel help?

PRE-REGISTERED HYPOTHESIS (written before looking at any result)
----------------------------------------------------------------
Adding the steric-height channel improves full-column TEMP RMSE, **with the
largest gain in the 100-300 m band**, because sea-surface height integrates the
density structure of the whole column and therefore encodes thermocline
displacement — information that SST/SSS, which see only the surface, do not
carry.  The 100-300 m band is currently the weakest layer of the depthwise
U-Net (0.1916 degC vs 0.1589 at 0-100 m, `audit_depthwise_e40`), so it has the
most room to move.

Pre-registered null / negative outcome: no significant change.  The leading
suspect would be that at 1 deg monthly resolution the eddy signal that makes
altimetry valuable is largely averaged away, leaving SSH nearly a linear
function of the upper-ocean heat content the model can already infer from SST
plus the WOA prior.  **A negative result is a result** and gets reported.

Caveat that must appear in the report: the pseudo-SSH is *derived from* the
same TEMP/SALT fields the model reconstructs (see ssh.py).  It is a test of
"does a vertically integrated surface constraint help", not of "does real
altimetry help" — the latter is a Phase-5 question.

Design
------
Two runs, identical in every respect except the cfg (`profiles_woa_surf` vs
`profiles_woa_surf_ssh`): same seed, same profile draws, same epochs, same
validation-selected checkpoint rule, same pinned test months.  The control is
retrained here rather than reusing `audit_depthwise_e40` so that both arms
share this script's exact training loop.

GPU memory
----------
The depthwise tensor stack is ~17 GB for 276 months at 12 channels, which does
not fit alongside other jobs on a shared card.  ``--cpu-tensors`` keeps the
stack in pinned host memory and ships one batch at a time, which fits in well
under 4 GB of device memory at some throughput cost.  Check `nvidia-smi` first.

Run:
    CUDA_VISIBLE_DEVICES=N python experiments/29_ssh_ablation.py
    CUDA_VISIBLE_DEVICES=N python experiments/29_ssh_ablation.py --cpu-tensors
    python experiments/29_ssh_ablation.py --smoke        # CPU, minutes
"""
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

from ocean_tokenizer import baselines as B, config as C, data, metrics
from ocean_tokenizer.anomaly import AnomNorm, Climatology
from ocean_tokenizer.ssh import SSHAnom, load_ssh_cache, ssh_for_indices
from ocean_tokenizer.unet import UNet2D

TRAIN_YEARS = (1985, 2007)
VAL_YEARS = (2008, 2010)
TEST_TIME_INDICES = [1933, 1935, 1936, 1938, 1942, 1946, 1952, 1953,
                     1956, 1965, 1967, 1976]
BANDS = [("0-100m", 0.0, 100.0), ("100-300m", 100.0, 300.0),
         ("300-max", 300.0, 1e9)]
ARMS = {"control_pws": ("profiles", "woa", "surf"),
        "treat_pws_ssh": ("profiles", "woa", "surf", "ssh")}
VARS = B.VARS

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--epochs", type=int, default=C.UNET_EPOCHS)
ap.add_argument("--ssh-cache", default=None)
ap.add_argument("--cpu-tensors", action="store_true",
                help="keep the training stack in host memory (low GPU memory)")
ap.add_argument("--arms", default="control_pws,treat_pws_ssh")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
ARM_NAMES = args.arms.split(",")

t0 = time.time()
device = C.DEVICE if torch.cuda.is_available() else "cpu"
print(f"device={device} cpu_tensors={args.cpu_tensors} smoke={args.smoke}", flush=True)
grid = data.CommonGrid()

tr_idx = data.select_month_indices(C.GT_SOURCE, TRAIN_YEARS)
va_idx = data.select_month_indices(C.GT_SOURCE, VAL_YEARS)
te_idx = np.asarray(TEST_TIME_INDICES)
if args.smoke:
    tr_idx, va_idx, te_idx = tr_idx[:6], va_idx[:2], te_idx[:2]
    args.epochs = 2
print(f"train={tr_idx.size} val={va_idx.size} test={te_idx.size}", flush=True)

ftrain = data.load_gt_fields(tr_idx, grid)
fval = data.load_gt_fields(va_idx, grid)
ftest = data.load_gt_fields(te_idx, grid)
woa = data.woa_prior(grid)
surf_train = {v: ftrain[v] for v in C.VARS_SURF if v in ftrain}
clim = Climatology(ftrain, surf_train)
norm = AnomNorm(clim, ftrain, surf_train)

# ---- pseudo-SSH: statistics from TRAIN months only ----
ssh_path = args.ssh_cache or os.path.join(
    C.CACHE, "ssh_dyn_smoke.npz" if args.smoke else "ssh_dyn.npz")
if not os.path.exists(ssh_path):
    raise SystemExit(f"missing {ssh_path} — run experiments/28_make_ssh.py first")
ssh_all, ssh_idx = load_ssh_cache(ssh_path)
ssh_tr = ssh_for_indices(ssh_all, ssh_idx, tr_idx)
sshnorm = SSHAnom(ssh_tr, ftrain["months"])
ssh_va = ssh_for_indices(ssh_all, ssh_idx, va_idx)
ssh_te = ssh_for_indices(ssh_all, ssh_idx, te_idx)
print(f"pseudo-SSH: train std {np.nanstd(ssh_tr - sshnorm.clim[ftrain['months']-1])*100:.2f} cm",
      flush=True)

rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)


def make_samples(fields, ssh_block):
    out = []
    for t in range(len(fields["months"])):
        s = B.prepare_month(fields, fields, woa, grid, t, rng, C.N_PROFILES)
        s["ssh_z"] = sshnorm.z(ssh_block[t], s["month"])
        out.append(s)
    return out


train_samples = make_samples(ftrain, ssh_tr)
val_samples = make_samples(fval, ssh_va)
test_samples = make_samples(ftest, ssh_te)

D, H, W = grid.ndepth, grid.nlat, grid.nlon
VAL_TRUE = {v: np.stack([s["gt"][v] for s in val_samples], 0) for v in VARS}
VAL_UNOBS = np.stack([s["unobs_mask"] for s in val_samples], 0)
TEST_TRUE = {v: np.stack([s["gt"][v] for s in test_samples], 0) for v in VARS}
TEST_UNOBS = np.stack([s["unobs_mask"] for s in test_samples], 0)
floor = metrics.evaluate_masked(
    {v: np.stack([B.predict_clim_floor(s, clim, grid)[v] for s in val_samples], 0)
     for v in VARS}, VAL_TRUE, VAL_UNOBS, grid.depth)
VAL_FLOOR = {v: floor["overall"][v] for v in VARS}
print(f"val clim floor: TEMP={VAL_FLOOR['TEMP']:.4f} SALT={VAL_FLOOR['SALT']:.4f}",
      flush=True)


def build_xy(samples, cfg):
    X = np.concatenate([B._unet_channels(s, grid, norm, cfg) for s in samples], 0)
    Y = np.concatenate([np.stack([np.nan_to_num(norm.z3d(v, s["gt"][v], s["month"]),
                                                nan=0.0) for v in VARS], 1)
                        for s in samples], 0)
    Wm = np.concatenate([np.repeat(s["unobs_mask"].astype("float32")[None], D, 0)
                         for s in samples], 0)
    return X, Y, Wm


def run_arm(name, cfg):
    Xtr, Ytr, Wtr = build_xy(train_samples, cfg)
    print(f"[{name}] cfg={cfg} c_in={Xtr.shape[1]} "
          f"Xtr {Xtr.shape} ({Xtr.nbytes/1e9:.1f} GB)", flush=True)
    hold = "cpu" if args.cpu_tensors else device
    Xtr_t = torch.from_numpy(Xtr).to(hold); del Xtr
    Ytr_t = torch.from_numpy(Ytr).to(hold); del Ytr
    Wtr_t = torch.from_numpy(Wtr).to(hold)
    Xva = torch.from_numpy(build_xy(val_samples, cfg)[0]).to(hold)
    Xte = torch.from_numpy(build_xy(test_samples, cfg)[0]).to(hold)

    torch.manual_seed(args.seed)
    model = UNet2D(Xtr_t.shape[1], len(VARS), base=C.UNET_BASE).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.UNET_LR)

    @torch.no_grad()
    def predict(X_t, samples):
        model.eval()
        outs = []
        for i in range(0, X_t.shape[0], 64):
            outs.append(model(X_t[i:i + 64].to(device)).float().cpu().numpy())
        model.train()
        out = np.concatenate(outs, 0).reshape(len(samples), D, len(VARS), H, W
                                              ).transpose(0, 2, 1, 3, 4)
        return {v: np.where(grid.ocean[None, None],
                            np.stack([norm.unz3d(v, out[n, k], samples[n]["month"])
                                      for n in range(len(samples))], 0),
                            np.nan).astype("float32")
                for k, v in enumerate(VARS)}

    N = Xtr_t.shape[0]
    nb = int(np.ceil(N / C.UNET_BATCH))
    best = {"score": float("inf"), "epoch": -1, "state": None}
    curves = {"epoch": [], "train_loss": [], "val_TEMP": [], "val_SALT": []}
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(N)
        ep_loss = 0.0
        for b in range(nb):
            sl = perm[b * C.UNET_BATCH:(b + 1) * C.UNET_BATCH]
            xb, yb = Xtr_t[sl].to(device), Ytr_t[sl].to(device)
            wb = Wtr_t[sl].to(device)[:, None]
            opt.zero_grad()
            loss = (((model(xb) - yb) ** 2) * wb).sum() / (wb.sum() * len(VARS) + 1e-8)
            loss.backward(); opt.step()
            ep_loss += float(loss)
        vr = metrics.evaluate_masked(predict(Xva, val_samples), VAL_TRUE,
                                     VAL_UNOBS, grid.depth)["overall"]
        score = float(np.mean([vr[v] / VAL_FLOOR[v] for v in VARS]))
        curves["epoch"].append(ep); curves["train_loss"].append(ep_loss / nb)
        curves["val_TEMP"].append(vr["TEMP"]); curves["val_SALT"].append(vr["SALT"])
        if score < best["score"]:
            best = {"score": score, "epoch": ep,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
        if ep % max(1, args.epochs // 20) == 0 or ep == 1:
            print(f"  [{name}] ep {ep:3d}/{args.epochs} loss={ep_loss/nb:.5f} "
                  f"val TEMP={vr['TEMP']:.4f} SALT={vr['SALT']:.4f} score={score:.4f}"
                  f"{'  *best*' if best['epoch'] == ep else ''}", flush=True)

    model.load_state_dict(best["state"])
    torch.save({"state_dict": best["state"], "epoch": best["epoch"], "cfg": cfg,
                "tag": f"ssh_{name}"}, os.path.join(C.CKPT, f"ssh_{name}.pt"))
    P = predict(Xte, test_samples)
    ev = metrics.evaluate_masked(P, TEST_TRUE, TEST_UNOBS, grid.depth)
    evb = metrics.evaluate_layers(P, TEST_TRUE, TEST_UNOBS, grid.depth, BANDS)
    res = {"cfg": list(cfg), "c_in": int(Xva.shape[1]), "best_epoch": best["epoch"],
           "val_score": best["score"], "curves": curves,
           "test": {v: ev["overall"][v] for v in VARS},
           "test_by_band": {v: evb["by_layer"][v] for v in VARS}}
    print(f"  [{name}] TEST TEMP={res['test']['TEMP']:.4f} "
          f"SALT={res['test']['SALT']:.4f}", flush=True)
    del Xtr_t, Ytr_t, Wtr_t, Xva, Xte
    if device == "cuda":
        torch.cuda.empty_cache()
    return res


results = {n: run_arm(n, ARMS[n]) for n in ARM_NAMES}

if {"control_pws", "treat_pws_ssh"} <= set(results):
    c, t = results["control_pws"], results["treat_pws_ssh"]
    delta = {v: t["test"][v] - c["test"][v] for v in VARS}
    dband = {v: {b[0]: t["test_by_band"][v][b[0]] - c["test_by_band"][v][b[0]]
                 for b in BANDS} for v in VARS}
    results["delta_ssh_minus_control"] = {"full": delta, "by_band": dband}
    print("\nSSH effect (negative = SSH helps):", flush=True)
    for v in VARS:
        print(f"  {v}: full {delta[v]:+.4f} | " +
              " ".join(f"{b[0]} {dband[v][b[0]]:+.4f}" for b in BANDS), flush=True)


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


out = {"task": "ssh_ablation", "protocol": "protocol_v1", "smoke": args.smoke,
       "git_commit": git_commit(), "seed": args.seed, "epochs": args.epochs,
       "device": device, "ssh_cache": os.path.basename(ssh_path),
       "hypothesis": "SSH improves TEMP, largest gain in 100-300 m",
       "results": results, "gpu_hours": round((time.time() - t0) / 3600, 3)}
path = os.path.join(C.CACHE,
                    f"ssh_ablation{'_smoke' if args.smoke else ''}.json")
json.dump(out, open(path, "w"), indent=2)
print(f"\nDONE in {(time.time()-t0)/60:.1f} min -> {path}", flush=True)
