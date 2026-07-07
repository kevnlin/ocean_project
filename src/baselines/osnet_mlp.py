"""OSnet-style MLP baseline (adapted from Pauthenet et al., Ocean Science 2022).

Reference: https://os.copernicus.org/articles/18/1221/2022/

OSnet maps surface + location + time predictors directly to full-depth T and S
profiles with a fully-connected network, and is trained as a *bootstrap
ensemble* whose spread gives an uncertainty estimate.  The original network also
emits an auxiliary mixed-layer "mask" that segments the well-mixed surface
layer; we reproduce that head (``K``) when the MLD is derivable from the data.

Architecture
------------
    in(8) -> 256 (ReLU) -> 256 (ReLU) -> [ T(D) | S(D) | K(D) ]
where D = 20 CESM2-LE depth levels (the same grid as the existing baselines).
T and S are predicted in per-depth z-scored space; K are logits.

Inputs (8) -- reduced from the paper's 12 (no SSH / SLA / surface currents):
    lat/90, sin(lon), cos(lon), sin(2*pi*doy/365), cos(2*pi*doy/365),
    bathymetry (normalised), SST, SSS
SST/SSS are read at the profile location (dense surface obs -> no leakage).

Loss = MSE(T_z) + MSE(S_z) + BCE(K)   (equal weight; T/S in z-space so the
regression terms are commensurate).  Invalid (below-seafloor) depths are masked
out of every term.

Ensemble: 15 members, each with a distinct seed AND a bootstrap resample (draw
with replacement) of the training profiles.  Prediction = ensemble mean; the
ensemble std is saved as the uncertainty estimate.

Departures from the paper (documented inline)
---------------------------------------------
* No SSH/SLA/currents -> fewer, weaker dynamical predictors.
* Bathymetry is approximated by the deepest valid CESM2 level (topography), as a
  real satellite/GEBCO bathymetry product is not part of this synthetic setup.
* MLD for the K head uses the simple dT = 0.2 degC threshold from the shallowest
  level (a standard, profile-only MLD criterion).

Run
---
    ~/.venv/bin/python src/baselines/osnet_mlp.py            # full (15 members)
    ~/.venv/bin/python src/baselines/osnet_mlp.py --smoke    # quick (3 members)
"""
from __future__ import annotations
import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as cm  # noqa: E402

NAME = "osnet_mlp"
HIDDEN = 256
N_MEMBERS = 15
EPOCHS = 200
PATIENCE = 25
BATCH = 4096
LR = 1e-3
VAL_FRAC = 0.1
MLD_DT = 0.2                     # degC threshold for mixed-layer depth


# --------------------------------------------------------------------------
# Mixed-layer-depth mask K (per-depth above-MLD indicator), from the T profile
# --------------------------------------------------------------------------
def mld_mask(profiles_T, depths):
    """K[d]=1 if depth d is within the mixed layer (|T-T_surf| <= MLD_DT).

    Returns (N,D) float in {0,1}; computed from the (gap-filled) T profile only.
    """
    filled, _ = cm.fill_profile_matrix(profiles_T)
    surf = filled[:, :1]
    return (np.abs(filled - surf) <= MLD_DT).astype("float32")


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
class Features:
    def __init__(self, tr, bathy):
        self.bathy = bathy
        self.smean = np.array([np.nanmean(tr["sst"]), np.nanmean(tr["sss"])], "float32")
        ss = np.array([np.nanstd(tr["sst"]), np.nanstd(tr["sss"])], "float32")
        self.sstd = np.where(ss < 1e-6, 1.0, ss).astype("float32")
        bv = bathy[np.isfinite(bathy)]
        self.bmean, self.bstd = float(bv.mean()), float(bv.std() + 1e-6)

    def build(self, d):
        b = self.bathy[d["i"], d["j"]]
        b = np.where(np.isfinite(b), b, self.bmean)
        cols = [
            (d["lat"] / 90.0).astype("float32"),
            np.sin(2 * np.pi * d["lon"] / 360.0), np.cos(2 * np.pi * d["lon"] / 360.0),
            d["sin_doy"], d["cos_doy"],
            ((b - self.bmean) / self.bstd).astype("float32"),
            np.nan_to_num((d["sst"] - self.smean[0]) / self.sstd[0], nan=0.0),
            np.nan_to_num((d["sss"] - self.smean[1]) / self.sstd[1], nan=0.0),
        ]
        return np.stack(cols, axis=1).astype("float32")


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------
class OSnet(nn.Module):
    def __init__(self, c_in, D):
        super().__init__()
        self.D = D
        self.body = nn.Sequential(
            nn.Linear(c_in, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        )
        self.head = nn.Linear(HIDDEN, 3 * D)        # T | S | K-logits

    def forward(self, x):
        h = self.head(self.body(x))
        return h[:, :self.D], h[:, self.D:2 * self.D], h[:, 2 * self.D:]


def train_member(Xtr, yT, yS, yK, mT, mS, c_in, D, seed, device):
    torch.manual_seed(seed)
    model = OSnet(c_in, D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()

    # fixed val split (same indices across members for comparability)
    rng = np.random.default_rng(cm.C.SEED + 7)
    N = Xtr.shape[0]
    perm = rng.permutation(N)
    nval = max(1, int(VAL_FRAC * N))
    vi, ti = perm[:nval], perm[nval:]
    # bootstrap resample of the training portion (member-specific)
    brng = np.random.default_rng(seed)
    bti = ti[brng.integers(0, ti.size, size=ti.size)]

    def to(idx, a): return torch.from_numpy(a[idx]).to(device)
    Xt, Xv = to(bti, Xtr), to(vi, Xtr)
    args_t = [to(bti, a) for a in (yT, yS, yK, mT, mS)]
    args_v = [to(vi, a) for a in (yT, yS, yK, mT, mS)]

    def loss_fn(X, tT, tS, tK, wT, wS):
        pT, pS, pK = model(X)
        lT = ((pT - tT) ** 2 * wT).sum() / wT.sum().clamp_min(1)
        lS = ((pS - tS) ** 2 * wS).sum() / wS.sum().clamp_min(1)
        lK = bce(pK, tK)
        return lT + lS + lK

    nb = int(np.ceil(Xt.shape[0] / BATCH))
    best, best_state, since = float("inf"), None, 0
    for ep in range(EPOCHS):
        model.train()
        pr = torch.randperm(Xt.shape[0], device=device)
        for b in range(nb):
            sl = pr[b * BATCH:(b + 1) * BATCH]
            opt.zero_grad()
            loss_fn(Xt[sl], *[a[sl] for a in args_t]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(Xv, *args_v).item()
        if v < best - 1e-5:
            best, since = v, 0
            best_state = {k: val.detach().cpu().clone() for k, val in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    global N_MEMBERS, EPOCHS, PATIENCE
    if args.smoke:
        N_MEMBERS, EPOCHS, PATIENCE = 3, 60, 15

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{NAME}] device={device} smoke={args.smoke}")

    S = cm.build_split(smoke=args.smoke)
    grid = S["grid"]
    D = grid.ndepth
    tr = cm.extract_profiles(S["train_samples"], grid)
    bathy = cm.bathy_field(S["ftrain"], grid)

    stdT = cm.DepthStandardizer(tr["TEMP"])
    stdS = cm.DepthStandardizer(tr["SALT"])
    feats = Features(tr, bathy)

    X = feats.build(tr)
    fT, mT = cm.fill_profile_matrix(tr["TEMP"])
    fS, mS = cm.fill_profile_matrix(tr["SALT"])
    yT = stdT.z(fT).astype("float32")
    yS = stdS.z(fS).astype("float32")
    yK = mld_mask(tr["TEMP"], grid.depth)
    mTf, mSf = mT.astype("float32"), mS.astype("float32")
    print(f"  profiles N={X.shape[0]}  features={X.shape[1]}  members={N_MEMBERS}")
    print(f"  MLP params/member: "
          f"{sum(p.numel() for p in OSnet(X.shape[1], D).parameters()):,}")

    # ---- train ensemble ----
    members = []
    for k in range(N_MEMBERS):
        m, v = train_member(X, yT, yS, yK, mTf, mSf, X.shape[1], D,
                            seed=cm.C.SEED + 100 + k, device=device)
        members.append(m)
        print(f"    member {k:2d}  val_loss={v:.5f}")

    # ---- predict test fields: ensemble mean + std ----
    def predict_cells(fdict):
        Xc = torch.from_numpy(feats.build(fdict)).to(device)
        Ts, Ss = [], []
        with torch.no_grad():
            for m in members:
                pT, pS, _ = m(Xc)
                Ts.append(stdT.unz(pT.cpu().numpy()))
                Ss.append(stdS.unz(pS.cpu().numpy()))
        Ts, Ss = np.stack(Ts), np.stack(Ss)          # (M, ncell, D)
        return {"TEMP": Ts.mean(0).astype("float32"),
                "SALT": Ss.mean(0).astype("float32"),
                "TEMP_std": Ts.std(0).astype("float32"),
                "SALT_std": Ss.std(0).astype("float32")}

    preds, stds = cm.predict_test_fields(S["test_samples"], grid, predict_cells,
                                         want_std=True)
    rmse_d, bands = cm.evaluate_fields(preds, S["test_samples"], grid)
    cm.print_band_table(NAME, bands)
    # mean ensemble spread (uncertainty sanity check)
    msT = np.nanmean(np.stack([s["TEMP"] for s in stds]))
    msS = np.nanmean(np.stack([s["SALT"] for s in stds]))
    print(f"  mean ensemble std: TEMP={msT:.4f} degC  SALT={msS:.4f} PSU")

    # ---- save ----
    cm.save_predictions(NAME, preds, grid, S["te_idx"], stds=stds)
    cm.save_depth_rmse(NAME, rmse_d, grid)
    cm.write_band_csv(os.path.join(cm.PRED_DIR, f"{NAME}_rmse.csv"),
                      "OSnet MLP (15x ens)", bands)
    torch.save({
        "members": [m.state_dict() for m in members],
        "feat": dict(smean=feats.smean, sstd=feats.sstd,
                     bmean=feats.bmean, bstd=feats.bstd),
        "stdT": dict(mean=stdT.mean, std=stdT.std),
        "stdS": dict(mean=stdS.mean, std=stdS.std),
        "bathy": bathy,
        "config": dict(hidden=HIDDEN, n_members=N_MEMBERS, depth=D, mld_dt=MLD_DT),
    }, os.path.join(cm.CKPT_DIR, f"{NAME}.pt"))
    print(f"  wrote {os.path.join(cm.CKPT_DIR, NAME + '.pt')}")
    print(f"[{NAME}] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
