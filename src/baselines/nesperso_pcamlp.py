"""NeSPReSO-style PCA + MLP baseline (reimplemented from the paper).

NeSPReSO (Neural Synthetic Profiles from Reduced-Order Spaces of Observations)
predicts the low-dimensional PCA representation of temperature and salinity
profiles from a small vector of surface / location / time predictors, then maps
the predicted PC scores back to full-depth profiles via the (fixed) PCA basis.

Pipeline
--------
1.  PCA (sklearn) on per-depth-standardised TRAINING profiles:
        15 components for T, 15 for S  (we assert each retains >=99% variance).
    PCA is fit on TRAINING profiles only -> no leakage from the test months.
2.  MLP   in(8) -> 512 (ReLU, 0.2 dropout) -> 512 (ReLU, 0.2 dropout) -> 30 (linear)
    Output = 30 *unit-variance* PC scores (15 T + 15 S).
3.  Inverse-PCA the predicted scores -> z-scored profiles -> physical T/S.
4.  Loss = WMSE (MSE on the unit-variance PC scores)
            + FMSE (MSE on the reconstructed z-scored profiles, masked to valid
                    depths so fabricated below-seafloor cells are not scored).
    Both terms live in unit-magnitude spaces (the paper's WMSE + FMSE).

Departures from the original paper (documented inline below)
------------------------------------------------------------
* The original NeSPReSO uses satellite SSH / ADT as a key predictor.  We do NOT
  have SSH in this synthetic CESM2-LE setup, so we DROP that input.  Expect
  degraded mid-depth (~100-600 m) skill, where SSH carries most of the dynamic-
  height information about the thermocline -- flagged again where relevant.
* Our profiles are on 20 CESM2-LE levels (5-985 m), not the paper's fine grid,
  and CESM2 bottom topography leaves NaNs below the seafloor -> we gap-fill for
  the PCA basis and mask those cells in the profile-space (FMSE) loss.
* Day-of-year is derived from the monthly snapshot (15th of the month).

Run
---
    ~/.venv/bin/python src/baselines/nesperso_pcamlp.py            # full
    ~/.venv/bin/python src/baselines/nesperso_pcamlp.py --smoke    # quick check
"""
from __future__ import annotations
import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as cm  # noqa: E402

NAME = "nesperso_pcamlp"
N_PC = 15                       # per variable (T, S)
HIDDEN = 512
DROPOUT = 0.2
LR = 1e-3
BATCH = 300
MAX_EPOCHS = 8000
PATIENCE = 500                  # early-stopping patience (epochs) on val loss
VAL_FRAC = 0.1


# --------------------------------------------------------------------------
# Feature construction (8 predictors; SSH/ADT intentionally absent)
# --------------------------------------------------------------------------
class FeatScaler:
    """Standardise SST/SSS using TRAINING-profile statistics (NaN-aware)."""
    def __init__(self, sst, sss):
        self.m = np.array([np.nanmean(sst), np.nanmean(sss)], "float32")
        s = np.array([np.nanstd(sst), np.nanstd(sss)], "float32")
        self.s = np.where(s < 1e-6, 1.0, s).astype("float32")

    def features(self, d):
        sst = (d["sst"] - self.m[0]) / self.s[0]
        sss = (d["sss"] - self.m[1]) / self.s[1]
        lat, lon = d["lat"], d["lon"]
        cols = [
            np.nan_to_num(sst, nan=0.0),
            np.nan_to_num(sss, nan=0.0),
            np.sin(2 * np.pi * lat / 180.0), np.cos(2 * np.pi * lat / 180.0),
            np.sin(2 * np.pi * lon / 360.0), np.cos(2 * np.pi * lon / 360.0),
            d["sin_doy"], d["cos_doy"],
        ]
        return np.stack(cols, axis=1).astype("float32")


# --------------------------------------------------------------------------
# Per-depth standardisation + PCA basis (fit on TRAINING profiles only)
# --------------------------------------------------------------------------
class ProfilePCA:
    def __init__(self, profiles, n_pc):
        filled, mask = cm.fill_profile_matrix(profiles)        # (N,D)
        self.mean_d = filled.mean(0).astype("float32")
        sd = filled.std(0).astype("float32")
        self.std_d = np.where(sd < 1e-6, 1.0, sd).astype("float32")
        Z = (filled - self.mean_d) / self.std_d
        self.pca = PCA(n_components=n_pc).fit(Z)
        self.evr = self.pca.explained_variance_ratio_
        self.eig = self.pca.explained_variance_.astype("float32")        # score var
        self.comp = self.pca.components_.astype("float32")               # (n_pc, D)
        self.pmean = self.pca.mean_.astype("float32")                    # (D,)
        self._filled, self._mask = filled, mask

    def z(self, profiles_filled):
        return (profiles_filled - self.mean_d) / self.std_d

    def scores_norm(self, profiles_filled):
        """Unit-variance PC scores for given (already filled) profiles."""
        s = self.pca.transform(self.z(profiles_filled))      # (N,n_pc)
        return (s / np.sqrt(self.eig)).astype("float32")


# --------------------------------------------------------------------------
# MLP
# --------------------------------------------------------------------------
class PCAMLP(nn.Module):
    def __init__(self, c_in, c_out=2 * N_PC):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c_in, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, c_out),
        )

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        global MAX_EPOCHS, PATIENCE
        MAX_EPOCHS, PATIENCE = 200, 40

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{NAME}] device={device} smoke={args.smoke}")

    S = cm.build_split(smoke=args.smoke)
    grid = S["grid"]
    tr = cm.extract_profiles(S["train_samples"], grid)

    # ---- PCA bases (training profiles only) ----
    pcaT = ProfilePCA(tr["TEMP"], N_PC)
    pcaS = ProfilePCA(tr["SALT"], N_PC)
    print(f"  PCA variance retained: TEMP={pcaT.evr.sum()*100:.2f}%  "
          f"SALT={pcaS.evr.sum()*100:.2f}%  (15 comps each)")
    assert pcaT.evr.sum() >= 0.99, "TEMP PCA < 99% variance"
    assert pcaS.evr.sum() >= 0.99, "SALT PCA < 99% variance"

    # ---- features + targets ----
    scaler = FeatScaler(tr["sst"], tr["sss"])
    X = scaler.features(tr)                                  # (N,8)
    filledT, maskT = cm.fill_profile_matrix(tr["TEMP"])
    filledS, maskS = cm.fill_profile_matrix(tr["SALT"])
    yT = pcaT.scores_norm(filledT)                           # (N,15)
    yS = pcaS.scores_norm(filledS)
    Y = np.concatenate([yT, yS], 1).astype("float32")        # (N,30)
    ZT = pcaT.z(filledT).astype("float32")                   # (N,D) z-profile targets
    ZS = pcaS.z(filledS).astype("float32")

    # ---- train/val split (separate RNG so the main stream is untouched) ----
    rng = np.random.default_rng(cm.C.SEED + 1)
    N = X.shape[0]
    perm = rng.permutation(N)
    nval = max(1, int(VAL_FRAC * N))
    vi, ti = perm[:nval], perm[nval:]
    print(f"  profiles: train={ti.size} val={vi.size}  features={X.shape[1]}")

    # torch tensors
    def T(a, idx): return torch.from_numpy(a[idx]).to(device)
    Xtr, Ytr, ZTtr, ZStr = T(X, ti), T(Y, ti), T(ZT, ti), T(ZS, ti)
    MTtr, MStr = T(maskT.astype("float32"), ti), T(maskS.astype("float32"), ti)
    Xva, Yva, ZTva, ZSva = T(X, vi), T(Y, vi), T(ZT, vi), T(ZS, vi)
    MTva, MSva = T(maskT.astype("float32"), vi), T(maskS.astype("float32"), vi)

    # PCA basis as torch tensors (for differentiable inverse transform / FMSE)
    def basis(p):
        return (torch.from_numpy(p.comp).to(device),          # (15,D)
                torch.from_numpy(p.pmean).to(device),          # (D,)
                torch.from_numpy(np.sqrt(p.eig)).to(device))   # (15,)
    compT, pmT, sqeT = basis(pcaT)
    compS, pmS, sqeS = basis(pcaS)

    def recon_z(scores_norm, comp, pmean, sqe):
        """unit-variance scores -> z-scored profile (matches PCA.inverse_transform)."""
        return (scores_norm * sqe) @ comp + pmean

    model = PCAMLP(X.shape[1]).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"  MLP params: {nparam:,}")
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def losses(out, y, zt, zs, mt, ms):
        wmse = ((out - y) ** 2).mean()                         # PC-score space
        zt_hat = recon_z(out[:, :N_PC], compT, pmT, sqeT)
        zs_hat = recon_z(out[:, N_PC:], compS, pmS, sqeS)
        fT = ((zt_hat - zt) ** 2 * mt).sum() / mt.sum().clamp_min(1)
        fS = ((zs_hat - zs) ** 2 * ms).sum() / ms.sum().clamp_min(1)
        return wmse + 0.5 * (fT + fS)

    nb = int(np.ceil(Xtr.shape[0] / BATCH))
    best_val, best_state, since = float("inf"), None, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        pr = torch.randperm(Xtr.shape[0], device=device)
        for b in range(nb):
            sl = pr[b * BATCH:(b + 1) * BATCH]
            opt.zero_grad()
            loss = losses(model(Xtr[sl]), Ytr[sl], ZTtr[sl], ZStr[sl],
                          MTtr[sl], MStr[sl])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = losses(model(Xva), Yva, ZTva, ZSva, MTva, MSva).item()
        if vloss < best_val - 1e-5:
            best_val, since = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if ep % 50 == 0 or since == 0:
            print(f"    epoch {ep:4d}  val={vloss:.5f}  best={best_val:.5f}  patience={since}")
        if since >= PATIENCE:
            print(f"  early stop at epoch {ep} (best val={best_val:.5f})")
            break
    model.load_state_dict(best_state)
    model.eval()

    # ---- predict full test fields (every ocean cell, dense surface inputs) ----
    def predict_cells(feats):
        X = torch.from_numpy(scaler.features(feats)).to(device)
        with torch.no_grad():
            out = model(X)
            zt = recon_z(out[:, :N_PC], compT, pmT, sqeT).cpu().numpy()
            zs = recon_z(out[:, N_PC:], compS, pmS, sqeS).cpu().numpy()
        TEMP = zt * pcaT.std_d + pcaT.mean_d
        SALT = zs * pcaS.std_d + pcaS.mean_d
        return {"TEMP": TEMP.astype("float32"), "SALT": SALT.astype("float32")}

    preds, _ = cm.predict_test_fields(S["test_samples"], grid, predict_cells)
    rmse_d, bands = cm.evaluate_fields(preds, S["test_samples"], grid)
    cm.print_band_table(NAME, bands)

    # ---- save ----
    cm.save_predictions(NAME, preds, grid, S["te_idx"])
    cm.save_depth_rmse(NAME, rmse_d, grid)
    cm.write_band_csv(os.path.join(cm.PRED_DIR, f"{NAME}_rmse.csv"),
                      "NeSPReSO PCA+MLP", bands)
    torch.save({
        "model": model.state_dict(),
        "feat_mean": scaler.m, "feat_std": scaler.s,
        "pcaT": dict(mean_d=pcaT.mean_d, std_d=pcaT.std_d, comp=pcaT.comp,
                     pmean=pcaT.pmean, eig=pcaT.eig, evr=pcaT.evr),
        "pcaS": dict(mean_d=pcaS.mean_d, std_d=pcaS.std_d, comp=pcaS.comp,
                     pmean=pcaS.pmean, eig=pcaS.eig, evr=pcaS.evr),
        "config": dict(n_pc=N_PC, hidden=HIDDEN, dropout=DROPOUT, lr=LR,
                       batch=BATCH, val_frac=VAL_FRAC),
    }, os.path.join(cm.CKPT_DIR, f"{NAME}.pt"))
    print(f"  wrote {os.path.join(cm.CKPT_DIR, NAME + '.pt')}")
    print(f"[{NAME}] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
