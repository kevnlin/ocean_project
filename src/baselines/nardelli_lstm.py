"""Buongiorno-Nardelli-style stacked-LSTM baseline (ported to PyTorch).

After Buongiorno Nardelli (2020), "A Deep Learning network to retrieve ocean
hydrographic profiles from combined satellite and in situ measurements".  Depth
is treated as the *sequence* dimension: a 2-layer stacked LSTM is unrolled over
the 20 CESM2-LE depth levels, fed the SAME per-profile surface/location/time
vector at every step, and emits a temperature- and salinity-*anomaly* at each
step.  The WOA23 climatology is added back to recover absolute T/S.

Monte-Carlo dropout (dropout left active at inference) provides a Bayesian-style
uncertainty estimate from 50 stochastic forward passes.

Inputs (6, all min-max scaled to [0,1])
    SST anomaly  = CESM2 SST - WOA23 SST climatology (same month & location)
    SSS anomaly  = CESM2 SSS - WOA23 SSS climatology
    lat, lon, sin(2*pi*doy/365), cos(2*pi*doy/365)

Targets per depth step
    T anomaly, S anomaly  (z-scored per depth so the two MSE terms are balanced)
Final prediction = predicted anomaly (un-z-scored) + WOA23 profile.

Loss = MSE(T anomaly) + MSE(S anomaly), masked to valid (finite, in-WOA) depths.

Departures from the original (documented inline)
------------------------------------------------
* The original ingests satellite SSH/ADT and SLA as anomaly inputs; we have no
  SSH here, so the dynamic-height predictor is absent (weaker thermocline skill).
* Anomalies are referenced to the WOA23 monthly climatology interpolated onto
  the 20 CESM2-LE levels; day-of-year is the 15th of each monthly snapshot.
* Anomaly targets are z-scored per depth so the equal-weight T+S MSE is not
  dominated by the (larger-variance) temperature term.

Run
---
    ~/.venv/bin/python src/baselines/nardelli_lstm.py            # full
    ~/.venv/bin/python src/baselines/nardelli_lstm.py --smoke    # quick
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

NAME = "nardelli_lstm"
HIDDEN = 35
N_LAYERS = 2
DROPOUT = 0.25
LR = 1e-3
BATCH = 2048
EPOCHS = 300
PATIENCE = 30
VAL_FRAC = 0.1
MC_PASSES = 50


# --------------------------------------------------------------------------
# WOA profile lookup + anomaly construction
# --------------------------------------------------------------------------
def woa_profiles(woa, var, month, i, j):
    """WOA monthly-climatology profile at (month,i,j) -> (N,D)."""
    return woa[var][np.asarray(month) - 1, :, np.asarray(i), np.asarray(j)]


def woa_surface(woa, var, month, i, j):
    """WOA monthly-climatology surface value at (month,i,j) -> (N,)."""
    return woa[var][np.asarray(month) - 1, np.asarray(i), np.asarray(j)]


class MinMax:
    """Per-feature min-max scaling to [0,1] (training stats; clipped at apply)."""
    def __init__(self, X):
        self.lo = np.nanmin(X, axis=0).astype("float32")
        self.hi = np.nanmax(X, axis=0).astype("float32")
        self.rng = np.where((self.hi - self.lo) < 1e-6, 1.0,
                            self.hi - self.lo).astype("float32")

    def __call__(self, X):
        z = (X - self.lo) / self.rng
        return np.clip(np.nan_to_num(z, nan=0.0), 0.0, 1.0).astype("float32")


def build_inputs(d, woa):
    """6-vector of raw (unscaled) inputs for profiles / cells described by dict d."""
    sst_anom = d["sst"] - woa_surface(woa, "SST", d["month_arr"], d["i"], d["j"])
    sss_anom = d["sss"] - woa_surface(woa, "SSS", d["month_arr"], d["i"], d["j"])
    return np.stack([
        sst_anom.astype("float32"), sss_anom.astype("float32"),
        d["lat"].astype("float32"), d["lon"].astype("float32"),
        d["sin_doy"].astype("float32"), d["cos_doy"].astype("float32"),
    ], axis=1)


# --------------------------------------------------------------------------
# Stacked LSTM (depth as sequence)
# --------------------------------------------------------------------------
class StackedLSTM(nn.Module):
    def __init__(self, c_in, D):
        super().__init__()
        self.D = D
        self.lstm = nn.LSTM(c_in, HIDDEN, num_layers=N_LAYERS,
                            batch_first=True, dropout=DROPOUT)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(HIDDEN, 2)               # T-anom, S-anom per step

    def forward(self, x):                              # x: (B, D, c_in)
        h, _ = self.lstm(x)
        return self.head(self.drop(h))                 # (B, D, 2)

    def enable_mc_dropout(self):
        """Keep dropout stochastic at inference (Monte-Carlo dropout)."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        # nn.LSTM's inter-layer dropout follows the module's training flag:
        self.lstm.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    global EPOCHS, PATIENCE, MC_PASSES
    if args.smoke:
        EPOCHS, PATIENCE, MC_PASSES = 60, 12, 15

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{NAME}] device={device} smoke={args.smoke}")

    S = cm.build_split(smoke=args.smoke)
    grid, woa = S["grid"], S["woa"]
    D = grid.ndepth
    tr = cm.extract_profiles(S["train_samples"], grid)
    tr["month_arr"] = tr["month"]

    # ---- inputs ----
    Xraw = build_inputs(tr, woa)
    scaler = MinMax(Xraw)
    X = scaler(Xraw)                                   # (N,6) in [0,1]

    # ---- anomaly targets (GT - WOA), z-scored per depth ----
    # Model self-climatology (per calendar month, per cell, mean over TRAINING
    # months): the fallback additive base where WOA23 is NaN.  WOA23 leaves
    # ~6-13% of CESM2 ocean cells uncovered (high-latitude / marginal seas);
    # without a fallback those cells would drop out of scoring and Nardelli would
    # be evaluated on an easier cell population than the other ML baselines.  The
    # LSTM still predicts the WOA-referenced anomaly; only the additive base
    # falls back from WOA -> spatially-resolved model climatology there.
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", category=RuntimeWarning)
        self_clim = {v: np.stack([
            (np.nanmean(S["ftrain"][v][S["ftrain"]["months"] == m], axis=0)
             if (S["ftrain"]["months"] == m).any()
             else np.full((D, grid.nlat, grid.nlon), np.nan, "float32"))
            for m in range(1, 13)], axis=0).astype("float32")     # (12,D,H,W)
            for v in cm.VARS}

    woaT = woa_profiles(woa, "TEMP", tr["month"], tr["i"], tr["j"])
    woaS = woa_profiles(woa, "SALT", tr["month"], tr["i"], tr["j"])
    anomT = tr["TEMP"] - woaT
    anomS = tr["SALT"] - woaS
    maskT = (np.isfinite(anomT)).astype("float32")
    maskS = (np.isfinite(anomS)).astype("float32")
    stdT = cm.DepthStandardizer(anomT)                 # fits on gap-filled anomalies
    stdS = cm.DepthStandardizer(anomS)
    fT, _ = cm.fill_profile_matrix(anomT)
    fS, _ = cm.fill_profile_matrix(anomS)
    yT = stdT.z(fT).astype("float32")
    yS = stdS.z(fS).astype("float32")
    print(f"  profiles N={X.shape[0]}  inputs={X.shape[1]}  seq_len(D)={D}")

    # ---- train/val split (separate RNG) ----
    rng = np.random.default_rng(cm.C.SEED + 2)
    N = X.shape[0]
    perm = rng.permutation(N)
    nval = max(1, int(VAL_FRAC * N))
    vi, ti = perm[:nval], perm[nval:]

    model = StackedLSTM(X.shape[1], D).to(device)
    print(f"  LSTM params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # tensors: input repeated across the depth/sequence axis
    def seq(idx):
        return torch.from_numpy(np.repeat(X[idx][:, None, :], D, axis=1)).to(device)

    def tgt(a, idx): return torch.from_numpy(a[idx]).to(device)
    Xt, Xv = seq(ti), seq(vi)
    yTt, ySt, mTt, mSt = (tgt(yT, ti), tgt(yS, ti), tgt(maskT, ti), tgt(maskS, ti))
    yTv, ySv, mTv, mSv = (tgt(yT, vi), tgt(yS, vi), tgt(maskT, vi), tgt(maskS, vi))

    def loss_fn(out, yt, ys, mt, ms):
        pt, ps = out[..., 0], out[..., 1]
        lT = ((pt - yt) ** 2 * mt).sum() / mt.sum().clamp_min(1)
        lS = ((ps - ys) ** 2 * ms).sum() / ms.sum().clamp_min(1)
        return lT + lS

    nb = int(np.ceil(Xt.shape[0] / BATCH))
    best, best_state, since = float("inf"), None, 0
    for ep in range(EPOCHS):
        model.train()
        pr = torch.randperm(Xt.shape[0], device=device)
        for b in range(nb):
            sl = pr[b * BATCH:(b + 1) * BATCH]
            opt.zero_grad()
            loss_fn(model(Xt[sl]), yTt[sl], ySt[sl], mTt[sl], mSt[sl]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xv), yTv, ySv, mTv, mSv).item()
        if v < best - 1e-5:
            best, since, best_state = v, 0, {k: val.detach().cpu().clone()
                                            for k, val in model.state_dict().items()}
        else:
            since += 1
        if ep % 20 == 0 or since == 0:
            print(f"    epoch {ep:3d}  val={v:.5f}  best={best:.5f}  patience={since}")
        if since >= PATIENCE:
            print(f"  early stop at epoch {ep} (best val={best:.5f})")
            break
    model.load_state_dict(best_state)

    # ---- MC-dropout prediction of full test fields ----
    def predict_cells(feats):
        feats = dict(feats)
        feats["month_arr"] = np.full(feats["i"].size, feats["month"])
        xc = scaler(build_inputs(feats, woa))
        xseq = torch.from_numpy(np.repeat(xc[:, None, :], D, axis=1)).to(device)
        wT = woa_profiles(woa, "TEMP", feats["month_arr"], feats["i"], feats["j"])
        wS = woa_profiles(woa, "SALT", feats["month_arr"], feats["i"], feats["j"])
        vT = np.isfinite(wT).astype("float32")           # WOA-available mask
        vS = np.isfinite(wS).astype("float32")
        # base = WOA where available, else spatially-resolved model self-climatology
        mi = feats["month"] - 1
        scT = self_clim["TEMP"][mi][:, feats["i"], feats["j"]].T   # (ncell,D)
        scS = self_clim["SALT"][mi][:, feats["i"], feats["j"]].T
        baseT = np.where(vT > 0, wT, scT)
        baseS = np.where(vS > 0, wS, scS)
        # Where WOA is missing the LSTM is out-of-distribution (it never trained
        # there), so we zero its anomaly and degrade gracefully to climatology
        # rather than extrapolate; this keeps full ocean coverage for a fair
        # same-cell comparison with the other ML baselines.
        model.eval(); model.enable_mc_dropout()         # dropout active at inference
        Ts, Ss = [], []
        with torch.no_grad():
            for _ in range(MC_PASSES):
                out = model(xseq).cpu().numpy()          # (ncell,D,2) z-anomaly
                Ts.append(stdT.unz(out[..., 0]) * vT + baseT)
                Ss.append(stdS.unz(out[..., 1]) * vS + baseS)
        Ts, Ss = np.stack(Ts), np.stack(Ss)              # (MC, ncell, D)
        return {"TEMP": Ts.mean(0).astype("float32"),
                "SALT": Ss.mean(0).astype("float32"),
                "TEMP_std": Ts.std(0).astype("float32"),
                "SALT_std": Ss.std(0).astype("float32")}

    preds, stds = cm.predict_test_fields(S["test_samples"], grid, predict_cells,
                                         want_std=True)
    rmse_d, bands = cm.evaluate_fields(preds, S["test_samples"], grid)
    cm.print_band_table(NAME, bands)
    msT = np.nanmean(np.stack([s["TEMP"] for s in stds]))
    msS = np.nanmean(np.stack([s["SALT"] for s in stds]))
    print(f"  mean MC-dropout std: TEMP={msT:.4f} degC  SALT={msS:.4f} PSU")

    # ---- save ----
    cm.save_predictions(NAME, preds, grid, S["te_idx"], stds=stds)
    cm.save_depth_rmse(NAME, rmse_d, grid)
    cm.write_band_csv(os.path.join(cm.PRED_DIR, f"{NAME}_rmse.csv"),
                      "Nardelli LSTM (MC-dropout)", bands)
    torch.save({
        "model": model.state_dict(),
        "scaler": dict(lo=scaler.lo, hi=scaler.hi, rng=scaler.rng),
        "stdT": dict(mean=stdT.mean, std=stdT.std),
        "stdS": dict(mean=stdS.mean, std=stdS.std),
        "config": dict(hidden=HIDDEN, layers=N_LAYERS, dropout=DROPOUT,
                       mc_passes=MC_PASSES, depth=D),
    }, os.path.join(cm.CKPT_DIR, f"{NAME}.pt"))
    print(f"  wrote {os.path.join(cm.CKPT_DIR, NAME + '.pt')}")
    print(f"[{NAME}] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
