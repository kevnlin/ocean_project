"""Baselines and the profiles/prior/surface configuration sweep.

Methods
-------
* climatology      : prediction == WOA23 monthly climatological prior
* nearest          : horizontal nearest-profile fill (optionally over WOA residual)
* mlp              : pointwise MLP on geo/time + (prior / nearest-obs / surface)
* unet             : depthwise 2D U-Net inpainting from sparse-obs maps + priors

Configs (input information available to a method)
------------------------------------------------
* profiles_only        : sparse Argo profiles only
* woa_only             : WOA climatology prior only
* profiles_woa         : profiles + WOA prior
* profiles_woa_surf    : profiles + WOA prior + SST/SSS surface fields
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

from . import config as C
from . import argo
from .unet import UNet2D

VARS = ("TEMP", "SALT")


# --------------------------------------------------------------------------
# Normalisation statistics (per variable, per depth) from training GT
# --------------------------------------------------------------------------
class Norm:
    def __init__(self, train_fields, surf_train):
        self.mean = {}
        self.std = {}
        for v in VARS:
            x = train_fields[v]                       # (T,D,H,W)
            self.mean[v] = np.nanmean(x, axis=(0, 2, 3)).astype("float32")  # (D,)
            sd = np.nanstd(x, axis=(0, 2, 3)).astype("float32")
            self.std[v] = np.where(sd < 1e-6, 1.0, sd)
        self.smean = {}
        self.sstd = {}
        for v in C.VARS_SURF:
            if v in surf_train:
                s = surf_train[v]
                self.smean[v] = float(np.nanmean(s))
                sd = float(np.nanstd(s))
                self.sstd[v] = sd if sd > 1e-6 else 1.0

    # `month` is accepted for interface parity with anomaly.AnomNorm (which is
    # month-dependent); the absolute normaliser ignores it.
    def z3d(self, v, arr, month=None):    # arr (D,H,W) or (...,D,H,W)
        return (arr - self.mean[v][:, None, None]) / self.std[v][:, None, None]

    def unz3d(self, v, arr, month=None):
        return arr * self.std[v][:, None, None] + self.mean[v][:, None, None]

    def zsurf(self, v, arr, month=None):
        return (arr - self.smean[v]) / self.sstd[v]


# --------------------------------------------------------------------------
# Geometry helper: lat/lon -> unit-sphere xyz (handles lon periodicity & poles)
# --------------------------------------------------------------------------
def _xyz(lat, lon):
    la = np.deg2rad(lat); lo = np.deg2rad(lon)
    return np.stack([np.cos(la) * np.cos(lo),
                     np.cos(la) * np.sin(lo),
                     np.sin(la)], axis=-1)


# --------------------------------------------------------------------------
# Per-month sample assembly
# --------------------------------------------------------------------------
def prepare_month(fields, surf, woa, grid, t, rng, n_profiles):
    """Assemble everything a method might need for monthly snapshot `t`."""
    mo = fields["months"][t]
    prof = argo.sample_profiles(fields, t, grid, n_profiles, rng)

    s = {"month": mo, "prof": prof}
    s["gt"] = {v: fields[v][t] for v in VARS}                       # (D,H,W)
    s["woa"] = {v: woa[v][mo - 1] for v in VARS}                    # (D,H,W)
    s["obs"] = {v: argo.build_obs_grid(prof, grid, v) for v in VARS}
    for sv in C.VARS_SURF:
        s.setdefault("surf", {})[sv] = surf[sv][t] if sv in surf else None

    # nearest-profile assignment for every ocean grid cell (shared across depth)
    oi, oj = np.where(grid.ocean)
    cell_xyz = _xyz(grid.lat[oi], grid.lon[oj])
    prof_xyz = _xyz(prof["lat"], prof["lon"])
    tree = cKDTree(prof_xyz)
    dist, nn = tree.query(cell_xyz, k=1)                            # (n_ocean,)
    s["ocean_ij"] = (oi, oj)
    s["nn"] = nn
    s["nn_dist"] = dist
    # observed-column mask: True where an ocean cell is *unobserved* (no profile
    # column there).  Broadcast across depth this excludes whole observed columns
    # from the loss and the metric (Week-1 unobserved-only fix).
    obs_col = np.zeros((grid.nlat, grid.nlon), dtype=bool)
    pi, pj = prof["ij"][:, 0], prof["ij"][:, 1]
    obs_col[pi, pj] = True
    s["unobs_mask"] = grid.ocean & (~obs_col)      # (H,W) bool
    # nearest-filled fields
    near = {}
    for v in VARS:
        g = np.full((grid.ndepth, grid.nlat, grid.nlon), np.nan, "float32")
        g[:, oi, oj] = prof[v][nn].T                                # (D, n_ocean)
        near[v] = g
    s["near"] = near
    return s


# ==========================================================================
# Method: climatology  (pred = WOA prior)
# ==========================================================================
def predict_climatology(sample):
    return {v: sample["woa"][v].copy() for v in VARS}


# ==========================================================================
# Method: climatology floor  (pred = train-only CESM2 monthly climatology)
# ==========================================================================
def predict_clim_floor(sample, clim, grid):
    """Predict the train-only CESM2 monthly climatology (= zero anomaly).

    This is the principled RMSE floor: any skill must beat *its own* held-out
    climatology, not the bias-dominated WOA prior.
    """
    mo = sample["month"]
    out = {}
    for v in VARS:
        arr = clim.clim3d(v, mo).astype("float32")
        out[v] = np.where(grid.ocean[None], arr, np.nan)
    return out


# ==========================================================================
# Method: nearest  (horizontal nearest-profile fill)
# ==========================================================================
def predict_nearest(sample, use_woa: bool):
    """If use_woa: interpolate the obs-minus-WOA residual and add WOA back
    (so far-from-obs cells fall back to climatology).  Else: raw nearest fill."""
    out = {}
    oi, oj = sample["ocean_ij"]
    for v in VARS:
        if use_woa:
            out[v] = sample["woa"][v].copy()
            # residual nearest fill at profile columns blended by distance
            res = sample["near"][v] - sample["woa"][v]
            # weight residual by exp(-d/scale): near obs -> trust profile, far -> WOA
            d = sample["nn_dist"]                       # chord distance on unit sphere
            w = np.exp(-(d / 0.05) ** 2)                # ~3 deg e-folding
            blended = sample["woa"][v].copy()
            blended[:, oi, oj] = (sample["woa"][v][:, oi, oj]
                                  + w[None] * res[:, oi, oj])
            out[v] = blended
        else:
            out[v] = sample["near"][v].copy()
    return out


# ==========================================================================
# Pointwise MLP
# ==========================================================================
class MLP(nn.Module):
    def __init__(self, c_in, hidden, c_out=2):
        super().__init__()
        layers = []
        d = c_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        layers += [nn.Linear(d, c_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _point_features(sample, grid, norm, cfg, idx_i, idx_j, depth_idx):
    """Build pointwise feature matrix for given (i,j,depth) sample points.

    Returns (X, has_dims). Geo/time always present; prior/nearest/surf gated by cfg.
    """
    lat = grid.lat[idx_i].astype("float32")
    lon = grid.lon[idx_j].astype("float32")
    dep = grid.depth[depth_idx].astype("float32")
    mo = sample["month"]
    cols = [
        (lat - grid.lat.mean()) / (grid.lat.std() + 1e-6),
        np.sin(np.deg2rad(lon)), np.cos(np.deg2rad(lon)),
        (dep - grid.depth.mean()) / (grid.depth.std() + 1e-6),
        np.full_like(lat, np.sin(2 * np.pi * mo / 12)),
        np.full_like(lat, np.cos(2 * np.pi * mo / 12)),
    ]
    if "woa" in cfg:
        for v in VARS:
            z = norm.z3d(v, sample["woa"][v], mo)[depth_idx, idx_i, idx_j]
            cols.append(np.nan_to_num(z, nan=0.0))
    if "profiles" in cfg:
        for v in VARS:
            z = norm.z3d(v, sample["near"][v], mo)[depth_idx, idx_i, idx_j]
            cols.append(np.nan_to_num(z, nan=0.0))
        # distance-to-nearest feature (per (i,j)); map ocean cells
        dmap = np.full((grid.nlat, grid.nlon), 1.0, "float32")
        oi, oj = sample["ocean_ij"]
        dmap[oi, oj] = sample["nn_dist"]
        cols.append(dmap[idx_i, idx_j])
    if "surf" in cfg:
        for sv in C.VARS_SURF:
            arr = sample["surf"].get(sv)
            if arr is None:
                cols.append(np.zeros_like(lat))
            else:
                z = norm.zsurf(sv, arr, mo)[idx_i, idx_j]
                cols.append(np.nan_to_num(z, nan=0.0))
    return np.stack(cols, axis=1).astype("float32")


def train_predict_mlp(train_samples, test_samples, grid, norm, cfg, rng, device):
    # ---- assemble training points ----
    Xtr, Ytr = [], []
    for s in train_samples:
        oi, oj = s["ocean_ij"]
        n = oi.size
        for _ in range(1):  # one depth-stratified draw per month
            take = min(C.MLP_POINTS_PER_MONTH, n * grid.ndepth)
            di = rng.integers(0, grid.ndepth, size=take)
            ci = rng.integers(0, n, size=take)
            ii, jj = oi[ci], oj[ci]
            X = _point_features(s, grid, norm, cfg, ii, jj, di)
            y = np.stack([norm.z3d(v, s["gt"][v], s["month"])[di, ii, jj]
                          for v in VARS], 1)
            m = np.isfinite(y).all(1)
            Xtr.append(X[m]); Ytr.append(y[m].astype("float32"))
    Xtr = np.concatenate(Xtr); Ytr = np.concatenate(Ytr)

    model = MLP(Xtr.shape[1], C.MLP_HIDDEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.MLP_LR)
    lossf = nn.MSELoss()
    Xt = torch.from_numpy(Xtr).to(device); Yt = torch.from_numpy(Ytr).to(device)
    nb = int(np.ceil(len(Xt) / C.MLP_BATCH))
    for ep in range(C.MLP_EPOCHS):
        perm = torch.randperm(len(Xt), device=device)
        for b in range(nb):
            sl = perm[b * C.MLP_BATCH:(b + 1) * C.MLP_BATCH]
            opt.zero_grad()
            loss = lossf(model(Xt[sl]), Yt[sl])
            loss.backward(); opt.step()

    # ---- predict full test fields ----
    preds = []
    model.eval()
    with torch.no_grad():
        for s in test_samples:
            oi, oj = s["ocean_ij"]
            pv = {v: np.full((grid.ndepth, grid.nlat, grid.nlon), np.nan, "float32")
                  for v in VARS}
            for d in range(grid.ndepth):
                X = _point_features(s, grid, norm, cfg, oi, oj,
                                    np.full(oi.size, d))
                yz = model(torch.from_numpy(X).to(device)).cpu().numpy()
                for k, v in enumerate(VARS):
                    pv[v][d, oi, oj] = yz[:, k]
            for v in VARS:
                pv[v] = norm.unz3d(v, pv[v], s["month"])
            preds.append(pv)
    return preds


# ==========================================================================
# Depthwise 2D U-Net
# ==========================================================================
def _unet_channels(sample, grid, norm, cfg):
    """Return input tensor (D, C_in, H, W) and the channel list for a sample."""
    D, H, W = grid.ndepth, grid.nlat, grid.nlon
    ocean = grid.ocean[None].astype("float32")        # (1,H,W)
    mo = sample["month"]
    chans = []

    def z_or_zero(arr3d, v):
        z = norm.z3d(v, arr3d, mo)
        return np.nan_to_num(z, nan=0.0).astype("float32")

    if "profiles" in cfg:
        for v in VARS:
            obs = sample["obs"][v]                     # (D,H,W) sparse
            chans.append(z_or_zero(obs, v))            # value
            chans.append((np.isfinite(obs)).astype("float32"))  # mask
    if "woa" in cfg:
        for v in VARS:
            chans.append(z_or_zero(sample["woa"][v], v))
    # depth positional channel (broadcast)
    dnorm = (grid.depth - grid.depth.mean()) / (grid.depth.std() + 1e-6)
    chans.append(np.broadcast_to(dnorm[:, None, None], (D, H, W)).astype("float32"))
    if "surf" in cfg:
        for sv in C.VARS_SURF:
            arr = sample["surf"].get(sv)
            if arr is None:
                chans.append(np.zeros((D, H, W), "float32"))
            else:
                z = np.nan_to_num(norm.zsurf(sv, arr, mo), nan=0.0).astype("float32")
                chans.append(np.broadcast_to(z[None], (D, H, W)))
    # ocean mask channel
    chans.append(np.broadcast_to(ocean, (D, H, W)).astype("float32"))
    X = np.stack(chans, axis=1)                        # (D, C_in, H, W)
    return X


def train_predict_unet(train_samples, test_samples, grid, norm, cfg, device,
                       unobs_loss=False):
    """Depthwise 2D U-Net: one shared 2D net applied per depth slice.

    ``unobs_loss``: if True the training loss is restricted to *unobserved*
    ocean cells (profile columns excluded) so the model is scored on its
    interpolation skill rather than on copying the obs it is fed.
    """
    D, H, W = grid.ndepth, grid.nlat, grid.nlon
    ocean_t = torch.from_numpy(grid.ocean.astype("float32")).to(device)

    # precompute tensors
    def targets(s):
        return np.stack([np.nan_to_num(norm.z3d(v, s["gt"][v], s["month"]), nan=0.0)
                         for v in VARS], axis=1)        # (D,2,H,W)

    Xtr = [(_unet_channels(s, grid, norm, cfg), targets(s)) for s in train_samples]
    c_in = Xtr[0][0].shape[1]
    model = UNet2D(c_in, len(VARS), base=C.UNET_BASE).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.UNET_LR)

    # flatten to slices
    Xall = np.concatenate([x for x, _ in Xtr], axis=0)       # (N*D, C, H, W)
    Yall = np.concatenate([y for _, y in Xtr], axis=0)       # (N*D, 2, H, W)
    Xall_t = torch.from_numpy(Xall).to(device)
    Yall_t = torch.from_numpy(Yall).to(device)
    # per-slice spatial loss weight (unobserved ocean, or all ocean)
    if unobs_loss:
        wm = np.stack([np.repeat(s["unobs_mask"].astype("float32")[None], D, axis=0)
                       for s in train_samples], axis=0).reshape(-1, H, W)  # (N*D,H,W)
        Wall_t = torch.from_numpy(wm).to(device)
    else:
        Wall_t = ocean_t[None].expand(Xall_t.shape[0], H, W)
    N = Xall_t.shape[0]
    nb = int(np.ceil(N / C.UNET_BATCH))
    for ep in range(C.UNET_EPOCHS):
        perm = torch.randperm(N, device=device)
        for b in range(nb):
            sl = perm[b * C.UNET_BATCH:(b + 1) * C.UNET_BATCH]
            opt.zero_grad()
            out = model(Xall_t[sl])
            w = Wall_t[sl][:, None]                          # (b,1,H,W)
            loss = (((out - Yall_t[sl]) ** 2) * w).sum() / (w.sum() * len(VARS) + 1e-8)
            loss.backward(); opt.step()

    preds = []
    model.eval()
    with torch.no_grad():
        for s in test_samples:
            X = torch.from_numpy(_unet_channels(s, grid, norm, cfg)).to(device)
            out = model(X).cpu().numpy()                # (D,2,H,W) z-scored
            pv = {}
            for k, v in enumerate(VARS):
                arr = norm.unz3d(v, out[:, k], s["month"])
                arr = np.where(grid.ocean[None], arr, np.nan)
                pv[v] = arr.astype("float32")
            preds.append(pv)
    return preds


# ==========================================================================
# Joint-depth 2D U-Net  (the Week-1 strong baseline)
# ==========================================================================
# The depthwise U-Net above reconstructs each depth level independently (a
# shared 2D net applied per slice), so it cannot exploit vertical correlation.
# The joint-depth U-Net stacks *all depths as input/output channels* and
# predicts the whole water column at once (2 vars x D levels = 40 output
# channels at D=20), letting the convolutions model the full T/S structure
# jointly.  This is the stronger control the shared-latent method must beat.
def _unet_channels_joint(sample, grid, norm, cfg):
    """Whole-column input tensor (C_in, H, W); depth encoded by channel identity."""
    D, H, W = grid.ndepth, grid.nlat, grid.nlon
    mo = sample["month"]
    chans = []                                          # each (H,W) or stacked

    def zc(arr3d, v):                                   # (D,H,W) -> D channels
        z = np.nan_to_num(norm.z3d(v, arr3d, mo), nan=0.0).astype("float32")
        return list(z)                                  # D x (H,W)

    if "profiles" in cfg:
        for v in VARS:
            obs = sample["obs"][v]                       # (D,H,W) sparse
            chans += zc(obs, v)                          # value, D channels
            chans += list(np.isfinite(obs).astype("float32"))  # mask, D channels
    if "woa" in cfg:
        for v in VARS:
            chans += zc(sample["woa"][v], v)
    if "surf" in cfg:
        for sv in C.VARS_SURF:
            arr = sample["surf"].get(sv)
            if arr is None:
                chans.append(np.zeros((H, W), "float32"))
            else:
                chans.append(np.nan_to_num(norm.zsurf(sv, arr, mo),
                                           nan=0.0).astype("float32"))
    chans.append(grid.ocean.astype("float32"))          # ocean mask, 1 channel
    return np.stack(chans, axis=0)                       # (C_in, H, W)


def train_predict_unet_joint(train_samples, test_samples, grid, norm, cfg, device,
                             unobs_loss=True):
    """Joint-depth U-Net: predicts (2*D) channels per month in one forward pass."""
    D, H, W = grid.ndepth, grid.nlat, grid.nlon
    Cout = len(VARS) * D

    def targets(s):                                      # (2*D, H, W)
        t = np.stack([np.nan_to_num(norm.z3d(v, s["gt"][v], s["month"]), nan=0.0)
                      for v in VARS], axis=0)            # (2, D, H, W)
        return t.reshape(Cout, H, W)

    Xtr = np.stack([_unet_channels_joint(s, grid, norm, cfg) for s in train_samples], 0)
    Ytr = np.stack([targets(s) for s in train_samples], 0)      # (N, 2D, H, W)
    c_in = Xtr.shape[1]
    model = UNet2D(c_in, Cout, base=C.UNET_JOINT_BASE).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.UNET_JOINT_LR)

    Xt = torch.from_numpy(Xtr).to(device)
    Yt = torch.from_numpy(Ytr).to(device)
    if unobs_loss:
        wm = np.stack([s["unobs_mask"].astype("float32") for s in train_samples], 0)
    else:
        wm = np.broadcast_to(grid.ocean.astype("float32"),
                             (len(train_samples), H, W)).copy()
    Wt = torch.from_numpy(wm).to(device)                # (N,H,W)
    N = Xt.shape[0]
    nb = int(np.ceil(N / C.UNET_JOINT_BATCH))
    for ep in range(C.UNET_JOINT_EPOCHS):
        perm = torch.randperm(N, device=device)
        for b in range(nb):
            sl = perm[b * C.UNET_JOINT_BATCH:(b + 1) * C.UNET_JOINT_BATCH]
            opt.zero_grad()
            out = model(Xt[sl])                          # (b, 2D, H, W)
            w = Wt[sl][:, None]                          # (b,1,H,W)
            loss = (((out - Yt[sl]) ** 2) * w).sum() / (w.sum() * Cout + 1e-8)
            loss.backward(); opt.step()

    preds = []
    model.eval()
    with torch.no_grad():
        for s in test_samples:
            X = torch.from_numpy(_unet_channels_joint(s, grid, norm, cfg)[None]).to(device)
            out = model(X).cpu().numpy()[0].reshape(len(VARS), D, H, W)
            pv = {}
            for k, v in enumerate(VARS):
                arr = norm.unz3d(v, out[k], s["month"])
                pv[v] = np.where(grid.ocean[None], arr, np.nan).astype("float32")
            preds.append(pv)
    return preds
