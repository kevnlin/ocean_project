"""Shared data plumbing for the Week-4 full-scale runs.

One class, ``FullRunData``, owns the GPU-resident z-space tensors and the
observation/query assembly used by BOTH the trainer
(experiments/18_full_train.py) and the post-training evaluation
(experiments/19_full_eval.py), so the two can never drift apart.  All values
are anomaly z-scores from a train-only ``AnomNorm`` (protocol_v1); NaN is
preserved end-to-end (encoders turn it into finite-flags).

Verified equal to the POC-verified ``token_api.sample_to_obs`` path
(profiles / surf / woa tensors and query targets, maxdiff 0.0).
"""
from __future__ import annotations
import numpy as np
import torch

from . import config as C

VARS = ("TEMP", "SALT")
BANDS = [("0-100m", 0.0, 100.0), ("100-300m", 100.0, 300.0),
         ("300-max", 300.0, 1e9)]


class FullRunData:
    """GPU tensors + assembly helpers for one (grid, norm, device) context."""

    def __init__(self, grid, norm, device):
        self.grid, self.norm, self.dev = grid, norm, device
        self.D, self.H, self.W = grid.ndepth, grid.nlat, grid.nlon
        self.HW = self.H * self.W
        self.oi, self.oj = np.where(grid.ocean)
        self.n_ocean = self.oi.size
        self.lat_t = torch.tensor(grid.lat, dtype=torch.float32, device=device)
        self.lon_t = torch.tensor(grid.lon, dtype=torch.float32, device=device)
        self.depth_t = torch.tensor(grid.depth, dtype=torch.float32,
                                    device=device)
        self.astd_t = {v: torch.tensor(norm.astd[v], dtype=torch.float32,
                                       device=device) for v in VARS}
        self.woaZ = None

    # ---------------- z-space tensor builders ----------------
    def z_volume(self, fields) -> np.ndarray:
        """(T, 2, D, H, W) float32 anomaly z-scores (NaN preserved)."""
        T = len(fields["months"])
        out = np.empty((T, 2, self.D, self.H, self.W), "float32")
        for t in range(T):
            mo = int(fields["months"][t])
            for k, v in enumerate(VARS):
                out[t, k] = self.norm.z3d(v, fields[v][t], mo)
        return out

    def z_surf(self, fields) -> np.ndarray:
        """(T, 2, H, W) float32 surface anomaly z-scores (NaN preserved)."""
        T = len(fields["months"])
        out = np.empty((T, 2, self.H, self.W), "float32")
        for t in range(T):
            mo = int(fields["months"][t])
            for k, sv in enumerate(C.VARS_SURF):
                out[t, k] = self.norm.zsurf(sv, fields[sv][t], mo)
        return out

    def load_woa(self, woa) -> torch.Tensor:
        """(12, 2, D, H, W) z-scored WOA prior, cached on the device."""
        arr = np.empty((12, 2, self.D, self.H, self.W), "float32")
        for mo in range(1, 13):
            for k, v in enumerate(VARS):
                arr[mo - 1, k] = self.norm.z3d(v, woa[v][mo - 1], mo)
        self.woaZ = torch.from_numpy(arr).to(self.dev)
        return self.woaZ

    # ---------------- observation / query assembly ----------------
    def obs_dict(self, ZAvol, surfZ, t, mo, prof_ii, prof_jj,
                 include=("profiles", "surf", "woa")) -> dict:
        """Observation dict for one month; prof_ii/jj are (K,) GPU int64."""
        dev = self.dev
        obs = {}
        if "profiles" in include:
            K = int(prof_ii.numel())
            if K:
                vals = ZAvol[t][:, :, prof_ii, prof_jj].permute(2, 0, 1)
            else:
                vals = torch.zeros(0, 2, self.D, device=dev)
            obs["profiles"] = dict(
                prof=vals[None], lat=self.lat_t[prof_ii][None],
                lon=self.lon_t[prof_jj][None],
                month=torch.tensor([mo], device=dev))
        if "surf" in include:
            obs["surf"] = dict(field=surfZ[t][None], lat=self.lat_t,
                               lon=self.lon_t,
                               month=torch.tensor([mo], device=dev))
        if "woa" in include:
            assert self.woaZ is not None, "call load_woa first"
            obs["woa"] = dict(field=self.woaZ[mo - 1][None], lat=self.lat_t,
                              lon=self.lon_t,
                              month=torch.tensor([mo], device=dev),
                              depth=self.depth_t)
        return obs

    def q_from_flat(self, idx_t, mo):
        """flat (D*HW) indices -> ((1,Q,4) query coords, (Q,) depth idx)."""
        di = idx_t // self.HW
        rem = idx_t % self.HW
        ii = rem // self.W
        jj = rem % self.W
        q = torch.stack([self.lat_t[ii], self.lon_t[jj], self.depth_t[di],
                         torch.full((idx_t.numel(),), float(mo),
                                    device=self.dev)], -1)
        return q[None], di

    def make_packs(self, fields, rng, n_profiles,
                   include=("profiles", "surf", "woa"), quiet=False,
                   subsample=0):
        """Fixed per-month eval packs: profile draw, obs dict, and the FULL
        unobserved-only query pool (q, y, depth idx).  Returns
        (packs, per-level query counts, per-level climatology-floor SSE)."""
        dev = self.dev
        D, HW, W = self.D, self.HW, self.W
        ZAv = torch.from_numpy(self.z_volume(fields)).to(dev)
        surfZ = torch.from_numpy(self.z_surf(fields)).to(dev)
        packs = []
        for t, mo in enumerate(fields["months"]):
            mo = int(mo)
            k = min(n_profiles, self.n_ocean)
            if k:
                pick = rng.choice(self.n_ocean, size=k, replace=False)
                ii_t = torch.from_numpy(self.oi[pick]).to(dev)
                jj_t = torch.from_numpy(self.oj[pick]).to(dev)
            else:
                ii_t = torch.zeros(0, dtype=torch.long, device=dev)
                jj_t = torch.zeros(0, dtype=torch.long, device=dev)
            col = torch.zeros(HW, dtype=torch.bool, device=dev)
            col[ii_t * W + jj_t] = True
            fin = torch.isfinite(ZAv[t]).all(0).view(D, HW)
            keep = fin & (~col)[None]
            idx = keep.view(-1).nonzero(as_tuple=True)[0]
            if subsample and idx.numel() > subsample:
                sel = rng.choice(idx.numel(), size=subsample, replace=False)
                idx = idx[torch.from_numpy(np.sort(sel)).to(dev)]
            y = ZAv[t].view(2, -1)[:, idx].T.contiguous()
            q, di = self.q_from_flat(idx, mo)
            packs.append(dict(obs=self.obs_dict(ZAv, surfZ, t, mo, ii_t, jj_t,
                                                include=include),
                              q=q, y=y, di=di, mo=mo, t=t))
        n_level = torch.zeros(D, dtype=torch.float64, device=dev)
        se0 = torch.zeros(D, 2, dtype=torch.float64, device=dev)
        for p in packs:
            n_level += torch.bincount(p["di"], minlength=D).double()
            se0.index_add_(0, p["di"], (p["y"].double() ** 2))
        del ZAv
        if not quiet:
            print(f"  packs: {len(packs)} months, "
                  f"{int(n_level.sum()):,} pooled queries", flush=True)
        return packs, n_level, se0

    # ---------------- metric ----------------
    def physical_rmse(self, se, n_level) -> dict:
        """Per-level z-space SSE (D,2) -> physical full/band/level RMSE."""
        out = {"full": {}, "by_band": {}, "by_depth": {}}
        dnp = self.grid.depth
        for k, v in enumerate(VARS):
            se_phys = se[:, k] * (self.astd_t[v].double() ** 2)
            out["full"][v] = float(torch.sqrt(se_phys.sum() / n_level.sum()))
            out["by_depth"][v] = torch.sqrt(
                se_phys / n_level.clamp(min=1)).cpu().numpy().tolist()
            out["by_band"][v] = {}
            for name, lo, hi in BANDS:
                sel = (dnp > lo) & (dnp <= hi)
                if lo <= dnp.min():
                    sel |= np.isclose(dnp, dnp.min())
                st = torch.from_numpy(sel).to(self.dev)
                out["by_band"][v][name] = float(torch.sqrt(
                    se_phys[st].sum() / n_level[st].sum().clamp(min=1)))
        return out

    @torch.no_grad()
    def eval_packs(self, model, packs, chunk=131072, obs_override=None):
        """Unobserved-only z-space SSE per level; fuse once per month.

        ``obs_override``: optional callable pack -> obs dict (e.g. modality
        dropping or token manipulations); default uses pack['obs'].
        """
        model.eval()
        se = torch.zeros(self.D, 2, dtype=torch.float64, device=self.dev)
        for p in packs:
            obs = p["obs"] if obs_override is None else obs_override(p)
            z = model.fuse(model.encode(obs, batch=1, device=self.dev))
            Q = p["q"].shape[1]
            for i in range(0, Q, chunk):
                out = model.decode(z, p["q"][:, i:i + chunk])[0]
                err2 = (out - p["y"][i:i + chunk]).double() ** 2
                se.index_add_(0, p["di"][i:i + chunk], err2)
        model.train()
        return se
