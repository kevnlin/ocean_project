"""Matched real-Argo scenes for the ocean latent architecture campaign.

The satellite first guess is the saved, year-cross-fitted anchor. Context and
targets are disjoint by float identity. Each observation token is one actual
depth, with separate temperature/salinity validity; no depth-band averaging.
This is retrospective same-month reconstruction, not causal forecasting.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from . import anchored as A
from .argo_obs import ArgoNorm
from .audit_tools import cohort_path, load_cohort
from .point_baselines import CH, Scores

SPLITS = {"train": (2016, 2020), "validation": (2021, 2021),
          "development": (2022, 2023)}
EVAL_SEED = 20260918


def satellite_features(c, sat):
    """Exactly the 56-feature recipe used by the saved daily first guess."""
    la, lo = np.deg2rad(c.lat), np.deg2rad(c.lon % 360.0)
    mon = c.month_index % 12 + 1
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    cols = [c.lat / 90.0, xyz]
    for k in range(6):
        cols += [np.sin(2 ** k * np.pi * xyz), np.cos(2 ** k * np.pi * xyz)]
    for h in (1, 2):
        cols += [np.sin(2 * np.pi * h * mon / 12.0), np.cos(2 * np.pi * h * mon / 12.0)]
    for name, scale in (("SLA_1deg_anom", .065), ("SST_1deg_anom", .65),
                        ("SSS_1deg_anom", .19)):
        v = sat[name].astype(float)
        cols += [np.nan_to_num(v / scale), np.isfinite(v).astype(float)]
    cols += [np.nan_to_num((sat["SST_1deg"].astype(float) - 14.) / 11.)]
    dc, de, dw, dn, ds = (sat[f"SLA_day_{k}"].astype(float) for k in "cewns")
    ok = np.isfinite(dc) & np.isfinite(sat["SLA_clim_1deg"])
    cols += [np.where(ok, (dc - sat["SLA_clim_1deg"]) / .09, 0.),
             np.where(ok & np.isfinite(sat["SLA_1deg"]), (dc - sat["SLA_1deg"]) / .05, 0.),
             np.nan_to_num((de - dw) / .05), np.nan_to_num((dn - ds) / .05),
             ok.astype(float)]
    return np.column_stack(cols)


def array_fingerprint(arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


@dataclass
class EvalMonth:
    month: int
    source: np.ndarray
    target: np.ndarray
    profile: np.ndarray
    level: np.ndarray


class OceanScenes:
    """Device-resident prepared data; scene construction never reads target values."""

    def __init__(self, root, device="cpu", anchor_tag="anc_satday_dfs", *, use_prepared_cache=True):
        self.root, self.device = Path(root), torch.device(device)
        if use_prepared_cache:
            from .latent_scene_cache import load_scene_cache
            if load_scene_cache(self, anchor_tag):
                return
        c, _ = load_cohort(str(root), "global", SPLITS, anomaly="cell")
        self.c = c
        self.norm = norm = ArgoNorm.fit(c, "train")
        self.levels = np.asarray(c.levels)
        self.n_levels = l = len(self.levels)
        y = np.concatenate([norm.z(ch, getattr(c, ch)) for ch in CH], axis=1)
        fingerprint = array_fingerprint((c.month_index, c.lat, c.lon, c.wmo, c.levels, y))
        anchor = self.root / "outputs/audit/global" / anchor_tag
        fg_checkpoint = torch.load(anchor / "first_guess_seed1234.pt", map_location="cpu", weights_only=True)
        fg_cache = torch.load(anchor / "first_guess_predictions_seed1234.pt", map_location="cpu", weights_only=True)
        if fingerprint != fg_checkpoint["cohort_fingerprint"] or fingerprint != fg_cache["cohort_fingerprint"]:
            raise ValueError("first-guess cache is not the loaded cohort/normalization")
        if fg_cache["training_predictions"] != "cross-fit by year":
            raise ValueError("training innovations require year-cross-fitted first guesses")
        sat = np.load(self.root / "data/argo_cohort/global_satellite.npz")
        if not np.array_equal(sat["month_index"], c.month_index):
            raise ValueError("satellite profile ordering mismatch")
        x = satellite_features(c, sat)
        if hashlib.sha256(memoryview(np.ascontiguousarray(x)).cast("B")).hexdigest() != fg_checkpoint["feature_fingerprint"]:
            raise ValueError("satellite features differ from the saved first guess")
        mu, sd = fg_checkpoint["feature_mean"].numpy(), fg_checkpoint["feature_std"].numpy()
        def tensor(a, dtype=torch.float32):
            if isinstance(a, np.ndarray) and not a.flags.writeable:
                a = a.copy()
            return torch.as_tensor(a, dtype=dtype, device=self.device)
        self.x = tensor((x - mu) / sd)
        self.y = tensor(y.reshape(-1, 2, l).transpose(0, 2, 1))
        self.fg = fg_cache["predictions"].reshape(-1, 2, l).transpose(1, 2).to(self.device)
        self.valid = torch.isfinite(self.y) & (self.y.abs() <= 10.)
        self.innovation = torch.where(self.valid, self.y - self.fg, torch.zeros_like(self.y))
        err = np.stack([getattr(c, f"{ch}_ERR") / norm.std[ch] for ch in CH], axis=-1)
        self.error = tensor(np.log1p(np.clip(np.nan_to_num(err, nan=0., posinf=100.), 0., 100.)))
        self.error_valid = tensor(np.isfinite(err) & (err >= 0.))
        self.lat, self.lon = tensor(c.lat), tensor(c.lon % 360.)
        with xr.open_dataset(cohort_path(str(root), "global")) as raw:
            order = np.argsort(np.asarray(raw.month_index.values, int), kind="stable")
            day = (np.asarray(raw["time"].values)[order].astype("datetime64[D]") - np.datetime64("2000-01-01")).astype(float)
        self.day = tensor(day)
        self.depth = tensor(c.levels)
        sla = sat["SLA_day_c"].astype(float) - sat["SLA_clim_1deg"].astype(float)
        self.state = tensor(np.nan_to_num(sla / .09))
        self.state_ok = tensor(np.isfinite(sla), torch.bool)
        self.months_available = set(int(m) for m in np.unique(c.month_index[np.isfinite(sat["SLA_1deg"])]))
        self.metadata = {"cohort_fingerprint": fingerprint, "feature_fingerprint": fg_checkpoint["feature_fingerprint"],
                         "anchor": str(anchor), "splits": SPLITS, "n_profiles": int(len(c.lat)),
                         "n_levels": l, "context_input_qc_z": 10., "task": "retrospective same-month reconstruction",
                         "evidence_status": "development; 2022-2023 has been used in earlier development",
                         "satellite_resolution": "1deg monthly SST/SSS/SLA and query-date daily SLA",
                         "normalization": "training years only", "training_first_guess": "cross-fit by year",
                         "observation_features": "same 56 satellite/position features + 2 first guesses + 2 reported error features + 2 error validity flags"}
        self.metadata["scene_fingerprint"] = array_fingerprint((day, c.TEMP_ERR, c.SALT_ERR,
            c.float_split, c.year_split, fg_cache["predictions"].numpy()))
        self.metadata["normalization_scope"] = "all training-year profiles, including heldout WMOs; inherited matched protocol"
        self.metadata["baseline_scope"] = "saved DFS-run first-guess cache plus saved kriging analysis; recombined matched anchor"
        self.fg_checkpoint = fg_checkpoint
        self.reference = json.loads((self.root / "outputs/audit/global/setconv_surface/summary_seed1234.json").read_text())
        self._months_cache = {}
        self._training_pools = {}

    def months(self, split):
        if split in self._months_cache:
            return self._months_cache[split]
        result = [int(m) for m in self.c.months_in(split) if int(m) in self.months_available
                and self.c.month(int(m), float_split="cohort_float").size
                and self.c.month(int(m), float_split="heldout_float").size]
        self._months_cache[split] = result
        return result

    def eval_months(self, split, max_cells=8000):
        out = []
        for m in self.months(split):
            src = self.c.month(m, float_split="cohort_float")
            tgt = self.c.month(m, float_split="heldout_float")
            prof = np.repeat(np.arange(tgt.size), self.n_levels)
            lev = np.tile(np.arange(self.n_levels), tgt.size)
            if max_cells and prof.size > max_cells:
                pick = np.random.default_rng([EVAL_SEED, m]).choice(prof.size, max_cells, replace=False)
                prof, lev = prof[pick], lev[pick]
            if np.intersect1d(self.c.wmo[src], self.c.wmo[tgt]).size:
                raise ValueError("input and heldout float identities overlap")
            out.append(EvalMonth(m, src, tgt, prof, lev))
        return out

    def identity_check(self, split, evs):
        s = Scores(self.levels, self.norm.std)
        for ev in evs:
            target = self.y[self.indices(ev.target[ev.profile]), self.indices(ev.level)].cpu().numpy()
            for j, ch in enumerate(CH):
                s.add(ch, np.zeros(len(ev.level)), target[:, j], ev.level)
        scores = s.result()
        for ch in CH:
            expected = self.reference["scores"][split][ch]
            got = scores[ch]
            if got["n"] != expected["n"] or abs(got["climatology_z"] / expected["climatology_z"] - 1.) > 1e-5:
                raise ValueError(f"evaluation fingerprint mismatch: {split} {ch}")
        return {ch: {k: scores[ch][k] for k in ("n", "climatology_z")} for ch in CH}

    def indices(self, rows):
        return torch.as_tensor(rows, dtype=torch.long, device=self.device)

    def training_draw(self, rng, n_queries):
        m = int(rng.choice(self.months("train")))
        if m not in self._training_pools:
            pool = self.c.month(m, float_split="cohort_float")
            floats, ids = np.unique(self.c.wmo[pool], return_inverse=True)
            self._training_pools[m] = pool, len(floats), ids
        pool, n_floats, ids = self._training_pools[m]
        held = rng.choice(n_floats, max(1, min(n_floats-1, int(round(.3 * n_floats)))), replace=False)
        held_mask = np.zeros(n_floats, dtype=bool)
        held_mask[held] = True
        target_mask = held_mask[ids]
        src, tgt = pool[~target_mask], pool[target_mask]
        # A training target float is never also a context float.
        draw = rng.choice(tgt.size * self.n_levels, n_queries,
                          replace=tgt.size * self.n_levels < n_queries)
        return m, src, tgt[draw // self.n_levels], draw % self.n_levels

    def context_rows(self, source, budget, seed):
        if budget <= 0 or len(source) <= budget:
            return np.asarray(source)
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        # Sampling is only for the latent branch. The exact local path always
        # retrieves from the complete, identical source set.
        return np.sort(rng.choice(source, budget, replace=False))

    def geometry(self, source, query_rows, k):
        src = self.indices(source)
        unique, inverse = np.unique(query_rows, return_inverse=True)
        rows = self.indices(unique)
        nb = A.neighbours(self.lat[rows], self.lon[rows], self.lat[src], self.lon[src], k)
        inv = self.indices(inverse)
        q = self.indices(query_rows)
        neigh = src[nb.idx[inv]]
        qstate = torch.where(self.state_ok[q], self.state[q], torch.zeros_like(self.state[q]))
        offsets = torch.stack([nb.dx[inv], nb.dy[inv], self.day[neigh]-self.day[q, None],
                               self.state[neigh]-qstate[:, None]], dim=-1)
        return {"neighbors": neigh, "offsets": offsets}

    def scene(self, source, query_rows, query_levels, analysis, context_profiles=512,
              context_seed=0, geometry=None, cached_baseline=None):
        q, lev = self.indices(query_rows), self.indices(query_levels)
        if geometry is None:
            geometry = self.geometry(source, query_rows, analysis.k)
        neighbors, offsets = geometry["neighbors"], geometry["offsets"]
        local_lev = lev[:, None].expand_as(neighbors)
        innovation = self.innovation[neighbors, local_lev]
        valid = self.valid[neighbors, local_lev]
        bg = self.fg[q, lev]
        field = torch.stack([lev, lev+self.n_levels], dim=-1)
        if cached_baseline is None:
            scales = lambda name: getattr(analysis, name)[field].exp()
            weights = A.kriging_weights(offsets[..., 0][:, None], offsets[..., 1][:, None],
                        valid.transpose(1, 2), scales("log_ell_x"), scales("log_ell_y"), scales("log_gamma"),
                        dt=offsets[..., 2][:, None], ell_t=scales("log_ell_t"),
                        ds=offsets[..., 3][:, None], ell_s=scales("log_ell_s"))
            baseline = bg + (weights * innovation.transpose(1, 2)).sum(-1)
        else:
            baseline = cached_baseline
        ctx = self.indices(self.context_rows(source, context_profiles, context_seed))
        n, l = len(ctx), self.n_levels
        coord = torch.stack([self.lat[ctx, None].expand(n, l), self.lon[ctx, None].expand(n, l),
                             self.depth[None].expand(n, l), self.day[ctx, None].expand(n, l)], -1).reshape(-1, 4)
        obs_features = torch.cat([self.x[ctx, None].expand(n, l, -1), self.fg[ctx], self.error[ctx], self.error_valid[ctx]], -1).reshape(-1, 62)
        local_features = torch.cat([self.x[neighbors], self.fg[neighbors, local_lev], self.error[neighbors, local_lev],
                                    self.error_valid[neighbors, local_lev],
                                    offsets / offsets.new_tensor([500., 500., 30., 1.])], -1)
        qcoord = torch.stack([self.lat[q], self.lon[q], self.depth[lev], self.day[q]], -1)
        satcoord = torch.stack([self.lat[ctx], self.lon[ctx], torch.zeros_like(self.lat[ctx]), self.day[ctx]], -1)
        return {"obs_features": obs_features, "obs_coord": coord,
                "obs_innovation": self.innovation[ctx].reshape(-1, 2), "obs_valid": self.valid[ctx].reshape(-1, 2),
                "satellite_features": self.x[ctx], "satellite_coord": satcoord,
                "query_features": torch.cat([self.x[q], bg], -1), "query_coord": qcoord,
                "query_background": bg, "baseline": baseline,
                "local_features": local_features, "local_offsets": offsets,
                "local_innovation": innovation, "local_valid": valid}

    def targets(self, rows, levels):
        q, lev = self.indices(rows), self.indices(levels)
        return self.y[q, lev], self.valid[q, lev]

    def analysis(self):
        m = A.InnovationAnalysis(2*self.n_levels, mode="kriging", k=32, use_time=True, use_state=True).to(self.device)
        path = self.root / "outputs/audit/global/anc_satday_kriging/model_seed1234.pt"
        m.load_state_dict(torch.load(path, map_location=self.device, weights_only=True), strict=True)
        return m


def masked_losses(output, target, valid, nll_weight=0.):
    """Channel-balanced masked MSE and Gaussian NLL, with explicit validity."""
    valid = valid & torch.isfinite(target)
    clean = torch.where(valid, target, torch.zeros_like(target))
    error = output["mean"] - clean
    counts_raw = valid.sum(0)
    active = counts_raw > 0
    counts = counts_raw.clamp_min(1)
    mse = (((error.square() * valid).sum(0) / counts) * active).sum() / active.sum().clamp_min(1)
    std = output["std"].clamp_min(.03)
    terms = .5*(error/std).square() + std.log() + .5*np.log(2*np.pi)
    nll = (((terms * valid).sum(0) / counts) * active).sum() / active.sum().clamp_min(1)
    return mse + nll_weight*nll + .001*output.get("aux_loss", mse.new_zeros(())), {"mse": mse, "nll": nll}
