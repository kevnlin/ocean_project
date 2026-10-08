"""CESM2 scenes matched to the earlier synthetic Argo reconstruction audit.

The OI anchor is the spherical, finite-observation, per-variable/depth-band
analysis in ``44_synth_argo_oi.py``.  It is deliberately distinct from the real
data experiment's learned anisotropic OI.  All source observations remain
available to OI and the numerical local path, independently of latent budget.
Synthetic measurements are noise free; finite extreme values are not clipped.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from . import anchored as A, synth_argo_eval as S
from .latent_experiment import EvalMonth, OceanScenes, array_fingerprint
from .oi import R_EARTH_KM, _lonlat_to_xyz
from .point_baselines import CH, Scores, band_of_levels

EVAL_SEED = S.EVAL_SEED


def position_features(c):
    """44 position/season fields plus 12 reserved zero satellite fields."""
    xyz = _lonlat_to_xyz(c.lat, c.lon)
    month = c.month_index % 12 + 1
    cols = [np.asarray(c.lat) / 90., xyz]
    for k in range(6):
        cols += [np.sin(2**k * np.pi * xyz), np.cos(2**k * np.pi * xyz)]
    for h in (1, 2):
        cols += [np.sin(2*np.pi*h*month/12), np.cos(2*np.pi*h*month/12)]
    position = np.column_stack(cols)
    assert position.shape[1] == 44
    return np.pad(position, ((0, 0), (0, 12)))


class SphericalOI(torch.nn.Module):
    """Frozen exact 44 OI covariance, including its selected neighbour counts.

    Padding rows are an identity system, so missing observations cannot change
    the valid-observation solve.  Distances and the solve remain float64 on
    CPU and CUDA.  Only the returned neural-network anchor is cast to float32.
    """

    def __init__(self, levels, selection):
        super().__init__()
        bands = band_of_levels(levels)
        ell = [[selection[ch][b]["L_km"] for b in bands] for ch in CH]
        gamma = [[selection[ch][b]["gamma"] for b in bands] for ch in CH]
        counts = [[selection[ch][b]["k"] for b in bands] for ch in CH]
        self.register_buffer("ell", torch.tensor(ell, dtype=torch.float64))
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float64))
        self.register_buffer("counts", torch.tensor(counts, dtype=torch.long))
        self.k = int(self.counts.max())

    def forward(self, geometry, levels, innovation, xyz):
        idx, supplied_valid = geometry["oi_neighbors"], geometry["oi_valid"]
        q, channels, k = idx.shape
        if q == 0:
            return innovation.new_empty((0, 2))
        field = torch.arange(channels, device=idx.device)[None, :, None]
        lev = levels[:, None, None]
        value = innovation[idx, lev, field].to(torch.float64)
        rank = torch.arange(k, device=idx.device)[None, None]
        valid = supplied_valid & (rank < self.counts[:, levels].T[..., None])
        valid = valid & torch.isfinite(value)
        v = valid.to(torch.float64)
        neighbor_xyz = xyz[idx]
        d_oo = 2*R_EARTH_KM*torch.asin((torch.linalg.vector_norm(
            neighbor_xyz[..., :, None, :] - neighbor_xyz[..., None, :, :], dim=-1)/2).clamp(0, 1))
        ell = self.ell[:, levels].T[..., None]
        covariance = torch.exp(-.5*(d_oo/ell[..., None]).square()) * v[..., :, None] * v[..., None, :]
        covariance = covariance + torch.diag_embed(self.gamma[:, levels].T[..., None]*v + (1-v))
        qcov = torch.exp(-.5*(geometry["oi_distance"]/ell).square())*v
        weight = torch.linalg.solve(covariance, qcov[..., None]).squeeze(-1)*v
        clean = torch.where(valid, value, 0.)
        return (weight*clean).sum(-1).to(innovation.dtype)


class SyntheticOceanScenes(OceanScenes):
    """Same train/evaluation masks and anomaly target as the older 64-slot model."""

    def __init__(self, root, device="cpu", *, surface=False, satellite_cache=None,
                 cohort=None, oi_summary=None):
        self.root, self.device = Path(root), torch.device(device)
        self.c, self.norm, obs = S.load(str(self.root), cohort=cohort)
        c = self.c
        self.levels = np.asarray(c.levels)
        self.n_levels = len(self.levels)
        self.obs = obs
        y = np.stack([obs[ch] for ch in CH], axis=-1)
        self.xyz_np = _lonlat_to_xyz(c.lat, c.lon)
        fingerprint = array_fingerprint((c.month_index, c.lat, c.lon, c.wmo,
                                         c.levels, y))
        self.surface = bool(surface)
        x = position_features(c)
        satellite_metadata = None
        if surface:
            if satellite_cache is None:
                raise ValueError("--surface requires an explicit, matched --satellite-cache")
            with np.load(satellite_cache, allow_pickle=False) as cache:
                for name in ("month_index", "lat", "lon"):
                    if not np.array_equal(cache[name], getattr(c, name)):
                        raise ValueError(f"satellite cache {name} ordering differs from cohort")
                x = cache["features"].copy()
                satellite_metadata = json.loads(str(cache["metadata"].item()))
            if x.shape != (len(c.lat), 56) or not np.isfinite(x).all():
                raise ValueError("satellite features must be finite [n_profiles,56]")
            if satellite_metadata.get("cohort_fingerprint") != fingerprint:
                raise ValueError("satellite cache cohort/anomaly/normalization fingerprint differs")
            if satellite_metadata.get("normalization_scope") != "training years only":
                raise ValueError("satellite normalization must use training years only")
        elif satellite_cache is not None:
            raise ValueError("satellite cache supplied without --surface")
        summary_path = Path(oi_summary) if oi_summary else self.root/"outputs/audit/synthetic/fixed_baselines/summary.json"
        baseline_summary = json.loads(summary_path.read_text())
        if baseline_summary["target"] != "anomaly_exact" or baseline_summary["splits"] != {k:list(v) for k,v in S.SPLITS.items()}:
            raise ValueError("OI baseline target or year splits differ")
        self.oi_selection = baseline_summary["oi_selection"]
        self.baseline_reference = baseline_summary["baselines"]["oi"]["scores"]
        self.valid_np = np.isfinite(y)
        def tensor(a, dtype=torch.float32):
            a = np.asarray(a)
            if not a.flags.writeable:
                a = a.copy()
            return torch.as_tensor(a, dtype=dtype, device=self.device)
        self.x = tensor(x)
        self.y = tensor(y)
        self.valid = torch.isfinite(self.y)
        self.fg = torch.zeros_like(self.y)
        self.innovation = torch.where(self.valid, self.y, 0.)
        self.error = torch.zeros_like(self.y)
        self.error_valid = torch.zeros_like(self.y)
        self.lat, self.lon = tensor(c.lat), tensor(c.lon % 360.)
        self.xyz = tensor(self.xyz_np, torch.float64)
        # The synthetic cohort has monthly timestamps, not invented daily dates.
        dates = np.array([np.datetime64(f"{2000+int(m)//12:04d}-{int(m)%12+1:02d}-15")
                          for m in c.month_index], dtype="datetime64[D]")
        day = (dates-np.datetime64("2000-01-01")).astype(float)
        self.day, self.depth = tensor(day), tensor(c.levels)
        self.state = torch.zeros(len(c.lat), device=self.device)
        self.state_ok = torch.zeros(len(c.lat), dtype=torch.bool, device=self.device)
        self.months_available = set(int(m) for m in np.unique(c.month_index))
        self._months_cache, self._training_pools = {}, {}
        self.fg_checkpoint = {"method": "zero standardized-anomaly background; no satellite first guess"}
        self.metadata = {
            "cohort_fingerprint": fingerprint,
            "feature_fingerprint": array_fingerprint((x,)),
            "scene_fingerprint": array_fingerprint((day, c.float_split, c.year_split)),
            "splits": S.SPLITS, "n_profiles": len(c.lat), "n_levels": self.n_levels,
            "levels_m": self.levels.tolist(), "anomaly_target": "at Argo profile position",
            "normalization": "training years only", "normalization_scope": "all training-year profiles; inherited old synthetic protocol",
            "inputs_per_evaluation_month": S.N_INPUT, "query_profiles_per_evaluation_month": S.N_QUERY,
            "validation_cell_cap": S.EVAL_CELLS, "context_input_qc_z": None,
            "task": "retrospective same-month reconstruction", "satellite": surface,
            "satellite_metadata": satellite_metadata,
            "feature_recipe": "44 positional/season fields, 12 reserved surface fields; no surface measurements in Argo-only arm",
            "observation_features": "56 positional/surface + 2 zero first guesses + 2 zero error fields + 2 missing-error flags",
            "baseline_scope": "frozen exact spherical 44 OI, per variable/depth-band finite-neighbor settings",
            "oi_selection": self.oi_selection,
            "oi_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "evidence_status": "2005 previously evaluated synthetic test; not new independent confirmation",
            "latent_budget_scope": "latent branch only; OI and local retrieval use complete source pool",
            "surface_sampling": "at source/query profile locations; not dense raster tokens" if surface else None,
        }

    def eval_months(self, split, max_cells=8000):
        result = []
        for ev in S.eval_set(self.c, self.obs, split, max_cells):
            if np.intersect1d(self.c.wmo[ev["src"]], self.c.wmo[ev["tgt"]]).size:
                raise ValueError("input and heldout float identities overlap")
            result.append(EvalMonth(ev["month"], ev["src"], ev["tgt"], ev["prof"], ev["lev"]))
        return result

    def identity_check(self, split, evs):
        scores = Scores(self.levels, self.norm.std)
        for ev in evs:
            for ch in CH:
                target = self.obs[ch][ev.target][ev.profile, ev.level]
                scores.add(ch, np.zeros(len(ev.level)), target, ev.level)
        return S.identity_check(str(self.root), split, scores.result())

    def geometry(self, source, query_rows, k=32, query_levels=None):
        source, query_rows = np.asarray(source), np.asarray(query_rows)
        if np.intersect1d(source, query_rows).size:
            raise ValueError("query measurements overlap the source pool")
        if query_levels is None:
            raise ValueError("exact per-depth OI geometry requires query_levels")
        levels = np.asarray(query_levels, dtype=int)
        src, q = self.indices(source), self.indices(query_rows)
        if self.device.type == "cuda":
            # One geometry matrix for all requested depths; finite selection is
            # performed independently for temperature and salinity below.
            chord = torch.cdist(self.xyz[q], self.xyz[src], compute_mode="donot_use_mm_for_euclid_dist")
            local_idx = chord.topk(min(k, len(source)), largest=False).indices
            neighbors = src[local_idx]
            oi_idx, oi_valid, oi_distance = [], [], []
            kmax = min(max(int(s["k"]) for v in self.oi_selection.values() for s in v.values()), len(source))
            li = self.indices(levels)
            for channel in range(2):
                finite = self.valid[src[:, None], li[None, :], channel].T
                distance, index = chord.masked_fill(~finite, float("inf")).topk(kmax, largest=False)
                oi_idx.append(src[index]); oi_valid.append(torch.isfinite(distance))
                oi_distance.append(2*R_EARTH_KM*torch.asin((distance/2).clamp(0, 1)))
            result = {"oi_neighbors": torch.stack(oi_idx, 1),
                      "oi_valid": torch.stack(oi_valid, 1),
                      "oi_distance": torch.stack(oi_distance, 1)}
        else:
            _, index = cKDTree(self.xyz_np[source]).query(self.xyz_np[query_rows], k=min(k, len(source)))
            neighbors = self.indices(source[np.asarray(index).reshape(len(query_rows), -1)])
            kmax = min(max(int(s["k"]) for v in self.oi_selection.values() for s in v.values()), len(source))
            idx = np.zeros((len(query_rows), 2, kmax), dtype=np.int64)
            valid = np.zeros(idx.shape, dtype=bool)
            distance = np.zeros(idx.shape, dtype=np.float64)
            for level in np.unique(levels):
                rows = np.flatnonzero(levels == level)
                for channel in range(2):
                    finite = source[self.valid_np[source, level, channel]]
                    if not len(finite):
                        continue
                    count = min(kmax, len(finite))
                    d, i = cKDTree(self.xyz_np[finite]).query(self.xyz_np[query_rows[rows]], k=count)
                    d, i = np.asarray(d).reshape(len(rows), count), np.asarray(i).reshape(len(rows), count)
                    idx[rows, channel, :count] = finite[i]
                    distance[rows, channel, :count] = 2*R_EARTH_KM*np.arcsin(np.clip(d/2, 0, 1))
                    valid[rows, channel, :count] = True
            result = {"oi_neighbors": self.indices(idx), "oi_valid": torch.as_tensor(valid, device=self.device),
                      "oi_distance": torch.as_tensor(distance, dtype=torch.float64, device=self.device)}
        local = A._offsets(self.lat[q], self.lon[q], self.lat, self.lon, neighbors)
        offsets = torch.stack([local.dx, local.dy, self.day[neighbors]-self.day[q, None],
                               torch.zeros_like(local.dx)], -1)
        result.update(neighbors=neighbors, offsets=offsets)
        return result

    def scene(self, source, query_rows, query_levels, analysis, context_profiles=256,
              context_seed=0, geometry=None, cached_baseline=None):
        if geometry is None:
            geometry = self.geometry(source, query_rows, 32, query_levels)
        levels = self.indices(query_levels)
        baseline = (analysis(geometry, levels, self.innovation, self.xyz)
                    if cached_baseline is None else cached_baseline)
        result = super().scene(source, query_rows, query_levels, analysis, context_profiles,
                               context_seed, geometry, cached_baseline=baseline)
        if not self.surface:
            result.pop("satellite_features")
            result.pop("satellite_coord")
        return result

    def analysis(self):
        return SphericalOI(self.levels, self.oi_selection).to(self.device)

    def baseline_identity_check(self, split, scores):
        for ch in CH:
            reference = self.baseline_reference[split][ch]
            if scores[ch]["n"] != reference["n"] or not np.isclose(
                    scores[ch]["rmse_physical"], reference["rmse_physical"], rtol=2e-5, atol=1e-8):
                raise ValueError(f"OI replay differs from 44 baseline: {split} {ch}")
        return {ch: {"rmse_physical": scores[ch]["rmse_physical"],
                     "reference_rmse_physical": self.baseline_reference[split][ch]["rmse_physical"]} for ch in CH}
