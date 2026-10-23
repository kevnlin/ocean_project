"""Forecast scenes of the gridded model on the real 2016-2023 Argo cohort.

The same scene keys, units and feature layout as ``grid_forecast_scenes.GridForecastScenes`` on
CESM2, so one model reads both:

* profiles: ``real_global.load_real`` (WOA23 anomaly at the profile's position, z-scored with the
  2016-2020 statistics, values beyond 10 standard deviations missing); no first guess and no
  reported errors are read (zeros, as the synthetic scenes);
* features, lagged features and dense surface grids: ``105_real_surface.py`` (OISST, OISSS L4 and
  DUACS SLA in the layout of the CESM2 caches);
* frozen OI: the selection of ``106_real_oi_baseline.py``; it is the persisted-OI reference the
  runner scores, the gridded model never reads it;
* real profile days (offsets in days are real, not mid-month).

Differences from the CESM2 scenes: monthly counts vary (no input parity), the splits are years
(train 2016-2020, validation 2021, development 2022-2023), and the training months are the
training split's, not a window before 2004.

``years=(2016, 2024)`` adds the 2024 test year (its rows are outside every split, so the
normalisation, the training months and every 2016-2023 row are unchanged); its caches are built on
that cohort, and ``oi_cohort_fingerprint`` names the 2016-2023 cohort whose frozen OI selection it
reuses, so no OI is fitted or scored on 2024 before the test.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .forecast_scenes import LEADS, ForecastMonth
from .grid_forecast_scenes import GridForecastScenes
from .latent_experiment import array_fingerprint
from .oi import _lonlat_to_xyz
from .point_baselines import CH
from .real_global import REAL_SPLITS, YEARS, cohort_fingerprint, load_real, raw_profiles, real_eval_month


class RealGridForecastScenes(GridForecastScenes):

    def __init__(self, root, device="cpu", *, caches, oi_summary, leads=LEADS, train_years=None,
                 years=YEARS, oi_cohort_fingerprint=None):
        self.root, self.device = Path(root), torch.device(device)
        years = tuple(int(y) for y in years)
        if years[0] != YEARS[0] or years[1] < YEARS[1] or years[1] > 2024:
            raise ValueError("the real cohort runs from 2016 to 2023 or 2024; 2025 is never read")
        c, norm, obs = load_real(self.root, years=years)
        self.c, self.norm, self.obs = c, norm, obs
        self.levels = np.asarray(c.levels)
        self.n_levels = len(self.levels)
        y = np.stack([obs[ch] for ch in CH], axis=-1)
        fingerprint = cohort_fingerprint(c, obs)
        caches = Path(caches)

        def cache(name):
            with np.load(caches / name, allow_pickle=False) as z:
                data = {k: z[k] for k in z.files}
            meta = json.loads(str(data.pop("metadata").item()))
            if meta["cohort_fingerprint"] != fingerprint:
                raise ValueError(f"{name} was built on another cohort or normalisation")
            for k in ("month_index", "lat", "lon"):
                if k in data and not np.array_equal(data[k], getattr(c, k)):
                    raise ValueError(f"{name}: {k} ordering differs from the cohort")
            return data, meta, hashlib.sha256((caches / name).read_bytes()).hexdigest()

        satellites, satellite_meta, satellites_sha = cache("satellites.npz")
        lagged, _, lagged_sha = cache("lagged_surface.npz")
        grids, _, grids_sha = cache("surface_grids.npz")
        summary_path = Path(oi_summary)
        summary = json.loads(summary_path.read_text())
        if (summary["target"] != "anomaly_exact" or summary["splits"] != {k: list(v) for k, v in REAL_SPLITS.items()}
                or summary["cohort"]["fingerprint"] != (oi_cohort_fingerprint or fingerprint)):
            raise ValueError("the OI baselines were computed on another cohort, target or split")
        self.oi_selection = summary["oi_selection"]
        self.baseline_reference = summary["baselines"]["oi"]["scores"]
        self.climatology_reference = summary["baselines"]["climatology"]["scores"]

        def tensor(a, dtype=torch.float32):
            return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=self.device)

        self.x = tensor(satellites["features"])
        self.y = tensor(y)
        self.valid_np = np.isfinite(y)
        self.valid = torch.isfinite(self.y)
        self.fg = torch.zeros_like(self.y)
        self.innovation = torch.where(self.valid, self.y, 0.)
        self.error = torch.zeros_like(self.y)
        self.error_valid = torch.zeros_like(self.y)
        self.lat, self.lon = tensor(c.lat), tensor(c.lon % 360.)
        self.xyz_np = _lonlat_to_xyz(c.lat, c.lon)
        self.xyz = tensor(self.xyz_np, torch.float64)
        temp, salt, day = raw_profiles(self.root, years)
        self.day, self.depth = tensor(day), tensor(c.levels)
        mean = np.stack([norm.mean[ch] for ch in CH], -1)
        std = np.stack([norm.std[ch] for ch in CH], -1)
        physical = y * std + mean
        climate = np.nan_to_num(np.stack([temp, salt], -1) - physical)
        self.profile_mean, self.profile_std = tensor(mean), tensor(std)
        self.profile_physical = tensor(physical)
        self.profile_climate = tensor(climate)
        self.profile_absolute = self.profile_physical + self.profile_climate
        self.surface, self.scale, self.scale_by, self.radius_km = True, None, None, None
        self.lagged = tensor(lagged["features"])
        self.lag_index = {int(lead): i for i, lead in enumerate(lagged["leads"])}
        self.leads = tuple(sorted({int(lead) for lead in leads}))
        if set(self.leads) - {0} - set(self.lag_index):
            raise ValueError(f"the lagged features hold leads {sorted(self.lag_index)}, not {list(self.leads)}")
        self.grid_lat, self.grid_lon = grids["grid_lat"], grids["grid_lon"]
        self.attach_surface_grids(torch.as_tensor(grids["grids"], device=self.device), grids["grid_month"])
        self.months_available = set(int(m) for m in np.unique(c.month_index))
        self._months_cache, self._training_pools = {}, {}
        self.train_years = None if train_years is None else int(train_years)
        self.fg_checkpoint = {"method": "zero standardized-anomaly background; no satellite first guess"}
        self.metadata = {
            "cohort_fingerprint": fingerprint, "cohort": f"real_global {years[0]}-{years[1]}",
            "feature_fingerprint": satellite_meta["feature_fingerprint"],
            "scene_fingerprint": array_fingerprint((day, c.float_split, c.year_split)),
            "splits": {k: list(v) for k, v in REAL_SPLITS.items()}, "n_profiles": int(len(c.lat)),
            "n_levels": self.n_levels, "levels_m": self.levels.tolist(),
            "anomaly_target": "WOA23 at the profile position", "normalization": "training years only",
            "qc": "values beyond 10 training standard deviations are missing",
            "task": "inputs of month t, held-out floats of month t + lead",
            "satellite": satellite_meta.get("source"), "satellite_sha256": satellites_sha,
            "lagged_surface_sha256": lagged_sha, "surface_grids_sha256": grids_sha,
            "oi_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "oi_selection": self.oi_selection, "forecast_leads_months": list(self.leads),
            "evidence_status": "2022-2023 development years; 2025 sealed holdout never read",
            "local_radius_km": None, "local_scale_sha256": None, "local_scale_by": None}

    def months(self, split):
        """Months of a split with inputs and queries; with ``train_years`` the training months are
        the last that many years of the training split (2020, 2019-2020, ...)."""
        if split not in self._months_cache:
            first = -np.inf
            if split == "train" and getattr(self, "train_years", None):
                first = REAL_SPLITS["train"][1] - self.train_years + 1
            self._months_cache[split] = [
                int(m) for m in self.c.months_in(split) if int(m) in self.months_available
                and 2000 + int(m) // 12 >= first
                and self.c.month(int(m), float_split="cohort_float").size
                and self.c.month(int(m), float_split="heldout_float").size]
        return self._months_cache[split]

    def split_of_year(self, year):
        return next((name for name, (lo, hi) in REAL_SPLITS.items() if lo <= year <= hi), None)

    def forecast_months(self, years, max_cells=8000):
        """Every target month of ``years`` at every lead whose input month is in the record; all
        leads of a month score the same cells (``real_global.real_eval_month``)."""
        out = []
        for month in sorted(self.months_available):
            if 2000 + month // 12 not in set(years) or not self.c.month(month, float_split="heldout_float").size:
                continue
            for lead in self.leads:
                if month - lead not in self.months_available:
                    continue
                ev = real_eval_month(self.c, self.obs, month, max_cells, lead)
                if ev["src"].size:
                    out.append(ForecastMonth(month, lead, ev["src"], ev["tgt"], ev["prof"], ev["lev"]))
        return out
