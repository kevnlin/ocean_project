"""Forecast scenes for the gridded state model (``grid_kriging_ocean``).

``forecast_scenes.ForecastScenes`` plus what a gridded state reads:

* ``surface_grid [3,H,W]``: the dense SST/SSS/steric-SSH z-scores of the input month
  (``100_surface_grids.py``; NaN over land), never of the query month;
* ``lead`` and ``input_month``: the scene's lead in months and the month its inputs come from;
* ``local_lat`` / ``local_lon``: positions of the 32 neighbour profiles, so the model can read
  its state there.

Everything else is the forecast scene unchanged, including the frozen OI anchor, which the
gridded model does not read (the runner still scores it as the persisted-OI reference).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .forecast_scenes import ForecastScenes


class GridForecastScenes(ForecastScenes):

    def __init__(self, root, device="cpu", *, surface_grids, **kwargs):
        super().__init__(root, device, **kwargs)
        path = Path(surface_grids)
        with np.load(path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadata"].item()))
            if metadata["cohort_fingerprint"] != self.metadata["cohort_fingerprint"]:
                raise ValueError("the surface grids were built on another cohort")
            grids, months = cache["grids"], cache["grid_month"]
            self.grid_lat, self.grid_lon = cache["grid_lat"], cache["grid_lon"]
        self.attach_surface_grids(torch.as_tensor(grids, device=self.device), months)
        self.metadata["surface_grids_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def attach_surface_grids(self, grids, months):
        """``grids [months, 3, H, W]`` (float16 is enough), ``months`` their month indices."""
        months = np.asarray(months)
        if len(np.unique(months)) != len(months) or len(months) != len(grids):
            raise ValueError("one surface grid per month, each month once")
        self.surface_grids = grids
        self.grid_index = {int(m): i for i, m in enumerate(months)}

    def scene(self, source, query_rows, query_levels, analysis, context_profiles=6080,
              context_seed=0, geometry=None, cached_baseline=None):
        result = super().scene(source, query_rows, query_levels, analysis, context_profiles,
                               context_seed, geometry, cached_baseline)
        month = int(np.unique(self.c.month_index[np.asarray(source)])[0])
        neighbors = result["local_profile_ids"]
        result.update(surface_grid=self.surface_grids[self.grid_index[month]].float(),
                      lead=torch.tensor(self.lead_of(source, query_rows)), input_month=torch.tensor(month),
                      local_lat=self.lat[neighbors], local_lon=self.lon[neighbors])
        return result
