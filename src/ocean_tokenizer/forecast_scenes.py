"""Forecast scenes: the observations of month t, scored at month t + lead.

``ForecastScenes`` is ``LongRecordScenes`` (the 1964-2005 cohort, a training window, the
complete neighbour profiles of the T/S alignment, the local OI-error scale) with a lead in
months:

* A training draw picks a lead from ``leads`` and a target month m of the window whose input
  month m - lead is in the window too. The inputs are month m - lead's context floats, drawn as
  the parent draws them (70 % of the month's input floats); the targets are cells of month m's
  input profiles. At lead 0 the draw is the parent's.
* A scene's lead is the month of its queries minus the month of its inputs. Everything the model
  reads about the observations (tokens, the 32 nearest profiles and their full columns, the frozen
  OI computed from them) is month t's, and the third coordinate of every local offset, the time
  from a neighbour to the query, is the lead: that is how one model tells the leads apart.
* A query's own surface features (columns 44-50) are replaced by month t's at the query position
  (``93_lagged_surface.py``): a forecast cannot read the target month's satellites.
* ``forecast_months`` builds evaluation cells: for target month m and each lead, month m - lead's
  6,080 inputs and month m's 1,520 queries with the cell draw of ``synth_argo_eval.eval_month``
  for m, so every lead is scored on the same cells and lead 0 is the usual evaluation.

The frozen OI from month t, read at month t + lead, is persisted OI: the baseline at every lead.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import synth_argo_eval as S
from .long_record_scenes import LongRecordScenes

LEADS = (0, 1, 3, 6)
#: the surface columns of a profile's 56 features: SST, SSS and SSH z-scores with validity, then SST
SURFACE_COLUMNS = slice(44, 51)


@dataclass
class ForecastMonth:
    month: int
    lead: int
    source: np.ndarray
    target: np.ndarray
    profile: np.ndarray
    level: np.ndarray


class ForecastScenes(LongRecordScenes):

    def __init__(self, root, device="cpu", *, leads=LEADS, lagged_surface=None, **kwargs):
        leads = tuple(sorted({int(lead) for lead in leads}))
        if not leads or leads[0] < 0:
            raise ValueError("leads are whole months, 0 or more")
        self.leads = leads
        super().__init__(root, device, **kwargs)
        self.lagged, self.lag_index = None, {}
        if self.surface and leads[-1] > 0:
            if lagged_surface is None:
                raise ValueError("a forecast with surface inputs needs the lagged surface cache")
            path = Path(lagged_surface)
            with np.load(path, allow_pickle=False) as cache:
                metadata = json.loads(str(cache["metadata"].item()))
                if metadata["cohort_fingerprint"] != self.metadata["cohort_fingerprint"]:
                    raise ValueError("the lagged surface cache was built on another cohort")
                for name in ("month_index", "lat", "lon"):
                    if not np.array_equal(cache[name], getattr(self.c, name)):
                        raise ValueError(f"lagged surface {name} ordering differs from the cohort")
                cached = [int(v) for v in cache["leads"]]
                if set(leads) - {0} - set(cached):
                    raise ValueError(f"the lagged surface cache holds leads {cached}, not {list(leads)}")
                self.lagged = torch.as_tensor(cache["features"], dtype=torch.float32, device=self.device)
            self.lag_index = {lead: i for i, lead in enumerate(cached)}
            self.metadata["lagged_surface_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.metadata.update(forecast_leads_months=list(leads),
                             forecast="inputs of month t, queries and targets of month t + lead; query surface "
                                      "features of month t; persisted frozen OI as the anchor")

    def lead_of(self, source, query_rows):
        """The months from a scene's inputs to its queries; each must come from one month."""
        months = self.c.month_index
        source_month = np.unique(months[np.asarray(source)])
        query_month = np.unique(months[np.asarray(query_rows)])
        if len(source_month) != 1 or len(query_month) != 1:
            raise ValueError("a forecast scene's inputs and its queries each come from one month")
        lead = int(query_month[0] - source_month[0])
        if lead < 0:
            raise ValueError("queries precede their inputs")
        return lead

    def _pool(self, month):
        if month not in self._training_pools:
            pool = self.c.month(month, float_split="cohort_float")
            floats, ids = np.unique(self.c.wmo[pool], return_inverse=True)
            self._training_pools[month] = pool, len(floats), ids
        return self._training_pools[month]

    def training_draw(self, rng, n_queries):
        lead = int(rng.choice(self.leads))
        window = self.months("train")
        inside = set(window)
        month = int(rng.choice([m for m in window if m - lead in inside]))
        pool, n_floats, ids = self._pool(month - lead)
        held = rng.choice(n_floats, max(1, min(n_floats - 1, int(round(.3 * n_floats)))), replace=False)
        held_mask = np.zeros(n_floats, dtype=bool)
        held_mask[held] = True
        target_mask = held_mask[ids]
        source = pool[~target_mask]
        # at lead 0 a training target float is never a context float; at a lead the targets are
        # another month's profiles
        target = pool[target_mask] if lead == 0 else self._pool(month)[0]
        draw = rng.choice(target.size * self.n_levels, n_queries,
                          replace=target.size * self.n_levels < n_queries)
        return month, source, target[draw // self.n_levels], draw % self.n_levels

    def scene(self, source, query_rows, query_levels, analysis, context_profiles=6080,
              context_seed=0, geometry=None, cached_baseline=None):
        lead = self.lead_of(source, query_rows)
        result = super().scene(source, query_rows, query_levels, analysis, context_profiles,
                               context_seed, geometry, cached_baseline)
        if lead > 0 and self.surface:
            features = result["query_features"].clone()
            features[:, SURFACE_COLUMNS] = self.lagged[self.indices(query_rows), self.lag_index[lead]]
            result["query_features"] = features
        return result

    def forecast_months(self, years, max_cells=8000):
        """Evaluation cells for every target month of ``years`` and every lead."""
        out = []
        for month in sorted(self.months_available):
            if 2000 + month // 12 not in set(years):
                continue
            cells = S.eval_month(self.c, self.obs, month, max_cells)
            for lead in self.leads:
                source = self.c.month(month - lead, float_split="cohort_float")
                if source.size != S.N_INPUT:
                    raise ValueError(f"month {month - lead} has {source.size} inputs, not {S.N_INPUT}")
                out.append(ForecastMonth(month, lead, source, cells["tgt"], cells["prof"], cells["lev"]))
        return out
