"""Scenes for training on a window of up to 40 years of the 1964-2005 cohort.

Three things are combined, each as its own module defines it:

* the matched synthetic scenes and their frozen OI anchor
  (``synthetic_latent_experiment``), on a named cohort with its own fixed
  baselines (``radius_scale_scenes.use_cohort_reference``);
* the complete neighbour profiles the T/S alignment of ``innovation_ocean``
  reads. ``innovation_scenes.InnovationOceanScenes`` adds them from
  ``cesm2_uniform.nc`` by name; here the same tensors come from the cohort in
  use;
* the local OI-error scale at the query (``radius_scale_scenes``, by query):
  the anchor and the training target are divided by it, and the scale is
  handed to the model as ``local_query_scale`` so that the neighbour
  innovations are divided after the alignment
  (``query_scaled_innovation``).

``train_years`` sets the training window: the last ``train_years`` years
before 2004. Only the months a training step draws from change. The per-depth
normalisation, the climatology, the OI anchor and its settings, the local
scale table and the 2004 validation and 2005 test cells stay the 2000-2003
fits of the audit's split, so the four-year window is the earlier setup
exactly.

No neighbour radius: the 32 nearest input profiles are read at any range.
Surface inputs reach the model as satellite tokens and query features; the
observation-operator tensors ``InnovationOceanScenes`` adds with them are read
only by the operator recipes and are not built.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from .audit_tools import cohort_path
from .point_baselines import CH
from .radius_scale_scenes import RadiusScaleScenes
from .synthetic_latent_experiment import SyntheticOceanScenes

#: the audit's training years end in 2003; month indices count from 2000-01
LAST_TRAINING_MONTH = 47


def window_months(available, train_years):
    """Months of the last ``train_years`` years up to 2003-12, sorted."""
    first = LAST_TRAINING_MONTH + 1 - 12 * int(train_years)
    return [m for m in sorted(available) if first <= m <= LAST_TRAINING_MONTH]


class LongRecordScenes(RadiusScaleScenes):

    def __init__(self, root, device="cpu", *, train_years=4, **kwargs):
        if kwargs.get("radius_km") is not None:
            raise ValueError("these scenes take no neighbour radius")
        if int(train_years) < 1:
            raise ValueError("the training window needs at least one year")
        self.train_years = int(train_years)
        super().__init__(root, device, **kwargs)
        # InnovationOceanScenes.__init__, with this cohort's stored climatology
        path = Path(cohort_path(str(self.root), "synthetic", cohort=kwargs.get("cohort")))
        with xr.open_dataset(path) as raw:
            order = np.argsort(np.asarray(raw.month_index.values), kind="stable")
            climate = np.stack([np.asarray(raw[f"CLIM_POS_{ch}"].values)[order] for ch in CH], -1)
        self.profile_climate = torch.as_tensor(climate, dtype=torch.float32, device=self.device)
        self.profile_mean = torch.as_tensor(np.stack([self.norm.mean[ch] for ch in CH], -1),
                                            dtype=torch.float32, device=self.device)
        self.profile_std = torch.as_tensor(np.stack([self.norm.std[ch] for ch in CH], -1),
                                           dtype=torch.float32, device=self.device)
        self.profile_physical = self.y * self.profile_std + self.profile_mean
        self.profile_absolute = self.profile_physical + self.profile_climate
        months = self.months("train")
        if len(months) != 12 * self.train_years:
            raise ValueError(f"a {self.train_years}-year window needs {12 * self.train_years} months "
                             f"with inputs and queries; the cohort has {len(months)}")
        self.metadata.update(
            full_profile_context="source-only absolute T/S plus stored training-year climatology at source coordinates",
            profile_climate_sha256=hashlib.sha256(climate.tobytes()).hexdigest(),
            train_years=self.train_years, training_months=[months[0], months[-1]],
            training_window="last train_years years before 2004; normalisation, climatology, OI and scale "
                            "tables stay fitted on 2000-2003")

    def months(self, split):
        if split != "train":
            return super().months(split)
        if "train" not in self._months_cache:
            self._months_cache["train"] = [
                m for m in window_months(self.months_available, self.train_years)
                if self.c.month(m, float_split="cohort_float").size
                and self.c.month(m, float_split="heldout_float").size]
        return self._months_cache["train"]

    def configure(self, radius_km=None, scale=None, scale_by="query"):
        if radius_km is not None or (scale is not None and scale_by != "query"):
            raise ValueError("these scenes take a query scale and no radius")
        super().configure(radius_km, scale, scale_by)

    def scene(self, source, query_rows, query_levels, analysis, context_profiles=6080,
              context_seed=0, geometry=None, cached_baseline=None):
        if geometry is None:
            geometry = self.geometry(source, query_rows, 32, query_levels)
        # the matched scene itself, without radius_scale_scenes' division of the innovations
        result = SyntheticOceanScenes.scene(self, source, query_rows, query_levels, analysis,
                                            context_profiles, context_seed, geometry, cached_baseline)
        neighbors = geometry["neighbors"]
        result.update(local_profile_values=self.profile_physical[neighbors],
                      local_profile_valid=self.valid[neighbors], local_profile_ids=neighbors,
                      local_profile_absolute=self.profile_absolute[neighbors],
                      local_profile_climatology=self.profile_climate[neighbors],
                      profile_depths=self.depth, profile_normalization_mean=self.profile_mean,
                      profile_normalization_std=self.profile_std, query_level=self.indices(query_levels))
        if self.scale is not None:
            at_query = self.scale[self.indices(query_rows), self.indices(query_levels)]
            result["baseline"] = result["baseline"] / at_query
            result["local_query_scale"] = at_query
        return result
