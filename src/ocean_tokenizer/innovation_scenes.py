"""Full source profiles and train-climatology operators for isolated trials."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from .synthetic_latent_experiment import SyntheticOceanScenes
from .point_baselines import CH
from .ocean_observation_operator import build_surface_observation_operator


class InnovationOceanScenes(SyntheticOceanScenes):
    def __init__(self, *args, operator_cache=None, **kwargs):
        super().__init__(*args, **kwargs)
        with xr.open_dataset(self.root / "data/synthetic_argo/cesm2_uniform.nc") as raw:
            order = np.argsort(np.asarray(raw.month_index.values), kind="stable")
            climate = np.stack([np.asarray(raw[f"CLIM_POS_{ch}"].values)[order] for ch in CH], -1)
        self.profile_climate = torch.as_tensor(climate, dtype=torch.float32, device=self.device)
        self.profile_mean = torch.as_tensor(np.stack([self.norm.mean[ch] for ch in CH], -1), dtype=torch.float32, device=self.device)
        self.profile_std = torch.as_tensor(np.stack([self.norm.std[ch] for ch in CH], -1), dtype=torch.float32, device=self.device)
        self.profile_physical = self.y * self.profile_std + self.profile_mean
        self.profile_absolute = self.profile_physical + self.profile_climate
        self.operator_h = self.operator_offset = self.operator_valid = None
        if self.surface and operator_cache is not None:
            path = Path(operator_cache)
            with np.load(path, allow_pickle=False) as z:
                metadata = json.loads(str(z["metadata"].item()))
                if metadata["cohort_fingerprint"] != self.metadata["cohort_fingerprint"]:
                    raise ValueError("Operator cache cohort mismatch")
                def tensor(key, dtype=torch.float32):
                    return torch.as_tensor(z[key], dtype=dtype, device=self.device)
                self.operator_h = tensor("operator")
                self.operator_offset = tensor("offset")
                self.operator_valid = tensor("valid", torch.bool)
            self.metadata["observation_operator"] = metadata
            self.metadata["operator_cache_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.metadata["full_profile_context"] = "source-only absolute T/S plus stored training-year climatology at source coordinates"
        self.metadata["profile_climate_sha256"] = hashlib.sha256(climate.tobytes()).hexdigest()

    def scene(self, source, query_rows, query_levels, analysis, context_profiles=6080,
              context_seed=0, geometry=None, cached_baseline=None):
        if geometry is None:
            geometry = self.geometry(source, query_rows, 32, query_levels)
        result = super().scene(source, query_rows, query_levels, analysis, context_profiles,
                               context_seed, geometry, cached_baseline)
        neighbors = geometry["neighbors"]
        q = self.indices(query_rows)
        result.update(local_profile_values=self.profile_physical[neighbors],
            local_profile_valid=self.valid[neighbors], local_profile_ids=neighbors,
            local_profile_absolute=self.profile_absolute[neighbors],
            local_profile_climatology=self.profile_climate[neighbors],
            profile_depths=self.depth, profile_normalization_mean=self.profile_mean,
            profile_normalization_std=self.profile_std, query_level=self.indices(query_levels))
        if self.surface:
            if self.operator_h is None:
                raise ValueError("Surface innovation trials require the registered operator cache")
            features = self.x[q]
            result.update(surface_observation=features[:, [44, 46, 48]],
                surface_operator=self.operator_h[q], surface_operator_offset=self.operator_offset[q],
                surface_operator_valid=self.operator_valid[q] & (features[:, [45, 47, 49]] > 0))
        return result
