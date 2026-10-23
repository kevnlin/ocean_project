"""The query-scaled innovation model with a choice of local weighting.

``local_weighting`` sets how the local path combines the innovations of the K
nearest profiles at the query's level:

  softmax        the existing learned scorer, unchanged;
  ka_oi          kriging attention with the frozen OI's L and gamma, fixed;
  ka_stationary  kriging attention, L and gamma learned per variable and level;
  ka_flow        kriging attention with per-point anisotropic length scales,
                 amplitudes and error ratios predicted from the query state and
                 the neighbour tokens (``kriging_attention``).

Everything else -- the latent, the T/S profile alignment, the query scaling, the
gate and the heads -- is ``query_scaled_innovation``'s. The kriging modes need
their starting L and gamma as ``[2][n_profile_levels]`` tuples (temperature,
salinity), normally the OI selection expanded from its depth bands to levels.
For a model without the frozen OI anchor, feed it
``kriging_attention.without_oi_anchor(scene)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .innovation_ocean import InnovationOceanConfig
from .kriging_attention import MODES, KrigingAttention
from .query_scaled_innovation import QueryScaledInnovationModel

LOCAL_WEIGHTINGS = ("softmax",) + MODES


@dataclass(frozen=True)
class KrigingOceanConfig(InnovationOceanConfig):
    local_weighting: str = "softmax"
    kriging_ell_km: tuple = ()
    kriging_gamma: tuple = ()

    def __post_init__(self):
        super().__post_init__()
        if self.local_weighting not in LOCAL_WEIGHTINGS:
            raise ValueError(f"local_weighting must be one of {LOCAL_WEIGHTINGS}")
        as_tuple = lambda rows: tuple(tuple(float(x) for x in row) for row in rows)
        object.__setattr__(self, "kriging_ell_km", as_tuple(self.kriging_ell_km))
        object.__setattr__(self, "kriging_gamma", as_tuple(self.kriging_gamma))
        if self.local_weighting != "softmax":
            shape = (2, self.n_profile_levels)
            for name in ("kriging_ell_km", "kriging_gamma"):
                rows = getattr(self, name)
                if len(rows) != shape[0] or any(len(row) != shape[1] for row in rows):
                    raise ValueError(f"{name} must be [2][{self.n_profile_levels}] for {self.local_weighting}")
                if any(x <= 0 for row in rows for x in row):
                    raise ValueError(f"{name} must be positive")


class KrigingOceanModel(QueryScaledInnovationModel):

    def __init__(self, config: KrigingOceanConfig):
        super().__init__(config)
        self.kriging = None
        if config.local_weighting != "softmax":
            self.kriging = KrigingAttention(config.width, config.local_weighting,
                                            config.kriging_ell_km, config.kriging_gamma)

    def _local_weights(self, h, local, geometry, innovation, valid, scene):
        if self.kriging is None:
            return super()._local_weights(h, local, geometry, innovation, valid, scene)
        out = self.kriging(h, local, geometry[..., :2], innovation, valid, scene["query_level"])
        weights = out["weights"]
        stats = torch.cat((weights.sum(-1), out["variance"]), -1).to(h.dtype)
        used = valid.transpose(1, 2)
        negative = ((weights < 0) & used).sum((0, 2)) / used.sum((0, 2)).clamp_min(1)
        return out["estimate"], stats, {"kriging_negative_share": negative.detach(),
                                        "kriging_length_scale_km": out["length_scale_km"].mean(0).detach()}
