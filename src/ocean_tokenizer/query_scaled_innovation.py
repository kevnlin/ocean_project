"""The innovation model with its local branch in units of a scale at the query.

``radius_scale_scenes`` (``scale_by="query"``) puts the local branch in the
units of a scale at the query by dividing the anchor and every neighbour
innovation by it before the model sees them. That cannot be combined with the
complete-profile T/S alignment of ``innovation_ocean`` as it stands: whenever
the alignment shifts a neighbour's sampling depth it re-derives that
neighbour's innovation from the full profile in per-depth z units, so shifted
and unshifted neighbours would arrive in different units.

Here a scene keeps its neighbour innovations in per-depth z units and supplies
the scale as ``local_query_scale [Q,2]`` (the anchor and the target are divided
by it as before). The innovations are divided by it once the alignment has
produced them, so the local branch sees one set of units whatever the shift.
At zero shift the numbers are exactly those of ``radius_scale_scenes``.

Every module of the model, its parameters and their initial values are
``innovation_ocean``'s, unchanged; only the order of the division differs.
"""
from __future__ import annotations

from .innovation_ocean import InnovationOceanModel
from .profile_alignment import FullProfileAlignment


class QueryScaledAlignment(FullProfileAlignment):
    """``FullProfileAlignment`` whose returned innovations are divided by the query scale."""

    def forward(self, scene, h):
        updated, diagnostics = super().forward(scene, h)
        scale = scene.get("local_query_scale")
        if scale is not None:
            innovation = updated["local_innovation"]
            updated["local_innovation"] = innovation / scale[:, None].to(innovation.dtype)
        return updated, diagnostics


class QueryScaledInnovationModel(InnovationOceanModel):

    def __init__(self, config):
        super().__init__(config)
        if self.alignment is not None:
            # the same module, parameters and initial values; only forward differs
            self.alignment.__class__ = QueryScaledAlignment

    def _local_evidence(self, scene, h, baseline_correction):
        if self.alignment is None and "local_query_scale" in scene:
            scene = {**scene, "local_innovation": scene["local_innovation"] / scene["local_query_scale"][:, None]}
        return super()._local_evidence(scene, h, baseline_correction)
