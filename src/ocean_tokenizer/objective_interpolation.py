"""Causal local objective interpolation (mentor doc §2.5).

Deterministic, **zero trainable parameters**.  For temperature and salinity
separately it combines observations with inverse-support-noise weights against
a zero-anomaly background:

    OI_c(q) = [rho0 * 0 + sum_i r_i k(q, S_i) v_ic]
              / [rho0 + sum_i r_i k(q, S_i)]

with ``r_i = sigma_i ** -p`` the inverse-noise weight and ``k`` a Gaussian RBF
in scaled distance, cut off beyond ``cutoff`` scale lengths.

Frozen settings (doc §2.5, selected on development validation only):
``ell_xy = 0.08``, ``ell_z = 0.075``, ``ell_t = 12`` months, ``rho0 = 0.30``,
noise exponent 1, cutoff 4.

Coordinates are ``(x, y, z, t)`` with x/y/z normalised to [0, 1] over the
domain and t in months relative to the analysis time — so ``ell_xy = 0.08`` is
8 % of the box (~2 deg latitude on the 25-50 N GODAS region) and
``ell_z = 0.075`` is ~71 m over a 949 m column.

Two deliberate notes on fidelity to the doc:

* §2.5 says "exact causal point observations are reproduced exactly".  With
  ``rho0 = 0.30`` that cannot hold: the background always keeps weight
  ``rho0 / (rho0 + r)``, so an analysis at an observation is shrunk toward
  zero anomaly by ~2 % at the profile-point noise of 0.08.  The formula in the
  doc is unambiguous, so the formula is implemented and the shrinkage is
  asserted in the tests; it vanishes only at ``rho0 = 0``.
* Standalone OI uses the T/S value channels only.  It does not invent an
  SSH-to-T/S cross-covariance, so SSH is simply not an input here (§2.5).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class OISettings:
    """Frozen §2.5 settings.  ``frozen=True`` so a row cannot tune them."""
    ell_xy: float = 0.08
    ell_z: float = 0.075
    ell_t: float = 12.0            # months
    rho0: float = 0.30             # background precision
    noise_exponent: float = 1.0
    cutoff: float = 4.0            # in scaled distances
    eps: float = 1e-12


def superobservations(coord: torch.Tensor, value: torch.Tensor,
                      sigma: torch.Tensor, decimals: int = 6):
    """Merge exact-duplicate supports into inverse-noise-weighted superobs.

    Doc §2.5: "Exact duplicate supports are first combined into
    inverse-noise-weighted superobservations."  Without this, re-ingesting one
    measurement n times multiplies its pull on the analysis by n, which is the
    duplicate-sensitivity the whole evidence line exists to avoid.

    Returns ``(coord, value, sigma)`` with one row per distinct support.

    The merged sigma is the **mean** precision, not the summed precision.  That
    distinction is the whole point: these are *exact duplicates* — one
    measurement re-ingested — so they carry one measurement's information.
    Summing precision would hand n copies n times the pull on the analysis,
    manufacturing evidence out of re-ingestion, which is exactly the failure
    the evidence line exists to prevent.  For identical sigmas the merge is
    therefore an identity on precision, and n copies move the analysis exactly
    as far as one does.

    Distinct-but-nearby observations are a different case and are NOT merged
    here — only bit-identical supports collapse.
    """
    if coord.shape[0] == 0:
        return coord, value, sigma
    key = torch.round(coord * 10 ** decimals) / 10 ** decimals
    uniq, inv = torch.unique(key, dim=0, return_inverse=True)
    w = sigma.clamp(min=1e-12) ** -1.0                     # precision weights
    n = uniq.shape[0]
    wsum = torch.zeros(n, dtype=w.dtype, device=w.device).index_add_(0, inv, w)
    vsum = torch.zeros(n, value.shape[1], dtype=value.dtype, device=value.device)
    vsum.index_add_(0, inv, w[:, None] * value)
    count = torch.zeros(n, dtype=w.dtype, device=w.device).index_add_(
        0, inv, torch.ones_like(w))
    mean_precision = wsum / count.clamp(min=1.0)
    return uniq, vsum / wsum[:, None].clamp(min=1e-12), mean_precision ** -1.0


class ObjectiveInterpolation(nn.Module):
    """Causal local Gaussian OI against a zero-anomaly background."""

    def __init__(self, settings: OISettings | None = None):
        super().__init__()
        self.s = settings or OISettings()

    def forward(self, query: torch.Tensor, query_cutoff: torch.Tensor,
                obs_coord: torch.Tensor, obs_value: torch.Tensor,
                obs_sigma: torch.Tensor) -> torch.Tensor:
        """(Q,4) queries -> (Q,C) analysis in the observations' units.

        ``query_cutoff`` (Q,) is the latest observation time each query may
        use, in the same month units as the coordinates' t axis.  An
        observation is admitted for a query only if ``t_obs <= cutoff``: this
        is applied per query, not per batch, so one future observation cannot
        leak into an earlier analysis (doc §3.4).
        """
        s = self.s
        Q, C = query.shape[0], obs_value.shape[1] if obs_value.numel() else 2
        if obs_coord.shape[0] == 0:
            return torch.zeros(Q, C, dtype=query.dtype, device=query.device)

        c, v, sig = superobservations(obs_coord, obs_value, obs_sigma)

        d2 = (((query[:, None, 0] - c[None, :, 0]) / s.ell_xy) ** 2
              + ((query[:, None, 1] - c[None, :, 1]) / s.ell_xy) ** 2
              + ((query[:, None, 2] - c[None, :, 2]) / s.ell_z) ** 2
              + ((query[:, None, 3] - c[None, :, 3]) / s.ell_t) ** 2)
        k = torch.exp(-0.5 * d2)
        k = torch.where(d2 <= s.cutoff ** 2, k, torch.zeros_like(k))

        causal = c[None, :, 3] <= query_cutoff[:, None] + s.eps
        r = sig.clamp(min=1e-12) ** (-s.noise_exponent)
        w = torch.where(causal, k * r[None, :], torch.zeros_like(k))   # (Q,N)

        num = w @ v                                                     # (Q,C)
        den = w.sum(dim=1, keepdim=True) + s.rho0
        return num / den.clamp(min=s.eps)
