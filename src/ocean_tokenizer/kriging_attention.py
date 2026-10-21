"""Kriging attention: covariance-aware weights for the local evidence path.

The local path of ``latent_ocean`` reads the innovations of the K nearest
profiles at the query's level. Its softmax weights score every neighbour on its
own (a Nadaraya-Watson average): a float sampled five times votes five times and
no weight can be negative. Kriging attention takes the weights of the system

    (K_NN + R) w = k_Nq,    estimate = w . d,    variance = k_qq - k_Nq . w

which sees how the neighbours relate to each other: copies of one observation
share its weight, a near neighbour screens a far one in the same direction, and
weights may be negative. With a Gaussian kernel of length scale L and
R = gamma I this is the frozen OI of ``synthetic_latent_experiment.SphericalOI``,
solved in the azimuthal-equidistant plane at the query (query-neighbour
distances are great-circle; neighbour-neighbour distances are the planar
difference of the two offsets).

Modes
  ka_oi          the OI's per-variable, per-level L and gamma, fixed;
  ka_stationary  the same two numbers per variable and level, learned;
  ka_flow        every point gets its own amplitude, a 2x2 length-scale matrix
                 (anisotropic and rotated) and every neighbour its own error
                 ratio, predicted from the query state and the neighbour tokens.
                 Points with different matrices are joined by the
                 Paciorek-Schervish non-stationary Gaussian, which keeps the
                 system positive definite. Zero-initialised heads start it as
                 ka_stationary.
  ka_cluster     CLOINet's cluster covariance (Cutolo et al. 2024) in the same
                 local solve: every point gets soft memberships of
                 ``n_clusters`` clusters (a softmax of a head on the state) and
                 two points correlate by the cosine of their membership
                 vectors (1 for the same memberships, 0 for disjoint clusters),
                 with no distance term; amplitudes and error ratios as in
                 ka_flow. CLOINet's own similarity 1 - sum_c (m_ac - m_bc)^2 is
                 indefinite, and in the local solve it made training diverge
                 (2026-10-21); the cosine is its positive semi-definite form.

The kernel and the solve run in float64, outside autocast.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn

MODES = ("ka_oi", "ka_stationary", "ka_flow", "ka_cluster")
#: bounds of the flow heads: length-scale factor e^1.4 ~ 4, shear 1, amplitude e^1, error ratio e^2
SHAPE_BOUND, SHEAR_BOUND, AMPLITUDE_BOUND, NUGGET_BOUND = 1.4, 1.0, 1.0, 2.0


def length_scale_matrices(ell: Tensor, shape: Tensor) -> Tensor:
    """``(s11, s12, s22)`` of ``Sigma = ell^2 L L^T`` with ``L = [[e^a, 0], [c, e^b]]``.

    ``shape`` holds ``(a, b, c)`` in its last dimension; zeros give ``ell^2 I``.
    ``ell`` broadcasts against ``shape[..., 0]``.
    """
    a, b, c = shape.unbind(-1)
    l11, l22 = a.exp(), b.exp()
    scale = ell.square()
    return torch.stack((scale * l11.square(), scale * l11 * c,
                        scale * (c.square() + l22.square())), -1)


def ps_covariance(xy_a: Tensor, sigma_a: Tensor, xy_b: Tensor, sigma_b: Tensor) -> Tensor:
    """Paciorek-Schervish correlation between two point sets, ``[..., A, B]``.

    ``C = |S_a|^1/4 |S_b|^1/4 |M|^-1/2 exp(-0.5 x^T M^-1 x)``, ``M = (S_a + S_b) / 2``,
    for 2x2 length-scale matrices given as ``(s11, s12, s22)``. Equal isotropic
    matrices ``L^2 I`` give the Gaussian ``exp(-0.5 |x|^2 / L^2)`` of OI.
    """
    a11, a12, a22 = (s[..., :, None] for s in sigma_a.unbind(-1))
    b11, b12, b22 = (s[..., None, :] for s in sigma_b.unbind(-1))
    m11, m12, m22 = (a11 + b11) / 2, (a12 + b12) / 2, (a22 + b22) / 2
    det_m = m11 * m22 - m12.square()
    det_a = a11 * a22 - a12.square()
    det_b = b11 * b22 - b12.square()
    dx = xy_a[..., :, None, 0] - xy_b[..., None, :, 0]
    dy = xy_a[..., :, None, 1] - xy_b[..., None, :, 1]
    quad = (m22 * dx.square() - 2 * m12 * dx * dy + m11 * dy.square()) / det_m
    return (det_a * det_b).pow(.25) / det_m.sqrt() * torch.exp(-.5 * quad)


def cluster_correlation(memberships: Tensor) -> Tensor:
    """Cluster correlation of soft memberships ``[..., A, C]``: the cosine of the membership
    vectors, positive semi-definite with a unit diagonal."""
    unit = memberships / memberships.norm(dim=-1, keepdim=True)
    return unit @ unit.transpose(-1, -2)


def without_oi_anchor(scene: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """The scene with the frozen OI anchor replaced by the background (climatology)."""
    return {**scene, "baseline": scene["query_background"]}


class KrigingAttention(nn.Module):
    """Weights of ``(K + R) w = k`` over the local neighbours, per query and variable.

    ``init_ell_km`` and ``init_gamma`` are ``[2, n_levels]`` (temperature, salinity),
    normally the frozen OI's settings expanded from its depth bands to levels.
    """

    def __init__(self, width: int, mode: str, init_ell_km, init_gamma, min_gamma: float = 1e-3,
                 n_clusters: int = 20):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        log_ell = torch.as_tensor(init_ell_km, dtype=torch.float64).log()
        log_gamma = torch.as_tensor(init_gamma, dtype=torch.float64).clamp_min(min_gamma).log()
        if log_ell.ndim != 2 or log_ell.shape[0] != 2 or log_gamma.shape != log_ell.shape:
            raise ValueError("init_ell_km and init_gamma must both be [2, n_levels]")
        self.mode, self.min_gamma, self.n_clusters = mode, float(min_gamma), int(n_clusters)
        if mode == "ka_oi":
            self.register_buffer("log_ell", log_ell)
            self.register_buffer("log_gamma", log_gamma)
        else:
            self.log_ell = nn.Parameter(log_ell.float())
            self.log_gamma = nn.Parameter(log_gamma.float())
        self.query_head = self.local_head = None
        if mode == "ka_flow":
            # per variable: query (a, b, c, log amplitude); neighbour the same plus its error ratio
            self.query_head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2 * 4))
            self.local_head = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, 2 * 5))
            for head in (self.query_head, self.local_head):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        elif mode == "ka_cluster":
            # per variable: query (n_clusters logits, log amplitude); neighbour also its error ratio.
            # The logits keep their random start (equal memberships are a saddle of B); the
            # amplitude and error-ratio outputs start at zero.
            c = self.n_clusters
            self.query_head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2 * (c + 1)))
            self.local_head = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, 2 * (c + 2)))
            for head, extra in ((self.query_head, 1), (self.local_head, 2)):
                with torch.no_grad():
                    head[-1].weight.view(2, c + extra, -1)[:, c:].zero_()
                nn.init.zeros_(head[-1].bias)

    def forward(self, h: Tensor, local: Tensor, offsets: Tensor, innovation: Tensor,
                valid: Tensor, level: Tensor, decay: Tensor | None = None) -> dict[str, Tensor]:
        """``h [Q,W]``, ``local [Q,K,W]``, ``offsets [Q,K,2]`` km east/north of the query,
        ``innovation`` and ``valid`` ``[Q,K,2]``, ``level [Q]``.

        ``decay [Q,2]`` in [0, 1] multiplies the query-neighbour covariance: the neighbours
        were observed earlier than the query (a forecast lead), so they explain less of it."""
        q, k = innovation.shape[:2]
        f64 = torch.float64
        with torch.autocast(device_type=h.device.type, enabled=False):
            log_ell = self.log_ell.to(f64).clamp(math.log(10.), math.log(5000.))
            log_gamma = self.log_gamma.to(f64).clamp(math.log(self.min_gamma), math.log(10.))
            ell, gamma = log_ell[:, level].T.exp(), log_gamma[:, level].T.exp()      # [Q,2]
            shape = torch.zeros(q, 2, k + 1, 3, dtype=f64, device=h.device)
            log_amplitude = torch.zeros(q, 2, k + 1, dtype=f64, device=h.device)
            nugget = gamma[..., None].expand(q, 2, k)
            memberships = None
            if self.mode == "ka_cluster":
                c = self.n_clusters
                dtype = self.query_head[0].weight.dtype
                hq = self.query_head(h.to(dtype)).to(f64).view(q, 2, c + 1)
                pair = torch.cat((local.to(dtype), h.to(dtype)[:, None].expand(-1, k, -1)), -1)
                hl = self.local_head(pair).to(f64).view(q, k, 2, c + 2).transpose(1, 2)  # [Q,2,K,C+2]
                memberships = torch.cat((hq[:, :, None, :c], hl[..., :c]), 2).softmax(-1)  # [Q,2,1+K,C]
                log_amplitude = AMPLITUDE_BOUND * torch.cat((hq[:, :, None, c], hl[..., c]), 2).tanh()
                nugget = gamma[..., None] * (NUGGET_BOUND * hl[..., c + 1].tanh()).exp()
            elif self.query_head is not None:
                dtype = self.query_head[0].weight.dtype
                hq = self.query_head(h.to(dtype)).to(f64).view(q, 2, 4)
                pair = torch.cat((local.to(dtype), h.to(dtype)[:, None].expand(-1, k, -1)), -1)
                hl = self.local_head(pair).to(f64).view(q, k, 2, 5).transpose(1, 2)    # [Q,2,K,5]
                raw = torch.cat((hq[:, :, None], hl[..., :4]), 2)                        # [Q,2,1+K,4]
                shape = torch.cat((SHAPE_BOUND * raw[..., :2].tanh(),
                                   SHEAR_BOUND * raw[..., 2:3].tanh()), -1)
                log_amplitude = AMPLITUDE_BOUND * raw[..., 3].tanh()
                nugget = gamma[..., None] * (NUGGET_BOUND * hl[..., 4].tanh()).exp()
            # point 0 is the query at the origin, points 1..K the neighbours
            xy = torch.cat((offsets.new_zeros(q, 1, 2), offsets), 1).to(f64)[:, None]
            sigma = length_scale_matrices(ell[..., None], shape)                        # [Q,2,1+K,3]
            amplitude = log_amplitude.exp()
            corr = ps_covariance(xy, sigma, xy, sigma) if memberships is None else cluster_correlation(memberships)
            cov = corr * amplitude[..., :, None] * amplitude[..., None, :]              # [Q,2,1+K,1+K]
            v = valid.transpose(1, 2).to(f64)                                           # [Q,2,K]
            # an invalid neighbour is an identity row: weight 0, the others' system untouched
            system = (cov[..., 1:, 1:] * v[..., :, None] * v[..., None, :]
                      + torch.diag_embed(nugget * v + (1. - v)))
            rhs = cov[..., 0, 1:] * v
            if decay is not None:
                rhs = rhs * decay.to(f64)[..., None]
            weights = torch.linalg.solve(system, rhs[..., None])[..., 0] * v
            d = torch.where(valid, innovation, torch.zeros_like(innovation)).transpose(1, 2).to(f64)
            estimate = (weights * d).sum(-1)
            variance = (cov[..., 0, 0] - (weights * rhs).sum(-1)).clamp_min(0.)
        dtype = innovation.dtype
        return {"estimate": estimate.to(dtype), "variance": variance.to(dtype), "weights": weights.to(dtype),
                # east and north length scales at the query, sqrt(s11), sqrt(s22)
                "length_scale_km": sigma[..., 0, ::2].detach().sqrt().to(dtype),
                "nugget": nugget.detach().to(dtype)}
