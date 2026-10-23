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
  ka_deep        a Gaussian kernel in a learned space (2026-10-21, after the
                 group meeting's "replace the Gaussian covariance by a net"):
                 every point gets ``n_features`` learned coordinates psi from
                 the state, and two points correlate by
                 exp(-1/2 (|dx|^2 / L^2 + |psi_a - psi_b|^2)). Any psi gives a
                 positive definite kernel (a Gaussian in R^(2+F)), and the
                 correlation can follow a curved current when psi does: no
                 length-scale matrices, the net replaces the geometry. Zero
                 initialised, it starts as ka_stationary.
  ka_product     ka_flow's kernel times the same feature Gaussian (a Schur
                 product of two positive definite kernels); starts as ka_flow.
  ka_multiscale  two Paciorek-Schervish kernels, the second with three times
                 the length scale, each with its own per-point amplitude:
                 s1_a s1_b PS(Sigma) + s2_a s2_b PS(9 Sigma). The second
                 amplitude starts at e^-1.5, so the start is nearly ka_stationary.

``multivariate=True`` solves temperature and salinity together with a linear
model of coregionalisation: Cov[(a,u),(b,v)] = B1_uv C^T_ab + B2_uv C^S_ab with
B1 and B2 positive semi-definite 2x2 matrices per level (Cholesky factors),
initialised to diag(1, 0) and diag(0, 1), so the start equals the per-variable
solve; one 2K x 2K system per query.

The kernel and the solve run in float64, outside autocast.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn

MODES = ("ka_oi", "ka_stationary", "ka_flow", "ka_cluster", "ka_deep", "ka_product", "ka_multiscale")
#: bounds of the flow heads: length-scale factor e^1.4 ~ 4, shear 1, amplitude e^1, error ratio e^2
SHAPE_BOUND, SHEAR_BOUND, AMPLITUDE_BOUND, NUGGET_BOUND = 1.4, 1.0, 1.0, 2.0
#: learned coordinates of ka_deep / ka_product: how many and how far (tanh bound, in correlation units)
N_FEATURES, FEATURE_BOUND = 4, 3.0
#: ka_multiscale: the second kernel's length-scale factor and the log amplitude it starts from
SECOND_SCALE, SECOND_LOG_AMPLITUDE = 3.0, -1.5


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


def feature_gaussian(phi: Tensor) -> Tensor:
    """``exp(-1/2 |phi_a - phi_b|^2)`` over the points ``[..., A, F]``: a Gaussian kernel in the
    feature space, positive definite for any features, with a unit diagonal."""
    square = (phi[..., :, None, :] - phi[..., None, :, :]).square().sum(-1)
    return torch.exp(-.5 * square)


def coregionalisation(factors: Tensor) -> Tensor:
    """``B = L L^T`` of lower-triangular ``L = [[l11, 0], [l21, l22]]`` given as ``(l11, l21, l22)``
    in the last dimension: ``[..., 2, 2]``, positive semi-definite."""
    l11, l21, l22 = factors.unbind(-1)
    return torch.stack((torch.stack((l11.square(), l11 * l21), -1),
                        torch.stack((l11 * l21, l21.square() + l22.square()), -1)), -2)


def without_oi_anchor(scene: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """The scene with the frozen OI anchor replaced by the background (climatology)."""
    return {**scene, "baseline": scene["query_background"]}


class KrigingAttention(nn.Module):
    """Weights of ``(K + R) w = k`` over the local neighbours, per query and variable.

    ``init_ell_km`` and ``init_gamma`` are ``[2, n_levels]`` (temperature, salinity),
    normally the frozen OI's settings expanded from its depth bands to levels.
    """

    def __init__(self, width: int, mode: str, init_ell_km, init_gamma, min_gamma: float = 1e-3,
                 n_clusters: int = 20, multivariate: bool = False):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        log_ell = torch.as_tensor(init_ell_km, dtype=torch.float64).log()
        log_gamma = torch.as_tensor(init_gamma, dtype=torch.float64).clamp_min(min_gamma).log()
        if log_ell.ndim != 2 or log_ell.shape[0] != 2 or log_gamma.shape != log_ell.shape:
            raise ValueError("init_ell_km and init_gamma must both be [2, n_levels]")
        self.mode, self.min_gamma, self.n_clusters = mode, float(min_gamma), int(n_clusters)
        self.multivariate = bool(multivariate)
        n_levels = log_ell.shape[1]
        if mode == "ka_oi":
            self.register_buffer("log_ell", log_ell)
            self.register_buffer("log_gamma", log_gamma)
        else:
            self.log_ell = nn.Parameter(log_ell.float())
            self.log_gamma = nn.Parameter(log_gamma.float())
        # per variable: the query's kernel outputs, and the same for a neighbour plus its error ratio
        self.n_query = {"ka_oi": 0, "ka_stationary": 0, "ka_flow": 4, "ka_cluster": self.n_clusters + 1,
                        "ka_deep": 1 + N_FEATURES, "ka_product": 4 + N_FEATURES, "ka_multiscale": 5}[mode]
        self.query_head = self.local_head = None
        if self.n_query:
            self.query_head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2 * self.n_query))
            self.local_head = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(),
                                            nn.Linear(width, 2 * (self.n_query + 1)))
            if mode == "ka_cluster":
                # the logits keep their random start (equal memberships are a saddle of B); the
                # amplitude and error-ratio outputs start at zero
                c = self.n_clusters
                for head, extra in ((self.query_head, 1), (self.local_head, 2)):
                    with torch.no_grad():
                        head[-1].weight.view(2, c + extra, -1)[:, c:].zero_()
                    nn.init.zeros_(head[-1].bias)
            else:
                for head in (self.query_head, self.local_head):
                    nn.init.zeros_(head[-1].weight)
                    nn.init.zeros_(head[-1].bias)
        self.coregionalisation = None
        if self.multivariate:
            # two components, (l11, l21, l22) per level: B1 = diag(1, 0), B2 = diag(0, 1) at the start
            factors = torch.zeros(2, n_levels, 3)
            factors[0, :, 0] = 1.
            factors[1, :, 2] = 1.
            self.coregionalisation = nn.Parameter(factors)

    def _covariance(self, h, local, offsets, level, q, k):
        """``cov [Q,2,1+K,1+K]`` (point 0 the query), ``nugget [Q,2,K]``, ``sigma [Q,2,1+K,3]``."""
        f64 = torch.float64
        log_ell = self.log_ell.to(f64).clamp(math.log(10.), math.log(5000.))
        log_gamma = self.log_gamma.to(f64).clamp(math.log(self.min_gamma), math.log(10.))
        ell, gamma = log_ell[:, level].T.exp(), log_gamma[:, level].T.exp()      # [Q,2]
        shape = torch.zeros(q, 2, k + 1, 3, dtype=f64, device=h.device)
        log_amplitude = torch.zeros(q, 2, k + 1, dtype=f64, device=h.device)
        nugget = gamma[..., None].expand(q, 2, k)
        memberships = features = log_amplitude_2 = None
        if self.query_head is not None:
            n = self.n_query
            dtype = self.query_head[0].weight.dtype
            hq = self.query_head(h.to(dtype)).to(f64).view(q, 2, n)
            pair = torch.cat((local.to(dtype), h.to(dtype)[:, None].expand(-1, k, -1)), -1)
            hl = self.local_head(pair).to(f64).view(q, k, 2, n + 1).transpose(1, 2)   # [Q,2,K,n+1]
            raw = torch.cat((hq[:, :, None], hl[..., :n]), 2)                          # [Q,2,1+K,n]
            nugget = gamma[..., None] * (NUGGET_BOUND * hl[..., n].tanh()).exp()
            if self.mode == "ka_cluster":
                c = self.n_clusters
                memberships = raw[..., :c].softmax(-1)                                  # [Q,2,1+K,C]
                log_amplitude = AMPLITUDE_BOUND * raw[..., c].tanh()
            elif self.mode == "ka_deep":
                log_amplitude = AMPLITUDE_BOUND * raw[..., 0].tanh()
                features = FEATURE_BOUND * raw[..., 1:].tanh()                          # [Q,2,1+K,F]
            else:
                shape = torch.cat((SHAPE_BOUND * raw[..., :2].tanh(),
                                   SHEAR_BOUND * raw[..., 2:3].tanh()), -1)
                log_amplitude = AMPLITUDE_BOUND * raw[..., 3].tanh()
                if self.mode == "ka_product":
                    features = FEATURE_BOUND * raw[..., 4:].tanh()
                elif self.mode == "ka_multiscale":
                    log_amplitude_2 = SECOND_LOG_AMPLITUDE + AMPLITUDE_BOUND * raw[..., 4].tanh()
        # point 0 is the query at the origin, points 1..K the neighbours
        xy = torch.cat((offsets.new_zeros(q, 1, 2), offsets), 1).to(f64)[:, None]      # [Q,1,1+K,2]
        sigma = length_scale_matrices(ell[..., None], shape)                            # [Q,2,1+K,3]
        if memberships is not None:
            corr = cluster_correlation(memberships)
        elif self.mode == "ka_deep":
            scaled = xy.expand(q, 2, k + 1, 2) / ell[..., None, None]
            corr = feature_gaussian(torch.cat((scaled, features), -1))
        else:
            corr = ps_covariance(xy, sigma, xy, sigma)
            if features is not None:
                corr = corr * feature_gaussian(features)
        amplitude = log_amplitude.exp()
        cov = corr * amplitude[..., :, None] * amplitude[..., None, :]                  # [Q,2,1+K,1+K]
        if log_amplitude_2 is not None:
            wide = ps_covariance(xy, SECOND_SCALE ** 2 * sigma, xy, SECOND_SCALE ** 2 * sigma)
            amplitude_2 = log_amplitude_2.exp()
            cov = cov + wide * amplitude_2[..., :, None] * amplitude_2[..., None, :]
        return cov, nugget, sigma

    def forward(self, h: Tensor, local: Tensor, offsets: Tensor, innovation: Tensor,
                valid: Tensor, level: Tensor, decay: Tensor | None = None) -> dict[str, Tensor]:
        """``h [Q,W]``, ``local [Q,K,W]``, ``offsets [Q,K,2]`` km east/north of the query,
        ``innovation`` and ``valid`` ``[Q,K,2]``, ``level [Q]``.

        ``decay [Q,2]`` in [0, 1] multiplies the query-neighbour covariance: the neighbours
        were observed earlier than the query (a forecast lead), so they explain less of it."""
        q, k = innovation.shape[:2]
        f64 = torch.float64
        with torch.autocast(device_type=h.device.type, enabled=False):
            cov, nugget, sigma = self._covariance(h, local, offsets, level, q, k)
            v = valid.transpose(1, 2).to(f64)                                           # [Q,2,K]
            d = torch.where(valid, innovation, torch.zeros_like(innovation)).transpose(1, 2).to(f64)
            if self.coregionalisation is None:
                # an invalid neighbour is an identity row: weight 0, the others' system untouched
                system = (cov[..., 1:, 1:] * v[..., :, None] * v[..., None, :]
                          + torch.diag_embed(nugget * v + (1. - v)))
                rhs = cov[..., 0, 1:] * v
                if decay is not None:
                    rhs = rhs * decay.to(f64)[..., None]
                weights = torch.linalg.solve(system, rhs[..., None])[..., 0] * v
                estimate = (weights * d).sum(-1)
                variance = (cov[..., 0, 0] - (weights * rhs).sum(-1)).clamp_min(0.)
                own = weights
            else:
                b = coregionalisation(self.coregionalisation.to(f64))[:, level]         # [2,Q,2,2]
                # joint covariance of the 2(1+K) points, variable-major: (T: query, neighbours; S: ...)
                joint = (b[0][:, :, :, None, None] * cov[:, 0][:, None, None]
                         + b[1][:, :, :, None, None] * cov[:, 1][:, None, None])         # [Q,2,2,1+K,1+K]
                joint = joint.permute(0, 1, 3, 2, 4).reshape(q, 2 * (k + 1), 2 * (k + 1))
                neighbours = torch.cat((torch.arange(1, k + 1), torch.arange(k + 2, 2 * k + 2))).to(h.device)
                queries = torch.tensor([0, k + 1], device=h.device)
                vv = v.reshape(q, 2 * k)
                system = (joint[:, neighbours][:, :, neighbours] * vv[:, :, None] * vv[:, None, :]
                          + torch.diag_embed(nugget.reshape(q, 2 * k) * vv + (1. - vv)))
                rhs = joint[:, queries][:, :, neighbours] * vv[:, None, :]                # [Q,2,2K]
                if decay is not None:
                    rhs = rhs * decay.to(f64).repeat_interleave(k, -1)[:, None, :]
                weights = torch.linalg.solve(system, rhs.transpose(1, 2)).transpose(1, 2) * vv[:, None, :]
                estimate = (weights * d.reshape(q, 1, 2 * k)).sum(-1)
                variance = (joint[:, queries, queries] - (weights * rhs).sum(-1)).clamp_min(0.)
                own = torch.stack((weights[:, 0, :k], weights[:, 1, k:]), 1)
        dtype = innovation.dtype
        out = {"estimate": estimate.to(dtype), "variance": variance.to(dtype), "weights": own.to(dtype),
               # east and north length scales at the query, sqrt(s11), sqrt(s22)
               "length_scale_km": sigma[..., 0, ::2].detach().sqrt().to(dtype),
               "nugget": nugget.detach().to(dtype)}
        if self.coregionalisation is not None:
            out["weights_joint"] = weights.to(dtype)
        return out
