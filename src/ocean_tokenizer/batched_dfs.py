"""Set-level DFS via random Fourier features (mentor doc §2.2).

    psi_i    = sum_q w_iq phi(x_iq)                 support-integrated features
    lambda_i = max(noise_density_i * sum_q w_iq, 1e-8)
    omega_i  = whitened ridge-leverage contribution of token i

This is *not* a per-token score heuristic.  Whitening by ``lambda`` and taking
ridge leverage in a **joint** solve makes overlapping or duplicate supports
compete for one finite pool of evidence:

    psi~_i = psi_i / sqrt(lambda_i)
    A      = Psi~^T Psi~ + I                 (F x F)
    omega_i = psi~_i^T A^-1 psi~_i           (= H_ii of the hat matrix)

Two consequences fall straight out of the algebra and are pinned by tests.
With ``s = |psi~|^2``, a lone token gets ``s / (1 + s)``; each of two identical
tokens gets ``s / (1 + 2s)``, so as ``s`` grows the pair splits *one* token's
worth of evidence rather than casting two votes.  And ``sum_i omega_i =
trace(A^-1 (A - I)) = F - trace(A^-1) <= F``: total evidence is bounded by the
feature count however many tokens arrive.

Structurally this differs from ``dfs.py``: the random-feature form solves an
``F x F`` system (32 x 32 here) rather than an ``N x N`` kriging system, so it
is cheap and exact-in-the-features instead of localised.  The two are separate
formulations of the same quantity and are deliberately kept apart — ``dfs.py``
belongs to the paper line.

The solve runs in **float64**.  The whole point is resolving how singular a
redundant neighbourhood is, and fp32 clamps near-duplicate leverage to 1.0,
silently restoring the independent-vote behaviour this exists to remove.

Final-run settings (§2.2): 32 random features, support length scales
(0.35, 0.35, 0.25, 2.0), fixed basis seed 0, no downstream mass calibration.
Support-noise densities of 0.08 (profile points) and 0.35 (patches) are pilot
heuristics, not calibrated GODAS observation errors; the 1e-8 floor is a
numerical guard, not an estimated variance.
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_FEATURES = 32
LENGTH_SCALES = (0.35, 0.35, 0.25, 2.0)      # (x, y, z, t)
BASIS_SEED = 0
LAMBDA_FLOOR = 1e-8
NOISE_DENSITY_POINT = 0.08
NOISE_DENSITY_PATCH = 0.35


class RandomFourierBasis(nn.Module):
    """Fixed random Fourier features approximating a Gaussian kernel.

    Frequencies are drawn once from a fixed seed and registered as buffers, so
    the basis is part of the *experiment definition* rather than something
    training can drift.  ``seed`` is held separate from the optimizer/data seed
    precisely so a seed sweep varies initialisation and sampling only.
    """

    def __init__(self, n_features: int = N_FEATURES,
                 length_scales=LENGTH_SCALES, seed: int = BASIS_SEED):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        ell = torch.as_tensor(length_scales, dtype=torch.float64)
        w = torch.randn(n_features, ell.numel(), generator=g,
                        dtype=torch.float64) / ell
        b = torch.rand(n_features, generator=g, dtype=torch.float64) * 2 * torch.pi
        self.register_buffer("freq", w)
        self.register_buffer("phase", b)
        self.n_features = int(n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(..., 4) -> (..., F) features, scaled so k(x, x) = 1.

        The phase-shifted form ``sqrt(2/F) cos(Wx + b)`` is used rather than
        concatenated cos/sin: it spans exactly ``n_features`` dimensions, so
        "32 random features" (§2.2) really does cap total set evidence at 32.
        The cos/sin pair would span 2F and quietly double that ceiling.
        """
        proj = x.to(self.freq.dtype) @ self.freq.T + self.phase
        return torch.cos(proj) * (2.0 / self.n_features) ** 0.5


def integrate_support(basis: RandomFourierBasis, nodes: torch.Tensor,
                      weights: torch.Tensor, noise_density: torch.Tensor):
    """Integrate the basis over each token's quadrature support.

    ``nodes`` (N, Q, 4), ``weights`` (N, Q), ``noise_density`` (N,).
    Returns ``(psi (N, F), lambda (N,))``.

    A wider support integrates more of the field, so it both accumulates more
    feature mass and carries proportionally more noise — hence lambda scales
    with the same ``sum_q w_iq``.
    """
    phi = basis(nodes)                                   # (N, Q, F)
    psi = (weights.to(phi.dtype)[..., None] * phi).sum(dim=1)
    wsum = weights.to(phi.dtype).sum(dim=1)
    lam = torch.clamp(noise_density.to(phi.dtype) * wsum, min=LAMBDA_FLOOR)
    return psi, lam


def dfs_omega(psi: torch.Tensor, lam: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    """Whitened ridge leverage of each active token. ``psi`` (N, F) -> (N,).

    Masked tokens are removed from the solve entirely — not merely zeroed
    afterwards — so an inactive token cannot compete for evidence with the
    active ones.
    """
    psi = psi.to(torch.float64)
    lam = lam.to(torch.float64)
    out = torch.zeros(psi.shape[0], dtype=torch.float64, device=psi.device)
    if not bool(mask.any()):
        return out
    p = psi[mask] / lam[mask].sqrt()[:, None]            # (M, F)
    F = p.shape[1]
    A = p.T @ p + torch.eye(F, dtype=p.dtype, device=p.device)
    sol = torch.linalg.solve(A, p.T)                     # (F, M)
    out[mask] = (p.T * sol).sum(dim=0)                   # diag(p A^-1 p^T)
    return out


class ConservativeResampler(nn.Module):
    """Transport token embeddings into a fixed slot budget, conserving mass.

    Doc §2.2: "the learned conservative resampler transports every token mass
    into 32 observation slots and preserves ``sum(slot_mass) == sum(active
    omega)``".

    The transport plan is a learned attention over slots, row-normalised so
    each token distributes exactly its own mass and nothing is created or
    destroyed.  Conservation is therefore structural — it holds at
    initialisation, after training, and for any input — rather than being a
    property the optimizer is asked to respect.

    ``uniform`` is the matched mechanism control: identical transport, with
    ``omega_i = 1`` supplied by the caller instead of measured evidence.
    """

    def __init__(self, d_model: int, n_slots: int = 32, temperature: float = 1.0):
        super().__init__()
        self.n_slots = int(n_slots)
        self.slot_query = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.temperature = float(temperature)

    def forward(self, emb: torch.Tensor, omega: torch.Tensor,
                mask: torch.Tensor):
        """(B,N,d) + (B,N) masses -> (B,S,d) slots, (B,S) mask, (B,S) mass."""
        B, N, d = emb.shape
        k = self.key(emb)                                        # (B,N,d)
        logits = torch.einsum("bnd,sd->bns", k, self.slot_query) / self.temperature
        logits = logits.masked_fill(~mask[..., None], float("-inf"))
        # normalise over SLOTS: each token splits its own mass across slots,
        # so the column sums reproduce the incoming masses exactly
        plan = torch.softmax(logits, dim=-1)                      # (B,N,S)
        plan = torch.where(mask[..., None], plan, torch.zeros_like(plan))

        # mass bookkeeping in omega's dtype (float64 in the estimator) so the
        # conservation identity is exact to that precision, not to float32's
        mm = (omega * mask).to(omega.dtype).unsqueeze(-1)         # (B,N,1)
        slot_mass = (plan.to(omega.dtype) * mm).sum(dim=1)        # (B,S)
        v = self.value(emb)
        w = (plan * (omega * mask).to(plan.dtype).unsqueeze(-1))
        slots = torch.einsum("bns,bnd->bsd", w, v)
        slots = slots / slot_mass.to(v.dtype).clamp(min=1e-12).unsqueeze(-1)
        return slots, slot_mass > 0, slot_mass
