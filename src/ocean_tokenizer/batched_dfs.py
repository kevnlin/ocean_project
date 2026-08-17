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

import math

import torch
import torch.nn as nn

N_FEATURES = 32
LENGTH_SCALES = (0.35, 0.35, 0.25, 2.0)      # (x, y, z, t)
BASIS_SEED = 0
# Kernel value between tokens carrying DIFFERENT observed variables at the same
# place and time.  1.0 would call them exact duplicates, which is what a purely
# geometric kernel does and what this constant exists to stop: SSH and surface
# temperature sit at the same (x, y, z=0, t), so before this they consolidated
# as one measurement and adding SSH *deleted* evidence from the surface stream
# it shadowed (surface omega 5.197 -> 3.499, -32.7 %).
#
# Distinct from ``dfs.S_CROSS = 0.8``, which is between processing streams of
# the SAME variable (real-time vs delayed-mode).  Different physical quantities
# are far less redundant than that.  Like the doc's own support-noise densities,
# this is a pilot heuristic, not a calibrated cross-covariance.
S_CROSS_VARIABLE = 0.3
LAMBDA_FLOOR = 1e-8
NOISE_DENSITY_POINT = 0.08
NOISE_DENSITY_PATCH = 0.35


class _VariableGroupCoords:
    """Embed a variable group as extra kernel coordinates.

    Groups are placed on a regular simplex with unit pairwise distance, so no
    two groups are closer than any other pair — an integer group index used
    directly would make groups 0 and 2 twice as far apart as 0 and 1, an
    artefact of labelling rather than physics.

    Because the RFF kernel is ``exp(-0.5 |du|^2)`` over the concatenated
    input, appending these coordinates multiplies the spatial kernel by a
    constant cross-group factor — exactly a separable ``k_space * k_variable``.
    """

    @staticmethod
    def scales(n_groups: int, s_cross: float = S_CROSS_VARIABLE) -> tuple:
        """Length scale(s) giving ``k = s_cross`` between different groups."""
        if n_groups < 2:
            return ()
        ell = math.sqrt(-0.5 / math.log(max(min(s_cross, 1 - 1e-12), 1e-12)))
        return (ell,) * (n_groups - 1)

    def __call__(self, group: torch.Tensor, n_groups: int) -> torch.Tensor:
        if n_groups < 2:
            return torch.zeros(*group.shape, 0, dtype=torch.float64,
                               device=group.device)
        v = self._simplex(n_groups).to(group.device)
        return v[group.long()]

    @staticmethod
    def _simplex(n: int) -> torch.Tensor:
        """(n, n-1) vertices of a regular simplex with unit pairwise distance.

        Centre the n standard basis vectors (pairwise distance sqrt 2), scale
        to unit distance, then drop to the n-1 dimensional subspace they span
        via SVD.  Distances are preserved by that rotation, so every pair of
        groups ends up exactly one unit apart.
        """
        v = torch.eye(n, dtype=torch.float64)
        v = (v - v.mean(0, keepdim=True)) / math.sqrt(2.0)
        u, sv, _ = torch.linalg.svd(v, full_matrices=False)
        return u[:, :n - 1] * sv[:n - 1]


variable_group_coords = _VariableGroupCoords()


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


class PerceiverResampler(nn.Module):
    """Conventional fixed-query Perceiver resampler — the `count` control.

    Doc §2.2: "`count` uses a conventional fixed-query Perceiver resampler; its
    reported unit ``omega`` is diagnostic only and does not drive its global
    resampler."

    The distinction from :class:`ConservativeResampler` is the normalisation
    axis, and it is the whole point of the control.  Here the softmax runs over
    **tokens**, so each slot forms a weighted average of whatever it attends to
    and token multiplicity feeds straight through — duplicate a token and its
    content gets more of the slot.  The conservative transport instead
    normalises over **slots**, so each token distributes exactly its own mass
    and nothing is created.  Passing unit masses to the conservative transport
    would make `count` a copy of `uniform` rather than a separate mechanism.
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
        B, N, d = emb.shape
        k = self.key(emb)
        logits = torch.einsum("bnd,sd->bsn", k, self.slot_query) / self.temperature
        logits = logits.masked_fill(~mask[:, None, :], float("-inf"))
        attn = torch.softmax(logits, dim=-1)                    # over TOKENS
        attn = torch.nan_to_num(attn)
        slots = torch.einsum("bsn,bnd->bsd", attn, self.value(emb))
        # reported mass is the token count each slot drew on: diagnostic only
        slot_mass = attn.sum(dim=-1).to(omega.dtype) * mask.sum(dim=1,
                                                               keepdim=True)
        live = mask.any(dim=1, keepdim=True).expand(-1, self.n_slots)
        return slots, live, slot_mass * live.to(slot_mass.dtype)
