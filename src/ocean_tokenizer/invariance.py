"""Task 6 — token-manipulation probes for the decisive invariance tests.

Pure helpers shared by tests/test_mbca_invariance.py, test_token_refinement.py,
test_profile_resampling.py and experiments/17_invariance_summary.py.  Every
probe manipulates a *TokenBatch* (post-encoder), so "same physical evidence,
different token representation" is constructed exactly, with all Task-4
metadata carried along.
"""
from __future__ import annotations
import torch

from .token_api import TokenBatch

_META = TokenBatch._OPT_INT + TokenBatch._OPT_FLOAT


def index_tokens(tb: TokenBatch, r: torch.Tensor) -> TokenBatch:
    """Gather tokens along the token axis with index vector ``r``."""
    kw = {}
    for name in _META:
        v = getattr(tb, name)
        kw[name] = None if v is None else v[:, r].clone()
    return TokenBatch(tb.emb[:, r].clone(), tb.coord[:, r].clone(),
                      tb.modality[:, r].clone(), tb.mask[:, r].clone(), **kw)


def split_token(tb: TokenBatch, idx: int, n: int) -> TokenBatch:
    """Test A: replace token ``idx`` with n identical children of mass w/n.

    Ideal partition — identical key/value content, mass divided.  MBCA must be
    exactly invariant; standard attention gives the token n-fold multiplicity.
    """
    N = tb.emb.shape[1]
    reps = torch.ones(N, dtype=torch.long)
    reps[idx] = n
    r = torch.repeat_interleave(torch.arange(N), reps)
    out = index_tokens(tb, r)
    if out.support_mass is not None:
        out.support_mass[:, idx:idx + n] /= n
    return out


def duplicate_tokens(tb: TokenBatch, token_idx, factor: int,
                     divide_mass: bool = True) -> TokenBatch:
    """Test B: append (factor-1) copies of the selected tokens.

    ``divide_mass=True`` divides each duplicated token's original mass across
    its copies (the controlled MBCA condition of the brief); ``False`` keeps
    full mass per copy (naive re-ingestion — copies masquerade as new
    evidence for every method).
    """
    token_idx = torch.as_tensor(token_idx, dtype=torch.long)
    N = tb.emb.shape[1]
    extra = token_idx.repeat(factor - 1)
    r = torch.cat([torch.arange(N), extra])
    out = index_tokens(tb, r)
    if divide_mass and out.support_mass is not None:
        out.support_mass[:, token_idx] /= factor
        out.support_mass[:, N:] /= factor
    return out


@torch.no_grad()
def output_change(model, tokens_a: TokenBatch, tokens_b: TokenBatch,
                  q: torch.Tensor) -> float:
    """Relative L2 change of decoded predictions between two token sets."""
    ya = model.decode(model.fuse(tokens_a), q)
    yb = model.decode(model.fuse(tokens_b), q)
    return float((ya - yb).norm() / (ya.norm() + 1e-12))


@torch.no_grad()
def latent_change(model, tokens_a: TokenBatch, tokens_b: TokenBatch) -> float:
    """Relative L2 change of the fused latent between two token sets."""
    za = model.fuse(tokens_a)
    zb = model.fuse(tokens_b)
    return float((za - zb).norm() / (za.norm() + 1e-12))


def modality_mass_totals(tb: TokenBatch) -> dict[int, float]:
    """Total support mass per modality id (valid tokens only)."""
    mass = (tb.mask.float() if tb.support_mass is None
            else tb.support_mass * tb.mask.float())
    out = {}
    for m in tb.modality.unique().tolist():
        out[int(m)] = float(mass[tb.modality == m].sum())
    return out
