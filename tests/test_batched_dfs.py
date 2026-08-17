"""Set-level DFS via random Fourier features (mentor doc §2.2).

    psi_i    = sum_q w_iq phi(x_iq)                 support-integrated features
    lambda_i = max(noise_density_i * sum_q w_iq, 1e-8)
    omega_i  = whitened ridge-leverage of token i in the active set

The point of the construction — and what these tests pin — is that overlapping
or duplicate supports **compete for finite set-level evidence** rather than
each casting an independent unit vote.
"""
import numpy as np
import pytest
import torch

from ocean_tokenizer.batched_dfs import (RandomFourierBasis, integrate_support,
                                         dfs_omega, ConservativeResampler,
                                         N_FEATURES, LENGTH_SCALES, BASIS_SEED)


def _point_tokens(coords, noise=0.08):
    """Point supports: one quadrature node of weight 1 per token."""
    c = torch.as_tensor(coords, dtype=torch.float64)
    nodes = c[:, None, :]                       # (N,1,4)
    weights = torch.ones(c.shape[0], 1, dtype=torch.float64)
    dens = torch.full((c.shape[0],), noise, dtype=torch.float64)
    return nodes, weights, dens


def _omega(coords, noise=0.08, mask=None):
    basis = RandomFourierBasis(N_FEATURES, LENGTH_SCALES, seed=BASIS_SEED)
    nodes, weights, dens = _point_tokens(coords, noise)
    psi, lam = integrate_support(basis, nodes, weights, dens)
    if mask is None:
        mask = torch.ones(psi.shape[0], dtype=torch.bool)
    return dfs_omega(psi, lam, mask)


# --------------------------------------------------------------------------
# The basis is fixed, not learned
# --------------------------------------------------------------------------
def test_basis_is_deterministic_for_a_fixed_seed():
    a = RandomFourierBasis(N_FEATURES, LENGTH_SCALES, seed=0)
    b = RandomFourierBasis(N_FEATURES, LENGTH_SCALES, seed=0)
    x = torch.randn(5, 4, dtype=torch.float64)
    assert torch.equal(a(x), b(x))


def test_a_different_seed_gives_a_different_basis():
    a = RandomFourierBasis(N_FEATURES, LENGTH_SCALES, seed=0)
    b = RandomFourierBasis(N_FEATURES, LENGTH_SCALES, seed=1)
    x = torch.randn(5, 4, dtype=torch.float64)
    assert not torch.allclose(a(x), b(x))


def test_basis_has_no_trainable_parameters():
    b = RandomFourierBasis(N_FEATURES, LENGTH_SCALES, seed=0)
    assert sum(p.numel() for p in b.parameters()) == 0


# --------------------------------------------------------------------------
# Set-level evidence: the DFS bound
# --------------------------------------------------------------------------
def test_total_evidence_never_exceeds_the_feature_count():
    """sum omega = trace(H) <= F. Evidence is finite no matter how many
    tokens are supplied — that is what makes it *set-level*."""
    rng = np.random.default_rng(0)
    for n in (5, 50, 500):
        c = rng.uniform(0, 1, (n, 4))
        assert float(_omega(c).sum()) <= N_FEATURES + 1e-6, n


def test_every_omega_is_between_zero_and_one():
    c = np.random.default_rng(1).uniform(0, 1, (30, 4))
    w = _omega(c)
    assert float(w.min()) >= -1e-9 and float(w.max()) <= 1.0 + 1e-9


# --------------------------------------------------------------------------
# Duplicates compete instead of voting independently
# --------------------------------------------------------------------------
def test_two_identical_tokens_split_one_token_worth_of_evidence():
    c1 = np.array([[0.5, 0.5, 0.5, 0.0]])
    one = float(_omega(c1).sum())
    two = _omega(np.repeat(c1, 2, axis=0))
    assert float(two[0]) == pytest.approx(float(two[1]), rel=1e-6)
    assert float(two[0]) < one, "a duplicate must not keep full evidence"
    assert float(two.sum()) < 2 * one, "duplicates must not sum to 2x"


def test_duplicate_evidence_grows_sublinearly_with_copy_count():
    c1 = np.array([[0.5, 0.5, 0.5, 0.0]])
    totals = [float(_omega(np.repeat(c1, k, axis=0)).sum())
              for k in (1, 2, 4, 8)]
    assert all(b > a for a, b in zip(totals, totals[1:])), "should still grow"
    assert totals[-1] < 2.0 * totals[0], \
        f"8 copies gained {totals[-1] / totals[0]:.2f}x — not sublinear"


def test_well_separated_tokens_each_keep_their_own_evidence():
    """Far apart in every axis, tokens are nearly independent."""
    far = np.array([[0.0, 0.0, 0.0, 0.0],
                    [9.0, 9.0, 9.0, 90.0],
                    [-9.0, -9.0, -9.0, -90.0]])
    w = _omega(far)
    solo = float(_omega(far[:1]).sum())
    assert all(abs(float(x) - solo) < 0.05 * solo for x in w)


def test_lower_noise_density_buys_more_evidence():
    c = np.array([[0.2, 0.3, 0.4, 0.0]])
    assert float(_omega(c, noise=0.01).sum()) > float(_omega(c, noise=1.0).sum())


# --------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------
def test_masked_tokens_get_zero_evidence():
    c = np.random.default_rng(2).uniform(0, 1, (10, 4))
    m = torch.ones(10, dtype=torch.bool); m[3:7] = False
    w = _omega(c, mask=m)
    assert torch.allclose(w[3:7], torch.zeros(4, dtype=w.dtype))


def test_masked_tokens_do_not_compete_with_active_ones():
    """Masking a duplicate must restore the survivor's full evidence."""
    c = np.array([[0.5, 0.5, 0.5, 0.0]] * 2)
    m = torch.tensor([True, False])
    masked = float(_omega(c, mask=m)[0])
    alone = float(_omega(c[:1])[0])
    assert masked == pytest.approx(alone, rel=1e-6)


def test_all_masked_gives_all_zero_without_nan():
    c = np.random.default_rng(3).uniform(0, 1, (6, 4))
    w = _omega(c, mask=torch.zeros(6, dtype=torch.bool))
    assert torch.isfinite(w).all() and float(w.abs().sum()) == 0.0


# --------------------------------------------------------------------------
# Conservative transport into slots
# --------------------------------------------------------------------------
def test_resampler_conserves_total_mass():
    """Doc §2.2: sum(slot_mass) == sum(active omega) to float tolerance."""
    torch.manual_seed(0)
    r = ConservativeResampler(d_model=32, n_slots=32)
    emb = torch.randn(2, 40, 32)
    omega = torch.rand(2, 40).double()
    mask = torch.ones(2, 40, dtype=torch.bool); mask[1, 30:] = False
    _, _, slot_mass = r(emb, omega, mask)
    expect = (omega * mask).sum(dim=1)
    # float32 tolerance, not float64: the transport plan comes from a float32
    # network, so conservation is exact only to that precision. The doc asks
    # for "up to floating-point tolerance", which this is.
    assert torch.allclose(slot_mass.sum(dim=1), expect, rtol=1e-5, atol=1e-6)


def test_resampler_emits_the_requested_number_of_slots():
    torch.manual_seed(0)
    r = ConservativeResampler(d_model=32, n_slots=32)
    emb = torch.randn(3, 17, 32)
    out, out_mask, mass = r(emb, torch.rand(3, 17).double(),
                            torch.ones(3, 17, dtype=torch.bool))
    assert out.shape == (3, 32, 32) and mass.shape == (3, 32)
    assert out_mask.shape == (3, 32)


def test_uniform_control_uses_unit_masses_but_the_same_transport():
    """`uniform` is the matched mechanism control: omega_i = 1 throughout."""
    torch.manual_seed(0)
    r = ConservativeResampler(d_model=32, n_slots=32)
    emb = torch.randn(1, 12, 32)
    mask = torch.ones(1, 12, dtype=torch.bool)
    _, _, mass = r(emb, torch.ones(1, 12).double(), mask)
    assert float(mass.sum()) == pytest.approx(12.0, rel=1e-5)


def test_zero_active_tokens_conserve_zero_without_nan():
    torch.manual_seed(0)
    r = ConservativeResampler(d_model=32, n_slots=32)
    emb = torch.randn(1, 5, 32)
    mask = torch.zeros(1, 5, dtype=torch.bool)
    _, _, mass = r(emb, torch.rand(1, 5).double(), mask)
    assert torch.isfinite(mass).all() and float(mass.sum()) == 0.0
