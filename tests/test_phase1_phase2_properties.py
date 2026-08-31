"""Phase 1 / Phase 2 properties, asserted rather than assumed.

work_plan.md Phase 2: *"Token-count invariance across layouts must be asserted
by test, not assumed."*  The same applies to the two Phase 1 mechanisms, whose
whole purpose is to make specific scenarios representable:

* vertical quadrature  -> a vertical resampling that conserves total thickness
                          conserves total evidence (the 2-dbar vs 10-dbar case)
* provenance noise     -> tokens sharing a provenance group collapse along
                          n_eff = k / (1 + (k-1) rho)  (the dual-stream float)

These run on constructed geometry, so they need no data and no training.
"""
import numpy as np
import torch

from ocean_tokenizer import batched_dfs as B
from ocean_tokenizer.batched_dfs import (RandomFourierBasis, integrate_support,
                                         dfs_omega, vertical_quadrature, n_eff,
                                         whiten_provenance)

AREA = 7854.0
NOISE_V = 7854.0 * 100.0


def _basis(p=256):
    return RandomFourierBasis(p, B.LENGTH_SCALES_KM, seed=B.BASIS_SEED)


def _column_evidence(nlev, top=0.0, bot=800.0, n_nodes=3):
    """Total evidence of one water column cut into ``nlev`` layers."""
    edges = torch.linspace(top, bot, nlev + 1, dtype=torch.float64)
    zc, dz = 0.5 * (edges[1:] + edges[:-1]), edges[1:] - edges[:-1]
    zq, wq = vertical_quadrature(zc, dz, n_nodes)
    nodes = torch.stack([torch.full_like(zq, 1000.0),
                         torch.full_like(zq, 1000.0), zq,
                         torch.zeros_like(zq)], dim=-1)
    psi, lam = integrate_support(_basis(), nodes, wq * AREA,
                                 torch.full((nlev,), NOISE_V, dtype=torch.float64))
    w = dfs_omega(psi, lam, torch.ones(nlev, dtype=torch.bool))
    return float(w.sum()), float((wq * AREA).sum())


# --------------------------------------------------------------------------
# Phase 1a — vertical support
# --------------------------------------------------------------------------
def test_vertical_quadrature_weights_sum_to_layer_thickness():
    """The rule integrates, it does not average: a thicker layer carries more."""
    dz = torch.tensor([10.0, 50.0, 200.0], dtype=torch.float64)
    zc = torch.tensor([5.0, 100.0, 400.0], dtype=torch.float64)
    _, w = vertical_quadrature(zc, dz, n_nodes=4)
    assert torch.allclose(w.sum(dim=1), dz, rtol=1e-12)


def test_vertical_resampling_conserves_support_exactly():
    """Total column support cannot depend on how finely the column is cut."""
    supports = [_column_evidence(n)[1] for n in (8, 16, 40, 80)]
    assert max(supports) - min(supports) < 1e-6 * supports[0]


def test_vertical_resampling_approximately_conserves_evidence():
    """The 2-dbar vs 10-dbar scenario: 10x the levels, same water, same evidence.

    Before Phase 1 a level was a point with zero vertical extent, so this was
    not merely violated but unrepresentable — thickness never entered the
    measure at all.  A few percent of drift remains because a finite quadrature
    of a continuous integral is not exact.
    """
    ev = [_column_evidence(n)[0] for n in (8, 16, 40, 80)]
    spread = (max(ev) - min(ev)) / min(ev)
    assert spread < 0.05, f"evidence varies {spread:.1%} with vertical cutting"


def test_midpoint_rule_is_the_one_node_special_case():
    """n_nodes=1 must reproduce the pre-Phase-1 behaviour scaled by thickness."""
    z = torch.tensor([100.0, 300.0], dtype=torch.float64)
    dz = torch.tensor([50.0, 80.0], dtype=torch.float64)
    zq, wq = vertical_quadrature(z, dz, n_nodes=1)
    assert torch.allclose(zq[:, 0], z) and torch.allclose(wq[:, 0], dz)


# --------------------------------------------------------------------------
# Phase 1b — provenance noise correlation
# --------------------------------------------------------------------------
def _group_mass(k, rho):
    nodes = torch.tensor([[[1000.0, 1000.0, 300.0, 0.0]]],
                         dtype=torch.float64).repeat(k, 1, 1)
    psi, lam = integrate_support(_basis(), nodes,
                                 torch.full((k, 1), AREA, dtype=torch.float64),
                                 torch.full((k,), AREA, dtype=torch.float64))
    w = dfs_omega(psi, lam, torch.ones(k, dtype=torch.bool),
                  torch.zeros(k, dtype=torch.long), rho)
    return float(w.sum())


def test_rho_zero_reproduces_diagonal_whitening():
    """The correlated path must be a strict generalisation, not a replacement."""
    psi = torch.randn(6, 16, dtype=torch.float64)
    lam = torch.rand(6, dtype=torch.float64) + 0.5
    prov = torch.tensor([0, 0, 1, 1, 2, 2])
    assert torch.allclose(whiten_provenance(psi, lam, prov, 0.0),
                          psi / lam.sqrt()[:, None])


def test_provenance_group_mass_matches_n_eff_theory():
    """group(k) = n*s/(1+n*s) with n = n_eff(k, rho) — Phase 1 exit criterion."""
    s = 1.0                       # support == noise area, so s_token = 1
    for rho in (0.0, 0.5, 0.9, 0.99):
        for k in (2, 4, 8):
            ne = n_eff(k, rho)
            pred = ne * s / (1.0 + ne * s)
            got = _group_mass(k, rho)
            assert abs(got - pred) / pred < 0.05, (rho, k, got, pred)


def test_high_correlation_collapses_duplicates_toward_one():
    """The paper's opening example: two streams of one float are one float."""
    solo = _group_mass(1, 0.99)
    pair = _group_mass(2, 0.99)
    assert pair / solo < 1.05, f"dual-stream ratio {pair/solo:.3f} did not collapse"


def test_independent_tokens_do_not_collapse():
    """Positive control: without shared provenance, k tokens must still count.

    Guards the failure mode where 'suppresses redundancy' is indistinguishable
    from 'ignores the stream' — both look flat without this contrast.
    """
    far = torch.tensor([[[1000.0, 1000.0, 300.0, 0.0]],
                        [[3500.0, 2400.0, 800.0, 0.0]]], dtype=torch.float64)
    psi, lam = integrate_support(_basis(), far,
                                 torch.full((2, 1), AREA, dtype=torch.float64),
                                 torch.full((2,), AREA, dtype=torch.float64))
    solo = dfs_omega(psi[:1], lam[:1], torch.ones(1, dtype=torch.bool))
    both = dfs_omega(psi, lam, torch.ones(2, dtype=torch.bool),
                     torch.tensor([0, 1]), 0.9)
    assert float(both.sum()) > 1.5 * float(solo.sum())


# --------------------------------------------------------------------------
# Phase 2 — token-count invariance
# --------------------------------------------------------------------------
def test_total_evidence_tracks_support_not_token_count():
    """Split one patch into 4 sub-patches of a quarter the area.

    Same field, same total support, 4x the tokens.  Evidence must follow the
    support.  This is the property that makes token layout a free choice rather
    than a way to buy influence.
    """
    cx, cy, area = 2000.0, 1400.0, 4 * 4 * B.cell_area_km2()
    coarse = torch.tensor([[[cx, cy, 0.0, 0.0]]], dtype=torch.float64)
    psi_c, lam_c = integrate_support(
        _basis(), coarse, torch.tensor([[area]], dtype=torch.float64),
        torch.tensor([area], dtype=torch.float64))
    ev_c = float(dfs_omega(psi_c, lam_c, torch.ones(1, dtype=torch.bool)).sum())

    dx = 2.0 * B.BOX_EXTENT[0] / (B.GRID_NX - 1)
    dy = 2.0 * B.BOX_EXTENT[1] / (B.GRID_NY - 1)
    fine = torch.tensor([[[cx + sx * dx, cy + sy * dy, 0.0, 0.0]]
                         for sx in (-0.5, 0.5) for sy in (-0.5, 0.5)],
                        dtype=torch.float64)
    psi_f, lam_f = integrate_support(
        _basis(), fine, torch.full((4, 1), area / 4, dtype=torch.float64),
        torch.full((4,), area, dtype=torch.float64))
    ev_f = float(dfs_omega(psi_f, lam_f, torch.ones(4, dtype=torch.bool)).sum())
    assert abs(ev_f - ev_c) / ev_c < 0.10, (ev_c, ev_f)


def test_clustered_sampling_carries_less_evidence_than_uniform():
    """Equal token count, different geometry: clustering is redundancy."""
    rng = np.random.default_rng(0)
    n = 40
    uni = torch.stack([
        torch.as_tensor(rng.uniform(0, B.BOX_EXTENT[0], n)),
        torch.as_tensor(rng.uniform(0, B.BOX_EXTENT[1], n)),
        torch.zeros(n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64)
    ], dim=-1)[:, None, :]
    clu = torch.stack([
        torch.as_tensor(rng.normal(2000.0, 20.0, n)),
        torch.as_tensor(rng.normal(1400.0, 20.0, n)),
        torch.zeros(n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64)
    ], dim=-1)[:, None, :]
    out = []
    for c in (uni, clu):
        psi, lam = integrate_support(
            _basis(), c, torch.full((n, 1), AREA, dtype=torch.float64),
            torch.full((n,), AREA, dtype=torch.float64))
        out.append(float(dfs_omega(psi, lam, torch.ones(n, dtype=torch.bool)).sum()))
    assert out[1] < 0.8 * out[0], f"clustered {out[1]:.3f} vs uniform {out[0]:.3f}"
