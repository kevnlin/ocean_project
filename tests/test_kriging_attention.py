"""Kriging attention: the covariance-aware weights of the local evidence path."""
import math

import numpy as np
import pytest
import torch

from ocean_tokenizer.kriging_attention import (
    KrigingAttention, length_scale_matrices, ps_covariance, without_oi_anchor,
)

N_LEVELS = 20


def _init(ell_t=250., ell_s=300., gamma_t=.01, gamma_s=.03):
    ell = [[ell_t] * N_LEVELS, [ell_s] * N_LEVELS]
    gamma = [[gamma_t] * N_LEVELS, [gamma_s] * N_LEVELS]
    return ell, gamma


def _inputs(q=6, k=9, width=16, seed=3):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(q, width, generator=g, dtype=torch.float64)
    local = torch.randn(q, k, width, generator=g, dtype=torch.float64)
    offsets = torch.randn(q, k, 2, generator=g, dtype=torch.float64) * 300.
    innovation = torch.randn(q, k, 2, generator=g, dtype=torch.float64)
    valid = torch.ones(q, k, 2, dtype=torch.bool)
    level = torch.randint(0, N_LEVELS, (q,), generator=g)
    return h, local, offsets, innovation, valid, level


def _oi(offsets, values, ell, gamma):
    """The frozen OI's solve in the plane: (C_oo + gamma I)^-1 C_go, Gaussian C."""
    xy = np.asarray(offsets, dtype=np.float64)
    d_go = np.linalg.norm(xy, axis=-1)
    d_oo = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    c_go = np.exp(-.5 * (d_go / ell) ** 2)
    c_oo = np.exp(-.5 * (d_oo / ell) ** 2) + gamma * np.eye(len(xy))
    w = np.linalg.solve(c_oo, c_go)
    return float(w @ np.asarray(values, dtype=np.float64)), w


def test_isotropic_equal_length_scales_give_the_oi_gaussian():
    xy = torch.tensor([[0., 0.], [120., -40.], [-300., 250.]], dtype=torch.float64)
    shape = torch.zeros(3, 3, dtype=torch.float64)
    sigma = length_scale_matrices(torch.tensor(250., dtype=torch.float64), shape)
    c = ps_covariance(xy, sigma, xy, sigma)
    d = torch.cdist(xy, xy)
    torch.testing.assert_close(c, torch.exp(-.5 * (d / 250.) ** 2))


def test_nonstationary_anisotropic_covariance_is_positive_definite():
    g = torch.Generator().manual_seed(11)
    xy = torch.randn(40, 2, generator=g, dtype=torch.float64) * 400.
    shape = torch.randn(40, 3, generator=g, dtype=torch.float64)
    sigma = length_scale_matrices(torch.tensor(250., dtype=torch.float64), shape)
    c = ps_covariance(xy, sigma, xy, sigma)
    torch.testing.assert_close(c, c.T)
    assert torch.linalg.eigvalsh(c).min() > -1e-9
    torch.testing.assert_close(torch.diagonal(c), torch.ones(40, dtype=torch.float64))


# ka_oi keeps the OI's numbers as float64 buffers; learned parameters are float32
@pytest.mark.parametrize("mode, tolerance, floor", [("ka_oi", 1e-9, 1e-12), ("ka_stationary", 1e-5, 1e-6)])
def test_oi_settings_reproduce_the_explicit_oi_solve(mode, tolerance, floor):
    ell, gamma = _init()
    attention = KrigingAttention(16, mode, ell, gamma).double()
    h, local, offsets, innovation, valid, level = _inputs()
    out = attention(h, local, offsets, innovation, valid, level)
    for i in range(len(level)):
        for j, (length, ratio) in enumerate(((250., .01), (300., .03))):
            expected, weights = _oi(offsets[i], innovation[i, :, j], length, ratio)
            assert out["estimate"][i, j].item() == pytest.approx(expected, rel=tolerance, abs=floor)
            np.testing.assert_allclose(out["weights"][i, j].detach().numpy(), weights, rtol=tolerance, atol=floor)


def test_invalid_neighbours_are_ignored_whatever_their_values():
    ell, gamma = _init()
    attention = KrigingAttention(16, "ka_oi", ell, gamma).double()
    h, local, offsets, innovation, valid, level = _inputs()
    valid[:, 4] = False
    valid[2, 1, 1] = False
    corrupted = innovation.clone()
    corrupted[:, 4] = 1e6
    corrupted[2, 1, 1] = -1e6
    out = attention(h, local, offsets, corrupted, valid, level)
    assert torch.isfinite(out["estimate"]).all()
    assert (out["weights"][:, :, 4] == 0).all() and out["weights"][2, 1, 1] == 0
    keep = [n for n in range(offsets.shape[1]) if n != 4]
    expected, _ = _oi(offsets[0, keep], innovation[0, keep, 0], 250., .01)
    assert out["estimate"][0, 0].item() == pytest.approx(expected, rel=1e-9)
    keep_s = [n for n in keep if n != 1]
    expected, _ = _oi(offsets[2, keep_s], innovation[2, keep_s, 1], 300., .03)
    assert out["estimate"][2, 1].item() == pytest.approx(expected, rel=1e-9)


def test_a_duplicated_neighbour_splits_its_weight_instead_of_voting_twice():
    ell, gamma = _init(ell_t=250., gamma_t=.1)
    attention = KrigingAttention(16, "ka_oi", ell, gamma).double()
    h, local, _, _, _, level = _inputs(q=1, k=2)
    offsets = torch.tensor([[[150., 0.], [150., 0.]]], dtype=torch.float64)
    innovation = torch.ones(1, 2, 2, dtype=torch.float64)
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    out = attention(h, local, offsets, innovation, valid, level)
    c = math.exp(-.5 * (150. / 250.) ** 2)
    torch.testing.assert_close(out["weights"][0, 0], torch.full((2,), c / (2 + .1), dtype=torch.float64))
    # two copies with error ratio gamma act as one observation with gamma / 2
    assert out["estimate"][0, 0].item() == pytest.approx(c / (1 + .1 / 2), rel=1e-12)


def test_a_screened_neighbour_behind_a_closer_one_gets_a_negative_weight():
    ell, gamma = _init(ell_t=250., gamma_t=.01)
    attention = KrigingAttention(16, "ka_oi", ell, gamma).double()
    h, local, _, _, _, level = _inputs(q=1, k=2)
    offsets = torch.tensor([[[100., 0.], [300., 0.]]], dtype=torch.float64)
    innovation = torch.ones(1, 2, 2, dtype=torch.float64)
    out = attention(h, local, offsets, innovation, torch.ones(1, 2, 2, dtype=torch.bool), level)
    near, far = out["weights"][0, 0]
    assert near > 1. and far < 0.


def test_flow_mode_starts_as_the_stationary_kernel():
    ell, gamma = _init()
    torch.manual_seed(5)
    flow = KrigingAttention(16, "ka_flow", ell, gamma).double()
    stationary = KrigingAttention(16, "ka_stationary", ell, gamma).double()
    inputs = _inputs()
    a, b = flow(*inputs), stationary(*inputs)
    torch.testing.assert_close(a["estimate"], b["estimate"])
    torch.testing.assert_close(a["variance"], b["variance"])


def test_flow_mode_heads_and_inputs_receive_finite_gradients():
    ell, gamma = _init()
    torch.manual_seed(6)
    flow = KrigingAttention(16, "ka_flow", ell, gamma).double()
    h, local, offsets, innovation, valid, level = _inputs()
    valid[1, 3, 0] = False
    h.requires_grad_(); local.requires_grad_(); innovation.requires_grad_()
    out = flow(h, local, offsets, innovation, valid, level)
    loss = (out["estimate"] - .3).square().mean() + out["variance"].mean()
    loss.backward()
    for head in (flow.query_head, flow.local_head):
        grad = head[-1].weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert torch.isfinite(innovation.grad).all() and innovation.grad.abs().sum() > 0
    assert innovation.grad[1, 3, 0] == 0


def test_posterior_variance_is_nonnegative_and_shrinks_near_observations():
    ell, gamma = _init()
    attention = KrigingAttention(16, "ka_oi", ell, gamma).double()
    h, local, offsets, innovation, valid, level = _inputs()
    # every neighbour at least ~4000 km away: no information, the prior variance remains
    away = offsets + 5000. * offsets / offsets.norm(dim=-1, keepdim=True)
    far = attention(h, local, away, innovation, valid, level)["variance"]
    near = attention(h, local, offsets * .05, innovation, valid, level)["variance"]
    assert (near >= 0).all() and (far >= 0).all()
    assert (near < far).all()
    torch.testing.assert_close(far, torch.ones_like(far), atol=1e-3, rtol=0)


def test_without_oi_anchor_replaces_only_the_baseline():
    scene = {"baseline": torch.tensor([[.5, -.2]]), "query_background": torch.zeros(1, 2),
             "local_innovation": torch.ones(1, 3, 2)}
    out = without_oi_anchor(scene)
    assert torch.equal(out["baseline"], scene["query_background"])
    assert out["local_innovation"] is scene["local_innovation"]
    assert torch.equal(scene["baseline"], torch.tensor([[.5, -.2]]))
