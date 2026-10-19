"""Softmax read-out of the gridded model: the attention baseline that kriging attention replaces."""
import math

import pytest
import torch

from ocean_tokenizer.grid_kriging_ocean import GridKrigingConfig, GridKrigingModel, SoftmaxReadout
from test_grid_kriging_ocean import ELL, GAMMA, GRID, _config, _scene


def _readout():
    torch.manual_seed(4)
    return SoftmaxReadout(16, ELL).double()


def _inputs(q=6, k=5):
    g = torch.Generator().manual_seed(9)
    h = torch.randn(q, 16, generator=g, dtype=torch.float64)
    local = torch.randn(q, k, 16, generator=g, dtype=torch.float64)
    offsets = torch.randn(q, k, 2, generator=g, dtype=torch.float64) * 400.
    innovation = torch.randn(q, k, 2, generator=g, dtype=torch.float64)
    valid = torch.ones(q, k, 2, dtype=torch.bool)
    return h, local, offsets, innovation, valid, torch.randint(0, 20, (q,), generator=g)


def test_softmax_weights_are_a_positive_average_against_a_null_key():
    readout = _readout()
    h, local, offsets, innovation, valid, level = _inputs()
    valid[2, 1] = False
    out = readout(h, local, offsets, innovation, valid, level)
    w = out["weights"]
    assert (w >= 0).all() and (w.sum(-1) <= 1 + 1e-12).all()
    assert (w[2, :, 1] == 0).all()
    torch.testing.assert_close(out["estimate"], (w * innovation.transpose(1, 2)).sum(-1))


def test_a_duplicated_neighbour_votes_twice_under_softmax():
    readout = _readout()
    h, local, _, _, _, level = _inputs(q=1, k=2)
    one = readout(h, local[:, :1], torch.tensor([[[150., 0.]]], dtype=torch.float64),
                  torch.ones(1, 1, 2, dtype=torch.float64), torch.ones(1, 1, 2, dtype=torch.bool), level)
    two = readout(h, local[:, :1].expand(1, 2, 16), torch.tensor([[[150., 0.], [150., 0.]]], dtype=torch.float64),
                  torch.ones(1, 2, 2, dtype=torch.float64), torch.ones(1, 2, 2, dtype=torch.bool), level)
    assert (two["weights"].sum(-1) > one["weights"].sum(-1)).all()


def test_the_lead_decay_scales_the_softmax_estimate():
    readout = _readout()
    inputs = _inputs()
    decay = torch.rand(6, 2, dtype=torch.float64)
    torch.testing.assert_close(readout(*inputs, decay=decay)["estimate"], decay * readout(*inputs)["estimate"])


@pytest.mark.parametrize("lead", [0, 3])
def test_a_softmax_model_trains_with_finite_gradients(lead):
    torch.manual_seed(3)
    model = GridKrigingModel(_config(kriging="softmax")).double()
    with torch.no_grad():
        model.mean_head[-1].weight.normal_(0., .1)
    scene = _scene(lead=lead)
    out = model(scene)
    assert torch.isfinite(out["mean"]).all() and torch.isfinite(out["std"]).all()
    ((out["mean"] - .3).square().mean() + .01 * out["std"].mean()).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.kriging.scorer[-1].weight.grad.abs().sum() > 0
    config = GridKrigingConfig(**model.config.to_dict())
    assert config == model.config
