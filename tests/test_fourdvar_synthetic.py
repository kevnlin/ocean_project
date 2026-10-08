"""Official solver pin, continuous observations and second-order training."""
from pathlib import Path

import numpy as np
import pytest
import torch

from ocean_tokenizer.fourdvar_synthetic import (
    FourDVarConfig, OfficialFourDVarNet, ScatterBilinearOperator,
    ScatterObservationCost, heldout_query_loss, make_batch,
    official_components, profiles_to_grid,
)

REPOSITORY = Path(__file__).resolve().parents[1] / "external/4dvarnet-starter"


def test_bilinear_operator_preserves_linear_field_at_continuous_position():
    height, width = 6, 12
    latitude = -90 + (torch.arange(height, dtype=torch.float64) + .5) * 180 / height
    longitude = (torch.arange(width, dtype=torch.float64) + .5) * 360 / width
    state = (2 * latitude[:, None] + 3 * longitude[None, :] + 7)[None, None]
    lat, lon = torch.tensor([-20., 42.], dtype=torch.float64), torch.tensor([40., 227.], dtype=torch.float64)
    expected = 2 * lat + 3 * lon + 7
    actual = ScatterBilinearOperator(lat, lon, height, width)(state)[0, :, 0]
    torch.testing.assert_close(actual, expected)


def test_periodic_longitude_and_polar_clamping():
    state = torch.arange(8., dtype=torch.float64).reshape(1, 1, 2, 4)
    op = ScatterBilinearOperator(torch.tensor([-90., -90., 90.]), torch.tensor([0., 360., 0.]), 2, 4)
    result = op(state)[0, :, 0]
    torch.testing.assert_close(result, torch.tensor([1.5, 1.5, 5.5], dtype=torch.float64))


def test_operator_supports_first_and_second_order_autograd():
    state = torch.randn(1, 2, 4, 6, dtype=torch.float64, requires_grad=True)
    op = ScatterBilinearOperator(torch.tensor([12., -34.]), torch.tensor([83., 211.]), 4, 6)
    assert torch.autograd.gradcheck(op, (state,))
    cost = op(state).square().mean()
    gradient, = torch.autograd.grad(cost, state, create_graph=True)
    second, = torch.autograd.grad(gradient.square().sum(), state)
    assert torch.isfinite(second).all() and second.abs().sum() > 0


def test_grid_initialization_masks_and_channel_order():
    obs = torch.tensor([[[1., 10.], [2., float("nan")]],
                        [[3., 30.], [float("nan"), 40.]]])
    grid = profiles_to_grid(obs, torch.tensor([0., 0.]), torch.tensor([10., 10.]), 4, 6)
    torch.testing.assert_close(grid[0, :, 2, 0], torch.tensor([2., 2., 20., 40.]))
    assert torch.isfinite(grid).sum() == 4


def test_cost_uses_original_profile_values_not_cell_average():
    obs = torch.tensor([[[1., 2.]], [[3., 4.]]])
    cfg = FourDVarConfig(levels=1, height=4, width=6)
    batch = make_batch(obs, torch.tensor([0., 0.]), torch.tensor([10., 10.]), cfg)
    state = torch.zeros(1, 2, 4, 6, requires_grad=True)
    assert ScatterObservationCost(2)(state, batch).item() == 7.5
    assert not batch.observed.requires_grad
    assert not batch.input.requires_grad


def test_auxiliary_dense_surface_constraints_are_explicit_and_separate():
    cfg = FourDVarConfig(levels=1, height=4, width=6, surface=True, auxiliary_weight=2.)
    obs = torch.ones(1, 1, 2)
    surface = torch.full((1, 3, 4, 6), 3.)
    batch = make_batch(obs, torch.tensor([0.]), torch.tensor([10.]), cfg, surface)
    state = torch.zeros_like(batch.input)
    assert batch.input.shape == (1, 5, 4, 6)
    assert ScatterObservationCost(2, auxiliary_weight=2.)(state, batch).item() == 19.
    with pytest.raises(ValueError, match="disagree"):
        make_batch(obs, torch.tensor([0.]), torch.tensor([10.]), cfg)


def test_heldout_loss_balances_variables_and_masks_them_independently():
    pred = torch.tensor([[1., 2.], [3., 10.]], requires_grad=True)
    target = torch.tensor([[0., 0.], [0., float("nan")]])
    loss = heldout_query_loss(pred, target)
    assert loss.item() == 4.5
    loss.backward()
    assert pred.grad[1, 1] == 0


@pytest.mark.parametrize("lat,lon", [([np.nan], [0.]), ([91.], [0.]), ([0.], [np.inf])])
def test_invalid_coordinates_fail(lat, lon):
    with pytest.raises(ValueError):
        ScatterBilinearOperator(torch.tensor(lat), torch.tensor(lon))


@pytest.mark.skipif(not REPOSITORY.exists(), reason="pinned external official source unavailable")
def test_official_classes_are_pinned_and_config_matches_author_defaults():
    components, metadata = official_components(REPOSITORY)
    assert set(components) == {"GradSolver", "ConvLstmGradModel", "BilinAEPriorCost"}
    assert metadata["class_source_sha256"] == {
        "GradSolver": "7ac0157af9a312b2a26ef4e677a97a93a4888cbcd35c22bf2139f3d5db883afe",
        "ConvLstmGradModel": "2acee443bf27d8bda58cd2060356fa8c93db4086692e01da513768a8e3212c44",
        "BilinAEPriorCost": "275d1503a03286d483ac329931a6f480755c74177cbfc08620ebad52f2d1019e"}
    c = FourDVarConfig()
    assert (c.n_step, c.lr_grad, c.prior_hidden, c.gradient_hidden, c.prior_downsampling) == (10, 1000., 32, 48, 2)
    assert not c.bilin_quad and c.channels == 40


@pytest.mark.skipif(not REPOSITORY.exists(), reason="pinned external official source unavailable")
def test_unrolled_official_solver_trains_through_state_gradients():
    torch.manual_seed(123)
    cfg = FourDVarConfig(levels=2, height=6, width=8, n_step=2, lr_grad=.2,
                        prior_hidden=3, gradient_hidden=4, dropout=.1)
    model = OfficialFourDVarNet(REPOSITORY, cfg)
    observed = torch.randn(6, 2, 2)
    lat, lon = torch.linspace(-60., 60., 6), torch.linspace(20., 300., 6)
    batch = make_batch(observed, lat, lon, cfg)
    state = model(batch)
    prediction = model.query(state, torch.tensor([-10., 12.]), torch.tensor([78., 229.]), torch.tensor([0, 1]))[0]
    loss = heldout_query_loss(prediction, torch.zeros_like(prediction))
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.solver.grad_mod.gates.weight.grad.abs().sum() > 0
    assert model.solver.prior_cost.conv_in.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        evaluated = model(batch)
    assert torch.isfinite(evaluated).all()
    assert evaluated.shape == state.shape


@pytest.mark.skipif(not REPOSITORY.exists(), reason="pinned external official source unavailable")
def test_queries_are_chunk_and_permutation_invariant():
    cfg = FourDVarConfig(levels=2, height=4, width=6, n_step=1, prior_hidden=2, gradient_hidden=2)
    model = OfficialFourDVarNet(REPOSITORY, cfg)
    state = torch.randn(1, 4, 4, 6)
    lat, lon, level = torch.tensor([-50., 20., 60.]), torch.tensor([21., 212., 300.]), torch.tensor([0, 1, 0])
    full = model.query(state, lat, lon, level)
    split = torch.cat([model.query(state, lat[i:i+1], lon[i:i+1], level[i:i+1]) for i in range(3)], dim=1)
    torch.testing.assert_close(full, split)
    permutation = torch.tensor([2, 0, 1])
    torch.testing.assert_close(full[:, permutation], model.query(state, lat[permutation], lon[permutation], level[permutation]))


@pytest.mark.skipif(not REPOSITORY.exists(), reason="pinned external official source unavailable")
def test_complete_optimizer_scheduler_and_rng_continuation_replays_training(tmp_path):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "experiments/synthetic/48_official_4dvarnet.py"
    spec = importlib.util.spec_from_file_location("official_training_state", path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    cfg = FourDVarConfig(levels=2, height=6, width=8, n_step=2, lr_grad=.2,
                        prior_hidden=3, gradient_hidden=4, dropout=.1)
    observed = torch.arange(48, dtype=torch.float32).reshape(12, 2, 2) / 40.
    latitude, longitude = torch.linspace(-60., 60., 12), torch.linspace(20., 300., 12)

    def initialize():
        torch.manual_seed(12); np.random.seed(41)
        model = OfficialFourDVarNet(REPOSITORY, cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .9 ** step)
        return model, optimizer, scheduler, np.random.default_rng(88)

    def step(model, optimizer, scheduler, rng):
        source = rng.choice(12, 6, replace=False)
        batch = make_batch(observed[source], latitude[source], longitude[source], cfg)
        state = model(batch)
        pred = model.query(state, torch.rand(3) * 100 - 50, torch.rand(3) * 300,
                           rng.integers(2, size=3))[0]
        target = torch.randn_like(pred) + float(np.random.random())
        optimizer.zero_grad(set_to_none=True)
        loss = heldout_query_loss(pred, target)
        loss.backward(); optimizer.step(); scheduler.step()
        return loss.detach()

    full, full_opt, full_sched, full_rng = initialize()
    full_losses = [step(full, full_opt, full_sched, full_rng) for _ in range(4)]
    full_random = trainer.random_state(full_rng)
    first, first_opt, first_sched, first_rng = initialize()
    for _ in range(2):
        step(first, first_opt, first_sched, first_rng)
    checkpoint = {"model": trainer.cpu_state(first), "optimizer": first_opt.state_dict(),
                  "scheduler": first_sched.state_dict(), "random": trainer.random_state(first_rng)}
    destination = tmp_path / "continuation.pt"
    trainer.atomic_save(checkpoint, destination)
    loaded = torch.load(destination, weights_only=True)
    resumed, resumed_opt, resumed_sched, resumed_rng = initialize()
    resumed.load_state_dict(loaded["model"])
    resumed_opt.load_state_dict(loaded["optimizer"])
    resumed_sched.load_state_dict(loaded["scheduler"])
    trainer.restore_random_state(resumed_rng, loaded["random"])
    resumed_losses = [step(resumed, resumed_opt, resumed_sched, resumed_rng) for _ in range(2)]
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    for original, replayed in zip(full_losses[2:], resumed_losses):
        torch.testing.assert_close(original, replayed, rtol=0, atol=0)
    assert full_sched.state_dict() == resumed_sched.state_dict()
    for key, value in full_opt.state_dict()["state"].items():
        for name, tensor in value.items():
            torch.testing.assert_close(tensor, resumed_opt.state_dict()["state"][key][name], rtol=0, atol=0)
    resumed_random = trainer.random_state(resumed_rng)
    assert full_random["numpy_generator"] == resumed_random["numpy_generator"]
    torch.testing.assert_close(full_random["torch_cpu"], resumed_random["torch_cpu"], rtol=0, atol=0)
    torch.testing.assert_close(full_random["numpy_global"]["state"], resumed_random["numpy_global"]["state"], rtol=0, atol=0)
