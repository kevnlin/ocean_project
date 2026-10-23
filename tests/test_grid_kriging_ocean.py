"""The gridded state model: rasterised inputs, U-Net state, lead evolution, kriging read-out."""
import math

import numpy as np
import pytest
import torch

from ocean_tokenizer.grid_kriging_ocean import (
    GridKrigingConfig, GridKrigingModel, GridSpec, UNet, rasterize_profiles, sample_grid_features,
)
from ocean_tokenizer.kriging_attention import KrigingAttention

GRID = GridSpec(lat0=-82.5, dlat=15., lon0=7.5, dlon=15., n_lat=12, n_lon=24)
ELL = ((900.,) * 20, (1100.,) * 20)
GAMMA = ((.05,) * 20, (.1,) * 20)


def _config(**kw):
    base = dict(grid=GRID.as_tuple(), n_levels=20, state_channels=8, encoder_channels=(8, 16),
                evolution_channels=(8, 16), hidden=16, kriging="ka_flow",
                kriging_ell_km=ELL, kriging_gamma=GAMMA)
    base.update(kw)
    return GridKrigingConfig(**base)


def _model(seed=3, **kw):
    torch.manual_seed(seed)
    return GridKrigingModel(_config(**kw)).double().eval()


def _scene(lead=0, q=7, k=5, n=9, seed=11):
    g = torch.Generator().manual_seed(seed)
    rand = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
    l = 20
    lat, lon = rand(n) * 30., (rand(n) * 90.) % 360.
    coord = torch.stack([lat[:, None].expand(n, l), lon[:, None].expand(n, l),
                         torch.arange(l, dtype=torch.float64)[None].expand(n, l) * 50., torch.zeros(n, l,
                         dtype=torch.float64)], -1).reshape(-1, 4)
    valid = torch.ones(n, l, 2, dtype=torch.bool)
    valid[0, 15:] = False
    surface = rand(3, GRID.n_lat, GRID.n_lon)
    surface[:, :2] = float("nan")              # land rows
    local_valid = torch.ones(q, k, 2, dtype=torch.bool)
    local_valid[1, 2, 1] = False
    offsets = rand(q, k, 4)
    offsets[..., :2] *= 600.
    return {"obs_coord": coord, "obs_innovation": rand(n * l, 2), "obs_valid": valid.reshape(-1, 2),
            "surface_grid": surface, "lead": torch.tensor(lead), "input_month": torch.tensor(13),
            "query_coord": torch.stack([rand(q) * 30., (rand(q) * 90.) % 360., rand(q).abs() * 300., rand(q)], -1),
            "query_level": torch.randint(0, l, (q,), generator=g), "query_features": rand(q, 58),
            "query_background": torch.zeros(q, 2, dtype=torch.float64),
            "local_innovation": rand(q, k, 2), "local_valid": local_valid, "local_offsets": offsets,
            "local_features": rand(q, k, 66), "local_lat": rand(q, k) * 30., "local_lon": (rand(q, k) * 90.) % 360.}


def test_rasterize_averages_profiles_in_a_cell_and_counts_them():
    lat = torch.tensor([10., 11., -40.], dtype=torch.float64)
    lon = torch.tensor([20., 21., 300.], dtype=torch.float64)
    values = torch.zeros(3, 20, 2, dtype=torch.float64)
    values[0], values[1], values[2] = 1., 3., -2.
    valid = torch.ones(3, 20, 2, dtype=torch.bool)
    valid[1, 5, 0] = False
    grid = rasterize_profiles(lat, lon, values, valid, GRID)
    assert grid.shape == (81, GRID.n_lat, GRID.n_lon)
    iy, ix = GRID.cell(10., 20.)
    assert grid[0, iy, ix] == 2. and grid[1, iy, ix] == 2.          # level 0, T and S: mean of 1 and 3
    assert grid[10, iy, ix] == 1.                                   # level 5 T: only the valid profile
    assert grid[40 + 10, iy, ix] == 1. and grid[40, iy, ix] == 1.   # coverage flags
    assert grid[80, iy, ix] == pytest.approx(math.log1p(2.))
    jy, jx = GRID.cell(-40., 300.)
    assert grid[0, jy, jx] == -2. and grid[80, jy, jx] == pytest.approx(math.log1p(1.))
    assert grid[:, 0, 0].abs().sum() == 0


def test_sampling_is_bilinear_and_periodic_in_longitude():
    field = torch.randn(4, GRID.n_lat, GRID.n_lon, dtype=torch.float64)
    lat = torch.tensor([GRID.lat0, 7.5, 3.1, -60.2, GRID.lat0 + 11 * GRID.dlat], dtype=torch.float64)
    lon = torch.tensor([GRID.lon0, 7.5 + 15 * 3, 359.9, 0.2, 181.], dtype=torch.float64)
    out = sample_grid_features(field, lat, lon, GRID)
    fy = (lat - GRID.lat0) / GRID.dlat
    fx = ((lon - GRID.lon0) % 360) / GRID.dlon
    i0, j0 = fy.floor().long().clamp(0, GRID.n_lat - 1), fx.floor().long() % GRID.n_lon
    i1, j1 = (i0 + 1).clamp(max=GRID.n_lat - 1), (j0 + 1) % GRID.n_lon
    wy, wx = (fy - fy.floor())[:, None], (fx - fx.floor())[:, None]
    f = field.permute(1, 2, 0)
    expected = (f[i0, j0] * (1 - wy) * (1 - wx) + f[i0, j1] * (1 - wy) * wx
                + f[i1, j0] * wy * (1 - wx) + f[i1, j1] * wy * wx)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(out[0], field[:, 0, 0])


def test_unet_keeps_odd_grid_shapes_and_a_zero_head_outputs_zero():
    net = UNet(5, 3, (4, 8, 8), zero_head=True).double()
    x = torch.randn(1, 5, 13, 26, dtype=torch.float64)
    out = net(x)
    assert out.shape == (1, 3, 13, 26) and (out == 0).all()
    net = UNet(5, 3, (4, 8), cond_dim=6).double()
    assert net(x, torch.randn(1, 6, dtype=torch.float64)).shape == (1, 3, 13, 26)


def test_at_initialization_a_reconstruction_is_kriging_of_the_neighbour_anomalies():
    model, scene = _model(), _scene(lead=0)
    out = model(scene)
    reference = KrigingAttention(16, "ka_stationary", ELL, GAMMA).double()
    h = torch.zeros(len(scene["query_level"]), 16, dtype=torch.float64)
    expected = reference(h, h[:, None].expand(-1, 5, -1), scene["local_offsets"][..., :2],
                         scene["local_innovation"], scene["local_valid"], scene["query_level"])["estimate"]
    torch.testing.assert_close(out["mean"], expected)
    assert torch.isfinite(out["std"]).all() and (out["std"] > 0).all()


def test_at_initialization_a_forecast_is_the_damped_analysis_and_the_state_persists():
    model = _model()
    analysis = model(_scene(lead=0))["mean"]
    state = model.encode(_scene(lead=3))
    torch.testing.assert_close(state["query_grid"], state["neighbour_grid"])        # the evolution starts at zero
    forecast = model.decode(_scene(lead=3), state)["mean"]
    decay = torch.exp(-3. / torch.tensor(model.config.init_decay_months, dtype=torch.float64))
    torch.testing.assert_close(forecast, decay * analysis)


@pytest.mark.parametrize("lead", [0, 3])
def test_gradients_are_finite_and_reach_every_part(lead):
    model, scene = _model(), _scene(lead=lead)
    with torch.no_grad():
        # the zero-initialised mean head after a first step: the gradient reaches the encoder through it
        model.mean_head[-1].weight.normal_(0., .1)
    model.train()
    scene["local_innovation"].requires_grad_()
    scene["obs_innovation"].requires_grad_()
    out = model(scene)
    loss = (out["mean"] - .3).square().mean() + .01 * out["std"].mean()
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters()}
    assert all(g is None or torch.isfinite(g).all() for g in grads.values())
    assert grads["encoder.head.weight"].abs().sum() > 0
    assert grads["kriging.local_head.2.weight"].abs().sum() > 0
    assert scene["local_innovation"].grad.abs().sum() > 0 and scene["obs_innovation"].grad.abs().sum() > 0
    if lead:
        assert grads["evolution.head.weight"].abs().sum() > 0 and grads["log_decay_months"].abs().sum() > 0


def test_queries_decode_the_same_alone_in_chunks_and_permuted():
    model, scene = _model(), _scene(lead=1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * .02)
    state = model.encode(scene)
    full = model.decode(scene, state)["mean"]
    query_keys = [k for k in scene if k.startswith(("query_", "local_"))]
    chunk = {k: (v[2:5] if k in query_keys else v) for k, v in scene.items()}
    torch.testing.assert_close(model.decode(chunk, state)["mean"], full[2:5])
    order = torch.randperm(len(full))
    permuted = {k: (v[order] if k in query_keys else v) for k, v in scene.items()}
    torch.testing.assert_close(model.decode(permuted, state)["mean"], full[order])


def test_missing_surface_fields_and_empty_neighbourhoods_stay_finite():
    model, scene = _model(), _scene(lead=6)
    scene["surface_grid"][:] = float("nan")
    scene["local_valid"][0] = False
    out = model(scene)
    assert torch.isfinite(out["mean"]).all() and torch.isfinite(out["std"]).all()


def test_serialized_config_replays_the_model():
    model, scene = _model(), _scene(lead=1)
    config = GridKrigingConfig(**model.config.to_dict())
    assert config == model.config
    replay = GridKrigingModel(config).double().eval()
    replay.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(replay(scene)["mean"], model(scene)["mean"])


def _active(model):
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * .02)
    return model


def test_the_prior_background_starts_as_kriging_too():
    model, scene = _model(background="prior"), _scene(lead=0)
    reference = _model(background="joint")
    torch.testing.assert_close(model(scene)["mean"], reference(scene)["mean"])


def test_with_a_prior_background_the_months_profiles_reach_a_reconstruction_only_through_kriging():
    model = _active(_model(background="prior"))
    scene, changed = _scene(lead=0), _scene(lead=0)
    changed["obs_innovation"] = changed["obs_innovation"] + 3.
    torch.testing.assert_close(model(changed)["mean"], model(scene)["mean"])
    forecast, changed_forecast = _scene(lead=3), _scene(lead=3)
    changed_forecast["obs_innovation"] = changed_forecast["obs_innovation"] + 3.
    assert not torch.allclose(model(changed_forecast)["mean"], model(forecast)["mean"])


def test_a_joint_background_reads_the_months_profiles_at_lead_zero():
    model = _active(_model(background="joint"))
    scene, changed = _scene(lead=0), _scene(lead=0)
    changed["obs_innovation"] = changed["obs_innovation"] + 3.
    assert not torch.allclose(model(changed)["mean"], model(scene)["mean"])


@pytest.mark.parametrize("lead", [0, 3])
def test_a_prior_model_trains_its_prior_at_every_lead_and_its_analysis_for_forecasts(lead):
    model, scene = _active(_model(background="prior")), _scene(lead=lead)
    model.train()
    out = model(scene)
    ((out["mean"] - .3).square().mean() + .01 * out["std"].mean()).backward()
    grads = {n: p.grad for n, p in model.named_parameters()}
    assert all(g is None or torch.isfinite(g).all() for g in grads.values())
    assert grads["prior_encoder.head.weight"].abs().sum() > 0
    analysis = grads["encoder.head.weight"]
    assert (analysis is not None and analysis.abs().sum() > 0) == bool(lead)


def test_without_evolution_a_forecast_reads_the_months_own_state():
    model = _active(_model(evolution=False))
    state = model.encode(_scene(lead=3))
    torch.testing.assert_close(state["query_grid"], state["neighbour_grid"])
    assert not any(name.startswith("evolution.") for name, _ in model.named_parameters())


def test_without_lead_decay_a_forecast_starts_as_the_undamped_analysis():
    model = _model(lead_decay=False)
    torch.testing.assert_close(model(_scene(lead=3))["mean"], model(_scene(lead=0))["mean"])


def test_the_new_switches_default_to_the_trained_model():
    config = _config()
    assert config.evolution and config.lead_decay
    old = {k: v for k, v in config.to_dict().items() if k not in ("evolution", "lead_decay")}
    assert GridKrigingConfig(**old) == config
