"""The official 4DVarNet behind the gridded-model interface: official paths, scene inputs, read-out."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ocean_tokenizer.fourdvar_grid import FourDVarGridModel
from ocean_tokenizer.fourdvar_synthetic import FourDVarConfig

REPOSITORY = Path(__file__).resolve().parents[1] / "external/4dvarnet-starter"
CONFIG = FourDVarConfig(levels=3, height=8, width=16, n_step=3, prior_hidden=4, gradient_hidden=5, surface=True)
pytestmark = pytest.mark.skipif(not (REPOSITORY / "src/models.py").exists(), reason="pinned 4dvarnet-starter absent")


def _scene(seed=0, lead=0):
    g = torch.Generator().manual_seed(seed)
    n, levels = 5, CONFIG.levels
    lat = torch.tensor([-60., -10., 0., 33., 71.])
    lon = torch.tensor([5., 100., 359., 181., 270.])
    coord = torch.stack([lat[:, None].expand(n, levels), lon[:, None].expand(n, levels),
                         torch.arange(levels, dtype=torch.float32)[None].expand(n, levels),
                         torch.zeros(n, levels)], -1).reshape(-1, 4)
    values = torch.randn(n * levels, 2, generator=g)
    valid = torch.ones(n * levels, 2, dtype=torch.bool)
    valid[4, 1] = False                       # profile 1, level 1: no salinity
    surface = torch.randn(3, CONFIG.height, CONFIG.width, generator=g)
    surface[:, :2] = float("nan")             # "land"
    return {"obs_coord": coord, "obs_innovation": torch.where(valid, values, 0.), "obs_valid": valid,
            "surface_grid": surface, "lead": torch.tensor(lead),
            "query_coord": torch.tensor([[12., 40., 0., 0.], [-33., 300., 0., 0.], [0., 359.5, 0., 0.]]),
            "query_level": torch.tensor([0, 2, 1])}


def _model(output="solver", config=CONFIG):
    torch.manual_seed(3)
    return FourDVarGridModel(REPOSITORY, config, output)


def _double(scene):
    return {k: v.double() if v.is_floating_point() else v for k, v in scene.items()}


def test_scene_inputs_enter_the_official_batch():
    model = _model()
    scene = _scene()
    batch = model.batch(scene)
    assert batch.input.shape == (1, 2 * CONFIG.levels + 3, CONFIG.height, CONFIG.width)
    # observed columns are T at every level, then S; the missing salinity is NaN, not the zero filler
    observed = batch.observed
    assert observed.shape == (5, 2 * CONFIG.levels)
    torch.testing.assert_close(observed[0, :CONFIG.levels], scene["obs_innovation"][:3, 0])
    torch.testing.assert_close(observed[0, CONFIG.levels:], scene["obs_innovation"][:3, 1])
    assert torch.isnan(observed[1, CONFIG.levels + 1]) and torch.isfinite(observed).sum() == 5 * 6 - 1
    # the surface grids are the last three state channels, NaN where missing
    torch.testing.assert_close(batch.input[0, -3:], scene["surface_grid"], equal_nan=True)


def test_evaluation_output_is_the_official_solver_output():
    model = _model("projection").eval()
    batch = model.batch(_scene())
    solver, projection = model.solve(batch)
    official = model.net(batch).detach()
    torch.testing.assert_close(projection, official, rtol=0, atol=0)
    assert not torch.equal(solver, projection)
    state = model.encode(_scene())
    torch.testing.assert_close(state["state"], official, rtol=0, atol=0)
    torch.testing.assert_close(state["solver"], solver, rtol=0, atol=0)


def test_training_output_is_the_state_before_the_projection():
    model = _model(config=replace(CONFIG, dropout=0.))
    scene = _scene()
    trained = model.train().encode(scene)
    solver, _ = model.eval().solve(model.batch(scene))
    torch.testing.assert_close(trained["state"].detach(), solver)
    assert trained["prior_cost"].requires_grad and trained["prior_cost"] > 0


def test_decode_reads_each_query_level_and_persists_the_state_across_leads():
    model = _model().eval()
    h, w = CONFIG.height, CONFIG.width
    lat = -90 + (torch.arange(h, dtype=torch.float64) + .5) * 180 / h
    lon = (torch.arange(w, dtype=torch.float64) + .5) * 360 / w
    # channel c holds the linear field c + lat/10 (T channels 0-2, S channels 3-5)
    field = (torch.arange(2 * CONFIG.levels + 3, dtype=torch.float64)[:, None, None]
             + lat[None, :, None] / 10 + 0 * lon[None, None, :])[None]
    state = {"state": field, "prior_cost": field.new_zeros(())}
    scene = _scene()
    out = model.decode(_double(scene), state)["mean"]
    qlat, level = scene["query_coord"][:, 0].double(), scene["query_level"]
    expected = torch.stack([level + qlat / 10, level + CONFIG.levels + qlat / 10], -1)
    torch.testing.assert_close(out, expected)
    later = model.decode(_double(_scene(lead=6)), state)["mean"]
    torch.testing.assert_close(later, out)


def test_official_objective_trains_the_solver_and_the_prior():
    model = _model().train()
    scene = _double(_scene())
    state = model.encode(scene)
    out = model.decode(scene, state)
    assert torch.isnan(out["std"]).all()
    loss = 50 * out["mean"].square().mean() + out["aux_loss"]
    loss.backward()
    for name in ("grad_mod", "prior_cost"):
        grads = [p.grad for p in getattr(model.net.solver, name).parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0, name


def test_unknown_output_and_missing_surface_are_refused():
    with pytest.raises(ValueError, match="output"):
        FourDVarGridModel(REPOSITORY, CONFIG, "mean")
    with pytest.raises(ValueError, match="surface"):
        FourDVarGridModel(REPOSITORY, replace(CONFIG, surface=False))
