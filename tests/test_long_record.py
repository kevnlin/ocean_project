"""Training window, query-scaled T/S alignment and long-record scenes; no CESM2 data needed."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ocean_tokenizer import long_record_scenes as LR
from ocean_tokenizer import query_scaled_innovation as QS
from ocean_tokenizer import radius_scale_scenes as R
from ocean_tokenizer.innovation_ocean import InnovationOceanModel
from test_innovation_ocean import _model as innovation_model, _scene as innovation_scene
from test_synthetic_latent_experiment import prepared

SOURCE = np.arange(6)
ROWS, LEVELS = np.array([6, 7, 8, 7, 6]), np.array([0, 1, 2, 0, 2])


def test_the_window_is_the_last_years_before_2004():
    available = set(range(-432, 72))
    assert LR.window_months(available, 4) == list(range(0, 48))
    assert LR.window_months(available, 10) == list(range(-72, 48))
    assert LR.window_months(available, 40) == list(range(-432, 48))


def test_training_months_follow_the_window_and_need_inputs_and_queries():
    data = LR.LongRecordScenes.__new__(LR.LongRecordScenes)
    data._months_cache, data.train_years = {}, 10
    data.months_available = set(range(-432, 72)) - {-5}
    data.c = SimpleNamespace(month=lambda m, float_split=None: np.arange(0 if m == -7 else 3))
    assert data.months("train") == [m for m in range(-72, 48) if m not in (-5, -7)]


def models(recipe, shift=0.):
    """The innovation model of test_innovation_ocean and the query-scaled one, same weights."""
    reference = innovation_model(recipe, active=True)
    if shift and reference.alignment is not None:
        with torch.no_grad():
            reference.alignment.shift_head[-1].bias.fill_(math.atanh(shift))
    scaled = QS.QueryScaledInnovationModel(reference.config).double().eval()
    scaled.load_state_dict(reference.state_dict())
    return reference, scaled


def test_same_modules_and_initial_values_as_the_innovation_model():
    config = innovation_model("profile_shared").config
    torch.manual_seed(5)
    reference = InnovationOceanModel(config)
    torch.manual_seed(5)
    scaled = QS.QueryScaledInnovationModel(config)
    assert type(scaled.alignment) is QS.QueryScaledAlignment
    expected, actual = reference.state_dict(), scaled.state_dict()
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name


@pytest.mark.parametrize("shift", [0., .4])
def test_a_unit_scale_is_the_innovation_model_exactly(shift):
    reference, scaled = models("profile_shared", shift)
    scene = innovation_scene()
    unit = {**scene, "local_query_scale": torch.ones_like(scene["baseline"])}
    with torch.no_grad():
        assert torch.equal(scaled(unit)["mean"], reference(scene)["mean"])


@pytest.mark.parametrize("shift", [0., .4])
def test_aligned_innovations_are_divided_by_the_query_scale_once(shift):
    reference, scaled = models("profile_shared", shift)
    scene = innovation_scene()
    scale = torch.rand(scene["baseline"].shape, generator=torch.Generator().manual_seed(4),
                       dtype=torch.float64) * 1.5 + .5
    h = torch.randn(len(scale), reference.config.width, generator=torch.Generator().manual_seed(6),
                    dtype=torch.float64)
    with torch.no_grad():
        plain, diagnostics = reference.alignment(scene, h)
        divided, _ = scaled.alignment({**scene, "local_query_scale": scale}, h)
    if shift:
        assert diagnostics["profile_shift_m"].abs().max() > 1.     # the depths really moved
    torch.testing.assert_close(divided["local_innovation"], plain["local_innovation"] / scale[:, None],
                               rtol=1e-12, atol=0)
    assert torch.equal(divided["local_valid"], plain["local_valid"])


def test_without_alignment_the_innovations_are_divided_before_the_local_branch():
    reference, scaled = models("local_control")
    assert scaled.alignment is None
    scene = innovation_scene()
    scale = torch.full_like(scene["baseline"], 2.)
    with torch.no_grad():
        expected = reference({**scene, "local_innovation": scene["local_innovation"] / scale[:, None]})["mean"]
        actual = scaled({**scene, "local_query_scale": scale})["mean"]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=0)


def long_record_scenes(scale=None):
    data = prepared()
    data.__class__ = LR.LongRecordScenes
    data.metadata, data.radius_km = {}, None
    climate = torch.as_tensor(np.random.default_rng(8).normal(size=(9, 3, 2)), dtype=torch.float32)
    data.profile_climate = climate
    data.profile_mean, data.profile_std = torch.zeros(3, 2), torch.ones(3, 2)
    data.profile_physical = data.y.clone()
    data.profile_absolute = data.profile_physical + climate
    data.configure(None, scale, "query")
    return data


def radius_scale_scenes(scale=None):
    data = prepared()
    data.__class__ = R.RadiusScaleScenes
    data.metadata = {}
    data.configure(None, scale, "query")
    return data


def test_the_scene_scales_the_anchor_and_hands_the_scale_to_the_model():
    scale = np.random.default_rng(3).uniform(.5, 2., size=(9, 3, 2))
    data, earlier = long_record_scenes(scale), radius_scale_scenes(scale)
    scene = data.scene(SOURCE, ROWS, LEVELS, data.analysis())
    before = earlier.scene(SOURCE, ROWS, LEVELS, earlier.analysis())
    at_query = torch.as_tensor(scale, dtype=torch.float32)[ROWS, LEVELS]
    assert torch.equal(scene["local_query_scale"], at_query)
    assert torch.equal(scene["baseline"], before["baseline"])          # the anchor: divided as before
    # neighbour innovations stay in z units here and are divided inside the model
    assert torch.equal(scene["local_innovation"] / at_query[:, None], before["local_innovation"])
    for name in before:
        if name not in ("baseline", "local_innovation"):
            assert torch.equal(scene[name], before[name]), name
    assert scene["local_profile_values"].shape == (5, 6, 3, 2)
    assert torch.equal(scene["query_level"], torch.as_tensor(LEVELS))


def test_without_a_scale_the_scene_is_the_matched_scene_plus_profiles():
    data, earlier = long_record_scenes(), radius_scale_scenes()
    scene = data.scene(SOURCE, ROWS, LEVELS, data.analysis())
    before = earlier.scene(SOURCE, ROWS, LEVELS, earlier.analysis())
    assert "local_query_scale" not in scene
    for name in before:
        assert torch.equal(scene[name], before[name]), name


def test_no_radius_and_no_per_profile_scale():
    data = long_record_scenes()
    with pytest.raises(ValueError):
        data.configure(250., None, "query")
    with pytest.raises(ValueError):
        data.configure(None, np.ones((9, 3, 2)), "profile")
