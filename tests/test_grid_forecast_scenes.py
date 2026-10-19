"""Gridded forecast scenes: the input month's surface grid, the lead and the neighbours' positions."""
import numpy as np
import pytest
import torch

from ocean_tokenizer import forecast_scenes as F
from ocean_tokenizer.grid_forecast_scenes import GridForecastScenes
from test_forecast import LEVELS, ROWS, SOURCE, same, scenes


def grid_scenes(lead):
    data = scenes(lead)
    data.__class__ = GridForecastScenes
    grids = torch.arange(3 * 3 * 4 * 6, dtype=torch.float16).reshape(3, 3, 4, 6)    # months -1, 0 and a later one
    data.attach_surface_grids(grids, np.array([-1, 0, lead or 9]))
    return data


@pytest.mark.parametrize("lead", [0, 1, 3])
def test_a_grid_scene_adds_the_input_months_surface_and_the_lead(lead):
    data = grid_scenes(lead)
    scene = data.scene(SOURCE, ROWS, LEVELS, data.analysis())
    reference = scenes(lead).scene(SOURCE, ROWS, LEVELS, data.analysis())
    for name in reference:
        assert same(scene[name], reference[name]), name
    assert int(scene["lead"]) == lead and int(scene["input_month"]) == 0
    assert torch.equal(scene["surface_grid"], data.surface_grids[1].float())     # month 0, never month lead
    ids = scene["local_profile_ids"]
    assert torch.equal(scene["local_lat"], data.lat[ids]) and torch.equal(scene["local_lon"], data.lon[ids])


def test_a_month_without_a_surface_grid_is_an_error():
    data = grid_scenes(1)
    data.attach_surface_grids(data.surface_grids, np.array([5, 6, 7]))
    with pytest.raises(KeyError):
        data.scene(SOURCE, ROWS, LEVELS, data.analysis())


def test_surface_grid_months_must_be_distinct():
    data = grid_scenes(1)
    with pytest.raises(ValueError):
        data.attach_surface_grids(data.surface_grids, np.array([0, 0, 1]))
