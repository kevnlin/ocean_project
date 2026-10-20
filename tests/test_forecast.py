"""Forecast scenes: lead draws, the lead of a scene and the query's surface features; no CESM2 data needed."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ocean_tokenizer import forecast_scenes as F
from test_long_record import long_record_scenes

SOURCE = np.arange(6)
ROWS, LEVELS = np.array([6, 7, 8, 7, 6]), np.array([0, 1, 2, 0, 2])


def drawer(leads=(0, 1, 3, 6), train_years=2):
    """Scenes with 10 profiles a month (8 inputs, 2 queries) over 2002-01 ... 2003-12."""
    data = F.ForecastScenes.__new__(F.ForecastScenes)
    months = np.repeat(np.arange(24, 48), 10)
    split = np.tile(np.r_[np.full(8, "cohort_float"), np.full(2, "heldout_float")], 24)
    def month(m, float_split=None):
        rows = np.flatnonzero(months == m)
        return rows if float_split is None else rows[split[rows] == float_split]
    data.c = SimpleNamespace(month_index=months, wmo=np.array([f"f{i}" for i in range(months.size)]), month=month)
    data.leads, data.train_years, data.n_levels = tuple(leads), train_years, 20
    data.months_available, data._months_cache, data._training_pools = set(range(24, 48)), {}, {}
    return data


def test_a_draw_reads_month_t_and_scores_month_t_plus_lead():
    data, rng = drawer(), np.random.default_rng(0)
    seen = set()
    for _ in range(300):
        month, source, rows, levels = data.training_draw(rng, 64)
        lead = data.lead_of(source, rows)
        seen.add(lead)
        assert lead in (0, 1, 3, 6) and 24 <= month - lead and month <= 47
        assert np.all(data.c.month_index[rows] == month) and np.all(data.c.month_index[source] == month - lead)
        assert source.size == 6 and np.all(levels < 20)                      # 70 % of 8 input floats
        if lead == 0:
            assert not np.intersect1d(source, rows).size                     # targets are held-out floats
        assert set(data.c.wmo[rows]) <= set(data.c.wmo[data.c.month(month, "cohort_float")])
    assert seen == {0, 1, 3, 6}


def test_a_scene_needs_one_input_month_and_one_later_query_month():
    data = drawer()
    with pytest.raises(ValueError):
        data.lead_of(np.array([0, 15]), np.array([30]))                      # inputs from two months
    with pytest.raises(ValueError):
        data.lead_of(np.array([30]), np.array([0]))                          # queries before inputs


def same(a, b):
    """Equal, with the fixture's missing values (NaN) equal to each other."""
    return a.shape == b.shape and torch.equal(torch.nan_to_num(a, nan=-999.), torch.nan_to_num(b, nan=-999.)) \
        and torch.equal(torch.isnan(a), torch.isnan(b)) if a.is_floating_point() else torch.equal(a, b)


def scenes(lead, surface=True):
    data = long_record_scenes(np.random.default_rng(3).uniform(.5, 2., size=(9, 3, 2)))
    data.__class__ = F.ForecastScenes
    data.leads, data.surface = (0, 1, 3, 6), surface
    data.c.month_index = np.r_[np.zeros(6, int), np.full(3, lead)]
    data.lagged = torch.arange(9 * 3 * 7, dtype=torch.float32).reshape(9, 3, 7)
    data.lag_index = {1: 0, 3: 1, 6: 2}
    return data


@pytest.mark.parametrize("lead", [1, 3, 6])
def test_at_a_lead_the_query_reads_its_input_months_surface(lead):
    data = scenes(lead)
    scene = data.scene(SOURCE, ROWS, LEVELS, data.analysis())
    reference = scenes(0).scene(SOURCE, ROWS, LEVELS, data.analysis())
    expected = data.lagged[torch.as_tensor(ROWS), data.lag_index[lead]]
    assert torch.equal(scene["query_features"][:, 44:51], expected)
    others = torch.ones(58, dtype=torch.bool)
    others[44:51] = False
    assert torch.equal(scene["query_features"][:, others], reference["query_features"][:, others])
    for name in reference:                       # the inputs, their neighbours and the anchor are month t's
        if name != "query_features":
            assert same(scene[name], reference[name]), name


def test_at_lead_zero_the_scene_is_the_long_record_scene():
    data = scenes(0)
    plain = long_record_scenes(np.random.default_rng(3).uniform(.5, 2., size=(9, 3, 2)))
    expected = plain.scene(SOURCE, ROWS, LEVELS, plain.analysis())
    actual = data.scene(SOURCE, ROWS, LEVELS, data.analysis())
    for name in expected:
        assert same(actual[name], expected[name]), name
