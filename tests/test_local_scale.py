"""A scale per depth level, 5-degree cell and calendar month; no CESM2 data needed."""
import numpy as np
import pytest

from ocean_tokenizer import local_scale as LS


def cohort(n=6000, levels=3, seed=0):
    """Profiles spread over two 5-degree cells and all twelve months."""
    rng = np.random.default_rng(seed)
    in_a = np.arange(n) % 2 == 0
    lat = np.where(in_a, 12., 42.) + rng.uniform(-2, 2, n)
    lon = np.where(in_a, 182., 302.) + rng.uniform(-2, 2, n)
    month = rng.integers(0, 12, n)
    values = rng.normal(size=(n, levels, 2))
    return values, lat, lon, month, in_a


def test_cell_index_wraps_longitude_and_keeps_the_pole_inside():
    cells = LS.cell_index(np.array([0., 0., 90., -90.]), np.array([-10., 350., 0., 0.]))
    assert cells[0] == cells[1]
    assert cells.min() >= 0 and cells.max() < LS.n_cells()
    assert cells[2] != cells[3]


def test_a_field_with_the_same_spread_everywhere_gets_scale_one():
    values, lat, lon, month, _ = cohort()
    table = LS.scale_table(values, lat, lon, month, np.ones(len(lat), bool))
    assert table.shape == (LS.n_cells(), 12, 3, 2)
    seen = np.unique(LS.cell_index(lat, lon))
    np.testing.assert_allclose(table[seen], 1., atol=.12)
    empty = np.setdiff1d(np.arange(LS.n_cells()), seen)
    assert np.array_equal(table[empty], np.ones_like(table[empty]))


def test_a_cell_with_twice_the_spread_gets_twice_the_scale():
    values, lat, lon, month, in_a = cohort()
    values[in_a] *= 2.
    table = LS.scale_table(values, lat, lon, month, np.ones(len(lat), bool), by_month=False)
    a, b = LS.cell_index(np.array([12., 42.]), np.array([182., 302.]))
    # second moments 4 and 1, level moment 2.5
    np.testing.assert_allclose(table[a], np.sqrt(4 / 2.5), rtol=.05)
    np.testing.assert_allclose(table[b], np.sqrt(1 / 2.5), rtol=.05)


def test_only_the_fitting_profiles_are_read():
    values, lat, lon, month, _ = cohort()
    fit = np.arange(len(lat)) < 4000
    before = LS.scale_table(values, lat, lon, month, fit)
    values[~fit] = 1e6
    after = LS.scale_table(values, lat, lon, month, fit)
    assert np.array_equal(before, after)


def test_month_entry_pools_the_month_and_its_two_neighbours():
    values, lat, lon, month, in_a = cohort(n=24000)
    values[in_a & (month == 6)] *= 5.
    table = LS.scale_table(values, lat, lon, month, np.ones(len(lat), bool))
    a = LS.cell_index(np.array([12.]), np.array([182.]))[0]
    by_month = table[a, :, 0, 0]
    assert by_month[[5, 6, 7]].min() > 1.5 * by_month[[0, 1, 2, 3, 9, 10, 11]].max()
    # the window wraps round the year
    values, lat, lon, month, in_a = cohort(n=24000)
    values[in_a & (month == 0)] *= 5.
    wrapped = LS.scale_table(values, lat, lon, month, np.ones(len(lat), bool))[a, :, 0, 0]
    assert wrapped[[11, 0, 1]].min() > 1.5 * wrapped[3:10].max()


def test_without_the_month_dimension_every_month_is_the_same():
    values, lat, lon, month, in_a = cohort()
    values[in_a & (month == 6)] *= 5.
    table = LS.scale_table(values, lat, lon, month, np.ones(len(lat), bool), by_month=False)
    assert np.array_equal(table, np.repeat(table[:, :1], 12, axis=1))


def test_a_thinly_sampled_cell_is_pulled_towards_the_level():
    values, lat, lon, month, in_a = cohort()
    values[in_a] *= 3.
    few = np.zeros(len(lat), bool)
    few[np.flatnonzero(in_a)[:10]] = True
    few[~in_a] = True
    a = LS.cell_index(np.array([12.]), np.array([182.]))[0]
    raw = LS.scale_table(values, lat, lon, month, few, by_month=False, pseudo=0.)[a, 0, 0, 0]
    shrunk = LS.scale_table(values, lat, lon, month, few, by_month=False, pseudo=20.)[a, 0, 0, 0]
    assert 1. < shrunk < raw


def test_a_constant_region_is_floored_not_amplified():
    values, lat, lon, month, in_a = cohort()
    values[in_a] = 0.
    table = LS.scale_table(values, lat, lon, month, np.ones(len(lat), bool), pseudo=0., floor=.2)
    a = LS.cell_index(np.array([12.]), np.array([182.]))[0]
    assert np.array_equal(table[a], np.full_like(table[a], .2))
    assert table.min() == .2


def test_missing_values_are_skipped_per_level_and_variable():
    values, lat, lon, month, in_a = cohort()
    reference = LS.scale_table(values, lat, lon, month, np.ones(len(lat), bool))
    values[::3, 1, 0] = np.nan
    table = LS.scale_table(values, lat, lon, month, np.ones(len(lat), bool))
    assert np.isfinite(table).all()
    assert np.array_equal(table[:, :, 0], reference[:, :, 0])
    assert np.array_equal(table[:, :, 1, 1], reference[:, :, 1, 1])
    # a level nobody observed keeps scale one
    values[:, 2, 1] = np.nan
    table = LS.scale_table(values, lat, lon, month, np.ones(len(lat), bool))
    assert np.array_equal(table[:, :, 2, 1], np.ones_like(table[:, :, 2, 1]))


def test_lookup_gives_each_profile_its_own_cell_and_month():
    values, lat, lon, month, in_a = cohort()
    values[in_a] *= 2.
    table = LS.scale_table(values, lat, lon, month, np.ones(len(lat), bool))
    scale = LS.lookup(table, lat, lon, month)
    assert scale.shape == values.shape
    i = 7
    assert np.array_equal(scale[i], table[LS.cell_index(lat[i:i+1], lon[i:i+1])[0], month[i]])


def test_shapes_that_do_not_match_are_refused():
    values, lat, lon, month, _ = cohort()
    with pytest.raises(ValueError, match="one row per profile"):
        LS.scale_table(values, lat[:-1], lon, month, np.ones(len(lat), bool))
    with pytest.raises(ValueError, match="calendar month"):
        LS.scale_table(values, lat, lon, month + 1, np.ones(len(lat), bool))
