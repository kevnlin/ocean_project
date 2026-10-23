"""The real global cohort for the gridded model: subset, splits, QC, fingerprint."""
import numpy as np
import pytest

from ocean_tokenizer.argo_obs import ArgoCohort
from ocean_tokenizer.real_global import REAL_SPLITS, quality_controlled, subset_cohort


def fake_cohort():
    months = np.array([200, 200, 201, 201, 201, 214])
    p, l = months.size, 3
    rng = np.random.default_rng(0)
    c = ArgoCohort(region="global", levels=np.array([5., 50., 500.]), month_index=months,
                   grid_y=np.arange(p), grid_x=np.arange(p), lat=rng.uniform(-60, 60, p),
                   lon=rng.uniform(0, 360, p), wmo=np.array(["a", "b", "a", "c", "d", "e"]),
                   year=2000 + months // 12, year_split=np.full(p, "train"),
                   float_split=np.array(["cohort_float", "heldout_float"] * 3),
                   TEMP=rng.normal(size=(p, l)), SALT=rng.normal(size=(p, l)),
                   TEMP_ERR=np.full((p, l), .01), SALT_ERR=np.full((p, l), .01))
    uniq, start = np.unique(months, return_index=True)
    stop = np.r_[start[1:], months.size]
    c._by_month = {int(m): (int(a), int(b)) for m, a, b in zip(uniq, start, stop)}
    return c


def test_a_subset_keeps_every_profile_array_aligned_and_reindexes_months():
    c = fake_cohort()
    keep = np.array([False, True, True, False, True, True])
    s = subset_cohort(c, keep)
    for name in ("month_index", "lat", "lon", "wmo", "year", "float_split", "grid_y"):
        assert np.array_equal(getattr(s, name), getattr(c, name)[keep]), name
    assert np.array_equal(s.TEMP, c.TEMP[keep]) and np.array_equal(s.SALT_ERR, c.SALT_ERR[keep])
    assert np.array_equal(s.month(200), [0]) and np.array_equal(s.month(201), [1, 2])
    assert np.array_equal(s.month(201, float_split="cohort_float"), [1, 2])      # rows 2 and 4 are input floats
    assert np.array_equal(s.month(200, float_split="heldout_float"), [0])
    assert np.array_equal(s.month(214), [3]) and s.month(199).size == 0


def test_values_far_from_the_training_mean_are_missing():
    z = np.array([[0., 9.9, -10.1], [np.nan, 11., 2.]])
    out = quality_controlled(z)
    assert np.array_equal(np.isnan(out), [[False, False, True], [True, True, False]])
    assert out[0, 1] == 9.9


def test_the_splits_cover_2016_to_2023_without_the_sealed_holdout():
    years = sorted(y for lo, hi in REAL_SPLITS.values() for y in range(lo, hi + 1))
    assert years == list(range(2016, 2024))
    assert REAL_SPLITS["validation"] == (2021, 2021)


from ocean_tokenizer.real_global import profile_surface_columns, surface_z


def test_surface_z_uses_training_months_only():
    rng = np.random.default_rng(1)
    months = np.arange(192, 192 + 36)                     # 2016-01 .. 2018-12
    fields = rng.normal(size=(36, 3, 4, 5)) + np.arange(36)[:, None, None, None] * .1
    fields[:, :, 0, 0] = np.nan                            # land
    train = months < 192 + 24
    z, climate, mean, std = surface_z(fields, months, train)
    expected_climate = np.stack([np.nanmean(fields[train & (months % 12 == m)], 0) for m in range(12)])
    np.testing.assert_allclose(climate, expected_climate)
    anomaly = fields - expected_climate[months % 12]
    np.testing.assert_allclose(mean, np.nanmean(anomaly[train], axis=(0, 2, 3)))
    np.testing.assert_allclose(z[5], (anomaly[5] - mean[:, None, None]) / std[:, None, None])
    assert np.isnan(z[:, :, 0, 0]).all()


def test_profile_columns_sample_the_input_months_grid():
    grid_lat, grid_lon = np.array([-0.5, 0.5]), np.arange(4) + 0.5
    months = np.array([192, 193, 194])
    z = np.zeros((3, 3, 2, 4))
    for i in range(3):
        z[i] = i + np.arange(3)[:, None, None]             # month i, field k -> i + k
    z[2, 1] = np.nan                                       # month 194 has no SSS
    absolute = np.full((3, 2, 4), 14.)
    lat, lon, month = np.array([-0.5, 0.5, 0.5]), np.array([1.5, 2.5, 3.5]), np.array([194, 194, 192])
    now = profile_surface_columns(z, absolute, months, grid_lat, grid_lon, lat, lon, month)
    np.testing.assert_allclose(now[0], [2., 1., 0., 0., 4., 1., 0.])     # SSS missing in 194
    lagged = profile_surface_columns(z, absolute, months, grid_lat, grid_lon, lat, lon, month, lag=1)
    np.testing.assert_allclose(lagged[0], [1., 1., 2., 1., 3., 1., 0.])  # month 193
    assert (lagged[2] == 0).all()                                        # 191 is before the record


from ocean_tokenizer.real_global import EVAL_SEED, real_eval_month


def test_an_evaluation_month_scores_heldout_floats_from_all_input_floats():
    c = fake_cohort()
    obs = {"TEMP": c.TEMP.copy(), "SALT": c.SALT.copy()}
    ev = real_eval_month(c, obs, 201)
    assert np.array_equal(ev["src"], c.month(201, float_split="cohort_float"))
    assert np.array_equal(ev["tgt"], c.month(201, float_split="heldout_float"))
    assert not set(c.wmo[ev["src"]]) & set(c.wmo[ev["tgt"]])
    assert len(ev["prof"]) == ev["tgt"].size * 3
    np.testing.assert_array_equal(ev["target"]["TEMP"], obs["TEMP"][ev["tgt"]][ev["prof"], ev["lev"]])


def test_a_forecast_month_reads_the_input_month_and_keeps_the_target_months_cells():
    c = fake_cohort()
    obs = {"TEMP": c.TEMP.copy(), "SALT": c.SALT.copy()}
    now, ahead = real_eval_month(c, obs, 201, max_cells=2), real_eval_month(c, obs, 201, max_cells=2, lead=1)
    assert np.array_equal(ahead["src"], c.month(200, float_split="cohort_float"))
    assert np.array_equal(ahead["tgt"], now["tgt"])
    assert np.array_equal(ahead["prof"], now["prof"]) and np.array_equal(ahead["lev"], now["lev"])
    pick = np.random.default_rng([EVAL_SEED, 201]).choice(3, 2, replace=False)
    assert np.array_equal(now["lev"], np.tile(np.arange(3), 1)[pick])
