"""Formula, masking, normalization and grouped-inference checks."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

from ocean_tokenizer.reconstruction_metrics import (
    evaluate_reconstruction, gaussian_metrics, inverse_standardize,
    paired_rmse_bootstrap, regression_metrics,
)


def test_regression_formulas_and_signed_bias():
    p, y = np.array([0., 3., 4.]), np.array([1., 2., 4.])
    m = regression_metrics(p, y)
    assert m["n"] == 3
    assert m["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert m["mae"] == pytest.approx(2 / 3)
    assert m["mean_bias"] == 0
    assert m["r2"] == pytest.approx(4 / 7)
    assert m["pearson_r"] == pytest.approx(np.corrcoef(p, y)[0, 1])
    assert regression_metrics(y + 2, y)["mean_bias"] == 2


def test_r2_is_not_squared_correlation_and_can_be_negative():
    m = regression_metrics([10., 12., 14.], [1., 2., 3.])
    assert m["pearson_r"] == pytest.approx(1)
    assert m["r2"] < 0


@pytest.mark.parametrize("p,y,expected_r2", [([1., 1.], [1., 1.], None),
                                           ([2., 2.], [1., 3.], 0.),
                                           ([1.], [2.], None)])
def test_constant_and_single_value_are_explicit(p, y, expected_r2):
    m = regression_metrics(p, y)
    assert m["r2"] == expected_r2
    assert m["pearson_r"] is None


def test_empty_targets_are_null_and_channel_masks_are_independent():
    m = regression_metrics([np.nan], [np.nan])
    assert m == {"n": 0, "rmse": None, "mae": None, "mean_bias": None,
                 "r2": None, "pearson_r": None}
    m = evaluate_reconstruction([[2., np.nan], [99., 6.]], [[1., np.nan], [np.nan, 4.]])
    assert m["TEMP"]["n"] == m["SALT"]["n"] == 1
    assert m["TEMP"]["rmse"] == 1
    assert m["SALT"]["rmse"] == 2


def test_explicit_mask_does_not_drop_nonfinite_prediction_for_scored_target():
    with pytest.raises(ValueError, match="finite"):
        regression_metrics([np.nan, 1.], [1., 1.])
    m = regression_metrics([np.nan, 1.], [1., 1.], np.array([False, True]))
    assert m["n"] == 1 and m["rmse"] == 0


@pytest.mark.parametrize("p,y,mask", [([1.], [1., 2.], None),
                                     ([1., 2.], [1., 2.], [1, 0]),
                                     ([1., 2.], [1., 2.], [True])])
def test_bad_regression_shapes_and_masks_fail(p, y, mask):
    with pytest.raises(ValueError):
        regression_metrics(p, y, mask)


def test_normal_crps_at_mean_and_nll_have_closed_form():
    sigma = 2.5
    m = gaussian_metrics([3.], [sigma], [3.])
    assert m["crps"] == pytest.approx(sigma * (np.sqrt(2) - 1) / np.sqrt(np.pi))
    assert m["nll"] == pytest.approx(np.log(sigma) + .5 * np.log(2 * np.pi))
    assert m["coverage_68"] == m["coverage_95"] == 1


@pytest.mark.parametrize("mean,sigma,target", [(0., 1., 1.3), (2., .2, -.1), (-4., 2., -3.7)])
def test_crps_matches_independent_cdf_integral(mean, sigma, target):
    left = quad(lambda x: ndtr((x - mean) / sigma) ** 2, -np.inf, target)[0]
    right = quad(lambda x: (1 - ndtr((x - mean) / sigma)) ** 2, target, np.inf)[0]
    assert gaussian_metrics([mean], [sigma], [target])["crps"] == pytest.approx(left + right, abs=1e-9)


def test_crps_point_mass_limit_and_physical_unit_change():
    assert gaussian_metrics([0.], [1e-8], [2.])["crps"] == pytest.approx(2., abs=1e-8)
    p, s, y = np.array([1., 3.]), np.array([.4, 1.]), np.array([1.3, 1.8])
    original = gaussian_metrics(p, s, y)
    transformed = gaussian_metrics(5 * p + 8, 5 * s, 5 * y + 8)
    assert transformed["nll"] == pytest.approx(original["nll"] + np.log(5))
    assert transformed["crps"] == pytest.approx(5 * original["crps"])
    assert transformed["coverage_68"] == original["coverage_68"]
    assert transformed["coverage_95"] == original["coverage_95"]


def test_coverage_uses_documented_gaussian_quantiles():
    m = gaussian_metrics(np.zeros(5), np.ones(5), [0., .99, 1.01, 1.95, 1.97])
    assert m["coverage_68"] == .4
    assert m["coverage_95"] == .8


@pytest.mark.parametrize("std", [[0.], [-1.], [np.nan], [np.inf], [1., 2.]])
def test_invalid_scales_fail(std):
    with pytest.raises(ValueError):
        gaussian_metrics([0.], std, [1.])


def test_uncertainty_mask_ignores_missing_channel_not_scored_channel():
    m = evaluate_reconstruction([[1., np.nan]], [[1., np.nan]], [[.5, np.nan]])
    assert m["TEMP"]["nll"] == pytest.approx(np.log(.5) + .5 * np.log(2 * np.pi))
    assert m["SALT"]["n"] == 0 and m["SALT"]["nll"] is None


def test_inverse_standardize_uses_each_depth_and_offsets_only_for_means():
    z = np.ones((2, 2))
    scale = np.array([[1., 2.], [3., 4.]])
    offset = np.array([[10., 20.], [30., 40.]])
    actual = inverse_standardize(z, np.array([1, 0]), scale, offset)
    np.testing.assert_array_equal(actual, [[33., 44.], [11., 22.]])
    np.testing.assert_array_equal(inverse_standardize(z, np.array([1, 0]), scale, uncertainty=True), [[3., 4.], [1., 2.]])
    error = inverse_standardize(z, np.array([0, 1]), scale)
    assert regression_metrics(error[:, 0], np.zeros(2))["rmse"] == pytest.approx(np.sqrt(5))
    assert np.sqrt(5) != np.mean(scale[:, 0])


@pytest.mark.parametrize("values,levels,scale,offset,uncertainty", [
    ([[1.]], [0.], [[1.]], None, False),
    ([[1.]], [-1], [[1.]], None, False),
    ([[1.]], [1], [[1.]], None, False),
    ([[1.]], [0], [[0.]], None, False),
    ([[1.]], [0], [[np.nan]], None, False),
    ([[1.]], [0], [[1.]], [[np.nan]], False),
    ([[1.]], [0], [[1.]], [[0.]], True),
])
def test_invalid_inverse_normalization_fails(values, levels, scale, offset, uncertainty):
    with pytest.raises(ValueError):
        inverse_standardize(values, levels, scale, offset, uncertainty=uncertainty)


def test_grouped_rmse_difference_pools_squared_error_and_replays_seed():
    p, b, y = [0., 0., 1.], [1., 2., 3.], [0., 0., 0.]
    args = (p, b, y, [1, 1, 2])
    m = paired_rmse_bootstrap(*args, draws=500, seed=8)
    assert m == paired_rmse_bootstrap(*args, draws=500, seed=8)
    assert m["delta_rmse"] == pytest.approx(np.sqrt(1 / 3) - np.sqrt(14 / 3))
    assert m["n"] == 3 and m["n_groups"] == 2
    assert m["ci95_delta_rmse"][1] < 0


def test_grouped_bootstrap_missing_target_and_empty_target():
    m = paired_rmse_bootstrap([1., np.nan], [2., np.nan], [0., np.nan], [1, 2], draws=20)
    assert m["n_groups"] == 1 and m["delta_rmse"] == -1
    assert m["ci95_delta_rmse"] == [-1., -1.]
    assert paired_rmse_bootstrap([0.], [0.], [np.nan], [1])["n"] == 0


@pytest.mark.parametrize("kwargs", [{"draws": 0}, {"draws": 1.5}])
def test_bootstrap_rejects_invalid_draws(kwargs):
    with pytest.raises(ValueError):
        paired_rmse_bootstrap([1.], [2.], [0.], [1], **kwargs)


def test_bootstrap_refuses_to_lose_targets_with_missing_group_identity():
    with pytest.raises(ValueError, match="group identity"):
        paired_rmse_bootstrap([1.], [2.], [0.], [np.nan])


def test_report_seed_sd_is_distinct_from_error_of_ensemble_prediction():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "experiments/real_data/78_standard_metrics.py"
    spec = importlib.util.spec_from_file_location("conventional_metrics_report", path)
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    target = np.zeros((2, 2))
    predictions = [np.ones_like(target), -np.ones_like(target), 3 * np.ones_like(target)]
    details = [{"pooled": evaluate_reconstruction(p, target, np.ones_like(target))} for p in predictions]
    stats = report.seed_statistics(details)["TEMP"]
    assert stats["rmse"]["values"] == [1., 1., 3.]
    assert stats["rmse"]["mean"] == pytest.approx(5 / 3)
    assert stats["rmse"]["sample_sd"] == pytest.approx(np.sqrt(4 / 3))
    ensemble = evaluate_reconstruction(np.mean(predictions, axis=0), target)["TEMP"]["rmse"]
    assert ensemble == 1.
    assert stats["rmse"]["mean"] != ensemble
    assert stats["r2"]["n_defined"] == 0 and stats["r2"]["mean"] is None
