"""Post-hoc spread calibration: one CRPS-optimal factor per lead, variable and depth band."""
import numpy as np
import pytest

from ocean_tokenizer import calibration as C
from ocean_tokenizer.reconstruction_metrics import gaussian_metrics

LEVELS_M = np.array([5., 150., 800.])           # one level in each of three bands


def test_crps_is_the_gaussian_crps_of_the_metrics():
    rng = np.random.default_rng(0)
    mean, std, target = rng.normal(size=500), rng.uniform(.2, 2., 500), rng.normal(size=500)
    assert C.crps(mean, std, target).mean() == pytest.approx(gaussian_metrics(mean, std, target)["crps"])


def cells(rng, n, lead, level, true_sd, predicted_sd):
    mean = rng.normal(size=(n, 2))
    target = mean + rng.normal(scale=true_sd, size=(n, 2))
    return mean, np.full((n, 2), predicted_sd), target, np.full(n, lead), np.full(n, level)


def stack(*groups):
    return [np.concatenate(parts) for parts in zip(*groups)]


def test_each_group_gets_the_factor_that_widens_its_spread_to_the_errors():
    rng = np.random.default_rng(1)
    mean, std, target, lead, level = stack(cells(rng, 20000, 0, 0, 1., 1.), cells(rng, 20000, 6, 0, 2., 1.),
                                           cells(rng, 20000, 6, 2, 1.5, 1.))
    scales = C.fit_scales(mean, std, target, lead, level, LEVELS_M)
    assert scales["0"]["TEMP"]["0-100m"] == pytest.approx(1., abs=.03)
    assert scales["6"]["SALT"]["0-100m"] == pytest.approx(2., abs=.06)
    assert scales["6"]["TEMP"]["700-1400m"] == pytest.approx(1.5, abs=.05)
    assert "100-300m" not in scales["6"]["TEMP"]


def test_missing_targets_are_skipped():
    rng = np.random.default_rng(2)
    mean, std, target, lead, level = cells(rng, 5000, 1, 1, 1.3, 1.)
    target[::3, 1] = np.nan
    scales = C.fit_scales(mean, std, target, lead, level, LEVELS_M)
    assert scales["1"]["SALT"]["100-300m"] == pytest.approx(1.3, abs=.06)


def test_applying_multiplies_each_cell_by_its_group_factor():
    scales = {"0": {"TEMP": {"0-100m": 1.1, "700-1400m": 1.2}, "SALT": {"0-100m": 1.3, "700-1400m": 1.4}},
              "6": {"TEMP": {"0-100m": 2.0, "700-1400m": 1.5}, "SALT": {"0-100m": 1.6, "700-1400m": 1.7}}}
    std = np.ones((4, 2))
    out = C.apply_scales(std, np.array([0, 0, 6, 6]), np.array([0, 2, 0, 2]), LEVELS_M, scales)
    assert out.tolist() == [[1.1, 1.3], [1.2, 1.4], [2.0, 1.6], [1.5, 1.7]]


def test_a_cell_without_a_fitted_factor_is_an_error():
    scales = {"0": {"TEMP": {"0-100m": 1.1}, "SALT": {"0-100m": 1.3}}}
    with pytest.raises(KeyError):
        C.apply_scales(np.ones((1, 2)), np.array([3]), np.array([0]), LEVELS_M, scales)
    with pytest.raises(KeyError):
        C.apply_scales(np.ones((1, 2)), np.array([0]), np.array([1]), LEVELS_M, scales)


def fake_run(folder, years, months, rng):
    import json
    folder.mkdir(parents=True)
    n = 4000
    level = np.tile(np.arange(3), n // 3 + 1)[:n]
    mean = rng.normal(size=(n, 2)).astype(np.float32)
    np.savez(folder / "forecast_seed1234.npz", mean=mean, std=np.ones((n, 2), np.float32),
             target=(mean + rng.normal(scale=1.4, size=(n, 2))).astype(np.float32),
             lead=np.repeat([0, 6], n // 2).astype(np.int8), level=level, month=rng.choice(months, n).astype(np.int16),
             normalization_scale=np.array([[2., .2], [1., .1], [.5, .05]]), normalization_offset=np.zeros((3, 2)))
    (folder / "best_seed1234.pt").write_bytes(b"weights")
    (folder / "summary_seed1234.json").write_text(json.dumps({
        "seed": 1234, "tag": folder.name, "best_step": 10, "val_years": years, "data": {"levels_m": LEVELS_M.tolist()}}))


def test_the_script_fits_a_run_on_its_selection_years_and_refuses_sealed_ones(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "fit116", Path(__file__).resolve().parents[1] / "experiments/synthetic/116_fit_calibration.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    rng = np.random.default_rng(3)
    fake_run(tmp_path / "ok", [2004, 2005], [48, 59, 60, 71], rng)
    record = script.fit(tmp_path / "ok")
    assert record["fit_years"] == [2004, 2005] and record["checkpoint_sha256"] == script.sha256(tmp_path / "ok/best_seed1234.pt")
    assert record["scales"]["6"]["TEMP"]["0-100m"] == pytest.approx(1.4, abs=.1)
    fake_run(tmp_path / "leaks", [2004, 2005], [48, 72], rng)            # month 72 is January 2006
    with pytest.raises(SystemExit):
        script.fit(tmp_path / "leaks")
    fake_run(tmp_path / "sealed", [2021, 2022], [252, 264], rng)
    with pytest.raises(SystemExit):
        script.fit(tmp_path / "sealed")
