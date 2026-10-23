"""Sealed scoring: registration once, verification, and the metrics of a sealed test."""
from types import SimpleNamespace

import numpy as np
import pytest

from ocean_tokenizer import sealed_scoring as SS


def files(tmp_path):
    a, b, data = tmp_path / "a.pt", tmp_path / "b.pt", tmp_path / "cohort.nc"
    a.write_bytes(b"model a")
    b.write_bytes(b"model b")
    data.write_bytes(b"cohort")
    return a, b, data


def test_a_registration_records_every_file_and_cannot_be_written_twice(tmp_path):
    a, b, data = files(tmp_path)
    path = tmp_path / "registration.json"
    record = SS.register(path, [("grid", a), ("reference", b)], {"cohort": data}, {"dataset": "x", "test_years": [1]})
    assert [e["family"] for e in record["entries"]] == ["grid", "reference"] and record["dataset"] == "x"
    with pytest.raises(SystemExit):
        SS.register(path, [("grid", a)], {"cohort": data}, {})
    SS.verify(record)
    b.write_bytes(b"retrained")
    with pytest.raises(SystemExit):
        SS.verify(record)


def test_an_unknown_family_is_refused(tmp_path):
    a, _, data = files(tmp_path)
    with pytest.raises(SystemExit):
        SS.register(tmp_path / "r.json", [("transformer", a)], {"cohort": data}, {})


def norm():
    return SimpleNamespace(mean={"TEMP": np.array([10., 5., 2.]), "SALT": np.array([35., 34.8, 34.6])},
                           std={"TEMP": np.array([2., 1., .5]), "SALT": np.array([.2, .1, .05])})


def test_metrics_are_physical_by_variable_band_and_level():
    rng = np.random.default_rng(0)
    n, levels_m = 600, np.array([5., 150., 800.])
    level = np.tile(np.arange(3), n // 3)
    target = rng.normal(size=(n, 2))
    target[5, 1] = np.nan                                       # a missing salinity
    mean, base = target + rng.normal(scale=.3, size=(n, 2)), target + rng.normal(scale=.5, size=(n, 2))
    std = np.full((n, 2), .3)
    out = SS.metrics(mean, std, base, target, level, norm(), levels_m)
    scale = norm().std["TEMP"][level]
    expected = np.sqrt(np.mean(((mean[:, 0] - target[:, 0]) * scale) ** 2))
    assert out["TEMP"]["model"]["rmse"] == pytest.approx(expected)
    assert out["TEMP"]["climatology"]["rmse"] == pytest.approx(np.sqrt(np.mean((target[:, 0] * scale) ** 2)))
    assert out["SALT"]["model"]["n"] == n - 1
    assert 0.5 < out["TEMP"]["model"]["coverage_68"] < 0.85 and out["TEMP"]["model"]["crps"] > 0
    assert set(out["TEMP"]["model"]["by_band"]) == {"0-100m", "100-300m", "700-1400m"}
    deep = level == 2
    assert out["TEMP"]["persisted_oi"]["by_level"][2] == pytest.approx(
        np.sqrt(np.mean(((base[deep, 0] - target[deep, 0]) * .5) ** 2)))
    assert "crps" not in out["TEMP"]["persisted_oi"]


def test_a_calibration_is_registered_with_its_checkpoint_and_verified(tmp_path):
    a, b, data = files(tmp_path)
    calibration = tmp_path / "a.json"
    calibration.write_text('{"checkpoint_sha256": "%s", "scales": {}}' % SS.sha256(a))
    record = SS.register(tmp_path / "registration.json", [("grid", a, calibration), ("reference", b)],
                         {"cohort": data}, {})
    assert record["entries"][0]["calibration"]["sha256"] == SS.sha256(calibration)
    assert "calibration" not in record["entries"][1]
    SS.verify(record)
    calibration.write_text('{"checkpoint_sha256": "%s", "scales": {"0": {}}}' % SS.sha256(a))
    with pytest.raises(SystemExit):
        SS.verify(record)


def test_a_calibration_of_another_checkpoint_is_refused(tmp_path):
    a, b, data = files(tmp_path)
    calibration = tmp_path / "b.json"
    calibration.write_text('{"checkpoint_sha256": "%s", "scales": {}}' % SS.sha256(b))
    with pytest.raises(SystemExit):
        SS.register(tmp_path / "registration.json", [("grid", a, calibration)], {"cohort": data}, {})


def test_calibrated_spread_adds_a_block_and_leaves_the_errors_alone():
    rng = np.random.default_rng(1)
    n, levels_m = 3000, np.array([5., 150., 800.])
    level = np.tile(np.arange(3), n // 3)
    target = rng.normal(size=(n, 2))
    mean, std = target + rng.normal(scale=.6, size=(n, 2)), np.full((n, 2), .3)
    collected = {6: {"mean": mean, "std": std, "base": mean, "target": target, "level": level,
                     "month": np.full(n, 60)}}
    scales = {"6": {ch: {band: 2. for band in ("0-100m", "100-300m", "700-1400m")} for ch in ("TEMP", "SALT")}}
    plain = SS.summarize(collected, norm(), levels_m, (2005,))
    out = SS.summarize(collected, norm(), levels_m, (2005,), scales=scales)
    block, raw = out["6"]["all"]["TEMP"]["model_calibrated"], out["6"]["all"]["TEMP"]["model"]
    assert "model_calibrated" not in plain["6"]["all"]["TEMP"]
    assert raw["rmse"] == plain["6"]["all"]["TEMP"]["model"]["rmse"]
    assert block["mean_std"] == pytest.approx(2 * raw["mean_std"]) and block["coverage_95"] > raw["coverage_95"]
    assert set(block["by_band"]) == {"0-100m", "100-300m", "700-1400m"} and "crps" in block["by_band"]["0-100m"]
