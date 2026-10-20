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
