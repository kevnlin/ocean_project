"""Exactness, provenance and memory contracts for reusable prepared scenes."""
import hashlib
import json

import numpy as np
import pytest
import torch
import xarray as xr

from ocean_tokenizer import anchored as A
from ocean_tokenizer import latent_experiment as E
from ocean_tokenizer import latent_scene_cache as C
from ocean_tokenizer.argo_obs import ArgoCohort, ArgoNorm


@pytest.fixture
def tiny_root(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    n, l = 8, 3
    c = ArgoCohort(
        region="global", levels=np.array([5., 125., 900.]), month_index=np.full(n, 192),
        grid_y=np.arange(n), grid_x=np.arange(n), lat=np.linspace(9., 11., n),
        lon=np.linspace(179., 181., n), wmo=np.array(["101", "101", "102", "103", "901", "901", "902", "104"]),
        year=np.full(n, 2016), year_split=np.full(n, "train"),
        float_split=np.array(["cohort_float"] * 4 + ["heldout_float"] * 3 + ["cohort_float"]),
        TEMP=rng.normal(size=(n, l)), SALT=rng.normal(size=(n, l)),
        TEMP_ERR=np.full((n, l), .2), SALT_ERR=np.full((n, l), .3), grid=(180, 360))
    c.TEMP[0, 1], c.TEMP_ERR[0, 0], c.SALT_ERR[1, 2] = np.nan, np.nan, -.2
    c._by_month = {192: (0, n)}
    monkeypatch.setattr(E, "load_cohort", lambda *args, **kwargs: (c, None))
    norm = ArgoNorm.fit(c, "train")
    sat = {key: np.linspace(.05, .3, n) for key in (
        "SLA_1deg_anom", "SST_1deg_anom", "SSS_1deg_anom", "SST_1deg", "SLA_day_c",
        "SLA_day_e", "SLA_day_w", "SLA_day_n", "SLA_day_s", "SLA_clim_1deg", "SLA_1deg")}
    sat["month_index"] = c.month_index.copy()
    sat["SLA_day_c"][5] = np.nan
    x = E.satellite_features(c, sat)
    y = np.concatenate([norm.z(ch, getattr(c, ch)) for ch in E.CH], axis=1)
    fingerprint = E.array_fingerprint((c.month_index, c.lat, c.lon, c.wmo, c.levels, y))
    checkpoint = {"cohort_fingerprint": fingerprint,
                  "feature_fingerprint": hashlib.sha256(memoryview(np.ascontiguousarray(x)).cast("B")).hexdigest(),
                  "feature_mean": torch.zeros(56, dtype=torch.float64),
                  "feature_std": torch.ones(56, dtype=torch.float64)}
    anchor = tmp_path / "outputs/audit/global/anc_satday_dfs"
    anchor.mkdir(parents=True)
    torch.save(checkpoint, anchor / "first_guess_seed1234.pt")
    torch.save({"cohort_fingerprint": fingerprint, "training_predictions": "cross-fit by year",
                "predictions": torch.arange(n * 2 * l, dtype=torch.float32).reshape(n, 2*l) / 100},
               anchor / "first_guess_predictions_seed1234.pt")
    kriging = tmp_path / "outputs/audit/global/anc_satday_kriging"
    kriging.mkdir(parents=True)
    torch.save(A.InnovationAnalysis(2*l, mode="kriging", k=32, use_time=True, use_state=True).state_dict(),
               kriging / "model_seed1234.pt")
    reference = tmp_path / "outputs/audit/global/setconv_surface"
    reference.mkdir(parents=True)
    (reference / "summary_seed1234.json").write_text(json.dumps({"scores": {}}))
    data = tmp_path / "data/argo_cohort"
    data.mkdir(parents=True)
    np.savez(data / "global_satellite.npz", **sat)
    np.savez(data / "global_anomcell.npz", TEMP=c.TEMP, SALT=c.SALT, n=n)
    xr.Dataset({"month_index": ("profile", c.month_index),
                "time": ("profile", np.arange(n).astype("timedelta64[D]") + np.datetime64("2016-01-01"))}).to_netcdf(
                    data / "global_global.nc", engine="scipy")
    return tmp_path


def test_cache_replay_is_bit_exact_and_bypasses_loading_and_normalization(tiny_root, monkeypatch):
    original = E.OceanScenes(tiny_root)
    manifest = C.prepare_scene_cache(tiny_root)
    monkeypatch.setattr(E, "load_cohort", lambda *a, **kw: pytest.fail("raw cohort reloaded"))
    monkeypatch.setattr(E.ArgoNorm, "fit", lambda *a, **kw: pytest.fail("normalization refitted"))
    cached = E.OceanScenes(tiny_root)
    assert cached.metadata == original.metadata
    assert cached.prepared_cache["cache_key"] == manifest["cache_key"]
    assert cached.reference == original.reference
    for field in C.TENSOR_FIELDS:
        actual, expected = getattr(cached, field), getattr(original, field)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    for field in C.COHORT_FIELDS:
        actual, expected = getattr(cached.c, field), getattr(original.c, field)
        if expected is not None:
            np.testing.assert_array_equal(actual, expected)
            assert isinstance(actual, np.memmap)
    for name in ("mean", "std"):
        for ch in E.CH:
            np.testing.assert_array_equal(getattr(cached.norm, name)[ch], getattr(original.norm, name)[ch])
    for split in E.SPLITS:
        assert cached.months(split) == original.months(split)
    rng_original, rng_cached = np.random.default_rng(456), np.random.default_rng(456)
    for _ in range(10):
        original_draw, cached_draw = original.training_draw(rng_original, 12), cached.training_draw(rng_cached, 12)
        for actual, expected in zip(cached_draw, original_draw):
            np.testing.assert_array_equal(actual, expected)
    original_ev, cached_ev = original.eval_months("train", max_cells=0), cached.eval_months("train", max_cells=0)
    for first, second in zip(original_ev, cached_ev):
        for field in ("source", "target", "profile", "level"):
            np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    first = original_ev[0]
    with torch.no_grad():
        old_scene = original.scene(first.source, first.target[first.profile], first.level, original.analysis())
        new_scene = cached.scene(first.source, first.target[first.profile], first.level, cached.analysis())
    for field in old_scene:
        np.testing.assert_array_equal(new_scene[field].numpy(), old_scene[field].numpy())


def test_cpu_tensor_storage_is_file_backed_and_copy_on_write(tiny_root):
    C.prepare_scene_cache(tiny_root)
    cached = E.OceanScenes(tiny_root)
    backing = cached._prepared_mmaps["tensor.x"]
    assert cached.x.data_ptr() == backing.__array_interface__["data"][0]
    saved_value = float(cached.x[0, 0])
    cached.x[0, 0] = 123456
    reloaded = E.OceanScenes(tiny_root)
    assert float(reloaded.x[0, 0]) == saved_value


@pytest.mark.parametrize("relative", [
    "data/argo_cohort/global_global.nc", "data/argo_cohort/global_anomcell.npz",
    "data/argo_cohort/global_satellite.npz", "outputs/audit/global/anc_satday_dfs/first_guess_seed1234.pt",
    "outputs/audit/global/anc_satday_dfs/first_guess_predictions_seed1234.pt",
    "outputs/audit/global/anc_satday_kriging/model_seed1234.pt",
    "outputs/audit/global/setconv_surface/summary_seed1234.json",
])
def test_changed_source_is_rejected_without_reloading_or_refitting(tiny_root, monkeypatch, relative):
    C.prepare_scene_cache(tiny_root)
    with (tiny_root / relative).open("ab") as stream:
        stream.write(b"changed-input")
    monkeypatch.setattr(E, "load_cohort", lambda *a, **kw: pytest.fail("stale cache fell back to raw loading"))
    with pytest.raises(ValueError, match="provenance mismatch"):
        E.OceanScenes(tiny_root)


def test_changed_recipe_is_rejected(tiny_root, monkeypatch):
    C.prepare_scene_cache(tiny_root)
    original = C.provenance
    def changed(*args, **kwargs):
        value = original(*args, **kwargs)
        value["recipes"]["latent_experiment.py"]["sha256"] = "new-recipe"
        return value
    monkeypatch.setattr(C, "provenance", changed)
    with pytest.raises(ValueError, match="provenance mismatch"):
        E.OceanScenes(tiny_root)


def test_corrupted_artifact_is_rejected(tiny_root):
    manifest = C.prepare_scene_cache(tiny_root)
    artifact = C.cache_directory(tiny_root) / manifest["arrays"]["tensor.x"]["path"]
    with artifact.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="artifact integrity mismatch"):
        E.OceanScenes(tiny_root)


def test_prepare_failure_does_not_publish_a_manifest(tiny_root, monkeypatch):
    monkeypatch.setattr(E, "load_cohort", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("interrupted")))
    with pytest.raises(RuntimeError, match="interrupted"):
        C.prepare_scene_cache(tiny_root)
    base = C.cache_directory(tiny_root)
    assert not (base / "manifest.json").exists()
    assert not list(base.glob(".preparing-*"))


def test_existing_verified_cache_does_not_prepare_again(tiny_root, monkeypatch):
    manifest = C.prepare_scene_cache(tiny_root)
    monkeypatch.setattr(E, "load_cohort", lambda *a, **kw: pytest.fail("cache prepared again"))
    assert C.prepare_scene_cache(tiny_root) == manifest
