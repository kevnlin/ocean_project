"""Geographic diagnostics must retain scoring identities and spherical geometry."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


PATH = Path(__file__).resolve().parents[1] / "experiments/real_data/72_latent_generalization.py"
spec = importlib.util.spec_from_file_location("latent_generalization", PATH)
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


def raw_fixture(path):
    # Deliberately unsorted. No raw T/S arrays exist in this metadata-only file.
    dataset = xr.Dataset({
        "month_index": ("profile", [264, 252, 264, 252]),
        "lat": ("profile", [1., 0., 0., 1.]),
        "lon": ("profile", [1., 0., 0., 1.]),
        "wmo": ("profile", ["B", "A", "A", "B"]),
        "year": ("profile", [2022, 2021, 2022, 2021]),
        "float_split": ("profile", ["heldout_float", "cohort_float", "cohort_float", "heldout_float"]),
    }, coords={"level": [5., 15.]})
    dataset.to_netcdf(path)
    return G.read_metadata(path)


def predictions(split="development", mean=0.):
    month, profile = (264, 2) if split == "development" else (252, 1)
    return {"month": np.array([month, month], dtype=np.int16), "profile": np.array([profile, profile]),
        "level": np.array([0, 1]), "mean": np.full((2, 2), mean, dtype=np.float32),
        "std": np.ones((2, 2), dtype=np.float32), "target": np.array([[1., 2.], [2., 3.]], dtype=np.float32),
        "baseline": np.full((2, 2), 2., dtype=np.float32)}


def job(seed=1234):
    return {"tag": f"dense_base_s{seed}", "seed": seed, "variant": "dense", "width": 128,
        "latents": 64, "blocks": 4, "experts": 4, "steps": 6000, "context_profiles": 512}


def summary(seed=1234):
    cfg = job(seed)
    val, dev = predictions("validation"), predictions(mean=float(seed - 1234))
    return {"tag": cfg["tag"], "seed": seed,
        "config": {"variant": "dense", "width": 128, "n_latents": 64, "n_blocks": 4, "n_experts": 4},
        "training": {"steps": 6000, "context_profiles": 512, "val_cells": 0},
        "data": {"cohort_fingerprint": "a" * 64, "feature_fingerprint": "b" * 64,
            "scene_fingerprint": "c" * 64, "n_profiles": 4, "n_levels": 2, "anchor": "/fixture/anchor"},
        **{name: {"scores": G.R.z_scores(a["mean"], a["target"]),
                  "baseline_scores": G.R.z_scores(a["baseline"], a["target"])}
           for name, a in (("validation", val), ("development", dev))}}


def save_item(root, seed=1234):
    folder = root / job(seed)["tag"]
    folder.mkdir(parents=True)
    (folder / f"summary_seed{seed}.json").write_text(json.dumps(summary(seed)))
    for split in ("validation", "development"):
        arrays = predictions(split, mean=float(seed - 1234) if split == "development" else 0.)
        np.savez(folder / f"{split}_seed{seed}.npz", **arrays)
    return folder


def test_metadata_read_is_stably_sorted_and_needs_no_raw_values(tmp_path):
    metadata = raw_fixture(tmp_path / "raw.nc")
    np.testing.assert_array_equal(metadata["month_index"], [252, 252, 264, 264])
    np.testing.assert_array_equal(metadata["float_split"], ["cohort_float", "heldout_float", "heldout_float", "cohort_float"])
    assert len(metadata["metadata_fingerprint"]) == 64
    assert G.read_metadata(tmp_path / "raw.nc")["metadata_fingerprint"] == metadata["metadata_fingerprint"]


def test_scoring_index_maps_to_sorted_raw_metadata(tmp_path):
    metadata = raw_fixture(tmp_path / "raw.nc")
    G.validate_identity(predictions(), metadata, "development")
    G.validate_identity(predictions("validation"), metadata, "validation")


@pytest.mark.parametrize("field,value,match", [
    ("profile", 3, "heldout_float"), ("profile", 99, "outside raw"),
    ("month", 265, "month/profile"), ("level", 2, "depth")])
def test_identity_rejects_bad_profile_month_depth(tmp_path, field, value, match):
    metadata = raw_fixture(tmp_path / "raw.nc")
    arrays = predictions()
    arrays[field][0] = value
    with pytest.raises(ValueError, match=match):
        G.validate_identity(arrays, metadata, "development")


def test_identity_rejects_reordered_or_incomplete_depth_cells(tmp_path):
    metadata = raw_fixture(tmp_path / "raw.nc")
    arrays = {k: v[::-1].copy() for k, v in predictions().items()}
    with pytest.raises(ValueError, match="registered"):
        G.validate_identity(arrays, metadata, "development")


def test_float_source_and_queries_must_be_disjoint(tmp_path):
    metadata = raw_fixture(tmp_path / "raw.nc")
    metadata["wmo"][3] = "B"
    with pytest.raises(ValueError, match="float identities"):
        G.validate_identity(predictions(), metadata, "development")


def test_spherical_distance_handles_date_line_and_duplicate_depths():
    metadata = {"month_index": np.array([264, 264, 264]), "lat": np.zeros(3),
        "lon": np.array([179., -179., 0.]), "float_split": np.array(["cohort_float", "heldout_float", "heldout_float"])}
    arrays = {"month": np.full(4, 264), "profile": np.array([1, 1, 2, 2])}
    distance = G.nearest_context_distances(arrays, metadata)
    np.testing.assert_allclose(distance[:2], G.EARTH_RADIUS_KM * np.deg2rad(2.), rtol=1e-12)
    np.testing.assert_allclose(distance[2:], G.EARTH_RADIUS_KM * np.deg2rad(179.), rtol=1e-12)


def test_nearest_distance_uses_same_month_only():
    metadata = {"month_index": np.array([264, 265, 264]), "lat": np.zeros(3),
        "lon": np.array([10., 0., 0.]), "float_split": np.array(["cohort_float", "cohort_float", "heldout_float"])}
    distance = G.nearest_context_distances({"month": np.array([264]), "profile": np.array([2])}, metadata)
    assert distance[0] == pytest.approx(G.EARTH_RADIUS_KM * np.deg2rad(10.))


def test_distance_bins_include_boundaries_once():
    distance = np.array([0., 25., 50., 100., 200., 500., 1000.])
    groups = G.interval_groups(distance, G.DISTANCE_EDGES, list("abcdef"))
    np.testing.assert_array_equal(np.stack([mask for _, mask in groups]).sum(axis=0), 1)
    np.testing.assert_array_equal([mask.sum() for _, mask in groups], [1, 1, 1, 1, 1, 2])


def test_group_partitions_cover_both_poles_longitude_and_all_seasons():
    metadata = {"lat": np.array([-90., -60., -30., 0., 30., 60., 90., 1.]),
        "lon": np.array([-180., -120., -60., 0., 60., 120., 180., 0.])}
    arrays = {"profile": np.arange(8), "month": np.array([264, 265, 266, 269, 272, 275, 267, 274])}
    groups = G.diagnostic_groups(arrays, metadata, np.arange(8.) * 100.)
    for partitions in groups.values():
        np.testing.assert_array_equal(np.stack([mask for _, mask in partitions]).sum(axis=0), 1)
    seasons = dict(groups["calendar_seasons"])
    np.testing.assert_array_equal(seasons["DJF"], [True, True, False, False, False, True, False, False])


def test_group_errors_and_missing_variable_counts_are_exact(tmp_path):
    metadata = raw_fixture(tmp_path / "raw.nc")
    arrays = predictions()
    arrays["target"][0, 0] = np.nan
    calibration = G.R.calibrate_baseline(predictions("validation"))
    results = G.analyse_groups(arrays, metadata, calibration, np.array([100., 100.]))
    active = next(x for x in results["nearest_context_distance_km"] if x["group"] == "[100, 200)")
    assert active["n_profiles"] == 1 and active["n_cells"] == 2
    assert active["scores"]["TEMP"]["n"] == 1
    assert active["scores"]["SALT"]["n"] == 2
    assert active["delta_J"]["TEMP"] == pytest.approx(1.)
    empty = results["nearest_context_distance_km"][0]
    assert empty["scores"]["TEMP"]["n"] == 0 and empty["delta_J"]["TEMP"] is None
    json.dumps(results, allow_nan=False)


@pytest.mark.parametrize("field", G.FINGERPRINT_KEYS)
def test_saved_data_fingerprint_mismatch_is_rejected(tmp_path, field):
    metadata = raw_fixture(tmp_path / "raw.nc")
    stored = summary()
    anchor = {"cohort_fingerprint": "a" * 64, "feature_fingerprint": "b" * 64, "levels": [5., 15.]}
    reference = stored["data"].copy()
    stored["data"][field] = "d" * 64
    with pytest.raises(ValueError, match=field):
        G.validate_fingerprints(stored, reference, metadata, anchor)


def test_wrong_npz_at_correct_path_cannot_pass_summary_scores():
    arrays = predictions(mean=1.)
    with pytest.raises(ValueError, match="arrays differ"):
        G.validate_summary_scores(arrays, summary(), "development")


def test_running_search_does_not_open_raw_metadata_or_integration_predictions(tmp_path):
    root = tmp_path / "outputs"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"jobs": [job()]}))
    result = G.build_report(manifest, root, tmp_path / "does_not_exist.nc")
    assert not result["selected_tags"] and result["raw_metadata"] is None
    assert not result["groups"] and len(result["search_pending"]) == 1
    assert "不填" in G.render_report(result)


def test_complete_selected_seeds_get_calibrated_groups_and_ensemble(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw.nc"
    raw_fixture(raw_path)
    root = tmp_path / "outputs"
    save_item(root, 1234)
    save_item(root, 1235)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"jobs": [job()]}))
    monkeypatch.setattr(G, "load_anchor", lambda _: {"cohort_fingerprint": "a" * 64, "feature_fingerprint": "b" * 64, "levels": [5., 15.]})
    result = G.build_report(manifest, root, raw_path)
    assert result["selected_tags"] == ["dense_base_s1234"]
    group = result["groups"][0]
    assert len(group["seeds"]) == 2 and group["ensemble"]["n_seeds"] == 2
    assert group["baseline_calibration"]["split"] == "validation_2021"
    assert result["raw_metadata"]["loaded_variables"][-1] == "level"
    assert not result["independent_ood"]
    text = G.render_report(result)
    assert "不是海盆" in text and "ΔJ" in text and "ensemble (2 seeds)" in text
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field", G.R.IDENTITY_KEYS)
def test_ensemble_refuses_all_five_identity_mismatches(tmp_path, field):
    paths = [tmp_path / "a.npz", tmp_path / "b.npz"]
    arrays = predictions()
    np.savez(paths[0], **arrays)
    arrays[field].flat[0] += 1
    np.savez(paths[1], **arrays)
    with pytest.raises(ValueError, match=field):
        G.R.ensemble_predictions(paths)


def test_changed_depth_coordinates_cannot_keep_the_same_dimension(tmp_path):
    metadata = raw_fixture(tmp_path / "raw.nc")
    stored = summary()
    anchor = {"cohort_fingerprint": "a" * 64, "feature_fingerprint": "b" * 64, "levels": [5., 16.]}
    with pytest.raises(ValueError, match="depth coordinates"):
        G.validate_fingerprints(stored, stored["data"], metadata, anchor)


def test_summary_valid_count_must_match_exactly_even_for_large_arrays():
    stored = summary()
    stored["development"]["scores"]["TEMP"]["n"] = 3
    with pytest.raises(ValueError, match="TEMP/n"):
        G.validate_summary_scores(predictions(), stored, "development")
