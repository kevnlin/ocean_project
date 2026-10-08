"""Scientific-report checks: frozen selection, paired evidence and uncertainty."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


REPORT_PATH = Path(__file__).resolve().parents[1] / "experiments/real_data/71_latent_report.py"
spec = importlib.util.spec_from_file_location("latent_report", REPORT_PATH)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def predictions(mean=0., std=1.):
    return {"mean": np.full((6, 2), mean, dtype=np.float32),
            "std": np.full((6, 2), std, dtype=np.float32),
            "target": np.array([[0., 1.], [1., 2.], [2., 3.], [3., 4.], [4., 5.], [5., 6.]], dtype=np.float32),
            "baseline": np.full((6, 2), 2., dtype=np.float32),
            "month": np.array([240, 240, 241, 241, 242, 242]),
            "profile": np.arange(6), "level": np.array([0, 2, 0, 2, 0, 2])}


def job(tag="dense_base_s1234", width=128, seed=1234, analysis_only=False):
    return {"tag": tag, "variant": "dense", "width": width, "latents": 64,
            "blocks": 4, "experts": 4, "context_profiles": 512,
            "seed": seed, "steps": 6000, "analysis_only": analysis_only}


def summary(cfg, macro=.5, dev=.9):
    scores = {"TEMP": {"J": macro, "rmse_z": macro},
              "SALT": {"J": macro, "rmse_z": macro}, "macro_z": macro}
    return {"tag": cfg["tag"], "seed": cfg["seed"], "params": 1000, "best_step": 500,
            "config": {"variant": cfg["variant"], "width": cfg["width"], "n_latents": cfg["latents"],
                       "n_blocks": cfg["blocks"], "n_experts": cfg["experts"]},
            "data": {"cohort_fingerprint": "cohort", "feature_fingerprint": "features", "scene_fingerprint": "scene"},
            "training": {"steps": cfg["steps"], "context_profiles": cfg["context_profiles"],
                         "analysis_only": cfg["analysis_only"]},
            "validation": {"scores": scores},
            "development": {"scores": {"TEMP": {"J": dev}, "SALT": {"J": dev}, "macro_z": dev}}}


def item(cfg, **kwargs):
    return {"job": cfg, "summary": summary(cfg, **kwargs), "folder": Path(cfg["tag"])}


def save_item(root, cfg, **kwargs):
    folder = root / cfg["tag"]
    folder.mkdir(parents=True)
    stored = summary(cfg, **kwargs)
    (folder / f"summary_seed{cfg['seed']}.json").write_text(json.dumps(stored))
    return folder


def test_selection_uses_validation_not_development():
    base = item(job(), macro=.3, dev=100.)
    large = item(job("dense_large_s1234", width=192), macro=.4, dev=0.)
    control = item(job("analysis_only_s1234", analysis_only=True), macro=.2)
    chosen = report.validation_selection([base, large, control], [])
    assert {x["job"]["tag"] for x in chosen} == {"dense_base_s1234", "analysis_only_s1234"}


def test_incomplete_search_cannot_select():
    assert report.validation_selection([item(job())], [{"tag": "other"}]) == []


def test_summary_must_match_registered_budget():
    cfg = job()
    stored = summary(cfg)
    stored["training"]["steps"] = 5
    with pytest.raises(ValueError, match="training steps"):
        report.match_job(cfg, stored)


def test_baseline_sigma_fits_validation_each_depth_variable():
    arrays = predictions()
    cal = report.calibrate_baseline(arrays)
    for level in (0, 2):
        mask = arrays["level"] == level
        expected = np.sqrt(np.mean((arrays["baseline"][mask] - arrays["target"][mask]) ** 2, axis=0))
        np.testing.assert_allclose(cal["sigma_z"][level], expected)
    np.testing.assert_allclose(cal["sigma_z"][1], cal["fallback_sigma_z"])
    np.testing.assert_array_equal(cal["counts"][1], [0, 0])
    assert cal["split"] == "validation_2021"


def test_calibration_application_cannot_refit_to_development_errors():
    cal = report.calibrate_baseline(predictions())
    levels = np.array([0, 1, 2, 19])
    sigma = report.calibrated_std(cal, levels)
    np.testing.assert_allclose(sigma[-1], cal["fallback_sigma_z"])
    assert np.all(sigma > 0)


def test_gaussian_metrics_known_coverage_and_nll():
    target = np.array([[0., 0.], [np.nan, 0.]])
    result = report.gaussian_metrics(np.zeros((2, 2)), np.ones((2, 2)), target)
    assert result["TEMP"]["n"] == 1
    assert result["SALT"]["n"] == 2
    assert result["TEMP"]["nll"] == pytest.approx(.5 * np.log(2 * np.pi))
    assert result["TEMP"]["coverage_68"] == 1


def test_ensemble_uses_total_variance_and_streams(tmp_path):
    paths = []
    for seed, mean in enumerate((0., 2.)):
        path = tmp_path / f"seed{seed}.npz"
        np.savez(path, **predictions(mean=mean, std=1.))
        paths.append(path)
    arrays = report.ensemble_predictions(paths)
    np.testing.assert_allclose(arrays["mean"], 1.)
    np.testing.assert_allclose(arrays["std"], np.sqrt(2.))


@pytest.mark.parametrize("field", report.IDENTITY_KEYS)
def test_ensemble_rejects_every_identity_mismatch(tmp_path, field):
    first, second = predictions(), predictions()
    second[field].flat[0] += 1
    paths = [tmp_path / "first.npz", tmp_path / "second.npz"]
    for path, arrays in zip(paths, (first, second)):
        np.savez(path, **arrays)
    with pytest.raises(ValueError, match=field):
        report.ensemble_predictions(paths)


def test_seed_identity_matches_nan_targets():
    arrays = predictions()
    arrays["target"][0, 0] = np.nan
    report.assert_identical(arrays, {k: v.copy() for k, v in arrays.items()})


def test_bootstrap_pairs_months_and_uses_pooled_scores():
    arrays = predictions()
    arrays["mean"] = arrays["target"].copy()
    stats = report.paired_month_bootstrap(arrays, draws=500, seed=8)
    base = report.z_scores(arrays["baseline"], arrays["target"])
    assert stats["n_months"] == 3
    assert stats["channels"]["TEMP"]["delta_J"] == pytest.approx(-base["TEMP"]["J"])
    assert stats["channels"]["TEMP"]["ci95_delta_J"][1] < 0
    assert stats == report.paired_month_bootstrap(arrays, draws=500, seed=8)


def test_single_seed_spread_not_reported_as_zero():
    stats = report.seed_statistics([summary(job())], "development")
    assert stats["n_seeds"] == 1
    assert stats["TEMP_J"]["std"] is None


def test_bootstrap_zero_anomalies_has_finite_climatology_floor():
    arrays = predictions()
    arrays["target"][:] = 0
    arrays["mean"][:] = 1
    arrays["baseline"][:] = 2
    result = report.paired_month_bootstrap(arrays, draws=20)
    assert result["channels"]["TEMP"]["delta_J"] == pytest.approx(-1e9)
    assert np.all(np.isfinite(result["channels"]["TEMP"]["ci95_delta_J"]))


def test_running_report_excludes_unregistered_integration(tmp_path):
    outputs = tmp_path / "outputs"
    cfg = job()
    save_item(outputs, job("integration_dense", width=64))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"jobs": [cfg]}))
    result = report.build_report(manifest, outputs, draws=20)
    assert result["completed_count"] == 0
    assert result["registered_count"] == 1
    assert result["selected_tags"] == []
    text = report.render_report(result)
    assert "未完成" in text and "integration_dense" not in text
    assert "20 个固定深度" in text
    assert "原位观测" in text
    json.dumps(result, allow_nan=False)


def test_replication_requires_same_training_protocol(tmp_path):
    outputs = tmp_path / "outputs"
    cfg = job()
    folder = save_item(outputs, cfg)
    selected = [{"job": cfg, "summary": summary(cfg), "folder": folder}]
    replicate_cfg = job("dense_base_s1235", seed=1235)
    replicate_folder = save_item(outputs, replicate_cfg)
    path = replicate_folder / "summary_seed1235.json"
    stored = json.loads(path.read_text())
    stored["config"]["use_latent"] = False
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="protocol differs"):
        report.collect_replicates(selected, outputs)


def test_finished_seed_report_builds_calibration_and_ensemble(tmp_path):
    outputs = tmp_path / "outputs"
    cfg = job()
    for seed in (1234, 1235):
        this_cfg = job(f"dense_base_s{seed}", seed=seed)
        folder = save_item(outputs, this_cfg)
        np.savez(folder / f"validation_seed{seed}.npz", **predictions())
        np.savez(folder / f"development_seed{seed}.npz", **predictions(mean=float(seed - 1234)))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"jobs": [cfg]}))
    result = report.build_report(manifest, outputs, draws=20)
    group = result["groups"][0]
    assert group["seed_statistics"]["development"]["n_seeds"] == 2
    assert group["ensemble"]["n_seeds"] == 2
    assert group["baseline_calibration"]["split"] == "validation_2021"
    assert len(group["ensemble"]["by_level"]) == 2
    assert len(group["development_details"]) == 2
    json.dumps(result, allow_nan=False)


def test_ablation_does_not_enter_capacity_selection(tmp_path):
    outputs = tmp_path / "outputs"
    cfg = job()
    save_item(outputs, cfg, macro=.5)
    ablation_cfg = job("dense_ablation_no_latent_s1234")
    ablation_cfg["latent_off"] = True
    folder = save_item(outputs, ablation_cfg, macro=.1)
    path = folder / "summary_seed1234.json"
    stored = json.loads(path.read_text())
    stored["training"]["latent_off"] = True
    path.write_text(json.dumps(stored))
    manifest, ablations = tmp_path / "manifest.json", tmp_path / "ablation.json"
    manifest.write_text(json.dumps({"jobs": [cfg]}))
    ablations.write_text(json.dumps({"jobs": [ablation_cfg]}))
    result = report.build_report(manifest, outputs, draws=20, ablation_manifest=ablations)
    assert result["selected_tags"] == [cfg["tag"]]
    assert result["ablations"][0]["summary"]["tag"] == ablation_cfg["tag"]
