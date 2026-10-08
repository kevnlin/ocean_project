"""Measured plotting contracts: evidence, frozen selection and missing results."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/real_data/75_latent_figures.py"
spec = importlib.util.spec_from_file_location("latent_figures", SCRIPT)
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)


def job(tag="dense_base_s1234", variant="dense", seed=1234, **flags):
    return {"tag": tag, "variant": variant, "width": 128, "latents": 64, "blocks": 4,
            "experts": 4, "context_profiles": 512, "seed": seed, "steps": 6000, **flags}


def scores(temp=.4, salt=.6):
    return {"TEMP": {"J": temp, "rmse_z": temp, "n": 60},
            "SALT": {"J": salt, "rmse_z": salt, "n": 60}, "macro_z": (temp + salt) / 2}


def summary(cfg, macro=.5):
    return {"tag": cfg["tag"], "seed": cfg["seed"], "params": 1000, "best_step": 1000,
            "config": {"variant": cfg["variant"], "width": cfg["width"], "n_latents": cfg["latents"],
                       "n_blocks": cfg["blocks"], "n_experts": cfg["experts"]},
            "training": {**cfg, "analysis_only": bool(cfg.get("analysis_only"))},
            "initial": {"scores": {"macro_z": .7}},
            "history": [{"step": 1000, "validation_macro_z": macro},
                        {"step": 6000, "validation_macro_z": macro + .01}],
            "validation": {"scores": {"macro_z": macro}},
            "development": {"scores": scores()}}


def details(temp=.4, salt=.6):
    return {"scores": scores(temp, salt), "baseline_scores": scores(.5, .8)}


def complete_report():
    jobs = [job(), job("soft_moe_base_s1234", "soft_moe"),
            job("local_transformer_base_s1234", "local_transformer"),
            job("analysis_only_s1234", analysis_only=True)]
    search = [{"job": cfg, "summary": summary(cfg, macro=.2 + i * .1)} for i, cfg in enumerate(jobs)]
    groups = []
    for i, cfg in enumerate(jobs):
        groups.append({"configuration": F.prefix(cfg["tag"]),
                       "development_details": {f"seed{seed}": details(temp=.7 - .1 * i + (seed - 1234) * .001)
                                               for seed in F.SEEDS},
                       "ensemble": {"n_seeds": 3,
                                    "by_level": [{"level_index": k, **details(temp=.7 - .1 * i + .001 * k)}
                                                 for k in range(20)]}})
    ablation_jobs, controls = [], []
    for name, flags in (("full", {}), ("latent_off", {"latent_off": True}),
                        ("local_off", {"local_off": True})):
        for seed in F.SEEDS:
            cfg = job(f"dense_base_fixed_oi_{name}_s{seed}", seed=seed, freeze_analysis=True, **flags)
            ablation_jobs.append(cfg)
            controls.append({"job": cfg, "summary": summary(cfg), "development_details": details()})
    report = {"schema_version": 1, "registered_count": len(jobs), "completed_count": len(jobs),
              "search": search, "pending": [], "selected_tags": [cfg["tag"] for cfg in jobs],
              "groups": groups, "ablations": controls, "ablation_pending": []}
    return (report, {"jobs": jobs}, {"selected": jobs},
            {"selected_neural": jobs[0]["tag"], "jobs": ablation_jobs}, np.linspace(5., 985., 20))


def test_curve_retains_only_actual_evaluations_and_step_zero_can_win():
    cfg = job()
    stored = summary(cfg)
    stored["initial"]["scores"]["macro_z"] = .3
    stored["best_step"] = 0
    stored["validation"]["scores"]["macro_z"] = .3
    curve = F.measured_curve(stored, cfg)
    assert curve["steps"] == [0, 1000, 6000]
    assert curve["macro_z"] == [.3, .5, .51]
    assert curve["best_step"] == 0


def test_completed_summary_cannot_disguise_short_training():
    cfg = job()
    stored = summary(cfg)
    stored["history"].pop()
    with pytest.raises(ValueError, match="registered budget"):
        F.measured_curve(stored, cfg)


def test_ratio_spread_uses_each_channels_own_anchor_and_sample_std():
    stored = {f"seed{seed}": details(temp=temp, salt=salt)
              for seed, temp, salt in zip(F.SEEDS, (.4, .5, .6), (.4, .8, 1.2))}
    result = F.aggregate_details(stored)
    assert result["channels"]["TEMP"]["ratio_mean"] == pytest.approx(1.)
    assert result["channels"]["TEMP"]["ratio_std"] == pytest.approx(.2)
    assert result["channels"]["SALT"]["ratio_mean"] == pytest.approx(1.)
    assert result["channels"]["SALT"]["ratio_std"] == pytest.approx(.5)


def test_missing_seed_omits_aggregate_instead_of_zero_spread():
    assert F.aggregate_details({"seed1234": details(), "seed1235": details()}) is None
    inputs = list(complete_report())
    del inputs[0]["groups"][0]["development_details"]["seed1236"]
    data = F.build_figure_data(*inputs)
    assert data["configurations"][0]["data"] is None
    assert "three-seed development: dense_base" in data["pending"]
    json.dumps(data, allow_nan=False)


def test_depth_uses_validation_winner_even_when_its_development_is_worst():
    data = F.build_figure_data(*complete_report())
    assert data["selected_neural"] == "dense_base_s1234"
    assert data["depth"]["configuration"] == "dense_base"
    assert data["depth"]["channels"]["TEMP"]["delta_J"][0] == pytest.approx(.2)
    assert not data["pending"]


def test_mechanism_selection_cannot_follow_development_winner():
    inputs = list(complete_report())
    inputs[3]["selected_neural"] = "local_transformer_base_s1234"
    with pytest.raises(ValueError, match="validation-selected"):
        F.build_figure_data(*inputs)


def test_integration_run_cannot_enter_the_registered_figure():
    inputs = list(complete_report())
    cfg = job("integration_dense")
    inputs[0]["search"].append({"job": cfg, "summary": summary(cfg)})
    inputs[0]["completed_count"] += 1
    with pytest.raises(ValueError, match="unregistered"):
        F.build_figure_data(*inputs)


def test_seed_anchor_mismatch_is_rejected_before_plotting():
    stored = {f"seed{seed}": details() for seed in F.SEEDS}
    stored["seed1236"]["baseline_scores"]["SALT"]["J"] += .01
    with pytest.raises(ValueError, match="matched anchor"):
        F.aggregate_details(stored)


def test_final_depth_plot_requires_the_twenty_observed_levels():
    inputs = list(complete_report())
    inputs[0]["groups"][0]["ensemble"]["by_level"].pop()
    with pytest.raises(ValueError, match="20 observed"):
        F.build_figure_data(*inputs)


def test_progress_figure_is_explicit_and_svg_keeps_text_editable(tmp_path):
    report = {"schema_version": 1, "registered_count": 1, "completed_count": 0,
              "search": [], "pending": [{"tag": "dense_base_s1234"}],
              "selected_tags": [], "groups": [], "ablations": []}
    data = F.build_figure_data(report, {"jobs": [job()]}, None, None, np.linspace(5., 985., 20))
    assert data["curves"] == []
    assert data["depth"] is None
    paths = F.render_figure(data, tmp_path / "figure")
    assert all(path.exists() and path.stat().st_size > 1000 for path in paths)
    svg = paths[1].read_text()
    assert "INCOMPLETE" in svg and "Pending selected-neural ensemble" in svg
    assert "<text" in svg
    assert "not an independent test" in svg


def test_stale_comparison_cannot_supply_curves_from_uncompleted_files(tmp_path):
    report, manifest, selection, ablations, levels = complete_report()
    comparison_path, manifest_path = tmp_path / "comparison.json", tmp_path / "manifest.json"
    comparison_path.write_text(json.dumps(report))
    manifest_path.write_text(json.dumps(manifest))
    levels_path = tmp_path / "levels.npy"
    np.save(levels_path, levels)
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps({"arrays": {"cohort.levels": {
        "path": "levels.npy", "sha256": F.source_record(levels_path)["sha256"]}}}))
    with pytest.raises(ValueError, match="completed source summary"):
        F.load_inputs(comparison_path, manifest_path, tmp_path / "selection.json",
                      tmp_path / "ablations.json", cache_path, tmp_path / "outputs")
