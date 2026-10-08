"""Completion, evidence attribution and honest final-backbone reporting."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/synthetic/58_final_research_report.py"
spec = importlib.util.spec_from_file_location("final_research_report_under_test", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def physical(error=.2, *, uncertainty=True):
    output = {}
    for ch, scale in (("TEMP", 1.), ("SALT", .1)):
        m = {"n": 351895, "rmse": error * scale, "mae": .7 * error * scale,
             "mean_bias": .1 * error * scale, "r2": .9, "pearson_r": .96,
             "absolute_r2": .99, "absolute_pearson_r": .995}
        if uncertainty:
            m.update(nll=-1., crps=.5 * error * scale, coverage_68=.68,
                     coverage_95=.95, mean_std=error * scale)
        output[ch] = m
    return output


def depths(metrics):
    output = {}
    for i, depth in enumerate((5,15,25,35,45,55,65,85,105,125,145,165.1,186.3,222.6,267.7,326.9,408.8,527.7,707.6,984.7)):
        output[str(float(depth))] = {ch: {**m, "n": 17595 if i < 15 else 17594} for ch, m in metrics.items()}
    return output


def contrast(delta=-.01, *, ci=None):
    return {ch: {"delta_rmse": delta * scale, "ci95_delta_rmse":
                 [value * scale for value in (ci or (delta - .005, delta + .005))]}
            for ch, scale in (("TEMP", 1.), ("SALT", .1))}


def literature():
    return {"papers": [{"id": "near_ocean", "title": "Direct ocean reconstruction source", "year": 2025,
                         "url": "https://arxiv.org/abs/2511.06041", "method_summary": "表层观测融合",
                         "overlap_with_ours": "共享表示", "important_difference": "未证实三维温盐",
                         "evidence_needed": "相同观测实验", "risk_level": "高"}],
            "claim_audit": [{"claim": "共享latent原创", "status": "不足", "reason": "已有来源",
                             "required_evidence": "新的更新机制", "related_paper_ids": ["near_ocean"]}]}


def completed_fixture(tmp_path, *, selected_family="prior_learned_oi"):
    output = tmp_path / "campaign"
    output.mkdir()
    groups = {"argo": ("previous_token64", "dense64", "dense192", "soft_moe192",
                       "local_transformer192", "soft_moe192_latent_off", "soft_moe192_local_off", "official4dvarnet"),
              "surface": ("previous_token64", "dense192", "soft_moe192", "soft_moe192_latent_off", "official4dvarnet")}
    jobs, runs, rows = [], {}, []
    for mode, families in groups.items():
        for family in families:
            kind = "previous" if family == "previous_token64" else "official" if family == "official4dvarnet" else "latent"
            error = .5 if family == "previous_token64" else .25
            m = physical(error)
            seed_metrics = [{"seed": seed, "physical_metrics": m, "by_depth_physical_metrics": depths(m)} for seed in report.SEEDS]
            seed_mean_sd = {ch: {key: {"mean": value, "sd": 0.} for key, value in channel.items() if key != "n"}
                            for ch, channel in m.items()}
            rows.append({"mode": mode, "family": family, "kind": kind, "seeds": list(report.SEEDS),
                         "seed_metrics": seed_metrics, "seed_mean_sd": seed_mean_sd,
                         "ensemble": {"physical_metrics": m, "by_depth_physical_metrics": depths(m)},
                         "uncertainty_method": "learned Gaussian"})
            for seed in report.SEEDS:
                tag = f"{family}_{mode}_s{seed}"
                folder = output / tag
                checkpoint = folder / f"best_seed{seed}.pt"
                checkpoint.parent.mkdir()
                checkpoint.write_bytes(b"frozen-model")
                config = {"n_obs_features": 62, "width": 192, "n_latents": 96, "n_heads": 8,
                          "n_blocks": 6, "n_query_blocks": 2, "n_experts": 4, "slots_per_expert": 2,
                          "variant": "soft_moe" if family.startswith("soft_moe") else "dense",
                          "use_latent": not family.endswith("latent_off"),
                          "use_local": not family.endswith("local_off"), "n_sat_features": 56 if mode == "surface" else 0}
                contract = {"seed": seed, "completed_steps": 15000, "training": {"steps": 15000},
                            "config": config, "params": 123, "best_step": 9000}
                summary = write(folder / f"summary_seed{seed}.json", contract)
                runs[tag] = {"training_contract": contract, "summary_path": str(summary),
                             "checkpoint_path": str(checkpoint), "checkpoint_sha256": report.sha256(checkpoint)}
                jobs.append({"tag": tag, "family": family, "kind": kind, "seed": seed,
                             "surface": mode == "surface", "steps": 15000, "output": str(folder)})
    for family, error in (("oi", .3), ("prior_learned_oi", .2), ("fixed_pointwise_mlp", .4), ("nearest_profile", .45), ("climatology", .7)):
        m = physical(error, uncertainty=family in ("oi", "prior_learned_oi"))
        rows.append({"mode": "argo", "family": family, "kind": "classical",
                     "seeds": [1234] if family in ("prior_learned_oi", "fixed_pointwise_mlp") else [],
                     "physical_metrics": m, "by_depth_physical_metrics": depths(m),
                     "uncertainty_method": "validation residual RMS" if family in ("oi", "prior_learned_oi") else "unavailable"})
    campaign_path = write(output / "campaign.json", {"jobs": jobs, "training_steps": 15000, "seeds": list(report.SEEDS)})
    prior_checkpoint = output / "prior.pt"; prior_checkpoint.write_bytes(b"old-oi")
    prior_summary = write(output / "prior_summary.json", {"first_guess": {"kind": "none"}, "weights": "kriging",
                            "k": 32, "use_time": False, "use_state": False, "lr": .02, "learned": {"ell_x": {"TEMP": {"0-100m": 416.}}}})
    selected_kind = "existing_learned_oi" if selected_family == "prior_learned_oi" else "classical" if selected_family == "oi" else "latent"
    choices = {mode: {"family": selected_family, "kind": selected_kind, "validation_mean_standardized_rmse": .2}
               for mode in groups}
    registry = {
                     "source_checkpoint": {"path": str(prior_checkpoint), "sha256": report.sha256(prior_checkpoint)},
                     "historical_summary": {"path": str(prior_summary), "sha256": report.sha256(prior_summary)}}
    registry_path = write(output / "prior_registry.json", registry)
    mlp_checkpoint = output / "fixed_mlp.pt"; mlp_checkpoint.write_bytes(b"fixed-mlp")
    selection = {"selected": choices, "runs": runs, "campaign_sha256": report.sha256(campaign_path),
                 "auxiliary_prior_learned_oi": {"parameters": 120, "registry": registry,
                     "registry_path": str(registry_path), "registry_sha256": report.sha256(registry_path),
                     "checkpoint_sha256": report.sha256(prior_checkpoint),
                     "historical_summary_sha256": report.sha256(prior_summary)},
                 "auxiliary_fixed_mlp": {"checkpoint_path": str(mlp_checkpoint), "checkpoint_sha256": report.sha256(mlp_checkpoint)}}
    selection_path = write(output / "selection.json", selection)
    component = contrast(-.03)
    component["paired_seed_rmse"] = {ch: {"per_seed_delta_rmse": [-.02, -.03, -.04]} for ch in report.CHANNELS}
    metric = {"selection": selection, "selection_file_sha256": report.sha256(selection_path), "rows": rows,
              "test_n_per_variable": {"TEMP": 351895, "SALT": 351895}, "levels_m": [float(d) for d in depths(physical())],
              "metric_source_sha256": report.sha256(report.ROOT / "src/ocean_tokenizer/reconstruction_metrics.py"),
              "reporter_source_sha256": report.sha256(report.ROOT / "experiments/synthetic/53_matched_reconstruction_report.py"),
              "development_array_sha256": {}, "limits": ["20 fixed depths"], "contrasts": {},
              "selected_against_previous_best": {mode: {"against_prior_learned_oi": contrast(0., ci=(0., 0.))} for mode in groups},
              "component_contrasts": {f"{mode}:full_minus_latent_off": component for mode in groups},
              "multimodal_contrasts": {}}
    for row in rows:
        if row["family"] != "oi":
            delta = row["ensemble"]["physical_metrics"]["TEMP"]["rmse"] if row["kind"] != "classical" else row["physical_metrics"]["TEMP"]["rmse"]
            metric["contrasts"][f"{row['mode']}:{row['family']}"] = {
                "against_oi": contrast(delta - .3), "against_previous_token64": contrast(delta - .5),
                "against_prior_learned_oi": contrast(delta - .2)}
    metrics_path = write(output / "metrics.json", metric)
    matched_report = tmp_path / "matched_report.md"; matched_report.write_text("Completed matched metrics")
    completion = {"phase": "complete", "training_runs": 39, "steps_per_run": 15000,
                  "metrics_sha256": report.sha256(metrics_path), "selection_sha256": report.sha256(selection_path),
                  "report_sha256": report.sha256(matched_report), "selected": choices}
    write(output / "completion.json", completion)
    protocol = tmp_path / "protocol.md"; protocol.write_text("exact protocol")
    real = write(tmp_path / "real.json", {"split": "previously used real development 2022–2023",
            "frozen_selection": {"selected_configuration": "soft_moe_large"}, "baseline": {"pooled": physical(.8)},
            "groups": [{"configuration": "soft_moe_large", "ensemble": {"pooled": physical(.79),
                        "paired_month_rmse_bootstrap": contrast(-.01)}, "seed_statistics": {}}]})
    return output, metrics_path, selection_path, matched_report, metric, completion, protocol, real


def test_completion_missing_is_pending_and_no_model_array_open(tmp_path, monkeypatch):
    monkeypatch.setattr(np, "load", lambda *_a, **_kw: pytest.fail("Must not open development before completion"))
    with pytest.raises(report.PendingEvidence):
        report.verify_completed(tmp_path, tmp_path / "metrics.json", tmp_path / "selection.json", tmp_path / "report.md")


def test_exact_39_run_contract_and_hashes_are_verified(tmp_path):
    output, metrics, selection, matched, _, _, _, _ = completed_fixture(tmp_path)
    completion, campaign, m = report.verify_completed(output, metrics, selection, matched)
    assert completion["training_runs"] == len(campaign["jobs"]) == 39
    assert m["test_n_per_variable"] == {"TEMP": 351895, "SALT": 351895}


@pytest.mark.parametrize("changed", ["metrics", "selection", "report", "checkpoint", "prior_checkpoint", "prior_summary", "prior_registry", "mlp"])
def test_changed_evidence_is_rejected(tmp_path, changed):
    output, metrics, selection, matched, _, _, _, _ = completed_fixture(tmp_path)
    path = {"metrics": metrics, "selection": selection, "report": matched,
            "prior_checkpoint": output / "prior.pt", "prior_summary": output / "prior_summary.json",
            "prior_registry": output / "prior_registry.json", "mlp": output / "fixed_mlp.pt",
            "checkpoint": Path(json.loads(selection.read_text())["runs"]["dense64_argo_s1234"]["checkpoint_path"])}[changed]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        report.verify_completed(output, metrics, selection, matched)


def test_current_metric_source_differs_rejected(tmp_path):
    output, metrics, selection, matched, m, completion, _, _ = completed_fixture(tmp_path)
    m["metric_source_sha256"] = "f" * 64
    write(metrics, m)
    completion["metrics_sha256"] = report.sha256(metrics)
    write(output / "completion.json", completion)
    with pytest.raises(ValueError, match="metric source"):
        report.verify_completed(output, metrics, selection, matched)


@pytest.mark.parametrize("mutation", ["steps", "count", "seed"])
def test_incomplete_scientific_contract_rejected(tmp_path, mutation):
    output, metrics, selection, matched, m, completion, _, _ = completed_fixture(tmp_path)
    if mutation == "steps":
        m["selection"]["runs"]["dense64_argo_s1234"]["training_contract"]["completed_steps"] = 14500
    elif mutation == "count":
        m["test_n_per_variable"]["TEMP"] = 351894
    else:
        next(row for row in m["rows"] if row["family"] == "dense64")["seeds"] = [1234, 1235]
    write(selection, m["selection"])
    m["selection_file_sha256"] = report.sha256(selection)
    write(metrics, m)
    completion.update(metrics_sha256=report.sha256(metrics), selection_sha256=report.sha256(selection))
    write(output / "completion.json", completion)
    with pytest.raises(ValueError):
        report.verify_completed(output, metrics, selection, matched)


def test_strong_prior_win_does_not_force_transformer_or_satellite(tmp_path):
    output, _, _, _, m, completion, protocol, real = completed_fixture(tmp_path)
    campaign = json.loads((output / "campaign.json").read_text())
    payload = report.build_payload(m, campaign, completion, literature(), protocol_path=protocol, real_path=real)
    assert payload["selected_architectures"]["argo"]["family"] == "prior_learned_oi"
    assert payload["selected_architectures"]["surface"]["uses_satellite"] is False
    assert payload["selected_architectures"]["argo"]["model_configs"][0]["kernel_parameters_summary"]
    assert payload["selected_architectures"]["argo"]["weights"][0]["path"].endswith("prior.pt")
    text = report.render_report(payload)
    assert "最终保留既有 learned OI" in text
    assert "共享 latent 的必要性未被当前选型证明" in text
    assert "j score" not in text.lower()
    assert "[Direct ocean reconstruction source]" in text
    assert "120个参数" in text
    assert "尚未确立能够取代它的新latent架构贡献" in text
    assert "[95% CI]" in text


def test_soft_moe_ablation_cannot_establish_dense_latent_necessity(tmp_path):
    _, _, _, _, m, _, _, _ = completed_fixture(tmp_path, selected_family="dense192")
    gate = report.evidence_gates(m)["argo"]
    assert gate["soft_moe_latent_control"]["stable_gain_in_both_variables"] is True
    assert gate["component_control_applies_to_selected_backbone"] is False
    assert gate["selected_latent_necessity_supported"] is False
    assert gate["sota_established"] is False


def test_latent_off_winner_has_no_executed_latent_diagram(tmp_path):
    output, _, _, _, m, completion, protocol, real = completed_fixture(tmp_path, selected_family="soft_moe192_latent_off")
    payload = report.build_payload(m, json.loads((output / "campaign.json").read_text()), completion,
                                   literature(), protocol_path=protocol, real_path=real)
    diagram = report.architecture_diagram(payload["selected_architectures"]["surface"])
    assert "E --> L" not in diagram
    assert "A --> E" not in diagram
    assert "S --> D" in diagram
    text = report.render_report(payload)
    assert "最终选择关闭 latent" in text
    assert "推理执行路径绕过共享latent" in text


def test_old_winner_boolean_historical_setting_is_not_a_backbone_config(tmp_path):
    output, _, _, _, m, _, _, _ = completed_fixture(tmp_path, selected_family="previous_token64")
    for c in m["selection"]["runs"].values():
        if "previous_token64" in c["checkpoint_path"]:
            c["training_contract"].pop("config")
            c["training_contract"].update(historical_setting=True, n_latent=64, n_slots=4,
                                          backbone="d4rt", mass_mode="dfs", refiner_km=500., refiner_gate=1.)
    architecture = report.selected_architectures(m, json.loads((output / "campaign.json").read_text()))["argo"]
    assert all(isinstance(c, dict) for c in architecture["model_configs"])
    assert architecture["model_configs"][0]["n_latents"] == 64
    assert architecture["model_configs"][0]["profile_depth_bands"] == 4


def test_local_transformer_is_drawn_in_neighbor_path_not_latent(tmp_path):
    output, _, _, _, m, _, _, _ = completed_fixture(tmp_path, selected_family="dense192")
    architecture = report.selected_architectures(m, json.loads((output / "campaign.json").read_text()))["argo"]
    architecture["family"] = "local_transformer192"
    architecture["model_configs"][0]["variant"] = "local_transformer"
    diagram = report.architecture_diagram(architecture)
    assert 'L["Dense latent Transformer"]' in diagram
    assert "N --> LT" in diagram
    assert "D --> LT" in diagram
    assert "ES --> U" in diagram


def test_negative_point_estimate_with_ci_crossing_zero_is_not_supported():
    value = contrast(-.001, ci=(-.02, .01))
    assert report.all_negative_ci(value) is False


def test_band_rmse_pools_squared_errors_and_counts():
    m1 = {ch: {"n": 1, "rmse": 1., "mae": 1., "mean_bias": -1.} for ch in report.CHANNELS}
    m2 = {ch: {"n": 3, "rmse": 3., "mae": 3., "mean_bias": 3.} for ch in report.CHANNELS}
    bands = report.aggregate_bands({"5": m1, "15": m2})
    temp = bands["0–100 m"]["TEMP"]
    assert temp["n"] == 4
    assert temp["rmse"] == pytest.approx(np.sqrt(7))
    assert temp["rmse"] != 2.
    assert temp["mae"] == temp["mean_bias"] + .5 == 2.5
    assert "r2" not in temp


def test_classic_surface_contrast_uses_surface_old_model(tmp_path):
    _, _, _, _, m, _, _, _ = completed_fixture(tmp_path)
    surface_old = report.row_for(m, "surface", "previous_token64")
    surface_old["ensemble"]["physical_metrics"] = physical(.6)
    value = report.selected_contrast(m, "surface", "prior_learned_oi", "previous_token64")
    assert value["TEMP"]["delta_rmse"] == pytest.approx(-.4)
    assert value["TEMP"]["ci95_delta_rmse"] is None
    assert m["contrasts"]["argo:prior_learned_oi"]["against_previous_token64"]["TEMP"]["delta_rmse"] == pytest.approx(-.3)


def test_literature_requires_complete_comparison_evidence():
    with pytest.raises(report.PendingEvidence):
        report.validate_literature({"papers": [], "claim_audit": []})
    value = literature()
    del value["papers"][0]["important_difference"]
    with pytest.raises(ValueError):
        report.validate_literature(value)


def test_improvement_fraction_uses_the_reference_rmse():
    learned = {"kind": "classical", "physical_metrics": physical(.15)}
    base = {"kind": "classical", "physical_metrics": physical(.2)}
    change = report.compare_rows(learned, base)
    assert change["TEMP"]["relative_rmse_reduction_percent"] == pytest.approx(25.)
    assert change["SALT"]["delta_rmse"] == pytest.approx(-.005)
