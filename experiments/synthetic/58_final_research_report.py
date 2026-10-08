"""Write the completed, evidence-gated Chinese ocean reconstruction report.

This is a reporting step, not training or model selection. It opens no new model
development artifact before the 57 completion guard has verified the campaign.
Missing completed evidence returns exit 2; inconsistent evidence is an error.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/synthetic_matched_20261007"
CHANNELS = ("TEMP", "SALT")
UNITS = {"TEMP": "°C", "SALT": "PSU"}
BANDS = (("0–100 m", 0., 100.), ("100–300 m", 100., 300.),
         ("300–700 m", 300., 700.), ("700–1,000 m", 700., 1000.))
SEEDS = (1234, 1235, 1236)


class PendingEvidence(RuntimeError):
    """The completed campaign or sealed literature review does not exist yet."""


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise PendingEvidence(f"Required evidence is pending: {path}")
    return json.loads(path.read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_script(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/synthetic" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_completed(output, metrics_path, selection_path, report_path):
    """Completion is a scientific prerequisite, not a file-existence shortcut."""
    output = Path(output)
    completion = read_json(output / "completion.json")
    if completion.get("phase") != "complete" or completion.get("training_runs") != 39:
        raise PendingEvidence("All 39 registered training runs must be completed")
    if completion.get("steps_per_run") != 15000:
        raise ValueError("Completed training budget differs from 15000 steps")
    for key, path in (("metrics_sha256", metrics_path), ("selection_sha256", selection_path),
                      ("report_sha256", report_path)):
        if not Path(path).is_file():
            raise PendingEvidence(f"Completion references a missing artifact: {path}")
        if sha256(path) != completion.get(key):
            raise ValueError(f"Completion evidence changed: {key}")
    metrics = read_json(metrics_path)
    manifest = read_json(output / "campaign.json")
    selection = read_json(selection_path)
    if metrics.get("selection_file_sha256") != sha256(selection_path):
        raise ValueError("Metrics refer to a different frozen selection")
    if metrics.get("selection") != selection or completion.get("selected") != selection.get("selected"):
        raise ValueError("Completed, scored and frozen architecture choices disagree")
    if selection.get("campaign_sha256") != sha256(output / "campaign.json"):
        raise ValueError("Campaign manifest changed after validation selection")
    helper_path = ROOT / "src/ocean_tokenizer/reconstruction_metrics.py"
    if metrics.get("metric_source_sha256") != sha256(helper_path):
        raise ValueError("Physical metric source differs from the scored experiment")
    reporter_path = ROOT / "experiments/synthetic/53_matched_reconstruction_report.py"
    if metrics.get("reporter_source_sha256") != sha256(reporter_path):
        raise ValueError("Matched reporting source differs from the scored experiment")
    finalizer = load_script("57_finalize_matched_campaign.py", "final_report_completion_contract")
    finalizer.verify(metrics, manifest)
    for tag, contract in selection["runs"].items():
        if sha256(contract["checkpoint_path"]) != contract["checkpoint_sha256"]:
            raise ValueError(f"Frozen best model changed: {tag}")
    prior = selection["auxiliary_prior_learned_oi"]
    for info, fingerprint in ((prior["registry"]["source_checkpoint"], prior["checkpoint_sha256"]),
                              (prior["registry"]["historical_summary"], prior["historical_summary_sha256"]),
                              ({"path": prior["registry_path"], "sha256": prior["registry_sha256"]}, prior["registry_sha256"])):
        if info["sha256"] != fingerprint or sha256(info["path"]) != fingerprint:
            raise ValueError(f"Frozen auxiliary learned OI evidence changed: {info['path']}")
    mlp = selection["auxiliary_fixed_mlp"]
    if sha256(mlp["checkpoint_path"]) != mlp["checkpoint_sha256"]:
        raise ValueError("Frozen fixed pointwise MLP checkpoint changed")
    for path, fingerprint in metrics.get("development_array_sha256", {}).items():
        if not Path(path).is_file():
            raise PendingEvidence(f"Scored prediction is missing: {path}")
        if sha256(path) != fingerprint:
            raise ValueError(f"Scored prediction changed: {path}")
    return completion, manifest, metrics


def validate_literature(literature):
    papers = literature.get("papers")
    claims = literature.get("claim_audit")
    if not isinstance(papers, list) or not papers or not isinstance(claims, list) or not claims:
        raise PendingEvidence("The literature similarity audit is not complete")
    required = ("id", "title", "year", "url", "method_summary", "overlap_with_ours",
                "important_difference", "evidence_needed", "risk_level")
    for paper in papers:
        if any(key not in paper for key in required):
            raise ValueError("Literature audit lacks the required evidence fields")
        if not paper["url"].startswith("https://"):
            raise ValueError("Literature citations require primary source links")
    if len({paper["id"] for paper in papers}) != len(papers):
        raise ValueError("Literature audit paper identifiers must be unique")
    return literature


def row_for(metrics, mode, family):
    matching = [row for row in metrics["rows"] if row["family"] == family and row["mode"] == mode]
    if not matching:
        matching = [row for row in metrics["rows"] if row["family"] == family
                    and row["mode"] == "argo" and row["kind"] == "classical"]
    if len(matching) != 1:
        raise ValueError(f"Selected/comparison method is missing or ambiguous: {mode}:{family}")
    return matching[0]


def pooled(row):
    return row["physical_metrics"] if row["kind"] == "classical" else row["ensemble"]["physical_metrics"]


def per_depth(row):
    return row["by_depth_physical_metrics"] if row["kind"] == "classical" else row["ensemble"]["by_depth_physical_metrics"]


def compare_rows(row, baseline):
    actual, reference = pooled(row), pooled(baseline)
    return {ch: {"delta_rmse": actual[ch]["rmse"] - reference[ch]["rmse"],
                 "relative_rmse_reduction_percent": 100 * (1 - actual[ch]["rmse"] / reference[ch]["rmse"])}
            for ch in CHANNELS}


def zero_contrast():
    return {ch: {"delta_rmse": 0., "ci95_delta_rmse": [0., 0.]} for ch in CHANNELS}


def selected_contrast(metrics, mode, family, baseline):
    extra = metrics.get("final_selected_contrasts", {}).get(mode, {}).get(baseline)
    if extra is not None:
        return extra
    if family == baseline:
        return zero_contrast()
    row = row_for(metrics, mode, family)
    key = f"{row['mode']}:{family}"
    contrasts = metrics["contrasts"].get(key, {})
    if baseline == "prior_learned_oi":
        return metrics["selected_against_previous_best"][mode]["against_prior_learned_oi"]
    name = "against_oi" if baseline == "oi" else "against_previous_token64"
    result = contrasts.get(name) if not (baseline == "previous_token64" and row["mode"] != mode) else None
    # Classical winners live in the Argo row but are compared with the matched
    # surface old model.  The report can display an exact difference without
    # inventing a confidence interval unavailable in the matched reporter.
    if result is None:
        delta = compare_rows(row, row_for(metrics, mode, baseline))
        return {ch: {"delta_rmse": delta[ch]["delta_rmse"], "ci95_delta_rmse": None}
                for ch in CHANNELS}
    return result


def complete_selected_contrasts(metrics, manifest):
    """Recompute final-winner contrasts, including classic/surface combinations.

    Caller must first pass verify_completed. The immutable 53 report legitimately
    stores classical predictions in its Argo group; this final reporting step
    also pairs a classical winner with the surface old model, without substituting
    the Argo old model or inventing a confidence interval.
    """
    helper = load_script("53_matched_reconstruction_report.py", "final_selected_contrasts")
    ref = helper.load_reference(metrics["selection"]["baseline_dir"], "development")
    cache = {}

    def prediction(mode, family):
        key = (mode, family)
        if key in cache:
            return cache[key]
        row = row_for(metrics, mode, family)
        if row["kind"] != "classical":
            jobs = sorted((job for job in manifest["jobs"] if job["family"] == family
                           and job["surface"] == (mode == "surface")), key=lambda job: job["seed"])
            values = []
            for job in jobs:
                arrays = helper.load_arrays(helper.prediction_path(job, "development"))
                helper.assert_identity(arrays, ref, job["tag"])
                values.append({"mean": np.asarray(arrays["mean"], np.float64)})
            value = helper.ensemble(values)
        else:
            if family == "prior_learned_oi":
                path = metrics["selection"]["auxiliary_prior_learned_oi"]["registry"]["arrays"]["development"]["path"]
            elif family == "fixed_pointwise_mlp":
                path = helper.prediction_path({"output": metrics["selection"]["auxiliary_fixed_mlp"]["directory"],
                                               "seed": 1234, "tag": family}, "development")
            else:
                path = helper.baseline_candidates(metrics["selection"]["baseline_dir"], "development")[family]
            arrays = helper.load_arrays(path)
            helper.assert_identity(arrays, ref, family)
            value = {"mean": np.asarray(arrays["mean"], np.float64)}
        cache[key] = value
        return value

    result = {}
    for mode, choice in metrics["selection"]["selected"].items():
        selected = prediction(mode, choice["family"])
        result[mode] = {baseline: helper.paired_contrast(selected, prediction(mode, baseline), ref)
                        for baseline in ("previous_token64", "oi", "prior_learned_oi")}
    return result


def all_negative_ci(contrast):
    return all(contrast[ch].get("delta_rmse") is not None
               and contrast[ch]["delta_rmse"] < 0
               and contrast[ch].get("ci95_delta_rmse") is not None
               and contrast[ch]["ci95_delta_rmse"][1] < 0 for ch in CHANNELS)


def stable_seed_improvement(row, baseline):
    if row["kind"] == "classical":
        return None
    ref = pooled(baseline)
    return all(seed["physical_metrics"][ch]["rmse"] < ref[ch]["rmse"]
               for seed in row["seed_metrics"] for ch in CHANNELS)


def component_gate(contrast):
    if not contrast:
        return {"tested": False, "stable_gain_in_both_variables": False}
    paired = contrast.get("paired_seed_rmse", {})
    stable = bool(paired) and all(all(value < 0 for value in paired[ch]["per_seed_delta_rmse"])
                                  for ch in CHANNELS)
    return {"tested": True, "lower_rmse_in_both_variables": all(contrast[ch]["delta_rmse"] < 0 for ch in CHANNELS),
            "monthly_ci_supports_both": all_negative_ci(contrast), "all_three_seed_pairs_support_both": stable,
            "stable_gain_in_both_variables": all_negative_ci(contrast) and stable}


def evidence_gates(metrics):
    decisions = {}
    prior = row_for(metrics, "argo", "prior_learned_oi")
    for mode, selected in metrics["selection"]["selected"].items():
        row = row_for(metrics, mode, selected["family"])
        delta = selected_contrast(metrics, mode, selected["family"], "prior_learned_oi")
        seeds = stable_seed_improvement(row, prior)
        latent = component_gate(metrics["component_contrasts"].get(f"{mode}:full_minus_latent_off"))
        local = component_gate(metrics["component_contrasts"].get(f"{mode}:full_minus_local_off"))
        applicable = selected["family"] == "soft_moe192"
        decisions[mode] = {"family": selected["family"], "kind": selected["kind"],
            "beats_strong_prior_in_both_variables": all(delta[ch]["delta_rmse"] < 0 for ch in CHANNELS),
            "monthly_ci_supports_strong_prior_gain": all_negative_ci(delta),
            "all_three_seeds_beat_strong_prior_in_both_variables": seeds,
            "stable_full_system_gain": all_negative_ci(delta) and seeds is True,
            "soft_moe_latent_control": latent, "soft_moe_local_control": local,
            "component_control_applies_to_selected_backbone": applicable,
            "selected_latent_necessity_supported": applicable and latent["stable_gain_in_both_variables"],
            "selected_local_necessity_supported": applicable and local["stable_gain_in_both_variables"],
            "novel_architecture_established": False, "sota_established": False,
            "reason": "Performance evidence cannot establish architectural originality or external SOTA; only SoftMoE has registered same-backbone component controls."}
    return decisions


def selected_architectures(metrics, manifest):
    result = {}
    selection = metrics["selection"]
    for mode, choice in selection["selected"].items():
        row = row_for(metrics, mode, choice["family"])
        item = {"family": choice["family"], "kind": choice["kind"],
                "uses_satellite": mode == "surface" and row["kind"] != "classical",
                "validation_mean_standardized_rmse": choice["validation_mean_standardized_rmse"],
                "weights": [], "model_configs": [], "training_configs": [], "params_per_seed": []}
        if row["kind"] != "classical":
            jobs = sorted((job for job in manifest["jobs"] if job["family"] == choice["family"]
                           and job["surface"] == (mode == "surface")), key=lambda job: job["seed"])
            for job in jobs:
                contract = selection["runs"][job["tag"]]
                c = contract["training_contract"]
                item["weights"].append({"seed": job["seed"], "path": contract["checkpoint_path"],
                                        "sha256": contract["checkpoint_sha256"], "best_step": c.get("best_step")})
                if choice["family"] == "previous_token64":
                    cfg = {"backbone": c.get("backbone", "d4rt"), "width": 64,
                           "n_latents": c.get("n_latent", 64), "n_self_blocks": 2,
                           "n_query_blocks": 2, "n_heads": 4, "profile_depth_bands": 4,
                           "mass_mode": c.get("mass_mode", "dfs"), "refiner_km": c.get("refiner_km", 500.),
                           "refiner_gate": c.get("refiner_gate", 1.), "refiner_dz": 100.,
                           "surface_patch_shape": [3, 3] if mode == "surface" else None,
                           "historical_setting": c.get("historical_setting"),
                           "note": "n_slots legacy summary field is not substituted for n_latent"}
                else:
                    cfg = c.get("config", {})
                    if not isinstance(cfg, dict):
                        raise ValueError("Model configuration must be a dictionary")
                item["model_configs"].append(cfg)
                item["training_configs"].append(c.get("training", {k: c[k] for k in ("steps", "lr", "weight_decay") if k in c}))
                item["params_per_seed"].append(c.get("params"))
        elif choice["family"] == "prior_learned_oi":
            c = selection["auxiliary_prior_learned_oi"]
            registry = c["registry"]
            historical = read_json(registry["historical_summary"]["path"])
            checkpoint = registry["source_checkpoint"]
            item["weights"] = [{"seed": 1234, **checkpoint}]
            item["model_configs"] = [{"backbone": "learned local optimal interpolation", "parameters": c["parameters"],
                                      "first_guess": historical.get("first_guess"),
                                      "weighting": historical.get("weights"), "nearest_profiles": historical.get("k"),
                                      "use_time": historical.get("use_time"), "use_state": historical.get("use_state"),
                                      "kernel_parameters_summary": historical.get("learned"),
                                      "source_summary": registry["historical_summary"]["path"]}]
            item["training_configs"] = [{"steps": 1500, "seed": 1234, "lr": historical.get("lr"),
                                          "role": "fixed existing checkpoint; not retrained"}]
            item["params_per_seed"] = [c["parameters"]]
        else:
            item["model_configs"] = [{"backbone": choice["family"], "learned_shared_latent": False,
                                      "reference": selection["baseline_dir"]}]
        result[mode] = item
    return result


def aggregate_bands(depth_metrics):
    """Pool errors, never average per-depth RMSE; no invented pooled R²/r."""
    result = {}
    for name, lo, hi in BANDS:
        layers = [m for depth, m in depth_metrics.items() if lo < float(depth) <= hi]
        output = {}
        for ch in CHANNELS:
            n = sum(m[ch]["n"] for m in layers)
            values = {"n": n, "rmse": None, "mae": None, "mean_bias": None}
            if n:
                values["rmse"] = float(np.sqrt(sum(m[ch]["n"] * m[ch]["rmse"] ** 2 for m in layers) / n))
                for key in ("mae", "mean_bias", "nll", "crps", "coverage_68", "coverage_95", "mean_std"):
                    if layers and all(key in m[ch] and m[ch][key] is not None for m in layers):
                        values[key] = float(sum(m[ch]["n"] * m[ch][key] for m in layers) / n)
            output[ch] = values
        result[name] = output
    return result


def real_data_supplement(path):
    report = read_json(path)
    selected = report["frozen_selection"]["selected_configuration"]
    matching = [group for group in report["groups"] if group["configuration"] == selected]
    if len(matching) != 1:
        raise ValueError("The real-Argo frozen selection is ambiguous")
    group = matching[0]
    return {"source_path": str(path), "source_sha256": sha256(path), "split": report["split"],
            "configuration": selected, "baseline": report["baseline"]["pooled"],
            "ensemble": group["ensemble"]["pooled"], "seed_statistics": group["seed_statistics"],
            "paired_month_rmse_bootstrap": group["ensemble"]["paired_month_rmse_bootstrap"],
            "relative_rmse_reduction_percent": {ch: 100 * (1 - group["ensemble"]["pooled"][ch]["rmse"] /
                                                          report["baseline"]["pooled"][ch]["rmse"]) for ch in CHANNELS},
            "never_rank_with_synthetic": True}


def build_payload(metrics, manifest, completion, literature, *, protocol_path, real_path, evidence=None):
    validate_literature(literature)
    architectures = selected_architectures(metrics, manifest)
    changes = {}
    for mode, choice in metrics["selection"]["selected"].items():
        row = row_for(metrics, mode, choice["family"])
        changes[mode] = {}
        for baseline in ("previous_token64", "oi", "prior_learned_oi"):
            comparison = compare_rows(row, row_for(metrics, mode, baseline))
            interval = selected_contrast(metrics, mode, choice["family"], baseline)
            for ch in CHANNELS:
                comparison[ch]["ci95_delta_rmse"] = interval[ch].get("ci95_delta_rmse")
            changes[mode][baseline] = comparison
    return {"format_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "task": "完成后的多模态三维海洋重建研究汇报", "completion": completion,
            "experiment": {key: manifest.get(key) for key in ("experiment", "train_years", "validation_year",
                           "previously_used_test_year", "inputs_per_month", "training_steps", "seeds")},
            "selection": metrics["selection"], "selected_architectures": architectures,
            "evidence_gates": evidence_gates(metrics), "improvements": changes,
            "test_n_per_variable": metrics["test_n_per_variable"], "levels_m": metrics["levels_m"],
            "rows": metrics["rows"], "contrasts": metrics["contrasts"],
            "component_contrasts": metrics["component_contrasts"], "multimodal_contrasts": metrics["multimodal_contrasts"],
            "bands_by_method": {f"{row['mode']}:{row['family']}": aggregate_bands(per_depth(row)) for row in metrics["rows"]},
            "literature": literature, "real_data_supplement": real_data_supplement(real_path),
            "evidence_contract": evidence,
            "limits": metrics["limits"], "sources": {"protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
                            "reporter": {"path": str(Path(__file__)), "sha256": sha256(Path(__file__))}}}


def fmt(value, digits=6):
    return "—" if value is None else f"{value:.{digits}f}"


def cell(value):
    if isinstance(value, (list, tuple)):
        value = "；".join(str(item) for item in value)
    elif isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).replace("|", "\\|").replace("\n", " ")


def citation(paper):
    return f"[{cell(paper['title'])}]({paper['url']})"


def mean_sd(row, ch, key):
    stat = row["seed_mean_sd"][ch][key]
    return f"{fmt(stat['mean'])} ± {fmt(stat['sd'])}"


def architecture_diagram(architecture):
    family = architecture["family"]
    lines = ["```mermaid", "flowchart LR", '    A["Argo 实际逐层温盐及掩码"]', '    Q["经纬度、深度、月份查询"]']
    if family in ("prior_learned_oi", "oi"):
        label = "既有 learned OI：学习局部协方差参数" if family == "prior_learned_oi" else "冻结球面 OI：验证集调参"
        lines += [f'    A --> O["{label}"]', "    Q --> O", '    O --> T["温度、盐度均值"]',
                  '    V["仅2004验证残差"] --> U["逐层、逐变量残差RMS尺度"]',
                  '    T --> R["均值与边际不确定性"]', "    U --> R"]
    elif family.startswith(("dense", "soft_moe", "local_transformer")):
        cfg = architecture["model_configs"][0]
        lines += ['    A --> O["冻结球面 OI 数值锚点"]', "    Q --> O",
                  '    Q --> D["独立查询 MLP / 解码器"]', '    O --> D']
        if cfg.get("use_latent", True):
            label = "Soft MoE latent Transformer" if cfg.get("variant") == "soft_moe" else "Dense latent Transformer"
            lines += ['    A --> E["逐层观测与新息 MLP 编码器"]', f'    E --> L["{label}"]', '    L --> D']
            if architecture["uses_satellite"]:
                lines += ['    S["SST / SSS / steric SSH"] --> SE["剖面位置表层编码器"]', "    SE --> L", "    S --> D"]
        elif architecture["uses_satellite"]:
            lines += ['    S["SST / SSS / steric SSH"]', "    S --> D"]
        if cfg.get("use_local", True):
            lines += ['    A --> N["32邻居实际逐层新息"]']
            if cfg.get("variant") == "local_transformer":
                lines += ['    N --> LT["32邻居 + null 的局部集合Transformer"]', '    D --> LT', '    LT --> G["局部候选与有符号gate"]']
            else:
                lines += ['    N --> G["局部候选与有符号gate"]']
            lines += ["    D --> G", '    G --> T["OI + residual + local correction"]',
                      '    N --> ES["局部coverage与新息方差统计"]', '    ES --> U']
        else:
            lines += ['    D --> T["OI + learned residual"]']
        lines += ["    O --> T", "    D --> T", '    D --> U["Gaussian 标准差head"]', '    T --> R["三种子集成温盐与不确定性"]', "    U --> R"]
    elif family == "official4dvarnet":
        lines += ['    A --> O["原位置双线性观测算子"]', '    O --> V["官方 GradSolver 10步 + ConvLSTM梯度模型"]',
                  '    P["官方 BilinAE prior"] --> V', '    V --> G["180×360、20层温盐网格"]', "    G --> D[\"独立位置双线性查询\"]", "    Q --> D", '    D --> T["三种子集成温盐均值"]']
        if architecture["uses_satellite"]:
            lines += ['    S["完整 SST / SSS / steric SSH 栅格"] --> V']
    elif family == "previous_token64":
        lines += ['    A --> E["逐层embedding → 4个depth-band pooling"]', '    E --> L["旧64-slot latent backbone"]',
                  '    L --> D["坐标query decoder + embedding refiner"]', "    Q --> D", '    D --> T["三种子集成温盐均值"]']
        if architecture["uses_satellite"]:
            lines += ['    S["SST/SSS 3×3 patch 与 SSH encoder"] --> L']
    else:
        lines += [f'    A --> M["{cell(family)}"]', "    Q --> M", '    M --> T["温盐均值"]']
    return "\n".join(lines + ["```"])


def render_report(report):
    selections = report["selection"]["selected"]
    papers = report["literature"]["papers"]
    by_id = {paper["id"]: paper for paper in papers}
    def references(ids):
        return "、".join(citation(by_id[key]) for key in ids if key in by_id)
    lines = ["# 多模态三维海洋重建：完整方法、架构选择与研究贡献汇报", "",
             "本报告在 **39 个注册训练任务全部完成、13 个架构设置各有 3 个种子、完整评分数组与验证选型通过完成审计后**生成。主指标为温度 RMSE（°C）和盐度 RMSE（PSU），同时给出常规误差、相关性和不确定性指标。", "",
             "## 1. 最终决定与最主要结果", "",
             "最终架构分别在 Argo-only 与允许表层观测的设置中，按 **2004 验证集温盐平均标准化 RMSE** 冻结选择：注册学习family用三种子集成，固定参照用其固定预测；2005 结果没有改变这一选择。以下‘表层设置’赢家也可能忽略卫星，如果仅使用 Argo 的强基线在验证集更好，就保留它。", "",
             "| 输入设置 | 最终采用方法 | 验证平均标准化RMSE | 2005温度RMSE (°C) | 2005盐度RMSE (PSU) |", "|---|---|---:|---:|---:|"]
    for mode, choice in selections.items():
        m = pooled(row_for(report, mode, choice["family"]))
        lines.append(f"| {'Argo-only' if mode == 'argo' else 'Argo + 表层产品可用'} | {choice['family']} | {fmt(choice['validation_mean_standardized_rmse'])} | {fmt(m['TEMP']['rmse'])} | {fmt(m['SALT']['rmse'])} |")
    lines += ["", "下表直接回答相同设置中改进了多少。ΔRMSE 为最终选择减基线，负值表示误差降低；相对降低百分比以该基线的 RMSE 为分母。三种子模型这里使用集成均值，固定既有方法使用其保存权重。", "",
              "| 设置 | 对照 | 温度ΔRMSE (°C) | 温度相对降低 | 盐度ΔRMSE (PSU) | 盐度相对降低 |", "|---|---|---:|---:|---:|---:|"]
    for mode, comparisons in report["improvements"].items():
        for base, delta in comparisons.items():
            t, s = delta["TEMP"], delta["SALT"]
            lines.append(f"| {mode} | {base} | {fmt(t['delta_rmse'])} | {t['relative_rmse_reduction_percent']:+.2f}% | {fmt(s['delta_rmse'])} | {s['relative_rmse_reduction_percent']:+.2f}% |")
    for mode, gate in report["evidence_gates"].items():
        if gate["family"] == "prior_learned_oi":
            verdict = "最终保留既有 learned OI；本轮新 Transformer/MoE 没有获得足以取代它的验证优势。共享 latent 的必要性未被当前选型证明。"
        elif gate["family"].endswith("latent_off"):
            verdict = "最终选择关闭 latent 的模型；当前证据不支持把共享 latent 保留为这一设置的必要模块。"
        elif gate["stable_full_system_gain"]:
            verdict = "最终系统对最强既有 learned OI 的两变量增益得到三种子和整月配对区间共同支持；这成立于本轮开发数据，不自动成立为外部 SOTA 或架构原创性。"
        elif gate["beats_strong_prior_in_both_variables"]:
            verdict = "最终系统开发集两变量 RMSE 都低于最强既有 learned OI，但种子稳定性或配对区间未同时支持两变量，不能把点估计优势写成稳健全面超越。"
        else:
            verdict = "最终系统没有在开发集两个变量同时超过最强既有 learned OI；验证选择与开发集排名可能不同，仍沿用冻结的验证选择。"
        lines += ["", f"**{mode} 的决定：** {verdict}"]
    lines += ["", "## 2. 研究目标、数据与公平比较的边界", "",
              "长期目标是从卫星 SST、SSS、海面高度与稀疏 Argo 剖面恢复一个可按位置、深度和日期查询的三维温盐状态，并提供不确定性。本轮实际完成的是 CESM2 月尺度、20 个固定深度的同月回顾性重建。连续位置接口已实现；任意新深度、未来预测和跨域迁移没有由这些实验验证。", "",
              "| 项目 | 本轮实际固定设置 |", "|---|---|",
              "| 数据与年份 | CESM2 synthetic Argo；训练 2000–2003、验证 2004、开发评分 2005 |",
              "| 输入与查询 | 每月6,080输入剖面；各方法相同source/target身份、相同评分点 |",
              "| 训练预算 | 13个设置 × 3种子 × 15,000 steps；每步1,024 queries |",
              "| 训练source池 | 输入cohort内30% float抽作target，剩余4,256条为source；验证/开发完整6,080条 |",
              "| 距平目标 | 气候态取Argo自身位置；训练年份拟合逐层T/S标准化 |",
              "| 评分支持 | 2005每变量351,895个有效值；2004每变量92,517个有效值 |",
              "| 表层观测 | 同一CESM2缓存：无噪声5 m SST/SSS、由目标温盐计算的steric sea-height proxy |",
              "| 固定强基线 | 经典OI；既有learned OI为1500-step、seed1234固定checkpoint，非本轮15k三seed训练 |", "",
              "2005 已在历史工作中查看，因此称 development。等输入、等步数与等评分点保证完整系统对比可复核，但不保证等参数、等FLOPs、等训练时间或优化器轨迹。原始 target 用同一个 float64 canonical 真值评分；float32只用于对齐保存精度不同的身份检查。卫星OSSE的SST/SSS直接等于5 m真值，steric proxy也由被重建温盐派生，不能把它当作独立测高观测或直接外推到真实产品。", "",
              "## 3. 最终采用架构与 backbone", ""]
    for mode, architecture in report["selected_architectures"].items():
        lines += [f"### 3.{1 if mode == 'argo' else 2}. {mode}：{architecture['family']}", "",
                  architecture_diagram(architecture), "",
                  "| 配置或权重 | 实际值 |", "|---|---|"]
        for i, config in enumerate(architecture["model_configs"]):
            lines.append(f"| seed/配置 {i + 1} | {cell(config)} |")
        for i, config in enumerate(architecture["training_configs"]):
            lines.append(f"| seed/训练配置 {i + 1} | {cell(config)} |")
        lines.append(f"| 每模型总参数 | {cell(architecture['params_per_seed']) if architecture['params_per_seed'] else '无神经网络参数'} |")
        lines.append(f"| 实际使用表层观测 | {'是' if architecture['uses_satellite'] else '否'} |")
        for weight in architecture["weights"]:
            lines.append(f"| seed {weight.get('seed')} 最优权重 | [{Path(weight['path']).name}]({weight['path']})；SHA256 `{weight['sha256']}`；best step {weight.get('best_step', '既有权重')} |")
        if architecture["kind"] == "existing_learned_oi" or architecture["family"] == "prior_learned_oi":
            lines += ["", "这个最终模型的 backbone 是局部 learned optimal interpolation，而非 latent Transformer。它在32条地理最近剖面读取实际逐层观测，对20层×T/S的40个field分别学习东西长度尺度、南北长度尺度与噪声比gamma，共120个参数，使用Gaussian kernel线性求解；没有time/state距离或卫星初猜。表中的kernel参数是历史按depth band聚合的摘要，实际逐层参数在冻结权重中。报告尺度另由2004逐层温盐残差RMS拟合，不是该模型给出的完整Bayesian后验。"]
        elif architecture["family"].startswith(("dense", "soft_moe", "local_transformer")):
            cfg = architecture["model_configs"][0]
            lines += ["", f"实际执行配置：width={cfg.get('width')}，latents={cfg.get('n_latents')}，heads={cfg.get('n_heads')}，latent blocks={cfg.get('n_blocks')}，query blocks={cfg.get('n_query_blocks')}，FFN variant={cfg.get('variant')}，experts={cfg.get('n_experts')}，每expert slots={cfg.get('slots_per_expert')}，latent启用={cfg.get('use_latent')}，local启用={cfg.get('use_local')}。"]
            if not cfg.get("use_latent", True):
                lines += ["", "latent相关模块虽然可能仍被分配并计入参数总数，但推理执行路径绕过共享latent。最终backbone应描述为OI锚定的query/local网络，不能继续称为共享latent Transformer。"]
    lines += ["", "### 3.3. 本轮新模型的完整组件与公式", "",
              "新系统逐层保留Argo的T/S数值、变量掩码、位置/深度/月编码，用两个新息MLP与观测投影编码。表层组在source剖面位置构造surface tokens，query还可直接使用同一产品的表层特征。模态prior按各模态总量平衡并保留null token；这不是独立信息量估计。", "",
              "共享latent每个block通过gated cross-attention重新读取完整观测memory，再执行latent self-attention与Dense FFN或Soft MoE。独立query解码器没有query间self-attention。Local Transformer只在每个query的32邻居集合内部增加attention。Soft MoE只替换latent block FFN，执行全部4个experts（各2slots），不存在top-k稀疏计算的已验证收益。", "",
              "共享latent/query结构和Soft MoE分别引用已有架构来源：" + references(("perceiver_io_2022", "soft_moe_2024")) + "。", "",
              "冻结球面OI使用原始query深度、逐变量有效观测与验证调参Gaussian covariance，距离与线性求解为float64。额外32邻居通路直接读取数值新息，形成局部候选、coverage与方差统计，再由有符号tanh gate调整均值：", "",
              r"\[\mu=a+r(h)+g(h,s)\odot\left[\sum_i\alpha_i d_i-(a-b)\right].\]", "",
              "其中a为冻结OI，b为背景（本synthetic设置为零标准化距平），d为实际逐层新息。残差与gate末层零初始化，使初始均值等于OI。局部gate允许负值，当前没有保证观测点严格插值。Gaussian尺度head给逐query边际标准差，不输出完整空间、垂直或温盐协方差。集成不确定性使用全方差公式，NLL/CRPS针对moment-matched Gaussian，而非完整Gaussian mixture。", "",
              "官方4DVarNet代码的本任务适配采用固定作者commit，GradSolver/ConvLstmGradModel/BilinAEPriorCost原AST执行：180×360全球网格、20层温盐40channels（表层组43）、10步solver、prior hidden32/gradient hidden48/downsampling2。观测算子在原Argo位置双线性读取网格，监督仅相同held-out queries，没有给予完整温盐真值场；它是明确的本任务适配，不是原论文benchmark复现。", "",
              "使用作者[4dvarnet-starter固定commit](https://github.com/CIA-Oceanix/4dvarnet-starter/tree/20f1b5f34b201342cde6dd21a30419d07541db54)；4DVarNet领域方法参照：" + references(("fourdvarnet_2024",)) + "。", "",
              "## 4. 与之前方法差异：改了什么，哪些能归因", "",
              "**旧64-slot本身已有Perceiver式共享latent。** 本轮不是首次把latent加入旧模型；旧→新同时改变了观测表示、数值OI锚点、直接新息通路、容量与训练目标。", "",
              "| 部分 | 之前64-slot | 本轮新系统 | 可作何种解释 |", "|---|---|---|---|",
              "| 观测粒度 | 逐层embedding后4个depth-band masked mean | 每实际深度独立行，T/S数值和掩码保留 | 垂直输入表示更细；不能单凭总提升证明这一组件贡献 |",
              "| 局部信息 | refiner读取压缩embedding | 原始query深度新息进入直接数值通路 | 需要full vs local_off验证；只对SoftMoE Argo设置注册 |",
              "| 数值均值 | 旧token网络输出 | 同一冻结OI + 零初始化residual/gate | 继承强基线本身是主要变化；不能归功于latent |",
              "| Backbone | width64、64 slots、2 blocks、4 heads | Dense64或width192、96 latents、6 blocks、8 heads；Dense/SoftMoE/Local | 容量/计算不同，完整系统比较不是参数匹配因果比较 |",
              "| 表层表示 | 旧patch encoders；本轮52在共同缓存适配 | 剖面位置surface tokens + query直接表层特征 | 相同产品，表示/压缩算子不同 |",
              "| Loss/优化 | pooled MSE、20% target-channel dropout；AdamW lr1e−3 FP32 | 变量均衡MSE +0.02NLL，无该dropout；AdamW lr3e−4 bf16 AMP | 等15k步不等优化轨迹，不能叫只改backbone |",
              "| 不确定性 | 旧注册网络无learned scale | query与局部统计Gaussian scale、三seed全方差 | 需同时比较CRPS/NLL/coverage/sharpness |", "",
              "旧Argo-only由49严格调用原62模型/训练AST重跑，补足缺失权重与公共预测。旧表层52是共同缓存下的新注册recipe，不能声称精确复现你贴出的0.2124/0.0371历史多模态数字；缺少完整原产物的旧4DVarNet数字也不用于制造提升。旧depth-band是embedding平均，不是简单把原始温盐值按band平均。", "",
              "## 5. 完整常规指标：3种子均值与集成分开", "",
              "RMSE/MAE/bias采用物理单位；bias=预测−真值。R²和Pearson r首先针对距平，同时另外列绝对温盐场的R²/r；加回同一气候态不改变RMSE/MAE/bias，但改变相关性和R²分母。‘三种子’为分别评分后mean±sample SD（ddof=1）；‘集成’为三份预测均值再评分，二者不可混用。"]
    for mode in ("argo", "surface"):
        lines += ["", f"### 5.{1 if mode == 'argo' else 2}. {mode}", "",
                  "| 方法 | 统计方式 | 温度RMSE (°C) | 盐度RMSE (PSU) | 温度MAE (°C) | 盐度MAE (PSU) |", "|---|---|---:|---:|---:|---:|"]
        rows = [r for r in report["rows"] if r["mode"] == mode or r["kind"] == "classical"]
        for row in rows:
            if row["kind"] != "classical":
                lines.append(f"| {row['family']} | 三seed mean ± SD | {mean_sd(row,'TEMP','rmse')} | {mean_sd(row,'SALT','rmse')} | {mean_sd(row,'TEMP','mae')} | {mean_sd(row,'SALT','mae')} |")
            m = pooled(row)
            role = "固定seed1234" if row["kind"] == "classical" and row["seeds"] else "固定方法" if row["kind"] == "classical" else "三seed集成"
            lines.append(f"| {row['family']} | {role} | {fmt(m['TEMP']['rmse'])} | {fmt(m['SALT']['rmse'])} | {fmt(m['TEMP']['mae'])} | {fmt(m['SALT']['mae'])} |")
    lines += ["", "模型规模按各已完成seed的实际保存合同列示；分配后绕过的模块也可能计入总参数，不能据此推断实际FLOPs。", "",
              "| 设置/方法 | 各seed总参数 |", "|---|---|"]
    for row in report["rows"]:
        if row["kind"] != "classical":
            counts = [c["training_contract"].get("params") for c in report["selection"]["runs"].values()
                      if Path(c["checkpoint_path"]).parent.name.startswith(f"{row['family']}_{row['mode']}_s")]
            lines.append(f"| {row['mode']} / {row['family']} | {cell(counts)} |")
    lines += ["", "### 5.3. 各方法 bias 与相关性（固定方法或集成）", "",
              "| 设置/方法 | 变量 | n | bias | 距平R² | 距平Pearson r | 绝对场R² | 绝对场Pearson r |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["rows"]:
        for ch, m in pooled(row).items():
            lines.append(f"| {row['mode']} / {row['family']} | {ch} ({UNITS[ch]}) | {m['n']:,} | {fmt(m['mean_bias'])} | {fmt(m['r2'])} | {fmt(m['pearson_r'])} | {fmt(m.get('absolute_r2'))} | {fmt(m.get('absolute_pearson_r'))} |")
    lines += ["", "## 6. 不确定性：精度、校准与区间宽度一起看", "",
              "NLL为物理单位自然对数密度，可为负且不可把温度/盐度数值直接混比；CRPS与目标同单位。68%用μ±σ（Gaussian名义68.2689%），95%用μ±1.95996398454σ。coverage接近名义值但σ过宽并不意味着可靠；必须一起检查NLL、CRPS和mean σ。OI和既有learned OI的σ为2004逐深度/变量残差RMS校准；它是验证残差尺度，不是解析posterior variance。没有scale的方法保留未提供，不编造校准。", "",
              "| 设置/方法 | 变量 | NLL ↓ | CRPS ↓ | 68% coverage | 95% coverage | mean σ | 尺度来源 |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in report["rows"]:
        for ch, m in pooled(row).items():
            lines.append(f"| {row['mode']} / {row['family']} | {ch} | {fmt(m.get('nll'))} | {fmt(m.get('crps'))} | {fmt(m.get('coverage_68'))} | {fmt(m.get('coverage_95'))} | {fmt(m.get('mean_std'))} | {cell(row['uncertainty_method'])} |")
    lines += ["", "## 7. 深度上的提升：最终模型、旧模型和最强既有方法", ""]
    for mode, choice in selections.items():
        selected = row_for(report, mode, choice["family"])
        old = row_for(report, mode, "previous_token64")
        prior = row_for(report, "argo", "prior_learned_oi")
        lines += [f"### 7.{1 if mode == 'argo' else 2}. {mode}逐深度", "",
                  "| 深度(m) | n T/S | 最终T RMSE | 旧T RMSE | prior T RMSE | 最终S RMSE | 旧S RMSE | prior S RMSE |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        details = [per_depth(row) for row in (selected, old, prior)]
        for depth in sorted(details[0], key=float):
            sm, om, pm = [d[depth] for d in details]
            lines.append(f"| {float(depth):g} | {sm['TEMP']['n']}/{sm['SALT']['n']} | {fmt(sm['TEMP']['rmse'])} | {fmt(om['TEMP']['rmse'])} | {fmt(pm['TEMP']['rmse'])} | {fmt(sm['SALT']['rmse'])} | {fmt(om['SALT']['rmse'])} | {fmt(pm['SALT']['rmse'])} |")
        lines += ["", "深度带按有效数值汇总平方误差后开方，非每层RMSE简单平均。", "",
                  "| 深度带 | n T/S | 最终温度RMSE | 旧温度RMSE | 最终盐度RMSE | 旧盐度RMSE |", "|---|---:|---:|---:|---:|---:|"]
        selected_band = aggregate_bands(details[0]); old_band = aggregate_bands(details[1])
        for band, m in selected_band.items():
            om = old_band[band]
            lines.append(f"| {band} | {m['TEMP']['n']}/{m['SALT']['n']} | {fmt(m['TEMP']['rmse'])} | {fmt(om['TEMP']['rmse'])} | {fmt(m['SALT']['rmse'])} | {fmt(om['SALT']['rmse'])} |")
    lines += ["", "## 8. 配对比较、组件消融与多模态增益", "",
              "下面95%区间来自同一评分点的整月配对bootstrap；它覆盖12个评分月的变化，不包括全部地理/长期气候不确定性。负差值表示前一个系统RMSE更低。seed配对另列，不能用bootstrap代替训练随机性。", "",
              "| 设置/最终模型 | 对照 | 温度ΔRMSE [95% CI] | 盐度ΔRMSE [95% CI] |", "|---|---|---|---|"]
    for mode, comparisons in report["improvements"].items():
        for baseline, delta in comparisons.items():
            def ci_cell(ch):
                ci = delta[ch]["ci95_delta_rmse"]
                return f"{fmt(delta[ch]['delta_rmse'])} [" + (f"{fmt(ci[0])}, {fmt(ci[1])}" if ci else "未单独计算") + "]"
            lines.append(f"| {mode} / {selections[mode]['family']} | {baseline} | {ci_cell('TEMP')} | {ci_cell('SALT')} |")
    lines += ["", "### 8.1. SoftMoE同系统组件控制", "",
              "这轮只有SoftMoE192注册full/latent_off/local_off对照，surface没有local_off。它们可以支持该backbone的组件判断；Dense、Local Transformer或旧模型获选时，不能把SoftMoE的消融结论移借给赢家。关闭latent后surface query仍可直接读表层产品；关闭新增local后仍有冻结OI数值通路。", "",
              "| 对照（full−off） | 温度ΔRMSE [95% CI] | 盐度ΔRMSE [95% CI] | 三seed配对温度差 | 三seed配对盐度差 |", "|---|---|---|---|---|"]
    for name, c in report["component_contrasts"].items():
        t, s = c["TEMP"], c["SALT"]
        tc, sc = t["ci95_delta_rmse"], s["ci95_delta_rmse"]
        p = c["paired_seed_rmse"]
        lines.append(f"| {name} | {fmt(t['delta_rmse'])} [{fmt(tc[0])}, {fmt(tc[1])}] | {fmt(s['delta_rmse'])} [{fmt(sc[0])}, {fmt(sc[1])}] | {cell([fmt(v) for v in p['TEMP']['per_seed_delta_rmse']])} | {cell([fmt(v) for v in p['SALT']['per_seed_delta_rmse']])} |")
    for mode, gate in report["evidence_gates"].items():
        support = "三seed与两变量配对区间支持额外latent收益" if gate["soft_moe_latent_control"]["stable_gain_in_both_variables"] else "未获得三seed与两变量配对区间同时支持的额外latent收益"
        attribution = "最终赢家正是SoftMoE full，该组件证据适用" if gate["component_control_applies_to_selected_backbone"] else "该结论只适用于被消融的SoftMoE，不能证明最终赢家的latent必要性"
        lines += ["", f"{mode}：{support}；{attribution}。"]
    lines += ["", "### 8.2. 同family增加表层产品（surface−Argo）", "",
              "| family | 温度ΔRMSE [95% CI] | 盐度ΔRMSE [95% CI] |", "|---|---|---|"]
    for family, c in report["multimodal_contrasts"].items():
        t, s = c["TEMP"], c["SALT"]
        tc, sc = t["ci95_delta_rmse"], s["ci95_delta_rmse"]
        lines.append(f"| {family} | {fmt(t['delta_rmse'])} [{fmt(tc[0])}, {fmt(tc[1])}] | {fmt(s['delta_rmse'])} [{fmt(sc[0])}, {fmt(sc[1])}] |")
    lines += ["", "这些相同family的差值才对应‘加入本轮表层产品后的系统变化’。两个输入setting各自的赢家可能是不同模型，直接比较赢家不能归因为卫星单组件。理想5 m输入的近表层增益须结合逐层表阅读，不证明真实卫星对深层温盐有同等信息增益。", "",
              "## 9. Contribution 与 novelty：哪些有证据，哪些仍不成立", "",
              "这里区分已实现的技术变化、获得实验支持的方法价值与足以支持论文原创性的机制。更高精度不自动等于新机制；严谨比较体系是必要证据，也不能代替方法novelty。", "",
              "| 候选贡献 | 本轮证据判断 | 允许的表述 |", "|---|---|---|"]
    for mode, gate in report["evidence_gates"].items():
        performance = "两变量均有配对区间与三seed支持" if gate["stable_full_system_gain"] else "未建立同时稳健超过最强既有方法"
        latent = "对最终SoftMoE backbone有注册消融支持" if gate["selected_latent_necessity_supported"] else "最终backbone的共享latent必要性未证实"
        local = "对最终SoftMoE backbone有注册消融支持" if gate["selected_local_necessity_supported"] else "最终backbone的额外local通路必要性未证实或未注册同backbone控制"
        lines += [f"| {mode}：完整新系统的额外精度 | {performance} | 仅陈述本任务开发结果；OI继承、表示与优化均发生变化 |",
                  f"| {mode}：共享潜在状态的独立价值 | {latent} | 只在适用的同系统控制范围内陈述，不将old→new归为首次添加latent |",
                  f"| {mode}：原始数值局部通路 | {local} | 已实现逐层直接读取，不等于严格观测插值/无损压缩证明 |"]
    lines += ["| 通用观测→latent→query架构与Soft MoE | 明确已有来源 | 引用Perceiver IO/Soft MoE；不能列为独立原创机制 |",
              "| 卫星背景/多模态输入与剖面修正 | 有直接海洋工作先例，且当前OSSE产品理想化 | 必须对最接近领域方法说明具体任务与更新机制差异 |",
              "| 独立信息、复制记录不增加证据、相关观测折扣 | 当前39任务没有验证 | 仅作为未来研究假设，不能写为已完成贡献 |",
              "| 连续任意深度、严格预报、迁移/transfer | 本轮未实验 | 仅可说明研究目标或待验证能力 |",
              "| SOTA | 目前内部适配与开发比较不构成外部SOTA证据 | 不作SOTA宣称 |", "",
              "**最重要的最终研究判断：** " + (
                  "本轮选择仍保留既有数值基线，尚未确立能够取代它的新latent架构贡献。明确了更复杂网络没有提供足够额外价值，后续模型应先保留获选数值backbone；这是一项实证决定，不把比较体系本身包装成方法novelty。"
                  if all(item["kind"] in ("classical", "existing_learned_oi") for item in selections.values()) else
                  "本轮确立的改进是上表中实际获选完整系统的性能与配置；具体是否超过最强已有方法、latent是否必要，分别由第1节和同backbone消融门禁判断。通用latent/MoE不是原创机制，当前独立新架构贡献仍不足以仅由这些结果宣称。"), "",
              "新系统‘OI数值锚定 + 逐层原始证据的直接通路 + 可选共享多模态表示’已实现，但只有获得适用对照支持的模块应保留。其组合是否足以作为新的方法贡献，还需要与最接近的海洋重建方法明确区分更新机制、观测处理和适用任务。若赢家是既有OI或latent_off，论文不能继续把共享latent/MoE写成已证明必要的核心创新。", "",
              "## 10. 类似论文与相似风险", "",
              "以下审计依据原论文/作者代码或官方来源。相似指技术与研究定位重叠，不能根据标题或模块名推断完全相同。直接海洋三维温盐重建工作优先于通用Transformer引用。", "",
              "| 文献 | 已有方法及与我们的重叠 | 重要差异 | 相似风险 | 还需何种证据 |", "|---|---|---|---|---|"]
    for paper in papers:
        lines.append(f"| {citation(paper)} ({paper['year']}) | {cell(paper['method_summary'])}；{cell(paper['overlap_with_ours'])} | {cell(paper['important_difference'])} | {cell(paper['risk_level'])} | {cell(paper['evidence_needed'])} |")
    lines += ["", "### 10.1. 新颖性主张逐项审核", "",
              "此表是文献审查时的机制/证据要求快照；其中待验证状态不代表本轮训练未完成，最终实证判断以第9节和完成门禁为准。", "",
              "| 主张 | 审核结论 | 原因与相关原始来源 | 必要证据 |", "|---|---|---|---|"]
    for claim in report["literature"]["claim_audit"]:
        links = "、".join(citation(by_id[pid]) for pid in claim.get("related_paper_ids", []) if pid in by_id)
        lines.append(f"| {cell(claim['claim'])} | {cell(claim['status'])} | {cell(claim['reason'])} {links} | {cell(claim['required_evidence'])} |")
    lines += ["", "因此，‘把卫星和稀疏剖面融合进latent，再按坐标查询’本身不足以构成可靠新颖性。与近邻区分必须依赖一项清楚的新观测更新/数值保真/不确定性机制，加上相同观测制度下的实际优势；当前代码并没有已验证的精确posterior update、复制不增信息或相关证据去重机制。若文献矩阵显示直接机制重叠，本轮应明确把原创性判断列为不足，而不是仅以更多模块或更大模型包装创新。", "",
              "## 11. 真实Argo补充：独立列示，不混入CESM2排名", ""]
    real = report["real_data_supplement"]
    lines += [f"真实观测实验的冻结选择为 `{real['configuration']}`；{real['split']}。与CESM2不同的数据、输入产品、训练预算和目标标准化，不能把两个表数值直接排大小。", "",
              "| 真实观测方法 | 温度RMSE (°C) | 盐度RMSE (PSU) | 温度MAE | 盐度MAE |", "|---|---:|---:|---:|---:|"]
    for label, m in (("真实数据匹配基线", real["baseline"]), ("冻结选择三seed集成", real["ensemble"])):
        lines.append(f"| {label} | {fmt(m['TEMP']['rmse'])} | {fmt(m['SALT']['rmse'])} | {fmt(m['TEMP']['mae'])} | {fmt(m['SALT']['mae'])} |")
    lines += ["", f"真实数据物理单位相对RMSE降低：温度 {real['relative_rmse_reduction_percent']['TEMP']:.2f}%，盐度 {real['relative_rmse_reduction_percent']['SALT']:.2f}%。完整物理单位定义、逐层/深度带和不确定性见[真实Argo标准指标报告](../real_data/standard_metrics_20261007.md)。这组结果支持有限的开发集增益，不能替CESM2的同setting比较，也不能成为独立年度泛化证明。", "",
              "## 12. 最终研究决定与尚未解决的问题", ""]
    for mode, architecture in report["selected_architectures"].items():
        lines.append(f"- **{mode} 后续执行模型：{architecture['family']}。** 使用上面列出的冻结权重与配置，不根据2005重新挑模型。")
    lines += ["- 论文中只将确有门禁支持的新增系统收益列为已验证贡献；如果没有超过强既有方法，明确当前新架构贡献不足。通用latent/MoE机制引用已有工作；无同backbone消融支持时，不宣布latent必要。",
              "- 新颖性不足的部分明确保留为待验证方向：相关来源/重复观测的信息一致性、测量保真与校准机制；它们不属于本轮已完成实验。",
              "- 独立验证需要新的年份/区域与真实卫星误差制度；预报必须另设因果观测截止时间。20个固定深度结果不证明任意深度重建。", "",
              "## 13. 产物、来源与可复核完成证据", "",
              "39任务的完整15,000-step训练、三seed记录、冻结权重和共同预测数组由completion门禁审核。该门禁要求完整评分计数、13family×3seed、OI/既有learned OI/固定MLP对照，并核验metrics、selection和基础report的SHA256；本报告再次核验所用权重与预测指纹。", "",
              "- [完整架构与实验协议](matched_architecture_protocol_20261007.md)",
              "- [完整物理指标及配对比较](matched_reconstruction_20261007.md)",
              "- [文献新颖性审计](literature_novelty_audit_20261007.md)",
              "- [独立证据与允许主张合同](final_report_evidence_contract_20261007.md)",
              f"- [冻结验证选型]({OUTPUT / 'selection.json'})",
              f"- [完成门禁证据]({OUTPUT / 'completion.json'})",
              f"- [可机器读取的完整最终汇报]({OUTPUT / 'final_research_report_20261007.json'})", "",
              "生成时间：" + report["created_at_utc"] + "。"]
    return "\n".join(lines) + "\n"


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=OUTPUT)
    p.add_argument("--metrics", type=Path)
    p.add_argument("--selection", type=Path)
    p.add_argument("--matched-report", type=Path, default=ROOT / "reports/synthetic/matched_reconstruction_20261007.md")
    p.add_argument("--literature", type=Path, default=ROOT / "reports/synthetic/literature_novelty_audit_20261007.json")
    p.add_argument("--protocol", type=Path, default=ROOT / "reports/synthetic/matched_architecture_protocol_20261007.md")
    p.add_argument("--evidence", type=Path, default=ROOT / "reports/synthetic/final_report_evidence_contract_20261007.json")
    p.add_argument("--real-metrics", type=Path, default=ROOT / "outputs/latent_ocean/standard_metrics_20261007.json")
    p.add_argument("--report", type=Path, default=ROOT / "reports/synthetic/final_research_report_20261007.md")
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    metrics_path = args.metrics or args.output_root / "metrics.json"
    selection_path = args.selection or args.output_root / "selection.json"
    destination = args.json or args.output_root / "final_research_report_20261007.json"
    try:
        completion, manifest, metrics = verify_completed(args.output_root, metrics_path, selection_path, args.matched_report)
        literature = validate_literature(read_json(args.literature))
        evidence = read_json(args.evidence)
        if evidence.get("is_final_results") is not False:
            raise ValueError("The static evidence contract must not masquerade as final results")
        metrics["final_selected_contrasts"] = complete_selected_contrasts(metrics, manifest)
        report = build_payload(metrics, manifest, completion, literature, protocol_path=args.protocol, real_path=args.real_metrics, evidence=evidence)
        report["sources"].update({"metrics": {"path": str(metrics_path), "sha256": sha256(metrics_path)},
            "selection": {"path": str(selection_path), "sha256": sha256(selection_path)},
            "completion": {"path": str(args.output_root / "completion.json"), "sha256": sha256(args.output_root / "completion.json")},
            "evidence_contract": {"path": str(args.evidence), "sha256": sha256(args.evidence)},
            "literature": {"path": str(args.literature), "sha256": sha256(args.literature)}})
        atomic_write(destination, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        atomic_write(args.report, render_report(report))
        print(json.dumps({"status": "complete", "report": str(args.report), "json": str(destination),
                          "selected": {mode: item["family"] for mode, item in report["selected_architectures"].items()}}, ensure_ascii=False))
        return 0
    except PendingEvidence as error:
        print(json.dumps({"status": "pending", "reason": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
