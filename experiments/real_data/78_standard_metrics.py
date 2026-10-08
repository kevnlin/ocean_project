"""Recompute conventional, physical-unit metrics from frozen reconstruction arrays.

This is reporting only. It never trains, changes frozen model selection, or opens
an independent test split. Predictions are streamed one seed at a time.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from ocean_tokenizer.reconstruction_metrics import (
    COVERAGE_68_NOMINAL, GAUSSIAN_KEYS, REGRESSION_KEYS,
    evaluate_reconstruction, inverse_standardize, paired_rmse_bootstrap,
)

ROOT = Path(__file__).resolve().parents[2]
CHANNELS = ("TEMP", "SALT")
UNITS = {"TEMP": "°C", "SALT": "PSU"}
BANDS = (("0–100 m", 0., 100.), ("100–300 m", 100., 300.),
         ("300–700 m", 300., 700.), ("700–1,000 m", 700., 1000.))
LABELS = {"dense_large": "Dense latent Transformer", "soft_moe_large": "Soft MoE latent Transformer",
          "local_transformer_large": "Local Transformer + latent", "analysis_only": "仅重新训练 OI",
          "dense_large_fixed_oi_full": "Dense，固定 OI，完整结构",
          "dense_large_fixed_oi_latent_off": "Dense，固定 OI，关闭 latent",
          "dense_large_fixed_oi_local_off": "Dense，固定 OI，关闭新增局部通路"}


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def reporting_helpers():
    """Reuse the campaign's strict saved-array identity and calibration readers."""
    path = Path(__file__).with_name("71_latent_report.py")
    spec = importlib.util.spec_from_file_location("frozen_latent_array_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def physical_arrays(arrays, scale, offset):
    return {key: inverse_standardize(arrays[key], arrays["level"], scale, offset)
            for key in ("mean", "target", "baseline")} | {
                "std": inverse_standardize(arrays["std"], arrays["level"], scale, uncertainty=True)}


def detail_metrics(physical, level_index, depths):
    evaluate = lambda mask: evaluate_reconstruction(
        physical["mean"][mask], physical["target"][mask], physical["std"][mask], channel_names=CHANNELS)
    return {"pooled": evaluate(np.ones(len(level_index), bool)),
            "by_depth": [{"level_index": i, "depth_m": float(d), "metrics": evaluate(level_index == i)}
                         for i, d in enumerate(depths)],
            "by_depth_band": [{"depth_band": name, "lower_exclusive_m": lo, "upper_inclusive_m": hi,
                               "metrics": evaluate((depths[level_index] > lo) & (depths[level_index] <= hi))}
                              for name, lo, hi in BANDS]}


def seed_statistics(details):
    """Statistics of separately scored models, not an ensemble's uncertainty."""
    output = {}
    for ch in CHANNELS:
        counts = [d["pooled"][ch]["n"] for d in details]
        if len(set(counts)) != 1:
            raise ValueError("per-variable scored counts differ across seeds")
        channel = {"n_per_seed": counts[0]}
        for key in REGRESSION_KEYS + GAUSSIAN_KEYS:
            values = [d["pooled"][ch][key] for d in details]
            defined = [value for value in values if value is not None]
            channel[key] = {"mean": float(np.mean(defined)) if defined else None,
                            "sample_sd": float(np.std(defined, ddof=1)) if len(defined) > 1 else None,
                            "values": values, "n_defined": len(defined)}
        output[ch] = channel
    return output


def baseline_details(reference, calibration, helpers, scale, offset, depths):
    arrays = {**reference, "mean": reference["baseline"],
              "std": helpers.calibrated_std(calibration, reference["level"])}
    physical = physical_arrays(arrays, scale, offset)
    details = detail_metrics(physical, arrays["level"], depths)
    details["uncertainty_source"] = "per-depth/channel residual RMS fitted on validation 2021 only; mean unchanged"
    return details


def assert_summary_scores(summary, arrays, physical):
    metrics = evaluate_reconstruction(physical["mean"], physical["target"])
    standardized = evaluate_reconstruction(arrays["mean"], arrays["target"])
    for ch in CHANNELS:
        saved = summary["development"]["scores"][ch]
        if metrics[ch]["n"] != saved["n"]:
            raise ValueError("saved count does not agree with physical-unit reconstruction")
        for computed, key in ((metrics[ch]["rmse"], "rmse_physical"),
                              (standardized[ch]["rmse"], "rmse_z")):
            if not np.isclose(computed, saved[key], rtol=2e-7, atol=1e-10):
                raise ValueError(f"saved {key} differs from recomputed arrays for {ch}")
    return {ch: {"n": standardized[ch]["n"], "standardized_rmse": standardized[ch]["rmse"]}
            for ch in CHANNELS}


def build_report(output_root, selection_path, first_guess_path, bootstrap_draws=4000):
    helpers = reporting_helpers()
    selection = read_json(selection_path)
    if selection.get("status") != "selected":
        raise ValueError("final validation-only selection must already be frozen")
    fg = torch.load(first_guess_path, map_location="cpu", weights_only=True)
    if tuple(fg["channels"]) != CHANNELS:
        raise ValueError("first-guess channel order differs")
    depths = np.asarray(fg["levels"], dtype=np.float64)
    scale = np.column_stack([np.asarray(fg["target_std"][ch]) for ch in CHANNELS])
    offset = np.column_stack([np.asarray(fg["target_mean"][ch]) for ch in CHANNELS])
    if scale.shape != (len(depths), 2) or offset.shape != scale.shape:
        raise ValueError("first-guess normalization does not match depths/channels")
    if len(depths) != 20 or np.any(np.diff(depths) <= 0):
        raise ValueError("unexpected campaign depth schema")
    candidates = selection["candidates"]
    if len(candidates) != 7 or len({c["tag"] for c in candidates}) != 7:
        raise ValueError("expected all seven registered final candidates")
    first_tag, first_seed = candidates[0]["tag"], candidates[0]["seeds"][0]
    first_folder = output_root / f"{first_tag}_s{first_seed}"
    reference = helpers.load_predictions(first_folder / f"development_seed{first_seed}.npz")
    validation = helpers.load_predictions(first_folder / f"validation_seed{first_seed}.npz")
    calibration = helpers.calibrate_baseline(validation)
    del validation
    if np.any(reference["level"] >= len(depths)):
        raise ValueError("prediction depths outside first-guess normalization")
    baseline = baseline_details(reference, calibration, helpers, scale, offset, depths)
    first_summary = read_json(first_folder / f"summary_seed{first_seed}.json")
    fingerprints = {key: first_summary["data"][key]
                    for key in ("cohort_fingerprint", "feature_fingerprint", "scene_fingerprint")}
    if fingerprints["cohort_fingerprint"] != fg["cohort_fingerprint"] or fingerprints["feature_fingerprint"] != fg["feature_fingerprint"]:
        raise ValueError("saved normalization and selected data fingerprints disagree")
    source_hashes = first_summary["source_sha256"]
    model = torch.load(first_folder / f"best_seed{first_seed}.pt", map_location="cpu", weights_only=True)
    for j, ch in enumerate(CHANNELS):
        if not np.array_equal(np.asarray(model["normalization"]["std"][ch]), scale[:, j]) or not np.array_equal(
                np.asarray(model["normalization"]["mean"][ch]), offset[:, j]):
            raise ValueError("model checkpoint and saved first-guess normalization differ")
    del model, fg
    report = {"schema_version": 1, "task": "retrospective real-Argo temperature/salinity reconstruction",
              "split": "previously used development 2022–2023; not independent test",
              "units": UNITS, "depths_m": depths.tolist(), "normalization": {
                  "source": str(first_guess_path), "source_sha256": sha256(first_guess_path),
                  "mean_by_depth_channel": offset.tolist(), "std_by_depth_channel": scale.tolist(),
                  "fit_scope": first_summary["data"]["normalization_scope"]},
              "definitions": {"rmse": "sqrt(sum((prediction-target)^2)/n), pooled over valid heldout values",
                  "mae": "sum(abs(prediction-target))/n", "mean_bias": "mean(prediction-target)",
                  "r2": "1-SSE/SST of physical-unit anomalies; null for constant target or fewer than two samples",
                  "pearson_r": "Pearson correlation of physical-unit anomalies; null for constant prediction/target",
                  "physical_target": "inverse-standardized temperature/salinity anomalies relative to 1° cell climatology",
                  "error_interpretation": "RMSE, MAE and bias equal absolute T/S errors because the same climatology cancels; R² and r are anomaly metrics",
                  "coverage_68_nominal": COVERAGE_68_NOMINAL, "coverage_95_nominal": .95,
                  "gaussian_nll": "natural-log predictive density in physical units; unit dependent and may be negative",
                  "gaussian_crps": "normal-distribution CRPS in target units; lower is better",
                  "ensemble_uncertainty": "moment-matched Gaussian using mean(std²+mean²)-ensemble_mean²; NLL/CRPS are not exact Gaussian-mixture scores",
                  "seed_statistics": "mean ± sample SD (ddof=1) of separately evaluated trained models; distinct from ensemble metric",
                  "depth_bands": "lower edge exclusive, upper edge inclusive; equal weighting per valid heldout value"},
              "frozen_selection": {"path": str(selection_path), "sha256": sha256(selection_path),
                  "selected_configuration": selection["selectedtag"], "selected_at_utc": selection["selected_at_utc"],
                  "criterion": "lowest 2021 validation mean temperature/salinity standardized RMSE of three-seed ensemble; unchanged"},
              "data_fingerprints": fingerprints, "training_source_sha256": source_hashes,
              "baseline": baseline, "baseline_calibration": calibration, "groups": [],
              "input_prediction_sha256": {}, "verification": {"counts": {ch: baseline["pooled"][ch]["n"] for ch in CHANNELS},
                  "all_seed_identity_fields": ["month", "profile", "level", "target", "baseline"],
                  "saved_physical_and_standardized_rmse_agreement": True}}
    for candidate in candidates:
        tag, seeds = candidate["tag"], candidate["seeds"]
        if seeds != [1234, 1235, 1236]:
            raise ValueError("all candidates must contain the three registered seeds")
        group = {"configuration": tag, "label": LABELS[tag], "seeds": seeds,
                 "params": candidate["neural_parameters"], "flags": candidate["flags"],
                 "config": candidate["config"], "individual_seeds": {}}
        mean_sum = np.zeros_like(reference["mean"], dtype=np.float64)
        second_sum = np.zeros_like(mean_sum)
        for seed in seeds:
            folder = output_root / f"{tag}_s{seed}"
            summary = read_json(folder / f"summary_seed{seed}.json")
            if summary["seed"] != seed or summary["tag"] != folder.name or summary["training"]["steps"] != 6000 or summary["history"][-1]["step"] != 6000:
                raise ValueError("only complete registered 6000-step models can be scored")
            if any(summary["data"][key] != value for key, value in fingerprints.items()) or summary["source_sha256"] != source_hashes:
                raise ValueError("matched data/training source fingerprints differ")
            if summary["config"] != candidate["config"]:
                raise ValueError("candidate architecture differs from frozen selection")
            path = folder / f"development_seed{seed}.npz"
            arrays = helpers.load_predictions(path)
            helpers.assert_identical(reference, arrays)
            physical = physical_arrays(arrays, scale, offset)
            details = detail_metrics(physical, arrays["level"], depths)
            details["standardized_rmse"] = assert_summary_scores(summary, arrays, physical)
            details["best_step"] = summary["best_step"]
            group["individual_seeds"][str(seed)] = details
            mean_sum += arrays["mean"]
            second_sum += np.asarray(arrays["mean"], dtype=np.float64) ** 2 + np.asarray(arrays["std"], dtype=np.float64) ** 2
            report["input_prediction_sha256"][str(path)] = sha256(path)
            del arrays, physical
        group["seed_statistics"] = seed_statistics(list(group["individual_seeds"].values()))
        seed_details = list(group["individual_seeds"].values())
        group["seed_statistics_by_depth"] = [
            {"level_index": i, "depth_m": float(depth), "metrics": seed_statistics([
                {"pooled": item["by_depth"][i]["metrics"]} for item in seed_details])}
            for i, depth in enumerate(depths)]
        group["seed_statistics_by_depth_band"] = [
            {"depth_band": name, "metrics": seed_statistics([
                {"pooled": item["by_depth_band"][i]["metrics"]} for item in seed_details])}
            for i, (name, _, _) in enumerate(BANDS)]
        mean = mean_sum / len(seeds)
        arrays = {**reference, "mean": mean,
                  "std": np.sqrt(np.maximum(second_sum / len(seeds) - mean ** 2, 0.))}
        physical = physical_arrays(arrays, scale, offset)
        group["ensemble"] = detail_metrics(physical, arrays["level"], depths)
        group["ensemble"]["paired_month_rmse_bootstrap"] = {
            ch: paired_rmse_bootstrap(physical["mean"][:, j], physical["baseline"][:, j],
                                     physical["target"][:, j], arrays["month"], draws=bootstrap_draws)
            for j, ch in enumerate(CHANNELS)}
        group["ensemble"]["standardized_rmse"] = {
            ch: {"n": scores["n"], "standardized_rmse": scores["rmse"]}
            for ch, scores in evaluate_reconstruction(mean, arrays["target"]).items()}
        report["groups"].append(group)
        print(json.dumps({"finished": tag, "ensemble_physical_rmse": {
            ch: group["ensemble"]["pooled"][ch]["rmse"] for ch in CHANNELS}}), flush=True)
        del mean_sum, second_sum, arrays, physical, mean
    return report


def format_number(value, digits=5):
    return "—" if value is None else f"{value:.{digits}f}"


def mean_sd(stat, digits=5):
    return f"{format_number(stat['mean'], digits)} ± {format_number(stat['sample_sd'], digits)}"


def render_report(report):
    chosen = report["frozen_selection"]["selected_configuration"]
    groups = report["groups"]
    selected = next(g for g in groups if g["configuration"] == chosen)
    baseline = report["baseline"]["pooled"]
    rows = ["# 真实 Argo 重建：常规指标与物理单位", "",
            "主要精度指标统一使用 **RMSE（均方根误差）**，温度以 °C、盐度以 PSU 表示；同时报告 MAE、平均偏差、R²、Pearson 相关系数，以及不确定性 NLL、CRPS 和覆盖率。", "",
            "本报告从已冻结的预测数组重新计算，不改变权重、评分点或验证集选型。训练年份为 2016–2020，验证为 2021；2022–2023 已用于此前开发，本表属于开发集而非独立测试集。每个变量的有效评分值为 "
            f"{baseline['TEMP']['n']:,} / {baseline['SALT']['n']:,}，覆盖 20 个固定深度。", "",
            "按每个样本所在深度，用保存的训练归一化反变换后再汇总平方误差；不能把整体标准化 RMSE 乘一个全局尺度当作物理 RMSE。各模型的反变换结果已逐一核验，和训练汇总中保存的物理 RMSE 一致。", "",
            "这里的目标是相对 1° 网格气候态的温盐异常。RMSE、MAE、偏差与绝对温盐误差相同，因为同一气候态在差值中抵消；**R² 与 Pearson r 是异常值指标**，不是绝对温盐的相关性。偏差符号为预测减观测。", "",
            "本设置使用真实 Argo 和真实卫星产品；它与用户提供的 CESM2、6,080 输入剖面/月、2005 测试年、15,000 步设置不同，不能将本表数值直接与那张合成实验表比较。", "",
            "## 三次独立训练的均值与样本标准差", "",
            "三个种子为 1234、1235、1236，每次训练 6,000 步。下表先分别评价每个模型，再报告 mean ± sample SD（ddof=1）；它不等于平均预测后的集成指标。固定匹配基线只评价一次。", "",
            "| 方法 | 温度 RMSE (°C) | 盐度 RMSE (PSU) | 温度 MAE (°C) | 盐度 MAE (PSU) |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| 卫星初猜 + 已保存 OI | {baseline['TEMP']['rmse']:.5f} | {baseline['SALT']['rmse']:.5f} | {baseline['TEMP']['mae']:.5f} | {baseline['SALT']['mae']:.5f} |"]
    for group in groups:
        stats = group["seed_statistics"]
        rows.append(f"| {group['label']} | {mean_sd(stats['TEMP']['rmse'])} | {mean_sd(stats['SALT']['rmse'])} | {mean_sd(stats['TEMP']['mae'])} | {mean_sd(stats['SALT']['mae'])} |")
    rows += ["", "## 三种子集成：完整常规指标", "",
             "各成员在同一评分身份上平均预测。最终采用 " + selected["label"] +
             "，选择依据是冻结前的 2021 验证集温盐标准化 RMSE 均值；本报告不按开发集重新挑选架构。", "",
             "| 方法 | 变量 | RMSE | MAE | 平均偏差 | 异常 R² | 异常 Pearson r |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, metrics in [("卫星初猜 + 已保存 OI", baseline)] + [(g["label"], g["ensemble"]["pooled"]) for g in groups]:
        for ch in CHANNELS:
            m = metrics[ch]
            rows.append(f"| {label} | {'温度' if ch == 'TEMP' else '盐度'} ({UNITS[ch]}) | " +
                        " | ".join(format_number(m[k]) for k in REGRESSION_KEYS) + " |")
    rows += ["", "## 验证集选中架构相对匹配基线", "",
             "差值为集成模型减基线；负数表示降低误差。区间使用 "
             f"{selected['ensemble']['paired_month_rmse_bootstrap']['TEMP']['n_groups']} 个整月配对 bootstrap（"
             f"{selected['ensemble']['paired_month_rmse_bootstrap']['TEMP']['draws']:,} 次），不包含训练种子或更长时间相关性的不确定性。", "",
             "| 变量 | 基线 RMSE | 选中集成 RMSE | RMSE 降幅 | RMSE 差值 95% 区间 |",
             "| --- | ---: | ---: | ---: | --- |"]
    for ch in CHANNELS:
        m = selected["ensemble"]["pooled"][ch]
        ci = selected["ensemble"]["paired_month_rmse_bootstrap"][ch]["ci95_delta_rmse"]
        improvement = 100 * (baseline[ch]["rmse"] - m["rmse"]) / baseline[ch]["rmse"]
        rows.append(f"| {'温度' if ch == 'TEMP' else '盐度'} ({UNITS[ch]}) | {baseline[ch]['rmse']:.5f} | {m['rmse']:.5f} | {improvement:.2f}% | [{ci[0]:.5f}, {ci[1]:.5f}] |")
    rows += ["", "## 不确定性指标", "",
             "NLL 是以 °C/PSU 为单位的高斯预测密度的自然对数负值，可能为负，且不能跨变量单位直接比较。CRPS 与对应变量同单位，越低越好。68% 列使用 ±1σ（名义 68.27%）；95% 列使用 ±1.959964σ。集成以总方差公式构造矩匹配高斯；所报 NLL/CRPS 是该高斯的指标，不是精确高斯混合密度指标。", "",
             "基线的每深度/变量 σ 仅用 2021 验证残差 RMS 拟合，均值未校正。OI-only 对照的标准化 σ 固定为 0.6，因此其不确定性不能解释为独立训练的校准模型。", "",
             "| 方法 | 变量 | 高斯 NLL | 高斯 CRPS | 68% 覆盖率 | 95% 覆盖率 | 平均 σ |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, metrics in [("卫星初猜 + OI（验证校准 σ）", baseline)] + [(g["label"], g["ensemble"]["pooled"]) for g in groups]:
        for ch in CHANNELS:
            m = metrics[ch]
            rows.append(f"| {label} | {'温度' if ch == 'TEMP' else '盐度'} | {m['nll']:.5f} | {m['crps']:.5f} | {100*m['coverage_68']:.2f}% | {100*m['coverage_95']:.2f}% | {m['mean_std']:.5f} |")
    rows += ["", "## 选中架构的逐深度与深度带 RMSE", "",
             "| 深度 (m) | 温度基线 RMSE (°C) | 温度集成 RMSE (°C) | 盐度基线 RMSE (PSU) | 盐度集成 RMSE (PSU) | 每变量有效 n |",
             "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for b, m in zip(report["baseline"]["by_depth"], selected["ensemble"]["by_depth"]):
        bm, mm = b["metrics"], m["metrics"]
        rows.append(f"| {m['depth_m']:.1f} | {bm['TEMP']['rmse']:.5f} | {mm['TEMP']['rmse']:.5f} | {bm['SALT']['rmse']:.5f} | {mm['SALT']['rmse']:.5f} | {mm['TEMP']['n']:,} / {mm['SALT']['n']:,} |")
    rows += ["", "| 深度带 | 温度基线 RMSE (°C) | 温度集成 RMSE (°C) | 盐度基线 RMSE (PSU) | 盐度集成 RMSE (PSU) |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for b, m in zip(report["baseline"]["by_depth_band"], selected["ensemble"]["by_depth_band"]):
        bm, mm = b["metrics"], m["metrics"]
        rows.append(f"| {m['depth_band']} | {bm['TEMP']['rmse']:.5f} | {mm['TEMP']['rmse']:.5f} | {bm['SALT']['rmse']:.5f} | {mm['SALT']['rmse']:.5f} |")
    rows += ["", "全部七种配置的单种子、集成、逐深度、深度带指标，以及各深度/深度带的种子均值与样本标准差、物理单位变换参数、预测文件 SHA256 和整月配对区间保存在同名 JSON。", "",
             "上述对照支持相同真实数据设置下的内部比较。卫星 L4 产品可能复用原位资料，当前同月输入也不是因果预报；这里尚不能据此宣称独立测试 SOTA 或任意深度连续推理已完成。", ""]
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/latent_ocean")
    parser.add_argument("--selection", type=Path, default=ROOT / "outputs/latent_ocean/final_selection_20261007.json")
    parser.add_argument("--first-guess", type=Path, default=ROOT / "outputs/audit/global/anc_satday_dfs/first_guess_seed1234.pt")
    parser.add_argument("--json", type=Path, default=ROOT / "outputs/latent_ocean/standard_metrics_20261007.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/real_data/standard_metrics_20261007.md")
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    args = parser.parse_args()
    report = build_report(args.output_root, args.selection, args.first_guess, args.bootstrap_draws)
    report["reporting_source_sha256"] = {str(Path(__file__)): sha256(__file__),
        str(ROOT / "src/ocean_tokenizer/reconstruction_metrics.py"): sha256(ROOT / "src/ocean_tokenizer/reconstruction_metrics.py")}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.report.write_text(render_report(report))
    print(json.dumps({"json": str(args.json), "report": str(args.report),
                      "candidates": len(report["groups"]), "scored_models": sum(len(g["seeds"]) for g in report["groups"]),
                      "selected": report["frozen_selection"]["selected_configuration"]}), flush=True)


if __name__ == "__main__":
    main()
