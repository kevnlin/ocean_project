"""Report the registered ocean latent campaign without using development to select.

Only manifest jobs are eligible. Large prediction arrays are read one seed at a
time; ensemble moments are accumulated without stacking seed predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHANNELS = ("TEMP", "SALT")
ARRAY_KEYS = ("mean", "std", "target", "baseline", "month", "profile", "level")
IDENTITY_KEYS = ("month", "profile", "level", "target", "baseline")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def same_configuration(a: dict, b: dict) -> bool:
    return (a["config"] == b["config"]
            and all(a["training"].get(k) == b["training"].get(k)
                    for k in ("steps", "queries", "context_profiles", "eval_context_profiles", "freeze_analysis", "analysis_only",
                              "lr", "analysis_lr", "weight_decay", "nll_weight", "amp", "val_cells", "val_every")))


def match_job(job: dict, summary: dict) -> None:
    """Reject accidental old/integration summaries at a registered output path."""
    expected = {"variant": job["variant"], "width": job["width"],
                "n_latents": job["latents"], "n_blocks": job["blocks"], "n_experts": job["experts"]}
    if summary["tag"] != job["tag"] or summary["seed"] != job["seed"]:
        raise ValueError(f"summary identity differs from manifest: {job['tag']}")
    if any(summary["config"].get(k) != value for k, value in expected.items()):
        raise ValueError(f"summary configuration differs from manifest: {job['tag']}")
    for k in ("steps", "context_profiles"):
        if summary["training"].get(k) != job[k]:
            raise ValueError(f"summary training {k} differs from manifest: {job['tag']}")
    if bool(summary["training"].get("analysis_only", False)) != bool(job.get("analysis_only", False)):
        raise ValueError(f"summary analysis-only status differs from manifest: {job['tag']}")
    for name in ("freeze_analysis", "latent_off", "local_off"):
        if name in job and bool(summary["training"].get(name, False)) != bool(job[name]):
            raise ValueError(f"summary {name} differs from manifest: {job['tag']}")


def load_campaign(manifest: dict, output_root: Path) -> tuple[list[dict], list[dict]]:
    completed, pending = [], []
    for job in manifest["jobs"]:
        folder = output_root / job["tag"]
        path = folder / f"summary_seed{job['seed']}.json"
        if path.exists():
            summary = read_json(path)
            match_job(job, summary)
            completed.append({"job": job, "summary": summary, "folder": folder})
        else:
            log = folder / "training.log"
            tail = log.read_text(errors="replace").splitlines()[-1] if log.exists() and log.stat().st_size else ""
            pending.append({"tag": job["tag"], "status": "unfinished", "last_log_line": tail})
    return completed, pending


def validation_selection(completed: list[dict], pending: list[dict]) -> list[dict]:
    """Freeze only after all registered search jobs finish, including OI control."""
    if pending:
        return []
    winners: dict[str, dict] = {}
    for item in completed:
        job, summary = item["job"], item["summary"]
        group = "analysis_only" if job.get("analysis_only") else job["variant"]
        score = float(summary["validation"]["scores"]["macro_z"])
        if not np.isfinite(score):
            raise ValueError(f"nonfinite validation selection score: {job['tag']}")
        if group not in winners or score < winners[group]["summary"]["validation"]["scores"]["macro_z"]:
            winners[group] = item
    return [winners[name] for name in sorted(winners)]


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        missing = set(ARRAY_KEYS) - set(stored.files)
        if missing:
            raise ValueError(f"prediction file lacks {sorted(missing)}: {path}")
        arrays = {k: stored[k] for k in ARRAY_KEYS}
    n = len(arrays["month"])
    for k in ("mean", "std", "target", "baseline"):
        if arrays[k].shape != (n, 2):
            raise ValueError(f"invalid {k} shape: {path}")
    for k in ("month", "profile", "level"):
        if arrays[k].shape != (n,):
            raise ValueError(f"invalid {k} shape: {path}")
    if n == 0 or np.any(arrays["level"] < 0):
        raise ValueError(f"empty/invalid scoring identity: {path}")
    mask = np.isfinite(arrays["target"])
    if any(np.any(~np.isfinite(arrays[k][mask])) for k in ("mean", "std", "baseline")):
        raise ValueError(f"nonfinite prediction for an observed target: {path}")
    if np.any(arrays["std"][mask] <= 0):
        raise ValueError(f"nonpositive uncertainty: {path}")
    return arrays


def assert_identical(reference: dict, candidate: dict) -> None:
    for k in IDENTITY_KEYS:
        if not np.array_equal(reference[k], candidate[k], equal_nan=True):
            raise ValueError(f"seed scoring identities differ in {k}; refusing ensemble")


def gaussian_metrics(mean: np.ndarray, std: np.ndarray, target: np.ndarray) -> dict:
    out = {}
    for j, ch in enumerate(CHANNELS):
        valid = np.isfinite(target[:, j])
        if not valid.any():
            out[ch] = {"n": 0, "nll": None, "coverage_68": None, "coverage_95": None, "mean_std_z": None}
            continue
        error = np.asarray(mean[valid, j], dtype=np.float64) - target[valid, j]
        sigma = np.asarray(std[valid, j], dtype=np.float64)
        out[ch] = {"n": int(valid.sum()),
                   "nll": float(np.mean(.5 * (error / sigma) ** 2 + np.log(sigma) + .5 * np.log(2 * np.pi))),
                   "coverage_68": float(np.mean(np.abs(error) <= sigma)),
                   "coverage_95": float(np.mean(np.abs(error) <= 1.959963984540054 * sigma)),
                   "mean_std_z": float(np.mean(sigma))}
    return out


def z_scores(mean: np.ndarray, target: np.ndarray) -> dict:
    out = {}
    for j, ch in enumerate(CHANNELS):
        mask = np.isfinite(target[:, j])
        if not mask.any():
            out[ch] = {"n": 0, "rmse_z": None, "J": None}
            continue
        error = np.asarray(mean[mask, j], dtype=np.float64) - target[mask, j]
        rmse = float(np.sqrt(np.mean(error ** 2)))
        rms_target = float(np.sqrt(np.mean(np.asarray(target[mask, j], dtype=np.float64) ** 2)))
        out[ch] = {"n": int(mask.sum()), "rmse_z": rmse, "J": rmse / max(rms_target, 1e-9)}
    active = [out[ch]["rmse_z"] for ch in CHANNELS if out[ch]["n"]]
    out["macro_z"] = float(np.mean(active)) if active else None
    return out


def calibrate_baseline(validation: dict) -> dict:
    """A zero-bias Gaussian scale per depth/variable, fit on 2021 only."""
    levels, target, mean = validation["level"], validation["target"], validation["baseline"]
    n_levels = int(levels.max()) + 1
    sigma = np.zeros((n_levels, 2), dtype=np.float64)
    counts = np.zeros_like(sigma, dtype=np.int64)
    fallback = np.zeros(2)
    for j in range(2):
        ok = np.isfinite(target[:, j])
        if not ok.any():
            raise ValueError("baseline calibration requires validation targets for each variable")
        errors = np.asarray(mean[:, j], dtype=np.float64) - target[:, j]
        fallback[j] = max(float(np.sqrt(np.mean(errors[ok] ** 2))), 1e-4)
        for level in range(n_levels):
            mask = ok & (levels == level)
            counts[level, j] = mask.sum()
            sigma[level, j] = max(float(np.sqrt(np.mean(errors[mask] ** 2))), 1e-4) if mask.any() else fallback[j]
    return {"split": "validation_2021", "method": "per-level, per-variable residual RMS, no bias correction",
            "sigma_z": sigma.tolist(), "counts": counts.tolist(), "fallback_sigma_z": fallback.tolist(),
            "sigma_floor_z": 1e-4}


def calibrated_std(calibration: dict, levels: np.ndarray) -> np.ndarray:
    sigma = np.asarray(calibration["sigma_z"], dtype=np.float64)
    result = np.broadcast_to(calibration["fallback_sigma_z"], (len(levels), 2)).copy()
    valid = (levels >= 0) & (levels < len(sigma))
    result[valid] = sigma[levels[valid]]
    return result


def paired_month_bootstrap(arrays: dict, draws: int = 4000, seed: int = 20261007) -> dict:
    """Compare pooled scores with paired whole-month sampling, not cell resampling."""
    months = np.unique(arrays["month"])
    stats = np.zeros((len(months), 2, 4), dtype=np.float64)
    for mi, month in enumerate(months):
        for j in range(2):
            mask = (arrays["month"] == month) & np.isfinite(arrays["target"][:, j])
            y = np.asarray(arrays["target"][mask, j], dtype=np.float64)
            stats[mi, j] = (np.sum((arrays["mean"][mask, j] - y) ** 2),
                           np.sum((arrays["baseline"][mask, j] - y) ** 2), np.sum(y ** 2), len(y))
    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    rng = np.random.default_rng(seed)
    # Only monthly sufficient statistics are expanded, never the underlying cells.
    sampled = rng.integers(0, len(months), size=(draws, len(months)))
    totals = stats[sampled].sum(axis=1)
    original = stats.sum(axis=0)
    channels = {}
    macro_draws, macro_estimates = [], []
    for j, ch in enumerate(CHANNELS):
        se, base_se, y2, count = original[j]
        if not count:
            channels[ch] = {"delta_J": None, "ci95_delta_J": None, "delta_rmse_z": None, "ci95_delta_rmse_z": None}
            continue
        eligible = totals[:, j, 3] > 0
        drawn = totals[eligible, j]
        delta_rmse = np.sqrt(drawn[:, 0] / drawn[:, 3]) - np.sqrt(drawn[:, 1] / drawn[:, 3])
        delta_j = delta_rmse / np.maximum(np.sqrt(drawn[:, 2] / drawn[:, 3]), 1e-9)
        estimate = float(np.sqrt(se / count) - np.sqrt(base_se / count))
        channels[ch] = {"delta_J": estimate / max(float(np.sqrt(y2 / count)), 1e-9),
                        "ci95_delta_J": np.quantile(delta_j, [.025, .975]).tolist(),
                        "delta_rmse_z": estimate, "ci95_delta_rmse_z": np.quantile(delta_rmse, [.025, .975]).tolist()}
        # These cohorts have both channels in every month. Keep generic sparse handling.
        full = np.full(draws, np.nan)
        full[eligible] = delta_rmse
        macro_draws.append(full)
        macro_estimates.append(estimate)
    macro_values = np.nanmean(macro_draws, axis=0) if macro_draws else np.array([])
    macro_values = macro_values[np.isfinite(macro_values)]
    return {"unit": "paired whole calendar month", "months": months.tolist(), "n_months": len(months),
            "draws": draws, "seed": seed, "sign": "negative means lower error than matched anchor",
            "channels": channels, "delta_macro_z": float(np.mean(macro_estimates)) if macro_estimates else None,
            "ci95_delta_macro_z": np.quantile(macro_values, [.025, .975]).tolist() if len(macro_values) else None,
            "limitation": "month resampling does not measure training-seed uncertainty or longer temporal dependence"}


def by_level(arrays: dict, baseline_std: np.ndarray) -> list[dict]:
    out = []
    for level in np.unique(arrays["level"]):
        mask = arrays["level"] == level
        out.append({"level_index": int(level),
                    "scores": z_scores(arrays["mean"][mask], arrays["target"][mask]),
                    "baseline_scores": z_scores(arrays["baseline"][mask], arrays["target"][mask]),
                    "uncertainty": gaussian_metrics(arrays["mean"][mask], arrays["std"][mask], arrays["target"][mask]),
                    "calibrated_baseline_uncertainty": gaussian_metrics(arrays["baseline"][mask], baseline_std[mask], arrays["target"][mask])})
    return out


def analyse_arrays(arrays: dict, calibration: dict, draws: int) -> dict:
    sigma = calibrated_std(calibration, arrays["level"])
    return {"scores": z_scores(arrays["mean"], arrays["target"]),
            "baseline_scores": z_scores(arrays["baseline"], arrays["target"]),
            "uncertainty": gaussian_metrics(arrays["mean"], arrays["std"], arrays["target"]),
            "calibrated_baseline_uncertainty": gaussian_metrics(arrays["baseline"], sigma, arrays["target"]),
            "paired_month_bootstrap": paired_month_bootstrap(arrays, draws), "by_level": by_level(arrays, sigma)}


def ensemble_predictions(paths: list[Path]) -> dict:
    """Streaming law-of-total-variance moments; reject nonidentical seed evidence."""
    if len(paths) < 2:
        raise ValueError("an ensemble requires at least two seeds")
    reference = load_predictions(paths[0])
    mean_sum = np.asarray(reference["mean"], dtype=np.float64).copy()
    second_sum = np.asarray(reference["std"], dtype=np.float64) ** 2 + mean_sum ** 2
    for path in paths[1:]:
        candidate = load_predictions(path)
        assert_identical(reference, candidate)
        mean_sum += candidate["mean"]
        second_sum += np.asarray(candidate["std"], dtype=np.float64) ** 2 + np.asarray(candidate["mean"], dtype=np.float64) ** 2
        del candidate
    mean = mean_sum / len(paths)
    variance = np.maximum(second_sum / len(paths) - mean ** 2, 0)
    reference["mean"] = mean
    reference["std"] = np.sqrt(variance)
    return reference


def seed_statistics(summaries: list[dict], split: str) -> dict:
    present = [s for s in summaries if split in s]
    result: dict[str, Any] = {"n_seeds": len(present), "seeds": [s["seed"] for s in present]}
    for metric in ("TEMP_J", "SALT_J", "macro_z"):
        values = [s[split]["scores"]["macro_z"] if metric == "macro_z"
                  else s[split]["scores"][metric.split("_")[0]]["J"] for s in present]
        result[metric] = {"mean": float(np.mean(values)) if values else None,
                          "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                          "values": values}
    return result


def collect_replicates(selected: list[dict], output_root: Path) -> list[list[dict]]:
    groups = []
    for winner in selected:
        items = [winner]
        prefix = winner["job"]["tag"].rsplit("_s", 1)[0]
        for folder in sorted(output_root.glob(prefix + "_s*")):
            if folder == winner["folder"]:
                continue
            for path in sorted(folder.glob("summary_seed*.json")):
                summary = read_json(path)
                if not same_configuration(winner["summary"], summary):
                    raise ValueError(f"replicate architecture/protocol differs: {path}")
                for fingerprint in ("cohort_fingerprint", "feature_fingerprint", "scene_fingerprint"):
                    if winner["summary"]["data"].get(fingerprint) != summary["data"].get(fingerprint):
                        raise ValueError(f"replicate data differs: {path}")
                job = {**winner["job"], "tag": summary["tag"], "seed": summary["seed"]}
                match_job(job, summary)
                items.append({"job": job, "summary": summary, "folder": folder})
        seeds = [item["summary"]["seed"] for item in items]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate replicate seeds: {prefix}")
        groups.append(sorted(items, key=lambda item: item["summary"]["seed"]))
    return groups


def format_value(value: float | None) -> str:
    return "—" if value is None else f"{value:.5f}"


def render_report(report: dict) -> str:
    rows = ["# 多模态海洋共享潜在状态实验", "",
            f"正式 search 已完成 {report['completed_count']}/{report['registered_count']}；开发集结果只能在模型和配置冻结后生成。", "",
            "目标：从 SST、SSS、SLA 和稀疏 Argo 逐深度观测重建连续海洋温盐状态，再通过独立位置、深度、日期查询得到温度、盐度与不确定性。当前数据与评估仅覆盖 20 个固定深度层；尚未证明任意深度的连续重建能力。", "",
            "## 架构与对照", "",
            "固定已保存的卫星初猜、训练归一化、输入剖面池和评分身份。个体深度观测编码器读入卫星特征、初猜、测量误差及有效性，并保留数值新息；共享 latent 通过观测 cross-attention 与 latent self-attention 交换信息，独立查询解码器输出温盐残差和高斯尺度，同时保留原始局部逐层新息通路。", "",
            "比较 dense latent Transformer、latent Soft MoE 和带每查询局部 Transformer 的 latent backbone；每类包含 base/large 两档。Soft MoE 使用连续 dispatch/combine，不能据此宣称稀疏计算加速。卫星 token 来自上下文剖面位置的卫星特征，尚未直接编码完整稠密卫星栅格。", "",
            "新模型默认同时训练 OI 参数，因此增加 analysis-only 对照。latent 的收益应同时相对固定初猜加 OI anchor，以及同设置重新训练的 OI-only 对照评估。当前新增模块并不单独保证重复记录证据不变性。", "",
            "## 2021 验证集架构搜索", "",
            "仅以验证集标准化温盐 RMSE 的平均值 macro_z 选取 checkpoint 与 base/large 配置；J 为 RMSE / 目标 RMS，越小越好。step 0 可成为最佳 checkpoint，表示当前训练没有超越初始化 anchor。", "",
            "| 配置 | 状态 | 神经参数 | 最佳 step | TEMP J | SALT J | macro_z |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for item in report["search"]:
        s = item["summary"]
        score = s["validation"]["scores"]
        rows.append(f"| {s['tag']} | 完成 | {s['params']:,} | {s['best_step']} | {score['TEMP']['J']:.5f} | {score['SALT']['J']:.5f} | {score['macro_z']:.5f} |")
    for item in report["pending"]:
        rows.append(f"| {item['tag']} | 未完成 | — | — | — | — | — |")
    rows += ["", "冻结选择：" + (", ".join(report["selected_tags"]) if report["selected_tags"] else "尚未冻结；必须完成全部注册 search。"), "",
             "## 多种子与已使用开发集", "",
             "训练为 2016–2020，验证为 2021。2022–2023 早已用于先前开发，属于 development，不能作为独立 test 确认或跨论文 SOTA 证据。同月上下文可包含查询日期之后的观测，因此这里是回顾性重建；预测实验需要严格另设时间截断。归一化沿用全部训练年份剖面的既定协议，包含划为上下文之外的训练 WMO，不能宣称这些浮标未参与预处理。", "",
             "SST/SSS L4 产品可能同化原位观测，其中的 Argo 证据可能回流到卫星输入；沿用 62_sanity_train.py 的既定协议，这一设置不是严格 source-independent 证据评估。所有方法使用相同产品仍可支持本协议下的匹配比较，但不能宣称已排除跨产品复用信息。", "",
             "| 冻结配置 | split | seeds | TEMP J mean ± std | SALT J mean ± std | macro_z mean ± std |",
             "| --- | --- | ---: | --- | --- | --- |"]
    for group in report["groups"]:
        for split in ("validation", "development"):
            stats = group["seed_statistics"][split]
            vals = [f"{format_value(stats[m]['mean'])} ± {format_value(stats[m]['std'])}" for m in ("TEMP_J", "SALT_J", "macro_z")]
            rows.append(f"| {group['configuration']} | {split} | {stats['n_seeds']} | " + " | ".join(vals) + " |")
    if not report["groups"]:
        rows += ["", "正式配置尚未冻结，未汇总开发集或种子 ensemble。"]
    if report.get("ablations"):
        rows += ["", "## 所选架构的固定 OI 与成分消融", "",
                 "消融单独收录，不参与 base/large 配置选择。", "",
                 "| 配置 | 固定 OI | latent | 局部通路 | 最佳 step | validation TEMP J | validation SALT J | development TEMP J | development SALT J |",
                 "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for item in report["ablations"]:
            s = item["summary"]
            config, training = s["config"], s["training"]
            validation = s["validation"]["scores"]
            dev = s.get("development", {}).get("scores", {})
            rows.append(f"| {s['tag']} | {bool(training.get('freeze_analysis'))} | {config.get('use_latent', True)} | {config.get('use_local', True)} | {s['best_step']} | {validation['TEMP']['J']:.5f} | {validation['SALT']['J']:.5f} | {format_value(dev.get('TEMP', {}).get('J'))} | {format_value(dev.get('SALT', {}).get('J'))} |")
        rows.append("")
    rows += ["", "## 不确定性与配对统计", "",
             "anchor 的温盐 Gaussian sigma 按 level/variable 仅在 2021 验证残差上拟合 RMS，不修正均值；冻结后应用于 development。其在验证集上的校准结果为拟合内结果，不能视作独立校准验证。模型 sigma 为标准化异常值单位，不是物理单位。", "",
             "多种子预测只有 month/profile/level/target/baseline 完全一致才可平均；ensemble variance = mean(sigma² + mean²) − ensemble_mean²。报告使用整月配对 bootstrap，差值为模型减匹配 anchor，负值代表改进；此区间未包含训练种子或长周期相关性的不确定性。", ""]
    for group in report["groups"]:
        results = list(group["development_details"].items())
        if group.get("ensemble"):
            results.append(("ensemble", group["ensemble"]))
        for label, details in results:
            rows += [f"### {group['configuration']} / {label}", "",
                     "| 变量 | J | anchor J | ΔJ 95% month CI | NLL | anchor calibrated NLL | 68% coverage | 95% coverage |",
                     "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
            for ch in CHANNELS:
                score, base = details["scores"][ch], details["baseline_scores"][ch]
                ci = details["paired_month_bootstrap"]["channels"][ch]["ci95_delta_J"]
                uncertainty, bu = details["uncertainty"][ch], details["calibrated_baseline_uncertainty"][ch]
                interval = "—" if ci is None else f"[{ci[0]:.5f}, {ci[1]:.5f}]"
                rows.append(f"| {ch} | {format_value(score['J'])} | {format_value(base['J'])} | {interval} | {format_value(uncertainty['nll'])} | {format_value(bu['nll'])} | {format_value(uncertainty['coverage_68'])} | {format_value(uncertainty['coverage_95'])} |")
            rows.append("")
    rows += ["逐深度指标、calibration sigma/counts、完整配对差值与种子列表保存在同名 JSON。未完成训练、未生成数组或缺少开发集时不填造数字。", "",
             "方法来源：[Perceiver IO](https://arxiv.org/abs/2107.14795)、[Soft MoE](https://arxiv.org/abs/2308.00951)、[ARMOR3D](https://os.copernicus.org/articles/8/845/2012/)。已有卫星初猜加剖面修正先例；这里的潜在贡献需由共享状态更新、准确率、校准和泛化实验支持。", ""]
    return "\n".join(rows)


def build_report(manifest_path: Path, output_root: Path, draws: int = 4000,
                 ablation_manifest: Path | None = None) -> dict:
    manifest = read_json(manifest_path)
    completed, pending = load_campaign(manifest, output_root)
    selected = validation_selection(completed, pending)
    report = {"schema_version": 1, "manifest": str(manifest_path),
              "selection": "2021 validation macro_z only; all registered search jobs must finish",
              "registered_count": len(manifest["jobs"]), "completed_count": len(completed),
              "search": [{"job": item["job"], "summary": item["summary"]} for item in completed],
              "pending": pending, "selected_tags": [item["job"]["tag"] for item in selected], "groups": [],
              "ablations": [], "ablation_pending": []}
    if ablation_manifest:
        ablations, ablation_pending = load_campaign(read_json(ablation_manifest), output_root)
        report["ablation_manifest"] = str(ablation_manifest)
        report["ablations"] = [{"job": item["job"], "summary": item["summary"]} for item in ablations]
        report["ablation_pending"] = ablation_pending
        for item, stored in zip(ablations, report["ablations"]):
            seed, folder = item["summary"]["seed"], item["folder"]
            val_path, dev_path = folder / f"validation_seed{seed}.npz", folder / f"development_seed{seed}.npz"
            if val_path.exists():
                validation = load_predictions(val_path)
                calibration = calibrate_baseline(validation)
                stored["baseline_calibration"] = calibration
                del validation
                if "development" in item["summary"] and dev_path.exists():
                    arrays = load_predictions(dev_path)
                    stored["development_details"] = analyse_arrays(arrays, calibration, draws)
                    del arrays
    for items in collect_replicates(selected, output_root):
        summaries = [item["summary"] for item in items]
        group = {"configuration": items[0]["job"]["tag"].rsplit("_s", 1)[0],
                 "seed_statistics": {split: seed_statistics(summaries, split) for split in ("validation", "development")},
                 "development_details": {}, "ensemble": None}
        dev_paths = []
        shared_calibration = None
        for item in items:
            summary, folder, seed = item["summary"], item["folder"], item["summary"]["seed"]
            val_path = folder / f"validation_seed{seed}.npz"
            if not val_path.exists():
                continue
            validation = load_predictions(val_path)
            calibration = calibrate_baseline(validation)
            if shared_calibration is None:
                shared_calibration = calibration
            elif calibration != shared_calibration:
                raise ValueError("matched baseline validation calibration differs across seeds")
            del validation
            dev_path = folder / f"development_seed{seed}.npz"
            if "development" in summary and dev_path.exists():
                arrays = load_predictions(dev_path)
                group["development_details"][f"seed{seed}"] = analyse_arrays(arrays, calibration, draws)
                del arrays
                dev_paths.append(dev_path)
        group["baseline_calibration"] = shared_calibration
        if len(dev_paths) >= 2:
            arrays = ensemble_predictions(dev_paths)
            group["ensemble"] = {**analyse_arrays(arrays, shared_calibration, draws),
                                 "n_seeds": len(dev_paths), "prediction_paths": [str(p) for p in dev_paths],
                                 "variance": "mean(sigma^2 + mean^2) - ensemble_mean^2"}
            del arrays
        report["groups"].append(group)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/latent_ocean/campaign_20261007.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/latent_ocean")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/real_data/latent_ocean_20261007.md")
    parser.add_argument("--json", type=Path, default=ROOT / "outputs/latent_ocean/comparison_20261007.json")
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    parser.add_argument("--ablation-manifest", type=Path, help="separate registered fixed-OI/component controls, never used for capacity selection")
    parser.add_argument("--freeze-selection", type=Path, help="write a validation-only selected manifest for 70 replication")
    args = parser.parse_args()
    report = build_report(args.manifest, args.output_root, args.bootstrap_draws, args.ablation_manifest)
    if args.freeze_selection:
        if report["pending"]:
            raise SystemExit("refusing to freeze selection before all registered search jobs finish")
        selected = [item["job"] for item in report["search"] if item["job"]["tag"] in report["selected_tags"]]
        frozen = {"version": 1, "phase": "frozen_selection", "source_manifest": str(args.manifest),
                  "selection": report["selection"], "selected": selected,
                  "validation_macro_z": {item["summary"]["tag"]: item["summary"]["validation"]["scores"]["macro_z"] for item in report["search"]},
                  "development": "not used in selection; 2022-2023 are previously used development years"}
        args.freeze_selection.parent.mkdir(parents=True, exist_ok=True)
        if args.freeze_selection.exists() and read_json(args.freeze_selection) != frozen:
            raise SystemExit("refusing to overwrite a different frozen selection")
        args.freeze_selection.write_text(json.dumps(frozen, indent=2, allow_nan=False) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(report))
    args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.report), "json": str(args.json),
                      "completed": report["completed_count"], "registered": report["registered_count"],
                      "selected": report["selected_tags"]}))


if __name__ == "__main__":
    main()
