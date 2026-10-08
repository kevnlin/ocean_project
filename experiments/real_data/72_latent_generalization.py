"""Geographic and context-proximity diagnostics for frozen ocean configurations.

This analysis consumes saved predictions only. Raw cohort reads are restricted
to profile metadata; it neither trains models nor opens raw T/S observations.
The geographic partitions are diagnostics within the inherited development
protocol, not independent out-of-distribution tests.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("ocean_latent_report", Path(__file__).with_name("71_latent_report.py"))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
EARTH_RADIUS_KM = 6371.0088
EVAL_SEED = 20260918
FINGERPRINT_KEYS = ("cohort_fingerprint", "feature_fingerprint", "scene_fingerprint")
LATITUDE_EDGES = np.array([-90., -60., -30., 0., 30., 60., 90.])
LONGITUDE_EDGES = np.arange(-180., 181., 60.)
DISTANCE_EDGES = np.array([0., 25., 50., 100., 200., 500., np.inf])
SEASONS = ("DJF", "MAM", "JJA", "SON")


def array_fingerprint(arrays) -> str:
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def read_metadata(path: Path) -> dict:
    """Reproduce ArgoCohort.load's stable month order without loading T/S."""
    names = ("month_index", "lat", "lon", "wmo", "year", "float_split")
    with xr.open_dataset(path) as raw:
        values = {name: np.asarray(raw[name].values) for name in names}
        levels = np.asarray(raw["level"].values, dtype=float)
    n = len(values["month_index"])
    if any(a.shape != (n,) for a in values.values()) or levels.ndim != 1 or not len(levels):
        raise ValueError("raw cohort metadata shapes differ")
    order = np.argsort(np.asarray(values["month_index"], dtype=int), kind="stable")
    values = {name: value[order] for name, value in values.items()}
    for name in ("month_index", "year"):
        values[name] = values[name].astype(np.int64)
    for name in ("lat", "lon"):
        values[name] = values[name].astype(np.float64)
    for name in ("wmo", "float_split"):
        values[name] = values[name].astype(str)
    if not n or np.any(~np.isfinite(values["lat"])) or np.any(~np.isfinite(values["lon"])):
        raise ValueError("raw cohort has empty or nonfinite geographic metadata")
    if np.any(np.abs(values["lat"]) > 90.) or np.any(values["year"] != 2000 + values["month_index"] // 12):
        raise ValueError("raw cohort latitude/year metadata is invalid")
    if not set(np.unique(values["float_split"])) <= {"cohort_float", "heldout_float"}:
        raise ValueError("unexpected raw float_split labels")
    values["levels"] = levels
    values["path"] = str(path)
    values["metadata_fingerprint"] = array_fingerprint([values[name] for name in names] + [levels])
    return values


def validate_identity(arrays: dict, metadata: dict, split: str, max_cells: int = 0) -> None:
    """Bind profile indices to sorted raw metadata and the exact scoring order."""
    bounds = {"validation": (2021, 2021), "development": (2022, 2023)}
    if split not in bounds:
        raise ValueError("diagnostics support validation and development only")
    for name in ("month", "profile", "level"):
        if np.asarray(arrays[name]).dtype.kind not in "iu":
            raise ValueError(f"noninteger scoring identity: {name}")
    profile, level, month = arrays["profile"], arrays["level"], arrays["month"]
    if np.any(profile < 0) or np.any(profile >= len(metadata["lat"])):
        raise ValueError("scoring profile is outside raw cohort")
    if np.any(level < 0) or np.any(level >= len(metadata["levels"])):
        raise ValueError("scoring depth is outside raw cohort")
    if not np.array_equal(month, metadata["month_index"][profile]):
        raise ValueError("scoring month/profile mapping differs from sorted raw metadata")
    if np.any(metadata["float_split"][profile] != "heldout_float"):
        raise ValueError("scoring queries must belong to heldout_float")
    lower, upper = bounds[split]
    if np.any((metadata["year"][profile] < lower) | (metadata["year"][profile] > upper)):
        raise ValueError(f"scoring year is outside {split}")
    if np.any(np.diff(month.astype(np.int64)) < 0):
        raise ValueError("scoring months are not ordered")
    expected_months = []
    for m in np.unique(metadata["month_index"][(metadata["year"] >= lower) & (metadata["year"] <= upper)]):
        labels = metadata["float_split"][metadata["month_index"] == m]
        if np.any(labels == "cohort_float") and np.any(labels == "heldout_float"):
            expected_months.append(m)
    if not np.array_equal(np.unique(month), expected_months):
        raise ValueError("scoring months differ from the registered global evaluation years")
    n_levels = len(metadata["levels"])
    for m in np.unique(month):
        rows = np.flatnonzero(metadata["month_index"] == m)
        source = rows[metadata["float_split"][rows] == "cohort_float"]
        target = rows[metadata["float_split"][rows] == "heldout_float"]
        if not len(source) or np.intersect1d(metadata["wmo"][source], metadata["wmo"][target]).size:
            raise ValueError("context/query float identities overlap or context is empty")
        expected_profile = np.repeat(target, n_levels)
        expected_level = np.tile(np.arange(n_levels), len(target))
        if max_cells and len(expected_profile) > max_cells:
            pick = np.random.default_rng([EVAL_SEED, int(m)]).choice(len(expected_profile), max_cells, replace=False)
            expected_profile, expected_level = expected_profile[pick], expected_level[pick]
        mask = month == m
        if (not np.array_equal(profile[mask], expected_profile)
                or not np.array_equal(level[mask], expected_level)):
            raise ValueError("scoring cells differ from the registered heldout-profile/depth order")


def validate_fingerprints(summary: dict, reference: dict, metadata: dict, anchor: dict) -> None:
    data = summary["data"]
    for name in FINGERPRINT_KEYS:
        value = data.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"missing/invalid saved {name}")
        if value != reference.get(name):
            raise ValueError(f"configuration/seed data differs in {name}")
    for name in ("cohort_fingerprint", "feature_fingerprint"):
        if data[name] != anchor.get(name):
            raise ValueError(f"saved first-guess anchor differs in {name}")
    if data.get("n_profiles") != len(metadata["lat"]) or data.get("n_levels") != len(metadata["levels"]):
        raise ValueError("summary cohort dimensions differ from raw metadata")
    if not np.array_equal(np.asarray(anchor.get("levels")), metadata["levels"]):
        raise ValueError("raw depth coordinates differ from the saved first-guess anchor")


def validate_summary_scores(arrays: dict, summary: dict, split: str) -> None:
    """A same-shaped, unrelated prediction file must not pass as this summary."""
    for mean_key, score_key in (("mean", "scores"), ("baseline", "baseline_scores")):
        scores = R.z_scores(arrays[mean_key], arrays["target"])
        expected = summary[split][score_key]
        for channel in R.CHANNELS:
            for metric in ("n", "rmse_z", "J"):
                got, want = scores[channel][metric], expected[channel][metric]
                if metric == "n":
                    equal = got == want
                elif got is None or want is None:
                    equal = got is want
                else:
                    equal = np.isclose(got, want, rtol=1e-6, atol=1e-8)
                if not equal:
                    raise ValueError(f"saved {split} arrays differ from summary {score_key}/{channel}/{metric}")


def sphere_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def nearest_context_distances(arrays: dict, metadata: dict) -> np.ndarray:
    """One spherical tree per month; query each profile once, not every depth."""
    result = np.empty(len(arrays["profile"]), dtype=np.float64)
    for month in np.unique(arrays["month"]):
        mask = arrays["month"] == month
        source = np.flatnonzero((metadata["month_index"] == month)
                                & (metadata["float_split"] == "cohort_float"))
        unique, inverse = np.unique(arrays["profile"][mask], return_inverse=True)
        if not len(source):
            raise ValueError("nearest-context distance requires nonempty same-month context")
        tree = cKDTree(sphere_xyz(metadata["lat"][source], metadata["lon"][source]))
        chord, _ = tree.query(sphere_xyz(metadata["lat"][unique], metadata["lon"][unique]), k=1)
        distance = 2. * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2., 0., 1.))
        result[mask] = distance[inverse]
    return result


def interval_groups(values: np.ndarray, edges: np.ndarray, labels: list[str]) -> list[tuple[str, np.ndarray]]:
    out = []
    for index, label in enumerate(labels):
        mask = (values >= edges[index]) & (values < edges[index + 1])
        if index == len(labels) - 1 and np.isfinite(edges[-1]):
            mask |= values == edges[-1]
        out.append((label, mask))
    return out


def diagnostic_groups(arrays: dict, metadata: dict, distance_km: np.ndarray) -> dict:
    profile = arrays["profile"]
    lat, lon = metadata["lat"][profile], (metadata["lon"][profile] + 180.) % 360. - 180.
    calendar_month = arrays["month"] % 12 + 1
    seasons = np.where(np.isin(calendar_month, (12, 1, 2)), "DJF",
                       np.where(calendar_month <= 5, "MAM", np.where(calendar_month <= 8, "JJA", "SON")))
    return {
        "latitude_bands": interval_groups(lat, LATITUDE_EDGES, [f"[{a:g}, {b:g}{']' if b == 90. else ')'} deg" for a, b in zip(LATITUDE_EDGES[:-1], LATITUDE_EDGES[1:])]),
        "longitude_sectors": interval_groups(lon, LONGITUDE_EDGES, [f"[{a:g}, {b:g}) deg" for a, b in zip(LONGITUDE_EDGES[:-1], LONGITUDE_EDGES[1:])]),
        "calendar_seasons": [(season, seasons == season) for season in SEASONS],
        "nearest_context_distance_km": interval_groups(distance_km, DISTANCE_EDGES,
            ["[0, 25)", "[25, 50)", "[50, 100)", "[100, 200)", "[200, 500)", "[500, infinity)"]),
    }


def analyse_groups(arrays: dict, metadata: dict, calibration: dict, distance_km: np.ndarray | None = None) -> dict:
    if distance_km is None:
        distance_km = nearest_context_distances(arrays, metadata)
    if len(distance_km) != len(arrays["profile"]) or not np.isfinite(distance_km).all() or np.any(distance_km < 0.):
        raise ValueError("invalid nearest-context distance array")
    baseline_std = R.calibrated_std(calibration, arrays["level"])
    result = {}
    for family, partitions in diagnostic_groups(arrays, metadata, distance_km).items():
        result[family] = []
        for label, mask in partitions:
            target = arrays["target"][mask]
            scores = R.z_scores(arrays["mean"][mask], target)
            baseline_scores = R.z_scores(arrays["baseline"][mask], target)
            delta = {channel: (None if not scores[channel]["n"] else scores[channel]["J"] - baseline_scores[channel]["J"])
                     for channel in R.CHANNELS}
            result[family].append({"group": label, "n_cells": int(mask.sum()),
                "n_profiles": int(len(np.unique(arrays["profile"][mask]))),
                "scores": scores, "baseline_scores": baseline_scores, "delta_J": delta,
                "uncertainty": R.gaussian_metrics(arrays["mean"][mask], arrays["std"][mask], target),
                "calibrated_baseline_uncertainty": R.gaussian_metrics(arrays["baseline"][mask], baseline_std[mask], target)})
    return result


def load_anchor(path: Path) -> dict:
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def build_report(manifest_path: Path, output_root: Path, raw_path: Path) -> dict:
    manifest = R.read_json(manifest_path)
    completed, pending = R.load_campaign(manifest, output_root)
    selected = R.validation_selection(completed, pending)
    report = {"schema_version": 1, "manifest": str(manifest_path),
        "selection": "2021 validation macro_z only; entire registered search must finish",
        "task": "same-protocol geographic and context-proximity development diagnostics",
        "independent_ood": False, "selected_tags": [x["job"]["tag"] for x in selected],
        "search_pending": pending, "prediction_pending": [], "raw_metadata": None, "groups": []}
    metadata, anchor, reference_data = None, None, None
    reference_arrays = None
    distance_km = None
    for items in R.collect_replicates(selected, output_root):
        group = {"configuration": items[0]["job"]["tag"].rsplit("_s", 1)[0], "seeds": [], "ensemble": None}
        dev_paths, shared_calibration = [], None
        for item in items:
            summary, folder, seed = item["summary"], item["folder"], item["summary"]["seed"]
            dev_path, val_path = folder / f"development_seed{seed}.npz", folder / f"validation_seed{seed}.npz"
            if "development" not in summary or not dev_path.exists() or not val_path.exists():
                report["prediction_pending"].append({"tag": summary["tag"], "status": "development arrays/summary or validation arrays unfinished"})
                continue
            if metadata is None:
                metadata = read_metadata(raw_path)
                anchor = load_anchor(Path(summary["data"]["anchor"]) / "first_guess_seed1234.pt")
                reference_data = summary["data"]
                report["raw_metadata"] = {"path": str(raw_path), "n_profiles": len(metadata["lat"]),
                    "n_levels": len(metadata["levels"]), "metadata_fingerprint": metadata["metadata_fingerprint"],
                    "loaded_variables": ["month_index", "lat", "lon", "wmo", "year", "float_split", "level"],
                    "fingerprint_scope": "saved cohort/feature fingerprints match first guess; scene fingerprints match all configurations/seeds; full T/S fingerprint is not independently recomputed"}
                report["saved_data_fingerprints"] = {k: reference_data[k] for k in FINGERPRINT_KEYS}
            validate_fingerprints(summary, reference_data, metadata, anchor)
            validation = R.load_predictions(val_path)
            validate_identity(validation, metadata, "validation", summary["training"]["val_cells"])
            validate_summary_scores(validation, summary, "validation")
            calibration = R.calibrate_baseline(validation)
            if shared_calibration is None:
                shared_calibration = calibration
            elif calibration != shared_calibration:
                raise ValueError("matched anchor validation calibration differs across seeds")
            del validation
            arrays = R.load_predictions(dev_path)
            validate_identity(arrays, metadata, "development")
            validate_summary_scores(arrays, summary, "development")
            if reference_arrays is None:
                # Keep only identity/evidence fields; mean/std are streamed one seed at a time.
                reference_arrays = {k: arrays[k] for k in R.IDENTITY_KEYS}
                distance_km = nearest_context_distances(arrays, metadata)
            else:
                R.assert_identical(reference_arrays, arrays)
            group["seeds"].append({"seed": seed, "tag": summary["tag"], "prediction_path": str(dev_path),
                "prediction_fingerprint": array_fingerprint([arrays[k] for k in R.ARRAY_KEYS]),
                "diagnostics": analyse_groups(arrays, metadata, calibration, distance_km)})
            dev_paths.append(dev_path)
            del arrays
        group["baseline_calibration"] = shared_calibration
        if len(dev_paths) >= 2:
            arrays = R.ensemble_predictions(dev_paths)
            R.assert_identical(reference_arrays, arrays)
            group["ensemble"] = {"n_seeds": len(dev_paths), "prediction_paths": [str(p) for p in dev_paths],
                "variance": "mean(sigma^2 + mean^2) - ensemble_mean^2",
                "diagnostics": analyse_groups(arrays, metadata, shared_calibration, distance_km)}
            del arrays
        report["groups"].append(group)
    return report


def render_report(report: dict) -> str:
    lines = ["# Ocean latent 地理与观测邻近度诊断", "",
        "这里分析已保存的预测数组，不重新训练。配置只按 2021 validation 冻结；2022–2023 是先前已使用的 development。纬度带、经度区间、日历季节和上下文距离分组属于同一协议内的诊断，不能作为独立 OOD 测试或 SOTA 证据。", "",
        "距离是查询剖面到同月全部 cohort_float 剖面位置的最近大圆距离，地球半径取 6371.0088 km。每月用球面三维坐标建树，每个查询剖面只查一次，再映射到各深度；不表示该深度最近有效 T/S 观测的距离。经度区间不是海盆定义。DJF/MAM/JJA/SON 是日历季节，不对应所有半球的同名季节。纬度最后一带包含 90°。", "",
        "J = 分组内 RMSE / 分组内目标异常 RMS；ΔJ = 模型减匹配 anchor，负数为改进。每组分母不同，应比较该组的 ΔJ，不能把各组 J 直接当作同一尺度的误差排名。NLL 和 sigma 使用各深度标准化异常单位。anchor sigma 仅由 2021 验证残差按深度与变量拟合，冻结后用于 development。", "",
        "检查 profile/month/level 与原始元数据稳定排序、heldout_float 身份及评分顺序；检查保存的 cohort/feature 指纹与 first guess、scene 指纹在配置与种子间一致。原始文件只读取位置、年月、WMO、float_split 和深度，完整 T/S 指纹不在本分析中重新计算。跨种子 ensemble 要求 month/profile/level/target/baseline 五字段完全一致。", ""]
    if report["search_pending"]:
        lines += [f"注册搜索尚有 {len(report['search_pending'])} 组未完成，配置尚未冻结，不填开发集诊断数字。", ""]
    if report["prediction_pending"]:
        lines += ["已冻结配置中仍有开发集输出未完成：" + "、".join(x["tag"] for x in report["prediction_pending"]) + "。", ""]
    for group in report["groups"]:
        analyses = [(f"seed {s['seed']}", s["diagnostics"]) for s in group["seeds"]]
        if group["ensemble"]:
            analyses.append((f"ensemble ({group['ensemble']['n_seeds']} seeds)", group["ensemble"]["diagnostics"]))
        for name, diagnostics in analyses:
            lines += [f"## {group['configuration']} / {name}", "",
                "| 分组类型 | 分组 | 变量 | 有效 n | J | anchor J | ΔJ | NLL | anchor NLL | 68%覆盖 | 95%覆盖 |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for family, partitions in diagnostics.items():
                for entry in partitions:
                    for channel in R.CHANNELS:
                        score, base = entry["scores"][channel], entry["baseline_scores"][channel]
                        unc, base_unc = entry["uncertainty"][channel], entry["calibrated_baseline_uncertainty"][channel]
                        metrics = (score["J"], base["J"], entry["delta_J"][channel], unc["nll"], base_unc["nll"], unc["coverage_68"], unc["coverage_95"])
                        lines.append(f"| {family} | {entry['group']} | {channel} | {score['n']} | " + " | ".join(R.format_value(v) for v in metrics) + " |")
            lines.append("")
    lines += ["各组剖面数、评分格点数、anchor 覆盖率、sigma 校准和预测指纹保存在同名 JSON；没有有效目标的组保留 n=0，其指标为 null。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/latent_ocean/campaign_20261007.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/latent_ocean")
    parser.add_argument("--raw-cohort", type=Path, default=ROOT / "data/argo_cohort/global_global.nc")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/real_data/latent_generalization_20261007.md")
    parser.add_argument("--json", type=Path, default=ROOT / "outputs/latent_ocean/generalization_20261007.json")
    args = parser.parse_args()
    report = build_report(args.manifest, args.output_root, args.raw_cohort)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(report))
    args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.report), "json": str(args.json),
        "selected": report["selected_tags"], "search_pending": len(report["search_pending"]),
        "prediction_pending": len(report["prediction_pending"])}))


if __name__ == "__main__":
    main()
