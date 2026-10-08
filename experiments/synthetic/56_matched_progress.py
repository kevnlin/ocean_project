"""Live validation-only progress; never read new-model 2005 predictions."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/synthetic_matched_20261007"
LABELS = {"previous_token64": "旧 64-slot 模型", "dense64": "Dense Transformer 64",
          "dense192": "Dense Transformer 192", "soft_moe192": "Soft MoE 192",
          "local_transformer192": "Local Transformer 192",
          "soft_moe192_latent_off": "Soft MoE，关闭 latent",
          "soft_moe192_local_off": "Soft MoE，关闭新增局部通路",
          "official4dvarnet": "作者代码 4DVarNet 适配"}


def progress(job):
    folder = Path(job["output"])
    summary_path = folder / f"summary_seed{job['seed']}.json"
    row = {"tag": job["tag"], "family": job["family"], "surface": job["surface"],
           "seed": job["seed"], "completed_steps": 0, "complete": False,
           "validation_temperature_rmse_C": None, "validation_salinity_rmse_PSU": None}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        row["completed_steps"] = summary.get("completed_steps", max(
            (h["step"] for h in summary.get("history", [])), default=0))
        row["complete"] = row["completed_steps"] == 15000
        scores = summary.get("validation", {}).get("scores", summary.get("scores", {}).get("validation", {}))
        physical_metrics = summary.get("validation", {}).get("physical_metrics", {})
        for channel, field in (("TEMP", "validation_temperature_rmse_C"), ("SALT", "validation_salinity_rmse_PSU")):
            row[field] = scores.get(channel, {}).get("rmse_physical")
            if row[field] is None:
                row[field] = physical_metrics.get(channel, {}).get("rmse")
        row["best_step"] = summary.get("best_step")
        return row
    history = folder / "history.jsonl"
    if history.exists():
        best = float("inf")
        for line in history.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # live writers may leave the final line incomplete
            row["completed_steps"] = max(row["completed_steps"], record.get("step", 0))
            criterion = record.get("validation/macro_z")
            if criterion is not None and criterion < best:
                best = criterion
                row["best_step"] = record["step"]
                row["validation_temperature_rmse_C"] = record.get("validation/TEMP_rmse_physical")
                row["validation_salinity_rmse_PSU"] = record.get("validation/SALT_rmse_physical")
    log = folder / "train.log"
    if log.exists():
        best = float("inf")
        for line in log.read_text().splitlines():
            match = re.match(r"step (\d+) train_mse", line)
            if match:
                row["completed_steps"] = max(row["completed_steps"], int(match.group(1)))
            if line.startswith("validation {"):
                try:
                    record = json.loads(line[len("validation "):].removesuffix(" *"))
                except json.JSONDecodeError:
                    continue
                criterion = record["validation_mean_standardized_rmse"]
                if criterion < best:
                    best = criterion
                    row["best_step"] = record["step"]
                    row["validation_temperature_rmse_C"] = record["validation_temperature_rmse_C"]
                    row["validation_salinity_rmse_PSU"] = record["validation_salinity_rmse_PSU"]
    return row


def refresh():
    campaign = json.loads((OUTPUT / "campaign.json").read_text())
    rows = [progress(job) for job in campaign["jobs"]]
    complete = sum(row["complete"] for row in rows)
    state_path = OUTPUT / "status.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    lines = ["# CESM2 同设置比较：运行进度", "",
        f"完整训练完成 **{complete}/{len(rows)}**。每次 15,000 步；每个架构配置使用 1234、1235、1236 三个种子。", "",
        "本文件只读取训练日志和 2004 验证指标。它不是最终测试结果，也不用于提前宣布胜出架构。", "",
        "统一数据：2000–2003 训练，2004 验证，2005 为此前已经使用的测试年；6080 输入剖面/月，20 个原生深度。最终每变量 351895 个有效评分值。", "",
        "| 架构 | 卫星输入 | 种子 | 已记录步数 | 完整训练 | 最好验证温度 RMSE (°C) | 最好验证盐度 RMSE (PSU) |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: |"]
    for row in rows:
        number = lambda value: "—" if value is None else f"{value:.6f}"
        lines.append(f"| {LABELS[row['family']]} | {'SST/SSS/海面高度' if row['surface'] else '无'} | {row['seed']} | {row['completed_steps']}/15000 | {'已完成' if row['complete'] else '待完成'} | {number(row['validation_temperature_rmse_C'])} | {number(row['validation_salinity_rmse_PSU'])} |")
    lines += ["", "固定比较方法包括气候态、最近邻、经典 OI、此前的 learned OI，以及按原配方重新训练的单种子 Pointwise MLP。", "",
        "先完成全部训练，再仅用验证集冻结选择，随后导出 2005 预测并计算 RMSE、MAE、偏差、R²、相关性与不确定性指标。最终报告会单独列出种子均值±标准差和预测集成。", "",
        f"当前运行阶段：`{state.get('phase', 'pending')}`。更新 UTC：`{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}`。"]
    if state.get("failure"):
        lines += ["", f"运行异常：`{state['failure']}`。尚未完成，不能将当前结果作为最终比较。"]
    if (OUTPUT / "metrics.json").exists():
        lines += ["", "最终指标已生成，见同目录 `metrics.json` 和 `matched_reconstruction_20261007.md`。"]
    path = ROOT / "reports/synthetic/matched_progress_20261007.md"
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(path)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60.)
    args = parser.parse_args()
    while True:
        state = refresh()
        if not args.watch or state.get("failure") or (OUTPUT / "metrics.json").exists():
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
