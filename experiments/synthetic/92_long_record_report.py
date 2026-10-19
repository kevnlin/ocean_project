"""Tables of the training-window x model-size grid (``90_long_record_campaign.py``).

Reads every finished ``summary_seed*.json`` under ``outputs/long_record_20261017``
and writes ``reports/synthetic/long_record_20261017.md`` and ``.json``:

  test        2005 RMSE (temperature degC, salinity PSU), mean and standard
              deviation over seeds, from the checkpoint selected on 2004; the
              frozen OI anchor of the same cells for reference
  gain        the change of that RMSE against the 4-year window of the same
              size, and against OI
  speed       training steps a run needs to reach the 2004 score (mean
              per-variable standardized RMSE, the checkpoint criterion) that
              the 4-year, smallest model ends with; the 2004 score after 1k,
              3k, 5k, 10k and 15k steps
  cost        parameters, wall-clock hours, peak GPU memory

The 30,000-step follow-up (runs ``k30_y<years>_<size>_s<seed>``, 4- and
40-year windows) gets its own section: its 2005 RMSE, the change against the
15,000-step run of the same cell, and its steps to the same 2004 target.

Runs still training are left out; a cell with fewer seeds than requested says
so. Run it at any time.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/long_record_20261017"
REPORT = ROOT / "reports/synthetic/long_record_20261017"
CH = ("TEMP", "SALT")
UNITS = {"TEMP": "°C", "SALT": "PSU"}
YEARS = (4, 10, 20, 40)
SIZES = {"s": "64 slots, width 192", "m": "128 slots, width 256", "l": "256 slots, width 384"}
SEEDS = 3
CURVE_STEPS = (1000, 3000, 5000, 10000, 15000)
LONG_CURVE_STEPS = (5000, 10000, 15000, 20000, 25000, 30000)
NAME = re.compile(r"^(?:k(\d+)_)?y(\d+)_([sml])_s(\d+)$")


def runs(output=OUTPUT, steps=15000):
    """Finished runs of one step budget, by (training years, size)."""
    found = {}
    for path in sorted(output.glob("*y*_*_s*/summary_seed*.json")):
        match = NAME.match(path.parent.name)
        if match is None or (int(match[1]) * 1000 if match[1] else 15000) != steps:
            continue
        summary = json.loads(path.read_text())
        if "development" not in summary or summary.get("completed_steps") != steps:
            continue
        found.setdefault((int(match[2]), match[3]), []).append(summary)
    return found


def rmse(summary, split, ch, key="scores"):
    return summary[split][key][ch]["rmse_physical"]


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else None


def reach(history, target):
    """First step at which the 2004 score is at most ``target``, linearly interpolated."""
    previous = None
    for record in history:
        step, score = record["step"], record["validation_mean_standardized_rmse"]
        if score <= target:
            if previous is None:
                return float(step)
            (s0, v0) = previous
            return float(s0 + (step - s0) * (v0 - target) / (v0 - score)) if v0 > target else float(step)
        previous = (step, score)
    return None


def curve(history, steps=CURVE_STEPS):
    at = {r["step"]: r["validation_mean_standardized_rmse"] for r in history}
    return {step: at.get(step) for step in steps}


def collect(found, target=None, curve_steps=CURVE_STEPS):
    """Per-cell statistics; ``target`` defaults to the 4-year small model's final 2004 score."""
    cells, oi = {}, {}
    reference = found.get((4, "s"), [])
    if target is None and reference:
        target = float(np.mean([s["validation"]["scores"]["macro_z"] for s in reference]))
    for (years, size), summaries in sorted(found.items()):
        cell = {"seeds": [s["seed"] for s in summaries], "params": summaries[0]["params"]}
        for ch in CH:
            cell[f"test_{ch}"] = mean_sd([rmse(s, "development", ch) for s in summaries])
            cell[f"validation_{ch}"] = mean_sd([rmse(s, "validation", ch) for s in summaries])
            oi.setdefault(ch, rmse(summaries[0], "development", ch, "baseline_scores"))
        cell["validation_score"] = mean_sd([s["validation"]["scores"]["macro_z"] for s in summaries])
        cell["best_step"] = [s["best_step"] for s in summaries]
        reached = [reach(s["history"], target) if target is not None else None for s in summaries]
        cell["steps_to_target"] = reached
        cell["curve"] = {step: mean_sd([v for v in values if v is not None]) if any(v is not None for v in values) else None
                         for step, values in zip(curve_steps, zip(*[curve(s["history"], curve_steps).values()
                                                                   for s in summaries]))}
        cell["hours"] = mean_sd([s["runtime_s"] / 3600 for s in summaries])
        cell["peak_gpu_gb"] = max((s.get("peak_gpu_memory_mb") or 0) / 1024 for s in summaries)
        cells[(years, size)] = cell
    return cells, oi, target


def fmt(stat, digits=4):
    if stat is None:
        return "–"
    mean, sd = stat
    return f"{mean:.{digits}f}" + (f" ± {sd:.{digits}f}" if sd is not None else "")


def change(new, old):
    return f"{100 * (new / old - 1):+.1f} %"


def markdown(cells, oi, target):
    sizes = [s for s in SIZES if any((y, s) in cells for y in YEARS)]
    lines = ["# Training window × model size, CESM2, Argo + surface inputs", "",
             "Zihao's shared T/S alignment model (`profile_shared`, Local Transformer, 6 blocks, 8 heads) "
             "with Argo plus SST, SSS and steric SSH, the local OI-error scale by depth × 5° cell, "
             "15,000 steps of 1,024 queries, checkpoint chosen on 2004, scored on 2005. Training windows end "
             "in 2003; the normalisation, climatology, OI anchor and scale table are the 2000–2003 fits for every "
             "window, so only the number of training months changes (48, 120, 240 or 480). Steric SSH is "
             "computed from the same T/S that is reconstructed and SST/SSS are its top level, without noise: "
             "an optimistic stand-in for satellite data.", "",
             "## 2005 test RMSE", "",
             "Mean ± standard deviation over seeds; temperature °C / salinity PSU. Frozen OI on the same cells: "
             f"{oi.get('TEMP', float('nan')):.4f} °C / {oi.get('SALT', float('nan')):.4f} PSU.", "",
             "| Training window | " + " | ".join(f"{s}: {SIZES[s]}" for s in sizes) + " |",
             "|---|" + "---|" * len(sizes)]
    for years in YEARS:
        row = []
        for size in sizes:
            cell = cells.get((years, size))
            if cell is None:
                row.append("–")
                continue
            seeds = "" if len(cell["seeds"]) == SEEDS else f" ({len(cell['seeds'])} seed{'s' * (len(cell['seeds']) > 1)})"
            row.append(f"{fmt(cell['test_TEMP'])} / {fmt(cell['test_SALT'])}{seeds}")
        lines.append(f"| {years} years | " + " | ".join(row) + " |")
    lines += ["", "## Change of the 2005 RMSE", "",
              "Against the 4-year window of the same size, then against OI; temperature / salinity.", "",
              "| Training window | " + " | ".join(f"{s}" for s in sizes) + " |", "|---|" + "---|" * len(sizes)]
    for years in YEARS:
        row = []
        for size in sizes:
            cell, base = cells.get((years, size)), cells.get((4, size))
            if cell is None:
                row.append("–")
                continue
            vs_oi = " / ".join(change(cell[f"test_{ch}"][0], oi[ch]) for ch in CH)
            vs_four = (" / ".join(change(cell[f"test_{ch}"][0], base[f"test_{ch}"][0]) for ch in CH)
                       if base is not None and years != 4 else "–")
            row.append(f"{vs_four}; vs OI {vs_oi}")
        lines.append(f"| {years} years | " + " | ".join(row) + " |")
    lines += ["", "## Training speed", "",
              ("Steps to reach the 2004 score the 4-year, smallest model ends with "
               f"({target:.4f}, mean per-variable standardized RMSE), per seed; – if not reached in 15,000 steps."
               if target is not None else "The 4-year, smallest model has not finished yet."), "",
              "| Training window | " + " | ".join(sizes) + " |", "|---|" + "---|" * len(sizes)]
    for years in YEARS:
        row = []
        for size in sizes:
            cell = cells.get((years, size))
            row.append("–" if cell is None else ", ".join("–" if s is None else f"{s / 1000:.1f}k"
                                                          for s in cell["steps_to_target"]))
        lines.append(f"| {years} years | " + " | ".join(row) + " |")
    lines += ["", "2004 score (mean per-variable standardized RMSE) after a number of steps, mean over seeds:", "",
              "| Window, size | " + " | ".join(f"{s // 1000}k" for s in CURVE_STEPS) + " | best step |",
              "|---|" + "---|" * (len(CURVE_STEPS) + 1)]
    for (years, size), cell in sorted(cells.items(), key=lambda kv: (kv[0][0], list(SIZES).index(kv[0][1]))):
        values = " | ".join("–" if cell["curve"][s] is None else f"{cell['curve'][s][0]:.4f}" for s in CURVE_STEPS)
        lines.append(f"| {years} y, {size} | {values} | {', '.join(str(b) for b in cell['best_step'])} |")
    lines += ["", "## Cost", "", "| Window, size | parameters | hours a run | peak GPU memory |", "|---|---|---|---|"]
    for (years, size), cell in sorted(cells.items(), key=lambda kv: (kv[0][0], list(SIZES).index(kv[0][1]))):
        lines.append(f"| {years} y, {size} | {cell['params'] / 1e6:.2f} M | {fmt(cell['hours'], 1)} | "
                     f"{cell['peak_gpu_gb']:.1f} GB |")
    return "\n".join(lines) + "\n"


def markdown_long(long, short, oi, target):
    """The 30,000-step follow-up against the 15,000-step cells."""
    sizes = [s for s in SIZES if any((y, s) in long for y in (4, 40))]
    lines = ["", "## 30,000 steps", "",
             "The 4- and 40-year windows trained twice as long (the learning-rate schedule spans 30,000 steps); "
             "everything else as above. 2005 RMSE, temperature °C / salinity PSU, mean ± sd over seeds; "
             "then the change against the 15,000-step run of the same cell.", "",
             "| Training window | " + " | ".join(f"{s}: {SIZES[s]}" for s in sizes) + " |", "|---|" + "---|" * len(sizes)]
    for years in (4, 40):
        row = []
        for size in sizes:
            cell = long.get((years, size))
            if cell is None:
                row.append("–")
                continue
            seeds = "" if len(cell["seeds"]) == SEEDS else f" ({len(cell['seeds'])} seed{'s' * (len(cell['seeds']) > 1)})"
            row.append(f"{fmt(cell['test_TEMP'])} / {fmt(cell['test_SALT'])}{seeds}")
        lines.append(f"| {years} years | " + " | ".join(row) + " |")
    lines += ["", "| Change against 15,000 steps | " + " | ".join(sizes) + " |", "|---|" + "---|" * len(sizes)]
    for years in (4, 40):
        row = []
        for size in sizes:
            cell, base = long.get((years, size)), short.get((years, size))
            row.append("–" if cell is None or base is None else
                       " / ".join(change(cell[f"test_{ch}"][0], base[f"test_{ch}"][0]) for ch in CH))
        lines.append(f"| {years} years | " + " | ".join(row) + " |")
    row = []
    for size in sizes:
        four, forty = long.get((4, size)), long.get((40, size))
        row.append("–" if four is None or forty is None else
                   " / ".join(change(forty[f"test_{ch}"][0], four[f"test_{ch}"][0]) for ch in CH))
    lines += ["", "40 years against 4 years at 30,000 steps: " + "; ".join(f"{s} {r}" for s, r in zip(sizes, row)), "",
              f"Steps to the 2004 target of the 15,000-step section ({target:.4f}), per seed; best step:" if target else
              "The 15,000-step target is not available yet.", "",
              "| Window, size | steps to target | " + " | ".join(f"{s // 1000}k" for s in LONG_CURVE_STEPS) + " | best step |",
              "|---|---|" + "---|" * (len(LONG_CURVE_STEPS) + 1)]
    for (years, size), cell in sorted(long.items(), key=lambda kv: (kv[0][0], list(SIZES).index(kv[0][1]))):
        reached = ", ".join("–" if s is None else f"{s / 1000:.1f}k" for s in cell["steps_to_target"])
        values = " | ".join("–" if cell["curve"][s] is None else f"{cell['curve'][s][0]:.4f}" for s in LONG_CURVE_STEPS)
        lines.append(f"| {years} y, {size} | {reached} | {values} | {', '.join(str(b) for b in cell['best_step'])} |")
    return "\n".join(lines) + "\n"


def main():
    found = runs()
    cells, oi, target = collect(found)
    found_long = runs(steps=30000)
    cells_long, oi_long, _ = collect(found_long, target, LONG_CURVE_STEPS)
    text = markdown(cells, oi or oi_long, target)
    if cells_long:
        text += markdown_long(cells_long, cells, oi or oi_long, target)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.with_suffix(".md").write_text(text)
    REPORT.with_suffix(".json").write_text(json.dumps(
        {"oi_test": oi or oi_long, "speed_target_validation_score": target,
         "cells": {f"y{y}_{s}": cell for (y, s), cell in cells.items()},
         "cells_30000_steps": {f"y{y}_{s}": cell for (y, s), cell in cells_long.items()}}, indent=2) + "\n")
    print(f"{sum(len(v) for v in found.values())} finished 15k runs in {len(cells)} cells, "
          f"{sum(len(v) for v in found_long.values())} finished 30k runs -> {REPORT}.md", flush=True)


if __name__ == "__main__":
    main()
