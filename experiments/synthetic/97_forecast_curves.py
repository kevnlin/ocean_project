"""Training curves of the forecast runs (``95_forecast_campaign.py``): is training converging?

Reads ``train.log`` of every run under ``outputs/forecast_20261019`` (``f_y<years>_s<seed>`` at
70,000 steps, ``k<thousands>_f_y<years>_s<seed>`` at other lengths) and draws
``outputs/forecast_20261019/training_curves.png``, the 4- and 40-year windows in two columns:

  top     the training MSE, as the runner logs it with every validation: the mean loss of every
          step of the 2,000 steps since the last one (log scale)
  bottom  the 2004-2005 validation score logged every 2,000 steps: the mean over leads of the
          per-variable standardized RMSE, the criterion that chooses the checkpoint; each run's
          best point is marked

One line per seed, coloured by training length. It also prints, for every run, the change of the
mean training MSE over the last tenth of training against the tenth before, or how far a run
still training has got, and the step of its best validation score. A run that crashed between a
validation and the next checkpoint validates that step again after resuming; the last record
counts. Run it at any time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/forecast_20261019"
STEPS = 70000
RUN = re.compile(r"^(?:k(\d+)_)?f_y(\d+)_s(\d+)$")
#: line colour by training length; the earlier 70,000-step runs recede in grey
COLOURS = {70000: "#898781", 100000: "#2a78d6"}


def runs(output=OUTPUT):
    found = []
    for folder in sorted(output.iterdir()):
        match = RUN.match(folder.name)
        if match and (folder / "train.log").exists():
            found.append({"tag": folder.name, "steps": int(match[1]) * 1000 if match[1] else STEPS,
                          "years": int(match[2]), "seed": int(match[3]), "log": folder / "train.log"})
    return found


def parse(path):
    """{step: training MSE since the last validation} and {step: validation score}; a step
    validated twice keeps its last record."""
    train, validation = {}, {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("validation {"):
            record = json.loads(line[len("validation "):].rstrip(" *"))
            train[record["step"]] = record["training_mse"]
            validation[record["step"]] = record["validation_mean_standardized_rmse"]
    return train, validation


def tail_change(train, steps):
    """Change in % of the mean training MSE over the last tenth of ``steps`` against the tenth
    before it; None while the run has not reached ``steps``."""
    if max(train, default=0) < steps:
        return None
    tenth = steps // 10
    last = [v for s, v in train.items() if steps - tenth < s <= steps]
    before = [v for s, v in train.items() if steps - 2 * tenth < s <= steps - tenth]
    return 100 * (np.mean(last) / np.mean(before) - 1)


def plot(found, path):
    windows = sorted({run["years"] for run in found})
    fig, axes = plt.subplots(2, len(windows), figsize=(6.5 * len(windows), 7.5), sharex="col", squeeze=False)
    for column, years in enumerate(windows):
        top, bottom = axes[0, column], axes[1, column]
        for run in (r for r in found if r["years"] == years):
            train, validation = parse(run["log"])
            colour, label = COLOURS.get(run["steps"], "#eb6834"), f"{run['steps']:,} steps"
            if validation:
                steps = sorted(validation)
                top.plot(steps, [train[s] for s in steps], color=colour, lw=1.5, label=label)
                bottom.plot(steps, [validation[s] for s in steps], color=colour, lw=1.5, label=label)
                best = min(steps, key=validation.get)
                bottom.plot(best, validation[best], "o", color=colour, ms=6, mec="white", mew=1.5)
        top.set_yscale("log")
        for axis in (top.yaxis.set_major_formatter, top.yaxis.set_minor_formatter):
            axis(FormatStrFormatter("%g"))
        top.set_title(f"{years}-year window")
        top.set_ylabel("training MSE, mean over 2,000 steps")
        bottom.set_ylabel("2004–05 validation score\n(mean standardized RMSE over leads)")
        bottom.set_xlabel("training step")
        for ax in (top, bottom):
            ax.grid(color="#e1e0d9", lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))  # one entry per length, not per seed
            ax.legend(unique.values(), unique.keys(), frameon=False)
    fig.suptitle("Forecast runs: one line per seed; dots mark each run's best validation score")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT, help="the campaign folder")
    output = parser.parse_args().output
    found = runs(output)
    if not found:
        raise SystemExit(f"no run with a train.log under {output}")
    for run in found:
        train, validation = parse(run["log"])
        change = tail_change(train, run["steps"])
        state = (f"last tenth {change:+.1f} %" if change is not None
                 else f"training, step {max(train, default=0):,} of {run['steps']:,}")
        best = min(validation, key=validation.get) if validation else None
        print(f"{run['tag']:18} {state:34} best validation step {best}", flush=True)
    path = output / "training_curves.png"
    plot(found, path)
    print(f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
