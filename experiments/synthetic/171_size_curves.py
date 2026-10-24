"""Training curves of the size study (``169_size_campaign.py``): is each size converged at 200k steps?

Reads ``train.log`` of every run under ``outputs/size_20261023`` (``y100_x<factor>_s<seed>``) with
``97_forecast_curves.parse`` and draws ``reports/synthetic/size_20261023_curves.png``: one column per
size; the training MSE (log scale) on top, the mean of every step since the previous record; the
2004 validation score below (mean standardised RMSE over leads, the checkpoint criterion); one line
per seed in the size's colour; each seed's best step a ringed dot; 180k, the start of the last
tenth, a dotted line; the y-axes fit the steps after the first tenth, so the tail is legible. It prints, per run, the last step logged, the change of the training MSE over
the last tenth against the tenth before (``97_forecast_curves.tail_change``), and the best
validation step and score; a best step inside the last tenth means the run was still improving.
Run it at any time; unfinished runs are drawn as far as they have got.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/size_20261023"
FIGURE = ROOT / "reports/synthetic/size_20261023_curves.png"
STEPS = 200000
RUN = re.compile(r"^(y100_x\d+)_s(\d+)$")
#: size -> (title, colour): an ordered tier, so one blue ramp light -> dark (validate_palette --ordinal passes)
SIZES = {"y100_x1": ("x1 · 9.4M", "#6da7ec"), "y100_x1375": ("x1.375 · 17.7M", "#2a78d6"),
         "y100_x175": ("x1.75 · 28.6M", "#184f95")}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/synthetic" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


curves = _script("97_forecast_curves")


def runs(output=OUTPUT):
    found = []
    for folder in sorted(output.iterdir()) if output.exists() else []:
        match = RUN.match(folder.name)
        if match and match[1] in SIZES and (folder / "train.log").exists():
            found.append({"tag": folder.name, "arm": match[1], "seed": int(match[2]), "log": folder / "train.log"})
    # by size, then seed: by name, "y100_x175" sorts before "y100_x1_"
    return sorted(found, key=lambda run: (list(SIZES).index(run["arm"]), run["seed"]))


def rows(found):
    out = []
    for run in found:
        train, validation = curves.parse(run["log"])
        best = min(validation, key=validation.get) if validation else None
        out.append({"tag": run["tag"], "arm": run["arm"], "seed": run["seed"], "last_step": max(validation, default=0),
                    "tail_change_pct": curves.tail_change(train, STEPS), "best_step": best,
                    "best_score": validation.get(best), "still_improving": best is not None and best > STEPS - STEPS // 10})
    return out


def plot(found, path=FIGURE):
    fig, axes = plt.subplots(2, len(SIZES), figsize=(12, 6.4), sharex=True, sharey="row", facecolor=SURFACE)
    tails = ([], [])          # values after the first tenth, which set the y-axes
    for column, (arm, (title, colour)) in enumerate(SIZES.items()):
        top, bottom = axes[0, column], axes[1, column]
        for run in (r for r in found if r["arm"] == arm):
            train, validation = curves.parse(run["log"])
            for row, (ax, values) in enumerate(((top, train), (bottom, validation))):
                steps = sorted(values)
                ax.plot(np.array(steps) / 1000, [values[s] for s in steps], color=colour, lw=1.0)   # 2 px at 150 dpi
                tails[row].extend(values[s] for s in steps if s >= STEPS // 10)
            if validation:
                best = min(validation, key=validation.get)
                bottom.plot(best / 1000, validation[best], "o", ms=5, color=colour, mec=SURFACE, mew=1.0, zorder=3)
        top.set_title(title, color=INK, fontsize=11)
        top.set_yscale("log")
        for ax in (top, bottom):
            ax.set_facecolor(SURFACE)
            ax.axvline((STEPS - STEPS // 10) / 1000, color=INK_2, lw=.8, ls=":")
            ax.grid(color=GRID, lw=.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=INK_2, labelsize=9)
        bottom.set_xlabel("step (thousands)", color=INK_2)
        bottom.set_xlim(0, STEPS / 1000)
    # the fast drop of the first tenth would flatten the tail, where convergence is judged
    if tails[0]:
        axes[0, 0].set_ylim(min(tails[0]) / 1.04, max(tails[0]) * 1.04)
        for ax in axes[0]:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
    if tails[1]:
        pad = .08 * (max(tails[1]) - min(tails[1])) or .01
        axes[1, 0].set_ylim(min(tails[1]) - pad, max(tails[1]) + pad)
    if any(tails):
        fig.text(.5, .008, "y-axes fit the steps after 20k; the initial drop runs off the top", color=INK_2, fontsize=8, ha="center")
    axes[0, 0].set_ylabel("training MSE (log)", color=INK_2)
    axes[1, 0].set_ylabel("2004 validation score\n(mean standardised RMSE over leads)", color=INK_2)
    handles = [Line2D([], [], color=c, lw=1.0, label=t) for t, c in SIZES.values()]
    handles += [Line2D([], [], ls="", marker="o", ms=5, color=INK_2, mec=SURFACE, label="best validation step"),
                Line2D([], [], color=INK_2, lw=.8, ls=":", label="180k: start of the last tenth")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .03), ncol=5, frameon=False, labelcolor=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    found = runs()
    for r in rows(found):
        tail = "running" if r["tail_change_pct"] is None else f"training MSE {r['tail_change_pct']:+.1f} % over the last tenth"
        best = "no validation yet" if r["best_step"] is None else f"best {r['best_score']:.4f} at {r['best_step']}"
        flag = " (still improving)" if r["still_improving"] and r["last_step"] >= STEPS else ""
        print(f"{r['tag']:20} step {r['last_step']:>6}  {tail:42} {best}{flag}", flush=True)
    plot(found)
    print(FIGURE.relative_to(ROOT))


if __name__ == "__main__":
    main()
