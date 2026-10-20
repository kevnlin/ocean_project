"""The figures of the paper draft, from saved results only (nothing is recomputed on test data).

  kernels.pdf   (a) learned query kernels over the SSH anomaly (CESM2, Gulf Stream, selection year),
                (b) median kernel anisotropy by latitude, (c) kernel tilt against SSH-isoline tilt
                where the SSH gradient is strongest; CESM2 and real data (117's saved cells)
  depth.pdf     sealed-test temperature RMSE by depth at leads 0 and 6 on both tests (109/113/121)

Colours: the dataviz reference categorical palette in fixed order, one colour per method in every
panel; reference lines in muted ink.

  .venv/bin/python experiments/synthetic/123_paper_figures.py PAPER_FOLDER
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("align117", ROOT / "experiments/synthetic/117_kernel_alignment.py")
align = importlib.util.module_from_spec(spec)
spec.loader.exec_module(align)
spec = importlib.util.spec_from_file_location("tables118", ROOT / "experiments/synthetic/118_validation_tables.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

FIGURES = ROOT / "outputs/figures_20261020"
CELLS = {"CESM2": FIGURES / "kernel_alignment_grid_ka_70k_20261019__gj_flow_s1234.npz",
         "Real": FIGURES / "kernel_alignment_real_global_20261019__scratch_s1234.npz"}
#: dataviz reference categorical palette (light mode), fixed order
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DATASET_COLOUR = {"CESM2": SLOT[0], "Real": SLOT[1]}
METHODS = [  # name, colour, CESM2 run, real run (None: not run there)
    ("KA-Ocean (ours)", SLOT[0], (T.PILOT, "gj_flow"), (T.REAL, "scratch")),
    ("ARMOR3D-style OI", SLOT[1], (T.FORMAL, "c_gp_oi"), (T.FORMAL, "r_gp_oi")),
    ("Aardvark-style", SLOT[2], (T.PILOT, "gj_none"), (T.FORMAL, "r_none")),
    ("Poseidon-T", SLOT[3], (T.POSEIDON, "p_none"), (T.SELECTION, "rp_none")),
    ("OI-anchored transformer", SLOT[4], (T.REFERENCE, "f_y40"), None)]
TESTS = {"CESM2": ROOT / "outputs/cesm2_test_20261019", "Real": ROOT / "outputs/real_global_20261019/sealed"}


def style(plt):
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": .6,
                         "lines.linewidth": 1.4, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})


def kernels(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    style(plt)
    cells = {name: dict(np.load(path, allow_pickle=False)) for name, path in CELLS.items()}
    fig = plt.figure(figsize=(7.0, 2.15))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=.55)
    # (a) map
    ax = fig.add_subplot(grid[0])
    c = cells["CESM2"]
    month, level = int(c["ssh_months"][0]), 0
    ssh = c["ssh_grids"][0].astype(np.float32)
    (la0, la1), (lo0, lo1) = (28., 46.), (282., 318.)
    rows, columns = (c["grid_lat"] >= la0) & (c["grid_lat"] <= la1), (c["grid_lon"] >= lo0) & (c["grid_lon"] <= lo1)
    mesh = ax.pcolormesh(c["grid_lon"][columns] - 360, c["grid_lat"][rows], ssh[np.ix_(rows, columns)], cmap="RdBu_r",
                         vmin=-3, vmax=3, shading="auto", rasterized=True)
    theta, major, minor = align.axes(c["sigma"][:, 0])
    lon = c["lon"] % 360.
    keep = np.flatnonzero((c["month"] == month) & (c["level"] == level) & (c["lat"] >= la0 + 1) & (c["lat"] <= la1 - 1)
                          & (lon >= lo0 + 1) & (lon <= lo1 - 1))
    for i in np.random.default_rng(1).permutation(keep)[:45]:
        scale = .35 / align.KM_PER_DEGREE
        ax.add_patch(Ellipse((lon[i] - 360, c["lat"][i]), 2 * major[i] * scale / np.cos(np.radians(c["lat"][i])),
                             2 * minor[i] * scale, angle=np.degrees(theta[i]), fill=False, lw=.7, color=INK))
    ax.set_xlim(lo0 - 360, lo1 - 360)
    ax.set_ylim(la0, la1)
    ax.set_aspect(1 / np.cos(np.radians((la0 + la1) / 2)))
    ax.set_xlabel("longitude ($^\\circ$E)")
    ax.set_ylabel("latitude ($^\\circ$N)")
    ax.set_title("(a) learned kernels, CESM2, 5 m", loc="left")
    bar = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=.05, pad=.28, aspect=30)
    bar.set_label("SSH anomaly (z)", fontsize=6.5)
    bar.ax.tick_params(labelsize=6)
    # (b) anisotropy by latitude
    ax = fig.add_subplot(grid[1])
    for name, c in cells.items():
        theta, major, minor = align.axes(c["sigma"][:, 0])
        centres, values = [], []
        for lo in range(-70, 70, 10):
            m = (c["lat"] >= lo) & (c["lat"] < lo + 10)
            if m.sum() > 500:
                centres.append(lo + 5)
                values.append(np.median(major[m] / np.maximum(minor[m], 1e-9)))
        ax.plot(centres, values, "-o", ms=3, color=DATASET_COLOUR[name], label=name)
        ax.annotate(name, (centres[-1], values[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=6.5, color=INK, va="center")
    ax.axhline(1, color=MUTED, lw=.6, ls=":")
    ax.set_xlim(-75, 88)
    ax.set_xlabel("latitude ($^\\circ$N)")
    ax.set_ylabel("median anisotropy (long / short axis)")
    ax.set_title("(b) zonal stretching by latitude", loc="left")
    ax.grid(color=GRID, lw=.5)
    ax.legend(frameon=False, loc="upper left")
    # (c) tilt relation
    ax = fig.add_subplot(grid[2])
    for name, c in cells.items():
        if name != "CESM2":                                 # the resolved, noise-free SSH of the simulation
            continue
        theta, _, _ = align.axes(c["sigma"][:, 0])
        g = c["gradient"]
        strength = np.hypot(g[:, 0], g[:, 1])
        finite = np.isfinite(strength)
        strong = finite & (strength >= np.quantile(strength[finite], .75))
        phi = np.arctan2(g[:, 0], -g[:, 1])
        x, y = np.sin(2 * phi[strong]), np.sin(2 * theta[strong])
        edges = np.linspace(-1, 1, 11)
        which = np.clip(np.digitize(x, edges) - 1, 0, 9)
        means = [y[which == b].mean() for b in range(10)]
        r = align.tilt_correlation(theta[strong], g[strong])
        ax.plot((edges[:-1] + edges[1:]) / 2, means, "-o", ms=3, color=DATASET_COLOUR[name], label=f"{name} (r = {r:.2f})")
    ax.plot([-1, 1], [-1, 1], ls=":", color=MUTED, lw=.7)
    ax.annotate("kernel turns fully\nwith the flow", (.55, .62), fontsize=6, color=MUTED)
    ax.axhline(0, color=GRID, lw=.6)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("SSH-isoline tilt  sin 2$\\varphi$")
    ax.set_ylabel("mean kernel tilt  sin 2$\\theta$")
    ax.set_title("(c) turning with the flow, CESM2", loc="left")
    ax.grid(color=GRID, lw=.5)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out / "kernels.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def depth(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    style(plt)
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1), sharey=True)
    for k, (test, lead) in enumerate((("CESM2", "0"), ("CESM2", "6"), ("Real", "0"), ("Real", "6"))):
        ax = axes[k]
        results = json.loads((TESTS[test] / "results.json").read_text())
        by_path = {e["checkpoint"]: e["by_lead"] for e in results["entries"]}
        levels = np.asarray(json.loads(next((T.PILOT / "gj_flow_s1234").glob("summary_seed*.json")).read_text())["data"]["levels_m"])
        first = results["entries"][0]["by_lead"][lead]["all"]["TEMP"]
        ax.plot(first["persisted_oi"]["by_level"], levels, ls="--", color=MUTED, lw=1, label="persisted OI")
        stat = json.loads((TESTS[test] / "results_statistical.json").read_text())["by_lead"][lead]["all"]["TEMP"]["model"]
        ax.plot(stat["by_level"], levels, ls=":", color=MUTED, lw=1.2, label="damped persistence")
        for name, colour, cesm2, real in METHODS:
            where = cesm2 if test == "CESM2" else real
            if where is None:
                continue
            folder, tag = where
            runs = [by_path[str(folder / f"{tag}_s{s}" / f"best_seed{s}.pt")][lead]["all"]["TEMP"]["model"]["by_level"]
                    for s in T.SEEDS]
            ax.plot(np.mean(runs, 0), levels, color=colour, lw=1.6 if "ours" in name else 1.1, label=name,
                    zorder=3 if "ours" in name else 2)
        ax.set_yscale("log")
        ax.set_ylim(1000, 4.5)
        ax.yaxis.set_major_locator(ticker.FixedLocator([5, 10, 25, 50, 100, 250, 500, 1000]))
        ax.yaxis.set_major_formatter(ticker.FixedFormatter(["5", "10", "25", "50", "100", "250", "500", "1000"]))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.set_title(f"{test}, lead {lead}", loc="left")
        ax.set_xlabel("T RMSE ($^\\circ$C)")
        ax.grid(color=GRID, lw=.5)
        if k == 0:
            ax.set_ylabel("depth (m)")
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.02))
    fig.subplots_adjust(wspace=.12)
    fig.savefig(out / "depth.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def sparsity(out):
    """Reconstruction error on the real selection year when only a fraction of the input floats is kept (124)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict
    style(plt)
    results = json.loads((ROOT / "outputs/sparsity_20261020/real.json").read_text())
    colours = {"KA-Ocean": SLOT[0], "ARMOR3D-style OI": SLOT[1], "Aardvark-style": SLOT[2]}
    fractions = ["1.0", "0.5", "0.25", "0.125"]
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.75))
    for ax, (ch, unit) in zip(axes, (("TEMP", "T RMSE ($^\\circ$C)"), ("SALT", "S RMSE (PSU)"))):
        model, oi = defaultdict(list), defaultdict(list)
        for key, res in results.items():
            name = key.split("|")[0]
            for f in fractions:
                model[(name, f)].append(res[f]["0"][ch]["model"])
                oi[f].append(res[f]["0"][ch]["oi"])
        x = [float(f) for f in fractions]
        ax.plot(x, [np.mean(oi[f]) for f in fractions], ls="--", color=MUTED, lw=1, label="OI")
        for name, colour in colours.items():
            ax.plot(x, [np.mean(model[(name, f)]) for f in fractions], "-o", ms=2.5, color=colour,
                    lw=1.6 if name == "KA-Ocean" else 1.1, label=name)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels(["1", "1/2", "1/4", "1/8"])
        ax.invert_xaxis()
        ax.set_xlabel("fraction of input floats")
        ax.set_ylabel(unit)
        ax.grid(color=GRID, lw=.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, 1.1), fontsize=6)
    fig.subplots_adjust(wspace=.55)
    fig.savefig(out / "sparsity.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper", type=Path)
    args = parser.parse_args()
    out = args.paper / "fig"
    out.mkdir(parents=True, exist_ok=True)
    kernels(out)
    depth(out)
    if (ROOT / "outputs/sparsity_20261020/real.json").exists():
        sparsity(out)
    print(f"wrote {out / 'kernels.pdf'} and {out / 'depth.pdf'}")


if __name__ == "__main__":
    main()
