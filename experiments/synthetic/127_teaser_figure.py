"""The teaser figure: softmax attention against Kriging Attention on a field with flow-dependent
covariance (the controlled benchmark of 125), and the errors of every read-out.

  (a) a test field, its observations and one query: the weights softmax attention gives the 32
      nearest observations (positive, summing to at most one)
  (b) the same query under KA: positive and negative weights stretched along the flow, with the
      learned (solid) and true (dashed) query kernels
  (c) RMSE on the 200 test fields: OI, softmax attention, KA stationary, KA flow, exact GP

  .venv/bin/python experiments/synthetic/127_teaser_figure.py PAPER_FOLDER
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gp125", ROOT / "experiments/synthetic/125_synthetic_gp_benchmark.py")
gp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gp)
spec = importlib.util.spec_from_file_location("align117", ROOT / "experiments/synthetic/117_kernel_alignment.py")
align = importlib.util.module_from_spec(spec)
spec.loader.exec_module(align)
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(kind):
    model = gp.ReadOut(kind)
    model.load_state_dict(torch.load(gp.OUTPUT / f"{kind}.pt", map_location="cpu", weights_only=True))
    return model.eval()


def pick_query(tb):
    """A query in a strongly anisotropic place, far from the domain edge."""
    sig = tb["sigma_q"]
    root = np.sqrt(((sig[:, 0] - sig[:, 2]) / 2) ** 2 + sig[:, 1] ** 2)
    ratio = np.sqrt(((sig[:, 0] + sig[:, 2]) / 2 + root) / np.maximum((sig[:, 0] + sig[:, 2]) / 2 - root, 1e-9))
    xy = tb["xy"][tb["query"]]
    inside = (xy.min(1) > 250) & (xy.max(1) < 750)
    candidates = np.flatnonzero(inside & (ratio > 3.0))
    return int(candidates[0])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper", type=Path)
    args = parser.parse_args()
    tb = dict(np.load(gp.OUTPUT / "test_batch.npz"))
    results = json.loads((gp.OUTPUT / "results.json").read_text())
    i = pick_query(tb)
    one = {k: v[i:i + 1] for k, v in tb.items() if k not in ("xy", "psi", "field")}
    batch = gp.to_device(one, "cpu")
    weights = {}
    with torch.no_grad():
        for kind in ("softmax", "ka_flow"):
            model = load(kind)
            _, _, out = model(batch)
            weights[kind] = out["weights"].float().numpy()[0, 0]                  # variable 0, [K]
            if kind == "ka_flow":
                learned = gp.KA_query_sigma(model.read, model.net(batch["hq"])).numpy()[0, 0]
    scene = int(tb["scene"][i])
    field = tb["field"][scene][:, 0].reshape(gp.N_GRID, gp.N_GRID)
    psi = tb["psi"][scene].reshape(gp.N_GRID, gp.N_GRID)
    query = tb["xy"][tb["query"][i]]
    neighbours = query + tb["offsets"][i]
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.linewidth": .6, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.0, 2.25))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, .95], wspace=.75)
    extent = [0, gp.SIZE_KM, 0, gp.SIZE_KM]
    window = 230.
    for k, (kind, title) in enumerate((("softmax", "(a) softmax attention"), ("ka_flow", "(b) Kriging Attention (ours)"))):
        ax = fig.add_subplot(grid[k])
        ax.imshow(field, origin="lower", extent=extent, cmap="Greys", alpha=.35, vmin=-3, vmax=3)
        ax.contour(np.linspace(0, gp.SIZE_KM, gp.N_GRID), np.linspace(0, gp.SIZE_KM, gp.N_GRID), psi, levels=10,
                   colors=MUTED, linewidths=.4, alpha=.7)
        w = weights[kind]
        scale = max(np.abs(w).max(), 1e-6)
        sc = ax.scatter(neighbours[:, 0], neighbours[:, 1], c=w, cmap="RdBu_r", vmin=-scale, vmax=scale,
                        s=12 + 160 * np.abs(w) / scale, edgecolors=INK, linewidths=.3, zorder=3)
        ax.scatter([query[0]], [query[1]], marker="*", s=90, color="#eda100", edgecolors=INK, linewidths=.5, zorder=4)
        if kind == "ka_flow":
            for sig, style, label in ((learned, "-", "learned kernel"), (tb["sigma_q"][i], "--", "true kernel")):
                theta, major, minor = align.axes(np.asarray(sig)[None])
                ax.add_patch(Ellipse(query, 2 * major[0], 2 * minor[0], angle=np.degrees(theta[0]), fill=False,
                                     lw=1.1, ls=style, color=INK, zorder=5, label=label))
            ax.legend(frameon=True, framealpha=.85, edgecolor="none", loc="lower left", fontsize=6)
            negative = (w < 0).sum()
            ax.text(.03, .96, f"{negative} of 32 weights negative", transform=ax.transAxes, va="top", fontsize=6.5, color=INK,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1.5}, zorder=6)
        else:
            ax.text(.03, .96, "all weights $\\geq 0$, sum $\\leq 1$", transform=ax.transAxes, va="top", fontsize=6.5, color=INK,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1.5}, zorder=6)
        ax.set_xlim(query[0] - window, query[0] + window)
        ax.set_ylim(query[1] - window, query[1] + window)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, loc="left")
    ax = fig.add_subplot(grid[2])
    names = [("oi", "OI (tuned)"), ("softmax", "softmax attention"), ("ka_stationary", "KA, stationary"),
             ("ka_flow", "KA, flow (ours)"), ("oracle", "exact GP, true cov.")]
    values = [results[k]["rmse"] for k, _ in names]
    colours = [MUTED, SLOT[1], SLOT[2], SLOT[0], "#b9b8b2"]
    bars = ax.barh(range(len(names)), values, color=colours, height=.6)
    for b, v in zip(bars, values):
        ax.text(v + .005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n for _, n in names])
    ax.invert_yaxis()
    ax.set_xlim(.3, max(values) * 1.12)
    ax.set_xlabel("RMSE on 200 test fields")
    ax.set_title("(c) error of each read-out", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=.5)
    out = args.paper / "fig" / "teaser.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"wrote {out}; query {i}, scene {scene}")


if __name__ == "__main__":
    main()
