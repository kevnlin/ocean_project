"""Reconstruction maps from 128's dense grid (January 2007, 105 m, North Atlantic; sealed CESM2 test).

  maps.pdf      truth with the month's input profiles, and the end-to-end models' reconstructions
                (KA-Ocean, ARMOR3D-style OI, Aardvark-style = KA-Ocean without KA, Poseidon-T) over
                their errors, each titled with its RMSE over the map
  maps_all.pdf  the same for every model, including OI and the OI-anchored transformer (supplement)

Reads only ``outputs/dense_maps_20261021/m84_l105.npz``; the month, depth and region were fixed in
128 before any map was computed.

  .venv/bin/python experiments/synthetic/129_dense_map_figure.py PAPER_FOLDER
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("figures123", ROOT / "experiments/synthetic/123_paper_figures.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)

MAPS = ROOT / "outputs/dense_maps_20261021/m84_l105.npz"
REGION = {"lat": (15., 60.), "lon": (280., 350.)}
LAND = "#d9d8d4"
MAIN = ["KA-Ocean", "ARMOR3D-style OI", "Aardvark-style", "Poseidon-T"]
ALL = ["KA-Ocean", "OI-anchored transformer", "ARMOR3D-style OI", "OI", "Aardvark-style", "Poseidon-T"]
LABEL = {"KA-Ocean": "KA-Ocean (ours)", "Aardvark-style": "Aardvark-style\n(KA-Ocean without KA)"}
SHORT = {"KA-Ocean": "KA-Ocean\n(ours)", "OI-anchored transformer": "OI-anchored\ntransformer",
         "ARMOR3D-style OI": "ARMOR3D-\nstyle OI", "OI": "OI\n", "Aardvark-style": "Aardvark-\nstyle",
         "Poseidon-T": "Poseidon-T\n"}


def load(channel=0):
    z = np.load(MAPS, allow_pickle=False)
    lat, lon = z["store_lat"], z["store_lon"]
    rows = (lat >= REGION["lat"][0]) & (lat <= REGION["lat"][1])
    cols = (lon >= REGION["lon"][0]) & (lon <= REGION["lon"][1])
    truth = z["truth"][channel]
    fields, errors, rmse = {}, {}, {}
    for name in ALL:
        full = np.full(truth.size, np.nan)
        full[z["cells"]] = z[f"pred_{name}"][:, channel]
        full = full.reshape(truth.shape)
        fields[name] = full[np.ix_(rows, cols)]
        errors[name] = fields[name] - truth[np.ix_(rows, cols)]
        rmse[name] = float(np.sqrt(np.nanmean(errors[name] ** 2)))
    keep = ((z["source_lat"] >= REGION["lat"][0]) & (z["source_lat"] <= REGION["lat"][1])
            & (z["source_lon"] % 360 >= REGION["lon"][0]) & (z["source_lon"] % 360 <= REGION["lon"][1]))
    # truth only where the models were read, so every panel shows the same cells
    shown = np.isfinite(fields["KA-Ocean"])
    return {"lat": lat[rows], "lon": lon[cols], "truth": np.where(shown, truth[np.ix_(rows, cols)], np.nan),
            "fields": fields, "errors": errors, "rmse": rmse, "level_m": float(z["level_m"]),
            "sources": (z["source_lat"][keep], z["source_lon"][keep] % 360)}


def panel(ax, data, field, cmap, limit, first_col, last_row, compact=False):
    edges = lambda c: np.r_[c - .5, c[-1] + .5]
    ax.set_facecolor(LAND)
    mesh = ax.pcolormesh(edges(data["lon"]), edges(data["lat"]), field, cmap=cmap, vmin=-limit, vmax=limit,
                         rasterized=True, shading="flat")
    ax.set_xlim(REGION["lon"]); ax.set_ylim(REGION["lat"])
    ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(REGION["lat"]))))
    ax.set_xticks([290, 320] if compact else [290, 310, 330]); ax.set_yticks([20, 40, 60])
    ax.set_xticklabels((["70°W", "40°W"] if compact else ["70°W", "50°W", "30°W"]) if last_row else [])
    ax.set_yticklabels(["20°N", "40°N", "60°N"] if first_col else [])
    ax.tick_params(length=2, pad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    return mesh


def figure(names, out, data, width, compact=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F.style(plt)
    n = len(names) + 1
    limit = float(np.ceil(np.nanpercentile(np.abs(data["truth"]), 98) * 4) / 4)
    elimit = float(np.ceil(np.nanpercentile(np.abs(data["errors"]["KA-Ocean"]), 98) * 2 * 10) / 10)
    fig, axes = plt.subplots(2, n, figsize=(width, width * .36), gridspec_kw={"wspace": .06, "hspace": .32})
    mesh = panel(axes[0, 0], data, data["truth"], "RdBu_r", limit, True, False, compact)
    axes[0, 0].scatter(data["sources"][1], data["sources"][0], s=.45, c=F.INK, alpha=.75, linewidths=0)
    axes[0, 0].set_title(f"Truth\n{data['level_m']:.0f} m, Jan 2007" if compact else
                         f"Truth, {data['level_m']:.0f} m, Jan 2007", fontsize=6 if compact else 7)
    for j, name in enumerate(names, 1):
        panel(axes[0, j], data, data["fields"][name], "RdBu_r", limit, False, False, compact)
        axes[0, j].set_title(SHORT[name] if compact else LABEL.get(name, name), fontsize=6 if compact else 7)
        emesh = panel(axes[1, j], data, data["errors"][name], "PuOr_r", elimit, False, True, compact)
        axes[1, j].set_title(f"RMSE {data['rmse'][name]:.3f} °C" if compact else
                             f"error, RMSE {data['rmse'][name]:.3f} °C", fontsize=6 if compact else 6.5, color=F.MUTED)
    axes[1, 0].axis("off")
    box = axes[1, 0].get_position()
    for mappable, height, label in ((mesh, .66, "anomaly (°C)"), (emesh, .18, "error (°C)")):
        cax = fig.add_axes([box.x0 + .08 * box.width, box.y0 + height * box.height, .84 * box.width, .07 * box.height])
        cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        cb.set_label(label, fontsize=6.5, labelpad=1); cb.ax.xaxis.set_label_position("top")
        cb.ax.tick_params(labelsize=6, length=2, pad=1); cb.outline.set_linewidth(.4)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper", type=Path)
    args = parser.parse_args()
    data = load(0)
    salinity = load(1)
    for name in ALL:
        print(f"{name:26s} RMSE T {data['rmse'][name]:.4f} °C  S {salinity['rmse'][name]:.4f} PSU")
    (args.paper / "fig").mkdir(parents=True, exist_ok=True)
    figure(MAIN, args.paper / "fig/maps.pdf", data, 6.9)
    figure(ALL, args.paper / "fig/maps_all.pdf", data, 6.9, compact=True)


if __name__ == "__main__":
    main()
