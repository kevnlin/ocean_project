"""Do the learned kriging kernels follow the flow? The flow kernel's long axis against SSH isolines.

For a gridded checkpoint with the flow kernel (``ka_flow``), scores every lead-0 cell of its own
selection years (CESM2 2004-2005, or real 2021; no sealed year is read), captures through a forward
hook the length-scale matrix ``Sigma = ell^2 L L^T`` the kriging module gives the query point for each
variable (``kriging_attention``; the hook recomputes it from the module's own heads and checks it
against the module's east/north length scales), and compares its long axis with the isolines of the
month's sea-surface-height anomaly at the query (steric SSH on CESM2, DUACS SLA on real data; the
input surface grid of that month). Alignment is |cos| of the angle between the two directions: 2/pi
(0.637) for orientations unrelated to the flow, 1 for kernels stretched along the isolines, below
2/pi for kernels stretched across them.

Writes ``outputs/figures_20261020/kernel_alignment_<campaign>__<run>.{npz,json,png,pdf}``: the
cells, the statistics by variable and depth band (all cells, and the quarters with the weakest and
strongest SSH gradient), and a figure (kernel ellipses over the SSH anomaly in the Gulf Stream and
Kuroshio regions, and the angle histogram of the strongest-gradient quarter).

  .venv/bin/python experiments/synthetic/117_kernel_alignment.py RUN_FOLDER [--level-m 5] [--month M]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ocean_tokenizer import calibration as CAL
from ocean_tokenizer.point_baselines import BANDS, CH, band_of_levels

OUTPUT = ROOT / "outputs/figures_20261020"
SEALED_YEARS = {2006, 2007, 2022, 2023, 2024, 2025}
KM_PER_DEGREE = 111.195
REGIONS = {"Gulf Stream": ((25., 50.), (280., 320.)), "Kuroshio": ((22., 45.), (125., 170.))}


def query_sigma(ka, h, level):
    """``[Q, 2, 3]`` (s11, s12, s22) in km^2, east/north, of the query point for each variable."""
    import torch
    from ocean_tokenizer import kriging_attention as KA
    f64 = torch.float64
    with torch.autocast(device_type=h.device.type, enabled=False):
        ell = ka.log_ell.to(f64).clamp(math.log(10.), math.log(5000.))[:, level].T.exp()      # [Q,2]
        dtype = ka.query_head[0].weight.dtype
        raw = ka.query_head(h.to(dtype)).to(f64).view(-1, 2, 4)
        shape = torch.cat((KA.SHAPE_BOUND * raw[..., :2].tanh(), KA.SHEAR_BOUND * raw[..., 2:3].tanh()), -1)
        return KA.length_scale_matrices(ell, shape)


def scenes(training, device):
    """The run's own data, as its runner built it."""
    if training.get("dataset") == "real_global":
        from ocean_tokenizer.real_forecast_scenes import RealGridForecastScenes
        return RealGridForecastScenes(ROOT, device, caches=Path(training["real_caches"]),
                                      oi_summary=Path(training["real_oi_summary"]), leads=training["leads"],
                                      train_years=training.get("real_train_years"))
    from ocean_tokenizer.grid_forecast_scenes import GridForecastScenes
    return GridForecastScenes(ROOT, device, surface_grids=Path(training["surface_grids"]), leads=training["leads"],
                              lagged_surface=Path(training["lagged_surface"]), train_years=training["train_years"],
                              surface=True, satellite_cache=Path(training["satellite_cache"]),
                              oi_summary=Path(training["oi_summary"]), cohort=training["cohort"])


def ssh_gradient(grid, lat, lon, grid_lat, grid_lon):
    """East and north derivatives (per km) of a [H, W] field on a regular global grid, at the nearest cells."""
    g = np.asarray(grid, np.float64)
    dlat, dlon = grid_lat[1] - grid_lat[0], grid_lon[1] - grid_lon[0]
    north = np.full_like(g, np.nan)
    north[1:-1] = (g[2:] - g[:-2]) / (2 * dlat * KM_PER_DEGREE)
    east = (np.roll(g, -1, 1) - np.roll(g, 1, 1)) / (2 * dlon * KM_PER_DEGREE * np.cos(np.radians(grid_lat))[:, None])
    i = np.clip(np.round((lat - grid_lat[0]) / dlat).astype(int), 0, len(grid_lat) - 1)
    j = np.round(((lon - grid_lon[0]) % 360.) / dlon).astype(int) % len(grid_lon)
    return np.stack((east[i, j], north[i, j]), -1)


def axes(sigma):
    """Long-axis angle (radians from east, counter-clockwise, modulo pi) and the two semi-axes (km)."""
    s11, s12, s22 = sigma[..., 0], sigma[..., 1], sigma[..., 2]
    theta = .5 * np.arctan2(2 * s12, s11 - s22)
    root = np.sqrt(((s11 - s22) / 2) ** 2 + s12 ** 2)
    return theta, np.sqrt((s11 + s22) / 2 + root), np.sqrt(np.maximum((s11 + s22) / 2 - root, 0.))


def alignment(theta, gradient):
    """|cos| of the angle between the long axis and the isoline direction (-dy, dx)."""
    phi = np.arctan2(gradient[..., 0], -gradient[..., 1])
    return np.abs(np.cos(theta - phi))


def statistics(cells):
    out = {}
    bands = band_of_levels(np.asarray(cells["levels_m"]), BANDS)[cells["level"]]
    strength = np.hypot(*np.moveaxis(cells["gradient"], -1, 0))
    for j, ch in enumerate(CH):
        theta, major, minor = axes(cells["sigma"][:, j])
        score = alignment(theta, cells["gradient"])
        out[ch] = {}
        for name in ["all", *dict.fromkeys(band_of_levels(np.asarray(cells["levels_m"]), BANDS))]:
            keep = np.isfinite(score) & ((bands == name) if name != "all" else True)
            if not keep.any():
                continue
            low, high = np.quantile(strength[keep], [.25, .75])
            block = lambda m: {"n": int(m.sum()), "alignment": float(score[m].mean()),
                               "anisotropy": float(np.median(major[m] / np.maximum(minor[m], 1e-9))),
                               "major_km": float(np.median(major[m])), "minor_km": float(np.median(minor[m]))}
            out[ch][name] = {"all": block(keep), "weakest_gradient_quarter": block(keep & (strength <= low)),
                             "strongest_gradient_quarter": block(keep & (strength >= high))}
    return out


def collect(path, device):
    import torch
    from ocean_tokenizer.grid_kriging_ocean import GridKrigingConfig, GridKrigingModel
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    training, config = checkpoint["training"], checkpoint["config"]
    if config["kriging"] != "ka_flow":
        raise SystemExit(f"{path}: the kernel is {config['kriging']}, not the flow kernel")
    if set(training["val_years"]) & SEALED_YEARS:
        raise SystemExit(f"{path}: its selection years include a sealed year")
    data = scenes(training, device)
    for kind in ("mean", "std"):
        for ch in CH:
            if not np.array_equal(np.asarray(checkpoint["normalization"][kind][ch]), getattr(data.norm, kind)[ch]):
                raise SystemExit(f"{path}: the scenes' normalisation differs from training's")
    model = GridKrigingModel(GridKrigingConfig(**config))
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(data.device).eval()
    captured = []

    def hook(module, args, kwargs, output):
        sigma = query_sigma(module, args[0], args[5])
        reference = output["length_scale_km"].double()                                # [Q, 2, east/north]
        if not torch.allclose(sigma[..., ::2].sqrt(), reference, rtol=2e-2):
            raise RuntimeError("the recomputed query kernel differs from the kriging module's")
        captured.append(sigma.cpu().numpy())

    handle = model.kriging.register_forward_hook(hook, with_kwargs=True)
    analysis, context = data.analysis(), training.get("eval_context_profiles") or training["context_profiles"]
    events = [ev for ev in data.forecast_months(list(training["val_years"]), 0) if ev.lead == 0]
    if {2000 + ev.month // 12 for ev in events} - set(training["val_years"]):
        raise SystemExit("an evaluated month lies outside the selection years")
    parts = {k: [] for k in ("lat", "lon", "level", "month", "gradient")}
    with torch.no_grad():
        for ev in events:
            geometry = data.geometry(ev.source, ev.target[ev.profile], 32, ev.level)
            state = None
            for start in range(0, len(ev.level), 2048):
                stop = min(start + 2048, len(ev.level))
                rows, lev = ev.target[ev.profile[start:stop]], ev.level[start:stop]
                scene = data.scene(ev.source, rows, lev, analysis, context, [20260918, ev.month],
                                   {k: v[start:stop] for k, v in geometry.items()})
                with torch.autocast(device_type=data.device.type, dtype=torch.bfloat16,
                                    enabled=training["amp"] and data.device.type == "cuda"):
                    if state is None:
                        state = model.encode(scene)
                    model.decode(scene, state)
                coord = scene["query_coord"][:, :2].double().cpu().numpy()
                parts["lat"].append(coord[:, 0])
                parts["lon"].append(coord[:, 1])
                parts["level"].append(np.asarray(lev))
                parts["month"].append(np.full(stop - start, ev.month))
                grid = scene["surface_grid"][2].double().cpu().numpy()
                parts["gradient"].append(ssh_gradient(grid, coord[:, 0], coord[:, 1],
                                                      np.asarray(data.grid_lat), np.asarray(data.grid_lon)))
    handle.remove()
    grids = {int(m): data.surface_grids[data.grid_index[int(m)]][2].float().cpu().numpy()
             for m in sorted({ev.month for ev in events})}
    cells = {k: np.concatenate(v) for k, v in parts.items()}
    cells["sigma"] = np.concatenate(captured)
    cells["levels_m"] = np.asarray(data.levels, np.float64)
    return checkpoint, cells, grids, np.asarray(data.grid_lat), np.asarray(data.grid_lon)


def figure(cells, grids, grid_lat, grid_lon, level, month, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    fig, panels = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.15, 1.25, .8]})
    ssh = grids[month]
    keep = (cells["month"] == month) & (cells["level"] == level)
    theta, major, minor = axes(cells["sigma"][:, 0])
    rng = np.random.default_rng(0)
    for ax, (name, ((la0, la1), (lo0, lo1))) in zip(panels[:2], REGIONS.items()):
        rows = (grid_lat >= la0) & (grid_lat <= la1)
        columns = (grid_lon >= lo0) & (grid_lon <= lo1)
        field = ssh[np.ix_(rows, columns)]
        mesh = ax.pcolormesh(grid_lon[columns], grid_lat[rows], field, cmap="RdBu_r", vmin=-3, vmax=3, shading="auto")
        ax.contour(grid_lon[columns], grid_lat[rows], field, levels=np.arange(-3, 3.1, .75), colors="k",
                   linewidths=.4, alpha=.6)
        lon = cells["lon"] % 360.
        inside = np.flatnonzero(keep & (cells["lat"] >= la0) & (cells["lat"] <= la1) & (lon >= lo0) & (lon <= lo1))
        for i in rng.permutation(inside)[:90]:
            scale = .45 / KM_PER_DEGREE                                         # draw at 0.45 of the length scale
            ax.add_patch(Ellipse((lon[i], cells["lat"][i]), 2 * major[i] * scale / np.cos(np.radians(cells["lat"][i])),
                                 2 * minor[i] * scale, angle=np.degrees(theta[i]), fill=False, lw=1.1, color="#1b1b1b"))
        ax.set_xlim(lo0, lo1)
        ax.set_ylim(la0, la1)
        ax.set_aspect(1 / np.cos(np.radians((la0 + la1) / 2)))
        ax.set_title(f"{name}: temperature kernels at {cells['levels_m'][level]:.0f} m")
        ax.set_xlabel("longitude (°E)")
        ax.set_ylabel("latitude (°N)")
    fig.colorbar(mesh, ax=panels[:2], shrink=.8, label="SSH anomaly (z)")
    strength = np.hypot(*np.moveaxis(cells["gradient"], -1, 0))
    finite = np.isfinite(strength)
    strong = finite & (strength >= np.quantile(strength[finite], .75))
    phi = np.arctan2(cells["gradient"][:, 0], -cells["gradient"][:, 1])
    difference = np.degrees((theta - phi + np.pi / 2) % np.pi - np.pi / 2)
    panels[2].hist(difference[strong], bins=36, range=(-90, 90), density=True, color="#3465a4", alpha=.85,
                   label="strongest-gradient quarter")
    panels[2].axhline(1 / 180, color="k", ls="--", lw=1, label="no relation to the flow")
    panels[2].set_xlabel("long axis minus SSH isoline direction (°)")
    panels[2].set_ylabel("density")
    panels[2].legend(fontsize=8)
    panels[2].set_title("temperature, all depths")
    fig.suptitle(title, fontsize=10)
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--level-m", type=float, default=5., help="depth of the map panels (nearest level)")
    parser.add_argument("--month", type=int, default=None, help="month index of the map panels (default: first)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summaries = sorted(args.run.glob("summary_seed*.json"))
    if len(summaries) != 1:
        raise SystemExit(f"{args.run}: expected one finished run")
    seed = json.loads(summaries[0].read_text())["seed"]
    checkpoint, cells, grids, grid_lat, grid_lon = collect(args.run / f"best_seed{seed}.pt", args.device)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / ("kernel_alignment_" + CAL.record_name(args.run)[:-len(".json")])
    np.savez_compressed(stem.with_suffix(".npz"), **cells)
    stats = {"run": str(args.run), "checkpoint_step": checkpoint["step"], "cells": int(len(cells["level"])),
             "months": sorted(set(cells["month"].tolist())), "random_alignment": 2 / math.pi,
             "statistics": statistics(cells)}
    stem.with_suffix(".json").write_text(json.dumps(stats, indent=1) + "\n")
    level = int(np.argmin(np.abs(cells["levels_m"] - args.level_m)))
    month = args.month if args.month is not None else stats["months"][0]
    every = stats["statistics"]["TEMP"]["all"]
    figure(cells, grids, grid_lat, grid_lon, level, month,
           f"{checkpoint['training']['tag']}: alignment |cos| {every['all']['alignment']:.3f} over all lead-0 cells, "
           f"{every['strongest_gradient_quarter']['alignment']:.3f} where the SSH gradient is strongest "
           f"(2/π = {2 / math.pi:.3f} for no relation); month {month}", stem)
    for ch in CH:
        block = stats["statistics"][ch]["all"]
        print(f"{ch}: alignment all {block['all']['alignment']:.3f}, weakest quarter "
              f"{block['weakest_gradient_quarter']['alignment']:.3f}, strongest quarter "
              f"{block['strongest_gradient_quarter']['alignment']:.3f}; median anisotropy {block['all']['anisotropy']:.2f}, "
              f"long/short axis {block['all']['major_km']:.0f}/{block['all']['minor_km']:.0f} km")
    print(f"wrote {stem}.{{npz,json,png,pdf}}")


if __name__ == "__main__":
    main()
