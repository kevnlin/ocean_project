"""Do the learned kriging kernels follow the flow? The flow kernel's long axis against SSH isolines.

For a gridded checkpoint with the flow kernel (``ka_flow``), scores every lead-0 cell of its own
selection years (CESM2 2004-2005, or real 2021; no sealed year is read), captures through a forward
hook the length-scale matrix ``Sigma = ell^2 L L^T`` the kriging module gives the query point for each
variable (``kriging_attention``; the hook recomputes it from the module's own heads and checks it
against the module's east/north length scales), and compares its long axis with the isolines of the
month's sea-surface-height anomaly at the query (steric SSH on CESM2, DUACS SLA on real data; the
input surface grid of that month). Alignment is |cos| of the angle between the two directions: 2/pi
(0.637) for orientations unrelated to the flow, 1 for kernels stretched along the isolines. Ocean
anomalies are stretched east-west, and so are many isolines, so two controls separate the parts:
``always_east`` is the alignment a kernel pointing east everywhere would get, and ``tilt_correlation``
correlates the kernel's and the isolines' tilts away from east-west (doubled angles, sin 2x); it is 0
when kernels do not turn with the local flow, whatever their mean orientation.

Writes ``outputs/figures_20261020/kernel_alignment_<campaign>__<run>.{npz,json,png,pdf}``: the
cells (with the months' SSH grids), the statistics by variable, by depth band (all cells, and the
quarters with the weakest and strongest SSH gradient) and by latitude band, and a figure: kernel
ellipses over the SSH anomaly in the Gulf Stream and Kuroshio regions, kernel anisotropy and
east-west orientation by latitude, and the kernel tilt against the isoline tilt where the SSH
gradient is strongest. ``--from-saved`` redoes statistics and figure from the saved cells.

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


def tilt_correlation(theta, gradient):
    """Correlation of the kernel's and the isolines' tilts away from east-west (doubled angles, sin 2x):
    0 when the kernels do not turn with the local flow, whatever their mean orientation."""
    phi = np.arctan2(gradient[..., 0], -gradient[..., 1])
    return float(np.corrcoef(np.sin(2 * theta), np.sin(2 * phi))[0, 1])


def block(theta, major, minor, gradient):
    """Orientation statistics of a set of cells. ``always_east`` is the alignment a kernel pointing
    east everywhere would get: the part of ``alignment`` explained by zonal elongation alone."""
    return {"n": int(len(theta)), "alignment": float(alignment(theta, gradient).mean()),
            "always_east": float(alignment(np.zeros_like(theta), gradient).mean()),
            "tilt_correlation": tilt_correlation(theta, gradient),
            "zonal_orientation": float(np.abs(np.cos(theta)).mean()),
            "anisotropy": float(np.median(major / np.maximum(minor, 1e-9))),
            "major_km": float(np.median(major)), "minor_km": float(np.median(minor))}


LATITUDES = ((-90, -40), (-40, -15), (-15, 15), (15, 40), (40, 90))


def statistics(cells):
    """By variable: per depth band (all cells, weakest and strongest SSH-gradient quarters) and per
    latitude band."""
    out = {}
    bands = band_of_levels(np.asarray(cells["levels_m"]), BANDS)[cells["level"]]
    strength = np.hypot(*np.moveaxis(cells["gradient"], -1, 0))
    finite = np.isfinite(strength)
    for j, ch in enumerate(CH):
        theta, major, minor = axes(cells["sigma"][:, j])
        cut = lambda m: block(theta[m], major[m], minor[m], cells["gradient"][m])
        out[ch] = {"by_band": {}, "by_latitude": {}}
        for name in ["all", *dict.fromkeys(band_of_levels(np.asarray(cells["levels_m"]), BANDS))]:
            keep = finite & ((bands == name) if name != "all" else True)
            if not keep.any():
                continue
            low, high = np.quantile(strength[keep], [.25, .75])
            out[ch]["by_band"][name] = {"all": cut(keep), "weakest_gradient_quarter": cut(keep & (strength <= low)),
                                        "strongest_gradient_quarter": cut(keep & (strength >= high))}
        for lo, hi in LATITUDES:
            keep = finite & (cells["lat"] >= lo) & (cells["lat"] < hi)
            if keep.sum() > 100:
                out[ch]["by_latitude"][f"{lo}..{hi}"] = cut(keep)
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


def figure(cells, level, month, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    fig, panels = plt.subplots(1, 4, figsize=(23, 4.8), gridspec_kw={"width_ratios": [1.15, 1.25, .75, .75], "wspace": .55})
    grid_lat, grid_lon = cells["grid_lat"], cells["grid_lon"]
    ssh = cells["ssh_grids"][list(cells["ssh_months"]).index(month)].astype(np.float32)
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
        for i in rng.permutation(inside)[:70]:
            scale = .4 / KM_PER_DEGREE                                          # drawn at 0.4 of the length scale
            ax.add_patch(Ellipse((lon[i], cells["lat"][i]), 2 * major[i] * scale / np.cos(np.radians(cells["lat"][i])),
                                 2 * minor[i] * scale, angle=np.degrees(theta[i]), fill=False, lw=.9, color="#1b1b1b"))
        ax.set_xlim(lo0, lo1)
        ax.set_ylim(la0, la1)
        ax.set_aspect(1 / np.cos(np.radians((la0 + la1) / 2)))
        ax.set_title(f"{name}: temperature kernels at {cells['levels_m'][level]:.0f} m", fontsize=10)
        ax.set_xlabel("longitude (°E)")
        ax.set_ylabel("latitude (°N)")
    fig.colorbar(mesh, ax=panels[:2], shrink=.8, label="SSH anomaly (z)")
    strength = np.hypot(*np.moveaxis(cells["gradient"], -1, 0))
    finite = np.isfinite(strength)
    centres, aniso, zonal = [], [], []
    for lo in range(-70, 70, 10):
        m = finite & (cells["lat"] >= lo) & (cells["lat"] < lo + 10)
        if m.sum() > 200:
            centres.append(lo + 5)
            aniso.append(np.median(major[m] / np.maximum(minor[m], 1e-9)))
            zonal.append(np.abs(np.cos(theta[m])).mean())
    panels[2].plot(centres, aniso, "o-", color="#3465a4", label="long / short axis")
    panels[2].set_xlabel("latitude (°N)")
    panels[2].set_ylabel("median anisotropy", color="#3465a4")
    twin = panels[2].twinx()
    twin.plot(centres, zonal, "s--", color="#cc0000", label="long axis east-west |cos|")
    twin.axhline(2 / np.pi, color="#cc0000", lw=.8, ls=":")
    twin.set_ylim(.5, 1.)
    twin.set_ylabel("east-west orientation |cos θ|", color="#cc0000")
    panels[2].set_title("kernel shape by latitude (temperature)", fontsize=10)
    strong = finite & (strength >= np.quantile(strength[finite], .75))
    phi = np.arctan2(cells["gradient"][:, 0], -cells["gradient"][:, 1])
    x, y = np.sin(2 * phi[strong]), np.sin(2 * theta[strong])
    edges = np.linspace(-1, 1, 11)
    which = np.clip(np.digitize(x, edges) - 1, 0, 9)
    means = [y[which == b].mean() if (which == b).any() else np.nan for b in range(10)]
    panels[3].plot((edges[:-1] + edges[1:]) / 2, means, "o-", color="#3465a4")
    panels[3].axhline(0, color="k", lw=.6)
    panels[3].plot([-1, 1], [-1, 1], ":", color="grey", lw=.8, label="kernel turns fully with the isolines")
    panels[3].set_xlabel("isoline tilt sin 2φ")
    panels[3].set_ylabel("mean kernel tilt sin 2θ")
    panels[3].set_title(f"tilt with the flow, strongest-gradient quarter (r = {tilt_correlation(theta[strong], cells['gradient'][strong]):+.2f})",
                        fontsize=10)
    panels[3].legend(fontsize=8)
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
    parser.add_argument("--from-saved", action="store_true", help="recompute statistics and figure from the saved cells")
    args = parser.parse_args()
    summaries = sorted(args.run.glob("summary_seed*.json"))
    if len(summaries) != 1:
        raise SystemExit(f"{args.run}: expected one finished run")
    seed = json.loads(summaries[0].read_text())["seed"]
    stem = OUTPUT / ("kernel_alignment_" + CAL.record_name(args.run)[:-len(".json")])
    if args.from_saved:
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as saved:
            cells = {k: saved[k] for k in saved.files}
        tag, step = str(cells["tag"]), int(cells["step"])
    else:
        checkpoint, cells, grids, grid_lat, grid_lon = collect(args.run / f"best_seed{seed}.pt", args.device)
        months = sorted(grids)
        tag, step = checkpoint["training"]["tag"], checkpoint["step"]
        cells.update(ssh_grids=np.stack([grids[m] for m in months]).astype(np.float16), ssh_months=np.asarray(months),
                     grid_lat=grid_lat, grid_lon=grid_lon, tag=np.asarray(tag), step=np.asarray(step))
        OUTPUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(stem.with_suffix(".npz"), **cells)
    stats = {"run": str(args.run), "checkpoint_step": step, "cells": int(len(cells["level"])),
             "months": sorted(set(cells["month"].tolist())), "random_alignment": 2 / math.pi,
             "statistics": statistics(cells)}
    stem.with_suffix(".json").write_text(json.dumps(stats, indent=1) + "\n")
    level = int(np.argmin(np.abs(cells["levels_m"] - args.level_m)))
    month = args.month if args.month is not None else stats["months"][0]
    every = stats["statistics"]["TEMP"]["by_band"]["all"]
    figure(cells, level, month,
           f"{tag}: long axis along the SSH-anomaly isolines |cos| {every['all']['alignment']:.3f} "
           f"(a kernel pointing east everywhere: {every['all']['always_east']:.3f}; 2/π = {2 / math.pi:.3f} for no relation); "
           f"map month {month}", stem)
    for ch in CH:
        blocks = stats["statistics"][ch]["by_band"]["all"]
        a, s = blocks["all"], blocks["strongest_gradient_quarter"]
        print(f"{ch}: alignment {a['alignment']:.3f} (always-east {a['always_east']:.3f}), strongest quarter "
              f"{s['alignment']:.3f} ({s['always_east']:.3f}); tilt correlation {a['tilt_correlation']:+.3f}, strongest "
              f"quarter {s['tilt_correlation']:+.3f}; east-west orientation {a['zonal_orientation']:.3f}; median anisotropy "
              f"{a['anisotropy']:.2f}, long/short axis {a['major_km']:.0f}/{a['minor_km']:.0f} km")
    print(f"wrote {stem}.{{npz,json,png,pdf}}")


if __name__ == "__main__":
    main()
