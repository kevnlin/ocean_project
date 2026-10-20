"""Dense maps: frozen models read at every ocean grid cell of one sealed-test month (qualitative figure).

For a CESM2 test month (fixed in advance: January 2007, month index 84) at one depth (105 m), query-only
pseudo rows are placed at every ocean cell of a region with the cohort's own feature recipe (position
features + the month's SST / SSS / steric-SSH columns of 88 and 110). Three checks must pass first:
the recipe reproduces the cached features of the month's real query rows; pseudo rows at those rows'
positions reproduce the models' predictions there; and the CESM2 anomaly field (store values minus the
2000-2003 monthly climatology), interpolated to the rows, reproduces their targets. Every model then
decodes the cells from the month's input profiles at lead 0. Writes
``outputs/dense_maps_20261021/m<month>_l<level>.npz``.

  .venv/bin/python experiments/synthetic/128_dense_maps.py [--month 84] [--level-m 105]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ocean_tokenizer import dense_queries as D
from ocean_tokenizer import local_scale
from ocean_tokenizer.synthetic_latent_experiment import position_features


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCORER = _module("scorer109", "experiments/synthetic/109_sealed_cesm2_test.py")
COHORT110 = _module("cohort110", "experiments/synthetic/110_cesm2_test_cohort.py")
TABLES = _module("tables118", "experiments/synthetic/118_validation_tables.py")
STORE = ROOT / "data/cesm2_long/cesm2_le_cohort_levels_1964_2007.zarr"
LONG = ROOT / "outputs/long_record_20261017/satellites.npz"
OUTPUT = ROOT / "outputs/dense_maps_20261021"
REGION = {"lat": (15., 60.), "lon": (280., 350.)}
MODELS = {"KA-Ocean": ("grid", TABLES.PILOT, "gj_flow"), "ARMOR3D-style OI": ("grid", TABLES.FORMAL, "c_gp_oi"),
          "Aardvark-style": ("grid", TABLES.PILOT, "gj_none"), "Poseidon-T": ("grid", TABLES.POSEIDON, "p_none"),
          "OI-anchored transformer": ("reference", TABLES.REFERENCE, "f_y40")}


def surface(month):
    """The month's absolute surface fields and their z scores (88's recipe)."""
    with np.load(LONG, allow_pickle=False) as cache:
        climate, mean, std = cache["surface_climatology"], cache["surface_mean"], cache["surface_std"]
        grid_lat, grid_lon = cache["grid_lat"], cache["grid_lon"]
    with xr.open_zarr(STORE, chunks=None) as ds:
        position = int(np.flatnonzero(np.asarray(ds.month_index.values) == month)[0])
        levels = np.asarray(ds.level.values)
    COHORT110.SURFACE_88.STORE = STORE
    _, fields = COHORT110.SURFACE_88.month_fields((position, levels))
    z = (fields - climate[month % 12] - mean[:, None, None]) / std[:, None, None]
    return fields, z, grid_lat, grid_lon


def features(lat, lon, month, fields, z, grid_lat, grid_lon):
    x = position_features(SimpleNamespace(lat=np.asarray(lat), lon=np.asarray(lon), month_index=np.full(len(lat), month)))
    x[:, 44:51] = COHORT110.columns(z, fields, np.asarray(lat), np.asarray(lon), grid_lat, grid_lon)
    return x.astype(np.float32)


def truth_anomaly(month, level):
    """CESM2 TEMP / SALT anomaly of the month at one level against the 2000-2003 monthly climatology."""
    with xr.open_zarr(STORE, chunks=None) as ds:
        index = np.asarray(ds.month_index.values)
        at = lambda m: int(np.flatnonzero(index == m)[0])
        out = []
        for name in ("TEMP", "SALT"):
            value = np.asarray(ds[name][at(month), level].values, np.float64)
            clim = np.mean([np.asarray(ds[name][at(m), level].values, np.float64) for m in range(month % 12, 48, 12)], 0)
            land = (np.asarray(ds.TEMP[at(month), level].values) == 0) & (np.asarray(ds.SALT[at(month), level].values) == 0)
            field = value - clim
            field[land] = np.nan
            out.append(field)
        return np.stack(out), np.asarray(ds.lat.values), np.asarray(ds.lon.values)


def bilinear(field, lat, lon, grid_lat, grid_lon):
    """87's stencil, with which the cohort's profiles were drawn (NaN where a corner is dry)."""
    idx, w = COHORT110.COHORT_87.stencil(np.asarray(lat, float), np.asarray(lon, float), grid_lat, grid_lon)
    return COHORT110.COHORT_87.interp(field[None], idx, w)[:, 0]


def predict(model, data, ev, rows, level, training):
    """Means and the OI baseline (z units) of ``rows`` at ``level`` from the month's input profiles."""
    import torch
    analysis = data.analysis()
    context = training.get("eval_context_profiles") or training["context_profiles"]
    lev = np.full(len(rows), level)
    geometry = data.geometry(ev.source, rows, 32, lev)
    means, bases, state = [], [], None
    with torch.no_grad():
        for start in range(0, len(rows), 2048):
            stop = min(start + 2048, len(rows))
            piece = {k: v[start:stop] for k, v in geometry.items()}
            scene = data.scene(ev.source, rows[start:stop], lev[start:stop], analysis, context, [20260918, ev.month], piece)
            with torch.autocast(device_type=data.device.type, dtype=torch.bfloat16, enabled=training["amp"] and data.device.type == "cuda"):
                if state is None:
                    state = model.encode(scene)
                prediction = model.decode(scene, state)
            means.append(data.to_z(prediction["mean"].float(), rows[start:stop], lev[start:stop]).cpu().numpy())
            bases.append(data.to_z(scene["baseline"], rows[start:stop], lev[start:stop]).cpu().numpy())
    return np.concatenate(means), np.concatenate(bases)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--month", type=int, default=84)
    parser.add_argument("--level-m", type=float, default=105.)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if 2000 + args.month // 12 not in SCORER.TEST_YEARS:
        raise SystemExit("the maps show a sealed-test month")
    fields, z, grid_lat, grid_lon = surface(args.month)
    out, data_cache, checks = {}, {}, {}
    for name, (family, folder, tag) in MODELS.items():
        path = folder / f"{tag}_s1234" / "best_seed1234.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        training = checkpoint["training"]
        key = (family, training["train_years"], training.get("scale"))
        if key not in data_cache:
            data_cache.clear()
            data_cache[key] = SCORER.scenes(family, training, args.device)
        data = data_cache[key]
        level = int(np.argmin(np.abs(np.asarray(data.levels) - args.level_m)))
        ev = next(e for e in data.forecast_months(list(SCORER.TEST_YEARS), 0) if e.month == args.month and e.lead == 0)
        model = SCORER.model_of(family, checkpoint, data.device)
        real = np.unique(np.asarray(ev.target)[np.asarray(ev.profile)][np.asarray(ev.level) == level])
        lat_r, lon_r = data.c.lat[real], data.c.lon[real]
        day = float(data.day[int(real[0])])
        # check 1: the feature recipe reproduces the cached features of the real query rows
        recipe = features(lat_r, lon_r, args.month, fields, z, grid_lat, grid_lon)
        cached = data.x[torch.as_tensor(real, device=data.device)].cpu().numpy()
        checks.setdefault("feature_recipe_max_abs", float(np.abs(recipe - cached).max()))
        if np.abs(recipe - cached).max() > 1e-4:
            raise SystemExit(f"the feature recipe differs from the cache ({np.abs(recipe - cached).max():.2e})")
        # check 2: pseudo rows at the real rows reproduce the predictions there
        scale_r = None if getattr(data, "scale", None) is None else data.scale[torch.as_tensor(real, device=data.device)].cpu().numpy()
        twin, twin_rows = D.with_pseudo_rows(data, lat_r, lon_r, month=args.month, day=day, features=cached,
                                             targets=data.y[torch.as_tensor(real, device=data.device)].cpu().numpy(), scale=scale_r)
        a, _ = predict(model, data, ev, real, level, training)
        b, _ = predict(model, twin, ev, twin_rows, level, training)
        checks[f"pseudo_rows_{name}"] = float(np.abs(a - b).max())
        if np.abs(a - b).max() > 1e-3:
            raise SystemExit(f"{name}: pseudo rows do not reproduce the predictions ({np.abs(a - b).max():.2e})")
        if "truth" not in out:
            truth, store_lat, store_lon = truth_anomaly(args.month, level)
            # check 3: the anomaly field reproduces the rows' targets (physical units)
            std = np.stack([np.asarray(data.norm.std[ch])[level] for ch in ("TEMP", "SALT")])
            mean = np.stack([np.asarray(data.norm.mean[ch])[level] for ch in ("TEMP", "SALT")])
            target = data.y[torch.as_tensor(real, device=data.device), level].cpu().numpy() * std + mean
            interp = np.stack([bilinear(truth[k], lat_r, lon_r, store_lat, store_lon) for k in range(2)], -1)
            ok = np.isfinite(interp) & np.isfinite(target)
            checks["truth_recipe_median_abs"] = float(np.median(np.abs(interp - target)[ok]))
            if checks["truth_recipe_median_abs"] > 1e-3:
                raise SystemExit(f"the anomaly recipe differs from the cohort's targets ({checks['truth_recipe_median_abs']:.2e})")
            la, lo = np.meshgrid(store_lat, store_lon, indexing="ij")
            region = ((la >= REGION["lat"][0]) & (la <= REGION["lat"][1]) & (lo >= REGION["lon"][0]) & (lo <= REGION["lon"][1])
                      & np.isfinite(truth).all(0) & np.isfinite(z[0]))
            cells = np.flatnonzero(region.ravel())
            out.update(truth=truth, store_lat=store_lat, store_lon=store_lon, cells=cells, level=level,
                       level_m=float(data.levels[level]), month=args.month, std=std, mean=mean,
                       source_lat=np.asarray(data.c.lat[ev.source]), source_lon=np.asarray(data.c.lon[ev.source]))
            out["cell_features"] = features(la.ravel()[cells], lo.ravel()[cells], args.month, fields, z, grid_lat, grid_lon)
        cells = out["cells"]
        la, lo = np.meshgrid(out["store_lat"], out["store_lon"], indexing="ij")
        lat_c, lon_c = la.ravel()[cells], lo.ravel()[cells]
        scale_c = None
        if getattr(data, "scale", None) is not None:
            with np.load(SCORER.DATA["scale_tables"], allow_pickle=False) as tables:
                scale_c = local_scale.lookup(tables[training["scale"]], lat_c, lon_c, np.full(len(cells), args.month % 12))
        dense, rows = D.with_pseudo_rows(data, lat_c, lon_c, month=args.month, day=day, features=out["cell_features"], scale=scale_c)
        mean_z, base_z = predict(model, dense, ev, rows, level, training)
        out[f"pred_{name}"] = mean_z * out["std"] + out["mean"]
        if "pred_OI" not in out:
            out["pred_OI"] = base_z * out["std"] + out["mean"]
        print(json.dumps({"model": name, **{k: v for k, v in checks.items()}}), flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / f"m{args.month}_l{int(out['level_m'])}.npz"
    np.savez_compressed(target, **{k: np.asarray(v) for k, v in out.items() if k != "cell_features"},
                        checks=json.dumps(checks))
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
