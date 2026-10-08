"""Exact surface and TEOS-10 finite-difference steric operators at train climate.

Only stored training climatology and geometry enter the Jacobian. No target-year
T/S measurement is used. Steric SSH is linearized, not an exact nonlinear operator.
"""
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import xarray as xr
from ocean_tokenizer.ssh import steric_height_columns


def jacobian(task):
    begin, temp, salt, depth, lat, lon = task
    good = np.isfinite(temp).all(1) & np.isfinite(salt).all(1)
    n, l = temp.shape
    gradient, base = np.zeros((n, l, 2), dtype=np.float64), np.zeros(n, dtype=np.float64)
    if good.any():
        t, s = temp[good].T.astype(np.float64), salt[good].T.astype(np.float64)
        la, lo = lat[good], lon[good]
        f = steric_height_columns(t, s, depth, la, lo)
        base[good] = f
        for j in range(l):
            for ch, eps in ((0, 0.001), (1, 0.0001)):
                perturbed = (t if ch == 0 else s).copy()
                perturbed[j] += eps
                f1 = steric_height_columns(perturbed if ch == 0 else t,
                    perturbed if ch == 1 else s, depth, la, lo)
                gradient[good, j, ch] = (f1 - f) / eps
    return begin, gradient, base, good


def prepare(root, satellite_path, output, workers=8):
    with np.load(satellite_path, allow_pickle=False) as cache:
        meta = json.loads(str(cache["metadata"].item()))
        sm, ss = cache["surface_mean"], cache["surface_std"]
        pm, ps = cache["normalization_mean"], cache["normalization_std"]
        surface_climate = cache["surface_climatology"]
        grid_lat, grid_lon = cache["grid_lat"], cache["grid_lon"]
    with xr.open_dataset(root / "data/synthetic_argo/cesm2_uniform.nc") as raw:
        order = np.argsort(raw.month_index.values, kind="stable")
        temp = np.asarray(raw.CLIM_POS_TEMP.values)[order]
        salt = np.asarray(raw.CLIM_POS_SALT.values)[order]
        lat, lon, month = raw.lat.values[order], raw.lon.values[order], raw.month_index.values[order]
        depth = raw.level.values
    spec = importlib.util.spec_from_file_location("prepare_satellite", root / "experiments/synthetic/50_prepare_matched_satellites.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    sample = np.zeros((len(temp), 3), dtype=np.float64)
    for m in range(12):
        rows = np.flatnonzero(month % 12 == m)
        sample[rows] = module.sample_grid(surface_climate[m], lat[rows], lon[rows], grid_lat, grid_lon)
    operator = np.zeros((len(temp), 3, len(depth)*2), dtype=np.float32)
    offset = np.zeros((len(temp), 3), dtype=np.float32)
    valid = np.zeros((len(temp), 3), dtype=bool)
    for ch in range(2):
        operator[:, ch, ch] = ps[0, ch] / ss[ch]
        climate = temp if ch == 0 else salt
        offset[:, ch] = (climate[:, 0] + pm[0, ch] - sample[:, ch] - sm[ch]) / ss[ch]
        valid[:, ch] = np.isfinite(offset[:, ch])
    tasks = ((i, temp[i:i+4096], salt[i:i+4096], depth, lat[i:i+4096], lon[i:i+4096])
             for i in range(0, len(temp), 4096))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for count, (start, grad, base, good) in enumerate(pool.map(jacobian, tasks)):
            stop = start + len(base)
            operator[start:stop, 2] = (grad * ps / ss[2]).reshape(len(base), -1)
            offset[start:stop, 2] = (base + (grad * pm).sum((1, 2)) - sample[start:stop, 2] - sm[2]) / ss[2]
            valid[start:stop, 2] = good & np.isfinite(offset[start:stop, 2])
            if count % 8 == 0:
                print(f"linearized operators {stop}/{len(temp)}", flush=True)
    operator = np.where(valid[..., None], operator, 0)
    offset = np.where(valid, offset, 0)
    metadata = {"cohort_fingerprint": meta["cohort_fingerprint"],
        "recipe": "exact SST/SSS and forward finite-difference TEOS-10 steric Jacobian around stored training-year monthly climatology",
        "T_perturbation_C": .001, "S_perturbation_PSU": .0001,
        "ssh_nonlinear_operator": False, "target_year_measurements_used": False,
        "surface_cache_sha256": hashlib.sha256(satellite_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "valid_per_modality": valid.sum(0).tolist()}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, operator=operator, offset=offset, valid=valid,
                        month_index=month, lat=lat, lon=lon, metadata=json.dumps(metadata))
    os.replace(temporary, output)
    print(json.dumps({"cache": str(output), **metadata}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--satellites", type=Path, default=ROOT / "outputs/synthetic_matched_20261007/satellites.npz")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/innovation_20261007/operators.npz")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    prepare(a.root, a.satellites, a.output, a.workers)
