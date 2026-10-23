"""Dense surface grids and lagged surface features of an iHESP 0.1-degree cohort, for the gridded model.

``86_ihesp_surface_cache.py`` averages the 0.1-degree SST, SSS and steric height into 1-degree
boxes every month and keeps their z-scores at the profiles (each profile reads its own box). The
gridded state reads the whole field and a forecast reads earlier months, so this writes, in the
layouts of ``100_surface_grids.py`` and ``93_lagged_surface.py``:

  surface_grids.npz   [M, 3, 180, 360] z-scores of the 1-degree boxes (86's 2000-2003 monthly
                      climatology, mean and spread, recomputed as 86 does from its 2000-2003 box
                      fields in float64, so they are 86's numbers to the bit)
  lagged_surface.npz  [P, 3, 7] the surface columns of month m - lead at each profile's box, for
                      leads 1, 3, 6; zeros (every flag off) where that month is not in the cohort

The monthly box fields are 86's (``--months-folder``); months not there are computed with 86's
``compute_month`` from ``--source-folder``. Checks before writing: every profile's box in these
grids gives its surface columns 44-50 of ``--satellites`` exactly, which lead 0 of the lagged
recipe must also give.

  .venv/bin/python experiments/synthetic/136_ihesp_grid_caches.py ihesp_hr_uniform
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

spec = importlib.util.spec_from_file_location("ihesp_surface86", ROOT / "experiments/synthetic/86_ihesp_surface_cache.py")
S86 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = S86
spec.loader.exec_module(S86)

RADIUS = ROOT / "outputs/radius_scale_20261008"
#: 86's box fields of the training cohort's months; the 2000-2003 ones fix the normalisation
TRAINING_MONTHS = RADIUS / "ihesp_hr_uniform/surface_months"
LEADS = (1, 3, 6)


def box_fields(months, folders):
    """``[M, 3, 64800]`` absolute SST, SSS and steric height of the 1-degree boxes, from the first
    of ``folders`` that holds each month (86's ``t<month>.npy``)."""
    folders = [Path(f) for f in ([folders] if isinstance(folders, (str, Path)) else folders)]
    out = []
    for t in months:
        path = next((f / f"t{int(t):02d}.npy" for f in folders if (f / f"t{int(t):02d}.npy").exists()), None)
        if path is None:
            raise SystemExit(f"no box fields of month {int(t)} in {[str(f) for f in folders]}")
        out.append(np.load(path))
    return np.stack(out).astype(np.float64)


def surface_columns(z, absolute_sst, box):
    """The seven surface columns (44-50) of profiles from ``z [n, 3]`` and SST at their boxes."""
    out = np.zeros((len(box), 7), np.float32)
    for k in range(3):
        out[:, 2 * k] = np.nan_to_num(z[:, k])
        out[:, 2 * k + 1] = np.isfinite(z[:, k])
    out[:, 6] = np.nan_to_num((absolute_sst - 14.0) / 11.0)
    return out


def main():
    from ocean_tokenizer import synth_argo_eval as S
    from ocean_tokenizer.latent_experiment import array_fingerprint
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cohort")
    p.add_argument("--satellites", type=Path, default=None, help="86's cache of this cohort")
    p.add_argument("--months-folder", type=Path, nargs="+", default=[TRAINING_MONTHS])
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    started = time.time()
    satellites = args.satellites or RADIUS / args.cohort / "satellites.npz"
    output = args.output or ROOT / "outputs/ihesp_20261021" / args.cohort
    c, norm, obs = S.load(str(ROOT), cohort=args.cohort)
    standardized = np.stack([obs[ch] for ch in ("TEMP", "SALT")], axis=-1)
    fingerprint = array_fingerprint((c.month_index, c.lat, c.lon, c.wmo, c.levels, standardized))
    with np.load(satellites, allow_pickle=False) as cache:
        meta = json.loads(str(cache["metadata"].item()))
        features = cache["features"]
    if meta["cohort_fingerprint"] != fingerprint:
        raise SystemExit("the satellite cache was built on another cohort")
    train_months = np.arange(48)                                                      # 2000-01 ... 2003-12
    _, climate, mean, std = S86.anomaly_z(box_fields(train_months, TRAINING_MONTHS), train_months % 12,
                                          np.ones(48, bool))
    months = np.unique(c.month_index)
    fields = box_fields(months, args.months_folder)                                  # [M, 3, 64800]
    z = (fields - climate[months % 12] - mean[None, :, None]) / std[None, :, None]
    where = {int(m): i for i, m in enumerate(months)}
    box = S86.one_degree_box(c.lat, c.lon)
    row = np.array([where[int(m)] for m in c.month_index])
    columns = surface_columns(z[row, :, box], fields[row, 0, box], box)
    if not np.array_equal(columns, features[:, 44:51]):
        worst = float(np.abs(columns - features[:, 44:51]).max())
        raise SystemExit(f"the grids do not give 86's surface columns at the profiles (largest difference {worst:.3g})")

    lagged = np.zeros((len(c.lat), len(LEADS), 7), np.float32)
    for j, lead in enumerate(LEADS):
        source = c.month_index - lead
        have = np.isin(source, months)
        r = np.array([where[int(m)] for m in source[have]])
        lagged[have, j] = surface_columns(z[r, :, box[have]], fields[r, 0, box[have]], box[have])
    common = {"cohort_fingerprint": fingerprint, "cohort": args.cohort, "source": str(satellites),
              "satellites_sha256": hashlib.sha256(Path(satellites).read_bytes()).hexdigest(),
              "normalization": "86's 2000-2003 monthly climatology, mean and spread",
              "profiles_read": "the 1-degree box that holds the profile",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=True)
    grids = z.reshape(len(months), 3, 180, 360).astype(np.float16)
    for name, arrays in (("surface_grids.npz", dict(grids=grids, grid_month=months.astype(int),
                                                     grid_lat=np.arange(-89.5, 90., 1.), grid_lon=np.arange(.5, 360., 1.),
                                                     metadata=json.dumps({**common, "units": "z-score"}))),
                         ("lagged_surface.npz", dict(features=lagged, leads=np.array(LEADS), month_index=c.month_index,
                                                     lat=c.lat, lon=c.lon, metadata=json.dumps(common)))):
        temporary = output / (name + ".tmp.npz")
        np.savez(temporary, **arrays)
        os.replace(temporary, output / name)
    print(json.dumps({"cohort": args.cohort, "months": int(len(months)), "profiles": int(len(c.lat)),
                      "lagged_available": {str(l): float((lagged[:, j, 1] > 0).mean()) for j, l in enumerate(LEADS)},
                      "output": str(output), "seconds": round(time.time() - started)}), flush=True)


if __name__ == "__main__":
    main()
