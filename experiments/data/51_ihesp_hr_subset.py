"""A subset of the iHESP high-resolution 3-D monthly ocean fields, cut while downloading.

The Hugging Face dataset ``SingleBicycle/iHESP-historical-3D-monthly-hr`` holds
TEMP and SALT on the 0.1-degree POP grid (2400 x 3600, 62 levels, 1920-2005) as
one 36-84 GB NetCDF-4 file per variable and decade: 1.2 TB in all, more than
this server has free. This script fetches only the compressed chunks that hold
the wanted depth levels of the wanted months, as HTTP byte ranges, and writes
one small NetCDF file per variable and month:

    <output>/<VAR>.<YYYYMM>.nc    (time=1, z_t=<levels>, nlat=2400, nlon=3600)
    <output>/grid.nc              TLAT, TLONG, KMT, TAREA, HT, REGION_MASK, all 62 depths
    <output>/manifest.json        source revision, levels, files written

Values are the source's stored float32 values, bit for bit; only the
compression changes. As in the source, salinity is stored in g/g with
``scale_factor = 1000``, so a reader that applies the attribute gets g/kg. By default the levels are the 20 depths of the synthetic Argo cohort
(5-985 m), which are levels of this grid, and the years are those of the
synthetic split (2000-2005). ``--verify`` re-reads random blocks of written
files through the HDF5 library and compares them with the remote file.

The dataset is gated: the token of an approved account is read from
``HF_TOKEN`` or ``~/.cache/huggingface/token``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import requests

from ocean_tokenizer import hdf5_chunks as H

REPO = "SingleBicycle/iHESP-historical-3D-monthly-hr"
#: depths of the synthetic Argo cohort, metres (data/synthetic_argo/cesm2_uniform.nc)
COHORT_DEPTHS_M = (5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 85.0, 105.0, 125.0, 145.0, 165.1,
                   186.3, 222.6, 267.7, 326.9, 408.8, 527.7, 707.6, 984.7)
GRID_FIELDS = ("TLAT", "TLONG", "KMT", "TAREA", "HT", "REGION_MASK", "z_t", "dz")


def token():
    value = os.environ.get("HF_TOKEN")
    if not value:
        value = (Path.home() / ".cache/huggingface/token").read_text().strip()
    return value


def repository():
    reply = requests.get(f"https://huggingface.co/api/datasets/{REPO}", timeout=60,
                         headers={"Authorization": f"Bearer {token()}"})
    reply.raise_for_status()
    info = reply.json()
    return info["sha"], [s["rfilename"] for s in info["siblings"] if s["rfilename"].endswith(".nc")]


def file_for(names, variable, year, month):
    """The decade file of ``variable`` that holds the month, and the month's index in it."""
    for name in names:
        match = re.search(rf"\.{variable}\.(\d{{4}})(\d{{2}})-(\d{{4}})(\d{{2}})\.nc$", name)
        if match:
            first = int(match[1]) * 12 + int(match[2]) - 1
            last = int(match[3]) * 12 + int(match[4]) - 1
            at = year * 12 + month - 1
            if first <= at <= last:
                return name, at - first
    raise ValueError(f"no file holds {variable} {year}-{month:02d}")


def level_indices(z_t_m, depths_m, tolerance=0.2):
    """Index of each wanted depth in the file's depth axis; a depth that is not a level is refused."""
    z = np.asarray(z_t_m, dtype=float)
    index = [int(np.abs(z - d).argmin()) for d in depths_m]
    off = [d for d, i in zip(depths_m, index) if abs(z[i] - d) > tolerance]
    if off:
        raise ValueError(f"depths that are not levels of this grid: {off}")
    return index


class Remote:
    """Byte ranges of one file of the dataset. The signed storage address is
    looked up once and again whenever it stops answering."""

    def __init__(self, revision, name):
        self.url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{name}"
        self.session = requests.Session()
        self.signed = None

    def _locate(self):
        reply = self.session.get(self.url, headers={"Authorization": f"Bearer {token()}"},
                                 allow_redirects=False, timeout=60)
        if reply.status_code not in (301, 302, 303, 307, 308):
            reply.raise_for_status()
            raise RuntimeError(f"expected a redirect to storage, got {reply.status_code}")
        self.signed = reply.headers["Location"]
        if self.signed.startswith("/"):
            self.signed = "https://huggingface.co" + self.signed

    def __call__(self, offset, size):
        problem = None
        for attempt in range(8):
            try:
                if self.signed is None:
                    self._locate()
                reply = self.session.get(self.signed, timeout=600,
                                         headers={"Range": f"bytes={offset}-{offset + size - 1}"})
                if reply.status_code == 206 and len(reply.content) == size:
                    return reply.content
                problem = f"status {reply.status_code}, {len(reply.content)} of {size} bytes"
            except requests.RequestException as error:
                problem = repr(error)
            self.signed = None
            time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"range {offset}+{size} of {self.url}: {problem}")

    def open(self):
        """The file through the HDF5 library (header, grid fields, verification)."""
        import fsspec
        import h5py
        fs = fsspec.filesystem("https", headers={"Authorization": f"Bearer {token()}"})
        handle = fs.open(self.url, "rb", block_size=256 * 1024, cache_type="blockcache")
        return h5py.File(handle, "r")


def describe(revision, name, variable, output):
    """Chunk index and header facts of one remote file, cached beside the output.

    Reading the index of an 80 GB file takes minutes (its B-tree is scattered
    through the file), so it is kept as ``index.<file>.npz``.
    """
    cache = Path(output) / f"index.{name}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            return {key: z[key] for key in z.files}
    with Remote(revision, name).open() as h:
        d = h[variable]
        text = lambda v: v.decode() if isinstance(v, bytes) else str(v)
        facts = {"rows": H.chunk_index(d), "shape": np.array(d.shape), "chunks": np.array(d.chunks),
                 "dtype": np.array(d.dtype.str), "shuffle": np.array(bool(d.shuffle)),
                 "fill": np.array(d.fillvalue), "z_t_cm": h["z_t"][...], "time": h["time"][...],
                 "time_units": np.array(text(h["time"].attrs["units"])),
                 "calendar": np.array(text(h["time"].attrs["calendar"])),
                 "long_name": np.array(text(d.attrs["long_name"])),
                 "units": np.array(text(d.attrs["units"])), "title": np.array(text(h.attrs["title"])),
                 "scale_factor": np.array(float(np.ravel(d.attrs.get("scale_factor", 1.0))[0]))}
    temporary = cache.with_name(cache.name + ".tmp.npz")
    np.savez(temporary, **facts)
    os.replace(temporary, cache)
    return facts


def write_month(path, variable, data, facts, time_index, levels, source):
    import netCDF4
    temporary = path.with_name(path.name + ".tmp")
    with netCDF4.Dataset(temporary, "w", format="NETCDF4") as nc:
        nc.setncatts({"title": str(facts["title"]), "source_dataset": f"huggingface.co/datasets/{REPO}",
                      "source_revision": source[0], "source_file": source[1],
                      "subset": "selected depth levels of one month; values unchanged (float32)",
                      "source_level_index": [int(k) for k in levels]})
        for dim, n in (("time", 1), ("z_t", len(levels)), ("nlat", data.shape[1]), ("nlon", data.shape[2])):
            nc.createDimension(dim, n)
        t = nc.createVariable("time", "f8", ("time",))
        t.units, t.calendar = str(facts["time_units"]), str(facts["calendar"])
        t.long_name = "time (end of the averaging month)"
        t[:] = facts["time"][time_index]
        z = nc.createVariable("z_t", "f4", ("z_t",))
        z.units, z.long_name, z.positive = "centimeters", "depth from surface to midpoint of layer", "down"
        z[:] = facts["z_t_cm"][levels]
        v = nc.createVariable(variable, "f4", ("time", "z_t", "nlat", "nlon"), zlib=True, complevel=4,
                              shuffle=True, chunksizes=(1, 1, 300, 450), fill_value=facts["fill"].item())
        v.long_name, v.units, v.coordinates = str(facts["long_name"]), str(facts["units"]), "TLONG TLAT z_t time"
        if float(facts["scale_factor"]) != 1.0:      # salinity: stored in g/g, unpacked to the stated g/kg
            v.scale_factor = np.float32(facts["scale_factor"])
        v.set_auto_maskandscale(False)
        v[0] = data
    os.replace(temporary, path)


def fetch(task):
    """One variable-month: download its chunks, keep the levels, write the file."""
    revision, name, variable, year, month, time_index, levels, output, min_free_gb = task
    path = Path(output) / f"{variable}.{year}{month:02d}.nc"
    if path.exists():
        return path.name, "present", 0., 0
    if shutil.disk_usage(output).free < min_free_gb * 1e9:
        return path.name, "skipped: free disk space is at the floor", 0., 0
    started = time.time()
    facts = describe(revision, name, variable, output)
    counted = []
    remote = Remote(revision, name)

    def read(offset, size):
        counted.append(size)
        return remote(offset, size)

    data = H.read_levels(read, facts["rows"], tuple(facts["shape"]), tuple(facts["chunks"]), time_index,
                         levels, np.dtype(str(facts["dtype"])), shuffle=bool(facts["shuffle"]),
                         fill=facts["fill"])
    write_month(path, variable, data, facts, time_index, levels, (revision, name))
    return path.name, "written", time.time() - started, sum(counted)


def write_grid(revision, name, output):
    import netCDF4
    path = Path(output) / "grid.nc"
    if path.exists():
        return
    temporary = path.with_name(path.name + ".tmp")
    with Remote(revision, name).open() as h, netCDF4.Dataset(temporary, "w", format="NETCDF4") as nc:
        nc.setncatts({"source_dataset": f"huggingface.co/datasets/{REPO}", "source_revision": revision,
                      "source_file": name, "contents": "grid fields of the 0.1-degree POP grid, unchanged"})
        nc.createDimension("nlat", h["TLAT"].shape[0])
        nc.createDimension("nlon", h["TLAT"].shape[1])
        nc.createDimension("z_t", h["z_t"].shape[0])
        for field in GRID_FIELDS:
            d = h[field]
            dims = ("nlat", "nlon") if d.ndim == 2 else ("z_t",)
            v = nc.createVariable(field, d.dtype.newbyteorder("="), dims, zlib=True, complevel=4, shuffle=True)
            for key in ("long_name", "units"):
                if key in d.attrs:
                    value = d.attrs[key]
                    v.setncattr(key, value.decode() if isinstance(value, bytes) else str(value))
            v.set_auto_maskandscale(False)
            v[:] = d[...]
    os.replace(temporary, path)


def verify(revision, names, output, count, seed=0):
    """Random blocks of written files against the remote file read by the HDF5 library."""
    import netCDF4
    files = sorted(Path(output).glob("[A-Z]*.[0-9][0-9][0-9][0-9][0-9][0-9].nc"))
    rng = np.random.default_rng(seed)
    checked = []
    for path in rng.choice(files, min(count, len(files)), replace=False):
        variable, stamp = path.name.split(".")[:2]
        name, time_index = file_for(names, variable, int(stamp[:4]), int(stamp[4:]))
        with netCDF4.Dataset(path) as nc, Remote(revision, name).open() as h:
            v = nc[variable]
            v.set_auto_maskandscale(False)
            if float(getattr(v, "scale_factor", 1.0)) != float(np.ravel(h[variable].attrs.get("scale_factor", 1.0))[0]):
                raise SystemExit(f"{path.name}: scale_factor differs from the source")
            levels = [int(k) for k in np.atleast_1d(nc.source_level_index)]
            if nc["time"][0] != h["time"][time_index]:
                raise SystemExit(f"{path.name}: time differs from the source")
            for _ in range(3):
                i = int(rng.integers(len(levels)))
                y, x = int(rng.integers(0, v.shape[2] - 300)), int(rng.integers(0, v.shape[3] - 500))
                ours = v[0, i, y:y + 300, x:x + 500]
                theirs = h[variable][time_index, levels[i], y:y + 300, x:x + 500]
                if not np.array_equal(ours, theirs):
                    raise SystemExit(f"{path.name}: level {levels[i]} block ({y}, {x}) differs from the source")
        checked.append(path.name)
        print(f"verified {path.name}: 3 blocks of 300 x 500 identical to the source", flush=True)
    return checked


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=ROOT / "data/ihesp/hr_cohort_levels")
    p.add_argument("--variables", nargs="+", default=["TEMP", "SALT"])
    p.add_argument("--years", nargs="+", type=int, default=[2004, 2005, 2000, 2001, 2002, 2003],
                   help="in the order they should arrive")
    p.add_argument("--depths-m", nargs="+", type=float, default=list(COHORT_DEPTHS_M))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--min-free-gb", type=float, default=20.,
                   help="stop writing when the output disk has less free space than this")
    p.add_argument("--verify", type=int, default=0, help="check this many written files and exit")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    revision, names = repository()
    if args.verify:
        verify(revision, names, args.output, args.verify)
        return
    started = time.time()
    tasks, levels = [], None
    for variable in args.variables:      # the index of each needed file, once, before the workers start
        for year in args.years:
            name, _ = file_for(names, variable, year, 1)
            facts = describe(revision, name, variable, args.output)
            found = level_indices(facts["z_t_cm"] / 100., args.depths_m)
            if levels not in (None, found):
                raise SystemExit("the files do not share one depth axis")
            levels = found
    for year in args.years:
        for month in range(1, 13):
            for variable in args.variables:
                name, time_index = file_for(names, variable, year, month)
                tasks.append((revision, name, variable, year, month, time_index, levels, str(args.output),
                              args.min_free_gb))
    write_grid(revision, tasks[0][1], args.output)
    print(json.dumps({"files": len(tasks), "levels": levels, "revision": revision,
                      "free_gb": shutil.disk_usage(args.output).free / 1e9}), flush=True)
    done = []
    with ProcessPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(fetch, task) for task in tasks]):
            name, state, seconds, fetched = future.result()
            done.append({"file": name, "state": state})
            print(f"{name}: {state}" + (f" in {seconds:.0f}s, {fetched / 1e6:.0f} MB fetched" if fetched else ""),
                  flush=True)
    written = sorted(Path(args.output).glob("[A-Z]*.[0-9][0-9][0-9][0-9][0-9][0-9].nc"))
    manifest = {"source_dataset": f"huggingface.co/datasets/{REPO}", "source_revision": revision,
                "variables": args.variables, "depths_m": args.depths_m, "source_level_index": levels,
                "grid": "0.1-degree POP tripole, 2400 x 3600; see grid.nc",
                "values": "float32, identical to the source; recompressed (zlib 4, shuffle)",
                "files": {path.name: path.stat().st_size for path in written},
                "total_gb": sum(path.stat().st_size for path in written) / 1e9,
                "not_written": [d for d in done if d["state"].startswith("skipped")],
                "runtime_s": time.time() - started}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"written": len(written), "total_gb": manifest["total_gb"],
                      "skipped": len(manifest["not_written"]), "runtime_s": manifest["runtime_s"]}), flush=True)


if __name__ == "__main__":
    main()
