"""P4 — the Senseiver, run as the Senseiver, then adapted to our evaluation.

The plan is strict about this one, and rightly:

    1. reproduce the official example / published behavior first
    2. then adapt to our held-out Argo evaluation
    3. never use a homemade approximation and label it Senseiver

So this script does not reimplement anything.  It clones/uses the official
LANL implementation (`OrchardLANL/Senseiver`, the code accompanying Santos et
al., *Nature Machine Intelligence* 2023) and calls it.

The one thing that has to be built here
---------------------------------------
The official repo ships `Data/*/placeholder*.txt` -- the datasets themselves are
not in the repository.  Its NOAA loader expects `Data/NOAA/sst_weekly.mat`, a
repackaging of **NOAA OI SST V2 weekly**.  `--stage prepare` rebuilds exactly
that array from NOAA's own primary file (`sst.wkmean.1990-present.nc`,
downloaded from NOAA PSL), in the layout the official loader reads.

This is obtaining the paper's input data from its source, not approximating the
paper's model.  The distinction is the one rule 3 is about, and it is worth
being explicit: **every line of model code executed here is the authors'.**

Why the Argo adaptation looks the way it does
---------------------------------------------
The Senseiver maps sparse sensor values to a **dense field**, and it is trained
on dense field snapshots.  Our real-Argo task has no dense truth -- the targets
are point measurements from held-out floats.  So the faithful adaptation is:

    train    on a gridded T/S product (GODAS), with sensors placed at the real
             Argo positions of that month
    evaluate at the held-out floats' real positions, against their own
             measurements -- the identical queries every other row is scored on

That asymmetry is a result, not an inconvenience, and the report says so: the
Senseiver **requires** a gridded training field where the DFS model does not.
Giving it GODAS is the most favourable honest choice available, since GODAS has
assimilated Argo and so encodes more than our model ever sees.

  .venv/bin/python experiments/real_data/48_senseiver.py --stage prepare
  .venv/bin/python experiments/real_data/48_senseiver.py --stage reproduce
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np

REPO_URL = "https://github.com/OrchardLANL/Senseiver.git"
NOAA_WEEKLY = "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2/sst.wkmean.1990-present.nc"
#: the official NOAA loader hard-codes this frame count
NOAA_FRAMES = 1914

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="prepare",
                choices=["clone", "prepare", "reproduce", "argo"])
ap.add_argument("--region", default="gulfstream")
ap.add_argument("--argo-sensors", type=int, default=24,
                help="Argo columns supplied as sensors, matching the registered "
                     "24-profile operating point of the DFS rows")
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--repo", default=None, help="default: <repo>/external/Senseiver")
ap.add_argument("--data", default=None, help="default: <repo>/data/senseiver")
ap.add_argument("--gpu", type=int, default=3)
ap.add_argument("--num-sensors", type=int, default=100)
ap.add_argument("--training-frames", type=int, default=50)
ap.add_argument("--seed", type=int, default=123)
ap.add_argument("--max-epochs", type=int, default=None,
                help="cap the official trainer for a bounded reproduction run")
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO = args.repo or os.path.join(ROOT, "external", "Senseiver")
DATA = args.data or os.path.join(ROOT, "data", "senseiver")
os.makedirs(DATA, exist_ok=True)
t0 = time.time()


def clone() -> None:
    if os.path.isdir(os.path.join(REPO, ".git")):
        print(f"already cloned: {REPO}")
    else:
        os.makedirs(os.path.dirname(REPO), exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO],
                       check=True)
    sha = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"],
                                  text=True).strip()
    print(f"Senseiver at {REPO}\n  upstream commit {sha}")
    return sha


def prepare() -> dict:
    """Build Data/NOAA/sst_weekly.mat from NOAA's primary weekly SST file.

    The official loader does:

        f = h5py.File('Data/NOAA/sst_weekly.mat','r')
        sst = np.nan_to_num(np.array(f['sst']))
        sea[t,:,:,0] = sst[t,:].reshape(180, 360, order='F')

    so it wants an HDF5 dataset named ``sst`` of shape (frames, 64800), each row
    a Fortran-ordered 180x360 grid.  MATLAB v7.3 files are HDF5 and store
    arrays transposed, which is why the loader's `sst[t,:]` indexes the FIRST
    axis; h5py therefore needs the dataset written as (64800, frames) so that
    `np.array(f['sst'])` comes back (frames, 64800).
    """
    import h5py
    import xarray as xr
    # The authors' loader hard-codes num_frames = 1914.  NOAA ships the weekly
    # series as two files; concatenated, frame 1914 lands on 2018-06-24, which
    # is consistent with the snapshot the 2023 paper used.  Neither file alone
    # reaches 1914 (427 + 1727 = 2154 combined), so both are required to
    # reproduce the published frame count rather than silently training on a
    # shorter series.
    parts = []
    for fn in ("sst.wkmean.1981-1989.nc", "sst.wkmean.1990-present.nc"):
        pth = os.path.join(DATA, fn)
        if not os.path.exists(pth):
            raise SystemExit(f"missing {pth}\n  fetch from "
                             f"https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2/{fn}")
        parts.append(xr.open_dataset(pth)["sst"])
    sst = xr.concat(parts, dim="time")
    if sst.sizes["time"] < NOAA_FRAMES:
        raise SystemExit(f"only {sst.sizes['time']} weekly frames available, "
                         f"the official loader needs {NOAA_FRAMES}")
    n = NOAA_FRAMES
    a = np.asarray(sst.isel(time=slice(0, n)).values, dtype=np.float64)
    if a.shape[1:] != (180, 360):
        raise SystemExit(f"unexpected NOAA grid {a.shape[1:]}, expected (180,360)")
    # The loader reads np.array(f['sst'])[t, :] and reshapes Fortran-order, so
    # h5py must hand back (frames, 64800) -- write it that way directly.
    flat = np.stack([a[t].reshape(-1, order="F") for t in range(n)])
    out = os.path.join(REPO, "Data", "NOAA", "sst_weekly.mat")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("sst", data=flat)        # (frames, 64800)
    meta = dict(source=NOAA_WEEKLY, frames=int(n), grid=[180, 360],
                target=os.path.relpath(out, ROOT),
                time_range=[str(sst.time.values[0])[:10],
                            str(sst.time.values[n - 1])[:10]],
                note="rebuilt from NOAA's primary weekly file in the layout the "
                     "official Senseiver NOAA loader reads; no model code changed")
    with open(os.path.join(DATA, "noaa_prepare.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {out}  ({n} frames, {flat.nbytes/1e6:.0f} MB)")

    # verify through the authors' own loader, not through ours
    cwd = os.getcwd()
    try:
        os.chdir(REPO)
        sys.path.insert(0, REPO)
        import datasets as ds_mod
        sea = ds_mod.NOAA()
        print(f"official loader returns {sea.shape}, "
              f"range [{sea.min():.3f}, {sea.max():.3f}]")
        meta["loader_shape"] = list(sea.shape)
    finally:
        os.chdir(cwd)
    return meta


def reproduce(sha: str | None) -> None:
    """Run the authors' train.py on the authors' NOAA example."""
    cmd = [os.path.join(ROOT, ".venv", "bin", "python"), "train.py",
           "--gpu", str(args.gpu), "--data", "noaa",
           "--num_sensors", str(args.num_sensors),
           "--training_frames", str(args.training_frames),
           "--cons", "False", "--seed", str(args.seed),
           "--enc_preproc", "32", "--dec_num_latent_channels", "32",
           "--enc_num_latent_channels", "32", "--num_latents", "256",
           "--dec_preproc_ch", "32", "--test", "False"]
    print("running the official trainer:\n  " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
    tail = (r.stdout or "")[-4000:] + (r.stderr or "")[-4000:]
    print(tail)
    rec = dict(returncode=r.returncode, upstream_commit=sha, command=cmd,
               tail=tail[-2000:])
    with open(os.path.join(DATA, "reproduce.json"), "w") as f:
        json.dump(rec, f, indent=1)
    if r.returncode != 0:
        print("\nREPRODUCTION DID NOT COMPLETE — per the plan's rule 1, the "
              "Argo adaptation must not be reported as 'Senseiver' until the "
              "official example runs.")


# ==========================================================================
# stage 3 (plan rule 2): adapt to our held-out Argo evaluation
# ==========================================================================
ARGO_DATASET = '''
def godas_argo():
    """GODAS T/S with depth folded into channels.

    Written by ocean_project/experiments/real_data/48_senseiver.py, NOT by the Senseiver
    authors. Shape (T, H, W, 2*D): one pixel per grid cell, one channel per
    (variable, depth). An Argo float measures the WHOLE COLUMN at one location,
    which is exactly what a Senseiver sensor is -- a pixel whose every channel
    is observed. Folding depth into channels therefore uses the authors'
    standard 2-D path with no change to their model.
    """
    import numpy as np
    d = np.load("Data/GODAS_ARGO/field.npz")
    return d["field"].astype("float32")
'''

ARGO_SENSORS = '''
def argo_sensors(data, n_sensors, rnd_seed):
    """Sensor pixels at REAL Argo positions.

    Written by ocean_project, not by the Senseiver authors. The authors'
    `sea_n_sensors` picks random wet pixels; here the pixels are the grid cells
    real floats actually occupied, so the Senseiver receives the same
    observations the DFS rows get rather than an idealised array.
    """
    import numpy as np
    c = np.load("Data/GODAS_ARGO/sensors.npz")
    # The authors index `sensors[x_sens, y_sens]` on an array shaped
    # (H, W), and their own sea_n_sensors draws x from data.shape[1] and y
    # from data.shape[2] -- so "x" is the ROW and "y" is the COLUMN. Saving
    # grid_x as "x" put column indices on the row axis and raised
    # IndexError once a float sat past row 26.
    xs, ys = c["row"], c["col"]
    if n_sensors and len(xs) > n_sensors:
        rng = np.random.default_rng(rnd_seed)
        k = rng.choice(len(xs), n_sensors, replace=False)
        xs, ys = xs[k], ys[k]
    return xs, ys
'''

ARGO_BRANCH = """
    if dataset_name == 'godas_argo':
        data = godas_argo()
        x_sens, y_sens = argo_sensors(data, num_sensors, seed)
        data = torch.tensor(data)
        return data, x_sens, y_sens
"""


def build_argo_dataset():
    """Write the field and the real Argo sensor pixels the adapter reads."""
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from ocean_tokenizer.godas import load_godas, GodasNorm
    from ocean_tokenizer.argo_obs import ArgoCohort

    data_dir = os.path.join(REPO, "Data", "GODAS_ARGO")
    os.makedirs(data_dir, exist_ok=True)
    fields = load_godas(os.path.join(ROOT, "data", "godas_" + args.region))
    months = fields["months"]
    year = months.astype("datetime64[Y]").astype(int) + 1970
    tr = np.where((year >= 2000) & (year <= 2018))[0]
    norm = GodasNorm(fields, slice(int(tr[0]), int(tr[-1]) + 1))
    T = norm.z("TEMP", fields["TEMP"], months)          # (T, Z, Y, X)
    S = norm.z("SALT", fields["SALT"], months)
    fld = np.concatenate([np.moveaxis(T, 1, -1), np.moveaxis(S, 1, -1)], axis=-1)
    fld = np.nan_to_num(fld).astype("float32")          # (T, Y, X, 2*Z)
    np.savez(os.path.join(data_dir, "field.npz"), field=fld)

    c = ArgoCohort.load(os.path.join(ROOT, "data", "argo_cohort",
                                     args.region + ".nc"))
    dev_months = [int(m) for m in c.months_in("development")
                  if c.month(m, float_split="cohort_float").size]
    rows = np.concatenate([c.month(m, float_split="cohort_float")
                           for m in dev_months])
    # The Senseiver's sensor set is fixed across frames, so the union of the
    # cells real floats occupied is the honest stand-in for "where Argo is".
    cells = np.unique(np.stack([c.grid_y[rows], c.grid_x[rows]], -1), axis=0)
    np.savez(os.path.join(data_dir, "sensors.npz"),
             row=cells[:, 0].astype(int),      # grid_y -> the authors' x_sens
             col=cells[:, 1].astype(int))      # grid_x -> the authors' y_sens
    print("  field " + str(fld.shape) + " -> " + data_dir + "/field.npz")
    print("  " + str(len(cells)) + " distinct real-Argo sensor cells (from "
          + format(len(rows), ",") + " profiles over "
          + str(len(dev_months)) + " months)")
    return fld.shape, len(cells)


def patch_upstream():
    """Append two DATA functions and one dispatch branch. No model change.

    Recorded verbatim in the run json so the diff against upstream is
    auditable: every line added is dataset plumbing, and network_light.py /
    model.py are untouched.
    """
    added = {}
    for fname, block, marker in (("datasets.py", ARGO_DATASET, "def godas_argo"),
                                 ("sensor_loc.py", ARGO_SENSORS, "def argo_sensors")):
        path_ = os.path.join(REPO, fname)
        txt = open(path_).read()
        if marker not in txt:
            open(path_, "a").write("\n" + block)
            added[fname] = block
    path_ = os.path.join(REPO, "dataloaders.py")
    txt = open(path_).read()
    if "godas_argo" not in txt:
        txt = txt.replace(
            "from datasets import cylinder, NOAA, pipe, plume, porous",
            "from datasets import cylinder, NOAA, pipe, plume, porous, godas_argo")
        txt = txt.replace(
            "from sensor_loc import ( cylinder_16_sensors, ",
            "from sensor_loc import ( argo_sensors,\n                         cylinder_16_sensors, ")
        anchor = "def load_data(dataset_name, num_sensors, seed=123):\n    \n"
        assert anchor in txt, "upstream load_data signature changed"
        txt = txt.replace(anchor, anchor + ARGO_BRANCH)
        open(path_, "w").write(txt)
        added["dataloaders.py"] = ARGO_BRANCH
    return added


def run_argo(sha):
    shape, n_cells = build_argo_dataset()
    added = patch_upstream()
    cmd = [os.path.join(ROOT, ".venv", "bin", "python"), "train.py",
           "--gpu", str(args.gpu), "--data", "godas_argo",
           "--num_sensors", str(args.argo_sensors),
           "--training_frames", "200", "--cons", "False",
           "--seed", str(args.seed),
           "--enc_preproc", "32", "--dec_num_latent_channels", "32",
           "--enc_num_latent_channels", "32", "--num_latents", "256",
           "--dec_preproc_ch", "32", "--test", "False"]
    print("running the authors' trainer on the Argo-sensored field:\n  "
          + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
    tail = (r.stdout or "")[-3000:] + (r.stderr or "")[-3000:]
    print(tail[-2500:])
    rec = dict(stage="argo", region=args.region, upstream_commit=sha,
               field_shape=list(shape), n_sensor_cells=n_cells,
               num_sensors=args.argo_sensors, returncode=r.returncode,
               upstream_additions=added, command=cmd, tail=tail[-2000:],
               note=("Only dataset plumbing was appended upstream; "
                     "network_light.py and model.py are untouched."))
    with open(os.path.join(DATA, "argo_stage.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print("\nrecord: " + os.path.join(DATA, "argo_stage.json"))


sha = clone()
if args.stage in ("prepare", "reproduce"):
    prepare()
if args.stage == "reproduce":
    reproduce(sha)
if args.stage == "argo":
    run_argo(sha)
print(f"total {time.time()-t0:.0f}s")
