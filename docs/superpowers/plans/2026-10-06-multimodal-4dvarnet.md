# Multi-modal Comparison with 4DVarNet — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the synthetic Argo cohort satellite-type fields, run 4DVarNet and our model with and without them, and put every method in one table of the synthetic report.

**Architecture:** A script builds a 1° monthly surface store (SST, SSS, steric-height sea level) from CESM2. A small numpy module bins profiles onto the grid and samples a grid at points. A driver trains 4DVarNet — the unmodified solver, prior and gradient model of `4dvarnet-starter`, imported from a pinned clone in `external/` — on Argo targets only, and scores it on the shared evaluation sets. `62_sanity_train.py` reads the new store through its existing `--surface` path. The report generator gains one section.

**Tech Stack:** Python 3.12, numpy, xarray/zarr, gsw (TEOS-10), PyTorch 2.13, kornia (import dependency of the starter), pytest.

**Spec:** `docs/superpowers/specs/2026-10-06-multimodal-4dvarnet-design.md`

**Conventions**

- Commands run from the repository root on branch `multimodal-4dvarnet`; Python is `./.venv/bin/python`.
- Commits are authored by the repo-local identity and end with the trailer `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- "Test year" is the split the summaries call `development` (2005); validation is 2004.
- GPU jobs take an explicit idle card from a fresh `nvidia-smi` reading (`CUDA_VISIBLE_DEVICES=<n>`).

## File structure

| File | Responsibility |
|---|---|
| `external/4dvarnet-starter/` (git-ignored clone, commit `20f1b5f`) | Upstream 4DVarNet, unmodified. |
| `experiments/synthetic/46_synth_surface_fields.py` (new) | Build `data/synthetic_argo/cesm2_surface_1deg.zarr`. |
| `src/ocean_tokenizer/gridded_obs.py` (new) | `bin_profiles`, `sample_bilinear`. Pure numpy. |
| `tests/test_gridded_obs.py` (new) | Tests of the above. |
| `experiments/synthetic/47_synth_argo_4dvarnet.py` (new) | 4DVarNet driver: data, training on Argo targets, scoring, JSON. |
| `experiments/real_data/62_sanity_train.py` (modify) | Surface store by region; synthetic accepts `--surface`. |
| `experiments/real_data/run_audit_queue.py` (modify) | Queue `syn_surface_k15_l64`. |
| `experiments/synthetic/43_synth_argo_report.py` (modify) | Section 12 and finding 12. |
| `docs/baseline_survey.md` (new) | Candidate baselines, papers, code. |
| `README.md` (modify) | Clone step and commands. |

---

### Task 1: Upstream clone and its one extra dependency

**Files:** none tracked (`external/` is git-ignored).

- [ ] **Step 1: Clone the starter at the pinned commit**

```bash
mkdir -p external && git clone -q https://github.com/CIA-Oceanix/4dvarnet-starter.git external/4dvarnet-starter
git -C external/4dvarnet-starter checkout -q 20f1b5f34b201342cde6dd21a30419d07541db54
git -C external/4dvarnet-starter rev-parse HEAD && git status --short | wc -l
```
Expected: the SHA `20f1b5f34b201342cde6dd21a30419d07541db54`, and `0` (the clone is ignored).

- [ ] **Step 2: Install kornia and check the upstream classes import**

`external/4dvarnet-starter/src/models.py` imports `kornia.filters` at module level.

```bash
./.venv/bin/pip install -q kornia
./.venv/bin/python -c "
import sys; sys.path.insert(0, 'external/4dvarnet-starter')
from src.models import BaseObsCost, BilinAEPriorCost, ConvLstmGradModel, GradSolver
import kornia, torch; print('ok', kornia.__version__, torch.__version__)"
```
Expected: `ok <kornia version> 2.13.0+cu130`.

---

### Task 2: Surface store

**Files:**
- Create: `experiments/synthetic/46_synth_surface_fields.py`
- Output (not tracked, `data/` is git-ignored): `data/synthetic_argo/cesm2_surface_1deg.zarr`

Facts about `data/cesm2_le_full_standard.zarr`: 72 months (2000-01 … 2005-12, a cftime `time`), `lat` −89.5…89.5, `lon` 0.5…359.5, 60 depth levels; `SST` and `SSS` are the model's 5 m level (attribute `SST_note`), NaN on land; the deepest wet level of 29 columns holds a `TEMP = 0, SALT = 0` fill, which `41_synth_argo_cohort.py` removes.

- [ ] **Step 1: Write the script**

Create `experiments/synthetic/46_synth_surface_fields.py`:

```python
"""Surface fields for the synthetic Argo cohort: SST, SSS and a sea-level stand-in.

The satellite-type inputs of the synthetic multi-modal comparison, one 1-degree
field a month for 2000-01 ... 2005-12, written with the variable names of the
real satellite store so `62_sanity_train.py --region synthetic --surface` reads
them through its existing path.

  SST, SSS   CESM2's own surface fields (its 5 m level)
  SLA        steric height relative to 990 dbar, TEOS-10, from CESM2's TEMP and
             SALT on the cohort's 20 levels (`ocean_tokenizer.ssh`); NaN where
             the column does not reach the reference level

Each field is an anomaly against its own train-year (2000-2003) monthly
climatology, the way the cohort's T/S target is defined.

Two limitations, to be repeated wherever a result uses this store:

* SST and SSS are noise-free and ARE the 5 m level of the truth, so a method
  given them is handed the answer at the shallowest level.
* SLA is a vertical integral of the T and S being reconstructed: a derived,
  optimistic stand-in for altimetry.

  .venv/bin/python experiments/synthetic/46_synth_surface_fields.py     # ~2 min CPU
"""
from __future__ import annotations

import os, shutil, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr

from ocean_tokenizer.ssh import P_REF_DBAR, steric_height_columns

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STORE = os.path.join(ROOT, "data", "cesm2_le_full_standard.zarr")
COHORT = os.path.join(ROOT, "data", "synthetic_argo", "cesm2_uniform.nc")
OUT = os.path.join(ROOT, "data", "synthetic_argo", "cesm2_surface_1deg.zarr")
TRAIN = (2000, 2003)
VARS = ("SST", "SLA", "SSS")
t0 = time.time()

LEVELS = np.asarray(xr.open_dataset(COHORT)["level"].values, dtype=float)
ds = xr.open_zarr(STORE)
di = [int(np.argmin(np.abs(ds.depth.values - d))) for d in LEVELS]
assert np.allclose(ds.depth.values[di], LEVELS, atol=0.2), ds.depth.values[di]
lat, lon = ds.lat.values.astype(float), ds.lon.values.astype(float)
NT = ds.sizes["time"]
assert NT == 72, NT
T = ds["TEMP"].isel(depth=di).values.astype("float32")        # (NT, L, NY, NX)
S = ds["SALT"].isel(depth=di).values.astype("float32")
fill = (T == 0) & (S == 0)                                    # the regridding fill 41 removes
T, S = np.where(fill, np.nan, T), np.where(fill, np.nan, S)
month_index = np.arange(NT)                                   # months since 2000-01
years, cal = 2000 + month_index // 12, month_index % 12
train = (years >= TRAIN[0]) & (years <= TRAIN[1])
print(f"loaded {NT} months x {LEVELS.size} levels ({time.time() - t0:.0f}s)", flush=True)

raw = {"SST": ds["SST"].values.astype("float32"), "SSS": ds["SSS"].values.astype("float32")}
lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
sla = np.full((NT, lat.size, lon.size), np.nan, dtype="float32")
for t in range(NT):
    # a column is used only where every level is finite, i.e. it reaches the reference
    ii, jj = np.where(np.isfinite(T[t]).all(axis=0) & np.isfinite(S[t]).all(axis=0))
    sla[t, ii, jj] = steric_height_columns(T[t][:, ii, jj], S[t][:, ii, jj], LEVELS,
                                           lat2d[ii, jj], lon2d[ii, jj])
raw["SLA"] = sla
print(f"steric height done ({time.time() - t0:.0f}s)", flush=True)


def anomaly(a):
    """a - its train-year monthly climatology; a is (NT, ...)."""
    clim = np.stack([np.nanmean(a[train & (cal == m)], axis=0) for m in range(12)])
    return (a - clim[cal]).astype("float32")


anom = {v: anomaly(raw[v]) for v in VARS}

# ---- diagnostics: what the fields contain
wet = np.isfinite(raw["SST"][0])
t_anom = anomaly(T)
band = (LEVELS > 100) & (LEVELS <= 300)
t_band = np.nanmean(t_anom[:, band], axis=1)                  # (NT, NY, NX)


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


print(f"wet cells {int(wet.sum()):,}; SLA defined on "
      f"{100 * np.isfinite(sla[0]).sum() / wet.sum():.0f} % of them")
for v in VARS:
    print(f"  {v}: anomaly std over train months {np.nanstd(anom[v][train]):.4f}, "
          f"finite on {int(np.isfinite(anom[v][0]).sum()):,} cells")
print(f"  5 m check: max |SST - TEMP(5 m)| = {np.nanmax(np.abs(raw['SST'] - T[:, 0])):.2e}, "
      f"max |SSS - SALT(5 m)| = {np.nanmax(np.abs(raw['SSS'] - S[:, 0])):.2e}")
print(f"  corr(SLA anomaly, 100-300 m TEMP anomaly) = {corr(anom['SLA'], t_band):.3f}; "
      f"corr(SLA anomaly, SST anomaly) = {corr(anom['SLA'], anom['SST']):.3f}")

out = xr.Dataset(
    {v: (("time", "lat", "lon"), anom[v]) for v in VARS},
    coords={"time": np.arange("2000-01", "2006-01", dtype="datetime64[M]").astype("datetime64[ns]"),
            "lat": lat.astype("float32"), "lon": lon.astype("float32")},
    attrs={"source": "CESM2-LE 1 degree, data/cesm2_le_full_standard.zarr",
           "role": "satellite-type inputs of the synthetic Argo cohort",
           "anomaly": f"against each field's {TRAIN[0]}-{TRAIN[1]} monthly climatology",
           "SST_SSS": "the model's 5 m level, noise-free: the truth at the cohort's shallowest level",
           "SLA": f"steric height relative to {P_REF_DBAR:.0f} dbar from TEMP/SALT on the cohort's "
                  "20 levels (ocean_tokenizer.ssh); derived from the truth, not an independent "
                  "observation; NaN where the column does not reach the reference"})
if os.path.exists(OUT):
    shutil.rmtree(OUT)
out.to_zarr(OUT, mode="w")
print(f"wrote {OUT} ({time.time() - t0:.0f}s)")
```

- [ ] **Step 2: Run it**

Run: `CUDA_VISIBLE_DEVICES="" ./.venv/bin/python experiments/synthetic/46_synth_surface_fields.py`
Expected: `loaded 72 months x 20 levels`, `steric height done`, a coverage line, three `anomaly std` lines, a `5 m check` line with both maxima at or near 0, a correlation line whose first value is clearly positive and larger than the second, and `wrote …/cesm2_surface_1deg.zarr`.

- [ ] **Step 3: Check the store reads the way 62 reads it**

```bash
./.venv/bin/python -c "
import xarray as xr, numpy as np
o = xr.open_zarr('data/synthetic_argo/cesm2_surface_1deg.zarr')
t = o.time.values.astype('datetime64[M]')
mi = ((t - np.datetime64('2000-01', 'M')) / np.timedelta64(1, 'M')).astype(int)
print(dict(o.sizes), list(o.data_vars), 'months', mi[0], mi[-1])"
```
Expected: `{'time': 72, 'lat': 180, 'lon': 360} ['SLA', 'SSS', 'SST'] months 0 71` (variable order may differ).

- [ ] **Step 4: Commit**

```bash
git add experiments/synthetic/46_synth_surface_fields.py
git commit -q -m "46_synth_surface_fields: SST, SSS and steric-height sea level for the synthetic cohort

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Gridding and point sampling

**Files:**
- Create: `src/ocean_tokenizer/gridded_obs.py`
- Test: `tests/test_gridded_obs.py`

The cohort stores each profile's cell as `grid_y = floor(lat + 90)`, `grid_x = floor(lon mod 360)`; cell centres are at −89.5…89.5 and 0.5…359.5. `41_synth_argo_cohort.py` samples CESM2 at a profile with a bilinear stencil on the cell centres, periodic in longitude; `sample_bilinear` is that stencil.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gridded_obs.py`:

```python
"""Profiles onto the 1-degree grid, and a grid back onto points."""
import numpy as np

from ocean_tokenizer.gridded_obs import bin_profiles, sample_bilinear


# --------------------------------------------------------------------------
# bin_profiles
# --------------------------------------------------------------------------
def test_bin_profiles_averages_per_cell_and_level():
    gy = np.array([2, 2, 0]); gx = np.array([3, 3, 1])
    v = np.array([[1.0, 10.0], [3.0, np.nan], [5.0, 6.0]])      # (n=3, L=2)
    g = bin_profiles(gy, gx, v, (4, 5))
    assert g.shape == (2, 4, 5)
    assert g[0, 2, 3] == 2.0                # mean of 1 and 3
    assert g[1, 2, 3] == 10.0               # the NaN level of the second profile is skipped
    assert g[0, 0, 1] == 5.0 and g[1, 0, 1] == 6.0
    assert np.isnan(g).sum() == 2 * 4 * 5 - 4      # every other cell is empty


def test_bin_profiles_leaves_a_cell_empty_when_all_its_values_are_missing():
    g = bin_profiles(np.array([1]), np.array([1]), np.array([[np.nan, 2.0]]), (3, 3))
    assert np.isnan(g[0, 1, 1]) and g[1, 1, 1] == 2.0


def test_bin_profiles_without_profiles_is_all_empty():
    g = bin_profiles(np.array([], dtype=int), np.array([], dtype=int), np.zeros((0, 3)), (2, 2))
    assert g.shape == (3, 2, 2) and np.isnan(g).all()


# --------------------------------------------------------------------------
# sample_bilinear
# --------------------------------------------------------------------------
def _field():
    lat = np.arange(180) - 89.5
    lon = np.arange(360) + 0.5
    # two channels: one linear in latitude, one linear in longitude
    return np.stack([np.repeat(lat[:, None], 360, axis=1),
                     np.repeat(lon[None, :], 180, axis=0)])


def test_sample_bilinear_is_exact_at_cell_centres():
    out = sample_bilinear(_field(), np.array([-0.5, 10.5]), np.array([0.5, 200.5]))
    assert out.shape == (2, 2)
    assert np.allclose(out, [[-0.5, 0.5], [10.5, 200.5]])


def test_sample_bilinear_interpolates_between_centres():
    out = sample_bilinear(_field(), np.array([0.25]), np.array([10.9]))
    assert np.allclose(out, [[0.25, 10.9]])


def test_sample_bilinear_is_periodic_in_longitude():
    f = np.zeros((1, 180, 360)); f[0, :, 0] = 2.0; f[0, :, 359] = 4.0
    # 0.0 E lies midway between the centres at 359.5 E and 0.5 E
    assert np.allclose(sample_bilinear(f, np.array([0.5]), np.array([0.0])), [[3.0]])
    assert np.allclose(sample_bilinear(f, np.array([0.5]), np.array([360.0])), [[3.0]])
    assert np.allclose(sample_bilinear(f, np.array([0.5]), np.array([359.75])), [[3.5]])


def test_sample_bilinear_clamps_at_the_poles():
    out = sample_bilinear(_field(), np.array([89.9, -89.9]), np.array([50.5, 50.5]))
    assert np.allclose(out[:, 0], [89.5, -89.5])


def test_binning_then_sampling_returns_a_profile_at_its_cell_centre():
    gy, gx = np.array([100]), np.array([40])
    g = np.nan_to_num(bin_profiles(gy, gx, np.array([[7.0, -3.0]]), (180, 360)))
    assert np.allclose(sample_bilinear(g, np.array([10.5]), np.array([40.5])), [[7.0, -3.0]])
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" ./.venv/bin/python -m pytest tests/test_gridded_obs.py -q 2>&1 | tail -3`
Expected: collection error, `No module named 'ocean_tokenizer.gridded_obs'`.

- [ ] **Step 3: Implement**

Create `src/ocean_tokenizer/gridded_obs.py`:

```python
"""Profiles onto the 1-degree grid, and a grid back onto points.

A gridded method (4DVarNet) needs scattered profiles as a field with gaps, and
its gridded answer has to be scored at the held-out profiles' own positions.
Both directions use the cohort's conventions: cell ``(grid_y, grid_x)`` =
``(floor(lat + 90), floor(lon mod 360))`` with centres at -89.5 ... 89.5 and
0.5 ... 359.5, and the bilinear stencil of
``experiments/synthetic/41_synth_argo_cohort.py``, periodic in longitude.
"""
from __future__ import annotations

import numpy as np


def bin_profiles(grid_y, grid_x, values, shape):
    """Mean of the profiles in each cell, level by level.

    ``values`` is ``(n, L)`` with NaN where a profile has no value. Returns
    ``(L, NY, NX)``, NaN in a cell that received no finite value at that level.
    """
    NY, NX = shape
    values = np.asarray(values, dtype=np.float64)
    flat = np.asarray(grid_y, dtype=np.int64) * NX + np.asarray(grid_x, dtype=np.int64)
    out = np.full((values.shape[1], NY * NX), np.nan)
    for l in range(values.shape[1]):
        ok = np.isfinite(values[:, l])
        n = np.bincount(flat[ok], minlength=NY * NX)
        s = np.bincount(flat[ok], weights=values[ok, l], minlength=NY * NX)
        m = n > 0
        out[l, m] = s[m] / n[m]
    return out.reshape(values.shape[1], NY, NX)


def sample_bilinear(field, lat, lon, lat0=-89.5, lon0=0.5):
    """``(C, NY, NX)`` on the 1-degree cell centres -> ``(P, C)`` at the points.

    Periodic in longitude; latitude is clamped to the first and last centre.
    """
    field = np.asarray(field)
    _, NY, NX = field.shape
    fy = np.clip(np.asarray(lat, dtype=np.float64) - lat0, 0.0, NY - 1.0)
    fx = (np.asarray(lon, dtype=np.float64) - lon0) % 360.0
    i0 = np.minimum(np.floor(fy).astype(int), NY - 1)
    i1 = np.minimum(i0 + 1, NY - 1)
    j0 = np.floor(fx).astype(int) % NX
    j1 = (j0 + 1) % NX
    wy, wx = fy - i0, fx - np.floor(fx)
    out = (field[:, i0, j0] * ((1 - wy) * (1 - wx)) + field[:, i0, j1] * ((1 - wy) * wx)
           + field[:, i1, j0] * (wy * (1 - wx)) + field[:, i1, j1] * (wy * wx))
    return out.T
```

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" ./.venv/bin/python -m pytest tests/test_gridded_obs.py -q 2>&1 | tail -3`
Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/ocean_tokenizer/gridded_obs.py tests/test_gridded_obs.py
git commit -q -m "gridded_obs: profiles onto the 1-degree grid and a grid back onto points

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: The 4DVarNet driver

Upstream, at commit `20f1b5f` (`external/4dvarnet-starter`):

- `src/models.py`: `GradSolver(prior_cost, obs_cost, grad_mod, n_step, lr_grad)`; its `forward(batch)` reads `batch.input` (NaN where unobserved), starts from `batch.input.nan_to_num()`, runs `n_step` updates, and in eval mode ends with `prior_cost.forward_ae(state)`. `BaseObsCost()(state, batch)` is the MSE on the finite cells of `batch.input`. `BilinAEPriorCost(dim_in, dim_hidden, downsamp, bilin_quad)`, `ConvLstmGradModel(dim_in, dim_hidden)`.
- `config/xp/base.yaml`: `n_step: 10`, `lr_grad: 1e3`, prior `dim_hidden: 32, bilin_quad: False, downsamp: 2`, gradient model `dim_hidden: 48`, `max_epochs: 150`, `batch_size: 4`, `gradient_clip_val: 0.5`, optimiser `cosanneal_lr_adam` with `lr: 1e-3`.
- `src/utils.py::cosanneal_lr_adam`: Adam with the learning rate on the gradient model and the observation term and half of it on the prior, cosine annealing over the epochs.
- `Lit4dVarNet.step`: loss `50 * mse + 1000 * sobel-gradient mse + 1 * prior_cost(output)`.
- `contrib/multimodal.MultiModalObsCost`: `BaseObsCost` plus `mse(conv(state), conv(field.nan_to_num()))` with two bias-free 3 × 3 convolutions to `dim_hidden: 5` channels.

**Files:**
- Create: `experiments/synthetic/47_synth_argo_4dvarnet.py`

- [ ] **Step 1: Write the driver**

Create `experiments/synthetic/47_synth_argo_4dvarnet.py`:

```python
"""4DVarNet baseline on the synthetic Argo cohort, Argo-only or with surface fields.

4DVarNet (Fablet et al. 2021) learns a variational interpolation: a prior and a
gradient-based solver are trained together so that a few solver steps turn
gappy gridded observations into a complete field. The solver, the prior and
the gradient model are imported UNMODIFIED from the authors' `4dvarnet-starter`
(https://github.com/CIA-Oceanix/4dvarnet-starter, CeCILL-C), cloned at a pinned
commit into the git-ignored `external/` folder, with the settings of its
`config/xp/base.yaml`. What is ours is how the cohort is put in front of it:

  state          TEMP and SALT z-anomalies on the 20 levels = 40 channels on the
                 global 1-degree grid, one month at a time
  observations   the month's input profiles binned to their cell (mean), NaN
                 elsewhere; padded circularly in longitude
  --surface      adds SST, SSS and SLA (46_synth_surface_fields.py) through an
                 observation term modelled on the starter's
                 contrib/multimodal.MultiModalObsCost
  targets        Argo profiles ONLY: each time a training month is drawn, 30 %
                 of its input profiles are held out and gridded as the target,
                 the rule of 62_sanity_train.py. The starter trains on a dense
                 field and adds a Sobel-gradient loss; neither exists here, so
                 the loss is its 50 x MSE on the cells that have a target plus
                 its prior cost
  selection      after each epoch the validation year is scored as the test
                 year will be; the weights with the lowest macro z are kept
  solver output  the state after the solver's steps, in training and in scoring.
                 Out of training the starter's `GradSolver.forward` also passes
                 that state through the prior's auto-encoder; on this cohort the
                 projection, which the training loss never sees, returns a field
                 no better than climatology (validation J 1.00 against 0.41
                 without it, same weights). The score along the starter's own
                 evaluation path is recorded next to the one used
  scoring        all 6,080 input profiles of the month in, the gridded answer
                 sampled bilinearly at the 1,520 query profiles, pooled on the
                 evaluation sets of ocean_tokenizer.synth_argo_eval after the
                 input-parity assertion and the zero-prediction identity check

Writes outputs/audit/synthetic/fourdvarnet[_surface]/summary_seed<seed>.json and
the selected weights.

  .venv/bin/python experiments/synthetic/47_synth_argo_4dvarnet.py                 # Argo only
  .venv/bin/python experiments/synthetic/47_synth_argo_4dvarnet.py --surface
  .venv/bin/python experiments/synthetic/47_synth_argo_4dvarnet.py --smoke         # prints only
"""
from __future__ import annotations

import argparse, collections, copy, hashlib, json, os, subprocess, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXT = os.path.join(ROOT, "external", "4dvarnet-starter")
STARTER_COMMIT = "20f1b5f34b201342cde6dd21a30419d07541db54"
sys.path.insert(0, os.path.join(ROOT, "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr


def _starter_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=EXT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


if _starter_commit() != STARTER_COMMIT:
    raise SystemExit(
        f"expected 4dvarnet-starter at {STARTER_COMMIT[:7]} in external/4dvarnet-starter:\n"
        "  git clone https://github.com/CIA-Oceanix/4dvarnet-starter.git external/4dvarnet-starter\n"
        f"  git -C external/4dvarnet-starter checkout {STARTER_COMMIT}")
sys.path.insert(0, EXT)
from src.models import BaseObsCost, BilinAEPriorCost, ConvLstmGradModel, GradSolver  # noqa: E402

from ocean_tokenizer import synth_argo_eval  # noqa: E402
from ocean_tokenizer.audit_tools import cohort_path  # noqa: E402
from ocean_tokenizer.gridded_obs import bin_profiles, sample_bilinear  # noqa: E402
from ocean_tokenizer.point_baselines import CH, Scores  # noqa: E402
from ocean_tokenizer.synth_argo_eval import (EVAL_CELLS, N_INPUT, N_QUERY, REFERENCE,  # noqa: E402
                                             SPLITS, identity_check)

# ---- the starter's base configuration (config/xp/base.yaml, utils.cosanneal_lr_adam)
N_STEP, LR_GRAD = 10, 1e3
PRIOR_HIDDEN, PRIOR_DOWNSAMP, GRAD_HIDDEN = 32, 2, 48
EPOCHS, BATCH, LR, CLIP = 150, 4, 1e-3, 0.5
W_MSE, W_PRIOR = 50.0, 1.0          # Lit4dVarNet.step, without its Sobel-gradient term
SURF_HIDDEN = 5                     # config/params/multimodal.yaml
# ---- ours
NY, NX = 180, 360
PAD = 8                             # circular longitude padding, in cells
TARGET_FRAC = 0.3                   # input profiles held out as targets (62)
SURF_VARS = ("SST", "SSS", "SLA")
SURF_STORE = os.path.join("data", "synthetic_argo", "cesm2_surface_1deg.zarr")

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=1234,
                help="target draws, month order and initial weights")
ap.add_argument("--surface", action="store_true",
                help="also give SST, SSS and SLA of the month")
ap.add_argument("--epochs", type=int, default=EPOCHS)
ap.add_argument("--out", default=None,
                help="default: outputs/audit/synthetic/fourdvarnet[_surface]")
ap.add_argument("--smoke", action="store_true",
                help="4 training months, 2 epochs, 1 validation month, no identity "
                     "check; prints and writes nothing")
args = ap.parse_args()
TAG = "fourdvarnet_surface" if args.surface else "fourdvarnet"
OUT = args.out or os.path.join(ROOT, "outputs", "audit", "synthetic", TAG)
dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()

# ------------------------------------------------------------------ data
c, norm, OBS = synth_argo_eval.load(ROOT)
LEV = c.levels
L = LEV.size
C_STATE = len(CH) * L               # TEMP levels, then SALT levels
Item = collections.namedtuple("Item", ["input", "tgt", "surface"])


def grid_profiles(rows):
    """(2L, NY, NX) float32: the profiles `rows` binned to their cells, NaN elsewhere."""
    return np.concatenate([bin_profiles(c.grid_y[rows], c.grid_x[rows], OBS[ch][rows], (NY, NX))
                           for ch in CH]).astype("float32")


SURF = None
if args.surface:
    _o = xr.open_zarr(os.path.join(ROOT, SURF_STORE))
    _t = _o.time.values.astype("datetime64[M]")
    _mi = ((_t - np.datetime64("2000-01", "M")) / np.timedelta64(1, "M")).astype(int)
    _a = np.stack([_o[v].values for v in SURF_VARS], axis=1).astype("float32")   # (T, 3, NY, NX)
    _tr = np.array([SPLITS["train"][0] <= 2000 + m // 12 <= SPLITS["train"][1] for m in _mi])
    # z-scored per field on the train years, as 62 scales the satellite fields
    _mu = np.nanmean(_a[_tr], axis=(0, 2, 3), keepdims=True)
    _sd = np.nanstd(_a[_tr], axis=(0, 2, 3), keepdims=True)
    SURF = {int(m): (_a[i] - _mu[0]) / _sd[0] for i, m in enumerate(_mi)}        # NaN kept


def pad_lon(x):
    """Wrap PAD columns around in longitude (last dim), keeping NaN."""
    return torch.cat([x[..., -PAD:], x, x[..., :PAD]], dim=-1)


def crop_lon(x):
    return x[..., PAD:-PAD]


def batch_of(inputs, months, targets=None):
    """An Item on the device, longitude-padded. inputs/targets: lists of (2L, NY, NX)."""
    t = lambda a: pad_lon(torch.from_numpy(np.stack(a)).to(dev))
    return Item(input=t(inputs), tgt=None if targets is None else t(targets),
                surface=None if SURF is None else t([SURF[int(m)] for m in months]))


val = synth_argo_eval.eval_set(c, OBS, "validation", EVAL_CELLS,
                               n_months=1 if args.smoke else None)
test = [] if args.smoke else synth_argo_eval.eval_set(c, OBS, "development")
print(f"{TAG} seed={args.seed} device={dev}: {len(val)} validation months, {len(test)} "
      f"test months, {N_INPUT} inputs and {N_QUERY} queries a month, state {C_STATE} x "
      f"{NY} x {NX} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ model
class SurfaceObsCost(nn.Module):
    """The profile observation term plus one tying the state to the surface fields.

    After contrib/multimodal.MultiModalObsCost of the starter: the MSE between a
    learned bias-free 3 x 3 convolution of the state and one of the fields, the
    fields zero-filled where missing. Rewritten because the original builds both
    convolutions for one channel count; here the state has 2L channels and the
    fields three.
    """

    def __init__(self, dim_state, dim_surface, dim_hidden):
        super().__init__()
        self.base_cost = BaseObsCost()
        self.conv_state = nn.Conv2d(dim_state, dim_hidden, (3, 3), padding=1, bias=False)
        self.conv_surface = nn.Conv2d(dim_surface, dim_hidden, (3, 3), padding=1, bias=False)

    def forward(self, state, batch):
        return self.base_cost(state, batch) + F.mse_loss(
            self.conv_state(state), self.conv_surface(batch.surface.nan_to_num()))


torch.manual_seed(args.seed)
solver = GradSolver(
    prior_cost=BilinAEPriorCost(dim_in=C_STATE, dim_hidden=PRIOR_HIDDEN,
                                bilin_quad=False, downsamp=PRIOR_DOWNSAMP),
    obs_cost=(SurfaceObsCost(C_STATE, len(SURF_VARS), SURF_HIDDEN) if args.surface
              else BaseObsCost()),
    grad_mod=ConvLstmGradModel(dim_in=C_STATE, dim_hidden=GRAD_HIDDEN),
    n_step=N_STEP, lr_grad=LR_GRAD).to(dev)
n_par = sum(p.numel() for p in solver.parameters())
groups = [{"params": list(solver.grad_mod.parameters()), "lr": LR},
          {"params": list(solver.obs_cost.parameters()), "lr": LR},
          {"params": list(solver.prior_cost.parameters()), "lr": LR / 2}]
epochs = 2 if args.smoke else args.epochs
opt = torch.optim.Adam([g for g in groups if g["params"]])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
print(f"  {n_par:,} parameters; starter {STARTER_COMMIT[:7]}; {epochs} epochs", flush=True)


# ------------------------------------------------------------------ scoring
def solve(model, batch, upstream_eval=False):
    """The model's answer for a batch, out of training.

    ``upstream_eval=True`` is the starter's `GradSolver.forward` in eval mode:
    the solver steps, then the prior's auto-encoder applied to the final state.
    The default stops before that projection and returns the state itself, the
    quantity the training loss is computed on.
    """
    model.eval()
    if upstream_eval:
        return model(batch)
    with torch.set_grad_enabled(True):
        state = model.init_state(batch)
        model.grad_mod.reset_state(batch.input)
        for step in range(model.n_step):
            state = model.solver_step(state, batch, step=step).detach().requires_grad_(True)
    return state


def score(evs, model=None, upstream_eval=False):
    """Pooled scores on the months' cells; ``model=None`` scores a zero prediction.
    The observations of a scored month are ALL of its input profiles."""
    s = Scores(LEV, norm.std)
    for ev in evs:
        if model is None:
            pred = {ch: np.zeros(ev["lev"].size) for ch in CH}
        else:
            out = crop_lon(solve(model, batch_of([grid_profiles(ev["src"])], [ev["month"]]),
                                 upstream_eval))[0]
            q = sample_bilinear(out.detach().cpu().numpy().astype(np.float64),
                                c.lat[ev["tgt"]], c.lon[ev["tgt"]])          # (R, 2L)
            pred = {ch: q[:, k * L:(k + 1) * L][ev["prof"], ev["lev"]] for k, ch in enumerate(CH)}
        for ch in CH:
            s.add(ch, pred[ch], ev["target"][ch], ev["lev"])
    return s.result()


ident = {}
if not args.smoke:
    ident = {split: identity_check(ROOT, split, score(evs))
             for split, evs in (("validation", val), ("development", test))}
    print(f"identity check passed: zero prediction matches {REFERENCE} on both splits",
          flush=True)

# ------------------------------------------------------------------ train
rng = np.random.default_rng(args.seed)
train_months = [int(m) for m in c.months_in("train")]
if args.smoke:
    train_months = train_months[:4]
for m in train_months:
    if c.month(m, float_split="cohort_float").size != N_INPUT:
        raise SystemExit(f"input parity violated in training month {m}")


def train_pair(m):
    """Observations and target of one draw: 70 % / 30 % of the month's input profiles."""
    pool = c.month(m, float_split="cohort_float")
    tgt = np.sort(rng.choice(pool, int(round(TARGET_FRAC * pool.size)), replace=False))
    return grid_profiles(np.setdiff1d(pool, tgt)), grid_profiles(tgt)


best = {"macro_z": float("inf"), "epoch": -1, "state": None}
history = []
for ep in range(epochs):
    solver.train()
    order = [train_months[i] for i in rng.permutation(len(train_months))]
    tot, nb = 0.0, 0
    for b in range(0, len(order), BATCH):
        months = order[b:b + BATCH]
        pairs = [train_pair(m) for m in months]
        batch = batch_of([p[0] for p in pairs], months, [p[1] for p in pairs])
        out = solver(batch)
        m = batch.tgt.isfinite()
        mse = F.mse_loss(out[m], batch.tgt[m])
        loss = W_MSE * mse + W_PRIOR * solver.prior_cost(out)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(solver.parameters(), CLIP)
        opt.step()
        tot += float(mse); nb += 1
    sched.step()
    sc = score(val, solver)
    history.append({"epoch": ep + 1, "train_mse": tot / nb, "val_macro_z": sc["macro_z"],
                    "val_TEMP_z": sc["TEMP"]["rmse_z"], "val_SALT_z": sc["SALT"]["rmse_z"]})
    star = ""
    if sc["macro_z"] < best["macro_z"]:
        best = {"macro_z": sc["macro_z"], "epoch": ep + 1,
                "state": copy.deepcopy(solver.state_dict())}
        star = " *"
    if (ep + 1) % 10 == 0 or ep < 3 or ep + 1 == epochs:
        print(f"  epoch {ep + 1:3d}/{epochs}  train mse {tot / nb:.4f}  validation TEMP "
              f"{sc['TEMP']['rmse_z']:.3f}z SALT {sc['SALT']['rmse_z']:.3f}z{star}  "
              f"({time.time() - t0:.0f}s)", flush=True)

solver.load_state_dict(best["state"])
scores = {"validation": score(val, solver)}
scores_upstream = {"validation": score(val, solver, upstream_eval=True)}
if test:
    scores["development"] = score(test, solver)
    scores_upstream["development"] = score(test, solver, upstream_eval=True)
for split, sc in scores.items():
    print(f"{split}: TEMP {sc['TEMP']['rmse_physical']:.4f} degC (J {sc['TEMP']['J']:.3f})  "
          f"SALT {sc['SALT']['rmse_physical']:.4f} PSU (J {sc['SALT']['J']:.3f})  "
          f"[best epoch {best['epoch']}]", flush=True)
for split, sc in scores_upstream.items():
    print(f"  with the starter's final prior projection, {split}: TEMP J "
          f"{sc['TEMP']['J']:.3f}  SALT J {sc['SALT']['J']:.3f}", flush=True)
if args.smoke:
    print(f"smoke ok ({time.time() - t0:.0f}s), nothing written")
    raise SystemExit(0)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


cpath = cohort_path(ROOT, "synthetic")
summary = {
    "tag": TAG, "region": "synthetic", "seed": args.seed, "target": "anomaly_exact",
    "splits": {k: list(v) for k, v in SPLITS.items()},
    "inputs_per_month": N_INPUT, "queries_per_month": N_QUERY, "eval_cells": EVAL_CELLS,
    "per_month": {name: {str(ev["month"]): {"inputs": int(ev["src"].size),
                                            "queries": int(ev["tgt"].size)}
                         for ev in evs}
                  for name, evs in (("validation", val), ("development", test))},
    "surface": list(SURF_VARS) if args.surface else None,
    "surface_store": SURF_STORE if args.surface else None,
    "upstream": {"repo": "https://github.com/CIA-Oceanix/4dvarnet-starter",
                 "commit": STARTER_COMMIT, "license": "CeCILL-C",
                 "classes": ["GradSolver", "BilinAEPriorCost", "ConvLstmGradModel",
                             "BaseObsCost"]},
    "state_channels": C_STATE, "grid": [NY, NX], "lon_pad": PAD,
    "n_step": N_STEP, "lr_grad": LR_GRAD, "prior_hidden": PRIOR_HIDDEN,
    "prior_downsamp": PRIOR_DOWNSAMP, "grad_hidden": GRAD_HIDDEN,
    "surface_hidden": SURF_HIDDEN if args.surface else None,
    "epochs": epochs, "batch": BATCH, "lr": LR, "clip": CLIP,
    "loss": {"mse": W_MSE, "prior": W_PRIOR, "sobel_gradient": 0.0},
    "training_targets": "argo_profiles", "target_fraction": TARGET_FRAC,
    "train_months": len(train_months), "params": n_par,
    "best_epoch": best["epoch"], "history": history,
    "identity_check": {"reference": REFERENCE, **ident},
    "cohort": {"path": os.path.relpath(cpath, ROOT), "sha256": sha256(cpath)},
    "solver_output": "state after the solver steps, without the starter's eval-mode "
                     "prior projection",
    "git_commit": git_commit(), "scores": scores,
    "scores_upstream_eval_path": scores_upstream, "runtime_s": time.time() - t0}
os.makedirs(OUT, exist_ok=True)
torch.save(best["state"], os.path.join(OUT, f"model_seed{args.seed}.pt"))
with open(os.path.join(OUT, f"summary_seed{args.seed}.json"), "w") as f:
    json.dump(summary, f, indent=1, default=float)
print(f"wrote {OUT}/summary_seed{args.seed}.json ({time.time() - t0:.0f}s)")
```

- [ ] **Step 2: Smoke run, both variants**

Pick an idle GPU `<n>` from `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`.

```bash
CUDA_VISIBLE_DEVICES=<n> ./.venv/bin/python experiments/synthetic/47_synth_argo_4dvarnet.py --smoke 2>&1 | tail -8
CUDA_VISIBLE_DEVICES=<n> ./.venv/bin/python experiments/synthetic/47_synth_argo_4dvarnet.py --smoke --surface 2>&1 | tail -8
```
Expected for each: a header line with `state 40 x 180 x 360`, a parameter count, two `epoch` lines with a finite training MSE and finite validation z, a `validation:` line, and `smoke ok (…s), nothing written`. Note the seconds per epoch (4 months) to estimate the full run: about 12 × that per epoch, times 150.

- [ ] **Step 3: Commit the driver**

```bash
git add experiments/synthetic/47_synth_argo_4dvarnet.py
git commit -q -m "47_synth_argo_4dvarnet: 4DVarNet (4dvarnet-starter solver) on the synthetic cohort

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Full runs**

```bash
CUDA_VISIBLE_DEVICES=<n> ./.venv/bin/python experiments/synthetic/47_synth_argo_4dvarnet.py 2>&1 | tee logs/synth_4dvarnet.log | tail -25
CUDA_VISIBLE_DEVICES=<n> ./.venv/bin/python experiments/synthetic/47_synth_argo_4dvarnet.py --surface 2>&1 | tee logs/synth_4dvarnet_surface.log | tail -25
```
(Run detached in `tmux` if the smoke timing projects more than about ten minutes each.)
Expected for each: `identity check passed`, a training MSE that falls over the epochs, a validation z below the climatology values (1.9145 TEMP, 1.7182 SALT), a `development:` line, and `wrote …/summary_seed1234.json`.
If the training MSE does not fall, or validation stays at climatology, stop and report the curve before changing any setting: the spec fixes the starter's settings.

(This happened on the first run: validation stayed at climatology while the training MSE fell. A diagnostic showed the cause is the starter's eval-mode prior projection, not the training; the driver above already scores the state before that projection and records the upstream-path score beside it.)

- [ ] **Step 5: Commit the results**

```bash
git add outputs/audit/synthetic/fourdvarnet/summary_seed1234.json outputs/audit/synthetic/fourdvarnet_surface/summary_seed1234.json
git commit -q -m "4DVarNet on the synthetic cohort: Argo-only and with surface fields, seed 1234

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Our model with the surface fields

**Files:**
- Modify: `experiments/real_data/62_sanity_train.py` (the `--surface` guard near line 170 and the store path near line 254)
- Modify: `experiments/real_data/run_audit_queue.py` (after `SYN_MASS_K15_L64`)

- [ ] **Step 1: Let the synthetic region read its own surface store**

In `experiments/real_data/62_sanity_train.py`, delete these three lines:

```python
if args.region == "synthetic" and args.surface:
    raise SystemExit("--surface reads the real satellite store; the synthetic "
                     "task is Argo-only")
```

and replace

```python
    _o = _xr.open_zarr(os.path.join(ROOT, "data", "real_obs_1deg.zarr"))
```

with

```python
    # the synthetic cohort has its own store (46_synth_surface_fields.py): CESM2
    # SST / SSS and a steric-height stand-in for altimetry, under the same names
    _o = _xr.open_zarr(os.path.join(
        ROOT, "data", *(("synthetic_argo", "cesm2_surface_1deg.zarr")
                        if args.region == "synthetic" else ("real_obs_1deg.zarr",))))
```

- [ ] **Step 2: Add the queue**

In `experiments/real_data/run_audit_queue.py`, replace

```python
SYN_QUEUES = {"syn_overfit": SYN_OVERFIT, "syn_refiner": SYN_REFINER,
              "syn_final": SYN_FINAL, "syn_refiner_k15": SYN_REFINER_K15,
              "syn_refiner_k15_l64": SYN_REFINER_K15_L64,
              "syn_mass_k15_l64": SYN_MASS_K15_L64}
```

with

```python
#: syn_surface_k15_l64: the current model (64 slots, 15 k steps, validated
#: refiner init) given the synthetic surface fields of 46_synth_surface_fields.py
#: (SST, SLA, SSS). Its Argo-only counterpart is `k15_l64_syn_r500_g1`.
SYN_SURFACE_K15_L64 = [("k15_l64_syn_r500_g1_surf",
                        FIX + ["--n-latent", "64", "--steps", "15000", "--surface"])]
SYN_QUEUES = {"syn_overfit": SYN_OVERFIT, "syn_refiner": SYN_REFINER,
              "syn_final": SYN_FINAL, "syn_refiner_k15": SYN_REFINER_K15,
              "syn_refiner_k15_l64": SYN_REFINER_K15_L64,
              "syn_mass_k15_l64": SYN_MASS_K15_L64,
              "syn_surface_k15_l64": SYN_SURFACE_K15_L64}
```

- [ ] **Step 3: Smoke run of the model with surface fields**

```bash
S=<scratch dir>; CUDA_VISIBLE_DEVICES=<n> ./.venv/bin/python experiments/real_data/62_sanity_train.py \
  --region synthetic --seed 1234 --tag smoke_surf --leads 0 --steps 60 --val-every 30 \
  --ablation anomaly_exact --refiner-km 500 --refiner-gate 1.0 --n-latent 64 --surface \
  --out-root "$S/smoke62_surf" 2>&1 | tail -10
```
Expected: a `surface channels: 72 months (2000-2005), 3 fields` line, two `step` lines, a final score line, and a runtime. Compare the runtime and `nvidia-smi` memory with the Argo-only smoke run (37 s for the same 60 steps). If it does not fit in memory or is more than about five times slower, re-run with `--surface-patch 5`, and use that value in the queue (append `"--surface-patch", "5"`), recording the change in the report.

- [ ] **Step 4: Tests, dry run, commit**

```bash
CUDA_VISIBLE_DEVICES="" ./.venv/bin/python -m pytest -q 2>&1 | tail -1
./.venv/bin/python experiments/real_data/run_audit_queue.py --queue syn_surface_k15_l64 --gpus <n> --seeds 1234,1235,1236 --dry-run | tail -4
git add experiments/real_data/62_sanity_train.py experiments/real_data/run_audit_queue.py
git commit -q -m "Surface fields on the synthetic cohort: 62 reads its store, queue syn_surface_k15_l64

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
Expected: all tests pass; three jobs listed, each ending `--n-latent 64 --steps 15000 --surface`.

- [ ] **Step 5: Launch the three seeds detached, and wait**

```bash
tmux new-session -d -s ocean_surf "cd $(pwd) && ./.venv/bin/python experiments/real_data/run_audit_queue.py --queue syn_surface_k15_l64 --gpus <idle list> --seeds 1234,1235,1236 --jobs 3 2>&1 | tee -a logs/audit/runner_syn_surface_k15_l64.log"
```
Progress: `logs/audit/queue_syn_surface_k15_l64.log`. Done when it prints `ALL DONE` with three `ok` lines and no `FAIL`.

- [ ] **Step 6: Commit the summaries**

```bash
git add outputs/audit/synthetic/k15_l64_syn_r500_g1_surf
git commit -q -m "Our model with surface fields on the synthetic cohort: run summaries, 3 seeds

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Report section, survey document, README

**Files:**
- Modify: `experiments/synthetic/43_synth_argo_report.py`
- Create: `docs/baseline_survey.md`
- Modify: `README.md`

- [ ] **Step 1: Section 12 in the report generator**

In `experiments/synthetic/43_synth_argo_report.py`, replace the function header and loop of `mlp_parity`

```python
def mlp_parity():
    """The pointwise MLP was scored with the models' inputs, on their cells."""
    want = (N_IN, T["queries_per_month"])
    ref = summary(FIX, SEEDS[0])["scores"]
    for x in seeds_done(MLPT):
        s = summary(MLPT, x)
```

with a version that serves any baseline driver:

```python
def baseline_parity(tag, name):
    """A baseline driver's runs were scored with the models' inputs, on their cells."""
    want = (N_IN, T["queries_per_month"])
    ref = summary(FIX, SEEDS[0])["scores"]
    for x in seeds_done(tag):
        s = summary(tag, x)
```

In the body that follows, replace the two occurrences of `pointwise MLP s{x}` with `{name} s{x}`, and after the function add:

```python
def mlp_parity():
    baseline_parity(MLPT, "pointwise MLP")
```

Then insert the following block immediately before the line `# ============================================ findings, one line per plan item`:

```python
# ============ 12. multi-modal comparison: Argo profiles and satellite-type fields
#: 47_synth_argo_4dvarnet.py (one fixed run each) and the current model given the
#: surface store of 46_synth_surface_fields.py (queue syn_surface_k15_l64)
FDV, FDV_SURF, FDV_SEED = "fourdvarnet", "fourdvarnet_surface", 1234
OURS, OURS_SURF = P64 + FIX, P64 + FIX + "_surf"
MM_DONE = (FB is not None and MLP_DONE and K15_DONE
           and all(summary(t, FDV_SEED) is not None for t in (FDV, FDV_SURF))
           and len(seeds_done(OURS_SURF)) == len(SEEDS))
md += ["## 12. Multi-modal comparison: Argo profiles and satellite-type fields\n"]
if not MM_DONE:
    md += ["Pending: `46_synth_surface_fields.py`, `47_synth_argo_4dvarnet.py` (with "
           "and without `--surface`) and the `syn_surface_k15_l64` queue.\n"]
else:
    for t in (FDV, FDV_SURF):
        baseline_parity(t, "4DVarNet")
    parity([OURS_SURF], steps=15000)
    want_surf = ",".join(sorted(summary(FDV_SURF, FDV_SEED)["surface"]))
    for x in SEEDS:
        s = summary(OURS_SURF, x)
        if s["n_latent"] != 64 or ",".join(sorted(s["surface"].split(","))) != want_surf:
            raise SystemExit(f"`{OURS_SURF}` s{x} is not the 64-slot run on {want_surf}")
    f0 = summary(FDV, FDV_SEED)

    def one(tag, ch, key="rmse_physical"):
        return val(tag, FDV_SEED, "development", ch, key)

    def three(tag, ch, key="rmse_physical"):
        return [val(tag, x, "development", ch, key) for x in SEEDS]

    ARGO, BOTH = "Argo", "Argo + SST, SSS, SLA"
    rows = [[lab, ARGO, how, f(fbv(n, "TEMP"), 4), f(fbv(n, "SALT"), 4),
             f(fbv(n, "TEMP", "J"), 3), f(fbv(n, "SALT", "J"), 3)]
            for n, lab, how in (("climatology", "climatology", "—"),
                                ("nearest_profile", "nearest profile", "—"),
                                ("oi", "OI", "tuned on validation"))]
    rows.insert(2, ["pointwise MLP", ARGO, "Argo profiles, one run", f(mlpv("TEMP"), 4),
                    f(mlpv("SALT"), 4), f(mlpv("TEMP", "J"), 3), f(mlpv("SALT", "J"), 3)])
    for tag, inp in ((FDV, ARGO), (FDV_SURF, BOTH)):
        rows.append(["4DVarNet", inp, "Argo profiles, one run", f(one(tag, "TEMP"), 4),
                     f(one(tag, "SALT"), 4), f(one(tag, "TEMP", "J"), 3),
                     f(one(tag, "SALT", "J"), 3)])
    for tag, inp in ((OURS, ARGO), (OURS_SURF, BOTH)):
        rows.append(["**our model**", inp, f"Argo profiles, {len(SEEDS)} seeds",
                     mean_sd(three(tag, "TEMP")), mean_sd(three(tag, "SALT")),
                     mean_sd(three(tag, "TEMP", "J")), mean_sd(three(tag, "SALT", "J"))])
    md += ["Every method on one table, test year 2005. A multi-modal method is given, "
           f"besides the month's {N_IN:,} input profiles, three 1° fields of the "
           "same month from CESM2 (`experiments/synthetic/46_synth_surface_fields.py`): "
           "SST, SSS and a sea-level field, each as an anomaly against its train-year "
           "monthly climatology. Every row is scored on the same "
           f"{N_TEST:,} values per variable, which this report checks.\n",
           "Two limits on what the multi-modal rows mean. **SST and SSS are the truth "
           "at the shallowest level**: CESM2's surface fields are its 5 m level, "
           "noise-free, so a method given them is handed the 0-100 m band's top level. "
           "**The sea-level field is derived from the truth**: CESM2's stored output "
           "has no sea level, so it is steric height computed from the very TEMP and "
           "SALT being reconstructed. Both make the satellite inputs more informative "
           "than real ones.\n",
           "4DVarNet (Fablet et al. 2021) is run from the authors' "
           f"[`4dvarnet-starter`]({f0['upstream']['repo']}) at commit "
           f"`{f0['upstream']['commit'][:7]}`: its solver, prior and gradient model "
           "unmodified, with the settings of its base configuration "
           f"({f0['n_step']} solver steps, {f0['epochs']} epochs, {f0['params']:,} "
           "parameters Argo-only), written by "
           "`experiments/synthetic/47_synth_argo_4dvarnet.py`. The state is TEMP and "
           f"SALT on the 20 levels ({f0['state_channels']} channels) on the 1° grid; "
           "profiles are binned to their cell; the answer is sampled at the query "
           "profiles. It is trained on Argo profiles only, "
           f"{100 * f0['target_fraction']:.0f} % of a month's input profiles held out "
           "as the target, where its authors train on a complete field with an added "
           "gradient loss; that loss is dropped here. It is one fixed run at seed "
           f"{FDV_SEED}, best epoch on validation "
           f"({f0['best_epoch']} Argo-only, {summary(FDV_SURF, FDV_SEED)['best_epoch']} "
           "with the fields).\n",
           md_table(["method", "inputs", "trained on", "test TEMP °C", "test SALT PSU",
                     "J TEMP", "J SALT"], rows)]
    rows = []
    for ch in CH:
        for lab, get in (
                ("OI, Argo", lambda b: float(FB["baselines"]["oi"]["scores"]["development"][ch]["by_band_z"][b])),
                ("4DVarNet, Argo", lambda b: float(summary(FDV, FDV_SEED)["scores"]["development"][ch]["by_band_z"][b])),
                ("4DVarNet, Argo + fields", lambda b: float(summary(FDV_SURF, FDV_SEED)["scores"]["development"][ch]["by_band_z"][b])),
                ("our model, Argo", lambda b: float(np.mean([summary(OURS, x)["scores"]["development"][ch]["by_band_z"][b] for x in SEEDS]))),
                ("our model, Argo + fields", lambda b: float(np.mean([summary(OURS_SURF, x)["scores"]["development"][ch]["by_band_z"][b] for x in SEEDS])))):
            rows.append([ch, lab] + [f(get(b), 4) for b in BK])
    d_ours = {ch: paired(OURS_SURF, OURS, "development", ch) for ch in CH}
    d_fdv = {ch: one(FDV_SURF, ch) - one(FDV, ch) for ch in CH}
    md += ["By depth band, test RMSE in z units (the fields act mostly where they "
           "are the answer, near the surface):\n",
           md_table(["", "", *BK], rows),
           f"Effect of the fields on test: our model "
           f"{fmt_pair(d_ours['TEMP'])} °C and {fmt_pair(d_ours['SALT'], 5)} PSU "
           f"(paired over seeds); 4DVarNet {d_fdv['TEMP']:+.4f} °C and "
           f"{d_fdv['SALT']:+.5f} PSU (one run each).\n"]

```

- [ ] **Step 2: Finding 12**

In the same file, directly before the line `def tm(tag, ch):`, insert:

```python
if MM_DONE:
    cand = {"OI (Argo)": (fbv("oi", "TEMP"), fbv("oi", "SALT")),
            "4DVarNet (Argo)": (one(FDV, "TEMP"), one(FDV, "SALT")),
            "4DVarNet (Argo + fields)": (one(FDV_SURF, "TEMP"), one(FDV_SURF, "SALT")),
            "our model (Argo)": (float(np.mean(three(OURS, "TEMP"))),
                                 float(np.mean(three(OURS, "SALT")))),
            "our model (Argo + fields)": (float(np.mean(three(OURS_SURF, "TEMP"))),
                                          float(np.mean(three(OURS_SURF, "SALT"))))}
    order = sorted(cand, key=lambda k: cand[k][0])
    find.append(
        "12. **Multi-modal comparison** — with SST, SSS and a sea-level field added "
        "(noise-free, and derived from the truth), test TEMP / SALT are "
        + "; ".join(f"{k} {cand[k][0]:.4f} °C / {cand[k][1]:.4f} PSU" for k in order)
        + f", lowest temperature error first. The fields change our model by "
        f"{d_ours['TEMP'][0]:+.4f} °C and 4DVarNet by {d_fdv['TEMP']:+.4f} °C.\n")

```

- [ ] **Step 3: Regenerate and verify**

```bash
CUDA_VISIBLE_DEVICES="" ./.venv/bin/python experiments/synthetic/43_synth_argo_report.py 2>&1 | grep -v findfont | tail -2
git diff reports/synthetic/synth_argo_audit.md | grep '^-' | grep -v '^---'
sed -n '/^## 12\./,$p' reports/synthetic/synth_argo_audit.md
grep -n "^12\. " reports/synthetic/synth_argo_audit.md
```
Expected: `wrote …`; no removed lines; section 12 with eight table rows and no `—` in a number column; finding 12 present. Check each number against the drivers' own printed `development:` lines.

- [ ] **Step 4: The survey document**

Create `docs/baseline_survey.md`:

```markdown
# Baselines for sparse-profile ocean reconstruction, with code

Survey of 2026-10-06 for the multi-modal comparison: methods that reconstruct
subsurface temperature and salinity from in-situ profiles, satellite fields, or
both, and whether code exists. "Status here" says what this repository has.

| Method | Reference | Code | Inputs | Status here |
|---|---|---|---|---|
| Optimal interpolation | Bretherton, Davis and Fandry 1976 | `src/ocean_tokenizer/oi.py` | profiles | run on the synthetic cohort (`44_synth_argo_oi.py`) |
| Pointwise MLP | this repository's gridded line | `src/ocean_tokenizer/baselines.py` | profiles | run on the synthetic cohort (`45_synth_argo_mlp.py`) |
| 4DVarNet | Fablet et al. 2021 | https://github.com/CIA-Oceanix/4dvarnet-starter (CeCILL-C, last commit 2024-12); the older https://github.com/CIA-Oceanix/4dvarnet-core (last commit 2023-05, PyTorch 1.11) | gridded observations with gaps, optional second field | run on the synthetic cohort (`47_synth_argo_4dvarnet.py`) |
| ARMOR3D-style | Guinehut et al. 2012, https://os.copernicus.org/articles/8/845/2012/ | none public | regression of T/S on altimetry and SST, then OI with profiles | not built; both pieces exist here |
| ConvNP | DeepSensor, https://github.com/alan-turing-institute/deepsensor | pip package `deepsensor` | off-grid and gridded inputs, any target point | the repo's SetConv U-Net backbone is this family; not run on the synthetic cohort with fields |
| OSnet | Pauthenet et al. 2022, https://os.copernicus.org/articles/18/1221/2022/ | https://github.com/euroargodev/OSnet-GulfStream | satellite fields to T/S profiles | re-implemented without sea level for the gridded line (`src/baselines/osnet_mlp.py`) |
| NeSPReSO | Miranda et al., https://data.coaps.fsu.edu/eric_pub/papers_html/Miranda_et_al_24.pdf | no public repository found; a web API | satellite fields to PCA scores of profiles | re-implemented without sea level for the gridded line (`src/baselines/nesperso_pcamlp.py`) |
| Stacked LSTM | Buongiorno Nardelli 2020 | not checked | satellite fields to profiles | re-implemented without sea level for the gridded line (`src/baselines/nardelli_lstm.py`) |
| Senseiver | Santos et al. 2023 | public | sparse sensors to a field | reproduced on real data (`48_senseiver.py`) |

Related, not methods to run:

- OceanDepths (Donike et al., August 2026, https://arxiv.org/abs/2608.16373):
  a global dataset of paired surface and subsurface observations with a
  reconstruction baseline task, at https://huggingface.co/datasets/ESA-philab/OceanDepths.
- TS-Cast (Ocean Science 2026, https://os.copernicus.org/articles/22/2161/2026/):
  satellite fields to subsurface T/S in the northwestern Pacific; code
  availability not confirmed.

## Notes on 4DVarNet

- `4dvarnet-core` is the original research code: PyTorch 1.11, Lightning 1.6,
  Python 3.9, configurations tied to its authors' clusters. `4dvarnet-starter`
  is the same group's maintained, smaller version (PyTorch 2, about 1,000
  lines) and is what this repository uses.
- Both are CeCILL-C. The clone lives unmodified in the git-ignored `external/`
  folder; our driver imports its solver, prior and gradient model.
- It interpolates gridded fields, so profiles are binned to the 1° grid and its
  answer is sampled back at the query profiles.
- It is normally trained on a complete field with a gradient loss. On the
  synthetic cohort it is trained on Argo profiles only, like every other
  learned method there.
```

- [ ] **Step 5: README**

In `README.md`, replace

```
.venv/bin/python experiments/synthetic/45_synth_argo_mlp.py --seed 1234  # pointwise MLP baseline, one fixed run
```

with

```
.venv/bin/python experiments/synthetic/45_synth_argo_mlp.py --seed 1234  # pointwise MLP baseline, one fixed run
# multi-modal comparison: surface fields, 4DVarNet (upstream clone, CeCILL-C, kept out of the tree), our model
.venv/bin/python experiments/synthetic/46_synth_surface_fields.py
git clone https://github.com/CIA-Oceanix/4dvarnet-starter.git external/4dvarnet-starter
git -C external/4dvarnet-starter checkout 20f1b5f34b201342cde6dd21a30419d07541db54 && .venv/bin/pip install kornia
.venv/bin/python experiments/synthetic/47_synth_argo_4dvarnet.py            # Argo only
.venv/bin/python experiments/synthetic/47_synth_argo_4dvarnet.py --surface  # with SST, SSS, SLA
.venv/bin/python $R --queue syn_surface_k15_l64 --gpus 0,1 --seeds 1234,1235,1236   # our model with the fields
```

- [ ] **Step 6: Final checks and commit**

```bash
CUDA_VISIBLE_DEVICES="" ./.venv/bin/python -m pytest -q 2>&1 | tail -1
git add experiments/synthetic/43_synth_argo_report.py reports/synthetic/synth_argo_audit.md docs/baseline_survey.md README.md
git commit -q -m "Synthetic audit report: multi-modal comparison (section 12); baseline survey

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git status --short | wc -l
```
Expected: all tests pass (522 before this plan, plus 8); `0`.

- [ ] **Step 7: Report to the user**

The table with measured numbers; what the satellite fields changed for each method; the two limitations of the fields; any setting that had to differ from the plan (for example the satellite patch size); that nothing was pushed.
