# Fixed Baselines (OI) on the Synthetic Argo Cohort — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score climatology, nearest-profile and a validation-tuned optimal interpolation on exactly the cells the trained synthetic models are scored on, and add the result plus a five-item todo status table to the synthetic audit report.

**Architecture:** A small numpy module (`point_baselines.py`) provides the baselines at scattered query points and reproduces the training script's scoring arithmetic. A driver script (`44_synth_argo_oi.py`) loads the synthetic cohort the way `62_sanity_train.py` does, proves it is scoring the model's cells (input-parity assertions and a zero-prediction identity check), tunes OI on the validation year, scores the test year once and writes JSON. The existing report generator (`43_synth_argo_report.py`) reads that JSON. No model is retrained.

**Tech Stack:** Python 3.12, numpy, scipy (`cKDTree`), the repo's `ocean_tokenizer.oi` solver, pytest. CPU only.

**Spec:** `docs/superpowers/specs/2026-10-04-synthetic-oi-baseline-design.md`

**Conventions used below**

- All commands run from the repository root on branch `synthetic-oi-baseline`.
- Python is always `./.venv/bin/python`.
- Every commit message ends with the trailer `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- "Test year" is the split the summaries call `development` (2005). Validation is 2004.

## File structure

| File | Responsibility |
|---|---|
| `src/ocean_tokenizer/point_baselines.py` (new) | Baselines at scattered queries (`nearest_profile`, `point_sweep`, `oi_points`) and pooled scoring (`band_of_levels`, `Scores`). Pure numpy/scipy, no I/O. |
| `tests/test_point_baselines.py` (new) | Closed-form tests of the module above. |
| `experiments/synthetic/44_synth_argo_oi.py` (new) | Driver: cohort, evaluation sets, parity and identity guards, OI tuning, scoring, JSON output. |
| `experiments/synthetic/43_synth_argo_report.py` (modify) | Todo status table, finding 9, §9. Sections 1-8 untouched. |
| `outputs/audit/synthetic/fixed_baselines/{summary,tuning}.json` (generated, committed) | Results. |
| `reports/synthetic/synth_argo_audit.md` (regenerated) | Report. |
| `README.md` (modify) | One command line. |

---

### Task 1: Environment

**Files:** none (installs into `.venv`).

- [ ] **Step 1: Install the pinned environment**

Run:
```bash
./.venv/bin/pip install -q -r requirements-lock.txt 2>&1 | tail -5
./.venv/bin/python -c "import torch, scipy, matplotlib, pytest; print(torch.__version__, scipy.__version__)"
```
Expected: `2.13.0 1.18.1` (a `+cu…` suffix on torch is fine).
If the lock file does not resolve on this machine, fall back to
`./.venv/bin/pip install -r requirements.txt scipy matplotlib pytest gsw pandas` and note the versions that were installed in the final report.

- [ ] **Step 2: Run the existing suite as the baseline**

Run: `./.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: `498 passed` (the README's count). Record the actual number; later steps expect it plus the new tests. If any test fails here, stop and report it before changing code.

---

### Task 2: `nearest_profile`

**Files:**
- Create: `src/ocean_tokenizer/point_baselines.py`
- Test: `tests/test_point_baselines.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_point_baselines.py`:

```python
"""Closed-form tests for the fixed baselines at scattered query points.

Like tests/test_oi.py these pin the arithmetic, not a regression snapshot, and
need no data store.
"""
import numpy as np
import pytest

from ocean_tokenizer.point_baselines import nearest_profile


def _obs(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-10, 10, n), rng.uniform(0, 20, n), rng.normal(size=n)


# --------------------------------------------------------------------------
# nearest_profile
# --------------------------------------------------------------------------
def test_nearest_profile_takes_the_closest_value():
    lat = np.array([0.0, 0.0, 5.0]); lon = np.array([0.0, 10.0, 5.0])
    out = nearest_profile(lat, lon, np.array([1.0, 2.0, 3.0]),
                          np.array([0.1, 0.0, 4.9]), np.array([0.2, 9.5, 5.1]))
    assert np.array_equal(out, [1.0, 2.0, 3.0])


def test_nearest_profile_skips_non_finite_levels():
    """The closest profile has no value at this level: the next one is used."""
    out = nearest_profile(np.array([0.0, 0.0]), np.array([0.0, 3.0]),
                          np.array([np.nan, 7.0]), np.array([0.0]), np.array([0.1]))
    assert np.array_equal(out, [7.0])


def test_nearest_profile_crosses_the_date_line():
    """359.9 E is 0.2 deg from 0.1 E, not 359.8 deg."""
    out = nearest_profile(np.array([0.0, 0.0]), np.array([359.9, 180.0]),
                          np.array([4.0, 9.0]), np.array([0.0]), np.array([0.1]))
    assert np.array_equal(out, [4.0])


def test_nearest_profile_without_observations_is_background():
    out = nearest_profile(np.array([0.0]), np.array([0.0]), np.array([np.nan]),
                          np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert np.array_equal(out, [0.0, 0.0])
```

- [ ] **Step 2: Run them to see them fail**

Run: `./.venv/bin/python -m pytest tests/test_point_baselines.py -q 2>&1 | tail -3`
Expected: collection error, `ModuleNotFoundError: No module named 'ocean_tokenizer.point_baselines'`.

- [ ] **Step 3: Implement**

Create `src/ocean_tokenizer/point_baselines.py`:

```python
"""Fixed (no trainable parameters) baselines at scattered query points.

The synthetic Argo audit scores a model at held-out profiles' own positions
rather than on a grid. These helpers give the non-learned references that
interface -- observations at (lat, lon) with one value per level, queries at
(lat, lon), one analysis per query -- and reproduce the scoring arithmetic of
``experiments/real_data/62_sanity_train.py`` on numpy arrays, so a baseline and
a trained model are pooled by the same formulas.

Everything is in z-scored anomaly space: the background is zero, and a
non-finite observation (a level below the sea floor) is simply absent.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .oi import LevelSweep, _lonlat_to_xyz


def _finite(obs_lat, obs_lon, obs_val):
    """Coordinates and values of the observations that are finite at this level."""
    v = np.asarray(obs_val, dtype=np.float64).ravel()
    ok = np.isfinite(v)
    return (np.asarray(obs_lat, dtype=np.float64).ravel()[ok],
            np.asarray(obs_lon, dtype=np.float64).ravel()[ok], v[ok])


def nearest_profile(obs_lat, obs_lon, obs_val, q_lat, q_lon):
    """(Q,) value of the nearest finite observation; 0 where there is none."""
    lat, lon, v = _finite(obs_lat, obs_lon, obs_val)
    q_lat = np.asarray(q_lat, dtype=np.float64).ravel()
    q_lon = np.asarray(q_lon, dtype=np.float64).ravel()
    if v.size == 0:
        return np.zeros(q_lat.size)
    # chord length on the unit sphere is monotone in great-circle distance
    _, idx = cKDTree(_lonlat_to_xyz(lat, lon)).query(_lonlat_to_xyz(q_lat, q_lon), k=1)
    return v[np.asarray(idx).ravel()]
```

- [ ] **Step 4: Run the tests**

Run: `./.venv/bin/python -m pytest tests/test_point_baselines.py -q 2>&1 | tail -3`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/ocean_tokenizer/point_baselines.py tests/test_point_baselines.py
git commit -q -m "point_baselines: nearest-profile baseline at scattered queries

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: OI at scattered queries (`point_sweep`, `oi_points`)

**Files:**
- Modify: `src/ocean_tokenizer/point_baselines.py`
- Test: `tests/test_point_baselines.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_point_baselines.py`, replace the import line
`from ocean_tokenizer.point_baselines import nearest_profile` with:

```python
from ocean_tokenizer.oi import oi_level
from ocean_tokenizer.point_baselines import nearest_profile, oi_points, point_sweep
```

and append:

```python
# --------------------------------------------------------------------------
# optimal interpolation at scattered queries
# --------------------------------------------------------------------------
def test_oi_points_equals_oi_level_on_the_same_points():
    """Scattered queries are the gridded solver's points, in any layout."""
    lat, lon, val = _obs()
    lat2d, lon2d = np.meshgrid(np.linspace(-8, 8, 9), np.linspace(2, 18, 7),
                               indexing="ij")
    grid = oi_level(lat, lon, val, lat2d, lon2d, np.ones_like(lat2d, dtype=bool),
                    L_km=400.0, gamma=0.1, k=12)
    pts = oi_points(lat, lon, val, lat2d.ravel(), lon2d.ravel(),
                    L_km=400.0, gamma=0.1, k=12)
    assert pts.shape == (lat2d.size,)
    assert np.allclose(pts, grid.ravel(), rtol=0, atol=1e-10)


def test_oi_points_returns_the_observation_as_gamma_vanishes():
    """A query on top of an observation gets that value back when gamma -> 0."""
    lat, lon, val = _obs(n=25, seed=1)
    out = oi_points(lat, lon, val, lat[:5], lon[:5], L_km=300.0, gamma=1e-9, k=10)
    assert np.allclose(out, val[:5], atol=1e-5)


def test_oi_points_ignores_non_finite_observations():
    lat, lon, val = _obs(n=40, seed=2)
    holed = val.copy(); holed[::4] = np.nan
    keep = np.isfinite(holed)
    q_lat, q_lon = np.array([0.0, 3.0]), np.array([10.0, 12.0])
    a = oi_points(lat, lon, holed, q_lat, q_lon, L_km=500.0, gamma=0.1, k=8)
    b = oi_points(lat[keep], lon[keep], val[keep], q_lat, q_lon,
                  L_km=500.0, gamma=0.1, k=8)
    assert np.allclose(a, b, rtol=0, atol=1e-12)


def test_oi_points_without_observations_is_background():
    out = oi_points(np.array([0.0]), np.array([0.0]), np.array([np.nan]),
                    np.array([1.0, 2.0]), np.array([1.0, 2.0]),
                    L_km=500.0, gamma=0.1, k=8)
    assert np.array_equal(out, [0.0, 0.0])


def test_point_sweep_sub_k_matches_a_fresh_solve():
    """One geometry at the largest k serves every smaller k exactly."""
    lat, lon, val = _obs(n=50, seed=3)
    q_lat, q_lon = np.array([-2.0, 4.0, 7.0]), np.array([5.0, 9.0, 15.0])
    sweep, v = point_sweep(lat, lon, val, q_lat, q_lon, k=20)
    assert np.allclose(sweep.sub_k(6).analyse(v, 350.0, 0.03),
                       oi_points(lat, lon, val, q_lat, q_lon, 350.0, 0.03, 6),
                       rtol=0, atol=1e-10)
```

- [ ] **Step 2: Run them to see them fail**

Run: `./.venv/bin/python -m pytest tests/test_point_baselines.py -q 2>&1 | tail -3`
Expected: `ImportError: cannot import name 'oi_points'`.

- [ ] **Step 3: Implement**

Append to `src/ocean_tokenizer/point_baselines.py`:

```python
def point_sweep(obs_lat, obs_lon, obs_val, q_lat, q_lon, k):
    """k-NN geometry over one level's finite observations, at scattered queries.

    Returns ``(sweep, values)``: an :class:`oi.LevelSweep` whose analysis has
    one entry per query, and the finite observation values it was built from
    (pass them to ``sweep.analyse``). ``sweep.sub_k`` gives any smaller k from
    the same geometry.
    """
    lat, lon, v = _finite(obs_lat, obs_lon, obs_val)
    q_lat = np.asarray(q_lat, dtype=np.float64).ravel()
    q_lon = np.asarray(q_lon, dtype=np.float64).ravel()
    sweep = LevelSweep(lat, lon, q_lat, q_lon, np.ones(q_lat.size, dtype=bool), k=k)
    return sweep, v


def oi_points(obs_lat, obs_lon, obs_val, q_lat, q_lon, L_km, gamma, k):
    """(Q,) optimal-interpolation analysis at scattered queries (see :mod:`oi`)."""
    sweep, v = point_sweep(obs_lat, obs_lon, obs_val, q_lat, q_lon, k)
    return sweep.analyse(v, L_km, gamma)
```

- [ ] **Step 4: Run the tests**

Run: `./.venv/bin/python -m pytest tests/test_point_baselines.py -q 2>&1 | tail -3`
Expected: `9 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/ocean_tokenizer/point_baselines.py tests/test_point_baselines.py
git commit -q -m "point_baselines: optimal interpolation at scattered queries

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Pooled scoring (`band_of_levels`, `Scores`)

This reproduces `score()` in `experiments/real_data/62_sanity_train.py` (lines 468-512): per channel, sum of squared z errors, cell count, sum of squared physical errors (`e² · std²` of the cell's level), sum of squared z targets; per band the same plus the sum of squared physical targets. `J = rmse_z / climatology_z`; `by_band_J = sqrt(Σ e² std² / Σ target² std²)`.

**Files:**
- Modify: `src/ocean_tokenizer/point_baselines.py`
- Test: `tests/test_point_baselines.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_point_baselines.py`, replace the line
`from ocean_tokenizer.point_baselines import nearest_profile, oi_points, point_sweep` with:

```python
from ocean_tokenizer.point_baselines import (Scores, band_of_levels, nearest_profile,
                                             oi_points, point_sweep)
```

and append:

```python
# --------------------------------------------------------------------------
# pooled scoring (the arithmetic of score() in 62_sanity_train.py)
# --------------------------------------------------------------------------
LEVELS = np.array([5.0, 50.0, 200.0, 500.0, 985.0])
STD = {"TEMP": np.array([2.0, 1.0, 0.5, 0.25, 0.1]), "SALT": np.ones(5)}


def test_band_of_levels_follows_the_training_script():
    assert list(band_of_levels(LEVELS)) == ["0-100m", "0-100m", "100-300m",
                                            "300-700m", "700-1400m"]


def test_scores_hand_computed_case():
    s = Scores(LEVELS, STD)
    # two cells in 0-100 m (levels 0 and 1), one in 300-700 m (level 3)
    s.add("TEMP", [1.0, 0.0, 2.0], [0.0, 2.0, 1.0], [0, 1, 3])
    r = s.result()["TEMP"]
    # squared errors 1, 4, 1; squared targets 0, 4, 1; std^2 4, 1, 0.0625
    assert r["n"] == 3 and r["unit"] == "degC"
    assert r["rmse_z"] == pytest.approx(np.sqrt(6 / 3))
    assert r["climatology_z"] == pytest.approx(np.sqrt(5 / 3))
    assert r["J"] == pytest.approx(np.sqrt(6 / 5))
    assert r["rmse_physical"] == pytest.approx(np.sqrt((4 + 4 + 0.0625) / 3))
    assert r["by_band_z"]["0-100m"] == pytest.approx(np.sqrt(5 / 2))
    assert r["by_band_z"]["300-700m"] == pytest.approx(1.0)
    assert np.isnan(r["by_band_z"]["100-300m"])
    assert r["by_band_physical"]["0-100m"] == pytest.approx(np.sqrt(8 / 2))
    assert r["by_band_J"]["0-100m"] == pytest.approx(np.sqrt(8 / 4))
    assert r["by_band_J"]["300-700m"] == pytest.approx(1.0)


def test_scores_skip_non_finite_targets_and_pool_across_calls():
    s = Scores(LEVELS, STD)
    s.add("SALT", [1.0, 5.0], [0.0, np.nan], [0, 1])
    s.add("SALT", [3.0], [0.0], [2])
    r = s.result()
    assert r["SALT"]["n"] == 2
    assert r["SALT"]["rmse_z"] == pytest.approx(np.sqrt((1 + 9) / 2))
    assert "TEMP" not in r
    assert r["macro_z"] == pytest.approx(r["SALT"]["rmse_z"])


def test_zero_prediction_scores_j_one():
    """Climatology (zero anomaly) is the J = 1 floor by construction."""
    rng = np.random.default_rng(0)
    t, li = rng.normal(size=200), rng.integers(0, 5, 200)
    s = Scores(LEVELS, STD)
    s.add("TEMP", np.zeros(200), t, li)
    r = s.result()["TEMP"]
    assert r["J"] == pytest.approx(1.0)
    assert r["rmse_z"] == pytest.approx(r["climatology_z"])
```

- [ ] **Step 2: Run them to see them fail**

Run: `./.venv/bin/python -m pytest tests/test_point_baselines.py -q 2>&1 | tail -3`
Expected: `ImportError: cannot import name 'Scores'`.

- [ ] **Step 3: Implement**

In `src/ocean_tokenizer/point_baselines.py`, add these constants directly below the `from .oi import LevelSweep, _lonlat_to_xyz` line:

```python
CH = ("TEMP", "SALT")
UNITS = {"TEMP": "degC", "SALT": "PSU"}
#: the depth bands of 62_sanity_train.py: same names, same edges
BANDS = (("0-100m", 0.0, 100.0), ("100-300m", 100.0, 300.0),
         ("300-700m", 300.0, 700.0), ("700-1400m", 700.0, 1401.0))
```

and append at the end of the file:

```python
def band_of_levels(levels, bands=BANDS):
    """Band name of every level, by the rule of 62_sanity_train.py."""
    levels = np.asarray(levels, dtype=float)
    out = []
    for d in levels:
        for name, lo, hi in bands:
            if (lo < d <= hi) or (d <= levels.min() and lo <= 0):
                out.append(name)
                break
        else:
            out.append(bands[-1][0])
    return np.array(out)


class Scores:
    """Pooled errors over cells, accumulated as ``score()`` in 62 does.

    ``add`` takes one channel's cells (prediction and target in z units, and
    each cell's level index); cells with a non-finite target are skipped.
    ``result`` returns the ``scores`` block 62 writes to its summaries:
    ``rmse_z``, ``rmse_physical``, ``unit``, ``J`` (RMSE / RMS of the target),
    ``climatology_z`` (RMS of the target), ``n``, the three ``by_band_*``
    dictionaries and ``macro_z`` (mean ``rmse_z`` over channels).
    """

    def __init__(self, levels, std, bands=BANDS):
        self.bands = bands
        self.band = band_of_levels(levels, bands)
        self.std = {ch: np.asarray(std[ch], dtype=np.float64) for ch in CH}
        # per channel: sum e^2 (z), cells, sum e^2 (physical), sum target^2 (z)
        self.tot = {ch: [0.0, 0, 0.0, 0.0] for ch in CH}
        # per band: sum e^2 (z), cells, sum e^2 (physical), sum target^2 (physical)
        self.by = {ch: {b: [0.0, 0, 0.0, 0.0] for b, _, _ in bands} for ch in CH}

    def add(self, ch, pred_z, target_z, level_index):
        pred = np.asarray(pred_z, dtype=np.float64).ravel()
        tgt = np.asarray(target_z, dtype=np.float64).ravel()
        li = np.asarray(level_index, dtype=int).ravel()
        m = np.isfinite(tgt)
        if not m.any():
            return
        e = (pred[m] - tgt[m]) ** 2
        sd2 = self.std[ch][li[m]] ** 2
        t2 = tgt[m] ** 2
        t = self.tot[ch]
        t[0] += float(e.sum()); t[1] += int(m.sum())
        t[2] += float((e * sd2).sum()); t[3] += float(t2.sum())
        bl = self.band[li[m]]
        for b, _, _ in self.bands:
            k = bl == b
            if k.any():
                v = self.by[ch][b]
                v[0] += float(e[k].sum()); v[1] += int(k.sum())
                v[2] += float((e[k] * sd2[k]).sum())
                v[3] += float((t2[k] * sd2[k]).sum())

    def result(self):
        nan = float("nan")
        out = {}
        for ch in CH:
            se, n, sep, se0 = self.tot[ch]
            if n == 0:
                continue
            rz, r0 = float(np.sqrt(se / n)), float(np.sqrt(se0 / n))
            by = self.by[ch]
            out[ch] = {
                "rmse_z": rz, "rmse_physical": float(np.sqrt(sep / n)),
                "unit": UNITS[ch], "J": rz / max(r0, 1e-9),
                "climatology_z": r0, "n": n,
                "by_band_z": {b: (float(np.sqrt(v[0] / v[1])) if v[1] else nan)
                              for b, v in by.items()},
                "by_band_physical": {b: (float(np.sqrt(v[2] / v[1])) if v[1] else nan)
                                     for b, v in by.items()},
                "by_band_J": {b: (float(np.sqrt(v[2] / v[3])) if v[3] else nan)
                              for b, v in by.items()}}
        out["macro_z"] = float(np.mean([out[ch]["rmse_z"] for ch in CH if ch in out]))
        return out
```

- [ ] **Step 4: Run the tests**

Run: `./.venv/bin/python -m pytest tests/test_point_baselines.py -q 2>&1 | tail -3`
Expected: `13 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/ocean_tokenizer/point_baselines.py tests/test_point_baselines.py
git commit -q -m "point_baselines: pooled scores matching 62_sanity_train.py

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: The driver script

How the evaluation set mirrors `62_sanity_train.py`:

- `build_eval(split, max_cells)` loops over `c.months_in(split)`; inputs are `c.month(m, float_split="cohort_float")` (all of them when `--n-profiles 0`), queries are `c.month(m, float_split="heldout_float")`.
- `make_sample` lays cells out as profile-major: cell `i` is profile `i // L`, level `i % L`. With a cap it keeps `np.random.default_rng([20260918, m]).choice(R * L, max_cells, replace=False)`.
- Validation uses `max_cells = 8000` (`--eval-cells` default); the test split is scored in full.

**Files:**
- Create: `experiments/synthetic/44_synth_argo_oi.py`

- [ ] **Step 1: Write the script**

Create `experiments/synthetic/44_synth_argo_oi.py`:

```python
"""Fixed baselines on the synthetic Argo cohort: climatology, nearest profile, OI.

The reference the synthetic audit lacked. Methods with no trainable parameters
are scored on exactly the cells the trained models are scored on
(`62_sanity_train.py --region synthetic --ablation anomaly_exact`): the same
cohort, at-position target and train-year normalisation, the same months, the
same 6,080 input profiles and 1,520 query profiles a month, the same 8,000-cell
validation draw, the same pooled metrics.

  climatology       zero anomaly (the J = 1 floor)
  nearest_profile   the nearest input profile's value at the same level
  oi                optimal interpolation (ocean_tokenizer.oi), level by level,
                    with (L_km, gamma, k) selected per variable and depth band
                    on the VALIDATION year; the test year is scored once
  oi_single         the same with one setting per variable (no band split)

Two guards run before any number is written:

* input parity -- every evaluated month hands every method all 6,080 input
  profiles and scores 1,520 query profiles, so no method sees fewer Argo
  profiles than another;
* identity -- a zero prediction reproduces the cell count and climatology
  error stored in a trained model's summary, so the cells and the
  normalisation are the model's.

Writes outputs/audit/synthetic/fixed_baselines/{summary,tuning}.json.

  .venv/bin/python experiments/synthetic/44_synth_argo_oi.py           # a few minutes, CPU
  .venv/bin/python experiments/synthetic/44_synth_argo_oi.py --smoke   # 1 month, prints only
"""
from __future__ import annotations

import os

# one BLAS thread per process: the work is many small solves spread over months
# by a process pool, and nested threading only oversubscribes a shared host
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse, hashlib, itertools, json, subprocess, sys, time
from multiprocessing import get_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np

from ocean_tokenizer.argo_obs import ArgoNorm
from ocean_tokenizer.audit_tools import cohort_path, load_cohort
from ocean_tokenizer.point_baselines import (BANDS, CH, Scores, band_of_levels,
                                             nearest_profile, oi_points, point_sweep)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPLITS = {"train": (2000, 2003), "validation": (2004, 2004), "development": (2005, 2005)}
N_INPUT, N_QUERY = 6080, 1520
EVAL_CELLS = 8000       # 62_sanity_train.py --eval-cells default (validation cap)
EVAL_SEED = 20260918    # 62's build_eval seeds the cap with [EVAL_SEED, month]
#: the trained model whose stored cell counts and climatology error the
#: identity check must reproduce
REFERENCE = os.path.join("outputs", "audit", "synthetic", "syn_r500_g1",
                         "summary_seed1234.json")
AXES = ("L_km", "gamma", "k")
GRID = {"L_km": [150.0, 250.0, 400.0, 600.0, 900.0, 1500.0],
        "gamma": [0.01, 0.03, 0.1, 0.3],
        "k": [10, 20, 40]}
#: edge rule: when a selection sits on an end of an axis, the next value of
#: this ladder is added on that side and the selection is repeated
LADDER = {"L_km": {"lo": [100.0, 60.0], "hi": [2500.0, 4000.0]},
          "gamma": {"lo": [0.003, 0.001], "hi": [1.0, 3.0]},
          "k": {"lo": [5], "hi": [80]}}
BAND_NAMES = [b for b, _, _ in BANDS]

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=12, help="processes, one month each")
ap.add_argument("--out", default=None,
                help="default: outputs/audit/synthetic/fixed_baselines")
ap.add_argument("--smoke", action="store_true",
                help="one validation month, a 2 x 2 x 1 grid, no edge rule, "
                     "no identity check; prints and writes nothing")
args = ap.parse_args()
OUT = args.out or os.path.join(ROOT, "outputs", "audit", "synthetic", "fixed_baselines")
t0 = time.time()

# ------------------------------------------------------------------ data
c, _ = load_cohort(ROOT, "synthetic", SPLITS, anomaly="exact")
norm = ArgoNorm.fit(c, "train")
LEV = c.levels
BAND_OF = band_of_levels(LEV)                          # (L,) band name per level
OBS = {ch: norm.z(ch, getattr(c, ch)) for ch in CH}    # (P, L) z-scored anomaly


def eval_month(m, max_cells):
    """One month's inputs and scored cells, as build_eval / make_sample in 62."""
    src = c.month(m, float_split="cohort_float")
    tgt = c.month(m, float_split="heldout_float")
    if (src.size, tgt.size) != (N_INPUT, N_QUERY):
        raise SystemExit(f"input parity violated in month {m}: {src.size} inputs, "
                         f"{tgt.size} queries (expected {N_INPUT} and {N_QUERY})")
    R, L = tgt.size, LEV.size
    prof = np.repeat(np.arange(R), L)        # query profile of each cell
    lev = np.tile(np.arange(L), R)           # level of each cell
    if max_cells and R * L > max_cells:
        pick = np.random.default_rng([EVAL_SEED, int(m)]).choice(
            R * L, max_cells, replace=False)
        prof, lev = prof[pick], lev[pick]
    return dict(month=int(m), src=src, tgt=tgt, prof=prof, lev=lev,
                target={ch: OBS[ch][tgt][prof, lev] for ch in CH})


def eval_set(split, max_cells=0, n_months=None):
    months = [int(m) for m in c.months_in(split)]
    return [eval_month(m, max_cells) for m in (months[:n_months] if n_months else months)]


# ------------------------------------------------------------------ methods
def predict(ev, method, setting=None):
    """{channel: (R, L)} analysis at the month's query profiles.

    Every method is given ALL of the month's input profiles. ``setting`` is,
    for OI, {channel: {band: (L_km, gamma, k)}}.
    """
    src, tgt = ev["src"], ev["tgt"]
    out = {}
    for ch in CH:
        p = np.zeros((tgt.size, LEV.size))
        if method != "climatology":
            for l in range(LEV.size):
                a = (c.lat[src], c.lon[src], OBS[ch][src, l], c.lat[tgt], c.lon[tgt])
                if method == "nearest_profile":
                    p[:, l] = nearest_profile(*a)
                else:
                    L_km, gamma, k = setting[ch][BAND_OF[l]]
                    p[:, l] = oi_points(*a, L_km=L_km, gamma=gamma, k=k)
        out[ch] = p
    return out


def _score_month(job):
    """Worker: one month's predictions at its scored cells."""
    ev, method, setting = job
    pred = predict(ev, method, setting)
    return {ch: pred[ch][ev["prof"], ev["lev"]] for ch in CH}


def score(evs, method, setting=None):
    s = Scores(LEV, norm.std)
    cells = POOL.map(_score_month, [(ev, method, setting) for ev in evs])
    for ev, cell in zip(evs, cells):
        for ch in CH:
            s.add(ch, cell[ch], ev["target"][ch], ev["lev"])
    return s.result()


# ------------------------------------------------------------------ tuning
def _sweep_month(job):
    """Worker: one month's squared error of every setting.

    Returns (n_settings, n_channels, n_bands). The k-NN geometry of a level is
    built once at the largest k and sliced for the smaller ones.
    """
    ev, combos = job
    kmax = max(k for _, _, k in combos)
    src, tgt = ev["src"], ev["tgt"]
    sse = np.zeros((len(combos), len(CH), len(BAND_NAMES)))
    for ci, ch in enumerate(CH):
        for l in range(LEV.size):
            cells = ev["lev"] == l
            rows, t = ev["prof"][cells], ev["target"][ch][cells]
            ok = np.isfinite(t)
            rows, t = rows[ok], t[ok]
            if rows.size == 0:
                continue
            # the analysis is needed only at the query profiles scored here
            need, inv = np.unique(rows, return_inverse=True)
            sweep, v = point_sweep(c.lat[src], c.lon[src], OBS[ch][src, l],
                                   c.lat[tgt][need], c.lon[tgt][need], kmax)
            subs = {}
            b = BAND_NAMES.index(BAND_OF[l])
            for i, (L_km, gamma, k) in enumerate(combos):
                if k not in subs:
                    subs[k] = sweep.sub_k(k)
                z = subs[k].analyse(v, L_km, gamma)[inv]
                sse[i, ci, b] += float(((z - t) ** 2).sum())
    return sse


def band_counts(evs):
    n = np.zeros((len(CH), len(BAND_NAMES)), dtype=int)
    for ev in evs:
        band = BAND_OF[ev["lev"]]
        for ci, ch in enumerate(CH):
            ok = np.isfinite(ev["target"][ch])
            for bi, b in enumerate(BAND_NAMES):
                n[ci, bi] += int((ok & (band == b)).sum())
    return n


def tune(evs, grid, extend=True):
    """Validation squared error of the whole grid, grown by the edge rule.

    Returns (axes, keys, sse, best, extensions): ``keys`` the settings as
    (L_km, gamma, k) tuples, ``sse`` (n_settings, n_channels, n_bands), and
    ``best[(ci, bi)]`` the selected setting of channel ci in band bi
    (``bi = None``: one setting for the whole column).
    """
    axes = {a: sorted(grid[a]) for a in AXES}
    ladder = {a: {d: list(v) for d, v in LADDER[a].items()} for a in AXES}
    done, extensions = {}, []
    while True:
        combos = [x for x in itertools.product(*(axes[a] for a in AXES))
                  if x not in done]
        if combos:
            tot = sum(POOL.map(_sweep_month, [(ev, combos) for ev in evs]))
            done.update({x: tot[i] for i, x in enumerate(combos)})
            print(f"  tuned {len(combos)} settings ({len(done)} in all, "
                  f"{time.time() - t0:.0f}s)", flush=True)
        keys = sorted(done)
        sse = np.stack([done[x] for x in keys])
        best = {(ci, bi): keys[int(np.argmin(sse[:, ci, bi]))]
                for ci in range(len(CH)) for bi in range(len(BAND_NAMES))}
        best.update({(ci, None): keys[int(np.argmin(sse[:, ci, :].sum(axis=1)))]
                     for ci in range(len(CH))})
        grown = set()
        if extend:
            ends = {a: {"lo": axes[a][0], "hi": axes[a][-1]} for a in AXES}
            for (ci, bi), x in best.items():
                for ai, a in enumerate(AXES):
                    for d in ("lo", "hi"):
                        if x[ai] == ends[a][d] and ladder[a][d] and (a, d) not in grown:
                            new = ladder[a][d].pop(0)
                            axes[a] = sorted(axes[a] + [new])
                            grown.add((a, d))
                            extensions.append({
                                "axis": a, "value": new,
                                "for": f"{CH[ci]} "
                                       f"{BAND_NAMES[bi] if bi is not None else 'all bands'}"})
        if not grown:
            return axes, keys, sse, best, extensions


def rmse(sq, n):
    return float(np.sqrt(sq / n)) if n else float("nan")


# ------------------------------------------------------------------ run
POOL = get_context("fork").Pool(max(1, args.workers))
val = eval_set("validation", EVAL_CELLS, n_months=1 if args.smoke else None)
test = [] if args.smoke else eval_set("development")
print(f"synthetic cohort: {len(val)} validation months, {len(test)} test months, "
      f"{N_INPUT} inputs and {N_QUERY} queries a month, {LEV.size} levels "
      f"({time.time() - t0:.0f}s)", flush=True)


def identity(evs, split):
    """A zero prediction must reproduce the trained model's stored fingerprint."""
    z = score(evs, "climatology")
    ref = json.load(open(os.path.join(ROOT, REFERENCE)))["scores"][split]
    rep = {}
    for ch in CH:
        rep[ch] = {"n": z[ch]["n"], "climatology_z": z[ch]["climatology_z"],
                   "reference_n": ref[ch]["n"],
                   "reference_climatology_z": ref[ch]["climatology_z"]}
        # the model's sums were accumulated in float32, hence 1e-5 and not exact
        if (z[ch]["n"] != ref[ch]["n"]
                or abs(z[ch]["climatology_z"] / ref[ch]["climatology_z"] - 1.0) > 1e-5):
            raise SystemExit(f"identity check failed on {split} {ch}: {rep[ch]}")
    return rep


ident = {}
if not args.smoke:
    ident = {"validation": identity(val, "validation"),
             "development": identity(test, "development")}
    print("identity check passed: zero prediction matches "
          f"{REFERENCE} on both splits", flush=True)

grid = ({"L_km": [250.0, 600.0], "gamma": [0.03, 0.1], "k": [10]}
        if args.smoke else GRID)
axes, keys, sse, best, extensions = tune(val, grid, extend=not args.smoke)
counts = band_counts(val)
ends = {a: (axes[a][0], axes[a][-1]) for a in AXES}


def describe(x, sq, n):
    return {"L_km": float(x[0]), "gamma": float(x[1]), "k": int(x[2]),
            "val_rmse_z": rmse(sq, n),
            "on_edge": [a for i, a in enumerate(AXES) if x[i] in ends[a]]}


sel_band = {ch: {b: best[(ci, bi)] for bi, b in enumerate(BAND_NAMES)}
            for ci, ch in enumerate(CH)}
sel_single = {ch: {b: best[(ci, None)] for b in BAND_NAMES}
              for ci, ch in enumerate(CH)}
oi_selection = {ch: {b: describe(best[(ci, bi)],
                                 sse[keys.index(best[(ci, bi)]), ci, bi], counts[ci, bi])
                     for bi, b in enumerate(BAND_NAMES)} for ci, ch in enumerate(CH)}
oi_single_selection = {ch: describe(best[(ci, None)],
                                    sse[keys.index(best[(ci, None)]), ci].sum(),
                                    counts[ci].sum()) for ci, ch in enumerate(CH)}

baselines = {}
for name, method, setting in (("climatology", "climatology", None),
                              ("nearest_profile", "nearest_profile", None),
                              ("oi", "oi", sel_band), ("oi_single", "oi", sel_single)):
    sc = {"validation": score(val, method, setting)}
    if test:
        sc["development"] = score(test, method, setting)
    baselines[name] = {"scores": sc}
    print(f"  scored {name} ({time.time() - t0:.0f}s)", flush=True)

# the tuned error and the scored error are the same cells: they must agree
for ch in CH:
    for b in BAND_NAMES:
        a = baselines["oi"]["scores"]["validation"][ch]["by_band_z"][b]
        if abs(a - oi_selection[ch][b]["val_rmse_z"]) > 1e-9:
            raise SystemExit(f"tuning and scoring disagree on {ch} {b}: "
                             f"{a} vs {oi_selection[ch][b]['val_rmse_z']}")

split = "validation" if args.smoke else "development"
print(f"\n{split}:  TEMP degC / SALT PSU / J TEMP / J SALT")
for name, b in baselines.items():
    s = b["scores"][split]
    print(f"  {name:16s} {s['TEMP']['rmse_physical']:.4f} / {s['SALT']['rmse_physical']:.4f}"
          f" / {s['TEMP']['J']:.3f} / {s['SALT']['J']:.3f}")
for ch in CH:
    for b in BAND_NAMES:
        o = oi_selection[ch][b]
        print(f"  OI {ch} {b:10s} L={o['L_km']:.0f} km gamma={o['gamma']} k={o['k']}"
              f" val rmse_z={o['val_rmse_z']:.4f}"
              + (f"  ON EDGE {o['on_edge']}" if o["on_edge"] else ""))
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
    "tag": "fixed_baselines", "region": "synthetic", "target": "anomaly_exact",
    "splits": {k: list(v) for k, v in SPLITS.items()},
    "cohort": {"path": os.path.relpath(cpath, ROOT), "sha256": sha256(cpath)},
    "git_commit": git_commit(),
    "inputs_per_month": N_INPUT, "queries_per_month": N_QUERY,
    "eval_cells": EVAL_CELLS,
    "per_month": {name: {str(ev["month"]): {"inputs": int(ev["src"].size),
                                            "queries": int(ev["tgt"].size)}
                         for ev in evs}
                  for name, evs in (("validation", val), ("development", test))},
    "identity_check": {"reference": REFERENCE, **ident},
    "oi_selection": oi_selection, "oi_single_selection": oi_single_selection,
    "baselines": baselines, "runtime_s": time.time() - t0}
tuning = {
    "axes": axes, "bands": BAND_NAMES, "extensions": extensions,
    "cells": {ch: {b: int(counts[ci, bi]) for bi, b in enumerate(BAND_NAMES)}
              for ci, ch in enumerate(CH)},
    "grid": [{"L_km": float(x[0]), "gamma": float(x[1]), "k": int(x[2]),
              "rmse_z": {ch: {**{b: rmse(sse[i, ci, bi], counts[ci, bi])
                                 for bi, b in enumerate(BAND_NAMES)},
                              "all": rmse(sse[i, ci].sum(), counts[ci].sum())}
                         for ci, ch in enumerate(CH)}}
             for i, x in enumerate(keys)]}
os.makedirs(OUT, exist_ok=True)
for name, obj in (("summary.json", summary), ("tuning.json", tuning)):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)
print(f"wrote {OUT}/summary.json and tuning.json ({time.time() - t0:.0f}s)")
```

- [ ] **Step 2: Smoke run**

Run: `./.venv/bin/python experiments/synthetic/44_synth_argo_oi.py --smoke 2>&1 | tail -20`
Expected: a line `tuned 4 settings (4 in all, …)`, four `scored …` lines, a `validation:` table in which `climatology` has `J` 1.000 / 1.000 and `oi` has `J` below 1 for both variables, and the last line `smoke ok (…s), nothing written`. `git status --short` shows only the new script.

- [ ] **Step 3: Commit**

```bash
git add experiments/synthetic/44_synth_argo_oi.py
git commit -q -m "44_synth_argo_oi: fixed baselines scored on the synthetic models' cells

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Run the baselines and commit the results

**Files:**
- Create (generated): `outputs/audit/synthetic/fixed_baselines/summary.json`, `tuning.json`

- [ ] **Step 1: Full run**

Run: `./.venv/bin/python experiments/synthetic/44_synth_argo_oi.py 2>&1 | tee logs/synth_oi.log | tail -40`
Expected, in order: the cohort line with `12 validation months, 12 test months, 6080 inputs and 1520 queries a month`; `identity check passed`; one or more `tuned N settings` lines (72 first; more only if the edge rule fires); four `scored` lines; a `development:` table; eight `OI …` selection lines; `wrote …/summary.json and tuning.json`.
If it exits with `input parity violated` or `identity check failed`, stop: the evaluation set does not match the models'. Compare `eval_month` with `make_sample`/`build_eval` in `62_sanity_train.py` before going on.

- [ ] **Step 2: Check the output against the guards**

Run:
```bash
./.venv/bin/python - <<'EOF'
import json
s = json.load(open("outputs/audit/synthetic/fixed_baselines/summary.json"))
t = json.load(open("outputs/audit/synthetic/fixed_baselines/tuning.json"))
pm = {(v["inputs"], v["queries"]) for sp in s["per_month"].values() for v in sp.values()}
print("inputs/queries per month:", pm)
for sp in ("validation", "development"):
    for ch in ("TEMP", "SALT"):
        i = s["identity_check"][sp][ch]
        print(sp, ch, "n", i["n"], "=", i["reference_n"], "| clim_z", round(i["climatology_z"], 6),
              "vs", round(i["reference_climatology_z"], 6))
for name, b in s["baselines"].items():
    d = b["scores"]["development"]
    print(f"{name:16s} TEMP {d['TEMP']['rmse_physical']:.4f} degC J {d['TEMP']['J']:.3f} | "
          f"SALT {d['SALT']['rmse_physical']:.4f} PSU J {d['SALT']['J']:.3f} | n {d['TEMP']['n']}")
print("grid settings:", len(t["grid"]), "| extensions:", t["extensions"])
print("still on an edge:", [(ch, b, o["on_edge"]) for ch, d in s["oi_selection"].items()
                            for b, o in d.items() if o["on_edge"]])
EOF
```
Expected: `inputs/queries per month: {(6080, 1520)}`; `n` equal to the reference on all four lines (351,895 on `development`, 92,517 on `validation`); every method's test `n` is 351895; `climatology` has `J 1.000`. Record the OI and nearest-profile numbers, the extensions and any selection still on an edge for the final report.

- [ ] **Step 3: Commit the results**

```bash
git add outputs/audit/synthetic/fixed_baselines/summary.json outputs/audit/synthetic/fixed_baselines/tuning.json
git commit -q -m "Fixed baselines on the synthetic cohort: climatology, nearest profile, tuned OI

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Report — todo status table, finding 9, §9

`43_synth_argo_report.py` builds a list `md`; the findings list `find` is inserted at index 3 by the line `md[3:3] = find` near the end. Sections 1-8 and findings 1-8 must not change.

**Files:**
- Modify: `experiments/synthetic/43_synth_argo_report.py`

- [ ] **Step 1: Confirm the report regenerates unchanged before editing**

Run:
```bash
./.venv/bin/python experiments/synthetic/43_synth_argo_report.py && git status --short reports/
```
Expected: `wrote …/synth_argo_audit.md` and no change to `reports/synthetic/synth_argo_audit.md`. If the two `fig_synth_*.png` files show as modified, that is only a rendering difference of this machine's matplotlib; restore them with `git checkout -- reports/synthetic/fig_synth_overfit.png reports/synthetic/fig_synth_refiner.png`. If the markdown itself changed, stop and report the diff: the downloaded outputs would not reproduce the committed report.

- [ ] **Step 2: Add §9, the parity checks and the todo table**

In `experiments/synthetic/43_synth_argo_report.py`, insert the following block immediately before the line `# ============================================ findings, one line per plan item`:

```python
# ============================================ 9. fixed baselines (OI)
FBP = os.path.join(OUTD, "fixed_baselines", "summary.json")
FB = json.load(open(FBP)) if os.path.exists(FBP) else None
N_IN = T["per_month"] - T["queries_per_month"]
BK = ("0-100m", "100-300m", "300-700m", "700-1400m")   # band keys in the summaries


def parity(tags):
    """No cited arm may have seen fewer Argo profiles, a different schedule or
    split, or be scored on other cells, than the validated configuration."""
    def key(s):
        return (s["n_profiles"], s["steps"], s["batch"],
                json.dumps(s["splits"], sort_keys=True),
                *[s["scores"][sp][ch]["n"] for sp in ("validation", "development")
                  for ch in CH])
    ref = summary(FIX, SEEDS[0])
    if ref["n_profiles"] != 0:
        raise SystemExit(f"{FIX} was not run on every input profile")
    bad = [f"{t}/s{x}" for t in tags for x in seeds_done(t)
           if key(summary(t, x)) != key(ref)]
    if bad:
        raise SystemExit(f"input parity violated against {FIX}: {bad}")


def fb_parity():
    """The fixed baselines saw the models' inputs and are scored on their cells."""
    want = (N_IN, T["queries_per_month"])
    got = {(v["inputs"], v["queries"]) for sp in FB["per_month"].values()
           for v in sp.values()}
    if got != {want} or (FB["inputs_per_month"], FB["queries_per_month"]) != want:
        raise SystemExit(f"fixed baselines saw {got}, the models {want}")
    ref = summary(FIX, SEEDS[0])["scores"]
    for name, b in FB["baselines"].items():
        for sp in ("validation", "development"):
            for ch in CH:
                if b["scores"][sp][ch]["n"] != ref[sp][ch]["n"]:
                    raise SystemExit(f"{name} is scored on other cells than {FIX} "
                                     f"({sp} {ch})")


TODO_TAGS = (["syn_reg_cell"] + [t for t, _ in SWEEP]
             + ["syn_fix_l64", "syn_fix_uniform", "syn_fix_count"])
parity(TODO_TAGS)
N_TEST = summary(FIX, SEEDS[0])["scores"]["development"]["TEMP"]["n"]


def fbv(name, ch, key="rmse_physical", split="development"):
    return float(FB["baselines"][name]["scores"][split][ch][key])


def model(ch, key="rmse_physical"):
    return [val(FIX, x, "development", ch, key) for x in SEEDS]


def gap(ch):
    """model - OI on test: (difference, per cent of OI, seed sd of the model)."""
    m = model(ch)
    d = float(np.mean(m)) - fbv("oi", ch)
    return d, 100.0 * d / fbv("oi", ch), float(np.std(m, ddof=1))


def verdict():
    g = [gap(ch) for ch in CH]
    if all(d < -2 * sd for d, _, sd in g):
        return "the model beats OI on both variables"
    if all(d > 2 * sd for d, _, sd in g):
        return "the model loses to OI on both variables"
    return "; ".join(
        f"{ch}: the model is "
        + ("better" if d < -2 * sd else "worse" if d > 2 * sd
           else "tied with OI (within 2 seed sd)")
        for ch, (d, _, sd) in zip(CH, g))


md += ["## 9. Fixed baselines: climatology, nearest profile, optimal interpolation\n"]
if FB is None:
    md += ["Pending: run `experiments/synthetic/44_synth_argo_oi.py`.\n"]
else:
    fb_parity()
    rows = [[lab, f"{N_IN:,}", tuned, f(fbv(n, "TEMP"), 4), f(fbv(n, "SALT"), 4),
             f(fbv(n, "TEMP", "J"), 3), f(fbv(n, "SALT", "J"), 3)]
            for n, lab, tuned in (
                ("climatology", "climatology (zero anomaly)", "—"),
                ("nearest_profile", "nearest profile", "—"),
                ("oi_single", "OI, one setting per variable", "validation 2004"),
                ("oi", "**OI, tuned per variable and depth band**", "validation 2004"))]
    rows.append([f"model `{FIX}` (mean ± sd, {len(seeds_done(FIX))} seeds)", f"{N_IN:,}",
                 "validation 2004", mean_sd(model("TEMP")), mean_sd(model("SALT")),
                 mean_sd(model("TEMP", "J")), mean_sd(model("SALT", "J"))])
    gT, gS = gap("TEMP"), gap("SALT")
    md += ["Methods with no trainable parameters, written by "
           "`experiments/synthetic/44_synth_argo_oi.py` and scored on the cells the "
           f"models are scored on: the same {N_IN:,} input profiles a month (every "
           f"method is given all of them), the same {T['queries_per_month']:,} query "
           f"profiles, the at-position target, the train-year normalisation, and "
           f"{N_TEST:,} test cells per variable. A zero prediction reproduces the "
           "cell counts and climatology error stored in the model summaries, and "
           "this report refuses to render if any cited arm saw a different number "
           "of profiles.\n",
           "OI is `ocean_tokenizer.oi` (Bretherton et al. 1976): level by level, "
           "Gaussian covariance in great-circle distance, the k nearest profiles. "
           "Length scale, noise ratio and k are selected on the validation year, "
           "one setting per variable and depth band; the test year is scored once.\n",
           md_table(["method", "inputs / month", "selected on", "test TEMP °C",
                     "test SALT PSU", "J TEMP", "J SALT"], rows),
           f"Model − OI on test: TEMP {gT[0]:+.4f} °C ({gT[1]:+.1f} %, model seed sd "
           f"{gT[2]:.4f}), SALT {gS[0]:+.5f} PSU ({gS[1]:+.1f} %, seed sd "
           f"{gS[2]:.5f}). Negative means the model is better.\n"]
    rows = []
    for ch in CH:
        oi = [float(FB["baselines"]["oi"]["scores"]["development"][ch]["by_band_z"][b])
              for b in BK]
        mo = [float(np.mean([summary(FIX, x)["scores"]["development"][ch]["by_band_z"][b]
                             for x in SEEDS])) for b in BK]
        rows.append([ch, "OI"] + [f(v, 4) for v in oi])
        rows.append([ch, "model (mean of seeds)"] + [f(v, 4) for v in mo])
        rows.append([ch, "model − OI"] + [f"{m - o:+.4f}" for m, o in zip(mo, oi)])
    md += ["By depth band, test RMSE in z units:\n",
           md_table(["", "", *BK], rows)]
    rows = [[ch, b, f"{o['L_km']:.0f}", o["gamma"], o["k"], f(o["val_rmse_z"], 4),
             ", ".join(o["on_edge"]) or "—"]
            for ch in CH for b, o in FB["oi_selection"][ch].items()]
    md += ["Selected OI settings:\n",
           md_table(["", "band", "L (km)", "gamma", "k", "validation RMSE z",
                     "on a grid edge"], rows)]

```

- [ ] **Step 3: Add finding 9 and the todo status table**

In the same file, replace the line

```python
md[3:3] = find
```

with:

```python
if FB is not None:
    gT, gS = gap("TEMP"), gap("SALT")
    find.append(
        f"9. **Fixed baselines** — on the same {N_IN:,} input profiles a month and "
        f"the same {N_TEST:,} test cells, validation-tuned OI scores "
        f"{fbv('oi', 'TEMP'):.4f} °C / {fbv('oi', 'SALT'):.4f} PSU and the model "
        f"{mean_sd(model('TEMP'))} °C / {mean_sd(model('SALT'))} PSU: {verdict()} "
        f"(model − OI {gT[0]:+.4f} °C, {gT[1]:+.1f} %; {gS[0]:+.5f} PSU, "
        f"{gS[1]:+.1f} %). Nearest profile: {fbv('nearest_profile', 'TEMP'):.4f} °C / "
        f"{fbv('nearest_profile', 'SALT'):.4f} PSU; climatology: "
        f"{fbv('climatology', 'TEMP'):.4f} °C / {fbv('climatology', 'SALT'):.4f} PSU.\n")


def tm(tag, ch):
    return float(np.mean([val(tag, x, "development", ch) for x in SEEDS]))


def pc(new, old):
    return f"{100.0 * (new - old) / old:+.0f} %"


todo_rows = [
    ["Anomaly target at the Argo position", "`syn_reg_cell` → `syn_reg`", f"{N_IN:,}", "3",
     f"{tm('syn_reg_cell', 'TEMP'):.3f} → {tm('syn_reg', 'TEMP'):.3f} °C "
     f"({pc(tm('syn_reg', 'TEMP'), tm('syn_reg_cell', 'TEMP'))}), "
     f"{tm('syn_reg_cell', 'SALT'):.4f} → {tm('syn_reg', 'SALT'):.4f} PSU "
     f"({pc(tm('syn_reg', 'SALT'), tm('syn_reg_cell', 'SALT'))})", "§5"],
    ["Compare with a fixed baseline (OI)", f"`fixed_baselines` vs `{FIX}`", f"{N_IN:,}",
     "3 (model)",
     ("pending" if FB is None else
      f"OI {fbv('oi', 'TEMP'):.4f} °C / {fbv('oi', 'SALT'):.4f} PSU; model "
      f"{mean_sd(model('TEMP'))} °C / {mean_sd(model('SALT'))} PSU; {verdict()}"),
     "§9"],
    ["Local refiner sweep, 3 seeds each",
     f"{len(SWEEP)} inits, `syn_reg` … `syn_r1500_g1`", f"{N_IN:,}", "3",
     f"selected on validation: `{FIX}` (500 km / 100 m, gate 1.0), "
     f"{mean_sd(model('TEMP'))} °C; vs the registered init "
     f"{fmt_pair(paired(FIX, REF, 'development', 'TEMP'))} °C", "§4"],
    ["Slots 32 → 64", f"`{FIX}` → `syn_fix_l64`", f"{N_IN:,}", "3",
     f"{fmt_pair(paired('syn_fix_l64', FIX, 'development', 'TEMP'))} °C, "
     f"{fmt_pair(paired('syn_fix_l64', FIX, 'development', 'SALT'), 5)} PSU", "§7"],
    ["DFS vs uniform vs count", f"`{FIX}`, `syn_fix_uniform`, `syn_fix_count`",
     f"{N_IN:,}", "3",
     f"uniform {fmt_pair(paired('syn_fix_uniform', FIX, 'development', 'TEMP'))} °C, "
     f"count {fmt_pair(paired('syn_fix_count', FIX, 'development', 'TEMP'))} °C "
     f"against DFS", "§8"]]
todo = ["## Todo status (2026-10-04)\n",
        f"The five items of the 2026-10-04 todo, all on CESM2 synthetic data. Every "
        f"arm below was given the same {N_IN:,} input profiles a month, 12 k steps "
        f"where it is trained, the same splits, and is scored on the same "
        f"{N_TEST:,} test cells per variable; the report checks this when it is "
        f"generated. Test year 2005; Δ are paired over seeds.\n",
        md_table(["todo item", "arms", "inputs / month", "seeds", "test result",
                  "section"], todo_rows)]
md[3:3] = todo + find
```

- [ ] **Step 4: Regenerate and verify the diff**

Run:
```bash
./.venv/bin/python experiments/synthetic/43_synth_argo_report.py
git diff --stat reports/synthetic/synth_argo_audit.md
git diff reports/synthetic/synth_argo_audit.md | grep '^-' | grep -v '^---'
```
Expected: the script prints `wrote …`; `--stat` shows insertions only; the last command prints nothing (no line of sections 1-8 or findings 1-8 was removed or changed). Restore the PNGs as in Step 1 if they show as modified. Then read the new parts:

```bash
sed -n '1,30p' reports/synthetic/synth_argo_audit.md
sed -n '/^## 9\./,$p' reports/synthetic/synth_argo_audit.md
```
Check that the todo table has five rows, each showing 6,080 inputs a month, with numbers (no `—`, no `pending`), that finding 9 is present, and that the numbers in §9 equal those printed in Task 6 Step 2.

- [ ] **Step 5: Prove the parity check bites**

Run:
```bash
./.venv/bin/python - <<'EOF'
import json, shutil, subprocess
p = "outputs/audit/synthetic/syn_fix_l64/summary_seed1234.json"
shutil.copy(p, p + ".bak")
try:
    s = json.load(open(p)); s["n_profiles"] = 1000
    json.dump(s, open(p, "w"))
    r = subprocess.run(["./.venv/bin/python", "experiments/synthetic/43_synth_argo_report.py"],
                       capture_output=True, text=True)
    print("exit", r.returncode, "|", (r.stderr or r.stdout).strip().splitlines()[-1])
finally:
    shutil.move(p + ".bak", p)
EOF
git status --short
```
Expected: `exit 1 | input parity violated against syn_r500_g1: ['syn_fix_l64/s1234']`, and `git status --short` shows only the intended changes (the summary file is restored byte for byte; the report was not rewritten by the failed run). Restore the PNGs as in Step 1 if they show as modified.

- [ ] **Step 6: Commit**

```bash
git add experiments/synthetic/43_synth_argo_report.py reports/synthetic/synth_argo_audit.md
git commit -q -m "Synthetic audit report: fixed baselines (section 9) and the todo status table

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: README and final verification

**Files:**
- Modify: `README.md` (the synthetic audit block, after the `syn_final` queue line)

- [ ] **Step 1: Add the command to the README**

In `README.md`, replace the line

```
.venv/bin/python experiments/synthetic/43_synth_argo_report.py # -> reports/synthetic/synth_argo_audit.md
```

with:

```
.venv/bin/python experiments/synthetic/44_synth_argo_oi.py     # fixed baselines: climatology, nearest profile, OI
.venv/bin/python experiments/synthetic/43_synth_argo_report.py # -> reports/synthetic/synth_argo_audit.md
```

- [ ] **Step 2: Full test suite**

Run: `./.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: the Task 1 baseline count plus 13, all passed.

- [ ] **Step 3: Final state check**

Run: `git status --short && git log --oneline main..HEAD`
Expected: a clean tree apart from the README edit, and the commits of Tasks 2-7 plus the spec and plan commits.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -q -m "README: fixed-baselines command in the synthetic audit

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Report to the user**

State, with the measured numbers: the test scores of climatology, nearest profile, OI and the model; model − OI with its percentage and the verdict; the selected OI settings and whether any sits on a grid edge; that the parity and identity checks passed; the test count; and that nothing was pushed.
