# Reference baselines (`src/baselines/`)

Three literature reference models for sparse-profile ocean T/S reconstruction,
added alongside the existing baselines (global mean, WOA climatology, nearest
profile, pointwise MLP, 2D U-Net) implemented in
`src/ocean_tokenizer/baselines.py`.

All three reuse the **existing data pipeline** (`ocean_tokenizer.data`,
`ocean_tokenizer.argo`, `ocean_tokenizer.baselines`) through the shared helper
[`_common.py`](./_common.py), so the train/test split, the synthetic Argo
profiles, the depth grid, the ocean mask and the WOA23 prior are **identical**
to the existing baselines (same `config.SEED = 1234`).

## Setup recap (from `ocean_tokenizer/config.py`)

| | |
|---|---|
| Ground truth | CESM2-LE full simulation (regular 1°×1°) |
| Grid | 180 lat × 360 lon, **20 depth levels** (≈5–985 m) |
| Train / test months | 48 (1985–2010) / 12 (2011–2014), seeded draw |
| Synthetic Argo profiles | 1500 ocean columns per month |
| Surface obs | SST / SSS, treated as **dense** (available on every ocean cell) |
| Metric | depth-banded RMSE, valid-cell-weighted, NaN-aware, ocean only |

> Note: an earlier task brief mentioned "500 floats / 96-24 months / 30 levels".
> The live pipeline config is 1500 profiles / 48-12 months / 20 levels — these
> scripts follow the **live config** so they line up with the existing table.

## Common paradigm (no data leakage)

These are *function-fitting* reconstructors: they learn
`(surface SST/SSS + location + time) → subsurface T/S profile`. The sparse Argo
profiles are the **training labels**; the surface fields are **dense inputs**.
At test time we feed each held-out month's dense surface field and predict the
subsurface on every ocean cell, then score NaN-aware against CESM2-LE truth. The
held-out subsurface truth is never an input. Surface SST/SSS used as inputs are
always read **at the profile/cell location**, not from the target field.

CESM2 bottom topography leaves NaNs below the seafloor (~13% of columns). For
the PCA basis and the dense targets we gap-fill each profile (downward
extension + per-depth mean) and **mask the fabricated cells out of the
profile-space losses**; predictions below the seafloor are never scored (truth
is NaN there).

## The three baselines

### 1. `nesperso_pcamlp.py` — NeSPReSO-style PCA + MLP (~283k params)
PCA (15 components each for T and S, ≥99% variance, fit on training profiles
only) compresses profiles to 30 PC scores; an MLP
`in(8) → 512 → 512 → 30` (ReLU, 0.2 dropout) predicts the *unit-variance* PC
scores from surface/location/time, and the fixed PCA basis maps them back to
full-depth profiles. Loss = WMSE (PC-score MSE) + FMSE (reconstructed-profile
MSE), each in a unit-magnitude space. Adam 1e-3, batch 300, ≤8000 epochs, early
stopping (patience 500) on a 10% validation split.
**Departure:** the original NeSPReSO uses satellite SSH/ADT as a predictor — we
have none, so it is dropped (8 inputs instead of 9). Expect degraded mid-depth
(100–600 m) skill, where SSH carries most thermocline information.

### 2. `osnet_mlp.py` — OSnet-style MLP + bootstrap ensemble
(Pauthenet et al., *Ocean Science* 2022,
<https://os.copernicus.org/articles/18/1221/2022/>.)
MLP `in(8) → 256 → 256 → [T(20) | S(20) | K(20)]`; T/S regressed in per-depth
z-space, K is a mixed-layer mask (logits). Loss = MSE(T)+MSE(S)+BCE(K). Trained
as a **15-member bootstrap ensemble** (distinct seed + resample-with-replacement
per member); prediction = ensemble mean, ensemble std = uncertainty.
**Departures:** no SSH/SLA/currents (8 inputs vs the paper's 12); bathymetry is
approximated by the deepest valid CESM2 level; the MLD `K` mask uses a
ΔT = 0.2 °C criterion from the shallowest level.

### 3. `nardelli_lstm.py` — Buongiorno-Nardelli stacked-LSTM + MC-dropout
2-layer LSTM (35 hidden) unrolled over **depth as the sequence dimension**; the
same 6-input vector (SST/SSS anomalies vs WOA23, lat, lon, sin/cos day-of-year,
min-max scaled to [0,1]) is fed at every step, emitting per-depth T/S anomalies
that are added back to the WOA23 climatology. Loss = MSE(T-anom)+MSE(S-anom)
(anomalies z-scored per depth so T and S are balanced). **Monte-Carlo dropout**
(dropout active at inference) with 50 forward passes gives mean + std.
**Departure:** no SSH/SLA inputs; anomalies referenced to WOA23 interpolated
onto the 20 CESM2-LE levels; day-of-year is the monthly snapshot's 15th.

## How to run

```bash
PY=~/.venv/bin/python

# quick correctness check (small months / few epochs / few members or MC passes)
$PY src/baselines/nesperso_pcamlp.py --smoke
$PY src/baselines/osnet_mlp.py       --smoke
$PY src/baselines/nardelli_lstm.py   --smoke

# full runs
$PY src/baselines/nesperso_pcamlp.py
$PY src/baselines/osnet_mlp.py
$PY src/baselines/nardelli_lstm.py

# combined comparison table (existing + new three)
$PY src/baselines/build_comparison_table.py
```

## Shared I/O interface

Every script (via `_common.py`) prints a depth-banded RMSE table
(surface ~5 m | 0–50 m | 50–200 m | 200 m+) for T and S, and writes:

| Artifact | Path |
|---|---|
| Model checkpoint | `checkpoints/<name>.pt` |
| Predicted test fields (+std for ens/MC) | `predictions/<name>.npz` |
| Per-method RMSE CSV | `predictions/<name>_rmse.csv` |
| Per-depth RMSE (for the combined table) | `outputs/cache/ref_<name>_depth.npz` |

`predictions/<name>.npz` holds `TEMP`, `SALT` `(N_test, D, H, W)` (plus
`TEMP_std`/`SALT_std` for the ensemble and MC-dropout models), `depths`, and
`te_idx`. `build_comparison_table.py` reads the per-depth caches and writes
`reports/baseline_comparison.csv` and `reports/baseline_comparison.md`.

`<name>` ∈ {`nesperso_pcamlp`, `osnet_mlp`, `nardelli_lstm`}.
