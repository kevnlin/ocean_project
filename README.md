# Ocean Tokenizer — Sparse-Profile Subsurface Ocean Reconstruction

Reconstruct the full 3-D subsurface **temperature & salinity** field from sparse,
Argo-like profiles. The problem is framed as an **observing-system simulation
experiment (OSSE)** on CESM2-LE, with WOA23 as an observational prior: the model
climate is the (fully known) ground truth, so reconstructions can be scored exactly.

## What's here

- **Unified data pipeline** — CESM2-LE (ground truth) + WOA23 (prior) standardized to a
  common 1° / 20-level grid as Zarr (`experiments/standardize.py`, `ocean_tokenizer.data`).
- **Four lossless tokenizers** — grid-patch, volume-patch, vertical-profile, point-query
  (`decode(encode(field)) == field` exactly; `ocean_tokenizer.tokenizers`).
- **Synthetic Argo sampling + baseline sweep** — climatology, nearest-profile, pointwise
  MLP, depthwise 2-D U-Net across input configs (`ocean_tokenizer.baselines`).
- **Three re-implemented literature reference models** — NeSPReSO (PCA+MLP), OSnet
  (15× bootstrap-ensemble MLP), Buongiorno-Nardelli stacked-LSTM (`src/baselines/`).
- **Depth-banded RMSE comparison** of every method on identical held-out data.

## Headline result (held-out CESM2-LE)

Our method of choice — the multi-modal **2-D U-Net (profiles + WOA + SST/SSS)** — has the
best full-column RMSE of every method (**0.47 °C / 0.10 PSU**) and the best skill through
the 50–200 m thermocline. Surface-only reference models lead only in the upper ocean.
Full tables: [`reports/baseline_comparison.md`](reports/baseline_comparison.md) ·
[`reports/baseline_table.md`](reports/baseline_table.md) ·
Stage-1 write-up: [`reports/final_report.md`](reports/final_report.md).

## Layout

```
src/ocean_tokenizer/   core package: config, data, tokenizers, argo, unet, baselines, metrics
src/baselines/         reference models (nesperso_pcamlp / osnet_mlp / nardelli_lstm) + README
                       + build_comparison_table.py
experiments/           runnable scripts 00–05 + standardize.py
reports/               generated markdown / CSV reports and data cards
checkpoints/           trained model weights (small; tracked)
data/, processed/      raw NetCDF + standardized Zarr  (NOT tracked — see "Data" below)
```

## Setup

Python 3.12; a single GPU is used if available (CPU works for smoke runs).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

```bash
# core pipeline
python experiments/00_data_cards.py            # data cards + common grid
python experiments/01_tokenizer_roundtrip.py   # lossless round-trip check
python experiments/02_synth_argo.py            # synthetic Argo example
python experiments/03_baselines.py             # baseline sweep (--smoke for a fast check)
python experiments/05_band_table.py            # -> reports/baseline_table.md

# reference baselines (each --smoke for a quick check)
python src/baselines/nesperso_pcamlp.py
python src/baselines/osnet_mlp.py
python src/baselines/nardelli_lstm.py
python src/baselines/build_comparison_table.py # -> reports/baseline_comparison.{md,csv}
```

See [`src/baselines/README.md`](src/baselines/README.md) for the reference-model details
and their documented departures from the original papers (chiefly: no SSH/ADT input).

## Experimental setup

| | |
|---|---|
| Ground truth | CESM2-LE full simulation (regular 1°, 180×360, 20 levels 5–985 m) |
| Prior | WOA23 monthly climatology |
| Synthetic Argo | 1500 random ocean columns / month (~3.5% coverage) |
| Split | 48 train months (1985–2010) / 12 test months (2011–2014), fixed seed 1234 |
| Metric | depth-banded RMSE — valid-cell-weighted, NaN-aware, ocean only |

## Data

The raw NetCDF (`data/`, ~4.7 GB) and standardized Zarr stores (`processed/`, ~42 GB) are
**not tracked in git**. Place the source CESM2-LE / WOA23 files under `data/` and run
`python experiments/standardize.py` to regenerate `processed/`. The large per-cell
prediction arrays (`predictions/*.npz`) are also untracked (they exceed GitHub's file-size
limit); the RMSE tables/CSVs and the trained checkpoints that produce them are included.

> **Note on numbers:** an earlier task brief cited 500 floats / 96–24 months / 30 levels;
> the live pipeline uses **1500 profiles/month, 48/12 months, 20 levels** — all reports
> reflect the live configuration.
