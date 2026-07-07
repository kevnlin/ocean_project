"""Central configuration: paths, common grid definition, and experiment knobs.

The standardized zarr stores already share dims (time, depth, lat, lon) and
variables (TEMP, SALT, SST, SSS).  This module pins down the *common analysis
grid* used by the tokenizer / baseline pipeline so every component agrees.
"""
from __future__ import annotations
import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = "/home/nvidia/ocean_project"
PROCESSED = os.path.join(ROOT, "processed")
OUTPUTS = os.path.join(ROOT, "outputs")
CACHE = os.path.join(OUTPUTS, "cache")
CKPT = os.path.join(OUTPUTS, "ckpt")
REPORTS = os.path.join(ROOT, "reports")

ZARR = {
    "cesm2":         os.path.join(PROCESSED, "cesm2_standard.zarr"),         # single member (curvilinear placeholder grid)
    "woa23":         os.path.join(PROCESSED, "woa23_standard.zarr"),         # observational climatology (prior/baseline)
    "cesm2_le_full": os.path.join(PROCESSED, "cesm2_le_full_standard.zarr"), # full LE simulation, regular 1deg (ground truth)
}

# --------------------------------------------------------------------------
# Common analysis grid
# --------------------------------------------------------------------------
# CESM2-LE (ground truth) and WOA23 share the exact 1deg lat grid (-89.5..89.5).
# LE lon is 0-360, WOA lon is -180..180; we standardise the analysis grid to the
# LE convention (0.5..359.5) and roll WOA onto it.
LON_CONVENTION = "0-360"          # analysis grid longitude convention

# Ground-truth depth levels: a curated subset of native CESM2-LE levels covering
# the surface to ~1000 m.  Using native LE levels keeps the GT lossless (no
# vertical interpolation of the target); WOA23 is interpolated onto these.
GT_SOURCE = "cesm2_le_full"
DEPTH_INDICES = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 21, 24, 27, 30, 33, 36, 39]
#   -> depths ~ 5,15,25,35,45,55,65,85,105,125,145,165,186,222,267,327,408,527,707,985 m

# --------------------------------------------------------------------------
# Experiment subset (kept modest: this is a prototype, not a foundation model)
# --------------------------------------------------------------------------
TRAIN_YEARS = (1985, 2010)        # inclusive range to sample training months from
TEST_YEARS = (2011, 2014)         # held-out evaluation period
N_TRAIN_MONTHS = 48               # monthly snapshots sampled from TRAIN_YEARS
N_TEST_MONTHS = 12                # monthly snapshots from TEST_YEARS

N_PROFILES = 1500                 # synthetic Argo columns sampled per monthly field
PROFILE_MAX_DEPTH_IDX = None      # None -> full target depth column per profile

SEED = 1234

VARS_3D = ["TEMP", "SALT"]        # volumetric variables to reconstruct
VARS_SURF = ["SST", "SSS"]        # surface variables (dense observations)

# physically plausible ranges; values outside -> NaN (drops fill/brine outliers)
PHYS_RANGE = {
    "TEMP": (-3.0, 40.0), "SST": (-3.0, 40.0),   # degC
    "SALT": (2.0, 45.0),  "SSS": (2.0, 45.0),    # PSU
}

# --------------------------------------------------------------------------
# Model knobs
# --------------------------------------------------------------------------
MLP_HIDDEN = [256, 256, 256]
MLP_EPOCHS = 30
MLP_BATCH = 65536
MLP_LR = 1e-3
MLP_POINTS_PER_MONTH = 120_000    # subsample of ocean points used as MLP targets

UNET_BASE = 32
UNET_EPOCHS = 40
UNET_LR = 2e-3
UNET_BATCH = 16                   # depth-slices per batch

DEVICE = "cuda"

os.makedirs(CACHE, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)
