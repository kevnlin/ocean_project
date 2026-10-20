"""Post-hoc calibration of the predictive spread: one factor per lead, variable and depth band.

The gridded models' Gaussian spread is too narrow at long leads (on CESM2 2004-2005 the 95 %
intervals of the 6-month forecasts cover about 0.8 of the errors). A factor multiplying the
predicted standard deviation is fitted on the selection years only, for every (lead, variable,
depth band) group, as the factor that minimises the group's mean CRPS in physical units. The
predicted mean is untouched, so every RMSE stays as it was.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

from .point_baselines import BANDS, CH, band_of_levels

#: the factors searched
BOUNDS = (0.25, 4.0)


def record_name(run_folder):
    """The file name of a run's calibration record: ``<campaign folder>__<run folder>.json``."""
    run = Path(run_folder).resolve()
    return f"{run.parent.name}__{run.name}.json"


def crps(mean, std, target):
    """Elementwise CRPS of a Gaussian forecast (the formula of ``reconstruction_metrics``)."""
    error = np.asarray(target, np.float64) - np.asarray(mean, np.float64)
    sigma = np.asarray(std, np.float64)
    z = error / sigma
    phi = np.exp(-.5 * z * z) / np.sqrt(2 * np.pi)
    return error * (2 * ndtr(z) - 1) + sigma * (2 * phi - 1 / np.sqrt(np.pi))


def _bands(level, levels_m):
    return band_of_levels(np.asarray(levels_m), BANDS)[np.asarray(level, int)]


def fit_scales(mean, std, target, lead, level, levels_m):
    """``{lead: {variable: {band: factor}}}`` minimising each group's mean CRPS; arrays are (n, 2)
    in physical units (``mean``, ``std``, ``target``) or (n,) (``lead``, ``level``); missing targets
    are skipped and groups without cells get no factor."""
    bands, lead = _bands(level, levels_m), np.asarray(lead)
    scales = {}
    for value in sorted(set(lead.tolist())):
        scales[str(value)] = {}
        for j, ch in enumerate(CH):
            scales[str(value)][ch] = {}
            for name in dict.fromkeys(band_of_levels(np.asarray(levels_m), BANDS)):
                keep = (lead == value) & (bands == name) & np.isfinite(target[:, j])
                if not keep.any():
                    continue
                m, s, y = (np.asarray(a[keep, j], np.float64) for a in (mean, std, target))
                best = minimize_scalar(lambda g: crps(m, s * g, y).mean(), bounds=BOUNDS, method="bounded",
                                       options={"xatol": 1e-4})
                scales[str(value)][ch][str(name)] = float(best.x)
    return scales


def apply_scales(std, lead, level, levels_m, scales):
    """``std`` times the factor of each cell's (lead, variable, band) group; a cell whose group has
    no factor is an error (``KeyError``)."""
    bands, lead = _bands(level, levels_m), np.asarray(lead)
    factor = np.empty(np.shape(std), np.float64)
    for value in set(lead.tolist()):
        for name in set(bands[lead == value].tolist()):
            rows = (lead == value) & (bands == name)
            for j, ch in enumerate(CH):
                factor[rows, j] = scales[str(value)][ch][name]
    return np.asarray(std, np.float64) * factor
