"""Conventional reconstruction metrics in the units supplied by the caller.

Missing targets are masked independently for each variable. A missing prediction
for a scored target is an error, rather than silently shrinking the denominator.
Bias has the sign prediction minus target. The normal CRPS is evaluated without
an unstable product of a very large standardized residual and a tiny scale.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import ndtr

REGRESSION_KEYS = ("rmse", "mae", "mean_bias", "r2", "pearson_r")
GAUSSIAN_KEYS = ("nll", "crps", "coverage_68", "coverage_95", "mean_std")
COVERAGE_68_NOMINAL = 0.6826894921370859
COVERAGE_95_Z = 1.959963984540054


def _scored(prediction, target, mask=None):
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError("prediction and target shapes must match exactly")
    valid = np.isfinite(y)
    if mask is not None:
        selected = np.asarray(mask)
        if selected.shape != y.shape or selected.dtype.kind != "b":
            raise ValueError("mask must be boolean and match the target shape")
        valid &= selected
    if np.any(~np.isfinite(p[valid])):
        raise ValueError("prediction must be finite for every scored target")
    return p[valid], y[valid], valid


def regression_metrics(prediction, target, mask=None) -> dict:
    """Pooled RMSE, MAE, signed bias, R² and Pearson r over selected values.

    R² is 1 - SSE/SST, including negative values, not squared correlation. It is
    null for a constant target or fewer than two values. Pearson r is additionally
    null for a constant prediction. No missing or constant case is changed to 0.
    """
    p, y, _ = _scored(prediction, target, mask)
    n = len(y)
    result = {"n": n, **{key: None for key in REGRESSION_KEYS}}
    if not n:
        return result
    residual = p - y
    mse = float(np.mean(residual * residual))
    result.update(rmse=float(np.sqrt(mse)), mae=float(np.mean(np.abs(residual))),
                  mean_bias=float(np.mean(residual)))
    if n >= 2:
        yc, pc = y - y.mean(), p - p.mean()
        sst, ssp = float(np.dot(yc, yc)), float(np.dot(pc, pc))
        if sst > 0:
            result["r2"] = float(1.0 - np.dot(residual, residual) / sst)
        if sst > 0 and ssp > 0:
            result["pearson_r"] = float(np.clip(np.dot(yc, pc) / np.sqrt(sst) / np.sqrt(ssp), -1., 1.))
    return result


def gaussian_metrics(mean, std, target, mask=None) -> dict:
    """Gaussian NLL, CRPS, one-sigma/95% coverage and mean predictive std.

    NLL uses natural logarithms and a density expressed in the input units; it can
    be negative. Coverage_68 uses ±1 sigma (nominal 68.2689492%). Coverage_95 uses
    ±1.95996398454 sigma (nominal 95%). CRPS has the target's physical unit.
    """
    p, y, valid = _scored(mean, target, mask)
    sigma = np.asarray(std, dtype=np.float64)
    if sigma.shape != np.asarray(target).shape:
        raise ValueError("std and target shapes must match exactly")
    sigma = sigma[valid]
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("std must be finite and positive for every scored target")
    result = {"n": len(y), **{key: None for key in GAUSSIAN_KEYS}}
    if not len(y):
        return result
    error = y - p
    with np.errstate(over="ignore", divide="ignore", under="ignore"):
        z = error / sigma
        nll = .5 * z * z + np.log(sigma) + .5 * np.log(2 * np.pi)
        phi = np.exp(-.5 * z * z) / np.sqrt(2 * np.pi)
    if not np.all(np.isfinite(nll)):
        raise ValueError("nonfinite Gaussian NLL; predictive scale is numerically too small")
    crps = error * (2 * ndtr(z) - 1) + sigma * (2 * phi - 1 / np.sqrt(np.pi))
    result.update(nll=float(nll.mean()), crps=float(crps.mean()),
                  coverage_68=float(np.mean(np.abs(error) <= sigma)),
                  coverage_95=float(np.mean(np.abs(error) <= COVERAGE_95_Z * sigma)),
                  mean_std=float(sigma.mean()))
    return result


def evaluate_reconstruction(mean, target, std=None, mask=None,
                            channel_names: Sequence[str] = ("TEMP", "SALT")) -> dict:
    """Evaluate [sample, channel] arrays separately for each named variable."""
    p, y = np.asarray(mean), np.asarray(target)
    if p.ndim != 2 or p.shape != y.shape or p.shape[1] != len(channel_names):
        raise ValueError("mean/target must have matching [sample, channel] shapes")
    if len(set(channel_names)) != len(channel_names):
        raise ValueError("channel names must be unique")
    if std is not None and np.asarray(std).shape != y.shape:
        raise ValueError("std shape must match target")
    if mask is not None and np.asarray(mask).shape != y.shape:
        raise ValueError("mask shape must match target")
    result = {}
    for j, name in enumerate(channel_names):
        selected = None if mask is None else np.asarray(mask)[:, j]
        result[name] = regression_metrics(p[:, j], y[:, j], selected)
        if std is not None:
            uncertainty = gaussian_metrics(p[:, j], np.asarray(std)[:, j], y[:, j], selected)
            result[name].update({k: uncertainty[k] for k in GAUSSIAN_KEYS})
    return result


def inverse_standardize(values, level_index, scale, offset=None, *, uncertainty=False):
    """Undo saved per-depth normalization on [sample, channel] arrays.

    Scale/offset must have shape [depth, channel]. Mean and target are transformed
    as z*scale + offset; std is transformed as std_z*scale with uncertainty=True.
    This function does not restore a spatially varying climatology.
    """
    a, level = np.asarray(values, dtype=np.float64), np.asarray(level_index)
    sd = np.asarray(scale, dtype=np.float64)
    if a.ndim != 2 or level.shape != (len(a),) or level.dtype.kind not in "iu":
        raise ValueError("values must be [sample, channel] and level_index a matching integer vector")
    if sd.ndim != 2 or sd.shape[1] != a.shape[1] or np.any(~np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError("scale must be a finite positive [depth, channel] matrix")
    if np.any(level < 0) or np.any(level >= len(sd)):
        raise ValueError("level_index lies outside saved normalization")
    if uncertainty and offset is not None:
        raise ValueError("uncertainty conversion must not include an offset")
    mu = np.zeros_like(sd) if offset is None else np.asarray(offset, dtype=np.float64)
    if mu.shape != sd.shape or np.any(~np.isfinite(mu)):
        raise ValueError("offset must be finite and match scale")
    return a * sd[level] + (0. if uncertainty else mu[level])


def paired_rmse_bootstrap(mean, reference, target, group, *, draws=4000, seed=20261007):
    """Paired pooled RMSE difference CI by resampling complete groups.

    Inputs are one variable in physical units. Groups typically represent months.
    Negative differences favor mean. This interval does not include training-seed
    uncertainty and assumes the selected grouping is an appropriate sampling unit.
    """
    p, y = np.asarray(mean, float), np.asarray(target, float)
    b, groups = np.asarray(reference, float), np.asarray(group)
    if p.ndim != 1 or p.shape != y.shape or b.shape != y.shape or groups.shape != y.shape:
        raise ValueError("bootstrap inputs must be matching vectors")
    if not isinstance(draws, (int, np.integer)) or draws < 1:
        raise ValueError("draws must be a positive integer")
    _, _, valid = _scored(p, y)
    _scored(b, y)
    if groups.dtype.kind in "iuf" and np.any(~np.isfinite(groups[valid])):
        raise ValueError("group identity must be finite for every scored target")
    names = np.unique(groups[valid])
    if not len(names):
        return {"n": 0, "n_groups": 0, "delta_rmse": None, "ci95_delta_rmse": None}
    stats = np.array([[np.sum((p[ok] - y[ok]) ** 2), np.sum((b[ok] - y[ok]) ** 2), np.sum(ok)]
                      for g in names for ok in [valid & (groups == g)]], dtype=np.float64)
    totals = stats.sum(axis=0)
    rng = np.random.default_rng(seed)
    sampled = stats[rng.integers(len(names), size=(draws, len(names)))].sum(axis=1)
    delta = np.sqrt(sampled[:, 0] / sampled[:, 2]) - np.sqrt(sampled[:, 1] / sampled[:, 2])
    return {"n": int(totals[2]), "n_groups": len(names), "groups": names.tolist(),
            "draws": int(draws), "seed": int(seed),
            "delta_rmse": float(np.sqrt(totals[0] / totals[2]) - np.sqrt(totals[1] / totals[2])),
            "ci95_delta_rmse": np.quantile(delta, [.025, .975]).tolist(),
            "sign": "prediction minus reference; negative means lower RMSE"}
