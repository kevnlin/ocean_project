"""RMSE metrics, NaN-aware, resolved overall / by-variable / by-depth."""
from __future__ import annotations
import numpy as np


def _rmse(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return np.nan
    d = a[m] - b[m]
    return float(np.sqrt(np.mean(d * d)))


def rmse_overall(pred, true):
    """pred,true: (..., ) arrays; returns scalar NaN-aware RMSE."""
    return _rmse(np.asarray(pred), np.asarray(true))


def rmse_by_depth(pred, true):
    """pred,true shape (N, D, H, W) -> array (D,) of per-depth RMSE."""
    pred = np.asarray(pred); true = np.asarray(true)
    D = pred.shape[1]
    return np.array([_rmse(pred[:, d], true[:, d]) for d in range(D)])


def evaluate(pred: dict, true: dict, depths) -> dict:
    """pred/true: dict var -> (N,D,H,W).  Returns nested metrics dict."""
    res = {"overall": {}, "by_depth": {}, "depths": np.asarray(depths)}
    for v in pred:
        res["overall"][v] = rmse_overall(pred[v], true[v])
        res["by_depth"][v] = rmse_by_depth(pred[v], true[v])
    return res
