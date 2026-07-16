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


def evaluate_layers(pred: dict, true: dict, masks, depths, layers) -> dict:
    """Per-depth-layer RMSE (unobserved-only, valid-cell-weighted = pooled).

    ``layers``: list of (name, lo_m, hi_m); a depth level d joins the layer with
    lo < d <= hi (the shallowest layer also takes the surface level d == min).
    RMSE for a layer pools the squared errors over every scored cell at every
    depth in the layer, so it equals the true RMSE over that ocean volume
    (deeper levels, with fewer wet cells, contribute proportionally).

    Returns {'by_layer': {var: {layer_name: rmse}}, 'depths', 'layers'}.
    """
    depths = np.asarray(depths)
    masks = np.asarray(masks).astype(bool)                     # (N,H,W)
    dmin = depths.min()
    res = {"by_layer": {}, "depths": depths, "layers": layers}
    for v in pred:
        p = np.asarray(pred[v]); t = np.asarray(true[v])       # (N,D,H,W)
        m = np.broadcast_to(masks[:, None, :, :], p.shape)
        res["by_layer"][v] = {}
        for name, lo, hi in layers:
            sel = (depths > lo) & (depths <= hi)
            if lo <= dmin:                                      # surface level
                sel = sel | np.isclose(depths, dmin)
            di = np.where(sel)[0]
            if di.size == 0:
                res["by_layer"][v][name] = np.nan
                continue
            keep = m[:, di]
            pv = np.where(keep, p[:, di], np.nan)
            tv = np.where(keep, t[:, di], np.nan)
            res["by_layer"][v][name] = rmse_overall(pv, tv)
    return res


def evaluate_masked(pred: dict, true: dict, masks, depths) -> dict:
    """Unobserved-only evaluation.

    ``masks``: (N, H, W) boolean array, True where a cell should be *scored*
    (typically ocean cells that are NOT profile columns).  The mask is
    broadcast across depth so an entire observed column is excluded.

    Excluding observed profile columns removes the leakage of scoring a model on
    cells where it was fed the noise-free truth — the Week-1 metric fix.
    """
    masks = np.asarray(masks).astype(bool)                     # (N,H,W)
    res = {"overall": {}, "by_depth": {}, "depths": np.asarray(depths)}
    for v in pred:
        p = np.asarray(pred[v]); t = np.asarray(true[v])       # (N,D,H,W)
        m = masks[:, None, :, :]                               # (N,1,H,W) -> broadcast
        keep = np.broadcast_to(m, p.shape)
        pv = np.where(keep, p, np.nan)
        tv = np.where(keep, t, np.nan)
        res["overall"][v] = rmse_overall(pv, tv)
        res["by_depth"][v] = rmse_by_depth(pv, tv)
    return res
