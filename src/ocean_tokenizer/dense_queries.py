"""Queries anywhere: pseudo rows appended to a scenes object so a trained model can be read on a grid.

A scenes object (``synthetic_latent_experiment`` and its subclasses) indexes every per-row quantity by
the row of a profile. ``with_pseudo_rows`` returns a shallow copy with ``n`` extra rows at the given
positions and month, with their query features (and lagged surface features for forecast leads); the
copy shares everything else with the original, which is left untouched. Pseudo rows are queries only:
they are never in a source pool, and without targets they are never valid.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from .oi import _lonlat_to_xyz


def lonlat_to_xyz(lat, lon):
    """Unit vectors of positions in degrees, the scenes' own convention (``oi._lonlat_to_xyz``)."""
    return _lonlat_to_xyz(np.asarray(lat, np.float64), np.asarray(lon, np.float64))


def with_pseudo_rows(data, lat, lon, *, month, day, features, targets=None, lagged=None, scale=None):
    """A copy of ``data`` with query-only rows at (``lat``, ``lon``) in month index ``month``.

    ``features`` are the rows' 56 query features, ``targets`` optional [n, levels, 2] (NaN where
    missing), ``lagged`` optional [n, leads, 7] surface columns of the input months, ``scale``
    optional [n, levels, 2] local scales. Returns the copy and the indices of the new rows."""
    n0, n = len(data.lat), len(lat)
    dev = data.device
    tensor = lambda a, dtype=torch.float32: torch.as_tensor(np.asarray(a), dtype=dtype, device=dev)
    levels = data.n_levels
    if targets is None:
        targets = np.full((n, levels, 2), np.nan, np.float32)
    new = copy.copy(data)
    new.c = copy.copy(data.c)
    new.c.month_index = np.concatenate([np.asarray(data.c.month_index), np.full(n, month, dtype=np.asarray(data.c.month_index).dtype)])
    new.lat = torch.cat([data.lat, tensor(lat)])
    new.lon = torch.cat([data.lon, tensor(np.asarray(lon, np.float64) % 360.)])
    xyz = lonlat_to_xyz(lat, lon)
    new.xyz_np = np.concatenate([data.xyz_np, xyz])
    new.xyz = torch.cat([data.xyz, tensor(xyz, torch.float64)])
    new.day = torch.cat([data.day, torch.full((n,), float(day), dtype=data.day.dtype, device=dev)])
    new.x = torch.cat([data.x, tensor(features)])
    y = tensor(targets)
    new.y = torch.cat([data.y, y])
    new.valid = torch.cat([data.valid, torch.isfinite(y)])
    new.valid_np = np.concatenate([data.valid_np, np.isfinite(np.asarray(targets))])
    new.innovation = torch.cat([data.innovation, torch.where(torch.isfinite(y), y, torch.zeros_like(y))])
    zeros = torch.zeros((n, levels, 2), dtype=data.fg.dtype, device=dev)
    new.fg = torch.cat([data.fg, zeros])
    new.error = torch.cat([data.error, zeros])
    new.error_valid = torch.cat([data.error_valid, zeros])
    if getattr(data, "state", None) is not None:
        new.state = torch.cat([data.state, torch.zeros(n, dtype=data.state.dtype, device=dev)])
        new.state_ok = torch.cat([data.state_ok, torch.zeros(n, dtype=data.state_ok.dtype, device=dev)])
    if getattr(data, "lagged", None) is not None:
        extra = np.zeros((n,) + tuple(data.lagged.shape[1:]), np.float32) if lagged is None else lagged
        new.lagged = torch.cat([data.lagged, tensor(extra)])
    if getattr(data, "scale", None) is not None:
        if scale is None:
            raise ValueError("the scenes use local scales; pass the pseudo rows' scales")
        new.scale = torch.cat([data.scale, tensor(scale, data.scale.dtype)])
    return new, np.arange(n0, n0 + n)
