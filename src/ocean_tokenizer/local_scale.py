"""A scale that depends on depth level, position and calendar month.

``ArgoNorm`` gives each depth level one standard deviation for the whole
ocean. Inside a level the spread of the anomalies, and of an analysis's error,
differs between regions by an order of magnitude and changes through the year.
A scale table records that:

    s[cell, month, level, variable]
        = sqrt(local second moment / the level's second moment)

so ``s = 1`` means "as variable as the level as a whole" and dividing a
per-depth z-score by ``s`` gives a value whose spread is about the same
everywhere. Cells are ``CELL_DEG`` degrees square; a month's entry pools that
calendar month and its two neighbours. Moments are taken about zero: the
quantities scaled here (anomalies from a climatology, analysis errors) have no
mean to remove, and a purely multiplicative scale keeps zero at zero.
"""
from __future__ import annotations

import numpy as np

CELL_DEG = 5.0


def n_cells(deg: float = CELL_DEG) -> int:
    return int(round(180.0 / deg)) * int(round(360.0 / deg))


def cell_index(lat, lon, deg: float = CELL_DEG) -> np.ndarray:
    """Row-major index of each position's cell; longitude wraps, 90N stays in."""
    ny, nx = int(round(180.0 / deg)), int(round(360.0 / deg))
    iy = np.clip(np.floor((np.asarray(lat, float) + 90.0) / deg).astype(int), 0, ny - 1)
    ix = np.floor((np.asarray(lon, float) % 360.0) / deg).astype(int) % nx
    return iy * nx + ix


def scale_table(values, lat, lon, month_of_year, fit, *, by_month: bool = True,
                deg: float = CELL_DEG, pseudo: float = 20.0,
                floor: float = 0.2) -> np.ndarray:
    """``(n_cells, 12, L, C)`` scales from the profiles selected by ``fit``.

    ``values`` is ``(P, L, C)`` with NaN where a profile has no value;
    ``month_of_year`` is 0-11. Only rows where ``fit`` is true are read, so a
    table fitted on the training years cannot see a later year.

    A thinly sampled entry is shrunk towards its parent with ``pseudo``
    pseudo-observations: a cell-month towards its cell, a cell towards the
    level. A cell nobody sampled therefore gets exactly 1. ``floor`` bounds the
    scale from below, so a nearly constant region is not blown up to unit size.
    With ``by_month=False`` the twelve month entries of a cell are equal.
    """
    values = np.asarray(values, dtype=np.float64)
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    month = np.asarray(month_of_year).astype(int)
    fit = np.asarray(fit, dtype=bool)
    p = values.shape[0]
    if values.ndim != 3 or not (lat.shape == lon.shape == month.shape == fit.shape == (p,)):
        raise ValueError("values must be (P, L, C) with one row per profile in lat, lon, "
                         "month_of_year and fit")
    if month.size and (month.min() < 0 or month.max() > 11):
        raise ValueError("month_of_year must be the calendar month, 0-11")
    cells, (_, l, c) = n_cells(deg), values.shape
    key = (cell_index(lat, lon, deg) * 12 + month)[fit]
    v = values[fit]
    ok = np.isfinite(v)
    sq = np.where(ok, v, 0.0) ** 2
    s2 = np.zeros((cells * 12, l, c))
    n = np.zeros((cells * 12, l, c))
    np.add.at(s2, key, sq)
    np.add.at(n, key, ok.astype(np.float64))
    s2, n = s2.reshape(cells, 12, l, c), n.reshape(cells, 12, l, c)

    n_level, s2_level = n.sum((0, 1)), s2.sum((0, 1))
    seen = (n_level > 0) & (s2_level > 0)
    m2_level = np.where(seen, s2_level / np.maximum(n_level, 1.0), 1.0)
    n_cell, s2_cell = n.sum(1), s2.sum(1)
    m2_cell = (s2_cell + pseudo * m2_level) / np.maximum(n_cell + pseudo, 1e-300)
    m2_cell = np.where(n_cell + pseudo > 0, m2_cell, m2_level)
    if by_month:
        n_win = n + np.roll(n, 1, axis=1) + np.roll(n, -1, axis=1)
        s2_win = s2 + np.roll(s2, 1, axis=1) + np.roll(s2, -1, axis=1)
        m2 = (s2_win + pseudo * m2_cell[:, None]) / np.maximum(n_win + pseudo, 1e-300)
        m2 = np.where(n_win + pseudo > 0, m2, m2_cell[:, None])
    else:
        m2 = np.repeat(m2_cell[:, None], 12, axis=1)
    scale = np.sqrt(m2 / m2_level)
    # an unobserved level, or one that is identically zero, has nothing to scale
    scale = np.where(seen, scale, 1.0)
    return np.maximum(scale, floor)


def lookup(table, lat, lon, month_of_year, deg: float = CELL_DEG) -> np.ndarray:
    """``(P, L, C)``: each profile's own cell and calendar month."""
    return table[cell_index(lat, lon, deg), np.asarray(month_of_year).astype(int)]
