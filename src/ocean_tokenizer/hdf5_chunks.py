"""Selected levels of a chunked, gzip-compressed HDF5 variable, from byte ranges.

For a file too large to download: the chunk index says where every compressed
chunk of a ``(time, level, y, x)`` variable sits, so the levels wanted at one
time step can be fetched as a few byte ranges and decoded here, without the
HDF5 library touching the remote file again.

``read(offset, size) -> bytes`` is the only access to the file, so the same
code reads a local file in the tests and an HTTP range in production.
"""
from __future__ import annotations

import zlib

import numpy as np


def chunk_index(dataset) -> np.ndarray:
    """``(n, 6)`` int64 rows ``[t, k, y, x, byte offset, stored bytes]`` of an h5py dataset."""
    rows = []
    dataset.id.chunk_iter(lambda info: rows.append((*info.chunk_offset, info.byte_offset, info.size)))
    return np.array(rows, dtype=np.int64).reshape(-1, 6)


def _check_levels(levels, n_levels):
    levels = [int(k) for k in levels]
    if (not levels or levels[0] < 0 or levels[-1] >= n_levels
            or any(b <= a for a, b in zip(levels, levels[1:]))):
        raise ValueError(f"levels must be a non-empty increasing list inside 0..{n_levels - 1}")
    return levels


def spans(rows, chunk_shape, time_index, levels):
    """One ``(offset, size, member rows)`` per depth chunk holding a wanted level.

    The chunks of one time step and depth chunk were written one after another,
    so their byte range is fetched in a single request.
    """
    wanted = sorted({k // chunk_shape[1] * chunk_shape[1] for k in levels})
    out = []
    for k0 in wanted:
        members = rows[(rows[:, 0] == time_index) & (rows[:, 1] == k0)]
        if len(members):
            start = int(members[:, 4].min())
            out.append((start, int((members[:, 4] + members[:, 5]).max()) - start, members))
    return out


def _decode(raw, chunk_shape, dtype, shuffle):
    data = zlib.decompress(raw)
    if shuffle:      # HDF5's shuffle filter stores byte 0 of every value, then byte 1, ...
        data = np.frombuffer(data, dtype=np.uint8).reshape(dtype.itemsize, -1).T.tobytes()
    return np.frombuffer(data, dtype=dtype).reshape(chunk_shape)


def read_levels(read, rows, shape, chunk_shape, time_index, levels, dtype, *, shuffle, fill):
    """``(len(levels), ny, nx)`` of one time step; a chunk absent from the index is ``fill``."""
    levels = _check_levels(levels, shape[1])
    dtype = np.dtype(dtype)
    _, _, ny, nx = shape
    out = np.full((len(levels), ny, nx), fill, dtype=dtype.newbyteorder("="))
    slab = (1, *chunk_shape[1:])
    for offset, size, members in spans(rows, chunk_shape, time_index, levels):
        blob = read(offset, size)
        if len(blob) != size:
            raise ValueError(f"asked for {size} bytes at {offset}, got {len(blob)}")
        k0 = int(members[0, 1])
        pick = [(i, k - k0) for i, k in enumerate(levels) if k0 <= k < k0 + chunk_shape[1]]
        dst, src = [i for i, _ in pick], [j for _, j in pick]
        for _, _, y0, x0, at, n in members:
            chunk = _decode(blob[at - offset:at - offset + n], slab, dtype, shuffle)[0]
            y1, x1 = min(y0 + chunk_shape[2], ny), min(x0 + chunk_shape[3], nx)
            out[dst, y0:y1, x0:x1] = chunk[src, :y1 - y0, :x1 - x0]
    return out
