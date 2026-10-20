"""Pseudo rows: query-only rows appended to a scenes object, the original left untouched."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ocean_tokenizer import dense_queries as D


def scenes(n=5, levels=3):
    y = np.random.default_rng(0).normal(size=(n, levels, 2)).astype(np.float32)
    y[0, 1, 0] = np.nan
    t = lambda a, dtype=torch.float32: torch.as_tensor(a, dtype=dtype)
    data = SimpleNamespace(device=torch.device("cpu"), n_levels=levels,
                           c=SimpleNamespace(month_index=np.arange(n) // 2, lat=np.linspace(-10, 10, n), lon=np.zeros(n)),
                           lat=t(np.linspace(-10, 10, n)), lon=t(np.zeros(n)), day=t(np.arange(n) * 30.),
                           x=t(np.ones((n, 56))), y=t(y), valid=t(np.isfinite(y), torch.bool), valid_np=np.isfinite(y),
                           fg=t(np.zeros((n, levels, 2))), error=t(np.zeros((n, levels, 2))),
                           error_valid=t(np.zeros((n, levels, 2))), state=t(np.zeros(n)), state_ok=t(np.zeros(n), torch.bool),
                           lagged=t(np.zeros((n, 3, 7))), scale=None)
    data.innovation = torch.where(data.valid, data.y, 0.)
    data.xyz_np = D.lonlat_to_xyz(np.asarray(data.c.lat), np.asarray(data.c.lon))
    data.xyz = torch.as_tensor(data.xyz_np, dtype=torch.float64)
    return data


def test_pseudo_rows_extend_every_per_row_array_and_leave_the_original_alone():
    data = scenes()
    lat, lon = np.array([1., 2.]), np.array([-30., 400.])
    features = np.full((2, 56), 7., np.float32)
    targets = np.full((2, 3, 2), .5, np.float32)
    new, rows = D.with_pseudo_rows(data, lat, lon, month=4, day=123., features=features, targets=targets,
                                   lagged=np.full((2, 3, 7), 2., np.float32))
    assert rows.tolist() == [5, 6] and len(data.lat) == 5
    assert torch.allclose(new.lon[rows], torch.tensor([330., 40.]))
    assert (new.x[rows] == 7).all() and (new.lagged[rows] == 2).all() and (new.day[rows] == 123.).all()
    assert new.c.month_index[rows].tolist() == [4, 4] and len(data.c.month_index) == 5
    assert torch.equal(new.innovation[rows], torch.full((2, 3, 2), .5)) and new.valid_np.shape == (7, 3, 2)
    assert np.allclose(new.xyz_np[rows], D.lonlat_to_xyz(lat, lon))
    assert torch.equal(new.x[:5], data.x)


def test_pseudo_rows_without_targets_are_never_valid():
    new, rows = D.with_pseudo_rows(scenes(), np.array([0.]), np.array([0.]), month=1, day=0.,
                                   features=np.zeros((1, 56), np.float32))
    assert not new.valid[rows].any() and (new.innovation[rows] == 0).all()
