"""`hold_device` must be a storage choice, never a behaviour change.

`train_predict_unet` produced every committed U-Net result in this repo. Adding
an offload path to fit a shared GPU is only safe if the default path is
bit-identical to what it was, and if the offload path computes the same thing.
These tests pin both on CPU, where the whole training loop is deterministic.
"""
import numpy as np
import pytest
import torch

from ocean_tokenizer import baselines as B, config as C


class _Grid:
    def __init__(self, D=3, H=8, W=12):
        self.lat = np.linspace(-30, 30, H)
        self.lon = np.linspace(0, 330, W)
        self.depth = np.linspace(5, 500, D)
        self.ndepth, self.nlat, self.nlon = D, H, W
        self.ocean = np.ones((H, W), bool)
        self.ocean[0, 0] = False


class _Norm:
    def z3d(self, v, arr, month=None):
        return np.asarray(arr, dtype="float32")

    def unz3d(self, v, arr, month=None):
        return np.asarray(arr, dtype="float32")

    def zsurf(self, v, arr, month=None):
        return np.asarray(arr, dtype="float32")


def _samples(grid, n, seed=0):
    rng = np.random.default_rng(seed)
    D, H, W = grid.ndepth, grid.nlat, grid.nlon
    out = []
    for i in range(n):
        obs = np.full((D, H, W), np.nan, "float32")
        obs[:, 1 + i % 3, 2] = rng.normal(size=D).astype("float32")
        unobs = grid.ocean.copy()
        unobs[1 + i % 3, 2] = False
        out.append({
            "month": 1 + i % 12,
            "gt": {v: rng.normal(size=(D, H, W)).astype("float32") for v in B.VARS},
            "woa": {v: rng.normal(size=(D, H, W)).astype("float32") for v in B.VARS},
            "obs": {v: obs.copy() for v in B.VARS},
            "surf": {sv: rng.normal(size=(H, W)).astype("float32")
                     for sv in ("SST", "SSS")},
            "unobs_mask": unobs,
        })
    return out


def _same(a, b):
    """Bit-identical comparison that treats NaN as equal.

    Land cells are NaN by construction, and plain ``array_equal`` reports two
    identical arrays as different when either holds a NaN (NaN != NaN) — which
    would turn every comparison below into a test that can never pass.
    """
    return np.array_equal(a, b, equal_nan=True)


def _run(hold_device, epochs=2):
    torch.set_num_threads(1)          # tiny tensors: 24 threads is pure overhead
    grid, norm = _Grid(), _Norm()
    tr, te = _samples(grid, 4, seed=1), _samples(grid, 2, seed=2)
    old_ep, old_b = C.UNET_EPOCHS, C.UNET_BATCH
    C.UNET_EPOCHS, C.UNET_BATCH = epochs, 3
    try:
        torch.manual_seed(1234)
        preds = B.train_predict_unet(tr, te, grid, norm, ("profiles", "woa", "surf"),
                                     "cpu", unobs_loss=True,
                                     hold_device=hold_device)
    finally:
        C.UNET_EPOCHS, C.UNET_BATCH = old_ep, old_b
    return np.stack([p[v] for p in preds for v in B.VARS])


def test_offload_path_matches_the_default_path():
    """hold_device='cpu' with device='cpu' must reproduce hold_device=None
    exactly -- same seed, same batches, same arithmetic."""
    assert _same(_run(None), _run("cpu"))


def test_default_is_reproducible():
    """Guards the comparison above: if the default path were nondeterministic,
    the equality test would prove nothing."""
    assert _same(_run(None), _run(None))


def test_batch_order_is_unchanged_by_offload():
    """The RNG stream must not move.  randperm is still drawn on `device`, so
    an offload run consumes torch randomness identically -- checked by the
    generator state after training rather than only by the outputs."""
    def state_after(hold):
        torch.manual_seed(7)
        _run_grid, _run_norm = _Grid(), _Norm()
        tr, te = _samples(_run_grid, 3, seed=5), _samples(_run_grid, 1, seed=6)
        old_ep, old_b = C.UNET_EPOCHS, C.UNET_BATCH
        C.UNET_EPOCHS, C.UNET_BATCH = 2, 2
        try:
            B.train_predict_unet(tr, te, _run_grid, _run_norm,
                                 ("profiles",), "cpu", unobs_loss=True,
                                 hold_device=hold)
        finally:
            C.UNET_EPOCHS, C.UNET_BATCH = old_ep, old_b
        return torch.random.get_rng_state()

    assert torch.equal(state_after(None), state_after("cpu"))


@pytest.mark.parametrize("hold", [None, "cpu"])
def test_predictions_are_finite_on_ocean_and_nan_on_land(hold):
    grid = _Grid()
    out = _run(hold)
    assert np.isfinite(out[..., 1:, :]).all()
    assert np.isnan(out[..., 0, 0]).all()      # the masked-out land cell
