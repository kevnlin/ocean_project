"""The FuXi-Ocean-style backbone: shapes, lead conditioning, windows that wrap in longitude only."""
import pytest
import torch

from ocean_tokenizer.fuxi_backbone import FuXiEncoder, SwinBlock, polar_mask


def _encoder(seed=0):
    torch.manual_seed(seed)
    return FuXiEncoder(93, 8, (12, 24), width=16, depth=2, heads=2, window=3).double()


def _x():
    return torch.randn(1, 93, 12, 24, dtype=torch.float64, generator=torch.Generator().manual_seed(5))


def test_the_state_has_the_grid_shape_and_is_finite_at_every_lead():
    model, x = _encoder(), _x()
    for lead in (0, 1, 3, 6):
        y = model(x, lead, 13.)
        assert y.shape == (1, 8, 12, 24) and torch.isfinite(y).all()


def test_the_lead_and_season_enter_through_a_time_context_that_starts_at_zero():
    model, x = _encoder(), _x()
    assert torch.equal(model(x, 0, 5.), model(x, 6, 5.))
    torch.nn.init.normal_(model.time[-1].weight, std=.1)
    assert not torch.allclose(model(x, 0, 5.), model(x, 6, 5.))
    assert not torch.allclose(model(x, 0, 5.), model(x, 0, 11.))


def test_a_shifted_window_wraps_in_longitude_and_not_across_the_poles():
    torch.manual_seed(1)
    block = SwinBlock(8, 2, 3, 1, (6, 6)).double()
    z, c = torch.randn(1, 6, 6, 8, dtype=torch.float64), torch.randn(1, 6, 6, 8, dtype=torch.float64)
    base = block(z, c)
    east, north = z.clone(), z.clone()
    east[0, 2, 0] += 1.                       # the first longitude column
    north[0, 0, 2] += 1.                      # the first latitude row
    assert not torch.allclose(block(east, c)[0, 2, 5], base[0, 2, 5])      # reaches the last column
    assert torch.allclose(block(north, c)[0, 5, 2], base[0, 5, 2])         # never the last row
    assert polar_mask(6, 6, 3, 1).flatten(1).any(1).tolist() == [False, False, True, True]


def test_every_parameter_learns():
    model = _encoder()
    model(_x(), 3, 5.).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_a_grid_the_windows_do_not_tile_is_refused():
    with pytest.raises(ValueError, match="multiples of 4"):
        FuXiEncoder(93, 8, (12, 24), width=16, depth=2, heads=2, window=9)
