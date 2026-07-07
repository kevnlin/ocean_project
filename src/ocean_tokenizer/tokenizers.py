"""Four ocean-field tokenizers with exact (lossless) round-trip.

All tokenizers are reversible re-arrangements / gathers of the input field, so
``decode(encode(field)) == field`` exactly (bit-for-bit, NaNs preserved) when no
masking/subsampling is applied.  Each exposes:

    ts = tok.encode(field)      # -> TokenSet (tokens + reconstruction metadata)
    field2 = tok.decode(ts)     # -> ndarray identical to `field`

Field conventions
-----------------
* 2D grid patch     : field shape (C, H, W)
* 3D volume patch   : field shape (C, D, H, W)
* vertical profile  : field shape (C, D, H, W)   -> one token per (lat,lon) column
* point query       : field shape (C, D, H, W) or (C, H, W)
"""
from __future__ import annotations
from dataclasses import dataclass, field as dc_field
from typing import Any
import numpy as np


@dataclass
class TokenSet:
    tokens: np.ndarray              # (n_tokens, token_dim)
    meta: dict[str, Any] = dc_field(default_factory=dict)

    @property
    def shape(self):
        return self.tokens.shape


# --------------------------------------------------------------------------
# (a) 2D grid patch tokenizer
# --------------------------------------------------------------------------
class GridPatchTokenizer2D:
    """Split (C, H, W) into non-overlapping (ph, pw) patches -> tokens.

    token_dim = C * ph * pw.  Padding (with NaN) makes H,W divisible; the
    original H,W are stored so decode crops back exactly.
    """

    def __init__(self, patch=(15, 16), pad_value=np.nan):
        self.ph, self.pw = patch
        self.pad_value = pad_value

    def encode(self, field: np.ndarray) -> TokenSet:
        assert field.ndim == 3, "expect (C, H, W)"
        C, H, W = field.shape
        Hp = int(np.ceil(H / self.ph) * self.ph)
        Wp = int(np.ceil(W / self.pw) * self.pw)
        x = np.full((C, Hp, Wp), self.pad_value, dtype=field.dtype)
        x[:, :H, :W] = field
        nh, nw = Hp // self.ph, Wp // self.pw
        # (C, nh, ph, nw, pw) -> (nh, nw, C, ph, pw)
        x = x.reshape(C, nh, self.ph, nw, self.pw)
        x = x.transpose(1, 3, 0, 2, 4).reshape(nh * nw, C * self.ph * self.pw)
        return TokenSet(x, dict(C=C, H=H, W=W, Hp=Hp, Wp=Wp, nh=nh, nw=nw,
                                ph=self.ph, pw=self.pw))

    def decode(self, ts: TokenSet) -> np.ndarray:
        m = ts.meta
        x = ts.tokens.reshape(m["nh"], m["nw"], m["C"], m["ph"], m["pw"])
        x = x.transpose(2, 0, 3, 1, 4).reshape(m["C"], m["Hp"], m["Wp"])
        return x[:, :m["H"], :m["W"]]


# --------------------------------------------------------------------------
# (b) 3D volume patch tokenizer
# --------------------------------------------------------------------------
class VolumePatchTokenizer3D:
    """Split (C, D, H, W) into (pd, ph, pw) volume patches -> tokens.

    token_dim = C * pd * ph * pw.
    """

    def __init__(self, patch=(5, 30, 36), pad_value=np.nan):
        self.pd, self.ph, self.pw = patch
        self.pad_value = pad_value

    def encode(self, field: np.ndarray) -> TokenSet:
        assert field.ndim == 4, "expect (C, D, H, W)"
        Cc, D, H, W = field.shape
        Dp = int(np.ceil(D / self.pd) * self.pd)
        Hp = int(np.ceil(H / self.ph) * self.ph)
        Wp = int(np.ceil(W / self.pw) * self.pw)
        x = np.full((Cc, Dp, Hp, Wp), self.pad_value, dtype=field.dtype)
        x[:, :D, :H, :W] = field
        nd, nh, nw = Dp // self.pd, Hp // self.ph, Wp // self.pw
        x = x.reshape(Cc, nd, self.pd, nh, self.ph, nw, self.pw)
        # -> (nd, nh, nw, C, pd, ph, pw)
        x = x.transpose(1, 3, 5, 0, 2, 4, 6).reshape(
            nd * nh * nw, Cc * self.pd * self.ph * self.pw)
        return TokenSet(x, dict(C=Cc, D=D, H=H, W=W, Dp=Dp, Hp=Hp, Wp=Wp,
                                nd=nd, nh=nh, nw=nw,
                                pd=self.pd, ph=self.ph, pw=self.pw))

    def decode(self, ts: TokenSet) -> np.ndarray:
        m = ts.meta
        x = ts.tokens.reshape(m["nd"], m["nh"], m["nw"], m["C"],
                              m["pd"], m["ph"], m["pw"])
        x = x.transpose(3, 0, 4, 1, 5, 2, 6).reshape(
            m["C"], m["Dp"], m["Hp"], m["Wp"])
        return x[:, :m["D"], :m["H"], :m["W"]]


# --------------------------------------------------------------------------
# (c) vertical profile tokenizer
# --------------------------------------------------------------------------
class VerticalProfileTokenizer:
    """One token per (lat, lon) column: the full vertical profile stacked over
    variables.  field (C, D, H, W) -> tokens (H*W, C*D)."""

    def encode(self, field: np.ndarray) -> TokenSet:
        assert field.ndim == 4, "expect (C, D, H, W)"
        Cc, D, H, W = field.shape
        # (C, D, H, W) -> (H, W, C, D) -> (H*W, C*D)
        x = field.transpose(2, 3, 0, 1).reshape(H * W, Cc * D)
        return TokenSet(x, dict(C=Cc, D=D, H=H, W=W))

    def decode(self, ts: TokenSet) -> np.ndarray:
        m = ts.meta
        x = ts.tokens.reshape(m["H"], m["W"], m["C"], m["D"])
        return x.transpose(2, 3, 0, 1)


# --------------------------------------------------------------------------
# (d) point query tokenizer
# --------------------------------------------------------------------------
class PointQueryTokenizer:
    """Represent a field as a set of (coordinate, value) tokens.

    Each token = [t_norm?, z_norm, lat_norm, lon_norm, v_0..v_{C-1}].
    Decoding scatters values back onto the grid via stored integer indices, so
    a *full* set of points reconstructs exactly.  Subsampling -> sparse/masked.
    """

    def __init__(self, coords=None):
        # coords: optional (lat_vals, lon_vals, depth_vals) for real normalisation
        self.coords = coords

    def _norm_axes(self, D, H, W):
        if self.coords is not None:
            lat, lon, dep = self.coords
            zc = (np.asarray(dep[:D]) - np.min(dep)) / (np.ptp(dep) + 1e-9)
            yc = (np.asarray(lat) - np.min(lat)) / (np.ptp(lat) + 1e-9)
            xc = (np.asarray(lon) - np.min(lon)) / (np.ptp(lon) + 1e-9)
        else:
            zc = np.linspace(0, 1, D)
            yc = np.linspace(0, 1, H)
            xc = np.linspace(0, 1, W)
        return zc, yc, xc

    def encode(self, field: np.ndarray, sample_idx: np.ndarray | None = None) -> TokenSet:
        if field.ndim == 3:                       # (C, H, W) -> add singleton depth
            field = field[:, None]
        assert field.ndim == 4, "expect (C, D, H, W) or (C, H, W)"
        Cc, D, H, W = field.shape
        zc, yc, xc = self._norm_axes(D, H, W)
        zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
        zz, yy, xx = zz.ravel(), yy.ravel(), xx.ravel()
        if sample_idx is not None:
            zz, yy, xx = zz[sample_idx], yy[sample_idx], xx[sample_idx]
        coord = np.stack([zc[zz], yc[yy], xc[xx]], axis=1)        # (N,3)
        vals = field[:, zz, yy, xx].T                            # (N,C)
        tokens = np.concatenate([coord, vals], axis=1).astype(field.dtype)
        return TokenSet(tokens, dict(C=Cc, D=D, H=H, W=W,
                                     idx=np.stack([zz, yy, xx], axis=1),
                                     full=sample_idx is None))

    def decode(self, ts: TokenSet, fill=np.nan) -> np.ndarray:
        m = ts.meta
        Cc, D, H, W = m["C"], m["D"], m["H"], m["W"]
        out = np.full((Cc, D, H, W), fill, dtype=ts.tokens.dtype)
        idx = m["idx"]
        vals = ts.tokens[:, 3:]                                  # drop 3 coord cols
        out[:, idx[:, 0], idx[:, 1], idx[:, 2]] = vals.T
        if D == 1:
            out = out[:, 0]
        return out


# --------------------------------------------------------------------------
# Registry / round-trip helper
# --------------------------------------------------------------------------
def roundtrip_error(field: np.ndarray, recon: np.ndarray) -> dict:
    """Max abs error and RMSE ignoring NaNs (NaN positions must match)."""
    a, b = field, recon
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    nan_match = bool(np.array_equal(nan_a, nan_b))
    m = ~nan_a
    if m.any():
        diff = np.abs(a[m] - b[m])
        max_abs = float(diff.max())
        rmse = float(np.sqrt(np.mean(diff ** 2)))
    else:
        max_abs = rmse = 0.0
    return dict(max_abs=max_abs, rmse=rmse, nan_match=nan_match,
                exact=bool(max_abs == 0.0 and nan_match))
