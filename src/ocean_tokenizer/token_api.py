"""Shared-latent model interface (Week 2): token schema + modality encoders.

This module pins down the *data contract* of the shared-latent method before
the fusion core exists:

* ``TokenBatch``       — the unified OceanObservationToken schema: every
                         observation, whatever its modality, becomes a set of
                         (embedding, physical coordinate, modality id, mask)
                         tokens.
* ``coord_features``   — the single coordinate featurisation shared by every
                         encoder AND the query decoder, so a location means the
                         same thing on both sides (the coordinate-consistency
                         contract).
* ``GridPatchEncoder`` — dense (surface or volumetric) gridded fields
                         -> one token per spatial patch (per depth level).
* ``ProfileEncoder``   — sparse vertical profiles -> several tokens per
                         profile (contiguous depth segments), preserving
                         vertical structure; native variable profile count.
* ``PointEncoder``     — scattered scalar observations -> one token each.
* ``SharedLatentModel``— the abstract encode -> fuse -> decode contract the
                         Week 3-4 Perceiver-style model will implement.
* ``SharedLatentStub`` — a deliberately trivial reference implementation
                         (masked mean-pool fusion + MLP query head).  It exists
                         so the interface, masks, and training plumbing can be
                         unit-tested end-to-end NOW; it is not the method.

Conventions
-----------
* Values entering an encoder are ALREADY normalised (anomaly-space z-scores via
  ``anomaly.AnomNorm``); encoders never see raw degC/PSU.  NaN = missing: it is
  converted to (value 0, finite-flag 0) and never contributes to a valid mask.
* Physical coordinates ride along as (lat, lon, depth_m, month) and are only
  consumed through ``coord_features``.
* Variable observation counts / missing modalities are handled structurally:
  every per-token operation is masked, and a missing modality is simply an
  absent key in the observation dict.

Known open problem (deliberately deferred to Weeks 3-4, with the fusion core):
the dense-grid-vs-sparse-profile token imbalance — a WOA volume yields ~10k
patch tokens while 1500 profiles yield ~6k; the fusion stage must not let the
dense prior drown out the observations.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# Modality registry
# --------------------------------------------------------------------------
MODALITIES = {"surf_grid": 0, "woa_grid": 1, "profile": 2, "point": 3}

# --------------------------------------------------------------------------
# Coordinate featurisation — shared by every encoder and the query decoder
# --------------------------------------------------------------------------
_DEPTH_SCALE = 1000.0             # ~ the deepest target level (985 m)
_N_FREQ_SPHERE = 6                # harmonics 2^0..2^5 on unit-sphere xyz
_N_FREQ_DEPTH = 5                 # harmonics 2^0..2^4 on linear + log depth
N_COORD_FEATS = 7 + 3 + 6 * _N_FREQ_SPHERE + 4 * _N_FREQ_DEPTH + 2   # = 68


def _fourier(x: torch.Tensor, n_freq: int) -> list[torch.Tensor]:
    """Multi-scale harmonics sin/cos(2^k pi x), k = 0..n_freq-1."""
    out = []
    for k in range(n_freq):
        a = (2.0 ** k) * math.pi * x
        out += [torch.sin(a), torch.cos(a)]
    return out


def coord_features(coord: torch.Tensor) -> torch.Tensor:
    """(..., 4) physical (lat_deg, lon_deg, depth_m, month 1-12) -> (..., 68).

    Deterministic and parameter-free, shared by every modality encoder AND the
    query decoder, so a location means the same thing on both sides (the
    coordinate-consistency contract).

    Week-4 revision (fourier_v2): the original 7 smooth features are kept and
    extended with multi-frequency Fourier features — unit-sphere xyz harmonics
    (finest wavelength ~3.6 deg, matched to the typical spacing of 1500
    profiles/month) and linear+log depth harmonics (the log scale resolves
    adjacent shallow levels).  Rationale: with the original first-harmonic
    features only, attention cannot form spatially selective patterns; the
    full-scale Week-4 runs collapsed to the zero-anomaly solution (train loss
    pinned at 1.0 even when queries were sampled AT observed profile columns —
    the copy diagnostic of experiments/18_full_train.py --probe-observed).
    """
    lat, lon, depth, month = coord.unbind(-1)
    lat_r = torch.deg2rad(lat)
    lon_r = torch.deg2rad(lon)
    dn = depth / _DEPTH_SCALE
    ldn = torch.log1p(depth.clamp(min=0.0)) / math.log1p(_DEPTH_SCALE)
    coslat = torch.cos(lat_r)
    x, y, z = coslat * torch.cos(lon_r), coslat * torch.sin(lon_r), torch.sin(lat_r)
    feats = [
        lat / 90.0,
        torch.sin(lon_r), torch.cos(lon_r),
        dn, ldn,
        torch.sin(2 * math.pi * month / 12.0),
        torch.cos(2 * math.pi * month / 12.0),
        x, y, z,
    ]
    for c in (x, y, z):
        feats += _fourier(c, _N_FREQ_SPHERE)
    feats += _fourier(dn, _N_FREQ_DEPTH)
    feats += _fourier(ldn, _N_FREQ_DEPTH)
    feats += [torch.sin(4 * math.pi * month / 12.0),
              torch.cos(4 * math.pi * month / 12.0)]
    return torch.stack(feats, dim=-1)


# --------------------------------------------------------------------------
# TokenBatch — the unified OceanObservationToken schema
# --------------------------------------------------------------------------
VAR_IDS = {"MULTI": -1, "TEMP": 0, "SALT": 1, "SST": 2, "SSS": 3}


@dataclass
class TokenBatch:
    """A batch of observation tokens from one or more modalities.

    Core (always present):
    emb      : (B, N, d)  encoded token embeddings (masked-out tokens are 0)
    coord    : (B, N, 4)  physical (lat, lon, depth_m, month); 0 where masked
    modality : (B, N)     int64 id into MODALITIES
    mask     : (B, N)     bool, True = real token (False = padding / all-NaN)

    Support & provenance (Task 4; None until an encoder supplies them —
    definitions in docs/token_measure_definition.md).  These exist so a test
    or the fusion stage can distinguish *new physical evidence* from *a
    different representation of the same evidence*:
    support_mass : (B, N) float  nonnegative physical support (see docs; MBCA
                   treats a missing mass as uniform-within-modality)
    parent_id    : (B, N) int64  id of the source observation within its
                   modality (profile index, whole-field 0, point index);
                   tokens sharing (modality, parent_id) represent one obs
    family_id    : (B, N) int64  tokenization-scheme id (patch/band layout),
                   so retokenizations of the same evidence are identifiable
    variable_id  : (B, N) int64  VAR_IDS; -1 = token carries multiple vars
    depth_lower  : (B, N) float  represented depth interval (m)
    depth_upper  : (B, N) float
    reliability  : (B, N) float  optional per-token quality in [0, 1]
    """
    emb: torch.Tensor
    coord: torch.Tensor
    modality: torch.Tensor
    mask: torch.Tensor
    support_mass: torch.Tensor | None = None
    parent_id: torch.Tensor | None = None
    family_id: torch.Tensor | None = None
    variable_id: torch.Tensor | None = None
    depth_lower: torch.Tensor | None = None
    depth_upper: torch.Tensor | None = None
    reliability: torch.Tensor | None = None

    _OPT_INT = ("parent_id", "family_id", "variable_id")
    _OPT_FLOAT = ("support_mass", "depth_lower", "depth_upper", "reliability")

    @property
    def n_valid(self) -> torch.Tensor:            # (B,) valid tokens per item
        return self.mask.sum(dim=1)

    def _opt(self, name):
        return getattr(self, name)

    def to(self, device) -> "TokenBatch":
        kw = {n: (None if self._opt(n) is None else self._opt(n).to(device))
              for n in self._OPT_INT + self._OPT_FLOAT}
        return TokenBatch(self.emb.to(device), self.coord.to(device),
                          self.modality.to(device), self.mask.to(device), **kw)

    @staticmethod
    def empty(batch: int, d_model: int, device=None) -> "TokenBatch":
        return TokenBatch(
            emb=torch.zeros(batch, 0, d_model, device=device),
            coord=torch.zeros(batch, 0, 4, device=device),
            modality=torch.zeros(batch, 0, dtype=torch.long, device=device),
            mask=torch.zeros(batch, 0, dtype=torch.bool, device=device),
            support_mass=torch.zeros(batch, 0, device=device))

    @staticmethod
    def cat(parts: list["TokenBatch"]) -> "TokenBatch":
        assert len(parts) > 0, "TokenBatch.cat needs at least one part"
        B = parts[0].emb.shape[0]
        assert all(p.emb.shape[0] == B for p in parts), "batch sizes differ"

        def cat_opt(name, int_default=None):
            vals = [p._opt(name) for p in parts]
            if all(v is None for v in vals):
                return None
            filled = []
            for p, v in zip(parts, vals):
                if v is not None:
                    filled.append(v)
                elif name == "support_mass":
                    # missing mass -> uniform (1 per valid token)
                    filled.append(p.mask.to(p.emb.dtype))
                elif int_default is not None:
                    filled.append(torch.full(p.mask.shape, int_default,
                                             dtype=torch.long,
                                             device=p.mask.device))
                else:
                    filled.append(torch.zeros(p.mask.shape, dtype=p.emb.dtype,
                                              device=p.mask.device))
            return torch.cat(filled, dim=1)

        kw = {n: cat_opt(n, int_default=-1) for n in TokenBatch._OPT_INT}
        kw.update({n: cat_opt(n) for n in TokenBatch._OPT_FLOAT})
        return TokenBatch(
            emb=torch.cat([p.emb for p in parts], dim=1),
            coord=torch.cat([p.coord for p in parts], dim=1),
            modality=torch.cat([p.modality for p in parts], dim=1),
            mask=torch.cat([p.mask for p in parts], dim=1), **kw)


def _finish_tokens(content_emb, coord, coord_proj, mask, modality_id,
                   support_mass=None, **meta):
    """Assemble a TokenBatch: content + coord embedding, zeroed where masked.

    ``meta`` may carry any of the Task-4 provenance fields (parent_id,
    family_id, variable_id, depth_lower, depth_upper, reliability).
    """
    coord = torch.nan_to_num(coord, nan=0.0)
    emb = content_emb + coord_proj(coord_features(coord))
    emb = emb * mask.unsqueeze(-1)
    coord = coord * mask.unsqueeze(-1)
    modality = torch.full(mask.shape, modality_id, dtype=torch.long,
                          device=mask.device)
    if support_mass is not None:
        support_mass = support_mass * mask.to(support_mass.dtype)
    return TokenBatch(emb=emb, coord=coord, modality=modality, mask=mask,
                      support_mass=support_mass, **meta)


# --------------------------------------------------------------------------
# (a) Grid-patch encoder — dense gridded fields (surface or volume)
# --------------------------------------------------------------------------
def _level_edges(depth: torch.Tensor):
    """(D,) level centres -> (lower, upper) represented interval edges (D,)."""
    d = depth
    if d.numel() == 1:
        return (d - 25.0).clamp(min=0.0), d + 25.0
    gap = (d[1:] - d[:-1]).clamp(min=0.0)
    lower = torch.cat([(d[:1] - gap[:1] / 2).clamp(min=0.0), d[:-1] + gap / 2])
    upper = torch.cat([d[:-1] + gap / 2, d[-1:] + gap[-1:] / 2])
    return lower, upper


class GridPatchEncoder(nn.Module):
    """Dense gridded field -> one token per (ph, pw) spatial patch.

    forward(field, lat, lon, month, depth=None)
      * field (B, C, H, W)     surface field; token depth = ``depth`` (default 0)
      * field (B, C, D, H, W)  volume; ``depth`` is the (D,) level grid and the
                               encoding is one token per (level, patch)
    lat (H,), lon (W,) in degrees; month (B,) 1-12 (broadcast to every token).
    NaN cells become (value 0, finite-flag 0); an all-NaN patch is masked out.

    Task-4 support mass (docs/token_measure_definition.md):
      surface patch  mu_i = sum over valid cells of the spherical area weight
                     cos(lat_cell)  (proportional to true cell area on an
                     equiangular grid);
      volume patch   mu_i = (same area sum) x represented layer thickness,
                     where a level's thickness is its inter-level interval.
    Provenance: parent_id = 0 (the whole field is one observation),
    family_id = 10000*ph + pw (the tokenization scheme), variable_id = MULTI,
    depth_lower/upper = the represented depth interval.
    """

    def __init__(self, c_in: int, d_model: int = 128, patch=(10, 12),
                 modality: str = "surf_grid", family_id: int | None = None):
        super().__init__()
        self.c_in = c_in
        self.ph, self.pw = patch
        self.modality_id = MODALITIES[modality]
        self.family_id = (10000 * self.ph + self.pw
                          if family_id is None else family_id)
        self.val_proj = nn.Linear(2 * c_in * self.ph * self.pw, d_model)
        self.coord_proj = nn.Linear(N_COORD_FEATS, d_model)

    def _encode_2d(self, field, lat, lon, month, depth_val):
        """field (B, C, H, W), depth_val (B,) -> tokens (B, nh*nw, ...)."""
        B, C, H, W = field.shape
        ph, pw = self.ph, self.pw
        Hp = math.ceil(H / ph) * ph
        Wp = math.ceil(W / pw) * pw
        if (Hp, Wp) != (H, W):
            padded = field.new_full((B, C, Hp, Wp), float("nan"))
            padded[:, :, :H, :W] = field
            field = padded
        nh, nw = Hp // ph, Wp // pw
        finite = torch.isfinite(field)
        vals = torch.nan_to_num(field, nan=0.0)
        # (B, C, nh, ph, nw, pw) -> (B, nh*nw, C*ph*pw)
        def patchify(x):
            xB, xC = x.shape[:2]
            x = x.reshape(xB, xC, nh, ph, nw, pw).permute(0, 2, 4, 1, 3, 5)
            return x.reshape(xB, nh * nw, xC * ph * pw)
        v = patchify(vals)
        m = patchify(finite.float())
        content = self.val_proj(torch.cat([v, m], dim=-1))          # (B, N, d)
        mask = m.bool().any(dim=-1)                                  # (B, N)
        N = nh * nw

        # ---- support mass: spherical area weight over valid cells ----
        # cell area ~ cos(lat) * dlat * dlon, spacings inferred from the
        # coordinate vectors so a genuinely refined grid (2x cells at half
        # spacing) conserves regional mass exactly.
        latt = torch.as_tensor(lat, device=field.device, dtype=field.dtype)
        lont = torch.as_tensor(lon, device=field.device, dtype=field.dtype)
        dlat = ((latt.max() - latt.min()) / max(H - 1, 1)).clamp(min=1e-6)
        dlon = ((lont.max() - lont.min()) / max(W - 1, 1)).clamp(min=1e-6)
        aw = torch.cos(torch.deg2rad(latt)) * dlat * dlon
        awmap = aw[:, None].expand(H, W)
        if (Hp, Wp) != (H, W):
            pad = awmap.new_zeros(Hp, Wp)
            pad[:H, :W] = awmap
            awmap = pad
        aw_p = patchify(awmap[None, None])[0]                        # (N, ph*pw)
        cell_ok = m.reshape(B, N, C, ph * pw).bool().any(dim=2)      # (B,N,cells)
        mass = (aw_p[None] * cell_ok.to(field.dtype)).sum(-1)        # (B, N)

        # patch-centre coordinates (edge-clamped index midpoint)
        dev = field.device
        ci = torch.clamp(torch.arange(nh, device=dev) * ph + ph // 2, max=H - 1)
        cj = torch.clamp(torch.arange(nw, device=dev) * pw + pw // 2, max=W - 1)
        clat = lat[ci][:, None].expand(nh, nw).reshape(-1)           # (N,)
        clon = lon[cj][None, :].expand(nh, nw).reshape(-1)
        coord = torch.stack([
            clat[None].expand(B, N),
            clon[None].expand(B, N),
            depth_val[:, None].expand(B, N),
            month[:, None].expand(B, N).to(field.dtype),
        ], dim=-1)                                                   # (B, N, 4)
        return content, coord, mask, mass

    def _meta(self, B, N, dlo, dhi, device, dtype):
        return dict(
            parent_id=torch.zeros(B, N, dtype=torch.long, device=device),
            family_id=torch.full((B, N), self.family_id, dtype=torch.long,
                                 device=device),
            variable_id=torch.full((B, N), VAR_IDS["MULTI"], dtype=torch.long,
                                   device=device),
            depth_lower=dlo.to(dtype), depth_upper=dhi.to(dtype))

    def forward(self, field, lat, lon, month, depth=None) -> TokenBatch:
        month = torch.as_tensor(month, device=field.device)
        if month.ndim == 0:
            month = month[None].expand(field.shape[0])
        if field.ndim == 4:                                          # surface
            B = field.shape[0]
            d = (torch.zeros(B, device=field.device) if depth is None
                 else torch.as_tensor(float(depth), device=field.device).expand(B))
            content, coord, mask, mass = self._encode_2d(field, lat, lon,
                                                         month, d)
            N = content.shape[1]
            dlo = d[:, None].expand(B, N)
            dhi = dlo
        elif field.ndim == 5:                                        # volume
            B, C, D, H, W = field.shape
            assert depth is not None and len(depth) == D, \
                "volume input needs the (D,) depth grid"
            flat = field.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            dgrid = torch.as_tensor(depth, device=field.device,
                                    dtype=field.dtype)
            dval = dgrid.repeat(B)
            mo = month.repeat_interleave(D)
            content, coord, mask, mass = self._encode_2d(flat, lat, lon,
                                                         mo, dval)
            n = content.shape[1]
            lo, hi = _level_edges(dgrid)                             # (D,)
            thick = (hi - lo).clamp(min=1e-6)
            # volume mass = area x represented layer thickness
            mass = mass * thick.repeat(B)[:, None]
            content = content.reshape(B, D * n, -1)
            coord = coord.reshape(B, D * n, 4)
            mask = mask.reshape(B, D * n)
            mass = mass.reshape(B, D * n)
            N = D * n
            dlo = lo.repeat_interleave(n)[None].expand(B, N)
            dhi = hi.repeat_interleave(n)[None].expand(B, N)
        else:
            raise ValueError(f"expected (B,C,H,W) or (B,C,D,H,W), got {field.shape}")
        return _finish_tokens(content, coord, self.coord_proj, mask,
                              self.modality_id, support_mass=mass,
                              **self._meta(field.shape[0], N, dlo, dhi,
                                           field.device, field.dtype))


# --------------------------------------------------------------------------
# (b) Profile encoder — sparse vertical columns -> physical depth-band tokens
# --------------------------------------------------------------------------
_BAND_SCALE = 1500.0     # deepest band edge across the 20-/23-level protocols


def default_depth_bands(max_depth: float) -> list[tuple[float, float]]:
    """Task-5 physical depth bands: 0-50 / 50-200 / 200-500 / 500-max, plus a
    1000-1500 band only when the grid actually extends below 1000 m (never
    report a >1000 m band on the 20-level task)."""
    bands = [(0.0, 50.0), (50.0, 200.0), (200.0, 500.0)]
    if max_depth > 1000.0:
        bands += [(500.0, 1000.0), (1000.0, float(min(max_depth, 1500.0)))]
    else:
        bands += [(500.0, float(max_depth))]
    return bands


class ProfileEncoder(nn.Module):
    """(B, P, C, D) profiles -> one token per *physical depth band* per profile.

    Levels are assigned to bands by their physical depth, not by index, so one
    trained encoder handles the 20-level grid, the extended 23-level grid, and
    irregular / ragged vertical sampling (each level is embedded by a small
    per-level MLP and masked-mean-pooled within its band — no dependence on
    the number of levels or their divisibility).

    Each band token carries (Task 5): pooled T/S content, validity mask,
    lat/lon, lower/upper/midpoint depth (via band features + the coordinate
    channel), the represented depth span, month, and a ``support_mass`` equal
    to the *valid* physical depth span (metres) the token actually represents
    (the Task-4 observation-measure seam; a level's represented thickness is
    its inter-level interval clipped to the band).

    A band with no finite level (e.g. below a shallow float's max depth, or a
    completely missing segment) is masked out.  Variable profile count P
    (including P=0) is native; ``valid`` (B, P) marks real profiles inside a
    padded batch.  ``depths`` may be the constructor grid (default), a shared
    (D,) grid, or per-profile (B, P, D) ragged depths (NaN = absent level).
    """

    def __init__(self, depth_grid=None, c_vars: int = 2, d_model: int = 128,
                 depth_bands=None, modality: str = "profile",
                 normalize_per_profile: bool = True,
                 family_id: int | None = None):
        super().__init__()
        # Task 4 (first implementation): normalize band masses within each
        # profile so a profile with more sampled depth levels does not
        # automatically receive more total mass.  Set False to emit the raw
        # represented span in metres.
        self.normalize_per_profile = normalize_per_profile
        if depth_bands is None:
            assert depth_grid is not None, \
                "need depth_grid (to derive default bands) or explicit depth_bands"
            depth_bands = default_depth_bands(
                float(np.nanmax(np.asarray(depth_grid, dtype="float64"))))
        self.bands = tuple((float(lo), float(hi)) for lo, hi in depth_bands)
        self.n_bands = len(self.bands)
        self.family_id = self.n_bands if family_id is None else family_id
        self.c_vars = c_vars
        self.modality_id = MODALITIES[modality]
        if depth_grid is not None:
            self.register_buffer("depth_grid", torch.as_tensor(
                np.asarray(depth_grid, dtype="float32")))
        else:
            self.depth_grid = None
        lo = torch.tensor([b[0] for b in self.bands], dtype=torch.float32)
        hi = torch.tensor([b[1] for b in self.bands], dtype=torch.float32)
        self.register_buffer("band_lo", lo)
        self.register_buffer("band_hi", hi)
        # per-level embedding: C values + C finite flags + 2 depth features
        self.level_mlp = nn.Sequential(
            nn.Linear(2 * c_vars + 2, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model))
        # band metadata: lo, hi, mid, valid-span fraction
        self.band_proj = nn.Linear(4, d_model)
        self.coord_proj = nn.Linear(N_COORD_FEATS, d_model)
        self.out_features = d_model

    def forward(self, prof, lat, lon, month, valid=None,
                depths=None) -> TokenBatch:
        B, P, C, D = prof.shape
        S = self.n_bands
        assert C == self.c_vars
        if P == 0:
            return TokenBatch.empty(B, self.out_features, device=prof.device)
        if valid is None:
            valid = torch.ones(B, P, dtype=torch.bool, device=prof.device)
        month = torch.as_tensor(month, device=prof.device)
        if month.ndim == 0:
            month = month[None, None].expand(B, P)
        elif month.ndim == 1:
            month = month[:, None].expand(B, P)

        # ---- depths -> (B, P, D), NaN = absent level ----
        if depths is None:
            assert self.depth_grid is not None and self.depth_grid.numel() == D, \
                "no depths given and constructor grid absent or wrong length"
            depths = self.depth_grid
        depths = torch.as_tensor(depths, device=prof.device, dtype=prof.dtype)
        if depths.ndim == 1:
            depths = depths[None, None, :].expand(B, P, D)
        d_ok = torch.isfinite(depths)
        d = torch.nan_to_num(depths, nan=0.0)

        # ---- represented thickness per level: inter-level interval ----
        if D > 1:
            gap = (d[..., 1:] - d[..., :-1]).clamp(min=0.0)
            lower = torch.cat([(d[..., :1] - gap[..., :1] / 2).clamp(min=0.0),
                               d[..., :-1] + gap / 2], dim=-1)
            upper = torch.cat([d[..., :-1] + gap / 2,
                               d[..., -1:] + gap[..., -1:] / 2], dim=-1)
        else:   # single-level profile: nominal 50 m of represented column
            lower = (d - 25.0).clamp(min=0.0)
            upper = d + 25.0

        # ---- band membership by level-centre depth ----
        lo = self.band_lo.view(1, 1, S, 1)
        hi = self.band_hi.view(1, 1, S, 1)
        dc = d.unsqueeze(2)                                    # (B,P,1,D)
        member = (dc >= lo) & (dc < hi)
        member[:, :, -1] |= (dc[:, :, 0] >= self.band_hi[-1])  # deepest level -> last band
        member &= d_ok.unsqueeze(2)

        # ---- per-level embedding (band-agnostic, pooled by membership) ----
        finite = torch.isfinite(prof)                          # (B,P,C,D)
        lvl_ok = finite.any(dim=2)                             # (B,P,D)
        vals = torch.nan_to_num(prof, nan=0.0).permute(0, 1, 3, 2)   # (B,P,D,C)
        flags = finite.to(prof.dtype).permute(0, 1, 3, 2)
        dfeat = torch.stack([d / _BAND_SCALE,
                             torch.log1p(d.clamp(min=0.0))
                             / math.log1p(_BAND_SCALE)], dim=-1)
        emb = self.level_mlp(torch.cat([vals, flags, dfeat], dim=-1))  # (B,P,D,d)

        w = (member & lvl_ok.unsqueeze(2)).to(prof.dtype)      # (B,P,S,D)
        count = w.sum(dim=-1)                                  # (B,P,S)
        pooled = torch.einsum("bpsd,bpde->bpse", w, emb) / count.clamp(min=1.0)[..., None]

        # ---- support mass: valid represented span, clipped to the band ----
        span = (torch.minimum(upper.unsqueeze(2), hi)
                - torch.maximum(lower.unsqueeze(2), lo)).clamp(min=0.0)
        mass = (span * w).sum(dim=-1)                          # (B,P,S) metres

        # ---- band metadata features ----
        mid = (self.band_lo + self.band_hi) / 2                # (S,)
        bfeat = torch.stack([
            (self.band_lo / _BAND_SCALE).expand(B, P, S),
            (self.band_hi / _BAND_SCALE).expand(B, P, S),
            (mid / _BAND_SCALE).expand(B, P, S),
            mass / (self.band_hi - self.band_lo).clamp(min=1e-6),
        ], dim=-1)
        content = pooled + self.band_proj(bfeat)               # (B,P,S,d)

        seg_ok = count > 0                                     # (B,P,S)
        mask = (valid[:, :, None] & seg_ok).reshape(B, P * S)
        coord = torch.stack([
            torch.nan_to_num(lat, nan=0.0)[:, :, None].expand(B, P, S),
            torch.nan_to_num(lon, nan=0.0)[:, :, None].expand(B, P, S),
            mid[None, None, :].expand(B, P, S),
            month[:, :, None].expand(B, P, S).to(prof.dtype),
        ], dim=-1).reshape(B, P * S, 4)

        out_mass = mass
        if self.normalize_per_profile:
            out_mass = mass / mass.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        dev = prof.device
        meta = dict(
            parent_id=torch.arange(P, device=dev)[None, :, None]
                .expand(B, P, S).reshape(B, P * S),
            family_id=torch.full((B, P * S), self.family_id,
                                 dtype=torch.long, device=dev),
            variable_id=torch.full((B, P * S), VAR_IDS["MULTI"],
                                   dtype=torch.long, device=dev),
            depth_lower=self.band_lo[None, None, :].expand(B, P, S)
                .reshape(B, P * S).to(prof.dtype),
            depth_upper=self.band_hi[None, None, :].expand(B, P, S)
                .reshape(B, P * S).to(prof.dtype))
        return _finish_tokens(content.reshape(B, P * S, -1), coord,
                              self.coord_proj, mask, self.modality_id,
                              support_mass=out_mass.reshape(B, P * S), **meta)


# --------------------------------------------------------------------------
# (c) Point encoder — scattered scalar observations
# --------------------------------------------------------------------------
class PointEncoder(nn.Module):
    """(B, N) scalar observations -> one token each.

    Each point is (value, var_id, lat, lon, depth, month); var_id indexes
    ``variables`` given at construction (e.g. ("TEMP", "SALT", "SST", "SSS")).
    """

    def __init__(self, variables=("TEMP", "SALT", "SST", "SSS"),
                 d_model: int = 128, modality: str = "point"):
        super().__init__()
        self.variables = tuple(variables)
        self.modality_id = MODALITIES[modality]
        self.var_emb = nn.Embedding(len(self.variables), 16)
        self.val_proj = nn.Linear(1 + 1 + 16, d_model)   # value, finite, var emb
        self.coord_proj = nn.Linear(N_COORD_FEATS, d_model)

    def forward(self, values, var_id, lat, lon, depth, month,
                valid=None) -> TokenBatch:
        B, N = values.shape
        if N == 0:
            return TokenBatch.empty(B, self.val_proj.out_features,
                                    device=values.device)
        if valid is None:
            valid = torch.ones(B, N, dtype=torch.bool, device=values.device)
        month = torch.as_tensor(month, device=values.device)
        if month.ndim == 0:
            month = month[None, None].expand(B, N)
        elif month.ndim == 1:
            month = month[:, None].expand(B, N)

        finite = torch.isfinite(values)
        v = torch.nan_to_num(values, nan=0.0)
        content = self.val_proj(torch.cat([
            v[..., None], finite[..., None].to(v.dtype), self.var_emb(var_id),
        ], dim=-1))
        mask = valid & finite
        coord = torch.stack([torch.nan_to_num(lat, nan=0.0),
                             torch.nan_to_num(lon, nan=0.0),
                             torch.nan_to_num(depth, nan=0.0),
                             month.to(v.dtype)], dim=-1)
        # Task 4: no support mass yet (points excluded from the first MVP's
        # mass scheme; MBCA falls back to uniform-within-modality), but
        # provenance is stamped so duplication is detectable.
        meta = dict(
            parent_id=torch.arange(N, device=values.device)[None].expand(B, N),
            family_id=torch.zeros(B, N, dtype=torch.long, device=values.device),
            variable_id=var_id.to(torch.long),
            depth_lower=coord[..., 2], depth_upper=coord[..., 2])
        return _finish_tokens(content, coord, self.coord_proj, mask,
                              self.modality_id, **meta)


# --------------------------------------------------------------------------
# Shared-latent model contract
# --------------------------------------------------------------------------
class SharedLatentModel(nn.Module):
    """encode -> fuse -> decode contract for the shared-latent method.

    ``obs`` is a dict mapping a modality key (a key of ``encoders``) to the
    kwargs of that encoder's forward.  A missing modality is an absent key —
    no retraining, no placeholder inputs.  Subclasses implement ``fuse``
    (tokens -> latent) and ``decode`` (latent + query coords -> values);
    Week 3-4 supplies the Perceiver-style implementations.
    """

    def __init__(self, encoders: dict[str, nn.Module], d_model: int = 128):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.d_model = d_model
        self.modality_emb = nn.Embedding(len(MODALITIES), d_model)

    def encode(self, obs: dict, batch: int, device=None) -> TokenBatch:
        parts = [self.encoders[k](**kw) for k, kw in obs.items()]
        tb = TokenBatch.cat(parts) if parts else TokenBatch.empty(
            batch, self.d_model, device=device)
        emb = tb.emb + self.modality_emb(tb.modality) * tb.mask.unsqueeze(-1)
        kw = {n: getattr(tb, n)
              for n in TokenBatch._OPT_INT + TokenBatch._OPT_FLOAT}
        return TokenBatch(emb, tb.coord, tb.modality, tb.mask, **kw)

    def fuse(self, tokens: TokenBatch) -> torch.Tensor:
        """TokenBatch -> latent (B, L, d)."""
        raise NotImplementedError

    def decode(self, latent: torch.Tensor, query_coord: torch.Tensor) -> torch.Tensor:
        """latent (B, L, d) + query (B, Q, 4) -> (B, Q, c_out)."""
        raise NotImplementedError

    def forward(self, obs: dict, query_coord: torch.Tensor) -> torch.Tensor:
        tokens = self.encode(obs, batch=query_coord.shape[0],
                             device=query_coord.device)
        return self.decode(self.fuse(tokens), query_coord)


class SharedLatentStub(SharedLatentModel):
    """Trivial reference implementation: masked mean-pool + MLP query head.

    Exists to exercise the interface (masks, variable counts, missing
    modalities, coordinate consistency, gradients) end-to-end before the real
    fusion core lands.  Permutation-invariant and padding-invariant by
    construction.  NOT the method — do not benchmark it as such.
    """

    def __init__(self, encoders, d_model: int = 128, c_out: int = 2,
                 hidden: int = 256):
        super().__init__(encoders, d_model)
        self.query_proj = nn.Linear(N_COORD_FEATS, d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, c_out))

    def fuse(self, tokens: TokenBatch) -> torch.Tensor:
        w = tokens.mask.to(tokens.emb.dtype).unsqueeze(-1)           # (B,N,1)
        total = (tokens.emb * w).sum(dim=1)
        count = w.sum(dim=1).clamp(min=1.0)
        return (total / count).unsqueeze(1)                          # (B,1,d)

    def decode(self, latent, query_coord):
        q = self.query_proj(coord_features(query_coord))             # (B,Q,d)
        z = latent.expand(-1, q.shape[1], -1)                        # (B,Q,d)
        return self.head(torch.cat([z, q], dim=-1))


# --------------------------------------------------------------------------
# Bridge from the existing baseline pipeline (prepare_month samples)
# --------------------------------------------------------------------------
def sample_to_obs(sample, grid, norm, device="cpu",
                  include=("profiles", "surf", "woa")) -> dict:
    """Convert a ``baselines.prepare_month`` sample into an observation dict.

    Values are anomaly-space z-scores (``norm`` is an ``anomaly.AnomNorm``).
    Returns kwargs keyed for a SharedLatentModel built with encoders named
    'profiles' (ProfileEncoder), 'surf' (GridPatchEncoder c_in=2) and 'woa'
    (GridPatchEncoder c_in=2, volume mode).  Batch dimension is 1 (one month).
    """
    mo = int(sample["month"])
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a, dtype="float32"),
                                  device=device)
    obs = {}
    if "profiles" in include:
        prof = sample["prof"]
        P = prof["lat"].size
        cols = []
        for v in ("TEMP", "SALT"):
            z = norm.z3d(v, sample["obs"][v], mo)                    # (D,H,W)
            ii, jj = prof["ij"][:, 0], prof["ij"][:, 1]
            cols.append(z[:, ii, jj].T)                              # (P, D)
        vals = np.stack(cols, axis=1) if P else np.zeros((0, 2, grid.ndepth))
        obs["profiles"] = dict(
            prof=t(vals)[None], lat=t(prof["lat"])[None],
            lon=t(prof["lon"])[None], month=torch.tensor([mo], device=device))
    if "surf" in include and sample.get("surf"):
        chans = []
        for sv in ("SST", "SSS"):
            arr = sample["surf"].get(sv)
            chans.append(np.full((grid.nlat, grid.nlon), np.nan, "float32")
                         if arr is None else norm.zsurf(sv, arr, mo))
        obs["surf"] = dict(
            field=t(np.stack(chans))[None], lat=t(grid.lat), lon=t(grid.lon),
            month=torch.tensor([mo], device=device))
    if "woa" in include:
        vol = np.stack([norm.z3d(v, sample["woa"][v], mo) for v in ("TEMP", "SALT")])
        obs["woa"] = dict(
            field=t(vol)[None], lat=t(grid.lat), lon=t(grid.lon),
            month=torch.tensor([mo], device=device), depth=t(grid.depth))
    return obs


def make_query_coords(lat, lon, depth, month, device="cpu") -> torch.Tensor:
    """Broadcast per-point (lat, lon, depth, month) arrays -> (1, Q, 4)."""
    arrs = np.broadcast_arrays(np.asarray(lat, "float32"),
                               np.asarray(lon, "float32"),
                               np.asarray(depth, "float32"),
                               np.asarray(month, "float32"))
    q = np.stack([a.ravel() for a in arrs], axis=-1)
    return torch.as_tensor(q, device=device)[None]


def build_stub(grid, d_model: int = 128, patch=(10, 12)) -> SharedLatentStub:
    """Wire the three modality encoders into the stub for this project's grid."""
    return SharedLatentStub({
        "profiles": ProfileEncoder(grid.depth, c_vars=2, d_model=d_model),
        "surf": GridPatchEncoder(2, d_model=d_model, patch=patch,
                                 modality="surf_grid"),
        "woa": GridPatchEncoder(2, d_model=d_model, patch=patch,
                                modality="woa_grid"),
    }, d_model=d_model)
