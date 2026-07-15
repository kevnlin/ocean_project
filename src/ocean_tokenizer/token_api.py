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
N_COORD_FEATS = 7
_DEPTH_SCALE = 1000.0             # ~ the deepest target level (985 m)


def coord_features(coord: torch.Tensor) -> torch.Tensor:
    """(..., 4) physical (lat_deg, lon_deg, depth_m, month 1-12) -> (..., 7).

    Deterministic, encoder-agnostic: the same location/time always produces the
    same features, whichever modality (or the query side) supplies it.
    """
    lat, lon, depth, month = coord.unbind(-1)
    lon_r = torch.deg2rad(lon)
    return torch.stack([
        lat / 90.0,
        torch.sin(lon_r), torch.cos(lon_r),
        depth / _DEPTH_SCALE,
        torch.log1p(depth.clamp(min=0.0)) / math.log1p(_DEPTH_SCALE),
        torch.sin(2 * math.pi * month / 12.0),
        torch.cos(2 * math.pi * month / 12.0),
    ], dim=-1)


# --------------------------------------------------------------------------
# TokenBatch — the unified OceanObservationToken schema
# --------------------------------------------------------------------------
@dataclass
class TokenBatch:
    """A batch of observation tokens from one or more modalities.

    emb      : (B, N, d)  encoded token embeddings (masked-out tokens are 0)
    coord    : (B, N, 4)  physical (lat, lon, depth_m, month); 0 where masked
    modality : (B, N)     int64 id into MODALITIES
    mask     : (B, N)     bool, True = real token (False = padding / all-NaN)
    """
    emb: torch.Tensor
    coord: torch.Tensor
    modality: torch.Tensor
    mask: torch.Tensor

    @property
    def n_valid(self) -> torch.Tensor:            # (B,) valid tokens per item
        return self.mask.sum(dim=1)

    def to(self, device) -> "TokenBatch":
        return TokenBatch(self.emb.to(device), self.coord.to(device),
                          self.modality.to(device), self.mask.to(device))

    @staticmethod
    def empty(batch: int, d_model: int, device=None) -> "TokenBatch":
        return TokenBatch(
            emb=torch.zeros(batch, 0, d_model, device=device),
            coord=torch.zeros(batch, 0, 4, device=device),
            modality=torch.zeros(batch, 0, dtype=torch.long, device=device),
            mask=torch.zeros(batch, 0, dtype=torch.bool, device=device))

    @staticmethod
    def cat(parts: list["TokenBatch"]) -> "TokenBatch":
        assert len(parts) > 0, "TokenBatch.cat needs at least one part"
        B = parts[0].emb.shape[0]
        assert all(p.emb.shape[0] == B for p in parts), "batch sizes differ"
        return TokenBatch(
            emb=torch.cat([p.emb for p in parts], dim=1),
            coord=torch.cat([p.coord for p in parts], dim=1),
            modality=torch.cat([p.modality for p in parts], dim=1),
            mask=torch.cat([p.mask for p in parts], dim=1))


def _finish_tokens(content_emb, coord, coord_proj, mask, modality_id):
    """Assemble a TokenBatch: content + coord embedding, zeroed where masked."""
    coord = torch.nan_to_num(coord, nan=0.0)
    emb = content_emb + coord_proj(coord_features(coord))
    emb = emb * mask.unsqueeze(-1)
    coord = coord * mask.unsqueeze(-1)
    modality = torch.full(mask.shape, modality_id, dtype=torch.long,
                          device=mask.device)
    return TokenBatch(emb=emb, coord=coord, modality=modality, mask=mask)


# --------------------------------------------------------------------------
# (a) Grid-patch encoder — dense gridded fields (surface or volume)
# --------------------------------------------------------------------------
class GridPatchEncoder(nn.Module):
    """Dense gridded field -> one token per (ph, pw) spatial patch.

    forward(field, lat, lon, month, depth=None)
      * field (B, C, H, W)     surface field; token depth = ``depth`` (default 0)
      * field (B, C, D, H, W)  volume; ``depth`` is the (D,) level grid and the
                               encoding is one token per (level, patch)
    lat (H,), lon (W,) in degrees; month (B,) 1-12 (broadcast to every token).
    NaN cells become (value 0, finite-flag 0); an all-NaN patch is masked out.
    """

    def __init__(self, c_in: int, d_model: int = 128, patch=(10, 12),
                 modality: str = "surf_grid"):
        super().__init__()
        self.c_in = c_in
        self.ph, self.pw = patch
        self.modality_id = MODALITIES[modality]
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
            x = x.reshape(B, C, nh, ph, nw, pw).permute(0, 2, 4, 1, 3, 5)
            return x.reshape(B, nh * nw, C * ph * pw)
        v = patchify(vals)
        m = patchify(finite.float())
        content = self.val_proj(torch.cat([v, m], dim=-1))          # (B, N, d)
        mask = m.bool().any(dim=-1)                                  # (B, N)

        # patch-centre coordinates (edge-clamped index midpoint)
        dev = field.device
        ci = torch.clamp(torch.arange(nh, device=dev) * ph + ph // 2, max=H - 1)
        cj = torch.clamp(torch.arange(nw, device=dev) * pw + pw // 2, max=W - 1)
        clat = lat[ci][:, None].expand(nh, nw).reshape(-1)           # (N,)
        clon = lon[cj][None, :].expand(nh, nw).reshape(-1)
        N = nh * nw
        coord = torch.stack([
            clat[None].expand(B, N),
            clon[None].expand(B, N),
            depth_val[:, None].expand(B, N),
            month[:, None].expand(B, N).to(field.dtype),
        ], dim=-1)                                                   # (B, N, 4)
        return content, coord, mask

    def forward(self, field, lat, lon, month, depth=None) -> TokenBatch:
        month = torch.as_tensor(month, device=field.device)
        if month.ndim == 0:
            month = month[None].expand(field.shape[0])
        if field.ndim == 4:                                          # surface
            B = field.shape[0]
            d = (torch.zeros(B, device=field.device) if depth is None
                 else torch.as_tensor(float(depth), device=field.device).expand(B))
            content, coord, mask = self._encode_2d(field, lat, lon, month, d)
        elif field.ndim == 5:                                        # volume
            B, C, D, H, W = field.shape
            assert depth is not None and len(depth) == D, \
                "volume input needs the (D,) depth grid"
            flat = field.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            dval = torch.as_tensor(depth, device=field.device,
                                   dtype=field.dtype).repeat(B)
            mo = month.repeat_interleave(D)
            content, coord, mask = self._encode_2d(flat, lat, lon, mo, dval)
            n = content.shape[1]
            content = content.reshape(B, D * n, -1)
            coord = coord.reshape(B, D * n, 4)
            mask = mask.reshape(B, D * n)
        else:
            raise ValueError(f"expected (B,C,H,W) or (B,C,D,H,W), got {field.shape}")
        return _finish_tokens(content, coord, self.coord_proj, mask,
                              self.modality_id)


# --------------------------------------------------------------------------
# (b) Profile encoder — sparse vertical columns -> multiple tokens each
# --------------------------------------------------------------------------
class ProfileEncoder(nn.Module):
    """(B, P, C, D) profiles -> n_segments tokens per profile.

    The column is split into contiguous depth segments (default 4 x 5 levels at
    D=20) so vertical / water-mass structure survives tokenisation instead of
    being flattened into one vector.  Variable profile count P (including P=0)
    is native; ``valid`` (B, P) marks real profiles inside a padded batch.
    A segment with no finite level (e.g. below a shallow float's max depth) is
    masked out.
    """

    def __init__(self, depth_grid, c_vars: int = 2, d_model: int = 128,
                 n_segments: int = 4, modality: str = "profile"):
        super().__init__()
        depth_grid = torch.as_tensor(np.asarray(depth_grid, dtype="float32"))
        D = depth_grid.numel()
        assert D % n_segments == 0, "depth levels must divide into segments"
        self.seg_len = D // n_segments
        self.n_segments = n_segments
        self.c_vars = c_vars
        self.modality_id = MODALITIES[modality]
        self.register_buffer("seg_depth",
                             depth_grid.reshape(n_segments, self.seg_len).mean(1))
        self.val_proj = nn.Linear(2 * c_vars * self.seg_len, d_model)
        self.coord_proj = nn.Linear(N_COORD_FEATS, d_model)

    def forward(self, prof, lat, lon, month, valid=None) -> TokenBatch:
        B, P, C, D = prof.shape
        S, L = self.n_segments, self.seg_len
        assert C == self.c_vars and D == S * L
        if P == 0:
            return TokenBatch.empty(B, self.val_proj.out_features,
                                    device=prof.device)
        if valid is None:
            valid = torch.ones(B, P, dtype=torch.bool, device=prof.device)
        month = torch.as_tensor(month, device=prof.device)
        if month.ndim == 0:
            month = month[None, None].expand(B, P)
        elif month.ndim == 1:
            month = month[:, None].expand(B, P)

        finite = torch.isfinite(prof)
        vals = torch.nan_to_num(prof, nan=0.0)
        # (B, P, C, S, L) -> (B, P, S, C*L)
        def segment(x):
            x = x.reshape(B, P, C, S, L).permute(0, 1, 3, 2, 4)
            return x.reshape(B, P, S, C * L)
        v = segment(vals)
        m = segment(finite.float())
        content = self.val_proj(torch.cat([v, m], dim=-1))           # (B,P,S,d)
        seg_ok = m.bool().any(dim=-1)                                # (B,P,S)
        mask = (valid[:, :, None] & seg_ok).reshape(B, P * S)

        coord = torch.stack([
            torch.nan_to_num(lat, nan=0.0)[:, :, None].expand(B, P, S),
            torch.nan_to_num(lon, nan=0.0)[:, :, None].expand(B, P, S),
            self.seg_depth[None, None, :].expand(B, P, S),
            month[:, :, None].expand(B, P, S).to(prof.dtype),
        ], dim=-1).reshape(B, P * S, 4)
        return _finish_tokens(content.reshape(B, P * S, -1), coord,
                              self.coord_proj, mask, self.modality_id)


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
        return _finish_tokens(content, coord, self.coord_proj, mask,
                              self.modality_id)


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
        return TokenBatch(emb, tb.coord, tb.modality, tb.mask)

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
