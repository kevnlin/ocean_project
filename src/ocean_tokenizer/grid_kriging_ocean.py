"""A gridded ocean state with lead evolution and a kriging read-out of the raw profiles.

One month of observations becomes one state on a regular latitude-longitude grid; a forecast
moves the state forward; a query reads the state, then corrects it with the exact values of the
profiles around it.

  encode   the month's profiles are averaged into their grid cells (20 levels x T/S, which
           levels were seen, how many profiles), stacked with the dense SST/SSS/SSH grids, a
           land/missing mask, position and season, and a U-Net turns them into the state G_t
           (``state_channels`` numbers per cell).
  evolve   G_{t+l} = G_t + Phi(G_t, l): a second U-Net, conditioned on the lead l in months,
           whose output layer starts at zero, so an untrained model persists the state.
  decode   a query (position, level, lead) reads G_{t+l} at its position (bilinear, periodic in
           longitude) and, with its own features and the level and lead, gives a background b
           and a hidden vector. Each of the 32 nearest input profiles is decoded the same way
           from G_t at its own position, at the query's level, lead 0, which gives its
           innovation d_j = observed - b_j. Kriging attention (``kriging_attention``) carries
           those innovations to the query, damped by exp(-l / tau) for a forecast:

               mean = background + b + sum_j w_j d_j,     w = rho (K + R)^-1 k

``background="prior"`` keeps the month's profiles out of the background the innovations are taken
against, as in data assimilation: a prior U-Net reads only the surface grids, position and season
(G^b_t); the analysis U-Net reads G^b_t and the rasterised profiles (G^a_t) and is what the lead
evolution moves. A reconstruction is then ``background + b(G^b_t) + kriging`` -- the profiles enter
it only through the kriging of their exact values -- and a forecast is ``background +
b(G^a_{t+l}) + damped kriging``; the neighbours' innovations are always against G^b_t. With
``background="joint"`` one U-Net reads everything and serves both.

The output layer of b starts at zero and the flow kernel starts at the OI's stationary kernel,
so an untrained model returns the OI-kernel kriging of the neighbours' anomalies at lead 0 and
that analysis damped by the lead decay at a forecast lead. No frozen OI and no local OI-error
scale are read. Units are the scene's per-depth z anomalies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .kriging_attention import MODES, KrigingAttention

KRIGING = MODES + ("softmax", "none")
BACKGROUNDS = ("joint", "prior")


class GridSpec(NamedTuple):
    """Cell centres ``lat0 + i dlat`` and ``lon0 + j dlon`` (degrees), ``n_lat x n_lon`` cells."""
    lat0: float = -89.5
    dlat: float = 1.
    lon0: float = .5
    dlon: float = 1.
    n_lat: int = 180
    n_lon: int = 360

    def as_tuple(self):
        return tuple(self)

    def cell(self, lat, lon):
        """Index of the cell containing each point (clamped in latitude, periodic in longitude)."""
        lat, lon = torch.as_tensor(lat, dtype=torch.float64), torch.as_tensor(lon, dtype=torch.float64)
        iy = torch.floor((lat - self.lat0 + self.dlat / 2) / self.dlat).long().clamp(0, self.n_lat - 1)
        ix = torch.floor(((lon - self.lon0 + self.dlon / 2) % 360.) / self.dlon).long() % self.n_lon
        return iy, ix


def rasterize_profiles(lat: Tensor, lon: Tensor, values: Tensor, valid: Tensor, grid: GridSpec) -> Tensor:
    """``[4L + 1, H, W]``: cell means of the valid values (level-major, T then S), which values
    were seen, and log(1 + profiles in the cell). ``values`` and ``valid`` are ``[n, L, 2]``."""
    n, levels = values.shape[:2]
    iy, ix = grid.cell(lat.detach(), lon.detach())
    cell = (iy * grid.n_lon + ix).to(values.device)
    ok = valid & torch.isfinite(values)
    flat = torch.where(ok, values, torch.zeros_like(values)).reshape(n, 2 * levels)
    seen = ok.reshape(n, 2 * levels).to(values.dtype)
    cells = grid.n_lat * grid.n_lon
    sums = values.new_zeros(cells, 2 * levels).index_add(0, cell, flat)
    counts = values.new_zeros(cells, 2 * levels).index_add(0, cell, seen)
    profiles = values.new_zeros(cells).index_add(0, cell, values.new_ones(n))
    out = torch.cat((sums / counts.clamp_min(1.), (counts > 0).to(values.dtype), profiles.log1p()[:, None]), 1)
    return out.T.reshape(4 * levels + 1, grid.n_lat, grid.n_lon)


def sample_grid_features(field: Tensor, lat: Tensor, lon: Tensor, grid: GridSpec) -> Tensor:
    """Bilinear sampling of ``field [C,H,W]`` at points of any shape, periodic in longitude:
    ``[..., C]``."""
    fy = (lat - grid.lat0) / grid.dlat
    fx = ((lon - grid.lon0) % 360.) / grid.dlon
    i0 = fy.floor().long().clamp(0, grid.n_lat - 1)
    j0 = fx.floor().long() % grid.n_lon
    i1, j1 = (i0 + 1).clamp(max=grid.n_lat - 1), (j0 + 1) % grid.n_lon
    wy, wx = (fy - fy.floor())[..., None].to(field.dtype), (fx - fx.floor())[..., None].to(field.dtype)
    f = field.permute(1, 2, 0)
    return (f[i0, j0] * (1 - wy) * (1 - wx) + f[i0, j1] * (1 - wy) * wx
            + f[i1, j0] * wy * (1 - wx) + f[i1, j1] * wy * wx)


class SoftmaxReadout(nn.Module):
    """The attention baseline for the read-out: a softmax over learned scores of the neighbours,
    a distance penalty with a learned length scale per variable and level, and a null key, so
    the weights are positive and sum to at most one (each neighbour votes on its own).

    Same inputs and outputs as ``KrigingAttention``; ``variance`` is the weighted spread of the
    innovations, as in the softmax path of ``latent_ocean``.
    """

    def __init__(self, width: int, init_ell_km):
        super().__init__()
        log_ell = torch.as_tensor(init_ell_km, dtype=torch.float32).log()
        if log_ell.ndim != 2 or log_ell.shape[0] != 2:
            raise ValueError("init_ell_km must be [2, n_levels]")
        self.log_ell = nn.Parameter(log_ell)
        self.scorer = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, 2))
        self.null_score = nn.Linear(width, 2)

    def forward(self, h, local, offsets, innovation, valid, level, decay=None):
        q, k = innovation.shape[:2]
        dtype = self.scorer[0].weight.dtype
        scores = self.scorer(torch.cat((h.to(dtype)[:, None].expand(-1, k, -1), local.to(dtype)), -1))
        ell = self.log_ell.clamp(math.log(10.), math.log(5000.))[:, level].T.exp()       # [Q,2]
        distance = offsets.to(dtype).square().sum(-1, keepdim=True) / ell[:, None].square()
        scores = (scores - .5 * distance).masked_fill(~valid, float("-inf"))
        scores = torch.cat((scores, self.null_score(h.to(dtype))[:, None]), 1)
        weights = scores.softmax(1)[:, :k].transpose(1, 2)                                # [Q,2,K]
        d = torch.where(valid, innovation, torch.zeros_like(innovation)).transpose(1, 2).to(dtype)
        estimate = (weights * d).sum(-1)
        variance = (weights * (d - estimate[..., None]).square()).sum(-1)
        if decay is not None:
            estimate = estimate * decay.to(dtype)
        out_dtype = innovation.dtype
        return {"estimate": estimate.to(out_dtype), "variance": variance.to(out_dtype),
                "weights": weights.to(out_dtype), "length_scale_km": ell.detach().to(out_dtype)}


def _groups(channels):
    return next(g for g in (8, 4, 2, 1) if channels % g == 0)


class PeriodicConv(nn.Module):
    """3x3 convolution, periodic in longitude (the last axis) and zero-padded in latitude."""

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=0)

    def forward(self, x):
        return self.conv(F.pad(F.pad(x, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1)))


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, cond_dim=None):
        super().__init__()
        self.norm1, self.conv1 = nn.GroupNorm(_groups(c_in), c_in), PeriodicConv(c_in, c_out)
        self.norm2, self.conv2 = nn.GroupNorm(_groups(c_out), c_out), PeriodicConv(c_out, c_out)
        self.skip = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1)
        self.film = None
        if cond_dim:
            self.film = nn.Linear(cond_dim, 2 * c_out)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x, cond=None):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.norm2(h)
        if self.film is not None and cond is not None:
            scale, shift = self.film(cond)[:, :, None, None].chunk(2, 1)
            h = h * (1 + scale) + shift
        return self.skip(x) + self.conv2(F.silu(h))


class UNet(nn.Module):
    """A small U-Net on the sphere's lat-lon grid; ``channels`` per resolution, coarsest last."""

    def __init__(self, c_in, c_out, channels, cond_dim=None, zero_head=False):
        super().__init__()
        channels = tuple(int(c) for c in channels)
        self.depth = len(channels) - 1
        self.stem = PeriodicConv(c_in, channels[0])
        self.down = nn.ModuleList(ResBlock(c, c, cond_dim) for c in channels)
        self.pool = nn.ModuleList(PeriodicConv(a, b, stride=2) for a, b in zip(channels[:-1], channels[1:]))
        self.middle = ResBlock(channels[-1], channels[-1], cond_dim)
        self.up = nn.ModuleList(ResBlock(b + a, a, cond_dim) for a, b in zip(channels[:-1], channels[1:]))
        self.head_norm = nn.GroupNorm(_groups(channels[0]), channels[0])
        self.head = nn.Conv2d(channels[0], c_out, 1)
        if zero_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x, cond=None):
        h, w = x.shape[-2:]
        multiple = 2 ** self.depth
        pad_h, pad_w = (-h) % multiple, (-w) % multiple
        if pad_w:
            x = F.pad(x, (0, pad_w, 0, 0), mode="circular")
        if pad_h:
            x = F.pad(x, (0, 0, pad_h // 2, pad_h - pad_h // 2))
        x = self.stem(x)
        skips = []
        for i, block in enumerate(self.down):
            x = block(x, cond)
            if i < self.depth:
                skips.append(x)
                x = self.pool[i](x)
        x = self.middle(x, cond)
        for i in reversed(range(self.depth)):
            x = F.interpolate(x, size=skips[i].shape[-2:], mode="nearest")
            x = self.up[i](torch.cat((x, skips[i]), 1), cond)
        x = self.head(F.silu(self.head_norm(x)))
        return x[..., pad_h // 2:pad_h // 2 + h, :w]


def _tuples(value):
    if isinstance(value, (list, tuple)):
        return tuple(_tuples(v) for v in value)
    return value


@dataclass(frozen=True)
class GridKrigingConfig:
    grid: tuple = GridSpec().as_tuple()
    n_levels: int = 20
    n_query_features: int = 58
    state_channels: int = 128
    encoder_channels: tuple = (64, 128, 192, 256)
    evolution_channels: tuple = (64, 128, 192)
    hidden: int = 256
    level_dim: int = 32
    lead_dim: int = 32
    max_lead: int = 6
    kriging: str = "ka_flow"
    background: str = "joint"
    kriging_ell_km: tuple = ()
    kriging_gamma: tuple = ()
    init_decay_months: float = 3.0
    std_min: float = .03
    std_max: float = 3.
    initial_std: float = .6

    def __post_init__(self):
        for name in ("grid", "encoder_channels", "evolution_channels", "kriging_ell_km", "kriging_gamma"):
            object.__setattr__(self, name, _tuples(getattr(self, name)))
        if self.kriging not in KRIGING:
            raise ValueError(f"kriging must be one of {KRIGING}")
        if self.background not in BACKGROUNDS:
            raise ValueError(f"background must be one of {BACKGROUNDS}")
        if self.kriging != "none" and (len(self.kriging_ell_km) != 2 or len(self.kriging_gamma) != 2):
            raise ValueError("a kriging read-out needs kriging_ell_km and kriging_gamma as [2][n_levels]")
        if not 0 < self.std_min < self.initial_std < self.std_max:
            raise ValueError("require 0 < std_min < initial_std < std_max")

    @property
    def grid_spec(self):
        return GridSpec(*self.grid)

    @property
    def profile_channels(self):
        return 4 * self.n_levels + 1

    @property
    def surface_channels(self):
        # SST/SSS/SSH and their masks (6), sin/cos of lat and lon (4), season (2)
        return 6 + 4 + 2

    @property
    def input_channels(self):
        return self.profile_channels + self.surface_channels

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GridKrigingModel(nn.Module):

    def __init__(self, config: GridKrigingConfig):
        super().__init__()
        self.config = c = config
        grid = c.grid_spec
        lat = torch.deg2rad(grid.lat0 + grid.dlat * torch.arange(grid.n_lat, dtype=torch.float32))
        lon = torch.deg2rad(grid.lon0 + grid.dlon * torch.arange(grid.n_lon, dtype=torch.float32))
        lat2, lon2 = torch.meshgrid(lat, lon, indexing="ij")
        self.register_buffer("static", torch.stack((lat2.sin(), lat2.cos(), lon2.sin(), lon2.cos())), persistent=False)
        self.prior_encoder = None
        if c.background == "prior":
            self.prior_encoder = UNet(c.surface_channels, c.state_channels, c.encoder_channels)
            self.encoder = UNet(c.state_channels + c.profile_channels, c.state_channels, c.encoder_channels)
        else:
            self.encoder = UNet(c.input_channels, c.state_channels, c.encoder_channels)
        self.lead_embedding = nn.Sequential(nn.Linear(3, c.lead_dim), nn.SiLU(), nn.Linear(c.lead_dim, c.lead_dim))
        self.evolution = UNet(c.state_channels, c.state_channels, c.evolution_channels, cond_dim=c.lead_dim,
                              zero_head=True)
        self.level_embedding = nn.Embedding(c.n_levels, c.level_dim)
        width = c.state_channels + c.n_query_features + c.level_dim + c.lead_dim
        self.decoder = nn.Sequential(nn.Linear(width, c.hidden), nn.SiLU(), nn.Linear(c.hidden, c.hidden),
                                     nn.SiLU(), nn.LayerNorm(c.hidden))
        self.mean_head = nn.Sequential(nn.Linear(c.hidden, c.hidden), nn.SiLU(), nn.Linear(c.hidden, 2))
        nn.init.zeros_(self.mean_head[-1].weight)
        nn.init.zeros_(self.mean_head[-1].bias)
        if c.kriging == "none":
            self.kriging = None
        elif c.kriging == "softmax":
            self.kriging = SoftmaxReadout(c.hidden, c.kriging_ell_km)
        else:
            self.kriging = KrigingAttention(c.hidden, c.kriging, c.kriging_ell_km, c.kriging_gamma)
        self.log_decay_months = nn.Parameter(torch.full((2, c.n_levels), math.log(c.init_decay_months)))
        self.std_head = nn.Sequential(nn.Linear(c.hidden + 2, c.hidden), nn.SiLU(), nn.Linear(c.hidden, 2))
        nn.init.zeros_(self.std_head[-1].weight)
        fraction = (c.initial_std - c.std_min) / (c.std_max - c.std_min)
        nn.init.constant_(self.std_head[-1].bias, math.log(fraction / (1 - fraction)))

    def _lead(self, lead, like):
        x = like.new_tensor(float(lead) / self.config.max_lead)
        return self.lead_embedding(torch.stack((x, x.square(), x.clamp_min(0).sqrt()))[None])[0]

    def encode(self, scene: Mapping[str, Tensor]) -> dict[str, Any]:
        """Inputs of the month only: its profiles (``obs_*`` tokens, level-major per profile),
        ``surface_grid [3,H,W]`` (NaN where missing), ``input_month`` and ``lead``."""
        c, grid = self.config, self.config.grid_spec
        coord, values = scene["obs_coord"], scene["obs_innovation"]
        n = coord.shape[0] // c.n_levels
        coord = coord.reshape(n, c.n_levels, 4)
        profiles = rasterize_profiles(coord[:, 0, 0], coord[:, 0, 1], values.reshape(n, c.n_levels, 2),
                                      scene["obs_valid"].reshape(n, c.n_levels, 2).bool(), grid)
        surface = scene["surface_grid"].to(profiles.dtype)
        seen = torch.isfinite(surface)
        month = 2 * math.pi * (float(scene["input_month"]) % 12) / 12.
        season = profiles.new_tensor([math.sin(month), math.cos(month)])[:, None, None].expand(2, grid.n_lat, grid.n_lon)
        context = torch.cat((torch.where(seen, surface, torch.zeros_like(surface)), seen.to(profiles.dtype),
                             self.static.to(profiles.dtype), season))[None]
        lead = int(scene["lead"])
        prior = None
        if self.prior_encoder is not None:
            prior = self.prior_encoder(context)
            grid0 = self.encoder(torch.cat((prior, profiles[None].to(prior.dtype)), 1)) if lead else None
        else:
            grid0 = self.encoder(torch.cat((profiles[None], context), 1))
        grid_lead = grid0 + self.evolution(grid0, self._lead(lead, grid0)[None]) if lead else grid0
        # the state a query reads, and the state the neighbours' innovations are taken against
        query_grid = grid_lead if grid_lead is not None else prior
        neighbour_grid = prior if prior is not None else grid0
        diagnostics = {"state_rms": query_grid.detach().float().square().mean().sqrt()}
        if lead:
            diagnostics["evolution_rms"] = (grid_lead - grid0).detach().float().square().mean().sqrt()
        return {"query_grid": query_grid[0], "neighbour_grid": neighbour_grid[0], "lead": lead,
                "aux_loss": query_grid.new_zeros(()), "diagnostics": diagnostics}

    def _hidden(self, field, lat, lon, features, level, lead_embedding):
        g = sample_grid_features(field, lat, lon, self.config.grid_spec)
        level = self.level_embedding(level)
        lead = lead_embedding.expand(*g.shape[:-1], -1)
        return self.decoder(torch.cat((g, features.to(g.dtype), level.to(g.dtype), lead.to(g.dtype)), -1))

    def decode(self, scene: Mapping[str, Tensor], state: Mapping[str, Any]) -> dict[str, Any]:
        c = self.config
        lead, level = state["lead"], scene["query_level"]
        q, k = scene["local_innovation"].shape[:2]
        coord = scene["query_coord"]
        h = self._hidden(state["query_grid"], coord[:, 0], coord[:, 1], scene["query_features"], level,
                         self._lead(lead, state["query_grid"]))
        background = self.mean_head(h)
        mean = scene["query_background"] + background
        diagnostics = {"background_rms": background.detach().float().square().mean().sqrt()}
        variance = mean.new_ones(q, 2)
        if self.kriging is not None:
            features = scene["local_features"][..., :c.n_query_features]
            h_local = self._hidden(state["neighbour_grid"], scene["local_lat"], scene["local_lon"], features,
                                   level[:, None].expand(q, k), self._lead(0, state["neighbour_grid"]))
            innovation, offsets = scene["local_innovation"], scene["local_offsets"][..., :2]
            valid = (scene["local_valid"].bool() & torch.isfinite(innovation)
                     & torch.isfinite(offsets).all(-1, keepdim=True))
            innovation = torch.where(valid, innovation - self.mean_head(h_local).to(innovation.dtype),
                                     torch.zeros_like(innovation))
            offsets = torch.nan_to_num(offsets)
            decay = torch.exp(-float(lead) / self.log_decay_months[:, level].T.exp())
            out = self.kriging(h, h_local, offsets, innovation, valid, level, decay=decay)
            mean = mean + out["estimate"]
            variance = out["variance"]
            used = valid.transpose(1, 2)
            diagnostics.update(
                kriging_negative_share=(((out["weights"] < 0) & used).sum((0, 2))
                                        / used.sum((0, 2)).clamp_min(1)).detach(),
                kriging_length_scale_km=out["length_scale_km"].mean(0).detach(),
                lead_decay=decay.mean(0).detach())
        logits = self.std_head(torch.cat((h, variance.to(h.dtype)), -1))
        std = c.std_min + (c.std_max - c.std_min) * logits.sigmoid()
        return {"mean": mean, "std": std.to(mean.dtype), "aux_loss": state["aux_loss"],
                "diagnostics": {**state.get("diagnostics", {}), **diagnostics}}

    def forward(self, scene: Mapping[str, Tensor]) -> dict[str, Any]:
        return self.decode(scene, self.encode(scene))
