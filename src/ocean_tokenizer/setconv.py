"""A translation-equivariant backbone for sparse profiles: SetConv -> U-Net -> query.

The Perceiver-IO line encodes every profile as five depth-band tokens, squeezes
the month through 32 unaddressed latent slots, and asks a query to find what it
needs by attention.  Two properties the reconstruction problem has are absent
from that design: the answer at a point depends mostly on observations *near*
it, and the map from observations to field is the same map everywhere in the
box.  Neither is built in, so both have to be learned from 264 months.

This backbone builds both in, following the convolutional conditional neural
process (Gordon et al., ICLR 2020; Vaughan et al. 2022 for climate fields;
Andersson et al. 2023 for sensor placement; Allen et al. 2025 use the same
encoder on raw weather observations):

  1 **SetConv encoding.**  Each profile is smeared onto a regular grid over the
    region with a Gaussian of learned width (one per variable group), giving two
    channels per level and variable: a value channel (kernel-weighted mean) and
    a density channel (how much evidence reached that cell).  A cell nothing
    reached has density 0 and value 0, which is the honest encoding of "no
    observation here" -- nothing is fabricated.
  2 **A U-Net processor** on that grid.  Convolution is translation
    equivariant, so a float 200 km east of a query is treated the way a float
    200 km east of any other query is, and the receptive field grows with depth
    instead of being set by an attention length scale that has to be learned.
  3 **Bilinear query readout.**  The processor writes the whole anomaly field
    (levels x channels); a query is answered by bilinear interpolation at its
    own position and level.  There is no bottleneck between observation and
    query: information is limited by grid resolution, not by slot count.

Levels stay separate end to end -- no depth-band pooling anywhere.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

KM_PER_DEG = 111.195


class _Block(nn.Module):
    def __init__(self, c_in: int, c_out: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        g = math.gcd(groups, c_out) or 1      # any width, not only multiples of 8
        self.n1 = nn.GroupNorm(g, c_out)
        self.n2 = nn.GroupNorm(g, c_out)
        self.skip = (nn.Identity() if c_in == c_out
                     else nn.Conv2d(c_in, c_out, 1))

    def forward(self, x):
        h = F.silu(self.n1(self.conv1(x)))
        h = F.silu(self.n2(self.conv2(h)))
        return h + self.skip(x)


class SetConvUNet(nn.Module):
    """Sparse profiles -> gridded anomaly field -> values at arbitrary queries.

    ``box`` is ``dict(lat=(lo, hi), lon=(lo, hi))``; ``levels`` the cohort's
    level depths in metres.  The grid is ``(ny, nx)`` cells over the box.
    ``ell_km`` initialises the SetConv width (learned, one per channel group).
    """

    def __init__(self, box: dict, levels, ny: int = 50, nx: int = 102,
                 width: int = 48, depth: int = 3, ell_km: float = 75.0,
                 max_lead: int = 6, c_vars: int = 2, n_surface: int = 0):
        super().__init__()
        #: extra gridded input channels (satellite SST / SLA / SSS and their
        #: validity mask), supplied already interpolated to this grid. They are
        #: the information a profiles-only optimal interpolation does not have.
        self.n_surface = int(n_surface)
        lev = torch.as_tensor(levels, dtype=torch.float32)
        self.register_buffer("levels", lev)
        self.L = lev.numel()
        self.C = c_vars
        (la0, la1), (lo0, lo1) = box["lat"], box["lon"]
        self.la0, self.la1, self.lo0, self.lo1 = (float(la0), float(la1),
                                                  float(lo0), float(lo1))
        self.ny, self.nx = int(ny), int(nx)
        gy = la0 + (torch.arange(ny, dtype=torch.float32) + 0.5) * (la1 - la0) / ny
        gx = lo0 + (torch.arange(nx, dtype=torch.float32) + 0.5) * (lo1 - lo0) / nx
        self.register_buffer("grid_lat", gy)
        self.register_buffer("grid_lon", gx)
        # one learned SetConv width per variable, in log km
        self.log_ell = nn.Parameter(torch.full((c_vars,), math.log(ell_km)))
        c_in = 2 * c_vars * self.L + 1 + self.n_surface   # value + density + lead (+ surface)
        chans = [width * (2 ** min(i, 2)) for i in range(depth)]
        self.stem = _Block(c_in, chans[0])
        self.down = nn.ModuleList(_Block(chans[i], chans[i + 1])
                                  for i in range(depth - 1))
        self.up = nn.ModuleList(_Block(chans[i + 1] + chans[i], chans[i])
                                for i in reversed(range(depth - 1)))
        self.head = nn.Conv2d(chans[0], c_vars * self.L, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.max_lead = int(max_lead)

    # ---- encoder ---------------------------------------------------------
    def setconv(self, prof, lat, lon):
        """(K,C,L) profile z-values at (K,) positions -> (2*C*L, ny, nx)."""
        valid = torch.isfinite(prof)
        v = torch.nan_to_num(prof) * valid
        dy = (self.grid_lat[:, None] - lat[None, :]) * KM_PER_DEG       # (ny,K)
        clat = torch.cos(torch.deg2rad(lat)).clamp(min=0.2)
        dlon = (self.grid_lon[:, None] - lon[None, :] + 180.0) % 360.0 - 180.0
        dx = dlon * KM_PER_DEG * clat[None, :]                          # (nx,K)
        d2 = dy[:, None, :] ** 2 + dx[None, :, :] ** 2                  # (ny,nx,K)
        ell = self.log_ell.exp().clamp(5.0, 2000.0)
        outs = []
        for ci in range(self.C):
            w = torch.exp(-0.5 * d2 / ell[ci] ** 2)                     # (ny,nx,K)
            num = torch.einsum("yxk,kl->lyx", w, v[:, ci, :])
            den = torch.einsum("yxk,kl->lyx", w, valid[:, ci, :].to(v.dtype))
            outs.append(num / den.clamp(min=1e-3))
            outs.append(torch.log1p(den))
        return torch.cat(outs, 0)

    # ---- decoder ---------------------------------------------------------
    def read(self, field, query):
        """``field`` (C*L, ny, nx), ``query`` (Q,4) lat/lon/depth_m/month -> (Q,C)."""
        gy = (query[:, 0] - self.la0) / (self.la1 - self.la0) * 2 - 1
        gx = (query[:, 1] - self.lo0) / (self.lo1 - self.lo0) * 2 - 1
        grid = torch.stack([gx, gy], -1)[None, :, None, :]             # (1,Q,1,2)
        s = F.grid_sample(field[None], grid, mode="bilinear",
                          padding_mode="border", align_corners=False)
        s = s[0, :, :, 0].transpose(0, 1).reshape(-1, self.C, self.L)  # (Q,C,L)
        li = torch.argmin((query[:, 2][:, None] - self.levels[None, :]).abs(), dim=1)
        return s[torch.arange(s.shape[0], device=s.device), :, li]

    def forward(self, prof, lat, lon, month, query, lead=None, surface=None):
        x = self.setconv(prof, lat, lon)
        lv = 0.0 if lead is None else float(int(lead)) / max(self.max_lead, 1)
        x = torch.cat([x, torch.full_like(x[:1], lv)], 0)
        if self.n_surface:
            if surface is None:
                raise ValueError(f"this model was built with n_surface="
                                 f"{self.n_surface}; pass the surface channels")
            x = torch.cat([x, torch.nan_to_num(surface)], 0)
        x = x[None]
        h = self.stem(x)
        skips = [h]
        for blk in self.down:
            h = blk(F.avg_pool2d(h, 2))
            skips.append(h)
        for i, blk in enumerate(self.up):
            s = skips[-2 - i]
            h = F.interpolate(h, size=s.shape[-2:], mode="bilinear",
                              align_corners=False)
            h = blk(torch.cat([h, s], 1))
        return self.read(self.head(h)[0], query)
