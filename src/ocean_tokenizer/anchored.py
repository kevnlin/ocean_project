"""Query-anchored fusion: a satellite first guess, corrected by in-situ innovations.

The token line pools every observation before a query can read it: a profile
becomes four depth-band means, a satellite field becomes 3-degree patch means,
and a month becomes 32 slots. The audit of 2026-10-07 measured what that costs
(a level's value cannot be read back from its band token; the satellite fields
move the score by 1-2 %). This module is the other way round. Nothing is pooled:

  first guess   ``FirstGuess`` maps what the satellites see AT THE QUERY (its own
                position and day) to a temperature / salinity anomaly profile;
  innovation    every nearby profile contributes ``observed - first guess at the
                profile's own position``, exact values at exact levels;
  analysis      ``InnovationAnalysis`` carries those innovations to the query,
                ``first guess + sum_i w_i * innovation_i``.

Three rules for the weights ``w`` share one set of length scales, so they can be
compared with nothing else changed:

  softmax   a normalised kernel average against a background key: positive
            weights, each neighbour one vote, so an observation ingested five
            times is counted five times;
  dfs       the same with the evidence prior ``beta * log tau`` of DFS-Attention,
            ``tau`` being the profile's ridge leverage (``profile_leverage``);
  kriging   the weights of the linear system the leverage itself comes from,
            ``(K + gamma I) w = k``: redundancy and the background are handled
            by the solve, and weights may be negative.

Offsets are kilometres east / north in the azimuthal-equidistant plane at the
query, so distances from the query are exact great-circle distances.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn

R_EARTH_KM = 6371.0


class Neighbours(NamedTuple):
    idx: torch.Tensor      # (Q, k) indices into the observations, nearest first
    dx: torch.Tensor       # (Q, k) km east of the query
    dy: torch.Tensor       # (Q, k) km north of the query
    dist: torch.Tensor     # (Q, k) great-circle km


def _xyz(lat, lon):
    la, lo = torch.deg2rad(lat), torch.deg2rad(lon)
    return torch.stack([torch.cos(la) * torch.cos(lo), torch.cos(la) * torch.sin(lo),
                        torch.sin(la)], dim=-1)


def _offsets(q_lat, q_lon, o_lat, o_lon, idx) -> Neighbours:
    p1, p2 = torch.deg2rad(q_lat)[:, None], torch.deg2rad(o_lat)[idx]
    dl = torch.deg2rad(o_lon[idx] - q_lon[:, None])
    a = (torch.sin((p2 - p1) / 2) ** 2
         + torch.cos(p1) * torch.cos(p2) * torch.sin(dl / 2) ** 2).clamp(0.0, 1.0)
    dist = 2.0 * R_EARTH_KM * torch.asin(torch.sqrt(a))
    bearing = torch.atan2(torch.sin(dl) * torch.cos(p2),
                          torch.cos(p1) * torch.sin(p2)
                          - torch.sin(p1) * torch.cos(p2) * torch.cos(dl))
    return Neighbours(idx, dist * torch.sin(bearing), dist * torch.cos(bearing), dist)


def neighbours(q_lat, q_lon, o_lat, o_lon, k: int, chunk: int = 4096) -> Neighbours:
    """The ``k`` observations nearest each query (all of them when fewer exist)."""
    k = min(int(k), o_lat.shape[0])
    oxyz = _xyz(o_lat, o_lon)
    idx = torch.cat([torch.cdist(_xyz(q_lat[i:i + chunk], q_lon[i:i + chunk]), oxyz)
                     .topk(k, dim=-1, largest=False).indices
                     for i in range(0, q_lat.shape[0], chunk)])
    return _offsets(q_lat, q_lon, o_lat, o_lon, idx)


def _separations(dx, dy, ell_x, ell_y, dt, ell_t, ds, ell_s):
    """Scaled squared separations: query-observation (..., k) and observation-
    observation (..., k, k). Every leg is an offset FROM THE QUERY, so the
    difference of two offsets is the separation of the two observations."""
    qo, oo = 0.0, 0.0
    for a, ell in ((dx, ell_x), (dy, ell_y), (dt, ell_t), (ds, ell_s)):
        if a is None:
            continue
        e = torch.as_tensor(ell, dtype=a.dtype, device=a.device)
        qo = qo + (a / e.unsqueeze(-1)) ** 2
        oo = oo + ((a.unsqueeze(-1) - a.unsqueeze(-2)) / e.unsqueeze(-1).unsqueeze(-1)) ** 2
    return qo, oo


def kriging_weights(dx, dy, valid, ell_x, ell_y, gamma, dt=None, ell_t=None, ds=None,
                    ell_s=None):
    """Weights of the optimal-interpolation analysis, ``(K + gamma I) w = k``.

    ``K`` and ``k`` are Gaussian in the scaled separations; ``gamma`` is the
    observation-to-background error variance ratio. A neighbour whose ``valid``
    is False gets weight 0 and leaves the others' system untouched. Shapes
    broadcast: offsets ``(..., k)``, scales and ``gamma`` ``(...)``.
    """
    qo, oo = _separations(dx, dy, ell_x, ell_y, dt, ell_t, ds, ell_s)
    v = valid.to(torch.float64)
    K = torch.exp(-0.5 * oo).to(torch.float64) * v.unsqueeze(-1) * v.unsqueeze(-2)
    kq = torch.exp(-0.5 * qo).to(torch.float64) * v
    g = torch.as_tensor(gamma, dtype=torch.float64, device=K.device).unsqueeze(-1)
    A = K + torch.diag_embed(g * v + (1.0 - v))        # a masked row is the identity
    w = torch.linalg.solve(A, kq.unsqueeze(-1)).squeeze(-1)
    return (w * v).to(dx.dtype)


def softmax_weights(logits, valid, bg_logit, log_mass=None, beta=1.0):
    """Attention over the neighbours and one background key; the background's
    share is ``1 - weights.sum(-1)``. ``log_mass`` adds the evidence prior."""
    z = logits if log_mass is None else logits + beta * log_mass
    z = z.masked_fill(~valid, float("-inf"))
    bg = torch.as_tensor(bg_logit, dtype=z.dtype, device=z.device).expand(z.shape[:-1])
    return torch.softmax(torch.cat([z, bg.unsqueeze(-1)], dim=-1), dim=-1)[..., :-1]


def profile_leverage(lat, lon, ell_km, noise, k: int, t=None, ell_days=None,
                     chunk: int = 4096):
    """Degrees of freedom for signal of each profile, ``tau_i = H_ii``.

    The ridge leverage of DFS-Attention (``dfs.dfs_scores``) at profile level:
    a Gaussian signal covariance in horizontal (and time) separation, error
    variance ``noise``, solved over each profile's ``k`` nearest neighbours.
    An isolated profile keeps ``1 / (1 + noise)``; ``c`` copies of one profile
    share a single degree of freedom, ``1 / (c + noise)`` each.
    """
    n = lat.shape[0]
    k = min(int(k), n)
    xyz = _xyz(lat, lon)
    tau = torch.empty(n, dtype=torch.float64, device=lat.device)
    for i in range(0, n, chunk):
        rows = torch.arange(i, min(i + chunk, n), device=lat.device)
        d = torch.cdist(xyz[rows], xyz)
        d[torch.arange(rows.numel(), device=lat.device), rows] = -1.0     # itself first
        nb = _offsets(lat[rows], lon[rows], lat, lon,
                      d.topk(k, dim=-1, largest=False).indices)
        dt = None if t is None else t[nb.idx] - t[rows, None]
        _, oo = _separations(nb.dx, nb.dy, ell_km, ell_km, dt, ell_days, None, None)
        A = torch.exp(-0.5 * oo).to(torch.float64)
        A = A + noise * torch.eye(k, dtype=torch.float64, device=lat.device)
        e0 = torch.zeros(rows.numel(), k, 1, dtype=torch.float64, device=lat.device)
        e0[:, 0] = 1.0
        tau[rows] = 1.0 - noise * torch.linalg.solve(A, e0)[:, 0, 0]
    return tau.clamp(0.0, 1.0)


class InnovationAnalysis(nn.Module):
    """Carries in-situ innovations to the queries: ``(Q, n_fields)`` increments.

    One learned length scale per field and separation leg (km east, km north,
    and, when used, days and "state", a scalar such as the sea-level anomaly at
    the profile). ``softmax`` and ``dfs`` weigh against a learned background
    logit; ``kriging`` solves with a learned error ratio ``gamma``.
    """

    MODES = ("softmax", "dfs", "kriging")

    def __init__(self, n_fields: int, mode: str = "kriging", k: int = 32,
                 init_km: float = 250.0, init_days: float = 20.0, init_gamma: float = 0.3,
                 use_time: bool = False, use_state: bool = False):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode, self.k = mode, int(k)
        full = lambda v: nn.Parameter(torch.full((n_fields,), float(v)))
        self.log_ell_x, self.log_ell_y = full(math.log(init_km)), full(math.log(init_km))
        self.log_ell_t = full(math.log(init_days)) if use_time else None
        self.log_ell_s = full(0.0) if use_state else None
        if mode == "kriging":
            self.log_gamma = full(math.log(init_gamma))
        else:
            self.bg_logit = full(0.0)
        if mode == "dfs":
            self.beta = full(1.0)

    def forward(self, q_lat, q_lon, o_lat, o_lon, innov, q_t=None, o_t=None, q_s=None,
                o_s=None, log_mass=None, chunk: int = 1024):
        out = [self._analyse(q_lat[i:i + chunk], q_lon[i:i + chunk], o_lat, o_lon, innov,
                             None if q_t is None else q_t[i:i + chunk], o_t,
                             None if q_s is None else q_s[i:i + chunk], o_s, log_mass)
               for i in range(0, q_lat.shape[0], chunk)]
        return torch.cat(out)

    def _analyse(self, q_lat, q_lon, o_lat, o_lon, innov, q_t, o_t, q_s, o_s, log_mass):
        nb = neighbours(q_lat, q_lon, o_lat, o_lon, self.k)
        d = innov[nb.idx].transpose(1, 2)                          # (Q, F, k)
        valid = torch.isfinite(d)
        d = torch.nan_to_num(d)
        dx, dy = nb.dx.unsqueeze(1), nb.dy.unsqueeze(1)            # (Q, 1, k)
        row = lambda p: None if p is None else torch.exp(p).unsqueeze(0)   # (1, F)
        dt = ds = None
        if self.log_ell_t is not None:
            dt = (o_t[nb.idx] - q_t[:, None]).unsqueeze(1)
        if self.log_ell_s is not None:
            ds = (o_s[nb.idx] - q_s[:, None]).unsqueeze(1)
        legs = dict(ell_x=row(self.log_ell_x), ell_y=row(self.log_ell_y),
                    dt=dt, ell_t=row(self.log_ell_t), ds=ds, ell_s=row(self.log_ell_s))
        if self.mode == "kriging":
            w = kriging_weights(dx, dy, valid, gamma=row(self.log_gamma), **legs)
        else:
            qo, _ = _separations(dx, dy, legs["ell_x"], legs["ell_y"], dt, legs["ell_t"],
                                 ds, legs["ell_s"])
            mass = beta = None
            if self.mode == "dfs":
                m = log_mass[nb.idx]                               # (Q, k) or (Q, k, F)
                mass = m.unsqueeze(1) if m.dim() == 2 else m.transpose(1, 2)
                beta = self.beta.view(1, -1, 1)
            w = softmax_weights(-0.5 * qo, valid, self.bg_logit.unsqueeze(0),
                                log_mass=mass, beta=beta)
        return (w * d).sum(-1)


class FirstGuess(nn.Module):
    """What the satellites see at a query -> anomaly at every field."""

    def __init__(self, n_in: int, n_fields: int, width: int = 512, depth: int = 3):
        super().__init__()
        layers, c = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(c, width), nn.SiLU()]
            c = width
        self.net = nn.Sequential(*layers, nn.Linear(c, n_fields))

    def forward(self, x):
        return self.net(x)
