"""DFS — Degrees of Freedom for Signal: scale-aware effective-evidence estimation.

This is the evidence half of DFS-Attention (docs/dfs_attention.md).  It answers
one question about a heterogeneous observation set:

    not "how many tokens are there?" but "how many independent, useful degrees
    of freedom do these tokens carry AT THE TARGET RECONSTRUCTION SCALE?"

Formulation
-----------
Treat the token set as observations of a field with signal covariance ``K`` and
observation-error covariance ``S``.  The analysis is the kriging/OI update
``f_hat = K (K + S)^-1 y`` and its influence (hat) matrix is

    H = K (K + S)^-1 ,      tau_i = H_ii in [0, 1] ,     DFS = tr H = sum_i tau_i

which is exactly the data-assimilation *degrees of freedom for signal*, and
simultaneously the ridge-leverage score of observation i (the ridge is the
observation noise; the "background" prior is K itself).  ``tau_i`` is the
fraction of observation i that the analysis takes from the observation rather
than from the background, so it is the marginal information i contributes after
every other observation is accounted for.  Reading off the identity

    tau_i = 1 - sigma_i^2 [(K + S)^-1]_ii

is all we need, and it is what this module computes.

Consequences (the behaviours Section 11 of the plan asks for) come out of the
mathematics rather than from hand-written rules:

* an isolated, precise observation gets tau ~ 1;
* c exact re-ingestions of ONE raw measurement share a single degree of freedom
  (see ``multiplicity`` below) — total DFS is *exactly* unchanged;
* two observations at the same place but different depths are nearly
  independent under the vertical kernel, so DFS ~ doubles;
* dense sampling in a vertically smooth column is discounted, while the same
  density across a thermocline is not, because the vertical length scale is
  stratification-dependent;
* coarsening the target resolution lengthens every kernel scale, so dense
  observations become mutually redundant — asking for a finer target retains
  more of them.

The three kernel legs and the provenance factor are the plan's
``k_ij = k_h(dx) k_z(dz) k_t(dt) k_s(source_i, source_j)``.  Each geometric leg
is a Gibbs (non-stationary) Gaussian kernel, which stays positive definite even
though every token carries its own length scales:

    k_d(i,j) = sqrt( 2 l_i l_j / (l_i^2 + l_j^2) ) * exp( -delta^2 / (l_i^2 + l_j^2) )

Cost
----
The exact ``(K+S)^-1`` is O(N^3).  The kernel decays, so we use the standard
data-assimilation *localisation* approximation: the diagonal of the inverse is
computed from each token's own neighbourhood (the same argument that makes the
LETKF local).  Cost is O(N k^3) with k ~ 32 neighbours — milliseconds for the
~6.5k observation tokens of one analysis month, and exact in the two limits
that matter (isolated tokens, and duplicate groups, which are each other's
nearest neighbours by construction).
"""
from __future__ import annotations
from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from .token_api import TokenBatch, MODALITIES, DEFAULT_SIGMA, KM_PER_DEG

R_EARTH_KM = 6371.0
_EPS = 1e-9

# --------------------------------------------------------------------------
# Target reconstruction scale — the "at what scale?" half of scale-awareness
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetScale:
    """Resolution of the field being reconstructed (the query's dx_q, dz_q).

    dx_km : horizontal target resolution (km)
    dz_m  : vertical target resolution (m)
    dt_d  : temporal target resolution (days)
    """
    dx_km: float = 1.0 * KM_PER_DEG      # protocol_v1: the 1 deg analysis grid
    dz_m: float = 50.0                   # protocol_v1: median level thickness
    dt_d: float = 30.0                   # protocol_v1: monthly analysis

    def as_tensor(self, device=None, dtype=torch.float32) -> torch.Tensor:
        return torch.tensor([self.dx_km, self.dz_m, self.dt_d],
                            device=device, dtype=dtype)


PROTOCOL_SCALE = TargetScale()
COARSE_SCALE = TargetScale(dx_km=5.0 * KM_PER_DEG, dz_m=250.0, dt_d=90.0)
FINE_SCALE = TargetScale(dx_km=0.25 * KM_PER_DEG, dz_m=10.0, dt_d=5.0)


# --------------------------------------------------------------------------
# Physical length-scale priors — "a practical first version can use several
# predefined depth regimes" (plan Section 6)
# --------------------------------------------------------------------------
# (name, depth_top_m, horizontal ell_h km, vertical ell_z m)
DEPTH_REGIMES = (
    ("mixed layer",  0.0,   150.0,  75.0),   # vertically near-uniform: levels
                                             # inside the ML are interchangeable
    ("thermocline",  50.0,  150.0,  35.0),   # sharp structure: short l_z, dense
                                             # levels stay informative
    ("intermediate", 300.0, 250.0, 150.0),   # smoother water masses
    ("deep ocean",   1000.0, 400.0, 400.0),  # slowly varying
)
_REG_TOP = torch.tensor([r[1] for r in DEPTH_REGIMES])
_REG_LH = torch.tensor([r[2] for r in DEPTH_REGIMES])
_REG_LZ = torch.tensor([r[3] for r in DEPTH_REGIMES])

TEMPORAL_SCALE_D = 15.0        # e-folding time of a monthly-scale anomaly
STRAT_BETA = 1.0               # strength of the stratification modifier
STRAT_REF = 1.0                # z-units per 100 m that halve l_z at beta = 1
MIN_LZ_M = 3.0                 # never claim structure finer than this
MIN_LH_KM = 10.0
S_CROSS = 0.8                  # k_s between different processing streams


def regime_scales(depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(...,) depth in m -> (l_h km, l_z m) from the depth-regime table.

    Piecewise-constant in the regime, then linearly interpolated across regime
    boundaries so the scales (and therefore the evidence) are continuous in
    depth — a token at 49 m and one at 51 m must not see a jump.
    """
    top = _REG_TOP.to(depth.device, depth.dtype)
    lh = _REG_LH.to(depth.device, depth.dtype)
    lz = _REG_LZ.to(depth.device, depth.dtype)
    # regime centres: midpoints of the regime spans (last one extrapolated)
    ctr = torch.cat([(top[:-1] + top[1:]) / 2, top[-1:] * 1.5])
    # linear interpolation in log-depth between regime centres
    ld = torch.log1p(depth.clamp(min=0.0))
    lc = torch.log1p(ctr)
    hi = torch.bucketize(ld, lc).clamp(1, len(ctr) - 1)
    lo = hi - 1
    x0, x1 = lc[lo], lc[hi]
    w = ((ld - x0) / (x1 - x0).clamp(min=_EPS)).clamp(0.0, 1.0)
    return lh[lo] * (1 - w) + lh[hi] * w, lz[lo] * (1 - w) + lz[hi] * w


class SupportScales(nn.Module):
    """Per-token 3-D length scales l_h, l_z, l_t at a given target resolution.

    Physics first (depth regimes + local stratification), target scale second:

        l_phys = l_regime(depth) / (1 + beta * strat/strat_ref)
        l_eff  = sqrt( l_phys^2 + delta_target^2 )

    The second line is the representer argument: a target field at resolution
    ``delta`` is the true field convolved with an averaging footprint of that
    width, so its covariance is the physical covariance convolved with the same
    footprint.  Coarse target -> long l -> nearby observations are mutually
    redundant; fine target -> l falls back to the physical scale and the same
    observations regain their independence.  This is the only place the target
    resolution enters, and it makes the resolution sweep monotone by
    construction.

    ``learn_residual=True`` adds the plan's optional learned correction: a small
    MLP predicts a bounded multiplicative residual on each log length scale, so
    the physical priors can be adjusted by data without being replaced (the
    residual is exactly 1.0 at initialisation).
    """

    def __init__(self, learn_residual: bool = False, beta: float = STRAT_BETA,
                 strat_ref: float = STRAT_REF, temporal_scale: float = TEMPORAL_SCALE_D,
                 max_residual: float = 2.0):
        super().__init__()
        self.beta = float(beta)
        self.strat_ref = float(strat_ref)
        self.temporal_scale = float(temporal_scale)
        self.log_max = math.log(float(max_residual))
        self.learn_residual = bool(learn_residual)
        if learn_residual:
            self.res = nn.Sequential(nn.Linear(6, 32), nn.SiLU(),
                                     nn.Linear(32, 3))
            nn.init.zeros_(self.res[-1].weight)
            nn.init.zeros_(self.res[-1].bias)
        else:
            self.res = None

    def forward(self, depth: torch.Tensor, strat: torch.Tensor,
                target: TargetScale) -> tuple[torch.Tensor, torch.Tensor,
                                              torch.Tensor]:
        lh0, lz0 = regime_scales(depth)
        damp = 1.0 + self.beta * (strat.clamp(min=0.0) / max(self.strat_ref, _EPS))
        lz0 = lz0 / damp                                   # thermocline -> short
        lt0 = torch.full_like(depth, self.temporal_scale)
        if self.res is not None:
            dn = depth.clamp(min=0.0) / 1000.0
            f = torch.stack([dn, torch.log1p(depth.clamp(min=0.0)) / 7.0,
                             strat.clamp(0.0, 10.0),
                             torch.log(lh0 / 100.0), torch.log(lz0 / 100.0),
                             torch.full_like(dn, math.log(max(target.dz_m, 1.0)
                                                          / 100.0))], dim=-1)
            r = torch.tanh(self.res(f)) * self.log_max
            lh0, lz0, lt0 = (lh0 * r[..., 0].exp(), lz0 * r[..., 1].exp(),
                             lt0 * r[..., 2].exp())
        lh = torch.sqrt(lh0 ** 2 + target.dx_km ** 2).clamp(min=MIN_LH_KM)
        lz = torch.sqrt(lz0 ** 2 + target.dz_m ** 2).clamp(min=MIN_LZ_M)
        lt = torch.sqrt(lt0 ** 2 + target.dt_d ** 2).clamp(min=0.5)
        return lh, lz, lt


# --------------------------------------------------------------------------
# Record multiplicity — exact duplication is consolidated, not merely damped
# --------------------------------------------------------------------------
def record_groups(tokens: TokenBatch, valid: torch.Tensor | None = None):
    """Group tokens by the raw measurement they stand for.

    Identity is ``(modality, record_id)``; ``record_id < 0`` means "no record
    identity" and is never merged.  A group of c tokens is the SAME measurement
    re-ingested c times — an exact duplicate, or a real-time / delayed-mode
    pair of one float cycle — so the group carries the information of ONE
    observation, not c.  Rather than solving a perfectly-correlated (and
    therefore singular) c x c block, we keep one representative per group in
    the linear system and split its degree of freedom evenly:

        tau_group = H_rep,rep     (solved once, well conditioned)
        tau_i     = tau_group / c for every member

    Total DFS is then *exactly* invariant to duplication, by construction.

    Returns (multiplicity (B,N) float, rep_index (B,N) long, is_rep (B,N) bool).
    """
    B, N = tokens.mask.shape
    dev = tokens.mask.device
    dt = tokens.emb.dtype
    valid = tokens.mask if valid is None else valid
    mult = torch.ones(B, N, device=dev, dtype=dt)
    ar = torch.arange(N, device=dev)
    rep = ar[None].expand(B, N).clone()
    if tokens.record_id is None:
        return mult, rep, torch.ones(B, N, dtype=torch.bool, device=dev)
    rid = tokens.record_id
    span = int(rid.abs().max()) + 2 if N else 2
    key = tokens.modality.to(torch.long) * span + rid
    # tokens without a record identity (or masked out) are their own group
    solo = (rid < 0) | ~valid
    key = torch.where(solo, -(ar[None].expand(B, N) + 1) - span * len(MODALITIES),
                      key)
    for b in range(B):
        uniq, inv, cnt = torch.unique(key[b], return_inverse=True,
                                      return_counts=True)
        mult[b] = cnt[inv].to(dt)
        first = torch.full((uniq.numel(),), N, device=dev, dtype=torch.long)
        first.scatter_reduce_(0, inv, ar, reduce="amin", include_self=True)
        rep[b] = first[inv]
    return mult, rep, rep == ar[None]


# --------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------
@dataclass
class DFSResult:
    tau: torch.Tensor                 # (B, N) per-token degrees of freedom
    total: torch.Tensor               # (B,)   DFS of the whole set
    by_modality: dict[str, torch.Tensor]
    ell_h: torch.Tensor
    ell_z: torch.Tensor
    multiplicity: torch.Tensor
    # localisation diagnostic: the mean kernel value to the FARTHEST neighbour
    # still inside each token's neighbourhood.  ~0 means the neighbourhood
    # covers all the correlation there is and tau is the exact ridge leverage;
    # a large value means correlation is being truncated and DFS is
    # OVERestimated (each token cannot see how redundant it really is).
    neighbour_cut: float = 0.0

    @property
    def localisation_is_tight(self) -> bool:
        return self.neighbour_cut < 0.05


def _sigma_of(tokens: TokenBatch) -> torch.Tensor:
    """(B,N) observation-error std, filling per-modality defaults.

    ``reliability`` in [0,1] (a QC flag / quality score) inflates the error:
    sigma_eff = sigma / max(reliability, 0.05).  A flagged-bad observation is
    therefore not discarded but *down-weighted* — it still occupies its place
    in the redundancy structure, which is what makes it stop crowding out a
    good neighbour.
    """
    dt = tokens.emb.dtype
    dev = tokens.emb.device
    default = torch.zeros(len(MODALITIES) + 1, device=dev, dtype=dt)
    for name, mid in MODALITIES.items():
        default[mid] = DEFAULT_SIGMA[name]
    fill = default[tokens.modality.clamp(min=0)]
    s = fill if tokens.sigma is None else torch.where(
        tokens.sigma.to(dt) > 0, tokens.sigma.to(dt), fill)
    if tokens.reliability is not None:
        r = tokens.reliability.to(dt)
        s = s / torch.where(r > 0, r, torch.ones_like(r)).clamp(min=0.05)
    return s


def _xyz_km(coord: torch.Tensor) -> torch.Tensor:
    """(B,N,4) -> (B,N,3) earth-centred coordinates in km (chordal metric)."""
    lat = torch.deg2rad(coord[..., 0])
    lon = torch.deg2rad(coord[..., 1])
    cl = torch.cos(lat)
    return torch.stack([R_EARTH_KM * cl * torch.cos(lon),
                        R_EARTH_KM * cl * torch.sin(lon),
                        R_EARTH_KM * torch.sin(lat)], dim=-1)


def _knn(u: torch.Tensor, valid: torch.Tensor, k: int,
         chunk: int = 2048) -> torch.Tensor:
    """(B,N,F) scaled coords -> (B,N,k) indices of the k nearest tokens.

    Index 0 of every row is the token itself, so the local system can be solved
    against e_0.  Invalid tokens are pushed to infinite distance (they are still
    returned as padding when fewer than k valid tokens exist, and are then
    neutralised by the caller).
    """
    B, N, _ = u.shape
    k = min(k, N)
    out = torch.zeros(B, N, k, dtype=torch.long, device=u.device)
    big = torch.finfo(u.dtype).max / 4
    for b in range(B):
        for i0 in range(0, N, chunk):
            i1 = min(i0 + chunk, N)
            d = torch.cdist(u[b, i0:i1], u[b])                  # (c, N)
            d = d.masked_fill(~valid[b][None, :], big)
            rows = torch.arange(i0, i1, device=u.device)
            d[torch.arange(i1 - i0, device=u.device), rows] = -1.0   # self first
            out[b, i0:i1] = d.topk(k, dim=-1, largest=False).indices
    return out


def dfs_scores(tokens: TokenBatch, target: TargetScale = PROTOCOL_SCALE,
               scales: SupportScales | None = None, k_neighbors: int | None = 32,
               s_cross: float = S_CROSS, evidence_mask: torch.Tensor | None = None,
               chunk: int = 4096) -> DFSResult:
    """Effective evidence tau_i (degrees of freedom for signal) per token.

    ``evidence_mask`` (B,N) restricts the estimate to the observation tokens
    (the climatological background is not evidence — it is the prior the
    evidence is measured against); tokens outside it get tau = 0.

    ``k_neighbors`` is the localisation radius: each token's leverage is solved
    against its k nearest tokens instead of the whole set.  ``None`` solves
    exactly (O(N^3) — use it for probes and small sets).  The approximation is
    tight exactly when the neighbourhood covers all the correlation there is;
    the returned ``neighbour_cut`` reports whether it did, and truncation
    always biases DFS *upward* (a token that cannot see its redundant partners
    believes it is informative).  On the protocol_v1 observing geometry
    (1500-3000 profiles, ~500 km apart, l_h ~ 190 km) k = 32 is converged:
    raising it to 256 moves the total by < 0.05 %.
    """
    dev, dt = tokens.emb.device, tokens.emb.dtype
    B, N = tokens.mask.shape
    scales = scales if scales is not None else SupportScales()
    valid = tokens.mask.clone()
    if evidence_mask is not None:
        valid = valid & evidence_mask
    zeros = torch.zeros(B, N, device=dev, dtype=dt)
    if N == 0 or not bool(valid.any()):
        return DFSResult(zeros, zeros.sum(-1), {}, zeros, zeros,
                         torch.ones_like(zeros))
    if k_neighbors is None:            # exact: every token sees the whole set
        k_neighbors = N
    # the local systems are (chunk, k, k) float64; keep that under ~64 MB so a
    # large k (exact mode on a small set) cannot blow up memory
    chunk = max(1, min(chunk, int(8e6 // max(k_neighbors ** 2, 1))))

    depth = tokens.coord[..., 2]
    strat = (torch.zeros_like(depth) if tokens.strat is None
             else tokens.strat.to(dt))
    lh, lz, lt = scales(depth, strat, target)
    pos = _xyz_km(tokens.coord)                                  # (B,N,3)
    tt = (torch.zeros_like(depth) if tokens.time_offset is None
          else tokens.time_offset.to(dt))
    src = (torch.zeros_like(tokens.modality) if tokens.source_id is None
           else tokens.source_id)
    # exact duplicates are consolidated: only one member of each record group
    # enters the linear system, and its degree of freedom is split evenly
    mult, rep_index, is_rep = record_groups(tokens, valid)
    solve_set = valid & is_rep
    sig = _sigma_of(tokens)

    # ---- neighbourhood search in a globally scaled metric ------------------
    def _mean(x):
        w = solve_set.to(dt)
        return (x * w).sum() / w.sum().clamp(min=1.0)
    u = torch.cat([pos / _mean(lh).clamp(min=_EPS),
                   (depth / _mean(lz).clamp(min=_EPS)).unsqueeze(-1),
                   (tt / _mean(lt).clamp(min=_EPS)).unsqueeze(-1)], dim=-1)
    idx = _knn(u, solve_set, k_neighbors)                        # (B,N,k)
    k = idx.shape[-1]

    g = lambda x: torch.gather(x, 1, idx.reshape(B, -1)).reshape(B, N, k)
    g3 = lambda x: torch.gather(x, 1, idx.reshape(B, -1, 1).expand(-1, -1, 3)
                                ).reshape(B, N, k, 3)
    n_pos, n_dep, n_tt = g3(pos), g(depth), g(tt)
    n_lh, n_lz, n_lt = g(lh), g(lz), g(lt)
    n_sig = g(sig)
    gi = lambda x: torch.gather(x, 1, idx.reshape(B, -1)).reshape(B, N, k)
    n_src = gi(src)
    rid = (torch.full_like(tokens.modality, -1) if tokens.record_id is None
           else tokens.record_id)
    n_rid, n_mod = gi(rid), gi(tokens.modality)
    # when fewer than k tokens are in the solve set, topk pads with excluded
    # ones; they are neutralised below (zero kernel row/col -> identity block)
    n_val = gi(solve_set)

    # The local system is solved in float64: a redundant neighbourhood makes
    # K near-singular, and the whole point is to resolve exactly how singular.
    sdt = torch.float64
    eye = torch.eye(k, device=dev, dtype=sdt)
    tau = torch.zeros(B, N, device=dev, dtype=dt)
    cut_num = cut_den = 0.0
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        sl = slice(i0, i1)

        def gibbs(delta, li, lj):
            s2 = (li ** 2 + lj ** 2).clamp(min=_EPS)
            amp = torch.sqrt((2.0 * li * lj / s2).clamp(min=0.0))
            return amp * torch.exp(-(delta ** 2) / s2)

        p = lambda x: x[:, sl].to(sdt)
        d_h = (p(n_pos)[:, :, :, None, :] - p(n_pos)[:, :, None, :, :]).norm(dim=-1)
        kh = gibbs(d_h, p(n_lh)[..., :, None], p(n_lh)[..., None, :])
        kz = gibbs(p(n_dep)[..., :, None] - p(n_dep)[..., None, :],
                   p(n_lz)[..., :, None], p(n_lz)[..., None, :])
        kt = gibbs(p(n_tt)[..., :, None] - p(n_tt)[..., None, :],
                   p(n_lt)[..., :, None], p(n_lt)[..., None, :])
        ks = torch.where(n_src[:, sl, :, None] == n_src[:, sl, None, :],
                         torch.ones((), device=dev, dtype=sdt),
                         torch.full((), s_cross, device=dev, dtype=sdt))
        K = kh * kz * kt * ks                                    # (B,c,k,k)
        # Two tokens standing for the SAME raw measurement are perfectly
        # correlated by definition, whatever stream they arrived on: this is
        # what makes a real-time / delayed-mode pair merge exactly rather than
        # merely mostly (it also overrides k_s < 1 across the two streams).
        same_rec = ((n_rid[:, sl, :, None] == n_rid[:, sl, None, :])
                    & (n_mod[:, sl, :, None] == n_mod[:, sl, None, :])
                    & (n_rid[:, sl, :, None] >= 0))
        K = torch.where(same_rec, torch.ones((), device=dev, dtype=sdt), K)

        ok = n_val[:, sl].to(sdt)
        K = K * ok[..., :, None] * ok[..., None, :]
        s2 = (p(n_sig) ** 2) * ok + (1.0 - ok)                   # dummy -> 1
        A = K + torch.diag_embed(s2) + eye * 1e-12
        e0 = torch.zeros_like(s2)
        e0[..., 0] = 1.0
        x = torch.linalg.solve(A, e0.unsqueeze(-1)).squeeze(-1)  # (B,c,k)
        tau[:, sl] = (1.0 - s2[..., 0] * x[..., 0]).clamp(0.0, 1.0).to(dt)
        # how much correlation is still present at the edge of the neighbourhood
        own = solve_set[:, sl].to(sdt)
        cut_num += float((K[..., 0, -1].abs() * own).sum())
        cut_den += float(own.sum())

    # every member of a record group reports its equal share of the group's
    # single degree of freedom -> total DFS is invariant to duplication
    tau = torch.gather(tau, 1, rep_index) / mult.clamp(min=1.0)
    tau = tau * valid.to(dt)
    by_mod = {}
    for name, mid in MODALITIES.items():
        sel = (tokens.modality == mid) & valid
        by_mod[name] = (tau * sel.to(dt)).sum(dim=1)
    return DFSResult(tau=tau, total=tau.sum(dim=1), by_modality=by_mod,
                     ell_h=lh, ell_z=lz, multiplicity=mult,
                     neighbour_cut=cut_num / max(cut_den, 1.0))


# --------------------------------------------------------------------------
# Conservative evidence transport through the resampler
# --------------------------------------------------------------------------
class EvidenceResampler(nn.Module):
    """Compress N observation tokens to ``k_slots`` while conserving evidence.

    The failure of MBCA that motivated this module: after a modality-specific
    resampler, the output tokens were given *equal* outgoing weight, so the
    evidence estimated before resampling was silently discarded.  Here the
    resampler is a transport plan.  Each token distributes its whole evidence
    ``tau_i`` over the slots (a softmax over SLOTS, not over tokens — the
    Slot-Attention normalisation), so

        A_is >= 0 ,  sum_s A_is = 1 ,  nu_s = sum_i A_is tau_i
        =>  sum_s nu_s = sum_i tau_i        (exactly, to float precision)

    and the slot content is the evidence-weighted mean of the tokens assigned
    to it, so a slot's value and its outgoing mass describe the same evidence.
    Empty slots get zero mass and are masked out downstream.
    """

    def __init__(self, d_model: int, k_slots: int = 32, n_heads: int = 4,
                 per_modality: bool = True):
        super().__init__()
        self.k = k_slots
        self.h = n_heads
        self.dh = d_model // n_heads
        assert d_model % n_heads == 0
        self.per_modality = per_modality
        n_groups = len(MODALITIES) if per_modality else 1
        self.slots = nn.Parameter(torch.randn(n_groups, k_slots, d_model) * 0.02)
        self.ln_kv = nn.LayerNorm(d_model)
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.n_groups = n_groups

    def forward(self, emb, mask, tau, modality):
        """-> (out (B, G*k, d), out_mask (B, G*k), nu (B, G*k))."""
        B = emb.shape[0]
        kv = self.ln_kv(emb)
        kh = self.wk(kv).view(B, -1, self.h, self.dh)
        vh = self.wv(kv).view(B, -1, self.h, self.dh)
        outs, masks, nus = [], [], []
        for gi in range(self.n_groups):
            sel = mask if not self.per_modality else (
                mask & (modality == list(MODALITIES.values())[gi]))
            q = self.wq(self.slots[gi])[None].expand(B, -1, -1)
            q = q.view(B, self.k, self.h, self.dh)
            # logits (B, h, N, k): each TOKEN is a distribution over slots
            lg = torch.einsum("bnhd,bkhd->bhnk", kh, q) / math.sqrt(self.dh)
            lg = lg.masked_fill(~sel[:, None, :, None], -float("inf"))
            A = torch.softmax(lg, dim=-1)
            A = torch.nan_to_num(A, nan=0.0) * sel[:, None, :, None]
            w = A * tau[:, None, :, None]                       # (B,h,N,k)
            nu = w.sum(dim=2).mean(dim=1)                       # (B,k) heads agree
            num = torch.einsum("bhnk,bnhd->bkhd", w, vh)
            den = w.sum(dim=2).clamp(min=_EPS)[..., None]       # (B,h,k,1)
            out = self.wo((num / den.transpose(1, 2)).reshape(B, self.k, -1))
            present = nu > _EPS
            outs.append(out * present[..., None])
            masks.append(present)
            nus.append(nu * present)
        return (torch.cat(outs, dim=1), torch.cat(masks, dim=1),
                torch.cat(nus, dim=1))
