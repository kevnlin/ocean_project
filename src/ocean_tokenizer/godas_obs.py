"""GODAS observation contract (mentor doc §2.1).

One sample at the default settings carries **664 tokens**:

===========================  ======================================  =====
stream                       composition                             count
===========================  ======================================  =====
profile points (T/S)         24 columns x 16 depths, at ``t_src``      384
surface patches (T/S)        70 patches x 2 context months             140
SSH patches                  70 patches x 2 context months             140
===========================  ======================================  =====

70 patches is not a free parameter: a 4 x 4 patch tiles the 38 x 26 experiment
grid as 10 x 7 = 70, which is what the doc's number implies.

**Five masks, kept deliberately distinct.**  Conflating any two is how a
padding slot starts contributing evidence, or a withheld target starts being
scored:

``mask``                real token rather than padding
``value_mask``          finite variables *within* that token
``support_mask``        token support eligible for evidence computation
``modality_available``  whole input stream retained or dropped
``target_mask``         finite target supervision, used only by the loss

Missing values are zero-filled **only at the encoder boundary** and always
travel with their mask, so a zero can never be mistaken for a measurement.

Coordinates are ``(x, y, z, t)``: x/y/z normalised to [0, 1] over the box, t in
months relative to ``t_src`` — so every causal token has ``t <= 0``.  Profiles
exist only at ``t_src``; the gridded streams span ``[t_src-1, t_src]``.

Training-time augmentation (§2.1): input modalities are dropped independently
with probability 0.2 retaining at least one, and one complete target channel is
withheld with probability 0.2.  Both are off outside training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .batched_dfs import NOISE_DENSITY_POINT, NOISE_DENSITY_PATCH

N_PROFILE_COLS = 24
PATCH = 4
N_PATCHES = 70                 # 4x4 patches tiling 38 x 26 -> 10 x 7
CONTEXT_MONTHS = 2
MOD_PROFILE, MOD_SURF, MOD_SSH = 0, 1, 2
N_MODALITIES = 3
N_CHANNELS = 2                 # TEMP, SALT

# Which observed VARIABLE each modality carries.  This is the axis that decides
# redundancy, and it is not the modality: profile points and surface patches
# both carry T/S, so co-located ones genuinely ARE redundant and must compete;
# SSH is a different physical quantity and must not be consolidated with them.
VARIABLE_GROUP = {MOD_PROFILE: 0, MOD_SURF: 0, MOD_SSH: 1}
N_VARIABLE_GROUPS = 2


@dataclass
class ObsConfig:
    n_profiles: int = N_PROFILE_COLS
    patch: int = PATCH
    context: int = CONTEXT_MONTHS
    modality_dropout: float = 0.2
    target_dropout: float = 0.2
    train: bool = False
    n_queries: int = 512
    max_lead: int = 3
    force_available: tuple | None = None   # §9 audit: pin the input streams


def _patch_boxes(ny: int, nx: int, p: int):
    """Top-left corners and extents of the patches tiling a (ny, nx) grid."""
    return [(y, x, min(p, ny - y), min(p, nx - x))
            for y in range(0, ny, p) for x in range(0, nx, p)]


def build_sample(fields: dict, t_src: int, cfg: ObsConfig | None = None,
                 rng: np.random.Generator | None = None,
                 lead: int = 0) -> dict:
    """Assemble one training/evaluation sample from z-scored GODAS fields.

    ``fields`` holds z-space ``TEMP``/``SALT`` (T, Z, Y, X) and ``SSH``
    (T, Y, X).  Returns the token arrays plus the five masks.

    ``lead`` is ``t_tgt - t_src`` in whole months.  Observations never move:
    the encoder window still stops at ``t_src``, so a positive lead is a
    genuine causal forecast request and not a relabelling of the inputs.
    """
    if not 0 <= lead <= (cfg.max_lead if cfg else 3):
        raise ValueError(f"lead must be in 0..{cfg.max_lead if cfg else 3}, "
                         f"got {lead}")
    cfg = cfg or ObsConfig()
    rng = rng or np.random.default_rng(0)
    T, Z, Y, X = fields["TEMP"].shape

    # which streams survive (training-time modality dropout, at least one)
    avail = np.ones(N_MODALITIES, dtype=bool)
    if cfg.force_available is not None:
        # the missing-input audit controls availability exactly; it must not
        # be at the mercy of the training-time dropout dice
        avail = np.asarray(cfg.force_available, dtype=bool)
    elif cfg.train and cfg.modality_dropout > 0:
        while True:
            avail = rng.random(N_MODALITIES) >= cfg.modality_dropout
            if avail.any():
                break

    coord, value, vmask, modality, noise = [], [], [], [], []

    # ---- profile point tokens: 24 columns x Z depths, at t_src -----------
    if avail[MOD_PROFILE]:
        ys = rng.integers(0, Y, cfg.n_profiles)
        xs = rng.integers(0, X, cfg.n_profiles)
        zz = np.arange(Z)
        for y, x in zip(ys, xs):
            t = np.stack([fields["TEMP"][t_src, :, y, x],
                          fields["SALT"][t_src, :, y, x]], axis=-1)   # (Z,2)
            coord.append(np.stack([np.full(Z, x / max(X - 1, 1)),
                                   np.full(Z, y / max(Y - 1, 1)),
                                   zz / max(Z - 1, 1),
                                   np.zeros(Z)], axis=-1))
            value.append(t)
            vmask.append(np.isfinite(t))
            modality.append(np.full(Z, MOD_PROFILE))
            noise.append(np.full(Z, NOISE_DENSITY_POINT))

    # ---- gridded patches over [t_src-context+1, t_src] ------------------
    boxes = _patch_boxes(Y, X, cfg.patch)
    for mod, keys in ((MOD_SURF, ("TEMP", "SALT")), (MOD_SSH, ("SSH",))):
        if not avail[mod]:
            continue
        for back in range(cfg.context - 1, -1, -1):
            tt = max(t_src - back, 0)
            for (y0, x0, hy, hx) in boxes:
                vals = []
                for k in keys:
                    a = fields[k]
                    box = (a[tt, 0, y0:y0 + hy, x0:x0 + hx] if a.ndim == 4
                           else a[tt, y0:y0 + hy, x0:x0 + hx])
                    with np.errstate(invalid="ignore"):
                        vals.append(np.nanmean(box) if np.isfinite(box).any()
                                    else np.nan)
                v = np.array(vals + [0.0] * (N_CHANNELS - len(vals)))
                m = np.array([np.isfinite(x) for x in vals]
                             + [False] * (N_CHANNELS - len(vals)))
                coord.append(np.array([[(x0 + hx / 2) / max(X - 1, 1),
                                        (y0 + hy / 2) / max(Y - 1, 1),
                                        0.0, -float(back)]]))
                value.append(v[None]); vmask.append(m[None])
                modality.append(np.array([mod]))
                noise.append(np.array([NOISE_DENSITY_PATCH]))

    if coord:
        coord = np.concatenate(coord); value = np.concatenate(value)
        vmask = np.concatenate(vmask); modality = np.concatenate(modality)
        noise = np.concatenate(noise)
    else:                                   # every stream dropped
        coord = np.zeros((0, 4)); value = np.zeros((0, N_CHANNELS))
        vmask = np.zeros((0, N_CHANNELS), bool); modality = np.zeros(0, int)
        noise = np.zeros(0)

    value = np.where(vmask, np.nan_to_num(value), 0.0)   # zero-fill at the boundary
    tok = torch.as_tensor(vmask.any(axis=-1))            # a token with no finite
    #                                                      value is not a token

    # ---- targets: unobserved cells at t_src ------------------------------
    tgt_mask_channels = np.ones(N_CHANNELS, dtype=bool)
    if cfg.train and cfg.target_dropout > 0 and rng.random() < cfg.target_dropout:
        tgt_mask_channels[rng.integers(0, N_CHANNELS)] = False

    t_tgt = min(t_src + lead, T - 1)
    fin = np.isfinite(fields["TEMP"][t_tgt]) & np.isfinite(fields["SALT"][t_tgt])
    idx = np.flatnonzero(fin.reshape(-1))
    take = rng.choice(idx, size=min(cfg.n_queries, idx.size), replace=False)
    zi, yi, xi = np.unravel_index(take, (Z, Y, X))
    qcoord = np.stack([xi / max(X - 1, 1), yi / max(Y - 1, 1),
                       zi / max(Z - 1, 1),
                       np.full(take.size, float(lead))], axis=-1)
    target = np.stack([fields["TEMP"][t_tgt][zi, yi, xi],
                       fields["SALT"][t_tgt][zi, yi, xi]], axis=-1)
    tmask = np.isfinite(target) & tgt_mask_channels[None, :]

    return dict(
        coord=torch.as_tensor(coord, dtype=torch.float64),
        value=torch.as_tensor(value, dtype=torch.float32),
        value_mask=torch.as_tensor(vmask),
        mask=tok,
        support_mask=tok.clone(),
        modality=torch.as_tensor(modality, dtype=torch.long),
        variable_group=torch.as_tensor(
            np.vectorize(VARIABLE_GROUP.get)(modality).astype("int64")
            if modality.size else np.zeros(0, dtype="int64")),
        modality_available=torch.as_tensor(avail),
        noise_density=torch.as_tensor(noise, dtype=torch.float64),
        query=torch.as_tensor(qcoord, dtype=torch.float64),
        target=torch.as_tensor(np.nan_to_num(target), dtype=torch.float32),
        target_mask=torch.as_tensor(tmask),
        t_src=int(t_src),
        lead=int(lead),
    )


def duplicate_profile_attack(s: dict, k: int, temp_bias: float = 2.0,
                             depths: int = 16) -> dict:
    """§9 duplicate attack: bias the first profile column, then copy it k times.

    The bias makes the attack *observable* — an unbiased duplicate moves
    nothing, so the amplification would be unmeasurable.  The copies are
    bit-exact, which is the point: they carry no new information, so any
    change in the prediction between k=1 and k=8 is the model being fooled by
    multiplicity rather than informed by evidence.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    out = {kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in s.items()}
    out["value"][:depths, 0] = torch.where(
        out["value_mask"][:depths, 0],
        out["value"][:depths, 0] + temp_bias,
        out["value"][:depths, 0])
    if k == 1:
        return out
    per_token = ("coord", "value", "value_mask", "mask", "support_mask",
                 "modality", "variable_group", "noise_density")
    block = {kk: out[kk][:depths] for kk in per_token}
    for kk in per_token:
        out[kk] = torch.cat([out[kk]] + [block[kk]] * (k - 1), dim=0)
    return out
