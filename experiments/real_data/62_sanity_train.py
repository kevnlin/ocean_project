"""Overfit sanity checks, one-component ablations, and backbone trials on real Argo.

Default setup: GLOBAL ocean state reconstruction — one domain over the whole
ocean, every profile a month delivers (~7 600, no cap), 20 levels to 985 m,
target = WOA23 monthly anomaly, split in the satellite era (train 2016-2020,
validate 2021, test 2022-23) so profile-only and satellite arms share one
table. `--region gulfstream|npac_gyre` reproduces the committed regional study
(with its own --split-table / --n-profiles).

The 2026-09-17 meeting asked for three things this script does:

* **an overfit test** — can the model drive RMSE below 0.1 on data it is allowed
  to memorise?  If it cannot, no result on held-out floats means anything,
  because the failure is in the pipeline rather than in the ocean.  Three rungs:

      --mode memorise   one fixed month, fixed input/target floats
      --mode copy       targets ARE the input profiles, at their own positions
                        and levels (can information reach a query at all?)
      --mode small      a fixed handful of months, fixed partitions

* **ablations, one component at a time**, each scored on a FIXED held-out set
  (same months, same seeded input draws, same queries for every arm), so two
  arms differ only by the switch named on the command line.

* **training curves** — every step's loss goes to a JSONL history and, with
  ``--wandb``, to Weights & Biases (offline unless WANDB_API_KEY is set).

Metrics are reported in z units AND in degC / PSU (comparable with the target
Prof. Wang set), because the per-level z scale varies by depth — one z unit is
not one number.

  .venv/bin/python experiments/real_data/62_sanity_train.py --mode memorise --steps 3000
  .venv/bin/python experiments/real_data/62_sanity_train.py --mode train --ablation refiner_local
"""
from __future__ import annotations

import argparse, json, os, sys, time, types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch

from ocean_tokenizer import protocol as P
from ocean_tokenizer.argo_obs import ArgoCohort, ArgoNorm, _select_profiles
from ocean_tokenizer.audit_tools import load_cohort
from ocean_tokenizer.fusion import build_fusion_model
from ocean_tokenizer.token_api import ProfileEncoder
from ocean_tokenizer.setconv import SetConvUNet

CH = ("TEMP", "SALT")
UNITS = {"TEMP": "degC", "SALT": "PSU"}
BANDS = (("0-100m", 0.0, 100.0), ("100-300m", 100.0, 300.0),
         ("300-700m", 300.0, 700.0), ("700-1400m", 700.0, 1401.0))

ap = argparse.ArgumentParser()
ap.add_argument("--region", default="global", choices=["global"] + list(P.REGIONS))
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--mode", default="train",
                choices=["train", "memorise", "copy", "small"])
ap.add_argument("--tag", default=None, help="name of this run (dir + W&B)")
ap.add_argument("--split-protocol", default="recent3", choices=sorted(P.SPLIT_PROTOCOLS))
ap.add_argument("--backbone", default="d4rt", choices=["d4rt", "gaot", "lno", "setconv"])
ap.add_argument("--mass-mode", default="dfs", choices=["dfs", "uniform", "count"])
ap.add_argument("--n-profiles", type=int, default=0,
                help="profiles per month (0 = every profile the month has)")
ap.add_argument("--steps", type=int, default=12000)
ap.add_argument("--batch", type=int, default=1, help="months per optimiser step")
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--warmup", type=int, default=300)
ap.add_argument("--weight-decay", type=float, default=0.01)
ap.add_argument("--queries", type=int, default=1024)
ap.add_argument("--val-every", type=int, default=1000)
ap.add_argument("--ckpt-every", type=int, default=0,
                help="also save model_seed<seed>_step<n>.pt every n steps "
                     "(0 = only the final/best state)")
ap.add_argument("--leads", default="0")
ap.add_argument("--d-model", type=int, default=64)
ap.add_argument("--n-latent", type=int, default=32)
ap.add_argument("--n-heads", type=int, default=4)
ap.add_argument("--n-self-blocks", type=int, default=2)
ap.add_argument("--n-dec-blocks", type=int, default=2)
ap.add_argument("--target-dropout", type=float, default=0.2)
ap.add_argument("--overfit-months", type=int, default=8)
# ---- the ablation switches (each is ONE component) ----
ap.add_argument("--ablation", default="none", help=(
    "none | qc | anomaly_exact | level_tokens | refiner_local | refiner_gate1 | "
    "coords_region | all_profiles | batch8 | no_latent | no_target_dropout | "
    "mass_uniform | mass_count  (comma-separate to stack, e.g. for the 'fixed' arm)"))
ap.add_argument("--setconv-grid", default="auto",
                help="SetConv grid ny x nx; auto = 1 deg (180x360) globally, "
                     "0.5 x 0.5 deg on a regional box")
ap.add_argument("--eval-cells", type=int, default=8000,
                help="fixed per-month cap on scored cells for the validation "
                     "and train-subset sets (the test set is always scored in "
                     "full); a global month has ~30 000 held-out cells")
ap.add_argument("--surface", action="store_true",
                help="add satellite SST/SLA/SSS — the information a profiles-only "
                     "OI does not have. Token backbones (d4rt/lno) get them as "
                     "gridded patch tokens through the surf/ssh encoders; the "
                     "satellite store covers 2016-2023 only")
ap.add_argument("--surface-vars", default="SST,SLA,SSS",
                help="which surface fields to use with --surface. SLA alone is "
                     "the clean control: altimetry assimilates no in-situ "
                     "profile, while the L4 SST/SSS analyses do")
ap.add_argument("--split-table",
                default='{"train":[2016,2020],"validation":[2021,2021],"development":[2022,2023]}',
                help='JSON override of the year splits, e.g. \'{"train":[2016,2020],'
                     '"validation":[2021,2021],"development":[2022,2023]}\'')
ap.add_argument("--setconv-width", type=int, default=28)
ap.add_argument("--n-slots", type=int, default=32,
                help="LNO latent slots (32 = the Perceiver's latent count)")
ap.add_argument("--proj-hidden", type=int, default=96,
                help="LNO position-MLP width (96 at 32 slots and 48 at 128 slots "
                     "keep the model within 4 %% of the Perceiver's size)")
ap.add_argument("--surface-patch", type=int, default=3,
                help="satellite patch size in grid cells for the token backbones "
                     "(3 = 3 x 3 deg; the encoder default of 10 x 12 deg would "
                     "average away the mesoscale signal altimetry carries)")
ap.add_argument("--wandb", action="store_true")
ap.add_argument("--wandb-project", default="ocean-audit")
ap.add_argument("--eval-split", default="development")
ap.add_argument("--out-root", default=None)
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ABL = {a for a in args.ablation.split(",") if a and a != "none"}
KNOWN = {"qc", "anomaly_exact", "level_tokens", "refiner_local", "refiner_gate1", "cap1000",
         "coords_region", "all_profiles", "batch8", "no_latent",
         "no_target_dropout", "mass_uniform", "mass_count",
         # one switch per architectural contribution, for the 20 k study
         "no_refiner",      # drop the query-local refiner branch entirely
         "no_refslots",     # drop the availability-conditioned reference slots
         "no_experts",      # both channels from the shared head, no salt expert
         "refiner_no_mass", # refiner scores ignore the DFS evidence (beta = 0)
         "loss_balanced"}   # equal loss weight per channel and per level
if ABL - KNOWN:
    raise SystemExit(f"unknown ablation(s): {sorted(ABL - KNOWN)}")
if "mass_uniform" in ABL: args.mass_mode = "uniform"
if "mass_count" in ABL: args.mass_mode = "count"
if "batch8" in ABL: args.batch = 8
if "no_target_dropout" in ABL: args.target_dropout = 0.0
if "all_profiles" in ABL: args.n_profiles = 0            # 0 = every profile
if "cap1000" in ABL: args.n_profiles = 1000
TAG = args.tag or f"{args.mode}_{args.backbone}_{args.mass_mode}" \
                  f"{'_' + '_'.join(sorted(ABL)) if ABL else ''}"
OUT = args.out_root or os.path.join(ROOT, "outputs", "audit", args.region, TAG)
os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
LEADS = [int(x) for x in args.leads.split(",")]
t0 = time.time()

# ------------------------------------------------------------------ data
SPLITS = P.SPLIT_PROTOCOLS[args.split_protocol]
if args.split_table:
    SPLITS = {k: tuple(v) for k, v in json.loads(args.split_table).items()}
c, qc_report = load_cohort(ROOT, args.region, SPLITS,
                           anomaly="exact" if "anomaly_exact" in ABL else "cell",
                           qc="qc" in ABL)
norm = ArgoNorm.fit(c, "train")
LEV = c.levels
#: the global run is one domain over the whole ocean; the regional boxes stay
#: available for comparison with the committed regional study
BOX = (dict(lat=(-90.0, 90.0), lon=(0.0, 360.0)) if args.region == "global"
       else P.REGIONS[args.region])
if args.setconv_grid == "auto":
    args.setconv_grid = "180x360" if args.region == "global" else "50x102"
LA0, LA1 = (float(x) for x in BOX["lat"]); LO0, LO1 = (float(x) for x in BOX["lon"])
STD = {ch: torch.as_tensor(norm.std[ch], dtype=torch.float32, device=dev) for ch in CH}
band_of_level = []
for d in LEV:
    for name, lo, hi in BANDS:
        if (lo < d <= hi) or (d <= LEV.min() and lo <= 0):
            band_of_level.append(name); break
    else:
        band_of_level.append(BANDS[-1][0])
band_of_level = np.array(band_of_level)

#: region-rescaled coordinates: stretch the box to fill the globe so the shared
#: Fourier features (finest wavelength 3.6 deg -> ~0.5 deg here) can resolve
#: structure at the anomaly's own correlation length. The local refiner's
#: length scales are rescaled by the same factors, so THIS switch changes
#: coordinate resolution only and not the refiner's physical reach.
SY = 180.0 / (LA1 - LA0)
SX = 360.0 / (LO1 - LO0)


def xlat(lat):
    return (np.asarray(lat) - LA0) * SY - 90.0 if "coords_region" in ABL else np.asarray(lat)


def xlon(lon):
    return (np.asarray(lon) - LO0) * SX if "coords_region" in ABL else np.asarray(lon)


SURF = None
SURF_Z: dict = {}          # month -> {var: (ny, nx) z-scored field, NaN kept}
if args.surface:
    import xarray as _xr
    _ny, _nx = (int(v) for v in args.setconv_grid.split("x"))
    _glat = LA0 + (np.arange(_ny) + 0.5) * (LA1 - LA0) / _ny
    _glon = LO0 + (np.arange(_nx) + 0.5) * (LO1 - LO0) / _nx
    _o = _xr.open_zarr(os.path.join(ROOT, "data", "real_obs_1deg.zarr"))
    _o = _o.assign_coords(lon=(_o.lon % 360.0)).sortby("lon")
    _t = _o.time.values.astype("datetime64[M]")
    _mi = ((_t - np.datetime64("2000-01", "M")) / np.timedelta64(1, "M")).astype(int)
    SURF = {}
    _VARS = [v for v in args.surface_vars.split(",") if v]
    for _v in _VARS:
        _a = _o[_v].interp(lat=_glat, lon=_glon, method="linear").values  # (T,ny,nx)
        for _i, _m in enumerate(_mi):
            SURF.setdefault(int(_m), []).append(_a[_i])
    # channel order SST, SLA, SSS + one shared validity mask; z-scored on the
    # TRAIN years of this split so the scaling cannot see the test era
    _tr = [m for m, _ in SURF.items()
           if SPLITS["train"][0] <= 2000 + m // 12 <= SPLITS["train"][1]]
    _stack = np.stack([np.stack(SURF[m]) for m in _tr])                  # (M,3,ny,nx)
    _mu = np.nanmean(_stack, axis=(0, 2, 3))[:, None, None]
    _sd = np.nanstd(_stack, axis=(0, 2, 3))[:, None, None]
    _sd = np.where(np.isfinite(_sd) & (_sd > 1e-6), _sd, 1.0)
    for _m, _ch in list(SURF.items()):
        _x = (np.stack(_ch) - _mu) / _sd
        _ok = np.isfinite(_x).all(0, keepdims=True).astype("float32")
        SURF_Z[_m] = {v: torch.as_tensor(_x[k], dtype=torch.float32, device=dev)
                      for k, v in enumerate(_VARS)}
        SURF[_m] = torch.as_tensor(np.concatenate([np.nan_to_num(_x), _ok]),
                                   dtype=torch.float32, device=dev)
    GRID_LAT = torch.as_tensor(_glat, dtype=torch.float32, device=dev)
    GRID_LON = torch.as_tensor(_glon, dtype=torch.float32, device=dev)
    print(f"  surface channels: {len(SURF)} months "
          f"({2000 + min(SURF) // 12}-{2000 + max(SURF) // 12}), "
          f"{len(_VARS)} fields {_VARS} + validity", flush=True)


def eligible(split, lead=0):
    return np.array([int(m) for m in c.months_in(split)
                     if c.month(m, float_split="cohort_float").size
                     and c.month(m + lead, float_split="heldout_float").size
                     and (SURF is None or int(m) in SURF)], int)


TRAIN_MONTHS = {int(x) for x in c.months_in("train")}
ELIG = {k: eligible(k) for k in ("train", "validation", args.eval_split)}


def pick_inputs(month, seed, lead=0):
    pool = c.month(month, float_split="cohort_float")
    if args.n_profiles <= 0:
        return pool
    return _select_profiles(c, pool, args.n_profiles,
                            np.random.default_rng([seed, month, lead]))


def make_sample(t_src, lead, src_rows, tgt_rows, n_queries, rng=None,
                target_dropout=0.0):
    """Input profiles + queries at the target profiles' own positions/levels."""
    if src_rows.size == 0 or tgt_rows.size == 0:
        return None
    prof = np.stack([norm.z("TEMP", c.TEMP[src_rows]),
                     norm.z("SALT", c.SALT[src_rows])], axis=1)      # (K,2,L)
    R = tgt_rows.size
    tz = np.stack([norm.z("TEMP", c.TEMP[tgt_rows]),
                   norm.z("SALT", c.SALT[tgt_rows])], axis=-1)       # (R,L,2)
    lat = np.repeat(c.lat[tgt_rows], LEV.size)
    lon = np.repeat(c.lon[tgt_rows], LEV.size)
    depth = np.tile(LEV, R)
    li = np.tile(np.arange(LEV.size), R)
    target = tz.reshape(R * LEV.size, 2)
    tmask = np.isfinite(target)
    if target_dropout > 0 and rng is not None and rng.random() < target_dropout:
        tmask[:, rng.integers(0, 2)] = False
    if n_queries and target.shape[0] > n_queries:
        pick = (rng or np.random.default_rng(0)).choice(
            target.shape[0], n_queries, replace=False)
        lat, lon, depth, li = lat[pick], lon[pick], depth[pick], li[pick]
        target, tmask = target[pick], tmask[pick]
    cm = float(int(t_src + lead) % 12 + 1)
    t = lambda a, d=torch.float32: torch.as_tensor(np.asarray(a), dtype=d, device=dev)
    return dict(
        prof=t(prof), lat=t(xlat(c.lat[src_rows])), lon=t(xlon(c.lon[src_rows])),
        lat_true=t(c.lat[src_rows]), lon_true=t(c.lon[src_rows]),
        month=t([float(int(t_src) % 12 + 1)], torch.long),
        query=t(np.stack([xlat(lat), xlon(lon), depth,
                          np.full(lat.size, cm)], -1)),
        query_true=t(np.stack([lat, lon, depth, np.full(lat.size, cm)], -1)),
        level_index=t(li, torch.long),
        target=t(np.nan_to_num(target)), target_mask=t(tmask, torch.bool),
        surface=(None if SURF is None else SURF.get(int(t_src))),
        surface_z=SURF_Z.get(int(t_src)),
        wmo=np.repeat(c.wmo[tgt_rows], LEV.size)[:target.shape[0]] if not n_queries
        else None, lead=int(lead), n_src=int(src_rows.size))


# ------------------------------------------------------------------ model
class Row(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        if args.backbone == "setconv":
            ny, nx = (int(v) for v in args.setconv_grid.split("x"))
            self.net = SetConvUNet(BOX, LEV, ny=ny, nx=nx,
                                   width=args.setconv_width,
                                   max_lead=max(LEADS + [1]),
                                   n_surface=(len([v for v in args.surface_vars.split(",") if v]) + 1
                                              if args.surface else 0))
            self.kind = "setconv"
            return
        self.kind = "fusion"
        variant = args.backbone + ("" if args.mass_mode == "dfs"
                                   else f"_{args.mass_mode}")

        class _Grid:
            depth = LEV
        self.net = build_fusion_model(
            variant, _Grid(), d_model=args.d_model, n_latent=args.n_latent,
            n_heads=args.n_heads, n_self_blocks=args.n_self_blocks,
            seed=args.seed, n_dec_blocks=args.n_dec_blocks,
            max_lead=max(LEADS + [1]), query_chunk=2048,
            with_ssh=bool(args.surface and "SLA" in args.surface_vars),
            patch_surf=((args.surface_patch, args.surface_patch)
                        if args.surface else None),
            **({"anchor_box": BOX} if args.backbone == "gaot" else {}),
            **({"n_slots": args.n_slots, "proj_hidden": args.proj_hidden}
               if args.backbone == "lno" else {}))
        if "level_tokens" in ABL:
            # one token per level instead of five depth-band tokens: band edges
            # at the midpoints between levels, so a token's own depth is its
            # level and nothing is pooled across levels
            edges = np.r_[LEV[0] - (LEV[1] - LEV[0]) / 2,
                          (LEV[1:] + LEV[:-1]) / 2,
                          LEV[-1] + (LEV[-1] - LEV[-2]) / 2]
            bands = [(float(edges[i]), float(edges[i + 1])) for i in range(LEV.size)]
            self.net.encoders["profiles"] = ProfileEncoder(
                LEV, c_vars=2, d_model=args.d_model, depth_bands=bands).to(dev)
        ref = [self.net.qdec.experts.shared_refiner, self.net.qdec.experts.salt_refiner]
        with torch.no_grad():
            if "refiner_local" in ABL:
                # 150 km horizontally, 100 m vertically, 1 month: the scales the
                # measured correlation actually has. The registered init is
                # (0.35, 0.35, 0.30, 3.0) on axes of lat/90 and lon/180, i.e.
                # 3500 km x 5600 km — wider than the region box.
                ell = torch.tensor([150.0 / (180.0 * 111.195 * np.cos(np.deg2rad((LA0 + LA1) / 2))),
                                    150.0 / (90.0 * 111.195), 100.0 / 1000.0, 1.0])
                if "coords_region" in ABL:
                    ell = ell * torch.tensor([SX, SY, 1.0, 1.0])
                for r in ref:
                    r.log_scale.copy_(torch.log(ell))
            elif "coords_region" in ABL:
                # keep the refiner's PHYSICAL reach fixed under the coordinate
                # rescaling, so this arm isolates coordinate resolution
                for r in ref:
                    r.log_scale.copy_(r.log_scale
                                      + torch.log(torch.tensor([SX, SY, 1.0, 1.0])))
            if "refiner_gate1" in ABL:
                for r in ref:
                    r.gate.fill_(1.0)
            if "no_refiner" in ABL:
                # the branch stays in the graph but contributes nothing and
                # cannot learn its way back: q + 0 * refiner(q)
                for r in ref:
                    r.gate.zero_(); r.gate.requires_grad_(False)
            if "refiner_no_mass" in ABL:
                # scores keep content + distance, lose beta_head * log tau
                for r in ref:
                    r.beta_head.zero_(); r.beta_head.requires_grad_(False)
        if "no_refslots" in ABL:
            # no availability-conditioned reference slots in the fusion kv
            self.net._extra_kv = lambda tokens: None
        if "no_experts" in ABL:
            # shared head emits both channels; the salt refiner/head go unused
            exp = self.net.qdec.experts
            def _shared_only(self_, q, query_coord, lead, emb, coord, tau,
                             time_offset, mask):
                h = self_.shared_refiner(q, query_coord, lead, emb=emb, coord=coord,
                                         tau=tau, time_offset=time_offset, mask=mask)
                return self_.shared_head(h)
            exp.forward = types.MethodType(_shared_only, exp)

    def forward(self, s):
        if self.kind == "setconv":
            return self.net(s["prof"], s["lat_true"], s["lon_true"], s["month"],
                            s["query_true"], lead=s["lead"],
                            surface=s.get("surface"))
        obs = {"profiles": dict(prof=s["prof"][None], lat=s["lat"][None],
                                lon=s["lon"][None], month=s["month"])}
        sz = s.get("surface_z")
        if args.surface and sz:
            # satellite fields as gridded patch tokens through the shared
            # surface / SSH encoders (the streams the global figure model uses)
            g = dict(lat=GRID_LAT, lon=GRID_LON, month=s["month"])
            if "SST" in sz and "SSS" in sz:
                obs["surf"] = dict(field=torch.stack([sz["SST"], sz["SSS"]])[None], **g)
            if "SLA" in sz:
                obs["ssh"] = dict(field=sz["SLA"][None, None], **g)
        q = s["query"][None]
        lead = torch.full(q.shape[:2], s["lead"], dtype=torch.long, device=dev)
        tokens = self.net.encode(obs, batch=1, device=dev)
        latent = self.net.fuse(tokens)
        if "no_latent" in ABL:
            latent = torch.zeros_like(latent)
        return self.net.decode(latent, q, None, lead)[0]


# ------------------------------------------------------------------ scoring
def score(model, samples):
    """Pooled RMSE in z and physical units, plus per-band z RMSE and J."""
    model.eval()
    se = {ch: 0.0 for ch in CH}; n = {ch: 0 for ch in CH}
    sep = {ch: 0.0 for ch in CH}; se0 = {ch: 0.0 for ch in CH}
    band = {ch: {b: [0.0, 0] for b, _, _ in BANDS} for ch in CH}
    with torch.no_grad():
        for s in samples:
            pred = model(s)
            for j, ch in enumerate(CH):
                m = s["target_mask"][:, j]
                if not bool(m.any()):
                    continue
                e = ((pred[:, j] - s["target"][:, j]) ** 2)[m]
                sd = STD[ch][s["level_index"]][m]
                se[ch] += float(e.sum()); n[ch] += int(m.sum())
                sep[ch] += float((e * sd ** 2).sum())
                se0[ch] += float((s["target"][:, j][m] ** 2).sum())
                bl = band_of_level[s["level_index"][m].cpu().numpy()]
                ev = e.cpu().numpy()
                for b, _, _ in BANDS:
                    k = bl == b
                    if k.any():
                        band[ch][b][0] += float(ev[k].sum()); band[ch][b][1] += int(k.sum())
    model.train()
    out = {}
    for ch in CH:
        if n[ch] == 0:
            continue
        rz = float(np.sqrt(se[ch] / n[ch])); r0 = float(np.sqrt(se0[ch] / n[ch]))
        out[ch] = {"rmse_z": rz, "rmse_physical": float(np.sqrt(sep[ch] / n[ch])),
                   "unit": UNITS[ch], "J": rz / max(r0, 1e-9),
                   "climatology_z": r0, "n": n[ch],
                   "by_band_z": {b: (float(np.sqrt(v[0] / v[1])) if v[1] else float("nan"))
                                 for b, v in band[ch].items()}}
    out["macro_z"] = float(np.mean([out[ch]["rmse_z"] for ch in CH if ch in out]))
    return out


def build_eval(split, n_months=None, seed=None, max_cells=0):
    """A FIXED evaluation set: same months, same input draws, same cells.

    ``max_cells`` caps the scored cells per month with a generator seeded by the
    month alone, so every arm and every seed is scored on the identical subset.
    """
    seed = args.seed if seed is None else seed
    months = ELIG[split]
    if n_months and months.size > n_months:
        months = months[np.linspace(0, months.size - 1, n_months).astype(int)]
    out = []
    for m in months:
        src = pick_inputs(int(m), seed)
        tgt = c.month(int(m), float_split="heldout_float")
        s = make_sample(int(m), 0, src, tgt, max_cells,
                        rng=np.random.default_rng([20260918, int(m)]))
        if s is not None:
            out.append(s)
    return out


QUERY_KEYS = ("query", "query_true", "level_index", "target", "target_mask")


def subsample(s, n, rng):
    """Train on n of a fixed sample's cells per step (all are scored).

    The overfit rungs fix up to --eval-cells cells per month; stepping on all of
    them would cost 8x the queries of a normal step. Drawing n of the SAME fixed
    cells each step is still memorisation of that set, at normal step cost.
    """
    Q = s["target"].shape[0]
    if not n or Q <= n:
        return s
    pick = torch.as_tensor(rng.choice(Q, n, replace=False), device=s["target"].device)
    out = dict(s)
    for k in QUERY_KEYS:
        out[k] = s[k][pick]
    return out


# ------------------------------------------------------------------ logging
hist_path = os.path.join(OUT, "history.jsonl")
hist = open(hist_path, "a")
wb = None
if args.wandb:
    import wandb
    os.environ.setdefault("WANDB_MODE", "offline" if not os.environ.get("WANDB_API_KEY")
                          else "online")
    os.environ.setdefault("WANDB_DIR", os.path.join(ROOT, "outputs", "wandb"))
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)
    wb = wandb.init(project=args.wandb_project, name=f"{args.region}/{TAG}/s{args.seed}",
                    config=vars(args) | {"ablations": sorted(ABL)}, reinit=True)


def log(step, **kw):
    # the seed travels with every record: several seeds of one arm append to the
    # same history file, and a plot that joins them draws a zigzag
    rec = {"step": step, "seed": args.seed, **kw}
    hist.write(json.dumps(rec, default=float) + "\n"); hist.flush()
    if wb is not None:
        wb.log(kw, step=step)


# ------------------------------------------------------------------ run
model = Row().to(dev)
n_par = sum(p.numel() for p in model.net.parameters())
print(f"{TAG}  region={args.region} seed={args.seed} backbone={args.backbone} "
      f"mass={args.mass_mode} ablations={sorted(ABL) or ['none']}\n"
      f"  params={n_par:,} device={dev} months train={ELIG['train'].size} "
      f"val={ELIG['validation'].size} {args.eval_split}={ELIG[args.eval_split].size}",
      flush=True)
if qc_report is not None:
    print(f"  QC: flagged {qc_report.n_values_flagged} values, dropped "
          f"{qc_report.n_profiles_dropped} profile-channels", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda i: (min(1.0, (i + 1) / max(args.warmup, 1))
                    * (0.5 * (1 + np.cos(np.pi * min(1.0, i / max(args.steps, 1))))) ** 0.5))
rng = np.random.default_rng(args.seed)

# ---- the fixed training material ----
if args.mode in ("memorise", "copy", "small"):
    k = 1 if args.mode in ("memorise", "copy") else args.overfit_months
    months = rng.choice(ELIG["train"], k, replace=False)
    fixed = []
    for m in months:
        pool = c.month(int(m), float_split="cohort_float")
        floats = np.unique(c.wmo[pool])
        if args.mode == "copy":
            src = (pool if args.n_profiles <= 0 else _select_profiles(
                c, pool, args.n_profiles, np.random.default_rng([args.seed, int(m)])))
            # answer what you were shown: targets are input profiles themselves
            # (512 of them, so the query set stays the size of the other rungs)
            tgt = np.sort(np.random.default_rng([args.seed, int(m), 1]).choice(
                src, min(512, src.size), replace=False))
        else:
            tf = set(rng.choice(floats, max(1, int(0.3 * floats.size)),
                                replace=False).tolist())
            src = _select_profiles(c, pool[~np.isin(c.wmo[pool], list(tf))],
                                   args.n_profiles,
                                   np.random.default_rng([args.seed, int(m)]))
            tgt = pool[np.isin(c.wmo[pool], list(tf))]
        s = make_sample(int(m), 0, src, tgt, args.eval_cells,
                        rng=np.random.default_rng([20260918, int(m)]))
        if s is not None:
            fixed.append(s)
    train_set, eval_sets = fixed, {"overfit": fixed}
    during = ["overfit"]
    print(f"  overfit material: {len(fixed)} month(s), "
          f"{sum(int(s['target_mask'].sum()) for s in fixed):,} scored cells, "
          f"{fixed[0]['n_src']} input profiles", flush=True)
else:
    train_set = None
    eval_sets = {"validation": build_eval("validation", max_cells=args.eval_cells),
                 args.eval_split: build_eval(args.eval_split),
                 "train_subset": build_eval("train", n_months=12,
                                            max_cells=args.eval_cells)}
    # only the selection set is scored during training — the held-out and
    # train-subset passes cost more than the training steps between them and
    # are needed once, on the selected weights
    during = ["validation"]
    for k, v in eval_sets.items():
        print(f"  eval[{k}]: {len(v)} months, "
              f"{sum(int(s['target_mask'].sum()) for s in v):,} cells", flush=True)

best = {"macro_z": float("inf"), "step": -1, "state": None}
loss_run = []
for step in range(args.steps):
    opt.zero_grad(set_to_none=True)
    tot = 0.0
    for _ in range(args.batch):
        if train_set is not None:
            s = subsample(train_set[int(rng.integers(len(train_set)))],
                          args.queries, rng)
        else:
            m = int(rng.choice(ELIG["train"]))
            ok = [L for L in LEADS if (m + L) in TRAIN_MONTHS]
            if not ok:
                continue
            lead = int(rng.choice(ok))
            pool = c.month(m, float_split="cohort_float")
            floats = np.unique(c.wmo[pool])
            tgt_pool = c.month(m + lead, float_split="cohort_float")
            if floats.size < 2 or tgt_pool.size == 0:
                continue
            tf = set(rng.choice(floats, max(1, int(round(0.3 * floats.size))),
                                replace=False).tolist())
            src = pool[~np.isin(c.wmo[pool], list(tf))]
            if args.n_profiles > 0 and src.size:
                src = _select_profiles(c, src, args.n_profiles, rng)
            tgt = tgt_pool[np.isin(c.wmo[tgt_pool], list(tf))]
            s = make_sample(m, lead, src, tgt, args.queries, rng,
                            target_dropout=args.target_dropout)
        if s is None:
            continue
        pred = model(s)
        msk = s["target_mask"]
        if not bool(msk.any()):
            continue
        se = ((pred - s["target"]) ** 2) * msk
        if "loss_balanced" in ABL:
            # plain MSE weights a (channel, level) by how many cells it happens
            # to have; this gives every channel and every level equal weight, so
            # the sparse deep levels are not drowned by the well-sampled ones
            li = s["level_index"].long()
            n = torch.zeros(LEV.size, 2, device=se.device).index_add_(
                0, li, msk.float())
            w = msk.float() / n.clamp(min=1.0)[li]
            loss = (se * w).sum() / w.sum()
        else:
            loss = se.sum() / msk.sum()
        (loss / args.batch).backward()
        tot += float(loss) / args.batch
    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    opt.step(); sched.step()
    loss_run.append(tot)
    if (step + 1) % 20 == 0:
        log(step + 1, train_loss=float(np.mean(loss_run[-20:])),
            lr=float(sched.get_last_lr()[0]), grad_norm=gn)
    if args.ckpt_every and (step + 1) % args.ckpt_every == 0:
        torch.save(model.state_dict(),
                   os.path.join(OUT, f"model_seed{args.seed}_step{step + 1}.pt"))
    if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
        sc = {k: score(model, eval_sets[k]) for k in during}
        flat = {f"{k}/{ch}_{m}": sc[k][ch][m]
                for k in sc for ch in CH if ch in sc[k]
                for m in ("rmse_z", "rmse_physical", "J")}
        flat.update({f"{k}/macro_z": sc[k]["macro_z"] for k in sc})
        log(step + 1, train_loss=float(np.mean(loss_run[-args.val_every:])), **flat)
        key = "overfit" if "overfit" in sc else "validation"
        line = "  ".join(
            f"{k}: " + " ".join(f"{ch} {sc[k][ch]['rmse_z']:.3f}z/"
                                f"{sc[k][ch]['rmse_physical']:.3f}{UNITS[ch]}"
                                for ch in CH if ch in sc[k]) for k in sc)
        star = ""
        if sc[key]["macro_z"] < best["macro_z"]:
            best = {"macro_z": sc[key]["macro_z"], "step": step + 1,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "scores": sc}
            star = " *"
        print(f"  step {step+1:6d} loss {np.mean(loss_run[-args.val_every:]):.4f}  "
              f"{line}{star}", flush=True)

if best["state"] is not None:
    model.load_state_dict(best["state"])
final = {k: score(model, v) for k, v in eval_sets.items()}
summary = {"tag": TAG, "region": args.region, "seed": args.seed,
           "surface": (args.surface_vars if args.surface else None),
           "splits": {k: list(v) for k, v in SPLITS.items()},
           "backbone": args.backbone, "mass_mode": args.mass_mode,
           "ablations": sorted(ABL), "mode": args.mode, "params": int(n_par),
           "steps": args.steps, "batch": args.batch, "n_profiles": args.n_profiles,
           "best_step": best["step"], "scores": final,
           "split_protocol": args.split_protocol,
           "runtime_s": time.time() - t0,
           "qc": (None if qc_report is None else
                  {"values_flagged": qc_report.n_values_flagged,
                   "profiles_dropped": qc_report.n_profiles_dropped})}
with open(os.path.join(OUT, f"summary_seed{args.seed}.json"), "w") as f:
    json.dump(summary, f, indent=1, default=float)
torch.save(model.state_dict(), os.path.join(OUT, f"model_seed{args.seed}.pt"))
hist.close()
if wb is not None:
    wb.summary.update({f"final/{k}/{ch}_rmse_z": final[k][ch]["rmse_z"]
                       for k in final for ch in CH if ch in final[k]})
    wb.finish()
print(f"\n{TAG}: " + "  ".join(
    f"{k} " + " ".join(f"{ch} {final[k][ch]['rmse_z']:.4f}z "
                       f"{final[k][ch]['rmse_physical']:.4f}{UNITS[ch]} "
                       f"J {final[k][ch]['J']:.3f}"
                       for ch in CH if ch in final[k]) for k in final)
      + f"\n  {time.time()-t0:.0f}s -> {OUT}")
