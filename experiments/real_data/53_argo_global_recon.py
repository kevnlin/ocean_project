"""Global 2-D T/S reconstruction on REAL data — DFS-Attention + Perceiver-IO
latent + D4RT query decoder (`fusion.D4RTFusion`, the model in the CESM2 figure).

This is the real-observation counterpart of `36_d4rt_recon_heatmap.py`. Same
architecture, same 1 deg x 20-level analysis grid, same depth bands — every
input replaced by an observation:

    stream    CESM2 figure                    here
    --------- ------------------------------- --------------------------------
    profiles  random columns of CESM2          real QC'd Argo profiles
    surf      CESM2 SST/SSS                    NOAA OISST + OISSS (satellite)
    woa       WOA23 climatology                WOA23 climatology (unchanged)
    ssh       pseudo-SSH from the model state  DUACS altimetry SLA
    TARGET    CESM2 at every unobserved cell   **held-out Argo floats' own T/S**

Two consequences follow from that last row, and both shape the figure.

**There is no dense truth.** The CESM2 map has an error value in every ocean
cell because the truth is a simulated field. Real error exists only where a
held-out float actually surfaced, so cells are binned and cells nobody visited
stay blank — not zero, not interpolated. `--bin-deg` sets the bin; 1 deg is what
the CESM2 figure uses and is far too fine for float coverage, so the default
coarsens until cells actually fill.

**The climatology is observational.** Targets are anomalies against WOA23 at the
float's own (lat, lon, depth, month), so "predict zero" is exactly "predict
WOA23" — an honest, fully observational floor.

Equator weighting
-----------------
`--eq-sigma` biases the profile draw toward the equator, where the CESM2 map is
brightest, while `--eq-uniform-frac` keeps a fraction drawn uniformly so the
rest of the ocean is still represented. The default is a 15 deg Gaussian with
25 % uniform: most profiles near the equator, some outside.

Period is 2016-2023, bounded by the satellite streams, not by Argo.

  .venv/bin/python experiments/53_argo_global_recon.py --smoke
  .venv/bin/python experiments/53_argo_global_recon.py --steps 12000
"""
from __future__ import annotations

import argparse, json, math, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from ocean_tokenizer import config as C, data as D_
from ocean_tokenizer import protocol as P
from ocean_tokenizer.fusion import build_fusion_model

VARS = ("TEMP", "SALT")
UNITS = {"TEMP": "degC", "SALT": "PSU"}
BANDS = [("0-100 m", 0.0, 100.0), ("100-300 m", 100.0, 300.0),
         ("300-985 m", 300.0, 1000.0)]

ap = argparse.ArgumentParser()
ap.add_argument("--cohort", default=None)
ap.add_argument("--variant", default="d4rt", choices=["d4rt", "dfs"])
ap.add_argument("--train-years", default="2016,2021")
ap.add_argument("--val-years", default="2022,2022")
ap.add_argument("--test-years", default="2023,2023")
ap.add_argument("--n-profiles", type=int, default=1000)
ap.add_argument("--eq-sigma", type=float, default=15.0,
                help="latitude sigma of the equator weighting, degrees")
ap.add_argument("--eq-uniform-frac", type=float, default=0.25,
                help="fraction of the draw taken uniformly instead")
ap.add_argument("--steps", type=int, default=12000)
ap.add_argument("--val-every", type=int, default=1000)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--weight-decay", type=float, default=0.01)
ap.add_argument("--warmup", type=int, default=500)
# Defaults matched to 36_d4rt_recon_heatmap.py, which is what produced the
# 445,703-param CESM2 figure. Same architecture and capacity, so the real-data
# map is readable against it rather than being a different model as well as a
# different dataset.
ap.add_argument("--d-model", type=int, default=64)
ap.add_argument("--n-latent", type=int, default=32)
ap.add_argument("--n-heads", type=int, default=4)
ap.add_argument("--n-self-blocks", type=int, default=2)
ap.add_argument("--n-dec-blocks", type=int, default=2)
ap.add_argument("--anchor-grid", default="12,24",
                help="geographically anchored latents (overrides --n-latent); "
                     "'' disables")
# The query-local refiner materialises a (B, heads, Q, N) relative-position
# bias, so memory is O(Q*N) and the chunk is the only thing bounding it. 8192
# OOMs a 16 GB share at global token counts; 1024 is what the CESM2 run used.
ap.add_argument("--query-chunk", type=int, default=1024)
# A training step keeps every chunk's activations for the backward pass, so
# memory grows with the TOTAL query count, not with the chunk. A month can
# offer >30k held-out values; subsampling per step bounds that. Evaluation runs
# under no_grad and is chunked, so it can use them all.
ap.add_argument("--train-queries", type=int, default=3072)
ap.add_argument("--eval-chunk", type=int, default=4096)
ap.add_argument("--seed", type=int, default=1234)
# 1 deg, matching the CESM2 figure's resolution. Float coverage cannot fill a
# 1 deg grid the way a simulated truth does, so the map is sparser than that
# figure and stays that way -- blank cells are cells no held-out float visited,
# and they are left blank rather than interpolated.
ap.add_argument("--bin-deg", type=float, default=1.0)
ap.add_argument("--en4-max-unc", type=float, default=1.0,
                help="drop EN4 cells whose temperature_uncertainty exceeds this "
                     "(degC). EN4 ships uncertainty rather than an observation "
                     "weight; it is large exactly where the analysis is poorly "
                     "constrained, so it is the usable proxy for down-weighting.")
ap.add_argument("--min-obs", type=int, default=3,
                help="scored values needed before a cell is drawn")
ap.add_argument("--no-ssh", action="store_true")
ap.add_argument("--demean", action="store_true",
                help="score the model-minus-EN4 difference about its own annual "
                     "mean at each cell and level, so a constant year-round "
                     "offset drops out and only the time-varying disagreement "
                     "is mapped")
ap.add_argument("--from-checkpoint", default=None,
                help="load a saved state_dict and skip training (re-plot only)")
ap.add_argument("--tag", default=None)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORTS = os.path.join(ROOT, "reports", "real_data"); os.makedirs(REPORTS, exist_ok=True)
CKPT = os.path.join(ROOT, "outputs", "ckpt"); os.makedirs(CKPT, exist_ok=True)
CACHE = os.path.join(ROOT, "outputs", "cache"); os.makedirs(CACHE, exist_ok=True)
tag = args.tag or f"argo_global_{args.variant}_s{args.seed}"
if args.smoke:
    args.steps, args.val_every, args.n_profiles = 40, 20, 200
dev = "cuda" if torch.cuda.is_available() else "cpu"
yrs = lambda s: tuple(int(x) for x in s.split(","))
TR, VA, TE = yrs(args.train_years), yrs(args.val_years), yrs(args.test_years)
t0 = time.time()
torch.manual_seed(args.seed); np.random.seed(args.seed)

# ===================================================================== grid
grid = D_.CommonGrid()
LAT, LON, DEPTH = grid.lat, grid.lon, grid.depth
H, W, Dn = grid.nlat, grid.nlon, grid.ndepth
print(f"[{tag}] {grid}", flush=True)

# ============================================================ real gridded obs
ro = xr.open_zarr(os.path.join(ROOT, "data", "real_obs_1deg.zarr")).load()
ro_months = ro.time.values.astype("datetime64[M]")
MONTH0 = np.datetime64("2000-01", "M")
ro_mi = (ro_months.astype(int) - MONTH0.astype(int))
print(f"  satellite streams: {len(ro_mi)} months "
      f"{str(ro_months[0])}..{str(ro_months[-1])}", flush=True)

woa = D_.woa_prior(grid)                      # TEMP/SALT (12,D,H,W); SST/SSS (12,H,W)
print(f"  WOA23 prior on the analysis grid: {woa['TEMP'].shape}", flush=True)

# ================================================================ Argo cohort
cpath = args.cohort or os.path.join(ROOT, "data", "argo_cohort", "global_global.nc")
if not os.path.exists(cpath):
    raise SystemExit(f"missing {cpath}\n  build it with:\n"
                     f"  .venv/bin/python experiments/44_build_argo_cohort.py "
                     f"--regions global --grid global --levels protocol "
                     f"--suffix _global")
ds = xr.open_dataset(cpath)
A = {k: np.asarray(ds[k].values) for k in
     ("TEMP", "SALT", "lat", "lon", "grid_y", "grid_x", "month_index", "year")}
A["wmo"] = np.asarray(ds["wmo"].values).astype(str)
A["float_split"] = np.asarray(ds["float_split"].values).astype(str)
ds.close()
assert A["TEMP"].shape[1] == Dn, (
    f"cohort has {A['TEMP'].shape[1]} levels, the analysis grid has {Dn}; "
    f"rebuild the cohort with --levels protocol")

held = A["float_split"] == "heldout_float"
in_split = lambda lo, hi: (A["year"] >= lo) & (A["year"] <= hi)
# only months the satellite streams actually cover: an input stream that is
# silently absent is not an ablation, it is an unrecorded change of model
have_sat = np.isin(A["month_index"], ro_mi)
tr_m = in_split(*TR) & have_sat
va_m = in_split(*VA) & have_sat
te_m = in_split(*TE) & have_sat
print(f"  cohort {A['TEMP'].shape[0]:,} profiles | train {tr_m.sum():,} "
      f"val {va_m.sum():,} test {te_m.sum():,} | held-out floats "
      f"{np.unique(A['wmo'][held]).size:,}", flush=True)

# ------------------------------------------------- anomaly vs WOA23, z-scored
def woa_at(rows: np.ndarray, var: str) -> np.ndarray:
    """WOA23 value at each profile's own (lat, lon, depth, calendar month)."""
    cm = (A["month_index"][rows] % 12)
    return woa[var][cm, :, A["grid_y"][rows], A["grid_x"][rows]]   # (R, D)


anom = {}
for v in VARS:
    anom[v] = A[v] - woa_at(np.arange(A[v].shape[0]), v)
# train-only per-level std: statistics that see val/test leak the target
# distribution into the input scaling, invisibly
astd = {}
for v in VARS:
    a = anom[v][tr_m & ~held]
    sd = np.nanstd(a, axis=0)
    astd[v] = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    print(f"  {v} anomaly std per level: {astd[v].min():.3f}..{astd[v].max():.3f}",
          flush=True)
zA = {v: anom[v] / astd[v] for v in VARS}

# ------------------------------------------------- gridded streams, z-scored
def _znorm(x, mask):
    mu = np.nanmean(x[mask], axis=0); sd = np.nanstd(x[mask], axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    return np.where(np.isfinite(mu), mu, 0.0), sd


sat = {}
_tr_months = set(A["month_index"][tr_m].tolist())
tr_sat = np.array([int(mi) in _tr_months for mi in ro_mi])
for v, key in (("SST", "SST"), ("SSS", "SSS"), ("SLA", "SLA")):
    x = np.asarray(ro[key].values, "float32")
    mu, sd = _znorm(x, tr_sat)
    sat[v] = (x - mu) / sd
woaZ = np.empty((12, 2, Dn, H, W), "float32")
for mo in range(12):
    for k, v in enumerate(VARS):
        woaZ[mo, k] = (woa[v][mo] - np.nanmean(woa[v][mo])) / max(
            float(np.nanstd(woa[v][mo])), 1e-6)
woaZ = np.nan_to_num(woaZ)
lat_t = torch.tensor(LAT, dtype=torch.float32, device=dev)
lon_t = torch.tensor(LON, dtype=torch.float32, device=dev)
dep_t = torch.tensor(DEPTH, dtype=torch.float32, device=dev)
woaZ_t = torch.from_numpy(woaZ).to(dev)
satT = {k: torch.from_numpy(np.nan_to_num(v)).to(dev) for k, v in sat.items()}
ro_pos = {int(m): i for i, m in enumerate(ro_mi)}

# =========================================================== equator weighting
def draw_profiles(rows: np.ndarray, n: int, rng) -> np.ndarray:
    """Sample input profiles, weighted toward the equator but not confined to it.

    `w = (1-f) * exp(-(lat/sigma)^2 / 2) + f` — a Gaussian in latitude plus a
    uniform floor, so the draw concentrates where the reconstruction is hardest
    without ever making the rest of the ocean unobservable. Setting f = 0 would
    starve the mid-latitudes entirely and the global map would go blank there.
    """
    if rows.size == 0 or n <= 0 or rows.size <= n:
        return rows
    la = A["lat"][rows]
    w = ((1.0 - args.eq_uniform_frac)
         * np.exp(-0.5 * (la / max(args.eq_sigma, 1e-6)) ** 2)
         + args.eq_uniform_frac)
    w = w / w.sum()
    return rows[rng.choice(rows.size, size=n, replace=False, p=w)]


by_month = {}
for m in np.unique(A["month_index"]):
    by_month[int(m)] = np.flatnonzero(A["month_index"] == m)


def month_rows(mi: int, mask: np.ndarray):
    idx = by_month.get(int(mi), np.zeros(0, int))
    if idx.size == 0:
        return idx, idx
    sel = mask[idx]
    return idx[sel & ~held[idx]], idx[sel & held[idx]]


def make_obs(mi: int, src_rows: np.ndarray) -> dict:
    """The model's observation dict for one month, all streams real."""
    cm = int(mi % 12) + 1
    obs = {}
    prof = np.stack([zA["TEMP"][src_rows], zA["SALT"][src_rows]], axis=1)  # (K,2,D)
    obs["profiles"] = dict(
        prof=torch.from_numpy(np.nan_to_num(prof).astype("float32"))[None].to(dev),
        lat=torch.from_numpy(A["lat"][src_rows].astype("float32"))[None].to(dev),
        lon=torch.from_numpy(A["lon"][src_rows].astype("float32"))[None].to(dev),
        month=torch.tensor([cm], device=dev))
    ri = ro_pos[int(mi)]
    obs["surf"] = dict(
        field=torch.stack([satT["SST"][ri], satT["SSS"][ri]])[None],
        lat=lat_t, lon=lon_t, month=torch.tensor([cm], device=dev))
    obs["woa"] = dict(field=woaZ_t[cm - 1][None], lat=lat_t, lon=lon_t,
                      month=torch.tensor([cm], device=dev), depth=dep_t)
    if not args.no_ssh:
        obs["ssh"] = dict(field=satT["SLA"][ri][None, None], lat=lat_t,
                          lon=lon_t, month=torch.tensor([cm], device=dev))
    return obs


def make_query(mi: int, tgt_rows: np.ndarray):
    """(1,Q,4) queries at the held-out floats' own positions, and their T/S."""
    R = tgt_rows.size
    cm = float(int(mi % 12) + 1)
    la = np.repeat(A["lat"][tgt_rows], Dn)
    lo = np.repeat(A["lon"][tgt_rows], Dn)
    dp = np.tile(DEPTH, R)
    q = np.stack([la, lo, dp, np.full(R * Dn, cm)], -1).astype("float32")
    y = np.stack([zA["TEMP"][tgt_rows].ravel(), zA["SALT"][tgt_rows].ravel()], -1)
    di = np.tile(np.arange(Dn), R)
    ok = np.isfinite(y).all(-1)
    return (torch.from_numpy(q[ok])[None].to(dev),
            torch.from_numpy(y[ok].astype("float32")).to(dev),
            torch.from_numpy(di[ok]).to(dev),
            np.repeat(tgt_rows, Dn)[ok])


# ===================================================================== model
anchor = (tuple(int(x) for x in args.anchor_grid.split(","))
          if args.anchor_grid else None)
build_kw = dict(d_model=args.d_model, n_latent=args.n_latent,
                n_heads=args.n_heads, n_self_blocks=args.n_self_blocks,
                seed=args.seed, anchor_grid=anchor, with_ssh=not args.no_ssh)
if args.variant == "d4rt":
    build_kw.update(n_dec_blocks=args.n_dec_blocks, max_lead=1,
                    query_chunk=args.query_chunk)
model = build_fusion_model(args.variant, grid, **build_kw).to(dev)
nparam = sum(p.numel() for p in model.parameters())
print(f"  model {args.variant}: params={nparam:,}", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                        weight_decay=args.weight_decay)


def lr_at(step):
    if args.warmup and step < args.warmup:
        return (step + 1) / args.warmup
    p = min(max((step - args.warmup) / max(args.steps - args.warmup, 1), 0.0), 1.0)
    return 0.05 + 0.475 * (1 + math.cos(math.pi * p))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
months_of = lambda mask: sorted({int(m) for m in A["month_index"][mask]})
TRM, VAM, TEM = months_of(tr_m), months_of(va_m), months_of(te_m)
print(f"  months: train {len(TRM)} val {len(VAM)} test {len(TEM)}", flush=True)


@torch.no_grad()
def evaluate(months, rng_seed):
    """Per-level z SSE on held-out floats, and the WOA (zero-anomaly) floor."""
    model.eval()
    se = torch.zeros(Dn, 2, dtype=torch.float64, device=dev)
    se0 = torch.zeros(Dn, 2, dtype=torch.float64, device=dev)
    n = torch.zeros(Dn, dtype=torch.float64, device=dev)
    rng = np.random.default_rng([rng_seed, 7])
    for mi in months:
        src, tgt = month_rows(mi, np.ones(A["lat"].size, bool))
        src = draw_profiles(src, args.n_profiles, rng)
        if src.size == 0 or tgt.size == 0 or int(mi) not in ro_pos:
            continue
        q, y, di, _ = make_query(mi, tgt)
        if y.numel() == 0:
            continue
        z = model.fuse(model.encode(make_obs(mi, src), batch=1, device=dev))
        for i in range(0, q.shape[1], args.eval_chunk):
            sl = slice(i, i + args.eval_chunk)
            out = model.decode(z, q[:, sl])[0]
            se.index_add_(0, di[sl], (out - y[sl]).double() ** 2)
            se0.index_add_(0, di[sl], y[sl].double() ** 2)
            n.index_add_(0, di[sl], torch.ones_like(di[sl], dtype=torch.float64))
    model.train()
    return se, se0, n


def phys(se, n):
    out = {}
    for k, v in enumerate(VARS):
        sp = se[:, k] * torch.tensor(astd[v] ** 2, device=dev).double()
        out[v] = float(torch.sqrt(sp.sum() / n.sum().clamp(min=1)))
    return out




# ==========================================================================
# Secondary truth: EN4 objective analysis, for the DENSE field
# ==========================================================================
# The primary number stays point-wise against held-out floats. But point
# observations cannot fill a 1 deg map, and a dense map is what shows WHERE the
# reconstruction fails. EN4 is the defensible gridded reference: a pure in-situ
# objective analysis, monthly, 1 deg, shipping its own uncertainty field.
#
# Two things this is NOT, and the report must say so:
#   * It is not an error. EN4 assimilates the very profiles fed in AND the
#     held-out ones, so this measures AGREEMENT WITH THE OPERATIONAL ANALYSIS.
#   * EN4 is smoothed. A model that correctly resolves mesoscale structure is
#     penalised against it, so EN4-RMSE partly rewards smoothness.
#
# EN4 sits on INTEGER degrees while our analysis grid uses half-degree centres,
# so the reference is never interpolated: the model is queried at EN4's own
# cell centres and depths, and only WOA23 -- a climatology -- is moved.
def load_en4_global(years):
    import glob
    fs = sorted(f for y in years for f in
                glob.glob(os.path.join(ROOT, "data", "reference", "en4",
                                       f"*.global.{y}.nc")))
    if not fs:
        return None
    ds = xr.open_mfdataset(fs, combine="by_coords").load()
    return ds


def woa_on(lat_e, lon_e, dep_e):
    """WOA23 monthly climatology interpolated onto EN4's axes: (12,2,D,H,W)."""
    src = xr.open_zarr(os.path.join(ROOT, "data", "woa23_standard.zarr"))
    src = src.assign_coords(lon=(src.lon % 360.0)).sortby("lon")
    out = np.empty((12, 2, dep_e.size, lat_e.size, lon_e.size), "float32")
    for k, v in enumerate(VARS):
        di = src[v].interp(lat=lat_e, lon=lon_e, depth=dep_e, method="linear",
                           kwargs={"fill_value": None}).values
        out[:, k] = np.asarray(di, "float32")
    return out


@torch.no_grad()
def dense_en4_map(model, months):
    """Model minus EN4 on EN4's own grid, per depth band. Returns maps + stats."""
    en4 = load_en4_global(range(TE[0], TE[1] + 1))
    if en4 is None:
        print("  EN4 global not present — skipping the dense map", flush=True)
        return None
    lat_e = np.asarray(en4.lat.values, float)
    lon_e = np.asarray(en4.lon.values, float) % 360.0
    dep_e = np.abs(np.asarray(en4.depth.values, float))
    keep = dep_e <= 1000.0
    dep_e = dep_e[keep]
    He, We, De = lat_e.size, lon_e.size, dep_e.size
    print(f"  EN4 grid {He}x{We}, {De} levels to {dep_e.max():.0f} m", flush=True)
    woaE = woa_on(lat_e, lon_e, dep_e)                       # (12,2,De,He,We)
    e_time = np.asarray(en4.time.values, "datetime64[M]")
    e_pos = {int(m.astype(int) - MONTH0.astype(int)): i for i, m in enumerate(e_time)}

    band_of = np.full(De, len(BANDS), int)
    for bi, (_, lo, hi) in enumerate(BANDS):
        sel = (dep_e > lo) & (dep_e <= hi)
        if lo <= dep_e.min():
            sel |= np.isclose(dep_e, dep_e.min())
        band_of[sel] = bi
    npanel = len(BANDS) + 1
    ss = {v: np.zeros((npanel, He, We)) for v in VARS}
    cc = {v: np.zeros((npanel, He, We)) for v in VARS}
    # For --demean the annual mean has to be removed at the (level, cell) the
    # difference actually lives on, BEFORE anything is pooled into a depth
    # band. De-meaning a band average instead would subtract a mean taken
    # across depths as well as months, which removes vertical structure rather
    # than the year-round offset.
    sd = sq = nl = None
    if args.demean:
        sd = {v: np.zeros((De, He, We)) for v in VARS}   # sum of d
        sq = {v: np.zeros((De, He, We)) for v in VARS}   # sum of d^2
        nl = {v: np.zeros((De, He, We)) for v in VARS}   # months contributing

    # query grid, built once: every EN4 cell at every level
    yy, xx = np.meshgrid(np.arange(He), np.arange(We), indexing="ij")
    q_lat = np.repeat(lat_e[yy].ravel(), De)
    q_lon = np.repeat(lon_e[xx].ravel(), De)
    q_dep = np.tile(dep_e, He * We)
    di_all = np.tile(np.arange(De), He * We)
    cell = np.repeat(np.arange(He * We), De)

    rng = np.random.default_rng([args.seed, 7])
    model.eval()
    for mi in months:
        if int(mi) not in e_pos or int(mi) not in ro_pos:
            continue
        src, _ = month_rows(mi, np.ones(A["lat"].size, bool))
        src = draw_profiles(src, args.n_profiles, rng)
        if src.size == 0:
            continue
        cm = int(mi % 12)
        ti = e_pos[int(mi)]
        eT = np.asarray(en4["temperature"].isel(time=ti).values, float)[keep] - 273.15
        eS = np.asarray(en4["salinity"].isel(time=ti).values, float)[keep]
        eU = np.asarray(en4["temperature_uncertainty"].isel(time=ti).values,
                        float)[keep]
        # EN4 as an anomaly against the same climatology the model predicts against
        # EN4 arrives (D, H, W) -- depth-major -- but the queries are built
        # cell-major (cell, depth), so a plain ravel pairs each prediction with
        # the WRONG depth and paints regular stripes across the map. Move depth
        # last so the flat index is cell*De + d, matching the query order.
        f = lambda a: np.moveaxis(a, 0, -1).reshape(-1)
        aT = f(eT - woaE[cm, 0])
        aS = f(eS - woaE[cm, 1])
        unc = f(eU)
        z = model.fuse(model.encode(make_obs(mi, src), batch=1, device=dev))
        q = torch.from_numpy(np.stack(
            [q_lat, q_lon, q_dep, np.full(q_lat.size, float(cm + 1))], -1
        ).astype("float32"))[None]
        preds = []
        for i in range(0, q.shape[1], args.eval_chunk):
            preds.append(model.decode(z, q[:, i:i + args.eval_chunk].to(dev))[0].cpu())
        pr = torch.cat(preds).numpy()
        for k, v in enumerate(VARS):
            truth = aT if v == "TEMP" else aS
            phys = pr[:, k] * astd[v][np.minimum(di_all, astd[v].size - 1)]
            ok = (np.isfinite(truth) & np.isfinite(phys)
                  & (unc <= args.en4_max_unc))
            if not ok.any():
                continue
            d = phys[ok] - truth[ok]
            cy, cx = cell[ok] // We, cell[ok] % We
            if args.demean:
                lv = di_all[ok]
                np.add.at(sd[v], (lv, cy, cx), d)
                np.add.at(sq[v], (lv, cy, cx), d * d)
                np.add.at(nl[v], (lv, cy, cx), 1.0)
                continue
            e2 = d ** 2
            for panel in (band_of[di_all[ok]], np.full(int(ok.sum()), len(BANDS))):
                np.add.at(ss[v], (panel, cy, cx), e2)
                np.add.at(cc[v], (panel, cy, cx), 1.0)
    model.train()
    if args.demean:
        # Sum of squares about each (level, cell)'s own annual mean, with the
        # one degree of freedom the mean costs. A cell seen in only one month
        # carries no information about variability and contributes nothing.
        for v in VARS:
            n = nl[v]
            SS = sq[v] - np.divide(sd[v] ** 2, np.maximum(n, 1.0),
                                   where=n > 0, out=np.zeros_like(sq[v]))
            SS = np.maximum(SS, 0.0)                 # guard rounding to <0
            dof = np.maximum(n - 1.0, 0.0)
            for lv in range(De):
                for panel in (band_of[lv], len(BANDS)):
                    ss[v][panel] += SS[lv]
                    cc[v][panel] += dof[lv]
    rm = {v: np.where(cc[v] > 0, np.sqrt(np.divide(ss[v], np.maximum(cc[v], 1))),
                      np.nan) for v in VARS}
    stats = {v: float(np.sqrt(ss[v][-1].sum() / max(cc[v][-1].sum(), 1)))
             for v in VARS}
    return dict(map=rm, counts=cc, lat=lat_e, lon=lon_e, rmse=stats)


# =================================================================== training
best, best_state, hist = float("inf"), None, []
rng = np.random.default_rng(args.seed)
if args.from_checkpoint:
    # Re-plotting an existing run: the figure is a pure function of the trained
    # weights, so nothing needs retraining. steps=0 empties the loop below.
    model.load_state_dict(torch.load(args.from_checkpoint, map_location=dev))
    args.steps = 0
    print(f"  loaded {args.from_checkpoint} — skipping training", flush=True)
else:
    print("  training ...", flush=True)
for step in range(1, args.steps + 1):
    mi = int(rng.choice(TRM))
    src, tgt = month_rows(mi, tr_m)
    # a float is an input or a target in a step, never both
    if src.size < 8 or int(mi) not in ro_pos:
        continue
    # Partition by WMO, not by profile. A float's cycles within a month sit a
    # few hundred km apart and share a sensor, so predicting one cycle from
    # another of the same instrument is close to copying -- the model learns to
    # lean on the nearest input profile instead of on the field. Same rule the
    # regional driver uses.
    fl = np.unique(A["wmo"][src])
    if fl.size < 2:
        continue
    n_t = max(1, min(fl.size - 1, int(round(0.3 * fl.size))))
    tgt_f = set(rng.choice(fl, n_t, replace=False).tolist())
    is_t = np.isin(A["wmo"][src], list(tgt_f))
    src_i, tgt_i = src[~is_t], src[is_t]
    if src_i.size == 0 or tgt_i.size == 0:
        continue
    src_i = draw_profiles(src_i, args.n_profiles, rng)
    if tgt_i.size == 0:
        continue
    q, y, di, _ = make_query(mi, tgt_i)
    if y.numel() == 0:
        continue
    if args.train_queries and y.shape[0] > args.train_queries:
        sel = torch.from_numpy(
            rng.choice(y.shape[0], args.train_queries, replace=False)).to(dev)
        q, y, di = q[:, sel], y[sel], di[sel]
    z = model.fuse(model.encode(make_obs(mi, src_i), batch=1, device=dev))
    out = model.decode(z, q)[0]
    loss = torch.nn.functional.mse_loss(out, y)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % args.val_every == 0 or step == args.steps:
        se, se0, n = evaluate(VAM, args.seed)
        r = phys(se, n); fl = phys(se0, n)
        sk = 1 - np.mean([r[v] / max(fl[v], 1e-9) for v in VARS])
        hist.append({"step": step, "loss": float(loss), **r, "skill": sk})
        star = ""
        if sk > -1e9 and (1 - sk) < best:
            best = 1 - sk
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            star = " *best*"
        print(f"    step {step:6d} loss {float(loss):.4f}  val TEMP {r['TEMP']:.4f} "
              f"SALT {r['SALT']:.4f}  skill {sk:.3f}{star}", flush=True)
if best_state is not None:
    model.load_state_dict(best_state)
if not args.from_checkpoint:
    # Never write the checkpoint on a re-plot: it would overwrite a trained
    # run with whatever was just loaded, under the same tag.
    torch.save(model.state_dict(), os.path.join(CKPT, f"{tag}.pt"))

# ==================================================================== testing
se, se0, n = evaluate(TEM, args.seed)
test, floor = phys(se, n), phys(se0, n)
print(f"\n  TEST  TEMP {test['TEMP']:.4f} (WOA floor {floor['TEMP']:.4f})  "
      f"SALT {test['SALT']:.4f} (floor {floor['SALT']:.4f})", flush=True)


# ================================================================= the figure
@torch.no_grad()
def error_map():
    """Binned physical RMSE per (bin, band) at held-out float positions."""
    model.eval()
    nb = int(round(args.bin_deg))
    BY, BX = int(np.ceil(180 / nb)), int(np.ceil(360 / nb))
    npanel = len(BANDS) + 1
    ss = {v: np.zeros((npanel, BY, BX)) for v in VARS}
    cc = {v: np.zeros((npanel, BY, BX)) for v in VARS}
    band_of = np.full(Dn, len(BANDS), int)
    for bi, (_, lo, hi) in enumerate(BANDS):
        sel = (DEPTH > lo) & (DEPTH <= hi)
        if lo <= DEPTH.min():
            sel |= np.isclose(DEPTH, DEPTH.min())
        band_of[sel] = bi
    rng = np.random.default_rng([args.seed, 7])
    for mi in TEM:
        src, tgt = month_rows(mi, np.ones(A["lat"].size, bool))
        src = draw_profiles(src, args.n_profiles, rng)
        if src.size == 0 or tgt.size == 0 or int(mi) not in ro_pos:
            continue
        q, y, di, rows = make_query(mi, tgt)
        if y.numel() == 0:
            continue
        z = model.fuse(model.encode(make_obs(mi, src), batch=1, device=dev))
        out = torch.cat([model.decode(z, q[:, i:i + args.eval_chunk])[0]
                         for i in range(0, q.shape[1], args.eval_chunk)])
        e = (out - y).cpu().numpy() ** 2
        dii = di.cpu().numpy()
        by = np.clip(((A["lat"][rows] + 90.0) / nb).astype(int), 0, BY - 1)
        bx = np.clip((A["lon"][rows] / nb).astype(int), 0, BX - 1)
        for k, v in enumerate(VARS):
            ph = e[:, k] * (astd[v][dii] ** 2)
            for panel in (band_of[dii], np.full(dii.size, len(BANDS))):
                np.add.at(ss[v], (panel, by, bx), ph)
                np.add.at(cc[v], (panel, by, bx), 1.0)
    rm = {v: np.where(cc[v] >= args.min_obs,
                      np.sqrt(np.divide(ss[v], np.maximum(cc[v], 1))), np.nan)
          for v in VARS}
    return rm, cc, nb, BY, BX


rm, cc, nb, BY, BX = error_map()
PANELS = [b[0] for b in BANDS] + ["full column"]
cmap = plt.cm.hot.copy(); cmap.set_bad("0.75")
fig, axes = plt.subplots(len(VARS), len(PANELS),
                         figsize=(4.1 * len(PANELS), 6.6), constrained_layout=True)
for r, v in enumerate(VARS):
    vmax = float(np.nanpercentile(rm[v], 98))
    for ci, nm in enumerate(PANELS):
        ax = axes[r, ci]
        im = ax.imshow(np.ma.masked_invalid(rm[v][ci]), origin="lower",
                       extent=[0, 360, -90, 90], aspect="auto", cmap=cmap,
                       vmin=0.0, vmax=vmax)
        if r == 0: ax.set_title(nm)
        if ci == 0: ax.set_ylabel(f"{v} ({UNITS[v]})\nlatitude")
        if r == len(VARS) - 1: ax.set_xlabel("longitude")
        ax.axhline(0.0, color="0.4", lw=0.6, ls=":")
    fig.colorbar(im, ax=axes[r, :], fraction=0.02,
                 label=f"{v} RMSE ({UNITS[v]})")
fig.suptitle(
    f"2-D T/S reconstruction error by depth layer — REAL DATA — "
    f"DFS-Attention + Perceiver-IO latent + D4RT query decoder ({nparam:,} params)\n"
    f"{len(TEM)} held-out months ({TE[0]}), {args.n_profiles} real Argo "
    f"profiles/month (equator-weighted, sigma={args.eq_sigma:.0f} deg, "
    f"{args.eq_uniform_frac:.0%} uniform), anomaly vs WOA23.  "
    f"Inputs: Argo + OISST/OISSS + WOA23 + DUACS SLA.\n"
    f"scored ONLY at WMO-disjoint held-out float positions, binned to "
    f"{nb} deg; dark = accurate, bright = more error, "
    f"grey = no held-out float there", fontsize=9)
fp = os.path.join(REPORTS, f"fig_{tag}.png")
fig.savefig(fp, dpi=130); plt.close(fig)
print(f"  wrote {os.path.relpath(fp, ROOT)}")


# ---------------------------- the DENSE map, scored against EN4 -------------
den = dense_en4_map(model, TEM)
if den is not None:
    ext = [float(den["lon"].min()), float(den["lon"].max()),
           float(den["lat"].min()), float(den["lat"].max())]
    fig, axes = plt.subplots(len(VARS), len(PANELS),
                             figsize=(4.1 * len(PANELS), 6.6),
                             constrained_layout=True)
    for r, v in enumerate(VARS):
        vmax = float(np.nanpercentile(den["map"][v], 98))
        for ci, nm in enumerate(PANELS):
            ax = axes[r, ci]
            im = ax.imshow(np.ma.masked_invalid(den["map"][v][ci]),
                           origin="lower", extent=ext, aspect="auto",
                           cmap=cmap, vmin=0.0, vmax=vmax)
            if r == 0: ax.set_title(nm)
            if ci == 0: ax.set_ylabel(f"{v} ({UNITS[v]})\nlatitude")
            if r == len(VARS) - 1: ax.set_xlabel("longitude")
        fig.colorbar(im, ax=axes[r, :], fraction=0.02,
                     label=f"{v} RMSE vs EN4 ({UNITS[v]})")
    fig.suptitle(
        f"2-D T/S reconstruction error by depth layer — REAL DATA — "
        f"DFS-Attention + Perceiver-IO latent + D4RT query decoder "
        f"({nparam:,} params)\n"
        f"{len(TEM)} held-out months ({TE[0]}), {args.n_profiles} real Argo "
        f"profiles/month (equator-weighted), anomaly target.  Inputs: Argo + "
        f"OISST/OISSS + WOA23 + DUACS SLA.\n"
        f"scored against the EN4 objective analysis on ITS OWN 1 deg grid "
        f"(agreement, not error — EN4 assimilates these same floats); "
        f"dark = closer to EN4, grey = land or EN4 uncertainty > "
        f"{args.en4_max_unc:g} degC"
        # own line: the note is long enough to run off the canvas if appended
        + ("\nANNUAL MEAN REMOVED at each cell and level — a constant "
           "year-round offset drops out, so this maps only the TIME-VARYING "
           "disagreement (RMSE about each cell's own 2023 mean, n-1 weighted)"
           if args.demean else ""), fontsize=9)
    fp2 = os.path.join(REPORTS, f"fig_{tag}_en4.png")
    fig.savefig(fp2, dpi=130); plt.close(fig)
    print(f"  wrote {os.path.relpath(fp2, ROOT)}")
    print(f"  EN4 agreement (full column): "
          + "  ".join(f"{v} {den['rmse'][v]:.4f}" for v in VARS), flush=True)

rec = {"tag": tag, "task": "real_argo_global_2d_reconstruction",
       "variant": args.variant, "params": nparam,
       "split": {"train": TR, "val": VA, "test": TE},
       "n_profiles": args.n_profiles, "eq_sigma": args.eq_sigma,
       "eq_uniform_frac": args.eq_uniform_frac,
       "test": test, "woa_floor": floor,
       "skill": {v: 1 - test[v] / max(floor[v], 1e-9) for v in VARS},
       "bin_deg": nb, "min_obs": args.min_obs,
       "cells_scored": int(np.isfinite(rm["TEMP"][-1]).sum()),
       "cells_total": int(BY * BX), "history": hist,
       "en4_agreement": (den["rmse"] if den is not None else None),
       "en4_max_unc": args.en4_max_unc,
       "en4_cells": (int(np.isfinite(den["map"]["TEMP"][-1]).sum())
                     if den is not None else None),
       "git_commit": P.git_commit(), "protocol_hash": P.protocol_hash()}
with open(os.path.join(CACHE, f"{tag}.json"), "w") as f:
    json.dump(rec, f, indent=1, default=str)
print(json.dumps({k: rec[k] for k in
                  ("test", "woa_floor", "skill", "cells_scored", "cells_total")},
                 indent=1))
print(f"total {time.time()-t0:.0f}s")
