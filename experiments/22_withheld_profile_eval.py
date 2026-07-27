"""Task 9b — withheld-profile evaluation (leave-profiles-out generalisation).

protocol_v1's headline metric scores *unobserved grid cells*; this script adds
the supporting decode demonstration the protocol lists but never ran: predict
full-depth T/S at the coordinates of profiles that were **held out of the
model input**, and score against the (noise-free OSSE) truth there.  It is the
synthetic-data twin of a real-Argo leave-one-profile-out test.

For each pinned test month a fixed 1500-profile draw is split into an INPUT set
(fed to the model / baselines) and a WITHHELD set (K_wh columns, never seen);
every method predicts the full column at the withheld columns and is scored on
the identical set.  Because anomaly RMSE equals absolute RMSE (the climatology
cancels), the shared-latent variants (anomaly z-space) and the field baselines
(absolute) are directly comparable.

Methods: the three fusion variants (from their fullA checkpoints, one per
seed) vs nearest-profile-from-input fill and the climatology floor.  Reported
per variable, full column + depth band + a near/far split by distance to the
nearest *input* profile (the axis withheld-profile skill actually rides on).

    !! REAL ARGO: this uses SYNTHETIC profiles — there is no real-Argo store on
    this machine (data/ holds only CESM2-LE + WOA23).  A true real-Argo LOPO
    needs the external Argo dataset and must additionally guard against
    reanalysis-assimilation leakage; that is deferred until the data lands.

Run:  CUDA_VISIBLE_DEVICES=6 python experiments/22_withheld_profile_eval.py
"""
import sys, os, json, time, argparse, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
from scipy.spatial import cKDTree

from ocean_tokenizer import data, baselines as B, config as C
from ocean_tokenizer.anomaly import Climatology, AnomNorm
from ocean_tokenizer.fusion import build_fusion_model
from ocean_tokenizer.fullrun import FullRunData, VARS

TRAIN_YEARS = (1985, 2007)
TEST_TIME_INDICES = [1933, 1935, 1936, 1938, 1942, 1946, 1952, 1953,
                     1956, 1965, 1967, 1976]
BANDS = [("0-100m", 0.0, 100.0), ("100-300m", 100.0, 300.0),
         ("300-max", 300.0, 1e9)]

ap = argparse.ArgumentParser()
ap.add_argument("--variants", default="mbca,perceiver,resampler")
ap.add_argument("--seeds", default="1234,1235,1236")
ap.add_argument("--prefix", default="fullA")
ap.add_argument("--withhold", type=int, default=300,
                help="profiles held out of the input per month (of 1500)")
ap.add_argument("--far-deg", type=float, default=3.0,
                help="near/far split: great-circle distance (deg) to nearest "
                     "input profile")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
VARIANTS = args.variants.split(",")
SEEDS = [int(s) for s in args.seeds.split(",")]

t0 = time.time()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

grid = data.CommonGrid()
tr_idx = data.select_month_indices(C.GT_SOURCE, TRAIN_YEARS)
te_idx = np.asarray(TEST_TIME_INDICES)
if args.smoke:
    tr_idx, te_idx = tr_idx[:8], te_idx[:2]
print(f"loading fields (train={tr_idx.size} test={te_idx.size}) ...", flush=True)
ftrain = data.load_gt_fields(tr_idx, grid)
ftest = data.load_gt_fields(te_idx, grid)
woa = data.woa_prior(grid)
surf_train = {v: ftrain[v] for v in C.VARS_SURF if v in ftrain}
clim = Climatology(ftrain, surf_train)
norm = AnomNorm(clim, ftrain, surf_train)
del ftrain

rd = FullRunData(grid, norm, dev)
rd.load_woa(woa)
D, H, W = rd.D, rd.H, rd.W
depth_np = grid.depth
astd = {v: norm.astd[v] for v in VARS}          # (D,)
ZA_te = {}                                       # per test month z-volume, GPU
surfZ_te = torch.from_numpy(rd.z_surf(ftest)).to(dev)
za_np = rd.z_volume(ftest)                        # (T,2,D,H,W) numpy z
ZA_te = torch.from_numpy(za_np).to(dev)


def _gc_deg(lat1, lon1, lat2, lon2):
    la1, la2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dlon = np.deg2rad(lon2 - lon1)
    c = (np.sin(la1) * np.sin(la2)
         + np.cos(la1) * np.cos(la2) * np.cos(dlon)).clip(-1, 1)
    return np.rad2deg(np.arccos(c))


def make_split(seed):
    """Per test month: input/withheld column split + nearest-input distances.

    Returns list of dicts (one per month) with GPU input-profile index tensors,
    withheld (D,K) flat query indices, withheld depth idx, and a near-mask.
    """
    rng = np.random.default_rng([seed, 7])
    oi, oj = rd.oi, rd.oj
    months = []
    for t, mo in enumerate(ftest["months"]):
        mo = int(mo)
        pick = rng.choice(rd.n_ocean, size=min(C.N_PROFILES, rd.n_ocean),
                          replace=False)
        n_wh = min(args.withhold, pick.size // 2)
        wh, inp = pick[:n_wh], pick[n_wh:]
        ii_in, jj_in = oi[inp], oj[inp]
        ii_wh, jj_wh = oi[wh], oj[wh]
        # distance of each withheld column to nearest INPUT profile (great circle)
        tree = cKDTree(np.deg2rad(np.stack([grid.lat[ii_in], grid.lon[jj_in]], 1)))
        # approximate NN in lat/lon rad space then exact gc on the match
        _, nn = tree.query(np.deg2rad(np.stack([grid.lat[ii_wh],
                                                grid.lon[jj_wh]], 1)), k=1)
        dist = _gc_deg(grid.lat[ii_wh], grid.lon[jj_wh],
                       grid.lat[ii_in][nn], grid.lon[jj_in][nn])
        months.append(dict(
            t=t, mo=mo,
            ii_in=torch.from_numpy(ii_in).to(dev),
            jj_in=torch.from_numpy(jj_in).to(dev),
            ii_wh=ii_wh, jj_wh=jj_wh, dist=dist))
    return months


def band_rmse(se_phys, n_lev):
    """se_phys (D,2), n_lev (D,2) -> {full,by_band} physical RMSE per var.

    NaN-aware: n_lev counts only the finite (ocean, in-bathymetry) cells that
    contributed to se_phys, so deep withheld columns that are NaN in the truth
    never inflate or deflate the RMSE."""
    out = {"full": {}, "by_band": {}}
    for k, v in enumerate(VARS):
        out["full"][v] = float(np.sqrt(se_phys[:, k].sum()
                                       / max(n_lev[:, k].sum(), 1)))
        out["by_band"][v] = {}
        for name, lo, hi in BANDS:
            sel = (depth_np > lo) & (depth_np <= hi)
            if lo <= depth_np.min():
                sel |= np.isclose(depth_np, depth_np.min())
            out["by_band"][v][name] = float(
                np.sqrt(se_phys[sel, k].sum() / max(n_lev[sel, k].sum(), 1)))
    return out


def accumulate(err_phys_by_month):
    """list of ((D,K,2) errors, (K,) near-mask) -> se/n for all + near + far."""
    def z(): return np.zeros((D, 2)), np.zeros((D, 2))
    se, n = z(); se_near, n_near = z(); se_far, n_far = z()
    for e, near in err_phys_by_month:
        fin = np.isfinite(e)
        e2 = np.where(fin, e, 0.0) ** 2               # (D,K,2)
        fc = fin.astype(float)
        se += e2.sum(1); n += fc.sum(1)
        se_near += e2[:, near].sum(1); n_near += fc[:, near].sum(1)
        se_far += e2[:, ~near].sum(1); n_far += fc[:, ~near].sum(1)
    return (band_rmse(se, n), band_rmse(se_near, n_near),
            band_rmse(se_far, n_far))


@torch.no_grad()
def model_errors(model, split):
    """(D,K,2) physical anomaly errors at withheld columns, per month."""
    out = []
    for m in split:
        t, mo = m["t"], m["mo"]
        obs = rd.obs_dict(ZA_te, surfZ_te, t, mo, m["ii_in"], m["jj_in"])
        z = model.fuse(model.encode(obs, batch=1, device=dev))
        ii_wh = torch.from_numpy(m["ii_wh"]).to(dev)
        jj_wh = torch.from_numpy(m["jj_wh"]).to(dev)
        K = ii_wh.numel()
        di = torch.arange(D, device=dev).repeat_interleave(K)
        q = torch.stack([rd.lat_t[ii_wh.repeat(D)], rd.lon_t[jj_wh.repeat(D)],
                         rd.depth_t[di],
                         torch.full((D * K,), float(mo), device=dev)], -1)[None]
        pred = model.decode(z, q)[0].reshape(D, K, 2).float().cpu().numpy()  # z
        tgt = za_np[t][:, :, m["ii_wh"], m["jj_wh"]].transpose(1, 2, 0)      # (2,D,K)->(D,K,2)
        err_z = pred - tgt
        err_phys = err_z * np.stack([astd[v] for v in VARS], 1)[:, None, :]
        near = m["dist"] <= args.far_deg
        out.append((err_phys, near))
    return out


def baseline_errors(predfn, split):
    """(D,K,2) physical errors for an absolute-field baseline at withheld cols."""
    out = []
    for m in split:
        t, mo = m["t"], m["mo"]
        # rebuild a prepare_month sample using ONLY input profiles this month
        s = _sample_from_split(m)
        pv = predfn(s)                                   # {v:(D,H,W)} absolute
        err = np.stack([pv[v][:, m["ii_wh"], m["jj_wh"]]
                        - ftest[v][t][:, m["ii_wh"], m["jj_wh"]]
                        for v in VARS], -1)               # (D,K,2)
        near = m["dist"] <= args.far_deg
        out.append((err, near))                           # NaN-aware downstream
    return out


def _sample_from_split(m):
    """Minimal prepare_month-style sample with only the INPUT profiles."""
    t = m["t"]
    ii = m["ii_in"].cpu().numpy(); jj = m["jj_in"].cpu().numpy()
    prof = {"ij": np.stack([ii, jj], 1),
            "lat": grid.lat[ii], "lon": grid.lon[jj],
            "month": np.full(ii.size, m["mo"], int)}
    for v in VARS:
        prof[v] = ftest[v][t][:, ii, jj].T
    s = {"month": m["mo"], "prof": prof,
         "gt": {v: ftest[v][t] for v in VARS},
         "woa": {v: woa[v][m["mo"] - 1] for v in VARS}}
    oi, oj = np.where(grid.ocean)
    cell_xyz = B._xyz(grid.lat[oi], grid.lon[oj])
    prof_xyz = B._xyz(prof["lat"], prof["lon"])
    dist, nn = cKDTree(prof_xyz).query(cell_xyz, k=1)
    s["ocean_ij"] = (oi, oj); s["nn"] = nn; s["nn_dist"] = dist
    near = {v: np.full((D, H, W), np.nan, "float32") for v in VARS}
    for v in VARS:
        near[v][:, oi, oj] = prof[v][nn].T
    s["near"] = near
    return s


results = {}
for seed in SEEDS:
    split = make_split(seed)
    kwh = sum(m["ii_wh"].size for m in split)
    nfar = sum((m["dist"] > args.far_deg).sum() for m in split)
    print(f"seed {seed}: {kwh} withheld cols pooled "
          f"({nfar} far > {args.far_deg} deg)", flush=True)
    # baselines (depend only on the seed split)
    results.setdefault("clim_floor", {})[seed] = accumulate(
        baseline_errors(lambda s: B.predict_clim_floor(s, clim, grid), split))
    results.setdefault("nearest", {})[seed] = accumulate(
        baseline_errors(lambda s: B.predict_nearest(s, use_woa=True), split))
    print(f"  [nearest   ] full TEMP="
          f"{results['nearest'][seed][0]['full']['TEMP']:.4f}", flush=True)
    for variant in VARIANTS:
        tag = f"{args.prefix}_{variant}_s{seed}"
        ckpt = torch.load(os.path.join(C.CKPT, f"{tag}.pt"), map_location=dev)
        ra = ckpt["args"]
        anchor = (tuple(int(x) for x in ra["anchor_grid"].split(","))
                  if ra.get("anchor_grid") else None)
        model = build_fusion_model(variant, grid, d_model=ra["d_model"],
                                   n_latent=ra["n_latent"], n_heads=ra["n_heads"],
                                   n_self_blocks=ra["n_self_blocks"],
                                   seed=seed, anchor_grid=anchor).to(dev)
        model.load_state_dict(ckpt["state_dict"]); model.eval()
        results.setdefault(variant, {})[seed] = accumulate(
            model_errors(model, split))
        print(f"  [{variant:10s}] full TEMP="
              f"{results[variant][seed][0]['full']['TEMP']:.4f} "
              f"far TEMP={results[variant][seed][2]['full']['TEMP']:.4f}",
              flush=True)


def agg(method, which, var, field="full", band=None):
    """which: 0 all / 1 near / 2 far.  Mean+/-std over seeds."""
    vals = []
    for seed in SEEDS:
        r = results[method][seed][which]
        vals.append(r["full"][var] if band is None else r["by_band"][var][band])
    return float(np.mean(vals)), float(np.std(vals))


summary = {}
for method in results:
    summary[method] = {}
    for lbl, which in (("all", 0), ("near", 1), ("far", 2)):
        m_t, s_t = agg(method, which, "TEMP")
        m_s, s_s = agg(method, which, "SALT")
        summary[method][lbl] = {
            "TEMP_mean": m_t, "TEMP_std": s_t,
            "SALT_mean": m_s, "SALT_std": s_s,
            "TEMP_bands": {b[0]: agg(method, which, "TEMP", band=b[0])[0]
                           for b in BANDS}}


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


out = {"task": "withheld_profile_eval", "protocol": "protocol_v1",
       "data": "synthetic_OSSE (no real Argo store present)",
       "git_commit": git_commit(), "smoke": args.smoke,
       "seeds": SEEDS, "variants": VARIANTS, "withhold_per_month": args.withhold,
       "far_deg": args.far_deg, "n_profiles": C.N_PROFILES,
       "test_months": te_idx.tolist(), "summary": summary,
       "gpu_hours": round((time.time() - t0) / 3600, 3)}
path = os.path.join(C.CACHE, "withheld_profile_eval.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nDONE in {(time.time()-t0)/60:.1f} min -> {path}", flush=True)
print("\nmethod        | all TEMP | near TEMP | far TEMP | all SALT")
for m in list(VARIANTS) + ["nearest", "clim_floor"]:
    if m not in summary:
        continue
    s = summary[m]
    print(f"{m:13s} | {s['all']['TEMP_mean']:.4f}   | "
          f"{s['near']['TEMP_mean']:.4f}    | {s['far']['TEMP_mean']:.4f}   | "
          f"{s['all']['SALT_mean']:.4f}", flush=True)
