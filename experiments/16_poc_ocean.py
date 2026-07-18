"""Task 7 Stage B — small ocean-subset proof of concept for the fusion variants.

A functionality gate, NOT the final result: 4 protocol_v1 train months, 1
validation month, real modality encoders (profiles / SST-SSS / WOA volume),
subsampled query targets.  For each variant (standard Perceiver /
fixed-budget resampler / MBCA):

  * tiny-subset train RMSE (can it overfit?)          — anomaly z-space
  * validation-month unobserved-only anomaly RMSE      — degC / PSU
  * duplicate sensitivity   (half the profiles duplicated; no new info)
  * patch-refinement sensitivity (same WOA/surf fields fed at 2x resolution
    through the same encoder -> 4x grid tokens, no new info)
  * profile-resampling sensitivity (same profiles linearly interpolated to
    2x vertical levels via the ragged-depth encoder path, no new info)

Success criteria (Monday brief): all three overfit; MBCA exact-partition test
passes (unit suite); MBCA more stable under duplication; no NaNs; missing-
modality batches run.

Run:  CUDA_VISIBLE_DEVICES=6 python experiments/16_poc_ocean.py
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from ocean_tokenizer import data, baselines as B, config as C
from ocean_tokenizer.anomaly import Climatology, AnomNorm
from ocean_tokenizer.token_api import sample_to_obs, TokenBatch
from ocean_tokenizer.fusion import build_fusion_model

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--queries", type=int, default=8192, help="query points per step")
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--d-model", type=int, default=128)
ap.add_argument("--n-latent", type=int, default=128)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
print(f"POC-B device={dev} steps={args.steps}", flush=True)

# ---- protocol_v1-consistent tiny subset: 4 train months, 1 val month ----
grid = data.CommonGrid()
tr_pool = data.select_month_indices(C.GT_SOURCE, (1985, 2007))
va_pool = data.select_month_indices(C.GT_SOURCE, (2008, 2010))
tr_idx = tr_pool[[0, 60, 120, 200]]          # spread over the era, 4 months
va_idx = va_pool[[6]]                        # one validation month
ftrain = data.load_gt_fields(tr_idx, grid)
fval = data.load_gt_fields(va_idx, grid)
woa = data.woa_prior(grid)

surf_train = {v: ftrain[v] for v in C.VARS_SURF if v in ftrain}
clim = Climatology(ftrain, surf_train)
norm = AnomNorm(clim, ftrain, surf_train)

rng = np.random.default_rng(args.seed)
train_samples = [B.prepare_month(ftrain, ftrain, woa, grid, t, rng, C.N_PROFILES)
                 for t in range(len(ftrain["months"]))]
val_samples = [B.prepare_month(fval, fval, woa, grid, t, rng, C.N_PROFILES)
               for t in range(len(fval["months"]))]

VARS = B.VARS
D, H, W = grid.ndepth, grid.nlat, grid.nlon


def month_obs(sample):
    obs = sample_to_obs(sample, grid, norm, device=dev)
    return {k: {kk: vv for kk, vv in v.items()} for k, v in obs.items()}


def query_pool(sample):
    """All unobserved ocean (i, j, d) query coords + z-space anomaly targets."""
    oi, oj = np.where(sample["unobs_mask"])
    mo = sample["month"]
    z = np.stack([norm.z3d(v, sample["gt"][v], mo) for v in VARS], -1)  # (D,H,W,2)
    di = np.repeat(np.arange(D), oi.size)
    ii = np.tile(oi, D); jj = np.tile(oj, D)
    y = z[di, ii, jj]                                   # (Q,2)
    ok = np.isfinite(y).all(1)
    q = np.stack([grid.lat[ii], grid.lon[jj], grid.depth[di],
                  np.full(ii.size, mo)], -1).astype("float32")
    return q[ok], y[ok].astype("float32")


print("preparing month observation dicts + query pools ...", flush=True)
train_data = [(month_obs(s), *query_pool(s)) for s in train_samples]
val_data = [(month_obs(s), *query_pool(s)) for s in val_samples]

results = {}
for variant in ("perceiver", "resampler", "mbca"):
    model = build_fusion_model(variant, grid, d_model=args.d_model,
                               n_latent=args.n_latent, n_heads=4,
                               n_self_blocks=4, seed=args.seed).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    srng = np.random.default_rng(args.seed)
    tv = time.time()
    losses = []
    nan_seen = False
    model.train()
    for step in range(1, args.steps + 1):
        obs, q, y = train_data[srng.integers(len(train_data))]
        take = srng.choice(q.shape[0], size=min(args.queries, q.shape[0]),
                           replace=False)
        qt = torch.as_tensor(q[take], device=dev)[None]
        yt = torch.as_tensor(y[take], device=dev)[None]
        opt.zero_grad()
        out = model(obs, qt)
        loss = ((out - yt) ** 2).mean()
        if not torch.isfinite(loss):
            nan_seen = True
            break
        loss.backward(); opt.step()
        losses.append(float(loss))
        if step % max(1, args.steps // 8) == 0:
            print(f"  [{variant:9s}] step {step:4d} loss {np.mean(losses[-20:]):.4f}",
                  flush=True)
    model.eval()

    @torch.no_grad()
    def eval_rmse(obs, q, y, chunk=65536):
        """Physical-unit anomaly RMSE per var over the full query pool."""
        se = np.zeros(2); n = np.zeros(2)
        for i in range(0, q.shape[0], chunk):
            qt = torch.as_tensor(q[i:i + chunk], device=dev)[None]
            out = model(obs, qt)[0].float().cpu().numpy()
            d_idx = np.abs(grid.depth[None, :]
                           - q[i:i + chunk, 2][:, None]).argmin(1)
            for k, v in enumerate(VARS):
                sd = norm.astd[v][d_idx]
                err = (out[:, k] - y[i:i + chunk, k]) * sd
                se[k] += float((err ** 2).sum()); n[k] += err.size
        return {v: float(np.sqrt(se[k] / n[k])) for k, v in enumerate(VARS)}

    @torch.no_grad()
    def rel_change(obs_a, obs_b, q, sub=20000):
        take = np.random.default_rng(0).choice(
            q.shape[0], size=min(sub, q.shape[0]), replace=False)
        qt = torch.as_tensor(q[take], device=dev)[None]
        ya = model(obs_a, qt); yb = model(obs_b, qt)
        return float((ya - yb).norm() / (ya.norm() + 1e-12))

    # -------- probes on the validation month --------
    obs_v, q_v, y_v = val_data[0]
    train_rmse = eval_rmse(*train_data[0])
    val_rmse = eval_rmse(obs_v, q_v, y_v)

    # (1) duplicate half the profiles (no new information)
    pv = obs_v["profiles"]
    Phalf = pv["prof"].shape[1] // 2
    obs_dup = dict(obs_v)
    obs_dup["profiles"] = {
        "prof": torch.cat([pv["prof"], pv["prof"][:, :Phalf]], 1),
        "lat": torch.cat([pv["lat"], pv["lat"][:, :Phalf]], 1),
        "lon": torch.cat([pv["lon"], pv["lon"][:, :Phalf]], 1),
        "month": pv["month"]}
    dup = rel_change(obs_v, obs_dup, q_v)

    # (2) same WOA/surf fields at 2x resolution through the same encoder
    obs_ref = dict(obs_v)
    for key in ("surf", "woa"):
        o = dict(obs_v[key])
        f = o["field"]
        o["field"] = torch.repeat_interleave(
            torch.repeat_interleave(f, 2, dim=-2), 2, dim=-1)
        o["lat"] = torch.repeat_interleave(o["lat"], 2)
        o["lon"] = torch.repeat_interleave(o["lon"], 2)
        obs_ref[key] = o
    refine = rel_change(obs_v, obs_ref, q_v)

    # (3) same profiles, 2x vertical sampling (linear interp, ragged path)
    dsub = grid.depth
    dmid = (dsub[:-1] + dsub[1:]) / 2
    d2 = np.sort(np.concatenate([dsub, dmid])).astype("float32")
    prof = pv["prof"].cpu().numpy()                      # (1,P,2,D)
    P = prof.shape[1]
    prof2 = np.stack([[np.interp(d2, dsub, prof[0, p, c])
                       for c in range(2)] for p in range(P)], 0)[None]
    obs_rs = dict(obs_v)
    obs_rs["profiles"] = {
        "prof": torch.as_tensor(prof2.astype("float32"), device=dev),
        "lat": pv["lat"], "lon": pv["lon"], "month": pv["month"],
        "depths": torch.as_tensor(d2, device=dev)}
    resamp = rel_change(obs_v, obs_rs, q_v)

    # (4) missing-modality batches run (functionality gate)
    with torch.no_grad():
        for drop in ("surf", "woa", "profiles"):
            o = {k: v for k, v in obs_v.items() if k != drop}
            qt = torch.as_tensor(q_v[:64], device=dev)[None]
            assert torch.isfinite(model(o, qt)).all(), f"NaN with {drop} missing"

    results[variant] = {
        "n_params": n_params, "train_secs": round(time.time() - tv, 1),
        "nan_seen": nan_seen, "final_train_loss": float(np.mean(losses[-20:])),
        "train_rmse": train_rmse, "val_rmse": val_rmse,
        "dup_sensitivity": dup, "refine_sensitivity": refine,
        "profile_resample_sensitivity": resamp,
    }
    print(f"  [{variant:9s}] train TEMP={train_rmse['TEMP']:.4f} "
          f"val TEMP={val_rmse['TEMP']:.4f} SALT={val_rmse['SALT']:.4f} | "
          f"dup={dup:.4f} refine={refine:.4f} resamp={resamp:.4f}", flush=True)

out_path = os.path.join(C.CACHE, "poc_ocean.json")
with open(out_path, "w") as f:
    json.dump({"config": vars(args),
               "train_months": tr_idx.tolist(), "val_month": va_idx.tolist(),
               "results": results}, f, indent=2)
print(f"\nDONE in {(time.time()-t0)/60:.1f} min -> {out_path}", flush=True)

print("\n| model | train TEMP | val TEMP | val SALT | dup | patch-refine | prof-resample |")
print("|---|---|---|---|---|---|---|")
for v, r in results.items():
    print(f"| {v} | {r['train_rmse']['TEMP']:.4f} | {r['val_rmse']['TEMP']:.4f} "
          f"| {r['val_rmse']['SALT']:.4f} | {r['dup_sensitivity']:.4f} "
          f"| {r['refine_sensitivity']:.4f} | {r['profile_resample_sensitivity']:.4f} |")
