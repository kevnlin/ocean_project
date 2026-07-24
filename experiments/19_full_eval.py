"""Task 8b (Week 4) — post-training evaluation of one full-run checkpoint.

For a completed experiments/18_full_train.py run (its best-validation
checkpoint), produces the remaining two report items of
reports/full_training_plan.md:

  (2) Task-6-style sensitivity probes on REAL data — duplicate half the
      profiles / feed the same grids at 2x resolution / resample profiles to
      2x vertical levels (no new information in any of them) — relative
      output change + probe RMSE, on validation months (never test).
  (3) The flexibility axis with ONE checkpoint, no retraining:
      profile-count sweep on the pinned test months (same density axis as the
      week-2 retrained-baseline ablation) and the missing-modality matrix
      (headline masks kept fixed; only the model's inputs are dropped).

Writes outputs/cache/full_eval_<tag>.json.

Run:  CUDA_VISIBLE_DEVICES=7 python experiments/19_full_eval.py --tag full_mbca_s1234
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from ocean_tokenizer import data, config as C
from ocean_tokenizer.anomaly import Climatology, AnomNorm
from ocean_tokenizer.fusion import build_fusion_model
from ocean_tokenizer.fullrun import FullRunData, VARS

TRAIN_YEARS = (1985, 2007)
VAL_YEARS = (2008, 2010)
TEST_TIME_INDICES = [1933, 1935, 1936, 1938, 1942, 1946, 1952, 1953,
                     1956, 1965, 1967, 1976]

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True, help="run tag, e.g. full_mbca_s1234")
ap.add_argument("--densities", default="0,50,150,375,750,1500,3000",
                help="profile-count sweep (week-2 density-ablation axis)")
ap.add_argument("--probe-months", type=int, default=3,
                help="validation months used for the sensitivity probes")
ap.add_argument("--probe-queries", type=int, default=20000)
args = ap.parse_args()
DENSITIES = [int(x) for x in args.densities.split(",")]

t0 = time.time()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ckpt_path = os.path.join(C.CKPT, f"{args.tag}.pt")
ckpt = torch.load(ckpt_path, map_location=dev)
run_args = ckpt["args"]
variant, seed = ckpt["variant"], run_args["seed"]
print(f"eval tag={args.tag} variant={variant} seed={seed} "
      f"(ckpt step {ckpt['step']})", flush=True)

# ---- data (identical to the trainer) ----
grid = data.CommonGrid()
tr_idx = data.select_month_indices(C.GT_SOURCE, TRAIN_YEARS)
va_idx = data.select_month_indices(C.GT_SOURCE, VAL_YEARS)
te_idx = np.asarray(TEST_TIME_INDICES)
print("loading fields ...", flush=True); ts = time.time()
ftrain = data.load_gt_fields(tr_idx, grid)
fval = data.load_gt_fields(va_idx, grid)
ftest = data.load_gt_fields(te_idx, grid)
woa = data.woa_prior(grid)
print(f"  loaded in {time.time()-ts:.1f}s", flush=True)
surf_train = {v: ftrain[v] for v in C.VARS_SURF if v in ftrain}
clim = Climatology(ftrain, surf_train)
norm = AnomNorm(clim, ftrain, surf_train)
del ftrain

rd = FullRunData(grid, norm, dev)
rd.load_woa(woa)
n_eval = run_args.get("n_profiles_eval", C.N_PROFILES)

anchor = (tuple(int(x) for x in run_args["anchor_grid"].split(","))
          if run_args.get("anchor_grid") else None)
model = build_fusion_model(variant, grid, d_model=run_args["d_model"],
                           n_latent=run_args["n_latent"],
                           n_heads=run_args["n_heads"],
                           n_self_blocks=run_args["n_self_blocks"],
                           seed=seed, anchor_grid=anchor).to(dev)
model.load_state_dict(ckpt["state_dict"])
model.eval()

out = {"tag": args.tag, "variant": variant, "seed": seed,
       "ckpt_step": ckpt["step"], "protocol": "protocol_v1",
       "n_profiles_eval": n_eval}

# ======================================================================
# (2) Sensitivity probes on validation months (Task-6 protocol, real data)
# ======================================================================
print("sensitivity probes ...", flush=True)
fval_small = {k: (v[:args.probe_months] if hasattr(v, "__len__")
                  and len(v) == len(fval["months"]) else v)
              for k, v in fval.items()}
fval_small["months"] = fval["months"][:args.probe_months]
probe_packs, _, _ = rd.make_packs(fval_small,
                                  np.random.default_rng([seed, 1]),
                                  n_eval, quiet=True)


def dup_half(obs):
    o = {k: dict(v) for k, v in obs.items()}
    pv = o["profiles"]
    Ph = pv["prof"].shape[1] // 2
    o["profiles"] = {
        "prof": torch.cat([pv["prof"], pv["prof"][:, :Ph]], 1),
        "lat": torch.cat([pv["lat"], pv["lat"][:, :Ph]], 1),
        "lon": torch.cat([pv["lon"], pv["lon"][:, :Ph]], 1),
        "month": pv["month"]}
    return o


def refine2x(obs):
    o = {k: dict(v) for k, v in obs.items()}
    for key in ("surf", "woa"):
        g = dict(o[key])
        g["field"] = torch.repeat_interleave(
            torch.repeat_interleave(g["field"], 2, dim=-2), 2, dim=-1)
        g["lat"] = torch.repeat_interleave(g["lat"], 2)
        g["lon"] = torch.repeat_interleave(g["lon"], 2)
        o[key] = g
    return o


def resample2x(obs):
    o = {k: dict(v) for k, v in obs.items()}
    pv = o["profiles"]
    dsub = grid.depth
    dmid = (dsub[:-1] + dsub[1:]) / 2
    d2 = np.sort(np.concatenate([dsub, dmid])).astype("float32")
    prof = pv["prof"][0].cpu().numpy()                      # (P,2,D)
    P = prof.shape[0]
    prof2 = np.stack([[np.interp(d2, dsub, prof[p, c]) for c in range(2)]
                      for p in range(P)], 0)[None] if P else \
        np.zeros((1, 0, 2, d2.size), "float32")
    o["profiles"] = {
        "prof": torch.as_tensor(prof2.astype("float32"), device=dev),
        "lat": pv["lat"], "lon": pv["lon"], "month": pv["month"],
        "depths": torch.as_tensor(d2, device=dev)}
    return o


@torch.no_grad()
def probe(pack, manip):
    qrng = np.random.default_rng(0)
    Q = pack["q"].shape[1]
    take = torch.from_numpy(qrng.choice(Q, size=min(args.probe_queries, Q),
                                        replace=False)).to(dev)
    q = pack["q"][:, take]
    y = pack["y"][take]
    di = pack["di"][take]
    za = model.fuse(model.encode(pack["obs"], batch=1, device=dev))
    zb = model.fuse(model.encode(manip(pack["obs"]), batch=1, device=dev))
    ya, yb = model.decode(za, q), model.decode(zb, q)
    rel = float((ya - yb).norm() / (ya.norm() + 1e-12))

    def rmse_of(pred):
        se = torch.zeros(rd.D, 2, dtype=torch.float64, device=dev)
        se.index_add_(0, di, (pred[0] - y).double() ** 2)
        n = torch.bincount(di, minlength=rd.D).double()
        return rd.physical_rmse(se, n)["full"]
    return rel, rmse_of(ya), rmse_of(yb)


probes = {}
for name, manip in (("duplicate_half", dup_half), ("patch_refine_2x", refine2x),
                    ("profile_resample_2x", resample2x)):
    rels, r_bases, r_mans = [], [], []
    for pack in probe_packs:
        rel, r_base, r_man = probe(pack, manip)
        rels.append(rel); r_bases.append(r_base); r_mans.append(r_man)
    probes[name] = {
        "rel_output_change_mean": float(np.mean(rels)),
        "rel_output_change_per_month": rels,
        "base_rmse_TEMP": float(np.mean([r["TEMP"] for r in r_bases])),
        "probe_rmse_TEMP": float(np.mean([r["TEMP"] for r in r_mans])),
        "base_rmse_SALT": float(np.mean([r["SALT"] for r in r_bases])),
        "probe_rmse_SALT": float(np.mean([r["SALT"] for r in r_mans])),
    }
    print(f"  {name:22s} rel={probes[name]['rel_output_change_mean']:.5f}  "
          f"TEMP {probes[name]['base_rmse_TEMP']:.4f}->"
          f"{probes[name]['probe_rmse_TEMP']:.4f}", flush=True)
out["probes"] = probes
del probe_packs, fval, fval_small

# ======================================================================
# (3a) One-checkpoint profile-count sweep on the pinned test months
# ======================================================================
print("profile-count sweep (one checkpoint, no retraining) ...", flush=True)
sweep = []
for density in DENSITIES:
    packs, n_level, se0 = rd.make_packs(
        ftest, np.random.default_rng([seed, 3, density]), density, quiet=True)
    res = rd.physical_rmse(rd.eval_packs(model, packs), n_level)
    floor = rd.physical_rmse(se0, n_level)["full"]
    row = {"density": density,
           "TEMP": res["full"]["TEMP"], "SALT": res["full"]["SALT"],
           "floor_TEMP": floor["TEMP"], "floor_SALT": floor["SALT"],
           "by_band": res["by_band"]}
    sweep.append(row)
    print(f"  density {density:5d}: TEMP={row['TEMP']:.4f} "
          f"SALT={row['SALT']:.4f} (floor {floor['TEMP']:.4f})", flush=True)
    del packs
out["count_sweep"] = sweep

# ======================================================================
# (3b) Missing-modality matrix (headline masks fixed; inputs dropped)
# ======================================================================
print("missing-modality matrix ...", flush=True)
packs, n_level, se0 = rd.make_packs(
    ftest, np.random.default_rng([seed, 2]), n_eval, quiet=True)
out["modality_floor"] = rd.physical_rmse(se0, n_level)["full"]
matrix = {}
for row_name, drop in (("full", None), ("drop_surf", "surf"),
                       ("drop_woa", "woa"), ("drop_profiles", "profiles")):
    ov = (None if drop is None
          else (lambda p, d=drop: {k: v for k, v in p["obs"].items() if k != d}))
    res = rd.physical_rmse(rd.eval_packs(model, packs, obs_override=ov),
                           n_level)
    matrix[row_name] = {"TEMP": res["full"]["TEMP"],
                        "SALT": res["full"]["SALT"],
                        "by_band": res["by_band"]}
    print(f"  {row_name:14s}: TEMP={matrix[row_name]['TEMP']:.4f} "
          f"SALT={matrix[row_name]['SALT']:.4f}", flush=True)
out["modality_matrix"] = matrix

path = os.path.join(C.CACHE, f"full_eval_{args.tag}.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print(f"DONE in {(time.time()-t0)/60:.1f} min -> {path}", flush=True)
