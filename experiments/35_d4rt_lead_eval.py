"""D4RT lead evaluation — unobserved-only RMSE at leads 0..3 (spec §2.1, §6).

Protocol mapping, which is an interpretation of mentor §3.7 and not a
transcription of it: protocol_v1's test set is 12 *scattered* pinned months,
not a contiguous era, so the rule "windows must not cross a split boundary"
is applied as

  * t_src IS the pinned test month (so the observation draw is identical to
    the protocol_v1 headline run and lead 0 reproduces it exactly);
  * targets are t_src + delta for delta in 0..3;
  * every t_src-1 (context) and t_src+delta (target) must fall strictly after
    the validation period ends — asserted at run time, not assumed.

Because the unobserved-cell mask is fixed by the t_src profile draw, all four
leads are scored on identical cells and are therefore directly comparable.

Baselines:
  * persistence  — the model's own lead-0 reconstruction reused as the
                   forecast for every lead;
  * climatology  — zero anomaly, floor recomputed against each target month.

WARNING: the mentor document's result tables are placeholders (never run).
Numbers produced here are fresh measurements and must not be merged with or
compared against them.

  CUDA_VISIBLE_DEVICES=6 .venv/bin/python experiments/35_d4rt_lead_eval.py \
      --ckpt outputs/ckpt/d4rt_s1234.pt
"""
import sys, os, json, time, argparse, subprocess
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
MAX_LEAD = 3
CONTEXT = 2

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--n-profiles", type=int, default=1500)
ap.add_argument("--chunk", type=int, default=32768)
ap.add_argument("--tag", default="")
ap.add_argument("--limit-months", type=int, default=0)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
cargs = ck["args"]
tag = args.tag or f"d4rt_lead_eval_s{args.seed}"
print(f"d4rt-lead-eval ckpt={args.ckpt} step={ck['step']} device={dev}",
      flush=True)

# ------------------------------------------------------- split arithmetic
grid = data.CommonGrid()
tr_idx = data.select_month_indices(C.GT_SOURCE, TRAIN_YEARS)
va_idx = data.select_month_indices(C.GT_SOURCE, VAL_YEARS)
val_end = int(va_idx.max())

src = np.asarray(TEST_TIME_INDICES)
if args.limit_months:
    src = src[:args.limit_months]
needed = sorted(set(src.tolist())
                | {int(s) - b for s in src for b in range(1, CONTEXT)}
                | {int(s) + d for s in src for d in range(MAX_LEAD + 1)})
bad_ctx = [int(s) - 1 for s in src if int(s) - 1 <= val_end]
bad_tgt = [int(s) + d for s in src for d in range(MAX_LEAD + 1)
           if int(s) + d <= val_end]
if bad_ctx or bad_tgt:
    raise SystemExit(
        f"split-boundary violation (spec §2.1): validation ends at index "
        f"{val_end}; context months {bad_ctx} / target months {bad_tgt} "
        f"are not strictly after it.")
print(f"val ends at index {val_end}; {len(src)} source months, "
      f"{len(needed)} months loaded (context + targets)", flush=True)

# ------------------------------------------------------------------ data
ftrain = data.load_gt_fields(tr_idx, grid)
surf_train = {v: ftrain[v] for v in C.VARS_SURF if v in ftrain}
clim = Climatology(ftrain, surf_train)
norm = AnomNorm(clim, ftrain, surf_train)          # train-only, protocol_v1
del ftrain

need_idx = np.asarray(needed)
fev = data.load_gt_fields(need_idx, grid)
woa = data.woa_prior(grid)
rd = FullRunData(grid, norm, dev)
D, H, W, HW = rd.D, rd.H, rd.W, rd.HW
ZA = torch.from_numpy(rd.z_volume(fev)).to(dev)
ZAf = ZA.view(len(need_idx), 2, D * HW)
surfZ = torch.from_numpy(rd.z_surf(fev)).to(dev)
ev_months = fev["months"].copy()
rd.load_woa(woa)
pos = {int(v): i for i, v in enumerate(need_idx)}      # zarr index -> row
del fev

# ----------------------------------------------------------------- model
model = build_fusion_model("d4rt", grid, d_model=cargs["d_model"],
                           n_latent=cargs["n_latent"],
                           n_heads=cargs["n_heads"],
                           n_self_blocks=cargs["n_self_blocks"],
                           n_dec_blocks=cargs["n_dec_blocks"],
                           n_ref_slots=cargs["n_ref_slots"],
                           max_lead=MAX_LEAD, seed=args.seed).to(dev)
model.load_state_dict(ck["state_dict"])
model.eval()

# ------------------------------------------------------------------ eval
rng = np.random.default_rng([args.seed, 2])
se = {l: torch.zeros(D, 2, dtype=torch.float64, device=dev)
      for l in range(MAX_LEAD + 1)}
se_pers = {l: torch.zeros(D, 2, dtype=torch.float64, device=dev)
           for l in range(MAX_LEAD + 1)}
se_clim = {l: torch.zeros(D, 2, dtype=torch.float64, device=dev)
           for l in range(MAX_LEAD + 1)}
n_lev = {l: torch.zeros(D, dtype=torch.float64, device=dev)
         for l in range(MAX_LEAD + 1)}
t0 = time.time()

for s in src:
    ti = pos[int(s)]
    mo = int(ev_months[ti])
    k = min(args.n_profiles, rd.n_ocean)
    pick = rng.choice(rd.n_ocean, size=k, replace=False)
    ii_t = torch.from_numpy(rd.oi[pick]).to(dev)
    jj_t = torch.from_numpy(rd.oj[pick]).to(dev)
    col = torch.zeros(HW, dtype=torch.bool, device=dev)
    col[ii_t * W + jj_t] = True

    obs = rd.obs_dict(ZA, surfZ, ti, mo, ii_t, jj_t, context=CONTEXT)
    with torch.no_grad():
        tokens = model.encode(obs, batch=1, device=dev)
        latent = model.fuse(tokens)

    pred0_cache = {}
    for lead in range(MAX_LEAD + 1):
        tj = pos[int(s) + lead]
        mo_t = int(ev_months[tj])
        # cells unobserved at t_src and finite in the target month
        fin = torch.isfinite(ZA[tj]).all(0).view(D, HW)
        keep = fin & (~col)[None]
        idx_t = keep.view(-1).nonzero(as_tuple=True)[0]
        y = ZAf[tj][:, idx_t].T
        q, di = rd.q_from_flat(idx_t, mo_t)
        lead_t = torch.full(q.shape[:2], lead, dtype=torch.long, device=dev)

        preds = []
        with torch.no_grad():
            for i in range(0, q.shape[1], args.chunk):
                preds.append(model.decode(latent, q[:, i:i + args.chunk],
                                          lead=lead_t[:, i:i + args.chunk])[0])
        pred = torch.cat(preds, dim=0)

        # persistence: the lead-0 reconstruction at these same cells, reused.
        # cells differ per lead only through target finiteness, so recompute
        # the lead-0 prediction on THIS lead's cell set.
        p0 = []
        with torch.no_grad():
            z0 = torch.zeros_like(lead_t)
            for i in range(0, q.shape[1], args.chunk):
                p0.append(model.decode(latent, q[:, i:i + args.chunk],
                                       lead=z0[:, i:i + args.chunk])[0])
        pred_pers = torch.cat(p0, dim=0)

        n_lev[lead].index_add_(0, di, torch.ones_like(di, dtype=torch.float64))
        se[lead].index_add_(0, di, (pred - y).double() ** 2)
        se_pers[lead].index_add_(0, di, (pred_pers - y).double() ** 2)
        se_clim[lead].index_add_(0, di, y.double() ** 2)
    print(f"  source {int(s)} done ({time.time()-t0:.0f}s)", flush=True)

rows = {}
for lead in range(MAX_LEAD + 1):
    rows[lead] = {
        "model": rd.physical_rmse(se[lead], n_lev[lead])["full"],
        "persistence": rd.physical_rmse(se_pers[lead], n_lev[lead])["full"],
        "climatology": rd.physical_rmse(se_clim[lead], n_lev[lead])["full"],
        "n_queries": int(n_lev[lead].sum()),
    }

print("\nlead |      model TEMP/SALT |  persistence TEMP/SALT | "
      "climatology TEMP/SALT")
for lead, r in rows.items():
    print(f"  {lead}  | {r['model']['TEMP']:.4f} / {r['model']['SALT']:.4f} "
          f"     | {r['persistence']['TEMP']:.4f} / {r['persistence']['SALT']:.4f} "
          f"      | {r['climatology']['TEMP']:.4f} / {r['climatology']['SALT']:.4f}")


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


os.makedirs(C.CACHE, exist_ok=True)
out = os.path.join(C.CACHE, f"{tag}.json")
json.dump({"tag": tag, "task": "d4rt_lead_eval", "protocol": "protocol_v1",
           "git_commit": git_commit(), "ckpt": args.ckpt,
           "ckpt_step": ck["step"], "seed": args.seed,
           "n_profiles": args.n_profiles, "source_months": src.tolist(),
           "val_end_index": val_end, "max_lead": MAX_LEAD,
           "context_months": CONTEXT, "results": rows,
           "note": ("Mentor doc tables are placeholders (never run); these are "
                    "fresh measurements and must not be compared to them."),
           "wall_s": round(time.time() - t0, 1)},
          open(out, "w"), indent=1)
print(f"\nsaved {out}", flush=True)
