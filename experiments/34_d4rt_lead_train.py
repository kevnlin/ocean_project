"""D4RT causal space-time query decoder — training (spec §2.3/§2.4 on protocol_v1).

Spec: docs/superpowers/specs/2026-08-16-d4rt-query-decoder-design.md
Mentor source: dfs_d4rt_intern_plan.md §2.3, §2.4.

WARNING about the mentor document: it is written in the past tense as a
finished audit but was never run.  Its result tables, checkpoint hashes,
manifest SHA and "106 passed" are targets, not measurements.  Nothing here
reproduces them and nothing here may be compared against them.

What this run does, per step:

  * draw a source month t_src from the eligible train months (eligible =
    t_src-1 and t_src+MAX_LEAD both inside the train split, mentor §3.7);
  * profile-count augmentation K ~ U{0..3000} at t_src only;
  * surface context over [t_src-1, t_src], each month tagged with its own
    time_offset;
  * sample a lead: 70 % delta=0 (reconstruction) and 30 % delta in {1,2,3}
    (forecast), the mentor §6.1 mix;
  * queries at unobserved ocean cells, scored against the truth at t_src+delta;
  * masked MSE in anomaly z-space.

Smoke:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python experiments/34_d4rt_lead_train.py --smoke
"""
import sys, os, json, time, argparse, subprocess, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from ocean_tokenizer import data, config as C
from ocean_tokenizer.anomaly import Climatology, AnomNorm
from ocean_tokenizer.fusion import build_fusion_model
from ocean_tokenizer.fullrun import (FullRunData, VARS,
                                       assert_nondegenerate_climatology)

TRAIN_YEARS = (1985, 2007)          # 276 months  (protocol_v1)
VAL_YEARS = (2008, 2010)            # 36 months
MAX_LEAD = 3
CONTEXT = 2                         # [t_src-1, t_src]  (mentor §2.1)
RECON_FRAC = 0.70                   # 70 % lead-0 / 30 % lead 1..3

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--steps", type=int, default=50_000)
ap.add_argument("--queries", type=int, default=8192)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--lr-min", type=float, default=1e-5)
ap.add_argument("--warmup", type=int, default=500)
# mentor §2.3 dimensions, verbatim
ap.add_argument("--d-model", type=int, default=64)
ap.add_argument("--n-latent", type=int, default=32)
ap.add_argument("--n-heads", type=int, default=4)
ap.add_argument("--n-self-blocks", type=int, default=2)
ap.add_argument("--n-dec-blocks", type=int, default=2)
ap.add_argument("--n-ref-slots", type=int, default=8)
ap.add_argument("--aug-min", type=int, default=0)
ap.add_argument("--aug-max", type=int, default=3000)
ap.add_argument("--log-every", type=int, default=100)
ap.add_argument("--val-every", type=int, default=2000)
ap.add_argument("--val-queries", type=int, default=2048,
                help="query subsample per (val month, lead)")
ap.add_argument("--n-profiles-eval", type=int, default=1500)
ap.add_argument("--patience", type=int, default=8,
                help="validations without improvement before stopping")
ap.add_argument("--min-steps", type=int, default=10_000,
                help="no early stop before this step: the loss leaves its "
                     "initial plateau only after a few thousand steps")
ap.add_argument("--tag", default="")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--limit-train", type=int, default=0)
args = ap.parse_args()

tag = args.tag or f"d4rt_s{args.seed}"
if args.smoke:
    args.steps, args.log_every, args.queries = 200, 20, 1024
    args.limit_train = 24
    args.val_every, args.val_queries, args.min_steps = 100, 256, 0
    args.n_profiles_eval = 200

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print(f"d4rt-train tag={tag} seed={args.seed} steps={args.steps} "
      f"device={dev} smoke={args.smoke}", flush=True)

# ---------------------------------------------------------------- data
grid = data.CommonGrid()
tr_idx = data.select_month_indices(C.GT_SOURCE, TRAIN_YEARS)
va_idx = data.select_month_indices(C.GT_SOURCE, VAL_YEARS)
if args.limit_train:
    tr_idx = tr_idx[:args.limit_train]
    va_idx = va_idx[:max(args.limit_train // 2, MAX_LEAD + CONTEXT)]
print(f"train months={tr_idx.size} val months={va_idx.size}", flush=True)

ts = time.time()
ftrain = data.load_gt_fields(tr_idx, grid)
woa = data.woa_prior(grid)
surf_train = {v: ftrain[v] for v in C.VARS_SURF if v in ftrain}
assert_nondegenerate_climatology(ftrain["months"])
clim = Climatology(ftrain, surf_train)
norm = AnomNorm(clim, ftrain, surf_train)
rd = FullRunData(grid, norm, dev)
D, H, W, HW = rd.D, rd.H, rd.W, rd.HW
oi, oj, n_ocean = rd.oi, rd.oj, rd.n_ocean

ZA = torch.from_numpy(rd.z_volume(ftrain)).to(dev)              # (T,2,D,H,W)
ZAf = ZA.view(len(tr_idx), 2, D * HW)
surfZ = torch.from_numpy(rd.z_surf(ftrain)).to(dev)
months = ftrain["months"].copy()
rd.load_woa(woa)
valid = [torch.isfinite(ZA[t]).all(0).view(-1).nonzero(as_tuple=True)[0]
         .cpu().numpy().astype("int64") for t in range(len(tr_idx))]
del ftrain
print(f"  data ready in {time.time()-ts:.1f}s", flush=True)

# eligible source months: the whole [t-1, t+MAX_LEAD] window must stay inside
# the split (mentor §3.7 — windows never cross a split boundary)
T = len(tr_idx)
eligible = np.arange(CONTEXT - 1, T - MAX_LEAD)
if eligible.size == 0:
    raise SystemExit(f"no eligible source months for T={T}")
print(f"eligible source months: {eligible.size} of {T}", flush=True)

# ------------------------------------------------------- validation packs
# protocol_v1 selects checkpoints on validation only.  Score is the mentor
# §6.2 rule: mean over lead 0..3 and variable {T,S} of RMSE / climatology RMSE
# — eight ratios, so no single lead or variable can dominate the choice.
fval = data.load_gt_fields(va_idx, grid)
ZAv = torch.from_numpy(rd.z_volume(fval)).to(dev)
ZAvf = ZAv.view(len(va_idx), 2, D * HW)
surfZv = torch.from_numpy(rd.z_surf(fval)).to(dev)
v_months = fval["months"].copy()
del fval
Tv = len(va_idx)
v_eligible = np.arange(CONTEXT - 1, Tv - MAX_LEAD)
val_rng = np.random.default_rng([args.seed, 1])
val_packs = []
for t in v_eligible:
    t = int(t)
    kk = min(args.n_profiles_eval, n_ocean)
    pk = val_rng.choice(n_ocean, size=kk, replace=False)
    vii = torch.from_numpy(oi[pk]).to(dev)
    vjj = torch.from_numpy(oj[pk]).to(dev)
    vcol = torch.zeros(HW, dtype=torch.bool, device=dev)
    vcol[vii * W + vjj] = True
    per_lead = []
    for l in range(MAX_LEAD + 1):
        tj = t + l
        fin = torch.isfinite(ZAv[tj]).all(0).view(D, HW)
        idx = (fin & (~vcol)[None]).view(-1).nonzero(as_tuple=True)[0]
        if idx.numel() > args.val_queries:
            sel = val_rng.choice(idx.numel(), size=args.val_queries,
                                 replace=False)
            idx = idx[torch.from_numpy(np.sort(sel)).to(dev)]
        qq, dd = rd.q_from_flat(idx, int(v_months[tj]))
        per_lead.append(dict(q=qq, y=ZAvf[tj][:, idx].T, di=dd, lead=l))
    val_packs.append(dict(t=t, mo=int(v_months[t]), ii=vii, jj=vjj,
                          leads=per_lead))
# climatology floor per lead (predicting zero anomaly)
val_floor = {}
for l in range(MAX_LEAD + 1):
    se0 = torch.zeros(D, 2, dtype=torch.float64, device=dev)
    nl = torch.zeros(D, dtype=torch.float64, device=dev)
    for p_ in val_packs:
        pl = p_["leads"][l]
        nl.index_add_(0, pl["di"], torch.ones_like(pl["di"], dtype=torch.float64))
        se0.index_add_(0, pl["di"], pl["y"].double() ** 2)
    val_floor[l] = rd.physical_rmse(se0, nl)["full"]
print(f"val: {len(val_packs)} eligible source months of {Tv}; floors "
      + " ".join(f"L{l} {val_floor[l]['TEMP']:.3f}/{val_floor[l]['SALT']:.3f}"
                 for l in range(MAX_LEAD + 1)), flush=True)


@torch.no_grad()
def validate():
    model.eval()
    se = {l: torch.zeros(D, 2, dtype=torch.float64, device=dev)
          for l in range(MAX_LEAD + 1)}
    nl = {l: torch.zeros(D, dtype=torch.float64, device=dev)
          for l in range(MAX_LEAD + 1)}
    for p_ in val_packs:
        obs_v = rd.obs_dict(ZAv, surfZv, p_["t"], p_["mo"], p_["ii"], p_["jj"],
                            context=CONTEXT)
        z = model.fuse(model.encode(obs_v, batch=1, device=dev))
        for pl in p_["leads"]:
            lt = torch.full(pl["q"].shape[:2], pl["lead"], dtype=torch.long,
                            device=dev)
            out = model.decode(z, pl["q"], lead=lt)[0]
            nl[pl["lead"]].index_add_(0, pl["di"],
                                      torch.ones_like(pl["di"], dtype=torch.float64))
            se[pl["lead"]].index_add_(0, pl["di"], (out - pl["y"]).double() ** 2)
    model.train()
    per_lead, ratios = {}, []
    for l in range(MAX_LEAD + 1):
        r = rd.physical_rmse(se[l], nl[l])["full"]
        per_lead[l] = r
        for v in ("TEMP", "SALT"):
            ratios.append(r[v] / max(val_floor[l][v], 1e-12))
    return float(np.mean(ratios)), per_lead


# ---------------------------------------------------------------- model
model = build_fusion_model("d4rt", grid, d_model=args.d_model,
                           n_latent=args.n_latent, n_heads=args.n_heads,
                           n_self_blocks=args.n_self_blocks,
                           n_dec_blocks=args.n_dec_blocks,
                           n_ref_slots=args.n_ref_slots,
                           max_lead=MAX_LEAD, seed=args.seed).to(dev)
n_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.Adam(model.parameters(), lr=args.lr)


def _lr_lambda(step):
    if args.warmup and step < args.warmup:
        return (step + 1) / args.warmup
    p = min(max((step - args.warmup) / max(args.steps - args.warmup, 1), 0.0), 1.0)
    lo = args.lr_min / args.lr
    return lo + 0.5 * (1 - lo) * (1 + math.cos(math.pi * p))


sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
print(f"model d4rt: params={n_params:,} d={args.d_model} L={model.n_latent} "
      f"dec_blocks={args.n_dec_blocks} ref_slots={args.n_ref_slots}", flush=True)


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- train
os.makedirs(C.CKPT, exist_ok=True)
os.makedirs(C.CACHE, exist_ok=True)
ckpt = os.path.join(C.CKPT, f"{tag}.pt")
rng = np.random.default_rng([args.seed, 0])
curves = {"step": [], "loss": [], "loss_lead0": [], "loss_leadN": [], "lr": [],
          "val_step": [], "val_score": [], "val_per_lead": []}
best = {"score": float("inf"), "step": -1, "state": None}
since_best = 0
win, win0, winN = [], [], []
t0 = time.time()

for step in range(1, args.steps + 1):
    t = int(rng.choice(eligible))
    mo = int(months[t])
    K = int(rng.integers(args.aug_min, args.aug_max + 1))
    if K:
        pick = rng.choice(n_ocean, size=min(K, n_ocean), replace=False)
        ii_t = torch.from_numpy(oi[pick]).to(dev)
        jj_t = torch.from_numpy(oj[pick]).to(dev)
        col = torch.zeros(HW, dtype=torch.bool, device=dev)
        col[ii_t * W + jj_t] = True
    else:
        ii_t = torch.zeros(0, dtype=torch.long, device=dev)
        jj_t = torch.zeros(0, dtype=torch.long, device=dev)
        col = torch.zeros(HW, dtype=torch.bool, device=dev)

    # lead for this step: 70 % reconstruction, 30 % forecast
    lead = 0 if rng.random() < RECON_FRAC else int(rng.integers(1, MAX_LEAD + 1))
    t_tgt = t + lead
    mo_tgt = int(months[t_tgt])

    # queries at cells unobserved in the SOURCE month (gate 10: lead-0 targets
    # exclude the supplied profile columns, so a point copy cannot score), and
    # finite in the TARGET month
    pool = valid[t_tgt]
    over = int(args.queries * (1.0 + 1.3 * K / max(n_ocean, 1)) + 64)
    cand = pool[rng.choice(pool.size, size=min(over, pool.size), replace=False)]
    idx_t = torch.from_numpy(cand).to(dev)
    idx_t = idx_t[~col[idx_t % HW]][:args.queries]
    if idx_t.numel() == 0:
        continue

    y = ZAf[t_tgt][:, idx_t].T                                   # (Q,2)
    q, _ = rd.q_from_flat(idx_t, mo_tgt)                         # target month
    lead_t = torch.full(q.shape[:2], lead, dtype=torch.long, device=dev)

    obs = rd.obs_dict(ZA, surfZ, t, mo, ii_t, jj_t, context=CONTEXT)
    out = model(obs, q, lead=lead_t)[0]
    loss = (out - y).pow(2).mean()

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()

    lv = float(loss.detach())
    win.append(lv)
    (win0 if lead == 0 else winN).append(lv)
    if step % args.log_every == 0:
        curves["step"].append(step)
        curves["loss"].append(float(np.mean(win)))
        curves["loss_lead0"].append(float(np.mean(win0)) if win0 else None)
        curves["loss_leadN"].append(float(np.mean(winN)) if winN else None)
        curves["lr"].append(sched.get_last_lr()[0])
        print(f"step {step:6d}  loss {np.mean(win):.4f}  "
              f"lead0 {np.mean(win0) if win0 else float('nan'):.4f}  "
              f"lead1-3 {np.mean(winN) if winN else float('nan'):.4f}  "
              f"lr {sched.get_last_lr()[0]:.2e}  "
              f"{time.time()-t0:.0f}s", flush=True)
        win, win0, winN = [], [], []

    if args.val_every and step % args.val_every == 0:
        vs, per_lead = validate()
        curves["val_step"].append(step)
        curves["val_score"].append(vs)
        curves["val_per_lead"].append(
            {str(l): per_lead[l] for l in per_lead})
        flag = ""
        if vs < best["score"]:
            best = {"score": vs, "step": step,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
            since_best = 0
            flag = "  *best*"
            # Write it NOW.  Holding the only copy of a multi-hour run in RAM
            # means a kill at hour 20 loses everything; tmp+rename keeps the
            # existing file intact if we die mid-write.
            torch.save({"state_dict": best["state"], "step": best["step"],
                        "val_score": best["score"], "tag": tag,
                        "variant": "d4rt", "args": vars(args),
                        "curves": curves}, ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
            flag += " (saved)"
        else:
            since_best += 1
        print(f"  val step {step:6d}  score {vs:.4f}  "
              + " ".join(f"L{l} {per_lead[l]['TEMP']:.3f}/{per_lead[l]['SALT']:.3f}"
                         for l in sorted(per_lead)) + flag, flush=True)
        if since_best >= args.patience and step >= args.min_steps:
            print(f"early stop: {since_best} validations without improvement "
                  f"(best {best['score']:.4f} @ step {best['step']})", flush=True)
            break

if best["state"] is None:                      # no validation ran
    best = {"score": None, "step": args.steps, "state": model.state_dict()}
print(f"selected checkpoint: step {best['step']} score {best['score']}",
      flush=True)

if best["score"] is None:                   # no validation ran at all
    torch.save({"state_dict": best["state"], "step": best["step"],
                "val_score": None, "tag": tag, "variant": "d4rt",
                "args": vars(args)}, ckpt)
out_json = os.path.join(C.CACHE, f"{tag}.json")
json.dump({"tag": tag, "task": "d4rt_lead_train", "protocol": "protocol_v1",
           "git_commit": git_commit(), "smoke": args.smoke,
           "seed": args.seed, "device": dev, "n_params": n_params,
           "max_lead": MAX_LEAD, "context_months": CONTEXT,
           "recon_frac": RECON_FRAC, "args": vars(args),
           "eligible_source_months": int(eligible.size),
           "curves": curves, "ckpt": ckpt,
           "selection": {"rule": "mean over lead 0..3 x {TEMP,SALT} of "
                                 "RMSE / val climatology RMSE",
                         "best_step": best["step"],
                         "best_score": best["score"],
                         "val_floor": {str(l): val_floor[l] for l in val_floor},
                         "val_months_eligible": int(len(val_packs))},
           "wall_s": round(time.time() - t0, 1)},
          open(out_json, "w"), indent=1)
print(f"saved {ckpt}\nsaved {out_json}\ntotal {time.time()-t0:.0f}s", flush=True)
