"""GODAS registered-row driver (mentor doc §6).

WARNING about the mentor document: `dfs_d4rt_intern_plan.md` is written in the
past tense as a finished audit but was never run.  Its §7/§8/§9 tables, §6.3
checkpoint hashes, §4 manifest SHA and "106 passed" are targets, not
measurements.  Numbers produced here are FRESH and must never be merged with
or compared against them.

Splits (doc §4), by calendar year on the downloaded subset:

    train       2000-2018    optimization and normalisation
    validation  2019-2021    hyperparameters and checkpoint selection
    development 2022-2024    already inspected; diagnostic only
    holdout     2025         one final evaluation after locking checkpoints

A month is an eligible SOURCE month only if its whole window stays inside its
split — one context month back, and the target at t_src — which is why the raw
228/36/36/12 months give the doc's 224/32/32/8 eligible sources.

Selection score (doc §6.2) is the mean over channels of RMSE divided by the
validation climatology RMSE, so no channel can dominate the choice.

  .venv/bin/python experiments/14_godas_dfs_d4rt.py --smoke \
      --configs dfs_oi_expert_cbottle
"""
import sys, os, json, time, argparse, subprocess, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from ocean_tokenizer.godas import load_godas, GodasNorm
from ocean_tokenizer.godas_obs import build_sample, ObsConfig, CONTEXT_MONTHS
from ocean_tokenizer.godas_model import ROWS, build_row
from ocean_tokenizer.objective_interpolation import (ObjectiveInterpolation,
                                                     OISettings)
from ocean_tokenizer.losses import CBottleMaskedLoss

SPLITS = {"train": (2000, 2018), "validation": (2019, 2021),
          "development": (2022, 2024), "holdout": (2025, 2025)}
CHANNELS = ("TEMP", "SALT")

ap = argparse.ArgumentParser()
ap.add_argument("--configs", default="dfs_oi_expert_cbottle",
                help="comma-separated registered rows")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--steps", type=int, default=5000)
ap.add_argument("--validation-interval", type=int, default=500)
ap.add_argument("--queries", type=int, default=512)
ap.add_argument("--eval-queries", type=int, default=1024)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--weight-decay", type=float, default=0.01)
ap.add_argument("--data", default=None)
ap.add_argument("--output", default=None)
ap.add_argument("--load-checkpoint", default=None,
                help="re-evaluate one row from a checkpoint; skips training")
ap.add_argument("--validation-only", action="store_true")
ap.add_argument("--test-only", action="store_true")
ap.add_argument("--test-year", type=int, default=2025)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = args.data or os.path.join(ROOT, "data", "godas_gulfstream")
OUT = args.output or os.path.join(ROOT, "outputs", "godas")
os.makedirs(OUT, exist_ok=True)
configs = [c.strip() for c in args.configs.split(",") if c.strip()]
for c in configs:
    if c not in ROWS:
        raise SystemExit(f"unknown row {c!r}; registered: {ROWS}")
if args.smoke:
    args.steps, args.validation_interval = 20, 10
    args.queries, args.eval_queries = 64, 64

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"godas driver rows={configs} seed={args.seed} steps={args.steps} "
      f"device={dev} smoke={args.smoke}", flush=True)

# ------------------------------------------------------------------ data
fields = load_godas(DATA)
months = fields["months"]
year = months.astype("datetime64[Y]").astype(int) + 1970
idx = {k: np.where((year >= lo) & (year <= hi))[0] for k, (lo, hi) in SPLITS.items()}
tr = idx["train"]
norm = GodasNorm(fields, slice(int(tr[0]), int(tr[-1]) + 1))
Z = {"months": months}
for v in ("TEMP", "SALT", "SSH"):
    Z[v] = norm.z(v, fields[v], months)


MAX_LEAD = 3


def eligible(split: str) -> np.ndarray:
    """Source months whose whole window stays inside the split.

    The window runs from the earliest context month ``t - context + 1`` to the
    furthest target ``t + MAX_LEAD``, so a split of N months yields
    N - (context - 1) - MAX_LEAD eligible sources. That is what reproduces the
    doc's 224/32/32/8 from the raw 228/36/36/12.
    """
    a = idx[split]
    return a[(CONTEXT_MONTHS - 1):len(a) - MAX_LEAD]


elig = {k: eligible(k) for k in SPLITS}
print("  ".join(f"{k}: {len(idx[k])} months / {len(elig[k])} eligible"
                for k in SPLITS), flush=True)

FLOOR = {}
for split in ("validation", "holdout"):
    se = np.zeros(len(CHANNELS)); n = 0
    for t in elig[split]:
        for ci, c in enumerate(CHANNELS):
            a = Z[c][t]
            se[ci] += np.nansum(a ** 2)
        n += np.isfinite(Z["TEMP"][t]).sum()
    FLOOR[split] = {c: float(np.sqrt(se[ci] / max(n, 1)))
                    for ci, c in enumerate(CHANNELS)}
print(f"  climatology floor (z): {FLOOR}", flush=True)


def to_device(s: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in s.items()}


def make_sample(t, train: bool, n_queries: int, rng, lead: int = 0):
    s = build_sample(Z, int(t), ObsConfig(train=train, n_queries=n_queries),
                     rng, lead=lead)
    return to_device(s, dev)


def evaluate(model, split: str, n_queries: int, seed: int) -> dict:
    """Unobserved-cell RMSE per channel, in z space, over a split."""
    rng = np.random.default_rng([seed, 7])
    se = torch.zeros(len(CHANNELS), dtype=torch.float64, device=dev)
    n = torch.zeros(len(CHANNELS), dtype=torch.float64, device=dev)
    is_oi = isinstance(model, ObjectiveInterpolation)
    if not is_oi:
        model.eval()
    for t in elig[split]:
        s = make_sample(t, False, n_queries, rng)
        with torch.no_grad():
            if is_oi:
                live = s["mask"] & s["value_mask"].all(dim=-1)
                out = model(s["query"],
                            torch.zeros(s["query"].shape[0],
                                        dtype=s["query"].dtype, device=dev),
                            s["coord"][live],
                            s["value"][live].to(s["query"].dtype),
                            s["noise_density"][live]).to(torch.float32)
            else:
                out = model(s)
        m = s["target_mask"]
        se += ((out - s["target"]) ** 2 * m).sum(dim=0).double()
        n += m.sum(dim=0).double()
    if not is_oi:
        model.train()
    rmse = (se / n.clamp(min=1)).sqrt()
    return {c: float(rmse[i]) for i, c in enumerate(CHANNELS)}


def selection_score(rmse: dict, split: str) -> float:
    return float(np.mean([rmse[c] / max(FLOOR[split][c], 1e-12)
                          for c in CHANNELS]))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ rows
results = {}
for row in configs:
    t0 = time.time()
    print(f"\n=== {row} ===", flush=True)

    if row == "objective_interpolation":
        oi = ObjectiveInterpolation(OISettings()).to(dev)
        val = evaluate(oi, "validation", args.eval_queries, args.seed)
        rec = {"row": row, "trainable": False, "validation": val,
               "validation_score": selection_score(val, "validation")}
        if not args.validation_only:
            rec["holdout"] = evaluate(oi, "holdout", args.eval_queries, args.seed)
        results[row] = rec
        print(f"  validation {val} score {rec['validation_score']:.4f}", flush=True)
        continue

    torch.manual_seed(args.seed)
    model = build_row(row).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    ckpt_path = os.path.join(OUT, f"{row}_seed{args.seed}.pt")

    if args.load_checkpoint:
        ck = torch.load(args.load_checkpoint, map_location="cpu",
                        weights_only=False)
        if ck["row"] != row:
            raise SystemExit(
                f"checkpoint is row {ck['row']!r}, not {row!r}; "
                f"--load-checkpoint accepts exactly one matching config")
        model.load_state_dict(ck["state_dict"])      # strict, per doc §6.3
        best = {"step": ck["step"], "score": ck["val_score"]}
        print(f"  loaded step {best['step']} score {best['score']:.4f}", flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        loss_fn = CBottleMaskedLoss()
        rng = np.random.default_rng([args.seed, 0])
        best = {"step": -1, "score": float("inf")}
        history = []
        for step in range(1, args.steps + 1):
            t = int(rng.choice(elig["train"]))
            s = make_sample(t, True, args.queries, rng)
            out = model(s)
            loss = loss_fn(out[None], s["target"][None], s["target_mask"][None])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % args.validation_interval == 0:
                val = evaluate(model, "validation", args.eval_queries, args.seed)
                sc = selection_score(val, "validation")
                history.append({"step": step, "score": sc, **val})
                flag = ""
                if sc < best["score"]:
                    best = {"step": step, "score": sc, "validation": val}
                    torch.save({"state_dict": model.state_dict(), "row": row,
                                "step": step, "val_score": sc, "seed": args.seed,
                                "args": vars(args)}, ckpt_path + ".tmp")
                    os.replace(ckpt_path + ".tmp", ckpt_path)
                    flag = " *best* (saved)"
                print(f"  step {step:5d} loss {float(loss):.4f} score {sc:.4f}"
                      f"  {val}{flag}", flush=True)
        best["history"] = history
        if best["step"] > 0:
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu",
                                             weights_only=False)["state_dict"])

    rec = {"row": row, "trainable": True, "n_params": n_params,
           "selected_step": best["step"], "validation_score": best.get("score"),
           "validation": best.get("validation"),
           "history": best.get("history", []),
           "checkpoint": ckpt_path,
           "checkpoint_sha256": sha256(ckpt_path) if os.path.exists(ckpt_path)
           else None}
    if not args.validation_only:
        rec["holdout"] = evaluate(model, "holdout", args.eval_queries, args.seed)
        print(f"  holdout {rec['holdout']}", flush=True)
    rec["wall_s"] = round(time.time() - t0, 1)
    results[row] = rec

out_json = os.path.join(OUT, f"metrics_seed{args.seed}.json")
json.dump({"task": "godas_dfs_d4rt", "git_commit": git_commit(),
           "seed": args.seed, "device": dev, "smoke": args.smoke,
           "splits": SPLITS, "climatology_floor_z": FLOOR,
           "eligible_source_months": {k: int(len(v)) for k, v in elig.items()},
           "args": vars(args), "results": results,
           "provenance_warning": (
               "The mentor document's result tables, checkpoint hashes and "
               "test counts are targets that were never run. These are fresh "
               "measurements and must not be compared against them.")},
          open(out_json, "w"), indent=1)
print(f"\nsaved {out_json}", flush=True)
