"""Task 7 Stage A — synthetic toy: does MBCA fix tokenization-multiplicity bias?

A smooth random 2-D field is observed through (a) a dense coarse grid and
(b) a handful of scattered points.  All three fusion variants (standard
Perceiver / fixed-budget resampler / MBCA) share encoders, latent size, and
query decoder, and are trained on identical stream of random fields to regress
the field at query locations.

Sensitivity probes on held-out fields (same physical evidence, different
token representation):
  * DUPLICATION  — every point token duplicated xn (MBCA: mass split /n).
  * REFINEMENT   — the same grid field re-tokenised at 2x resolution
                   (4x the grid tokens, no new information).
Reported: relative output change ||y' - y|| / ||y||, plus held-out RMSE for
each representation.

Run:  python experiments/15_poc_toy.py [--steps 2000] [--device cuda]
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch

from ocean_tokenizer import config as C
from ocean_tokenizer.token_api import (TokenBatch, GridPatchEncoder,
                                       PointEncoder)
from ocean_tokenizer.fusion import (StandardPerceiver, FixedBudgetResampler,
                                    MBCA)

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()
dev = args.device

H = W = 16            # coarse grid resolution
NPTS = 15             # scattered point observations
NQ = 64               # training queries per field
D_MODEL, N_LATENT = 64, 32

# The toy lives on a [0,80]x[0,345] "lat/lon" carpet so the ocean encoders
# can be reused unchanged (coord featurisation is generic).
LAT = torch.linspace(0, 80, H)
LON = torch.linspace(0, 345, W)
LAT2 = torch.linspace(0, 80, 2 * H)
LON2 = torch.linspace(0, 345, 2 * W)


def sample_fields(B, rng):
    """Smooth random field: sum of K random sinusoids, evaluated everywhere."""
    K = 4
    a = rng.normal(size=(B, K)) / np.sqrt(K)
    u = rng.uniform(0.3, 1.2, size=(B, K))
    v = rng.uniform(0.3, 1.2, size=(B, K))
    ph = rng.uniform(0, 2 * np.pi, size=(B, K))

    def f(lat, lon):                      # lat, lon: (B, N) -> (B, N)
        x = lat / 80.0
        y = lon / 345.0
        out = np.zeros_like(x, dtype="float64")
        for k in range(K):
            out += a[:, k, None] * np.sin(
                2 * np.pi * (u[:, k, None] * x + v[:, k, None] * y) + ph[:, k, None])
        return out
    return f


def make_batch(B, rng, grid_hw=(LAT, LON)):
    f = sample_fields(B, rng)
    glat, glon = grid_hw
    gh, gw = len(glat), len(glon)
    la = np.broadcast_to(np.repeat(glat.numpy(), gw), (B, gh * gw))
    lo = np.broadcast_to(np.tile(glon.numpy(), gh), (B, gh * gw))
    grid_field = torch.tensor(f(la, lo).reshape(B, 1, gh, gw), dtype=torch.float32)

    pl = rng.uniform(0, 80, (B, NPTS)); po = rng.uniform(0, 345, (B, NPTS))
    pv = f(pl, po)
    ql = rng.uniform(0, 80, (B, NQ)); qo = rng.uniform(0, 345, (B, NQ))
    qv = f(ql, qo)

    obs = {
        "grid": dict(field=grid_field, lat=glat, lon=glon,
                     month=torch.full((B,), 3)),
        "points": dict(values=torch.tensor(pv, dtype=torch.float32),
                       var_id=torch.zeros(B, NPTS, dtype=torch.long),
                       lat=torch.tensor(pl, dtype=torch.float32),
                       lon=torch.tensor(po, dtype=torch.float32),
                       depth=torch.zeros(B, NPTS),
                       month=torch.full((B,), 3)),
    }
    q = torch.stack([torch.tensor(ql, dtype=torch.float32),
                     torch.tensor(qo, dtype=torch.float32),
                     torch.zeros(B, NQ), torch.full((B, NQ), 3.0)], -1)
    y = torch.tensor(qv, dtype=torch.float32)[..., None]
    return obs, q, y


def to_dev(obs, q, y):
    return ({k: {kk: vv.to(dev) for kk, vv in v.items()} for k, v in obs.items()},
            q.to(dev), y.to(dev))


def build(variant, seed):
    torch.manual_seed(seed)
    encoders = {
        "grid": GridPatchEncoder(1, d_model=D_MODEL, patch=(4, 4),
                                 modality="surf_grid"),
        "points": PointEncoder(variables=("F",), d_model=D_MODEL),
    }
    cls = {"perceiver": StandardPerceiver, "resampler": FixedBudgetResampler,
           "mbca": MBCA}[variant]
    return cls(encoders, d_model=D_MODEL, n_latent=N_LATENT, n_heads=4,
               n_self_blocks=2, c_out=1).to(dev)


def dup_points(obs, n):
    """Duplicate every point token xn (no new information)."""
    o = {k: dict(v) for k, v in obs.items()}
    p = o["points"]
    o["points"] = {k: (torch.cat([v] * n, dim=1)
                       if v.ndim >= 2 and v.shape[1] == NPTS else v)
                   for k, v in p.items()}
    return o


@torch.no_grad()
def rel_change(model, obs_a, obs_b, q):
    ya = model(obs_a, q)
    yb = model(obs_b, q)
    return float((ya - yb).norm() / (ya.norm() + 1e-12))


@torch.no_grad()
def rmse(model, obs, q, y):
    return float(((model(obs, q) - y) ** 2).mean().sqrt())


def main():
    print(f"device={dev} steps={args.steps}", flush=True)
    results = {}
    for variant in ("perceiver", "resampler", "mbca"):
        rng = np.random.default_rng(args.seed)
        model = build(variant, args.seed)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
        t0 = time.time()
        for step in range(1, args.steps + 1):
            obs, q, y = to_dev(*make_batch(args.batch, rng))
            opt.zero_grad()
            loss = ((model(obs, q) - y) ** 2).mean()
            loss.backward(); opt.step(); sched.step()
            if step % max(1, args.steps // 5) == 0:
                print(f"  [{variant:9s}] step {step:5d} loss {float(loss):.4f}",
                      flush=True)
        model.eval()

        # -------- held-out sensitivity probes (fixed eval rng) --------
        erng = np.random.default_rng(9999)
        obs, q, y = to_dev(*make_batch(32, erng))
        # duplication (mass handling is inside the encoders/fusion: PointEncoder
        # has no mass -> uniform; duplication multiplies point tokens)
        dup = {n: rel_change(model, obs, dup_points(obs, n), q) for n in (2, 4, 8)}
        dup_rmse = {n: rmse(model, dup_points(obs, n), q, y) for n in (1, 2, 4, 8)}
        # refinement: same fields re-evaluated on a 2x grid (4x tokens)
        erng2 = np.random.default_rng(9999)
        obs2, _, _ = to_dev(*make_batch(32, erng2, grid_hw=(LAT2, LON2)))
        obs_ref = {"grid": obs2["grid"], "points": obs["points"]}
        refine = rel_change(model, obs, obs_ref, q)
        refine_rmse = rmse(model, obs_ref, q, y)
        base_rmse = rmse(model, obs, q, y)
        results[variant] = {
            "train_secs": round(time.time() - t0, 1),
            "heldout_rmse": base_rmse,
            "dup_rel_change": dup, "dup_rmse": dup_rmse,
            "refine_rel_change": refine, "refine_rmse": refine_rmse,
        }
        print(f"  [{variant:9s}] rmse={base_rmse:.4f}  "
              f"dup8 delta={dup[8]:.4f} (rmse {dup_rmse[8]:.4f})  "
              f"refine delta={refine:.4f} (rmse {refine_rmse:.4f})", flush=True)

    out = os.path.join(C.CACHE, "poc_toy.json")
    with open(out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"\n-> {out}", flush=True)

    print("\n| model | held-out RMSE | dup x2 | dup x4 | dup x8 | refine 2x |")
    print("|---|---|---|---|---|---|")
    for v, r in results.items():
        d = r["dup_rel_change"]
        print(f"| {v} | {r['heldout_rmse']:.4f} | {d[2]:.4f} | {d[4]:.4f} "
              f"| {d[8]:.4f} | {r['refine_rel_change']:.4f} |")


if __name__ == "__main__":
    main()
