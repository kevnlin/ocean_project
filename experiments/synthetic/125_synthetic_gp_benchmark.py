"""A controlled test of the read-out: Gaussian random fields whose covariance follows a known flow.

Each field lives on a 1000 x 1000 km square (48 x 48 grid). A random stream function psi (a sum of
four Gaussian bumps) is the dense auxiliary image; the field's covariance is the Paciorek-Schervish
kernel with length-scale matrices stretched along the isolines of psi (anisotropy 1-4, growing with
|grad psi|). Two independent fields per psi play the two variables. Each field has 150 noisy point
observations (noise sd 0.15) and 200 query points; a model sees the observations, psi and grad psi.

Read-outs over the 32 nearest observations, with the same feature network (an MLP of psi, grad psi
and position) and loss (MSE + 0.02 Gaussian NLL):
  oi             stationary isotropic Gaussian kernel, length and noise tuned on the training fields
  softmax        softmax attention (``SoftmaxReadout``)
  ka_stationary  Kriging Attention with one learned stationary kernel
  ka_flow        Kriging Attention with the learned flow-dependent kernel (ours)
  oracle         the exact GP with the true covariance and noise (an upper bound)

Writes ``outputs/synthetic_gp_20261020/results.json`` (RMSE, CRPS and, for ka_flow, how well the
learned kernels recover the true orientation and anisotropy) and the trained read-outs.

  .venv/bin/python experiments/synthetic/125_synthetic_gp_benchmark.py [--steps 4000]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ocean_tokenizer import kriging_attention as KA
from ocean_tokenizer.grid_kriging_ocean import SoftmaxReadout
from ocean_tokenizer.calibration import crps

OUTPUT = ROOT / "outputs/synthetic_gp_20261020"
N_GRID, SIZE_KM = 48, 1000.
N_OBS, N_QUERY, K = 150, 200, 32
NOISE_SD, BASE_KM = .15, 80.
WIDTH = 64


def grid_points():
    axis = (np.arange(N_GRID) + .5) * SIZE_KM / N_GRID
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], -1)                       # [P, 2] east, north (km)


def stream_function(rng, xy):
    psi = np.zeros(len(xy))
    for _ in range(4):
        centre = rng.uniform(0, SIZE_KM, 2)
        psi += rng.normal() * np.exp(-((xy - centre) ** 2).sum(-1) / (2 * 250. ** 2))
    return (psi - psi.mean()) / max(psi.std(), 1e-6)


def gradient(psi):
    field = psi.reshape(N_GRID, N_GRID)
    dy, dx = np.gradient(field, SIZE_KM / N_GRID)
    return np.stack([dx.ravel(), dy.ravel()], -1)                       # per km


def true_sigma(grad):
    """(s11, s12, s22) in km^2 stretched along the isolines (direction (-dy, dx))."""
    strength = np.hypot(grad[:, 0], grad[:, 1])
    g = np.clip(strength / max(np.quantile(strength, .95), 1e-9), 0, 1)
    ratio = 1 + 3 * g
    theta = np.arctan2(grad[:, 0], -grad[:, 1])
    major, minor = BASE_KM * np.sqrt(ratio), BASE_KM / np.sqrt(ratio)
    c, s = np.cos(theta), np.sin(theta)
    s11 = (c * major) ** 2 + (s * minor) ** 2
    s22 = (s * major) ** 2 + (c * minor) ** 2
    s12 = c * s * (major ** 2 - minor ** 2)
    return np.stack([s11, s12, s22], -1), theta, ratio


def make_fields(n, seed, device):
    """``n`` scenes: psi, grad psi, the true kernel and two field samples on the grid."""
    rng = np.random.default_rng(seed)
    xy = grid_points()
    xy_t = torch.as_tensor(xy, dtype=torch.float64, device=device)
    scenes = []
    for _ in range(n):
        psi = stream_function(rng, xy)
        grad = gradient(psi)
        sigma, theta, ratio = true_sigma(grad)
        s = torch.as_tensor(sigma, dtype=torch.float64, device=device)
        cov = KA.ps_covariance(xy_t, s, xy_t, s) + 1e-6 * torch.eye(len(xy), dtype=torch.float64, device=device)
        chol = torch.linalg.cholesky(cov)
        z = torch.as_tensor(rng.normal(size=(len(xy), 2)), dtype=torch.float64, device=device)
        field = (chol @ z).float().cpu().numpy()
        scenes.append({"psi": psi.astype(np.float32), "grad": grad.astype(np.float32), "sigma": sigma, "theta": theta,
                       "ratio": ratio, "field": field})
    return xy, scenes


def features(scene, idx, xy):
    g = scene["grad"][idx] * 250.                                         # gradients of order one
    return np.concatenate([scene["psi"][idx, None], g, xy[idx] / SIZE_KM], -1)


def sample_batch(scenes, xy, rng, n_scenes, n_query=N_QUERY):
    """Observations, queries and the 32-nearest-neighbour geometry of ``n_scenes`` random scenes."""
    out = {k: [] for k in ("hq", "hl", "offsets", "innovation", "target", "sigma_q", "sigma_l", "scene", "query")}
    for _ in range(n_scenes):
        i = int(rng.integers(len(scenes)))
        sc = scenes[i]
        perm = rng.permutation(len(xy))
        obs, query = perm[:N_OBS], perm[N_OBS:N_OBS + n_query]
        y = sc["field"][obs] + rng.normal(scale=NOISE_SD, size=(N_OBS, 2)).astype(np.float32)
        d = ((xy[query, None] - xy[obs][None]) ** 2).sum(-1)
        nn_idx = np.argsort(d, 1)[:, :K]                                 # [Q, K] into obs
        out["hq"].append(features(sc, query, xy))
        out["hl"].append(features(sc, obs, xy)[nn_idx])
        out["offsets"].append(xy[obs][nn_idx] - xy[query][:, None])
        out["innovation"].append(y[nn_idx])
        out["target"].append(sc["field"][query])
        out["sigma_q"].append(sc["sigma"][query])
        out["sigma_l"].append(sc["sigma"][obs][nn_idx])
        out["scene"].append(np.full(len(query), i))
        out["query"].append(query)
    return {k: np.concatenate(v) for k, v in out.items()}


class ReadOut(nn.Module):
    """Feature network + a read-out over the neighbours + a spread head."""

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.net = nn.Sequential(nn.Linear(5, WIDTH), nn.SiLU(), nn.Linear(WIDTH, WIDTH), nn.SiLU(), nn.Linear(WIDTH, WIDTH))
        ell, gamma = [[BASE_KM], [BASE_KM]], [[.05], [.05]]
        self.read = (SoftmaxReadout(WIDTH, ell) if kind == "softmax" else KA.KrigingAttention(WIDTH, kind, ell, gamma))
        self.spread = nn.Sequential(nn.Linear(WIDTH + 1, WIDTH), nn.SiLU(), nn.Linear(WIDTH, 1))

    def forward(self, b):
        h, local = self.net(b["hq"]), self.net(b["hl"])
        valid = torch.ones(b["innovation"].shape, dtype=torch.bool, device=h.device)
        level = torch.zeros(len(h), dtype=torch.long, device=h.device)
        out = self.read(h, local, b["offsets"], b["innovation"], valid, level)
        v = out["variance"].float()
        std = torch.stack([.02 + nn.functional.softplus(self.spread(torch.cat([h, v[:, j:j + 1]], -1)))[:, 0]
                           for j in range(2)], -1)
        return out["estimate"].float(), std, out


def to_device(b, device):
    keys = ("hq", "hl", "offsets", "innovation", "target")
    return {k: torch.as_tensor(b[k], dtype=torch.float32, device=device) for k in keys}


def gp_estimate(b, sigma_q, sigma_l, ell=None, gamma=NOISE_SD ** 2):
    """Exact local GP with the given covariance (true PS kernel, or an isotropic Gaussian of length ell)."""
    off = torch.as_tensor(b["offsets"], dtype=torch.float64)
    q, k = off.shape[:2]
    xy = torch.cat([torch.zeros(q, 1, 2, dtype=torch.float64), off], 1)
    if ell is None:
        s = torch.cat([torch.as_tensor(sigma_q, dtype=torch.float64)[:, None], torch.as_tensor(sigma_l, dtype=torch.float64)], 1)
    else:
        s = torch.zeros(q, k + 1, 3, dtype=torch.float64)
        s[..., 0], s[..., 2] = ell ** 2, ell ** 2
    cov = KA.ps_covariance(xy, s, xy, s)
    system = cov[:, 1:, 1:] + gamma * torch.eye(k, dtype=torch.float64)
    w = torch.linalg.solve(system, cov[:, 1:, :1])[..., 0]               # [Q, K]
    y = torch.as_tensor(b["innovation"], dtype=torch.float64)            # [Q, K, 2]
    mean = (w[..., None] * y).sum(1)
    var = (cov[:, 0, 0] - (w * cov[:, 1:, 0]).sum(1)).clamp_min(1e-6)
    return mean.numpy(), np.sqrt(var.numpy())[:, None].repeat(2, 1), w.numpy()


def scores(mean, std, target):
    return {"rmse": float(np.sqrt(np.mean((mean - target) ** 2))), "crps": float(crps(mean, std, target).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--train-fields", type=int, default=2000)
    parser.add_argument("--test-fields", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    xy, train = make_fields(args.train_fields, 1, device)
    _, test = make_fields(args.test_fields, 2, device)
    print(f"fields ready in {time.time() - started:.0f} s", flush=True)
    rng_test = np.random.default_rng(3)
    test_batches = [sample_batch([sc], xy, rng_test, 1) for sc in test]
    tb = {k: np.concatenate([b[k] for b in test_batches]) for k in test_batches[0]}
    results = {"setup": {"grid": N_GRID, "size_km": SIZE_KM, "obs": N_OBS, "queries": N_QUERY, "neighbours": K,
                         "noise_sd": NOISE_SD, "base_km": BASE_KM, "train_fields": args.train_fields,
                         "test_fields": args.test_fields, "steps": args.steps}}
    # oracle and tuned OI
    mean, std, w_oracle = gp_estimate(tb, tb["sigma_q"], tb["sigma_l"])
    results["oracle"] = scores(mean, std, tb["target"])
    rng_tune = np.random.default_rng(4)
    tune_batches = [sample_batch(train, xy, rng_tune, 1) for _ in range(100)]       # drawn once, then joined
    tune = {k: np.concatenate([b[k] for b in tune_batches]) for k in tune_batches[0]}
    best = None
    for ell in (15., 20., 30., 40., 50., 60., 80., 100., 130., 160.):
        for gamma in (.01, .0225, .05, .1, .2, .4, .8):
            m, s, _ = gp_estimate(tune, None, None, ell, gamma)
            r = scores(m, s, tune["target"])["rmse"]
            if best is None or r < best[0]:
                best = (r, ell, gamma)
    mean, std, _ = gp_estimate(tb, None, None, best[1], best[2])
    results["oi"] = {**scores(mean, std, tb["target"]), "ell_km": best[1], "gamma": best[2]}
    print(json.dumps({"oracle": results["oracle"], "oi": results["oi"]}), flush=True)
    rng = np.random.default_rng(5)
    for kind in ("softmax", "ka_stationary", "ka_flow"):
        torch.manual_seed(0)
        model = ReadOut(kind).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=args.steps, pct_start=.05)
        for step in range(args.steps):
            b = to_device(sample_batch(train, xy, rng, 8, 128), device)
            mean, std, _ = model(b)
            err = mean - b["target"]
            loss = err.square().mean() + .02 * (.5 * (err / std).square() + std.log()).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            means, stds, extra = [], [], []
            for i in range(0, len(tb["target"]), 4000):
                chunk = to_device({k: v[i:i + 4000] for k, v in tb.items()}, device)
                m, s, out = model(chunk)
                means.append(m.cpu().numpy())
                stds.append(s.cpu().numpy())
                if kind == "ka_flow":
                    h = model.net(chunk["hq"])
                    sig = KA_query_sigma(model.read, h)
                    extra.append(sig.cpu().numpy())
                if i == 0:
                    first_weights = out["weights"].float().cpu().numpy()[:, 0]
        mean, std = np.concatenate(means), np.concatenate(stds)
        results[kind] = scores(mean, std, tb["target"])
        results[kind]["negative_weight_share"] = float((first_weights < 0).mean())
        if kind == "ka_flow":
            sig = np.concatenate(extra)[:, 0]                                # variable 0
            learned_theta = .5 * np.arctan2(2 * sig[:, 1], sig[:, 0] - sig[:, 2])
            true_theta = .5 * np.arctan2(2 * tb["sigma_q"][:, 1], tb["sigma_q"][:, 0] - tb["sigma_q"][:, 2])
            root = lambda s: np.sqrt(((s[:, 0] - s[:, 2]) / 2) ** 2 + s[:, 1] ** 2)
            ratio = lambda s: np.sqrt(((s[:, 0] + s[:, 2]) / 2 + root(s)) / np.maximum((s[:, 0] + s[:, 2]) / 2 - root(s), 1e-9))
            strong = ratio(tb["sigma_q"]) > 1.6
            results[kind]["orientation_alignment"] = float(np.abs(np.cos(learned_theta - true_theta))[strong].mean())
            results[kind]["anisotropy_correlation"] = float(np.corrcoef(np.log(ratio(sig)), np.log(ratio(tb["sigma_q"])))[0, 1])
        torch.save(model.state_dict(), OUTPUT / f"{kind}.pt")
        print(json.dumps({kind: results[kind]}), flush=True)
    np.savez_compressed(OUTPUT / "test_batch.npz", **{k: v for k, v in tb.items()}, xy=xy,
                        psi=np.stack([sc["psi"] for sc in test]), field=np.stack([sc["field"] for sc in test]))
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {OUTPUT / 'results.json'} in {time.time() - started:.0f} s")


def KA_query_sigma(ka, h):
    f64 = torch.float64
    ell = ka.log_ell.to(f64).clamp(math.log(10.), math.log(5000.))[:, torch.zeros(len(h), dtype=torch.long, device=h.device)].T.exp()
    raw = ka.query_head(h).to(f64).view(-1, 2, 4)
    shape = torch.cat((KA.SHAPE_BOUND * raw[..., :2].tanh(), KA.SHEAR_BOUND * raw[..., 2:3].tanh()), -1)
    return KA.length_scale_matrices(ell, shape)


if __name__ == "__main__":
    main()
