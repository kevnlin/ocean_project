"""The controlled Gaussian-process benchmark of 125 on four truths, three of them non-Gaussian.

A reviewer's objection to the read-out: "kriging attention still assumes a Gaussian covariance
decay". This script keeps 125's protocol (1000 x 1000 km square, 48 x 48 grid, a stream function
psi of four Gaussian bumps as the auxiliary image, 150 noisy observations with noise sd 0.15, 200
queries, read-outs over the 32 nearest observations with the same feature MLP of psi, grad psi and
position, loss MSE + 0.02 NLL, 2000 training and 200 test fields, 4000 steps) and changes only the
truth's covariance:

  flow         125's truth unchanged: the Paciorek-Schervish kernel with length-scale matrices
               stretched along the isolines of psi (the control; reproduces 125's numbers)
  warped       a Gaussian kernel in a warped plane, C = exp(-|w(x) - w(y)|^2 / (2 L^2)) with
               w(x) = (x1 + A psi(x), x2), A = 100 km, L = 80 km: the correlation lobes follow the
               curved isolines of x1 + A psi, not an ellipse (the "shape like a curve" case)
  exponential  the stationary Matern-1/2 kernel exp(-|x - y| / L): a cusp at zero, no Gaussian decay
  mixture      0.7 Gaussian at L plus 0.3 Gaussian at 3 L: two scales

Each truth is sampled exactly: the dense covariance of the 2304 grid points plus a 1e-6 jitter is
factorised by Cholesky (the jitter is raised tenfold if a factorisation fails, and the jitter used is
recorded); observations and queries are grid points, as in 125. The same random stream (psi and the
white noise) is used for every truth, so the four truths share psi and differ only in covariance.

Read-outs: oi, softmax, ka_stationary, ka_flow (125's four), ka_deep (a Gaussian in a learned feature
space), ka_product (flow kernel times feature Gaussian), ka_multiscale (two flow kernels) and the
oracle (the exact local GP with the true covariance and noise). Three training seeds per read-out
(seed 0 is 125's: the control's first three read-outs repeat 125 step for step); the fields and the
test queries are the same for every seed, so the oracle and OI have no seed spread.

For ka_deep on the warped truth, a diagnostic of whether the learned features follow the warp: the
correlation, over query-neighbour pairs, between the learned kernel distance |dx|^2 / ell^2 +
|phi_a - phi_b|^2 and the true warped distance |w(x) - w(y)|^2 / L^2, against the plain Euclidean
distance |dx|^2 / L^2 as the reference.

Writes outputs/synthetic_gp_nongaussian_20261021/results.json, results.md and rmse.png.

  .venv/bin/python experiments/synthetic/151_synthetic_gp_nongaussian.py --truths flow     # one truth
  .venv/bin/python experiments/synthetic/151_synthetic_gp_nongaussian.py --aggregate       # join them
"""
from __future__ import annotations

import argparse
import importlib.util
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

_spec = importlib.util.spec_from_file_location("gp125", ROOT / "experiments/synthetic/125_synthetic_gp_benchmark.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

OUTPUT = ROOT / "outputs/synthetic_gp_nongaussian_20261021"
TRUTHS = ("flow", "warped", "exponential", "mixture")
TRAINED = ("softmax", "ka_stationary", "ka_flow", "ka_deep", "ka_product", "ka_multiscale")   # 125's order first
READOUTS = ("oi",) + TRAINED + ("oracle",)
L_KM, WARP_KM = B.BASE_KM, 100.
MIXTURE = ((.7, 1.), (.3, 3.))                                                              # (weight, scale / L)
N_OBS, N_QUERY, K, NOISE_SD = B.N_OBS, B.N_QUERY, B.K, B.NOISE_SD


def truth_covariance(truth, xy_a, aux_a, xy_b, aux_b):
    """Correlation ``[..., A, B]`` between point sets ``xy [..., A, 2]`` (km) with per-point auxiliaries
    ``aux`` (flow: the length-scale matrix (s11, s12, s22); warped: psi; otherwise unused)."""
    if truth == "flow":
        return KA.ps_covariance(xy_a, aux_a, xy_b, aux_b)
    dx = xy_a[..., :, None, 0] - xy_b[..., None, :, 0]
    dy = xy_a[..., :, None, 1] - xy_b[..., None, :, 1]
    if truth == "warped":
        dx = dx + WARP_KM * (aux_a[..., :, None, 0] - aux_b[..., None, :, 0])
        return torch.exp(-.5 * (dx.square() + dy.square()) / L_KM ** 2)
    r2 = dx.square() + dy.square()
    if truth == "exponential":
        return torch.exp(-r2.clamp_min(0.).sqrt() / L_KM)
    if truth == "mixture":
        return sum(w * torch.exp(-.5 * r2 / (s * L_KM) ** 2) for w, s in MIXTURE)
    raise ValueError(truth)


def auxiliary(truth, psi, sigma):
    if truth == "flow":
        return sigma
    if truth == "warped":
        return psi[:, None].astype(np.float64)
    return np.zeros((len(psi), 1))


def make_fields(truth, n, seed, device):
    """``n`` scenes with the same random stream as 125's ``make_fields`` (psi, then the white noise)."""
    rng = np.random.default_rng(seed)
    xy = B.grid_points()
    xy_t = torch.as_tensor(xy, dtype=torch.float64, device=device)
    eye = torch.eye(len(xy), dtype=torch.float64, device=device)
    scenes, jitters = [], []
    for _ in range(n):
        psi = B.stream_function(rng, xy)
        grad = B.gradient(psi)
        sigma, theta, ratio = B.true_sigma(grad)
        aux = torch.as_tensor(auxiliary(truth, psi, sigma), dtype=torch.float64, device=device)
        cov = truth_covariance(truth, xy_t, aux, xy_t, aux)
        for jitter in (1e-6, 1e-5, 1e-4, 1e-3):
            try:
                chol = torch.linalg.cholesky(cov + jitter * eye)
                break
            except torch.linalg.LinAlgError:
                continue
        else:
            raise RuntimeError(f"{truth}: covariance not positive definite with jitter up to 1e-3")
        jitters.append(jitter)
        z = torch.as_tensor(rng.normal(size=(len(xy), 2)), dtype=torch.float64, device=device)
        field = (chol @ z).float().cpu().numpy()
        scenes.append({"psi": psi.astype(np.float32), "grad": grad.astype(np.float32), "sigma": sigma, "theta": theta,
                       "ratio": ratio, "field": field, "aux": aux.cpu().numpy()})
    return xy, scenes, {"max_jitter": float(max(jitters)), "jitter_counts": {str(j): int(jitters.count(j)) for j in sorted(set(jitters))},
                        "field_sd": float(np.std(np.stack([sc["field"] for sc in scenes])))}


def sample_batch(scenes, xy, rng, n_scenes, n_query=N_QUERY):
    """125's ``sample_batch`` (same random calls) with the truth's auxiliaries instead of sigma."""
    out = {k: [] for k in ("hq", "hl", "offsets", "innovation", "target", "aux_q", "aux_l", "sigma_q", "scene", "query")}
    for _ in range(n_scenes):
        i = int(rng.integers(len(scenes)))
        sc = scenes[i]
        perm = rng.permutation(len(xy))
        obs, query = perm[:N_OBS], perm[N_OBS:N_OBS + n_query]
        y = sc["field"][obs] + rng.normal(scale=NOISE_SD, size=(N_OBS, 2)).astype(np.float32)
        d = ((xy[query, None] - xy[obs][None]) ** 2).sum(-1)
        nn_idx = np.argsort(d, 1)[:, :K]
        out["hq"].append(B.features(sc, query, xy))
        out["hl"].append(B.features(sc, obs, xy)[nn_idx])
        out["offsets"].append(xy[obs][nn_idx] - xy[query][:, None])
        out["innovation"].append(y[nn_idx])
        out["target"].append(sc["field"][query])
        out["aux_q"].append(sc["aux"][query])
        out["aux_l"].append(sc["aux"][obs][nn_idx])
        out["sigma_q"].append(sc["sigma"][query])
        out["scene"].append(np.full(len(query), i))
        out["query"].append(query)
    return {k: np.concatenate(v) for k, v in out.items()}


def oracle_estimate(truth, b, gamma=NOISE_SD ** 2):
    """The exact local GP over the K neighbours with the true covariance and noise (125's ``gp_estimate``)."""
    off = torch.as_tensor(b["offsets"], dtype=torch.float64)
    q, k = off.shape[:2]
    xy = torch.cat([torch.zeros(q, 1, 2, dtype=torch.float64), off], 1)
    aux = torch.cat([torch.as_tensor(b["aux_q"], dtype=torch.float64)[:, None], torch.as_tensor(b["aux_l"], dtype=torch.float64)], 1)
    cov = truth_covariance(truth, xy, aux, xy, aux)
    system = cov[:, 1:, 1:] + gamma * torch.eye(k, dtype=torch.float64)
    w = torch.linalg.solve(system, cov[:, 1:, :1])[..., 0]
    y = torch.as_tensor(b["innovation"], dtype=torch.float64)
    mean = (w[..., None] * y).sum(1)
    var = (cov[:, 0, 0] - (w * cov[:, 1:, 0]).sum(1)).clamp_min(1e-6)
    return mean.numpy(), np.sqrt(var.numpy())[:, None].repeat(2, 1)


def tune_oi(train, xy):
    rng_tune = np.random.default_rng(4)
    tune_batches = [sample_batch(train, xy, rng_tune, 1) for _ in range(100)]
    tune = {k: np.concatenate([b[k] for b in tune_batches]) for k in tune_batches[0]}
    best = None
    for ell in (15., 20., 30., 40., 50., 60., 80., 100., 130., 160.):
        for gamma in (.01, .0225, .05, .1, .2, .4, .8):
            m, s, _ = B.gp_estimate(tune, None, None, ell, gamma)
            r = B.scores(m, s, tune["target"])["rmse"]
            if best is None or r < best[0]:
                best = (r, ell, gamma)
    return best[1], best[2]


def deep_distances(ka, h, local, offsets):
    """ka_deep's learned squared kernel distance ``|dx|^2 / ell^2 + |phi_a - phi_b|^2`` between each query
    and its neighbours, variable 0: ``[Q, K]`` (the quantity whose Gaussian is the learned correlation)."""
    f64 = torch.float64
    q, k = offsets.shape[:2]
    n = ka.n_query
    ell = ka.log_ell.to(f64).clamp(math.log(10.), math.log(5000.))[0, 0].exp()
    hq = ka.query_head(h).to(f64).view(q, 2, n)[:, 0]                                      # [Q, n]
    pair = torch.cat((local, h[:, None].expand(-1, k, -1)), -1)
    hl = ka.local_head(pair).to(f64).view(q, k, 2, n + 1)[:, :, 0, :n]                     # [Q, K, n]
    phi_q = KA.FEATURE_BOUND * hq[:, 1:].tanh()
    phi_l = KA.FEATURE_BOUND * hl[..., 1:].tanh()
    euclid = offsets.to(f64).square().sum(-1) / ell.square()
    return euclid + (phi_q[:, None] - phi_l).square().sum(-1), ell


def warp_diagnostic(model, tb, device):
    """How well ka_deep's learned distance follows the true warped distance over query-neighbour pairs."""
    from scipy.stats import spearmanr
    learned, true, euclid = [], [], []
    with torch.no_grad():
        for i in range(0, len(tb["target"]), 4000):
            chunk = {k: torch.as_tensor(v[i:i + 4000], dtype=torch.float32, device=device) for k, v in tb.items()
                     if k in ("hq", "hl", "offsets")}
            h, local = model.net(chunk["hq"]), model.net(chunk["hl"])
            d2, ell = deep_distances(model.read, h, local, chunk["offsets"])
            learned.append(d2.cpu().numpy().ravel())
            off = tb["offsets"][i:i + 4000].astype(np.float64)
            dpsi = tb["aux_q"][i:i + 4000, None, 0] - tb["aux_l"][i:i + 4000, :, 0]
            true.append((((off[..., 0] + WARP_KM * dpsi) ** 2 + off[..., 1] ** 2) / L_KM ** 2).ravel())
            euclid.append(((off ** 2).sum(-1) / L_KM ** 2).ravel())
    learned, true, euclid = map(np.concatenate, (learned, true, euclid))
    # the non-Euclidean part of each distance: what the features (or the warp) add to |dx|^2 / L^2
    return {"pearson_learned_vs_true": float(np.corrcoef(learned, true)[0, 1]),
            "pearson_euclid_vs_true": float(np.corrcoef(euclid, true)[0, 1]),
            "spearman_learned_vs_true": float(spearmanr(learned, true)[0]),
            "spearman_euclid_vs_true": float(spearmanr(euclid, true)[0]),
            "pearson_learned_vs_true_nonEuclidean_part": float(np.corrcoef(learned - euclid, true - euclid)[0, 1]),
            "learned_ell_km": float(ell), "pairs": int(len(true))}


def flow_diagnostic(model, tb, device):
    """125's orientation alignment and anisotropy correlation of ka_flow's query kernels."""
    with torch.no_grad():
        sig = []
        for i in range(0, len(tb["target"]), 4000):
            h = model.net(torch.as_tensor(tb["hq"][i:i + 4000], dtype=torch.float32, device=device))
            sig.append(B.KA_query_sigma(model.read, h).cpu().numpy())
    sig = np.concatenate(sig)[:, 0]
    learned_theta = .5 * np.arctan2(2 * sig[:, 1], sig[:, 0] - sig[:, 2])
    true_theta = .5 * np.arctan2(2 * tb["sigma_q"][:, 1], tb["sigma_q"][:, 0] - tb["sigma_q"][:, 2])
    root = lambda s: np.sqrt(((s[:, 0] - s[:, 2]) / 2) ** 2 + s[:, 1] ** 2)
    ratio = lambda s: np.sqrt(((s[:, 0] + s[:, 2]) / 2 + root(s)) / np.maximum((s[:, 0] + s[:, 2]) / 2 - root(s), 1e-9))
    strong = ratio(tb["sigma_q"]) > 1.6
    return {"orientation_alignment": float(np.abs(np.cos(learned_theta - true_theta))[strong].mean()),
            "anisotropy_correlation": float(np.corrcoef(np.log(ratio(sig)), np.log(ratio(tb["sigma_q"])))[0, 1])}


def train(kind, train_scenes, xy, rng, seed, steps, device):
    torch.manual_seed(seed)
    model = B.ReadOut(kind).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=.05)
    for step in range(steps):
        b = B.to_device(sample_batch(train_scenes, xy, rng, 8, 128), device)
        mean, std, _ = model(b)
        err = mean - b["target"]
        loss = err.square().mean() + .02 * (.5 * (err / std).square() + std.log()).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        sched.step()
    return model.eval()


def evaluate(model, tb, device):
    means, stds, first_weights = [], [], None
    with torch.no_grad():
        for i in range(0, len(tb["target"]), 4000):
            chunk = B.to_device({k: v[i:i + 4000] for k, v in tb.items() if k in ("hq", "hl", "offsets", "innovation", "target")}, device)
            m, s, out = model(chunk)
            means.append(m.cpu().numpy())
            stds.append(s.cpu().numpy())
            if i == 0:
                first_weights = out["weights"].float().cpu().numpy()[:, 0]
    r = B.scores(np.concatenate(means), np.concatenate(stds), tb["target"])
    r["negative_weight_share"] = float((first_weights < 0).mean())
    return r


def run_truth(truth, args, device):
    started = time.time()
    xy, train_scenes, info = make_fields(truth, args.train_fields, 1, device)
    _, test_scenes, test_info = make_fields(truth, args.test_fields, 2, device)
    print(f"[{truth}] fields ready in {time.time() - started:.0f} s; {info}", flush=True)
    rng_test = np.random.default_rng(3)
    test_batches = [sample_batch([sc], xy, rng_test, 1) for sc in test_scenes]
    tb = {k: np.concatenate([b[k] for b in test_batches]) for k in test_batches[0]}
    out = {"sampling": {"train": info, "test": test_info}, "runs": {}}
    mean, std = oracle_estimate(truth, tb)
    out["runs"]["oracle"] = [B.scores(mean, std, tb["target"])]
    ell, gamma = tune_oi(train_scenes, xy)
    mean, std, _ = B.gp_estimate(tb, None, None, ell, gamma)
    out["runs"]["oi"] = [{**B.scores(mean, std, tb["target"]), "ell_km": ell, "gamma": gamma}]
    print(f"[{truth}] " + json.dumps({"oracle": out["runs"]["oracle"][0], "oi": out["runs"]["oi"][0]}), flush=True)
    for seed in args.seeds:
        rng = np.random.default_rng(5 + seed)                                        # shared across kinds, as in 125
        for kind in args.kinds:
            t0 = time.time()
            model = train(kind, train_scenes, xy, rng, seed, args.steps, device)
            r = {"seed": seed, **evaluate(model, tb, device), "train_s": round(time.time() - t0)}
            if kind == "ka_flow" and truth == "flow":
                r.update(flow_diagnostic(model, tb, device))
            if kind == "ka_deep" and truth == "warped":
                r["warp_diagnostic"] = warp_diagnostic(model, tb, device)
            torch.save(model.state_dict(), OUTPUT / f"{truth}_{kind}_seed{seed}.pt")
            out["runs"].setdefault(kind, []).append(r)
            print(f"[{truth}] " + json.dumps({kind: r}), flush=True)
            (OUTPUT / f"partial_{truth}.json").write_text(json.dumps(out, indent=1) + "\n")
    out["elapsed_s"] = round(time.time() - started)
    (OUTPUT / f"partial_{truth}.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def summarise(runs):
    rows = {}
    for kind, items in runs.items():
        rows[kind] = {}
        for metric in ("rmse", "crps"):
            v = np.array([r[metric] for r in items])
            rows[kind][metric] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0., "n": int(len(v))}
    return rows


def aggregate(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {"setup": {"grid": B.N_GRID, "size_km": B.SIZE_KM, "obs": N_OBS, "queries": N_QUERY, "neighbours": K,
                         "noise_sd": NOISE_SD, "length_km": L_KM, "warp_km": WARP_KM, "mixture": MIXTURE,
                         "train_fields": args.train_fields, "test_fields": args.test_fields, "steps": args.steps,
                         "seeds": list(args.seeds), "readouts": list(READOUTS)},
               "truths": {}}
    control = json.loads((ROOT / "outputs/synthetic_gp_20261020/results.json").read_text()) \
        if (ROOT / "outputs/synthetic_gp_20261020/results.json").exists() else None
    if control is None:
        alt = Path("/DATA2/linke/ocean_project_ka_dev3/outputs/synthetic_gp_20261020/results.json")
        control = json.loads(alt.read_text()) if alt.exists() else None
    for truth in TRUTHS:
        path = OUTPUT / f"partial_{truth}.json"
        if not path.exists():
            print(f"missing {path}", flush=True)
            continue
        part = json.loads(path.read_text())
        entry = {"sampling": part["sampling"], "summary": summarise(part["runs"]), "runs": part["runs"]}
        if truth == "flow" and control is not None:
            entry["control_125"] = {k: {m: control[k][m] for m in ("rmse", "crps")} for k in ("oracle", "oi", "softmax", "ka_stationary", "ka_flow")}
        results["truths"][truth] = entry
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=1) + "\n")

    # markdown table
    lines = ["# Synthetic GP benchmark on four truths (151)", "",
             f"Protocol of 125: {args.train_fields} training / {args.test_fields} test fields, {N_OBS} observations (noise sd {NOISE_SD}), "
             f"{N_QUERY} queries, {K} neighbours, {args.steps} steps, seeds {list(args.seeds)}. "
             "RMSE and CRPS as mean +- sd over the training seeds (the oracle and OI have no training seed).", ""]
    for truth, entry in results["truths"].items():
        s = entry["summary"]
        jit = entry["sampling"]["train"]["max_jitter"]
        lines += [f"## {truth}", "", f"Cholesky jitter {jit:g}; sampled field sd {entry['sampling']['train']['field_sd']:.3f}.", "",
                  "| read-out | RMSE | CRPS | RMSE / oracle |", "|---|---|---|---|"]
        oracle = s["oracle"]["rmse"]["mean"]
        for kind in READOUTS:
            if kind not in s:
                continue
            r, c = s[kind]["rmse"], s[kind]["crps"]
            lines.append(f"| {kind} | {r['mean']:.4f} +- {r['sd']:.4f} | {c['mean']:.4f} +- {c['sd']:.4f} | {r['mean'] / oracle:.3f} |")
        if truth == "flow" and "control_125" in entry:
            lines += ["", "Control versus 125 (RMSE, CRPS): " + "; ".join(
                f"{k} 125 {v['rmse']:.4f}/{v['crps']:.4f} vs 151 {s[k]['rmse']['mean']:.4f}/{s[k]['crps']['mean']:.4f}"
                for k, v in entry["control_125"].items())]
            flow_runs = entry["runs"].get("ka_flow", [])
            if flow_runs and "orientation_alignment" in flow_runs[0]:
                lines += ["", "ka_flow kernel recovery per seed: " + "; ".join(
                    f"seed {r['seed']} alignment {r['orientation_alignment']:.3f}, anisotropy corr {r['anisotropy_correlation']:.3f}"
                    for r in flow_runs)]
        if truth == "warped" and "ka_deep" in entry["runs"]:
            lines += ["", "ka_deep warp diagnostic (learned kernel distance vs true warped distance over query-neighbour pairs):", ""]
            for r in entry["runs"]["ka_deep"]:
                d = r["warp_diagnostic"]
                lines.append(f"- seed {r['seed']}: Pearson learned {d['pearson_learned_vs_true']:.3f} vs Euclidean {d['pearson_euclid_vs_true']:.3f}; "
                             f"Spearman learned {d['spearman_learned_vs_true']:.3f} vs Euclidean {d['spearman_euclid_vs_true']:.3f}; "
                             f"non-Euclidean part {d['pearson_learned_vs_true_nonEuclidean_part']:.3f}; learned ell {d['learned_ell_km']:.0f} km")
        lines.append("")
    (OUTPUT / "results.md").write_text("\n".join(lines))

    # figure: one panel per truth, RMSE per read-out (dot and seed spread), the oracle as a dashed line
    truths = [t for t in TRUTHS if t in results["truths"]]
    kinds = [k for k in READOUTS if k != "oracle"]
    group = {"oi": "baseline", "softmax": "baseline", "ka_stationary": "KA (125)", "ka_flow": "KA (125)",
             "ka_deep": "KA, new kernels", "ka_product": "KA, new kernels", "ka_multiscale": "KA, new kernels"}
    colour = {"baseline": "#8a8984", "KA (125)": "#2a78d6", "KA, new kernels": "#eb6834"}
    fig, axes = plt.subplots(1, len(truths), figsize=(2.6 * len(truths) + 1.2, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(kinds))[::-1]
    for ax, truth in zip(axes, truths):
        s = results["truths"][truth]["summary"]
        for yi, k in zip(y, kinds):
            ax.errorbar(s[k]["rmse"]["mean"], yi, xerr=s[k]["rmse"]["sd"], fmt="o", ms=5, color=colour[group[k]],
                        ecolor=colour[group[k]], elinewidth=1.2, capsize=2)
        ax.axvline(s["oracle"]["rmse"]["mean"], color="#0b0b0b", ls="--", lw=1)
        ax.set_title(truth, fontsize=10)
        ax.grid(axis="x", color="#e4e3df", lw=.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=8, length=0)
        ax.set_xlabel("RMSE", fontsize=8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(kinds, fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=g) for g, c in colour.items()]
    handles.append(plt.Line2D([], [], ls="--", color="#0b0b0b", label="oracle (exact GP, true covariance)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(.5, -.02))
    fig.suptitle(f"Read-out RMSE on four truths (mean and sd over {len(args.seeds)} training seeds)", fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(OUTPUT / "rmse.png", dpi=180, bbox_inches="tight")
    print(f"wrote {OUTPUT / 'results.json'}, results.md, rmse.png", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--train-fields", type=int, default=2000)
    parser.add_argument("--test-fields", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--truths", nargs="+", default=list(TRUTHS), choices=TRUTHS)
    parser.add_argument("--kinds", nargs="+", default=list(TRAINED), help="trained read-outs to run (ka-dev8: the fixed feature-row initialisation of ka_deep / ka_product)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--aggregate", action="store_true", help="only join the per-truth partial results")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not args.aggregate:
        device = torch.device(args.device)
        for truth in args.truths:
            run_truth(truth, args, device)
    aggregate(args)


if __name__ == "__main__":
    main()
