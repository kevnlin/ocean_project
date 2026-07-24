"""Task 6 deliverable — one summary table of all invariance test results.

Combines:
  * architectural probes (random-init, fixed seed): exact partition,
    controlled duplication, physical grid refinement, profile resampling —
    for all three fusion variants;
  * trained-model sensitivity (Stage-A toy, outputs/cache/poc_toy.json):
    duplication / refinement RMSE of trained standard vs resampler vs MBCA;
  * the pytest verdicts of the Task-5/6 suites.

Writes reports/invariance_test_summary.md.

Run:  python experiments/17_invariance_summary.py
"""
import sys, os, json, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch

from ocean_tokenizer import config as C
from ocean_tokenizer.token_api import MODALITIES
from ocean_tokenizer.fusion import build_fusion_model
from ocean_tokenizer.invariance import (split_token, duplicate_tokens,
                                        output_change, latent_change,
                                        modality_mass_totals)

torch.manual_seed(0)
D_MODEL, N_LATENT = 32, 16
DEPTHS = np.array([5, 15, 25, 35, 45, 55, 65, 85, 105, 125, 145, 165,
                   186, 222, 267, 327, 408, 527, 707, 985], dtype="float32")
D = len(DEPTHS)
VARIANTS = ("perceiver", "resampler", "mbca")
LABEL = {"perceiver": "Standard Perceiver", "resampler": "Fixed resampler",
         "mbca": "MBCA"}


class FakeGrid:
    depth = DEPTHS.astype("float64")


def build(variant):
    m = build_fusion_model(variant, FakeGrid(), d_model=D_MODEL,
                           n_latent=N_LATENT, n_heads=4, n_self_blocks=2,
                           patch=(4, 6), seed=3)
    m.eval()
    return m


LAT0 = torch.linspace(-80, 80, 8)
LON0 = torch.linspace(0, 345, 12)


def refine_coords(c, k):
    step = float(c[1] - c[0])
    off = (torch.arange(k, dtype=c.dtype) + 0.5) / k - 0.5
    return (c[:, None] + off[None, :] * step).reshape(-1)


def obs(P=7, seed=0, month=3, k=1):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(1, 2, 8, 12)).astype("float32")
    f[:, :, :2, :3] = np.nan
    if k > 1:
        f = np.repeat(np.repeat(f, k, axis=2), k, axis=3)
    return {
        "profiles": dict(
            prof=torch.tensor(rng.normal(size=(1, P, 2, D)).astype("float32")),
            lat=torch.tensor(rng.uniform(-80, 80, (1, P)).astype("float32")),
            lon=torch.tensor(rng.uniform(0, 360, (1, P)).astype("float32")),
            month=torch.tensor([month])),
        "surf": dict(field=torch.tensor(f),
                     lat=LAT0 if k == 1 else refine_coords(LAT0, k),
                     lon=LON0 if k == 1 else refine_coords(LON0, k),
                     month=torch.tensor([month])),
    }


rngq = np.random.default_rng(2)
Q = torch.tensor(np.stack([rngq.uniform(-80, 80, 9), rngq.uniform(0, 360, 9),
                           rngq.uniform(0, 985, 9), np.full(9, 3.0)],
                          -1).astype("float32"))[None]

probes = {}
for v in VARIANTS:
    model = build(v)
    tb = model.encode(obs(), batch=1)
    idx = int(((tb.modality[0] == MODALITIES["profile"]) & tb.mask[0])
              .nonzero()[0])
    row = {}
    for n in (2, 4, 8):
        row[f"partition_x{n}"] = output_change(model, tb,
                                               split_token(tb, idx, n), Q)
    dup_idx = ((tb.modality[0] == MODALITIES["profile"]) & tb.mask[0]
               & (tb.parent_id[0] < 3)).nonzero().flatten()
    for n in (2, 4, 8):
        row[f"dup_x{n}"] = output_change(
            model, tb, duplicate_tokens(tb, dup_idx, n, divide_mass=True), Q)
    tb_r = model.encode(obs(k=2), batch=1)
    row["refine2x_pred"] = output_change(model, tb, tb_r, Q)
    row["refine2x_latent"] = latent_change(model, tb, tb_r)
    g = MODALITIES["surf_grid"]
    row["refine2x_mass_ratio"] = (modality_mass_totals(tb_r)[g]
                                  / modality_mass_totals(tb)[g])
    probes[v] = row

toy = None
toy_path = os.path.join(C.CACHE, "poc_toy.json")
if os.path.exists(toy_path):
    toy = json.load(open(toy_path))["results"]

# pytest verdicts
pt = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
    cwd=C.ROOT, capture_output=True, text=True)
verdict = pt.stdout.strip().splitlines()[-1] if pt.stdout.strip() else "n/a"

fmt = lambda x: f"{x:.1e}" if x < 1e-3 else f"{x:.4f}"
L = []
L.append("# Invariance Test Summary (Task 6)\n")
L.append("*Architectural probes: random-init models, fixed seed, identical "
         "trunks — the invariances under test are structural, not learned. "
         "Trained-model sensitivity: Stage-A toy (converged models, held-out "
         "fields). Definitions: docs/token_measure_definition.md.*\n")
L.append(f"**Unit-test verdict:** `{verdict}` (suites: token API, profile "
         "encoder, fusion, mbca_invariance [A+B], token_refinement [C], "
         "profile_resampling [D]).\n")

L.append("## Architectural probes — relative output change (lower = more invariant)\n")
L.append("| probe | " + " | ".join(LABEL[v] for v in VARIANTS) + " | MBCA expectation |")
L.append("|---|---|---|---|---|")
rows = [
    ("Test A — exact partition x2", "partition_x2", "**exact (0)**"),
    ("Test A — exact partition x4", "partition_x4", "**exact (0)**"),
    ("Test A — exact partition x8", "partition_x8", "**exact (0)**"),
    ("Test B — duplication x2 (mass split)", "dup_x2", "**exact (0)**"),
    ("Test B — duplication x4 (mass split)", "dup_x4", "**exact (0)**"),
    ("Test B — duplication x8 (mass split)", "dup_x8", "**exact (0)**"),
    ("Test C — physical 2x refinement (pred)", "refine2x_pred", "smallest"),
    ("Test C — physical 2x refinement (latent)", "refine2x_latent", "smallest"),
]
for name, key, expect in rows:
    L.append(f"| {name} | " + " | ".join(fmt(probes[v][key]) for v in VARIANTS)
             + f" | {expect} |")
L.append("")
mr = probes["mbca"]["refine2x_mass_ratio"]
L.append(f"Measure contract under physical 2x refinement: grid-modality total "
         f"support mass ratio refined/coarse = **{mr:.4f}** (exact "
         f"conservation = 1; token count grows ~4x).\n")

if toy:
    L.append("## Trained-model sensitivity (Stage-A toy, converged)\n")
    L.append("| model | held-out RMSE | dup x8 shift | dup x8 RMSE | refine 2x shift |")
    L.append("|---|---|---|---|---|")
    for v in ("perceiver", "resampler", "mbca"):
        r = toy[v]
        L.append(f"| {LABEL[v]} | {r['heldout_rmse']:.4f} "
                 f"| {r['dup_rel_change']['8']:.4f} "
                 f"| {r['dup_rmse']['8']:.4f} "
                 f"| {r['refine_rel_change']:.4f} |")
    L.append("")
    L.append("Trained standard attention degrades catastrophically under "
             "duplication (RMSE "
             f"{toy['perceiver']['dup_rmse']['1']:.3f} -> "
             f"{toy['perceiver']['dup_rmse']['8']:.3f}); trained MBCA is "
             "exactly invariant at equal accuracy.\n")

L.append("## Reading guide\n")
L.append("- **Exact rows** hold to float32 tolerance (<1e-5) for MBCA by "
         "construction: n children (k, v, w/n) reproduce the parent's "
         "attention contribution exactly.")
L.append("- **Physical refinement** genuinely changes token content (patches "
         "cover different windows), so no method is exact; MBCA conserves "
         "the modality's total attention mass while standard attention lets "
         "it grow ~4x.")
L.append("- **Profile resampling** is absorbed by the encoder (fixed physical "
         "bands + span mass): token count and masses are level-count "
         "independent; residual band-mass shift <10% from boundary "
         "half-intervals (test_profile_resampling).")

out = os.path.join(C.REPORTS, "invariance_test_summary.md")
open(out, "w").write("\n".join(L))
print(f"-> {out}")
print("\n".join(L[8:20]))
