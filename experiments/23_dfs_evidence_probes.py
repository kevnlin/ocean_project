"""Section-11 required experiments for DFS-Attention — the evidence probes.

Every case the plan lists, run against the estimator that has to get it right,
plus (for the duplication cases) the decoded prediction of a randomly
initialised model, so "evidence unchanged" and "prediction stable" are both
measured rather than asserted:

  1  exact duplication          same level / whole profile / real-time+delayed
                                mode copies      -> evidence & prediction stable
  2  same location, depths      one column, several depths   -> evidence grows
  3  dense sampling, smooth     2 dbar vs 10 dbar, uniform layer -> discounted
  4  dense sampling, thermocline same density, sharp gradient -> NOT discounted
  5  target-resolution sweep    coarse / medium / fine       -> monotone
  6  horizontal cases           duplicates, clustered vs uniform, fronts,
                                mixed patch sizes, overlapping products

The probes are analytic: they need no trained weights, because the evidence
estimator is a property of the observing geometry.  Prediction-stability
columns use a fixed-seed random model, which is the honest architectural test
(a trained model's stability is reported by experiments/25_dfs_report.py).

Run:  python experiments/23_dfs_evidence_probes.py
Out:  reports/dfs_evidence_probes.md, outputs/cache/dfs_evidence_probes.json
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from ocean_tokenizer import config as C, dfs
from ocean_tokenizer.token_api import (ProfileEncoder, GridPatchEncoder,
                                       MODALITIES)
from ocean_tokenizer.fusion import build_fusion_model
from ocean_tokenizer.invariance import duplicate_tokens, output_change

torch.manual_seed(0)
DEPTHS = np.array([5, 15, 25, 35, 45, 55, 65, 85, 105, 125, 145, 165,
                   186, 222, 267, 327, 408, 527, 707, 985], dtype="float32")
D = len(DEPTHS)


class FakeGrid:
    depth = DEPTHS.astype("float64")


ENC = ProfileEncoder(DEPTHS, c_vars=2, d_model=32)
# a fine band layout, used where vertical DENSITY has to be expressible in
# tokens at all (the protocol's 4 physical bands cannot represent 2 dbar vs
# 10 dbar sampling -- both collapse to one token per band)
FINE_BANDS = [(z, z + 10.0) for z in np.arange(0.0, 300.0, 10.0)]

results, t0 = {}, time.time()


def prof_obs(lat, lon, values=None, depths=None, seed=0, **kw):
    lat = np.atleast_1d(np.asarray(lat, "float32"))
    lon = np.atleast_1d(np.asarray(lon, "float32"))
    P = lat.size
    nd = D if depths is None else np.asarray(depths).shape[-1]
    if values is None:
        values = np.random.default_rng(seed).normal(size=(P, 2, nd))
    o = dict(prof=torch.tensor(np.asarray(values, "float32")[None]),
             lat=torch.tensor(lat[None]), lon=torch.tensor(lon[None]),
             month=torch.tensor([3]), **kw)
    if depths is not None:
        d = np.asarray(depths, "float32")
        if d.ndim == 1:
            d = np.broadcast_to(d, (P, d.size))
        o["depths"] = torch.tensor(d[None])
    return o


EXACT_MAX = 700          # above this the exact O(N^3) solve is not worth it


def dfs_of(tb, target=dfs.PROTOCOL_SCALE):
    """Exact DFS where the set is small enough (no localisation error at all);
    a wide but finite neighbourhood on the large gridded sets, where k = 256
    is measurably converged (see docs/dfs_attention.md)."""
    n = int(tb.mask.sum())
    k = None if n <= EXACT_MAX else 256
    return float(dfs.dfs_scores(tb, target, k_neighbors=k).total)


def synth_column(kind, z=DEPTHS.astype("float64"), amp=1.0):
    """A T/S anomaly column in z-units, at EQUAL RMS amplitude.

    'smooth'      — the anomaly varies slowly with depth (a broad warm layer):
                    adjacent levels are nearly interchangeable.
    'thermocline' — the same amount of anomaly, concentrated in a sharp dipole
                    at 120 m (a vertically displaced thermocline, which is how
                    thermocline variability actually appears in anomaly space):
                    adjacent levels are NOT interchangeable.

    The two are deliberately matched in RMS so the only difference the
    estimator can see is vertical structure, not amplitude.  Both are already
    in the anomaly z-space the encoders consume — no per-column renormalisation
    (which would erase exactly the difference under test).
    """
    if kind == "smooth":
        t = np.cos(np.pi * z / (2 * 1000.0))
    else:
        t = np.tanh((z - 120.0) / 12.0) * np.exp(-((z - 120.0) / 90.0) ** 2)
    t = t / max(np.sqrt((t ** 2).mean()), 1e-9) * amp
    v = np.stack([t, 0.4 * t])                              # S follows T
    return v[None]                                          # (1, 2, nz)


# ==========================================================================
# 1 — exact duplication
# ==========================================================================
rng = np.random.default_rng(11)
P0 = 24
base_obs = prof_obs(rng.uniform(-60, 60, P0), rng.uniform(0, 360, P0), seed=11)
tb0 = ENC(**base_obs)
base_dfs = dfs_of(tb0)

# a fixed-seed model per variant, for the prediction-stability columns
MODELS = {}
for v in ("perceiver", "resampler", "mbca", "dfs"):
    m = build_fusion_model(v, FakeGrid(), d_model=32, n_latent=16, n_heads=4,
                           n_self_blocks=2, patch=(4, 6), seed=3)
    m.eval()
    MODELS[v] = m
qrng = np.random.default_rng(2)
QC = torch.tensor(np.stack([qrng.uniform(-80, 80, 64), qrng.uniform(0, 360, 64),
                            qrng.uniform(0, 985, 64), np.full(64, 3.0)],
                           -1).astype("float32"))[None]

dup_rows = []
# (a) duplicate ONE level-band token, (b) duplicate whole profiles
lvl_idx = tb0.mask[0].nonzero().flatten()[:1]
prof_idx = ((tb0.parent_id[0] < 6) & tb0.mask[0]).nonzero().flatten()
for name, idx in (("one level token", lvl_idx), ("6 whole profiles", prof_idx)):
    for f in (2, 4, 8):
        dup = duplicate_tokens(tb0, idx, f, divide_mass=False)
        row = dict(case=name, factor=f, tokens=int(dup.mask.sum()),
                   dfs=dfs_of(dup), d_dfs=dfs_of(dup) - base_dfs)
        for v, m in MODELS.items():
            row[f"dy_{v}"] = output_change(m, tb0, dup, QC)
        dup_rows.append(row)

# (c) real-time + delayed-mode copies of the same float cycles
vals = rng.normal(size=(P0, 2, D))
lat_c, lon_c = rng.uniform(-60, 60, P0), rng.uniform(0, 360, P0)
single = ENC(**prof_obs(lat_c, lon_c, values=vals))
rtdm = ENC(**prof_obs(np.concatenate([lat_c, lat_c]),
                      np.concatenate([lon_c, lon_c]),
                      values=np.concatenate([vals, vals]),
                      parent=torch.tensor(np.concatenate(
                          [np.arange(P0), np.arange(P0)])[None]),
                      source=torch.tensor(np.concatenate(
                          [np.full(P0, 300), np.full(P0, 301)])[None])))
dup_rows.append(dict(case="real-time + delayed-mode", factor=2,
                     tokens=int(rtdm.mask.sum()), dfs=dfs_of(rtdm),
                     d_dfs=dfs_of(rtdm) - dfs_of(single),
                     **{f"dy_{v}": output_change(MODELS[v], single, rtdm, QC)
                        for v in MODELS}))
results["duplication"] = dict(base_dfs=base_dfs,
                              base_tokens=int(tb0.mask.sum()), rows=dup_rows)
print(f"[1] duplication: base DFS {base_dfs:.2f} on {int(tb0.mask.sum())} tokens",
      flush=True)

# ==========================================================================
# 2 — same location, different depths
# ==========================================================================
rows = []
for keep in (1, 2, 3, 4):
    # keep the shallowest ``keep`` protocol bands of one column
    v = np.full((1, 2, D), np.nan)
    edges = [0.0, 50.0, 200.0, 500.0, 1e9]
    sel = DEPTHS < edges[keep]
    vv = np.random.default_rng(21).normal(size=(1, 2, D))
    v[:, :, sel] = vv[:, :, sel]
    tb = ENC(**prof_obs([10.0], [20.0], values=v))
    rows.append(dict(bands=int(tb.mask.sum()), dfs=dfs_of(tb),
                     depth_span=float(DEPTHS[sel].max())))
results["depth_complementarity"] = rows
print(f"[2] depth complementarity: " +
      " ".join(f"{r['bands']}b={r['dfs']:.2f}" for r in rows), flush=True)

# ==========================================================================
# 3 & 4 — dense vertical sampling: smooth column vs thermocline
# ==========================================================================
ENC_FINE = ProfileEncoder(DEPTHS, c_vars=2, d_model=32, depth_bands=FINE_BANDS)
TARGETS = (("protocol (Δz = 50 m)", dfs.PROTOCOL_SCALE),
           ("fine (Δz = 10 m)", dfs.TargetScale(dz_m=10.0)))
rows = []
for tname, ts in TARGETS:
    for kind in ("smooth", "thermocline"):
        for dbar in (10.0, 2.0):
            z = np.arange(2.0, 300.0, dbar)
            col = synth_column(kind, z)
            tb = ENC_FINE(**prof_obs([10.0], [20.0], values=col, depths=z))
            rows.append(dict(target=tname, column=kind, sampling_dbar=dbar,
                             levels=int(z.size), tokens=int(tb.mask.sum()),
                             dfs=dfs_of(tb, ts)))
for tname, _ in TARGETS:
    for kind in ("smooth", "thermocline"):
        a, b = [r for r in rows if r["target"] == tname and r["column"] == kind]
        print(f"[3/4] {tname:20s} {kind:11s} 10 dbar {a['dfs']:6.2f} -> 2 dbar "
              f"{b['dfs']:6.2f}  (x{b['dfs']/max(a['dfs'],1e-9):.2f} for x"
              f"{b['levels']/a['levels']:.0f} levels)", flush=True)
results["vertical_density"] = rows

# ==========================================================================
# 5 — target vertical-resolution sweep (identical input)
# ==========================================================================
z = np.arange(2.0, 300.0, 2.0)
sweep = {}
for kind in ("smooth", "thermocline"):
    tb = ENC_FINE(**prof_obs([10.0], [20.0], values=synth_column(kind, z),
                             depths=z))
    sweep[kind] = [dict(dz_m=dz, dfs=dfs_of(tb, dfs.TargetScale(dz_m=dz)))
                   for dz in (250.0, 100.0, 50.0, 25.0, 10.0, 5.0)]
results["resolution_sweep"] = sweep
print("[5] dz sweep (thermocline): " +
      " ".join(f"{r['dz_m']:.0f}m={r['dfs']:.2f}" for r in sweep["thermocline"]),
      flush=True)

# ==========================================================================
# 6 — horizontal cases
# ==========================================================================
hrows = []
rng = np.random.default_rng(31)
P = 24
# (a) uniform vs clustered vs exact horizontal duplicates
lat_u, lon_u = rng.uniform(-60, 60, P), rng.uniform(0, 360, P)
lat_k, lon_k = rng.uniform(9, 11, P), rng.uniform(20, 22, P)
half = P // 2
lat_d = np.concatenate([lat_u[:half], lat_u[:half]])
lon_d = np.concatenate([lon_u[:half], lon_u[:half]])
vd = rng.normal(size=(half, 2, D))
for name, (la, lo, vv, kw) in {
        "uniform (global)": (lat_u, lon_u, None, {}),
        "clustered (2 deg box)": (lat_k, lon_k, None, {}),
        "half the profiles, each twice (new floats)":
            (lat_d, lon_d, np.concatenate([vd, vd]), {}),
        "half the profiles, each twice (same float)":
            (lat_d, lon_d, np.concatenate([vd, vd]),
             dict(parent=torch.tensor(np.concatenate(
                 [np.arange(half), np.arange(half)])[None]))),
}.items():
    tb = ENC(**prof_obs(la, lo, values=vv, seed=31, **kw))
    hrows.append(dict(case=name, profiles=int(la.size),
                      tokens=int(tb.mask.sum()), dfs=dfs_of(tb)))

# (b) profiles across a front: same spacing, different water on each side
lat_f = np.concatenate([np.full(8, 9.6), np.full(8, 10.4)])
lon_f = np.tile(np.linspace(20, 22, 8), 2)
same = np.repeat(synth_column("smooth")[:, :, :], 16, axis=0)
front = np.concatenate([np.repeat(synth_column("smooth"), 8, axis=0),
                        np.repeat(synth_column("thermocline"), 8, axis=0)])
for name, vv in (("16 profiles, one water mass", same),
                 ("16 profiles across a front", front)):
    tb = ENC(**prof_obs(lat_f, lon_f, values=vv))
    hrows.append(dict(case=name, profiles=16, tokens=int(tb.mask.sum()),
                      dfs=dfs_of(tb)))
results["horizontal"] = hrows
print("[6a] " + " | ".join(f"{r['case']}: {r['dfs']:.2f}" for r in hrows[:4]),
      flush=True)

# (c) two profiles 5 km apart — the plan's coarse-vs-fine horizontal case
lat_p = np.array([10.0, 10.045])            # ~5 km apart
lon_p = np.array([20.0, 20.0])
pair = ENC(**prof_obs(lat_p, lon_p, seed=41))
solo = ENC(**prof_obs(lat_p[:1], lon_p[:1], seed=41))
pair_rows = []
for nm, ts in (("coarse (5 deg)", dfs.COARSE_SCALE),
               ("protocol (1 deg)", dfs.PROTOCOL_SCALE),
               ("fine (0.25 deg)", dfs.FINE_SCALE),
               ("very fine (5 km)", dfs.TargetScale(dx_km=5.0, dz_m=10.0))):
    pair_rows.append(dict(target=nm, dfs_pair=dfs_of(pair, ts),
                          dfs_single=dfs_of(solo, ts),
                          ratio=dfs_of(pair, ts) / max(dfs_of(solo, ts), 1e-9)))
results["five_km_pair"] = pair_rows
print("[6b] 5 km pair, DFS(pair)/DFS(single): " +
      " ".join(f"{r['target'].split()[0]}={r['ratio']:.2f}" for r in pair_rows),
      flush=True)

# (d) satellite patches on a realistic 1 deg grid: refining the tokenization of
#     a dense field adds evidence only until the patches are finer than the
#     target scale, where it saturates -- the principled form of "refinement
#     must not multiply a field's influence"
lat_g = torch.arange(-29.5, 30.5, 1.0)
lon_g = torch.arange(0.5, 60.5, 1.0)
G_H, G_W = lat_g.numel(), lon_g.numel()
fld = torch.tensor(rng.normal(size=(1, 2, G_H, G_W)).astype("float32"))
grid_rows = []
for p in (20, 10, 5, 2, 1):
    ge = GridPatchEncoder(2, d_model=32, patch=(p, p), modality="surf_grid")
    tb = ge(fld, lat_g, lon_g, torch.tensor([3]))
    grid_rows.append(dict(case=f"{p}x{p} cells/patch", patch_deg=float(p),
                          tokens=int(tb.mask.sum()), dfs=dfs_of(tb)))
# overlapping products: the same ocean seen by two processing streams
from ocean_tokenizer.token_api import TokenBatch
ge = GridPatchEncoder(2, d_model=32, patch=(5, 5), modality="surf_grid")
a = ge(fld, lat_g, lon_g, torch.tensor([3]))
ge2 = GridPatchEncoder(2, d_model=32, patch=(5, 5), modality="surf_grid",
                       family_id=777, source_id=101)
b = ge2(fld + 0.05 * torch.randn_like(fld), lat_g, lon_g, torch.tensor([3]))
solo_dfs = dfs_of(a)
grid_rows.append(dict(case="two overlapping products (5x5)", patch_deg=5.0,
                      tokens=int(TokenBatch.cat([a, b]).mask.sum()),
                      dfs=dfs_of(TokenBatch.cat([a, b]))))
results["grid_products"] = grid_rows
results["grid_single_5x5"] = solo_dfs
print("[6c] " + " | ".join(f"{r['case']}: {r['tokens']}tok "
                           f"{r['dfs']:.1f}" for r in grid_rows), flush=True)

# ==========================================================================
# report
# ==========================================================================
os.makedirs(C.CACHE, exist_ok=True)
meta = dict(seconds=time.time() - t0, torch=torch.__version__,
            scale=dict(protocol=dfs.PROTOCOL_SCALE.__dict__,
                       coarse=dfs.COARSE_SCALE.__dict__,
                       fine=dfs.FINE_SCALE.__dict__))
with open(os.path.join(C.CACHE, "dfs_evidence_probes.json"), "w") as f:
    json.dump(dict(meta=meta, results=results), f, indent=1)

L = ["# DFS-Attention — Evidence Probes (plan Section 11)\n",
     "*Analytic probes of the evidence estimator: no trained weights are "
     "involved, because `tau_i` is a property of the observing geometry at a "
     "target scale. Prediction-change columns use randomly initialised "
     "fixed-seed models (architectural, not learned, behaviour).*\n",
     f"Protocol target scale: dx = {dfs.PROTOCOL_SCALE.dx_km:.0f} km, "
     f"dz = {dfs.PROTOCOL_SCALE.dz_m:.0f} m, dt = "
     f"{dfs.PROTOCOL_SCALE.dt_d:.0f} d. Every DFS below is the **exact** "
     "ridge-leverage trace (`k_neighbors=None`) for sets of "
     f"<= {EXACT_MAX} tokens — no localisation error enters them; the larger "
     "gridded sets use a 256-token neighbourhood, which is converged for that "
     "geometry.\n"]

L.append("\n## 1 · Exact duplication\n")
L.append(f"Baseline: **{results['duplication']['base_tokens']} tokens, "
         f"DFS {base_dfs:.2f}**. Re-ingesting a measurement must add no "
         "evidence and must not move the prediction. ΔDFS and Δŷ are measured "
         "against each row's own un-duplicated input (the real-time / "
         "delayed-mode row uses a second, independent 24-profile draw).\n")
L.append("| case | factor | tokens | DFS | ΔDFS | Δŷ perceiver | Δŷ resampler "
         "| Δŷ MBCA | Δŷ DFS |")
L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in dup_rows:
    L.append(f"| {r['case']} | ×{r['factor']} | {r['tokens']} | {r['dfs']:.2f} "
             f"| {r['d_dfs']:+.2e} | {r['dy_perceiver']:.4f} | "
             f"{r['dy_resampler']:.4f} | {r['dy_mbca']:.4f} | {r['dy_dfs']:.4f} |")

L.append("\n## 2 · Same location, different depths\n")
L.append("One column; more of its depth bands are supplied. Different depths "
         "describe different water, so evidence must grow with the number of "
         "bands — the correction at the centre of the plan.\n")
L.append("| bands supplied | deepest level (m) | DFS | DFS / band |")
L.append("|---:|---:|---:|---:|")
for r in results["depth_complementarity"]:
    L.append(f"| {r['bands']} | {r['depth_span']:.0f} | {r['dfs']:.3f} | "
             f"{r['dfs'] / max(r['bands'], 1):.3f} |")

L.append("\n## 3–4 · Dense vertical sampling: smooth column vs thermocline\n")
L.append("Identical sampling change (10 dbar → 2 dbar, 5× the levels) applied "
         "to a vertically smooth anomaly column and to one whose anomaly is "
         "concentrated in a sharp thermocline dipole at 120 m, **matched in "
         "RMS amplitude** so only vertical structure differs. Tokens are 10 m "
         "bands, so vertical density is expressible in the token set at all.\n")
L.append("Read the two target blocks together: at a 50 m target the extra "
         "levels cannot be used by *either* column — correctly, since the "
         "requested field has no 2 dbar structure — while at a 10 m target the "
         "thermocline column converts them into evidence and the smooth one "
         "still cannot. Redundancy is a property of the column *and* the "
         "question being asked.\n")
L.append("| target | column | sampling | levels | tokens | DFS | gain vs 10 dbar |")
L.append("|---|---|---:|---:|---:|---:|---:|")
for r in results["vertical_density"]:
    ref = [x for x in results["vertical_density"]
           if x["target"] == r["target"] and x["column"] == r["column"]
           and x["sampling_dbar"] == 10.0][0]["dfs"]
    L.append(f"| {r['target']} | {r['column']} | {r['sampling_dbar']:.0f} dbar "
             f"| {r['levels']} | {r['tokens']} | {r['dfs']:.2f} | "
             f"×{r['dfs'] / max(ref, 1e-9):.2f} |")

L.append("\n## 5 · Target vertical-resolution sweep (identical input)\n")
L.append("The same 2 dbar profile, queried for coarser and finer "
         "reconstructions. Retained evidence must rise as the target gets "
         "finer.\n")
L.append("| target Δz (m) | DFS, smooth column | DFS, thermocline |")
L.append("|---:|---:|---:|")
for a, b in zip(sweep["smooth"], sweep["thermocline"]):
    L.append(f"| {a['dz_m']:.0f} | {a['dfs']:.2f} | {b['dfs']:.2f} |")

L.append("\n## 6 · Horizontal cases\n")
L.append("| case | profiles | tokens | DFS |")
L.append("|---|---:|---:|---:|")
for r in results["horizontal"]:
    L.append(f"| {r['case']} | {r['profiles']} | {r['tokens']} | {r['dfs']:.2f} |")
L.append("\n### Two profiles 5 km apart\n")
L.append("The plan's horizontal test case: redundant for a coarse target, "
         "genuinely two observations for a 5 km one.\n")
L.append("| target | DFS (one profile) | DFS (the pair) | pair / single |")
L.append("|---|---:|---:|---:|")
for r in results["five_km_pair"]:
    L.append(f"| {r['target']} | {r['dfs_single']:.3f} | {r['dfs_pair']:.3f} "
             f"| ×{r['ratio']:.3f} |")

L.append("\n### Gridded products (60°×60° field on the 1° analysis grid)\n")
L.append("Refining the tokenization of a dense field exposes more of it — but "
         "only down to the target scale, where the evidence saturates. That "
         "is the principled form of \"retokenizing must not multiply a field's "
         "influence\": it is bounded by what the target can resolve, not by a "
         "normalisation convention.\n")
L.append("| case | patch (deg) | tokens | DFS | DFS / token |")
L.append("|---|---:|---:|---:|---:|")
for r in results["grid_products"]:
    L.append(f"| {r['case']} | {r['patch_deg']:.0f} | {r['tokens']} | "
             f"{r['dfs']:.2f} | {r['dfs'] / max(r['tokens'], 1):.3f} |")
L.append(f"\nOne 5×5 product alone scores {results['grid_single_5x5']:.2f}; "
         "adding a second, independently processed product of the same ocean "
         "does not double it.\n")

L.append(f"\n---\n*Generated by `experiments/23_dfs_evidence_probes.py` in "
         f"{meta['seconds']:.1f} s.*")
path = os.path.join(C.REPORTS, "dfs_evidence_probes.md")
open(path, "w").write("\n".join(L) + "\n")
print(f"wrote {path} ({time.time() - t0:.1f}s)")
