"""Score the adapted Senseiver at held-out Argo float positions (plan P4).

`48_senseiver.py --stage argo` trains the authors' model on a GODAS field whose
sensors sit at real Argo cells. This carries its predictions through the SAME
held-out-float evaluation every DFS row uses, so the Senseiver finally has a row
in the P0 table rather than only a training curve.

What is actually being measured, stated plainly
-----------------------------------------------
The Senseiver reconstructs a **dense field** and must be trained on dense
snapshots, so the field it learns is GODAS, not the ocean. Scoring it against
held-out floats therefore asks: *how close does a GODAS emulator come to real
measurements it never saw?* That is the fair question given the architecture's
requirement, and it is not the same question the DFS rows answer — they are
trained against float measurements directly.

That asymmetry is the result, not a nuisance: **the Senseiver needs a gridded
training field and the DFS rows do not.** The number below should be read with
that in front of it.

Every line of model code executed is the authors'. This script only loads their
checkpoint, calls their `test()`, and scores the output.

  .venv/bin/python experiments/real_data/56_senseiver_score.py --region gulfstream
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch

from ocean_tokenizer import protocol as P
from ocean_tokenizer.argo_obs import ArgoCohort
from ocean_tokenizer.clustered_ci import accumulate, ci_rmse
from ocean_tokenizer.godas import load_godas, GodasNorm

ap = argparse.ArgumentParser()
ap.add_argument("--region", default="gulfstream")
ap.add_argument("--repo", default=None)
ap.add_argument("--version", type=int, default=None,
                help="lightning_logs version; default = newest godas_argo run")
ap.add_argument("--eval-split", default="development")
ap.add_argument("--num-pix", type=int, default=512)
ap.add_argument("--output", default=None)
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO = args.repo or os.path.join(ROOT, "external", "Senseiver")
OUT = args.output or os.path.join(ROOT, "outputs", f"senseiver_{args.region}")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

# ---- find the godas_argo run -------------------------------------------
import yaml
cands = []
for d in sorted(glob.glob(os.path.join(REPO, "lightning_logs", "version_*"))):
    hp = os.path.join(d, "hparams.yaml")
    if not os.path.exists(hp):
        continue
    h = yaml.safe_load(open(hp)) or {}
    if h.get("data_name") == "godas_argo":
        ck = sorted(glob.glob(os.path.join(d, "checkpoints", "*.ckpt")))
        if ck:
            cands.append((int(d.rsplit("_", 1)[1]), d, ck[-1], h))
if not cands:
    raise SystemExit("no trained godas_argo run under lightning_logs; run "
                     "experiments/real_data/48_senseiver.py --stage argo first")
ver, vdir, ckpt, hp = (cands[-1] if args.version is None
                       else next(c for c in cands if c[0] == args.version))
print(f"senseiver run version_{ver}\n  {os.path.relpath(ckpt, ROOT)}", flush=True)

# ---- rebuild the authors' dataloader and model -------------------------
cwd = os.getcwd()
sys.path.insert(0, REPO)
os.chdir(REPO)
try:
    from dataloaders import senseiver_dataloader
    from network_light import Senseiver
    dl = senseiver_dataloader(dict(hp), num_workers=0)
    # Their `test()` reads dataset.data and pos_encodings directly, which live
    # on CPU, while Lightning restores the module to the device it trained on.
    # The field is 988 pixels, so CPU inference is cheap and keeps every tensor
    # on one device without touching their code.
    model = Senseiver.load_from_checkpoint(ckpt, **dict(hp)).cpu()
    model.eval()
    data = dl.dataset.data
    print(f"  field {tuple(data.shape)}  sensors {len(dl.dataset.sensors)}",
          flush=True)
    with torch.no_grad():
        recon = model.test(dl, num_pix=args.num_pix, split_time=8)
finally:
    os.chdir(cwd)
recon = np.asarray(recon.cpu().numpy())          # (T, H, W, 2*Z) in GODAS z
print(f"  reconstruction {recon.shape}  ({time.time()-t0:.0f}s)", flush=True)

# ---- de-normalise and score at held-out float positions ----------------
fields = load_godas(os.path.join(ROOT, "data", f"godas_{args.region}"))
months = fields["months"]
year = months.astype("datetime64[Y]").astype(int) + 1970
tr = np.where((year >= 2000) & (year <= 2018))[0]
norm = GodasNorm(fields, slice(int(tr[0]), int(tr[-1]) + 1))
Z = fields["TEMP"].shape[1]
mi_of_t = (months.astype("datetime64[M]").astype(int)
           - np.datetime64("2000-01", "M").astype(int))
pos = {int(m): i for i, m in enumerate(mi_of_t)}

c = ArgoCohort.load(os.path.join(ROOT, "data", "argo_cohort", f"{args.region}.nc"))
ev = [int(m) for m in c.months_in(args.eval_split)
      if c.month(m, float_split="heldout_float").size and int(m) in pos]
labs = {ch: [] for ch in P.CHANNELS}
errs = {ch: [] for ch in P.CHANNELS}
for m in ev:
    t = pos[int(m)]
    rows = c.month(m, float_split="heldout_float")
    gy, gx = c.grid_y[rows], c.grid_x[rows]
    zc = recon[t][gy, gx]                          # (R, 2*Z) in z space
    for j, ch in enumerate(P.CHANNELS):
        zz = zc[:, j * Z:(j + 1) * Z]                     # (R, Z)
        # GodasNorm.unz expects whole (T, Z, Y, X) fields; here we hold point
        # samples at specific cells, so the inverse is applied directly:
        # clim is (12, Z, Y, X) per calendar month and scale is (Z,).
        cal = int(str(np.datetime64(months[t], "M"))[5:7])
        clim_col = norm.clim[ch][cal - 1][:, gy, gx].T     # (R, Z)
        pred = zz * np.asarray(norm.scale[ch]).reshape(1, -1) + clim_col
        truth = (c.TEMP if ch == "TEMP" else c.SALT)[rows]
        ok = np.isfinite(pred) & np.isfinite(truth)
        if ok.any():
            wm = np.repeat(c.wmo[rows], Z).reshape(rows.size, Z)
            labs[ch].append(wm[ok]); errs[ch].append((pred[ok] - truth[ok]) ** 2)

# GODAS itself, scored at the same held-out float positions. The Senseiver is
# trained to reproduce GODAS, so this is the ceiling it cannot beat: any gap
# between GODAS and the floats is inherited, not the model's error.
ceil_labs = {ch: [] for ch in P.CHANNELS}
ceil_errs = {ch: [] for ch in P.CHANNELS}
for m in ev:
    t = pos[int(m)]
    rows = c.month(m, float_split="heldout_float")
    gy, gx = c.grid_y[rows], c.grid_x[rows]
    for ch in P.CHANNELS:
        g = np.asarray(fields[ch])[t][:, gy, gx].T          # (R, Z)
        truth = (c.TEMP if ch == "TEMP" else c.SALT)[rows]
        ok = np.isfinite(g) & np.isfinite(truth)
        if ok.any():
            wm = np.repeat(c.wmo[rows], Z).reshape(rows.size, Z)
            ceil_labs[ch].append(wm[ok])
            ceil_errs[ch].append((g[ok] - truth[ok]) ** 2)

res = {"version": ver, "checkpoint": os.path.relpath(ckpt, ROOT),
       "months": len(ev), "channels": {}, "godas_vs_argo_ceiling": {}}
for ch in P.CHANNELS:
    if ceil_labs[ch]:
        st = accumulate(np.concatenate(ceil_labs[ch]),
                        np.concatenate(ceil_errs[ch]),
                        cluster_unit="wmo", channel=ch, method="godas")
        res["godas_vs_argo_ceiling"][ch] = {"rmse": st.rmse(),
                                            "n_wmos": st.n_clusters}
        print(f"  GODAS itself vs the same floats — {ch}: {st.rmse():.4f}",
              flush=True)
for ch in P.CHANNELS:
    if not labs[ch]:
        continue
    st = accumulate(np.concatenate(labs[ch]), np.concatenate(errs[ch]),
                    cluster_unit="wmo", channel=ch, method="senseiver")
    iv = ci_rmse(st, n_boot=4000, seed=20260907)
    res["channels"][ch] = {"rmse": st.rmse(), "n_wmos": st.n_clusters,
                           "n": int(st.n.sum()), "ci_wmo": iv.to_dict()}
    print(f"  {ch}: RMSE {st.rmse():.4f} over {st.n_clusters} held-out floats "
          f"[{iv.lo:.4f}, {iv.hi:.4f}]", flush=True)

art = P.ResultArtifact(
    package="P4", track="A", region=args.region, results=res,
    seeds=[int(hp.get("seed", 123))], command=" ".join(sys.argv),
    counts={"months": len(ev), "eval_split": args.eval_split},
    warnings=["The Senseiver is trained to reproduce a GRIDDED field (GODAS) "
              "because its architecture requires dense training snapshots. "
              "This score therefore measures how closely a GODAS emulator "
              "matches real held-out floats, which is not the same question "
              "the DFS rows answer."],
    notes="Authors' model and inference; this script only loads the checkpoint "
          "and scores the output at held-out float positions."
    ).finalize(ROOT)
p_ = os.path.join(OUT, f"artifact_P4_senseiver_{args.region}.json")
print(f"\nartifact: {p_}\n  sha256 {art.write(p_)}\ntotal {time.time()-t0:.0f}s")
