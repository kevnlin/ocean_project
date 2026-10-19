"""The sealed CESM2 test: frozen checkpoints scored once on 2006-2007 of ``cesm2_long_test``.

No model was trained or selected on 2006-2007; ``110_cesm2_test_cohort.py`` drew those months after
every run had been set up. Two steps, each refusing to run twice:

  --register FAMILY:CHECKPOINT ...   after training: records every checkpoint (sha256, its run's
                                     summary), the test years and leads, and the sha256 of the
                                     cohort, the caches and the OI selection. FAMILY is "grid"
                                     (runner 101) or "reference" (runner 94, the OI-anchored local
                                     transformer, with its local OI-error scale table)
  --score                            verifies every recorded file, then scores each checkpoint on
                                     every cell of 2006-2007 at leads 0, 1, 3, 6, with the persisted
                                     OI of the input month on the same cells; writes results.json
                                     and never overwrites it

The OI selection is 44's on this cohort, tuned on 2004 (identical to the training cohort's). Scores
are runner 94's (``point_baselines.Scores``): RMSE in degC / PSU and J against the climatology.

  --check FAMILY:CHECKPOINT          engineering check that opens nothing: scores the checkpoint on
                                     2004-2005 of the test cohort (rows identical to training's) and
                                     requires its run's own 2004-2005 forecast scores back

  .venv/bin/python experiments/synthetic/109_sealed_cesm2_test.py --check grid:PATH
  .venv/bin/python experiments/synthetic/109_sealed_cesm2_test.py --register grid:PATH reference:PATH ...
  .venv/bin/python experiments/synthetic/109_sealed_cesm2_test.py --score
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUTPUT = ROOT / "outputs/cesm2_test_20261019"
REGISTRATION = OUTPUT / "registration.json"
RESULTS = OUTPUT / "results.json"
COHORT = "cesm2_long_test"
TEST_YEARS = (2006, 2007)
LEADS = (0, 1, 3, 6)
FAMILIES = ("grid", "reference")
DATA = {"cohort": ROOT / f"data/synthetic_argo/{COHORT}.nc",
        "satellites": OUTPUT / "satellites.npz", "surface_grids": OUTPUT / "surface_grids.npz",
        "lagged_surface": OUTPUT / "lagged_surface.npz",
        "oi_summary": ROOT / f"outputs/audit/synthetic/fixed_baselines_{COHORT}/summary.json",
        "scale_tables": ROOT / "outputs/long_record_20261017/scale_tables.npz"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def register(path, entries, data):
    """Write the registration once: ``entries`` are (family, checkpoint) pairs, ``data`` named files."""
    path = Path(path)
    if path.exists():
        raise SystemExit(f"{path} exists: the test is registered once")
    for family, _ in entries:
        if family not in FAMILIES:
            raise SystemExit(f"unknown family {family!r}; one of {FAMILIES}")
    record = {"created_unix": time.time(), "cohort": COHORT, "test_years": list(TEST_YEARS), "leads": list(LEADS),
              "cells": "every held-out cell of every test month",
              "entries": [{"family": family, "checkpoint": str(ckpt), "sha256": sha256(ckpt)} for family, ckpt in entries],
              "data": {name: {"path": str(p), "sha256": sha256(p)} for name, p in data.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=1) + "\n")
    os.replace(temporary, path)
    return record


def verify(record):
    """Every recorded file must still have its recorded content."""
    for item in [*record["entries"], *record["data"].values()]:
        path = Path(item.get("checkpoint", item.get("path")))
        if not path.exists() or sha256(path) != item["sha256"]:
            raise SystemExit(f"{path} changed or vanished since the registration")


def scenes(family, training, device):
    """The test cohort's forecast scenes as the checkpoint's runner built them."""
    import numpy as np
    from ocean_tokenizer import local_scale
    from ocean_tokenizer.forecast_scenes import ForecastScenes
    from ocean_tokenizer.grid_forecast_scenes import GridForecastScenes
    common = dict(leads=LEADS, lagged_surface=DATA["lagged_surface"], train_years=training["train_years"],
                  surface=True, satellite_cache=DATA["satellites"], oi_summary=DATA["oi_summary"], cohort=COHORT)
    if family == "grid":
        return GridForecastScenes(ROOT, device, surface_grids=DATA["surface_grids"], **common)
    data = ForecastScenes(ROOT, device, **common)
    if training["scale"] is not None:
        # the table is a function of position and calendar month fitted on 2000-2003; the test rows only add lookups
        with np.load(DATA["scale_tables"], allow_pickle=False) as tables:
            scale = local_scale.lookup(tables[training["scale"]], data.c.lat, data.c.lon, data.c.month_index % 12)
        data.configure(None, scale, "query")
    return data


def model_of(family, checkpoint, device):
    if family == "grid":
        from ocean_tokenizer.grid_kriging_ocean import GridKrigingConfig, GridKrigingModel
        model = GridKrigingModel(GridKrigingConfig(**checkpoint["config"]))
    else:
        from ocean_tokenizer.innovation_ocean import InnovationOceanConfig
        from ocean_tokenizer.query_scaled_innovation import QueryScaledInnovationModel
        model = QueryScaledInnovationModel(InnovationOceanConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def score_checkpoint(family, data, model, training, chunk=2048, years=TEST_YEARS):
    """Runner 94's full-cell evaluation on the test years: Scores of the model and of the persisted OI
    by lead and by year."""
    import numpy as np
    import torch
    from ocean_tokenizer.point_baselines import CH, Scores
    from ocean_tokenizer.synthetic_latent_experiment import EVAL_SEED
    analysis = data.analysis()
    evs = data.forecast_months(list(years), 0)
    new = lambda: Scores(data.levels, data.norm.std)
    model_scores, oi_scores = {}, {}
    context = training.get("eval_context_profiles") or training["context_profiles"]
    with torch.no_grad():
        for ev in evs:
            year = 2000 + ev.month // 12
            geometry = data.geometry(ev.source, ev.target[ev.profile], 32, ev.level)
            state, preds, bases = None, [], []
            for start in range(0, len(ev.level), chunk):
                stop = min(start + chunk, len(ev.level))
                rows, lev = ev.target[ev.profile[start:stop]], ev.level[start:stop]
                part = {k: v[start:stop] for k, v in geometry.items()}
                scene = data.scene(ev.source, rows, lev, analysis, context, [EVAL_SEED, ev.month], part)
                with torch.autocast(device_type=data.device.type, dtype=torch.bfloat16,
                                    enabled=training["amp"] and data.device.type == "cuda"):
                    if state is None:
                        state = model.encode(scene)
                    prediction = model.decode(scene, state)
                p = data.to_z(prediction["mean"].float(), rows, lev)
                if not torch.isfinite(p).all():
                    raise FloatingPointError(f"nonfinite prediction in month {ev.month}, lead {ev.lead}")
                preds.append(p.cpu().numpy())
                bases.append(data.to_z(scene["baseline"], rows, lev).cpu().numpy())
            p, b = np.concatenate(preds), np.concatenate(bases)
            y = data.targets(ev.target[ev.profile], ev.level)[0].cpu().numpy()
            for key in (ev.lead, (ev.lead, year)):
                model_scores.setdefault(key, new())
                oi_scores.setdefault(key, new())
                for j, ch in enumerate(CH):
                    model_scores[key].add(ch, p[:, j], y[:, j], ev.level)
                    oi_scores[key].add(ch, b[:, j], y[:, j], ev.level)
    name = lambda key: str(key) if isinstance(key, int) else f"{key[0]}_{key[1]}"
    return {"model": {name(k): s.result() for k, s in model_scores.items()},
            "persisted_oi": {name(k): s.result() for k, s in oi_scores.items()},
            "months": sorted({ev.month for ev in evs}), "cells_per_lead": {str(lead): int(sum(
                len(ev.level) for ev in evs if ev.lead == lead)) for lead in LEADS}}


def check(family, path, device):
    """Score a checkpoint on 2004-2005 of the test cohort; its run's summary must come back."""
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    training = checkpoint["training"]
    summary = json.loads((Path(path).parent / f"summary_seed{checkpoint['seed']}.json").read_text())
    if summary["best_step"] != checkpoint["step"]:
        raise SystemExit("the checkpoint is not the run's chosen one")
    data = scenes(family, training, device)
    outcome = score_checkpoint(family, data, model_of(family, checkpoint, data.device), training, years=(2004, 2005))
    worst = 0.
    for lead in map(str, LEADS):
        for ch in ("TEMP", "SALT"):
            mine = outcome["model"][lead][ch]["rmse_physical"]
            theirs = summary["forecast"]["best"]["by_lead"][lead]["scores"][ch]["rmse_physical"]
            oi_mine = outcome["persisted_oi"][lead][ch]["rmse_physical"]
            oi_theirs = summary["forecast"]["best"]["by_lead"][lead]["baseline_scores"][ch]["rmse_physical"]
            worst = max(worst, abs(mine / theirs - 1), abs(oi_mine / oi_theirs - 1))
            print(f"lead {lead} {ch}: model {mine:.5f} (run {theirs:.5f}), persisted OI {oi_mine:.5f} (run {oi_theirs:.5f})")
    if worst > 1e-3:
        raise SystemExit(f"the test cohort does not reproduce the run's 2004-2005 scores (relative {worst:.2g})")
    print(f"check passed: 2004-2005 reproduced to a relative {worst:.2g}")


def score(device):
    import torch
    if RESULTS.exists():
        raise SystemExit(f"{RESULTS} exists: the test is scored once")
    record = json.loads(REGISTRATION.read_text())
    verify(record)
    results, built = {"registration_sha256": sha256(REGISTRATION), "started_unix": time.time(), "entries": []}, {}
    for entry in record["entries"]:
        checkpoint = torch.load(entry["checkpoint"], map_location="cpu", weights_only=True)
        training, family = checkpoint["training"], entry["family"]
        key = (family, training["train_years"], training.get("scale"))
        if key not in built:
            built.clear()
            built[key] = scenes(family, training, device)
        data = built[key]
        if any(not (torch.as_tensor(checkpoint["normalization"][kind][ch]).numpy() == getattr(data.norm, kind)[ch]).all()
               for kind in ("mean", "std") for ch in ("TEMP", "SALT")):
            raise SystemExit(f"{entry['checkpoint']}: the test cohort's normalisation differs from training")
        started = time.time()
        outcome = score_checkpoint(family, data, model_of(family, checkpoint, data.device), training)
        results["entries"].append({**entry, "tag": training["tag"], "seed": checkpoint["seed"],
                                   "step": checkpoint["step"], "seconds": time.time() - started, **outcome})
        print(json.dumps({"scored": training["tag"], "lead_rmse": {
            lead: [round(outcome["model"][lead][ch]["rmse_physical"], 4) for ch in ("TEMP", "SALT")]
            for lead in map(str, LEADS)}}), flush=True)
        del checkpoint
    temporary = RESULTS.with_suffix(".tmp")
    temporary.write_text(json.dumps(results, indent=1, default=float) + "\n")
    os.replace(temporary, RESULTS)
    print(f"wrote {RESULTS}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--register", nargs="+", metavar="FAMILY:CHECKPOINT")
    p.add_argument("--score", action="store_true")
    p.add_argument("--check", metavar="FAMILY:CHECKPOINT")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if sum(map(bool, (args.register, args.score, args.check))) != 1:
        raise SystemExit("pass one of --register, --score, --check")
    if args.check:
        family, path = args.check.split(":", 1)
        check(family, path, args.device)
    elif args.register:
        entries = [(item.split(":", 1)[0], Path(item.split(":", 1)[1])) for item in args.register]
        record = register(REGISTRATION, entries, DATA)
        print(f"registered {len(record['entries'])} checkpoints -> {REGISTRATION}")
    else:
        score(args.device)


if __name__ == "__main__":
    main()
