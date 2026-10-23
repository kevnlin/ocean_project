"""The "reference analyses" rows: GLORYS12 and ARMOR3D next to the paper's rows, and the manifest.

Reads the scores ``160_reference_products.py`` wrote for every product and period, the sealed
tests' results (read-only: the paper's KA-Ocean, persisted-OI and climatology rows and the cell
counts, exactly the files ``122_paper_tables.py`` reads) and the three scratch seeds' summaries
for the 2021 validation, and writes

  outputs/reference_products_20261021/reference_rows.md   the tables, the paper's column order
  outputs/reference_products_20261021/manifest.json       products, months, method, counts

Both products assimilate every Argo profile, including the held-out floats, so every table
says: reference analyses, never fair baselines.

  .venv/bin/python experiments/real_data/161_reference_rows.py
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ocean_tokenizer.copernicus_reference import PRODUCTS      # noqa: E402

OUTPUT = ROOT / "outputs/reference_products_20261021"
SEALED = {"2024": Path("/DATA2/linke/ocean_project_ka_dev4/outputs/real_2024_test/sealed/results.json"),
          "2022-2023": Path("/DATA2/linke/ocean_project_ka_dev3/outputs/real_global_20261019/sealed/results.json")}
SUMMARIES_2021 = sorted(glob.glob("/DATA2/linke/ocean_project_ka_dev/outputs/real_global_20261019/scratch_s123?/summary_seed*.json"))
LEADS = ("0", "1", "3", "6")
CH = ("TEMP", "SALT")
BANDS = ("0-100m", "100-300m", "300-700m", "700-1400m")
NAMES = {"glorys12": "GLORYS12", "armor3d": "ARMOR3D"}
NOTE = ("Both products assimilate every Argo profile, including the held-out floats scored here, so they are "
        "**reference analyses, never fair baselines**: they have seen the answer.")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def fmt(ch, v):
    return "--" if v is None or not np.isfinite(v) else (f"{v:.3f}" if ch == "TEMP" else f"{v:.4f}")


def paper_rows(period):
    """KA-Ocean (mean of the scratch seeds), persisted OI and climatology of a sealed test, by lead."""
    r = json.loads(SEALED[period].read_text())
    first, scratch = r["entries"][0], [e for e in r["entries"] if e["tag"].startswith("scratch_s")]
    rows = {}
    for lead in LEADS:
        rows[lead] = {"cells": first["by_lead"][lead]["cells"]}
        for ch in CH:
            a = first["by_lead"][lead]["all"][ch]
            rows[lead][ch] = {"n": a["model"]["n"],
                              "ka_ocean": {k: float(np.mean([e["by_lead"][lead]["all"][ch]["model"][k] for e in scratch]))
                                           for k in ("rmse", "mae", "mean_bias")},
                              "persisted_oi": {k: a["persisted_oi"][k] for k in ("rmse", "mae", "mean_bias")},
                              "climatology": {k: a["climatology"][k] for k in ("rmse", "mae", "mean_bias")},
                              "persisted_oi_by_band": {b: a["persisted_oi"]["by_band"][b]["rmse"] for b in BANDS},
                              "climatology_by_band": {b: a["climatology"]["by_band"][b]["rmse"] for b in BANDS},
                              "ka_ocean_by_band": {b: float(np.mean([e["by_lead"][lead]["all"][ch]["model"]["by_band"][b]["rmse"]
                                                                     for e in scratch])) for b in BANDS}}
    return rows, {"path": str(SEALED[period]), "sha256": sha256(SEALED[period]), "ka_ocean_seeds": [e["tag"] for e in scratch]}


def rows_2021():
    ss = [json.loads(Path(p).read_text()) for p in SUMMARIES_2021]
    rows = {}
    for lead in LEADS:
        rows[lead] = {}
        for ch in CH:
            b = [s["forecast"]["best"]["by_lead"][lead] for s in ss]
            rows[lead][ch] = {"n": b[0]["baseline_scores"][ch]["n"],
                              "persisted_oi": {"rmse": b[0]["baseline_scores"][ch]["rmse_physical"]},
                              "ka_ocean": {"rmse": float(np.mean([x["scores"][ch]["rmse_physical"] for x in b]))},
                              "persisted_oi_by_band": b[0]["baseline_scores"][ch]["by_band_physical"]}
    return rows


def main():
    results = {p: {per: json.loads((OUTPUT / f"{p}_{per}.json").read_text())
                   for per in ("2021", "2024", "2022-2023") if (OUTPUT / f"{p}_{per}.json").exists()}
               for p in PRODUCTS}
    lines = ["# Reference analyses at the held-out real Argo cells", "",
             NOTE, "",
             "RMSE of the anomaly in physical units on every held-out cell of the period, the paper's recipe "
             "(WOA23 anomaly at the profile's position, 2016-2020 z-scoring, cells with a finite target; "
             "`sealed_scoring.metrics`). Leads 1, 3, 6 of a product are its anomaly of the input month (field minus WOA23 of "
             "that calendar month at the float's position) persisted to the target month, as the paper's persisted-OI row "
             "persists the OI anomaly; lead 0 is its analysis of the target month. "
             "KA-Ocean, persisted OI and climatology are copied from the sealed results the paper tables are "
             "generated from (KA-Ocean: mean of seeds 1234-1236). Columns: T 0/1/3/6, S 0/1/3/6.", ""]
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "role": NOTE.replace("**", ""),
                "products": {}, "sealed_sources": {}, "periods": {}, "method": None}
    header = "| Row | T 0 | T 1 | T 3 | T 6 | S 0 | S 1 | S 3 | S 6 |\n|---|---|---|---|---|---|---|---|---|"

    def product_cells(res, lead, ch):
        return res["by_lead"][lead][ch]

    for period, title in (("2024", "Real Argo, sealed test 2024"), ("2022-2023", "Real Argo, held-out test 2022-2023")):
        paper, source = paper_rows(period)
        manifest["sealed_sources"][period] = source
        lines += [f"## {title}", "", f"Cells: {paper['0']['cells']:,} held-out cells per lead "
                  f"(finite targets: T {paper['0']['TEMP']['n']:,}, S {paper['0']['SALT']['n']:,}), from the sealed results.", "", header]
        for name, key in (("Climatology (paper)", "climatology"), ("Persisted OI (paper)", "persisted_oi"), ("KA-Ocean (paper, mean of 3 seeds)", "ka_ocean")):
            lines.append(f"| {name} | " + " | ".join(fmt(ch, paper[l][ch][key]["rmse"]) for ch in CH for l in LEADS) + " |")
        for p, res in results.items():
            if period not in res:
                lines.append(f"| {NAMES[p]} | " + " | ".join("pending" for _ in range(8)) + " |")
                continue
            r = res[period]
            lines.append(f"| {NAMES[p]} reference analysis (0) / persisted analysis (1,3,6) | "
                         + " | ".join(fmt(ch, product_cells(r, l, ch)["product"]["rmse"]) if l in r["by_lead"] else "--"
                                      for ch in CH for l in LEADS) + " |")
        lines += ["", "MAE / bias (product minus float) on the same cells:", "", header]
        for name, key in (("Persisted OI MAE (paper)", ("persisted_oi", "mae")), ("Persisted OI bias (paper)", ("persisted_oi", "mean_bias")),
                          ("KA-Ocean MAE (paper)", ("ka_ocean", "mae")), ("KA-Ocean bias (paper)", ("ka_ocean", "mean_bias"))):
            lines.append(f"| {name} | " + " | ".join(fmt(ch, paper[l][ch][key[0]][key[1]]) for ch in CH for l in LEADS) + " |")
        for p, res in results.items():
            if period not in res:
                continue
            r = res[period]
            for label, key in (("MAE", "mae"), ("bias", "mean_bias")):
                lines.append(f"| {NAMES[p]} {label} | " + " | ".join(fmt(ch, product_cells(r, l, ch)["product"][key]) if l in r["by_lead"] else "--"
                                                                   for ch in CH for l in LEADS) + " |")
        lines += ["", "RMSE by depth band (rows: lead 0 and lead 6; product cells where it is finite):", "",
                  "| Row | " + " | ".join(f"T {b}" for b in BANDS) + " | " + " | ".join(f"S {b}" for b in BANDS) + " |",
                  "|---|" + "---|" * 8]
        for lead in ("0", "6"):
            lines.append(f"| Climatology (paper) | " + " | ".join(fmt(ch, paper[lead][ch]["climatology_by_band"][b]) for ch in CH for b in BANDS) + " |")
            lines.append(f"| Persisted OI lead {lead} (paper) | " + " | ".join(fmt(ch, paper[lead][ch]["persisted_oi_by_band"][b]) for ch in CH for b in BANDS) + " |")
            lines.append(f"| KA-Ocean lead {lead} (paper) | " + " | ".join(fmt(ch, paper[lead][ch]["ka_ocean_by_band"][b]) for ch in CH for b in BANDS) + " |")
            for p, res in results.items():
                if period in res and lead in res[period]["by_lead"]:
                    r = res[period]
                    lines.append(f"| {NAMES[p]} lead {lead} | " + " | ".join(fmt(ch, product_cells(r, lead, ch)["product"]["by_band"][b]["rmse"])
                                                                          for ch in CH for b in BANDS) + " |")
        counts = {}
        for p, res in results.items():
            if period in res:
                counts[p] = {l: {ch: {"scored": product_cells(res[period], l, ch)["product"]["n"],
                                      "finite_target": product_cells(res[period], l, ch)["cells_finite_target"],
                                      "dropped_product_missing": product_cells(res[period], l, ch)["cells_dropped_product_missing"],
                                      "sealed_n": paper[l][ch]["n"],
                                      "climatology_rmse_on_product_cells": product_cells(res[period], l, ch)["climatology_same_cells"]["rmse"],
                                      "climatology_rmse_all_cells": product_cells(res[period], l, ch)["climatology_all_cells"]["rmse"],
                                      "climatology_rmse_sealed": paper[l][ch]["climatology"]["rmse"]}
                                 for ch in CH} for l in LEADS if l in res[period]["by_lead"]}
                counts[p]["cells_per_lead"] = {l: res[period]["by_lead"][l]["cells"] for l in LEADS if l in res[period]["by_lead"]}
                counts[p]["sealed_cells_per_lead"] = {l: paper[l]["cells"] for l in LEADS}
        manifest["periods"][period] = {"sealed_cells": {l: paper[l]["cells"] for l in LEADS},
                                       "sealed_finite": {l: {ch: paper[l][ch]["n"] for ch in CH} for l in LEADS}, "products": counts}
        lines += ["", "Cells scored (product finite) / finite-target cells (= sealed test's n) and cells dropped because the product is missing there:", ""]
        for p, cnt in counts.items():
            for l in LEADS:
                if l in cnt:
                    lines.append(f"- {NAMES[p]} lead {l}: " + "; ".join(
                        f"{ch} {cnt[l][ch]['scored']:,} / {cnt[l][ch]['finite_target']:,} (sealed n {cnt[l][ch]['sealed_n']:,}), "
                        f"dropped {cnt[l][ch]['dropped_product_missing']:,}; climatology RMSE on the product's cells "
                        f"{fmt(ch, cnt[l][ch]['climatology_rmse_on_product_cells'])} vs sealed {fmt(ch, cnt[l][ch]['climatology_rmse_sealed'])}"
                        for ch in CH))
        if period == "2022-2023":
            lines += ["", "By year (product RMSE, lead 0 / 1 / 3 / 6):", ""]
            for p, res in results.items():
                if period in res:
                    for y in ("2022", "2023"):
                        lines.append(f"- {NAMES[p]} {y}: T " + " / ".join(fmt("TEMP", res[period]["by_lead_by_year"][l][y]["TEMP"]["product"]["rmse"]) for l in LEADS)
                                     + "; S " + " / ".join(fmt("SALT", res[period]["by_lead_by_year"][l][y]["SALT"]["product"]["rmse"]) for l in LEADS))
        lines.append("")

    # 2021 validation
    v = rows_2021()
    lines += ["## Validation on 2021 (selection year, not a test year)", "",
              "Every held-out cell of 2021. Persisted OI and KA-Ocean: the scratch runs' own 2021 scores (summary_seed*.json, "
              "the numbers `133 --check` reproduces); climatology: computed here on the same cells (a zero anomaly).", "", header]
    first = next((res["2021"] for res in results.values() if "2021" in res), None)
    if first is not None:
        lines.append("| Climatology (computed, all finite-target cells) | " + " | ".join(fmt(ch, first["by_lead"][l][ch]["climatology_all_cells"]["rmse"]) for ch in CH for l in LEADS) + " |")
    lines.append("| Persisted OI (runs' 2021) | " + " | ".join(fmt(ch, v[l][ch]["persisted_oi"]["rmse"]) for ch in CH for l in LEADS) + " |")
    lines.append("| KA-Ocean (runs' 2021, mean of 3 seeds) | " + " | ".join(fmt(ch, v[l][ch]["ka_ocean"]["rmse"]) for ch in CH for l in LEADS) + " |")
    for p, res in results.items():
        if "2021" in res:
            lines.append(f"| {NAMES[p]} | " + " | ".join(fmt(ch, res["2021"]["by_lead"][l][ch]["product"]["rmse"]) for ch in CH for l in LEADS) + " |")
            lines.append(f"| {NAMES[p]} MAE | " + " | ".join(fmt(ch, res["2021"]["by_lead"][l][ch]["product"]["mae"]) for ch in CH for l in LEADS) + " |")
            lines.append(f"| {NAMES[p]} bias | " + " | ".join(fmt(ch, res["2021"]["by_lead"][l][ch]["product"]["mean_bias"]) for ch in CH for l in LEADS) + " |")
    lines += ["", "Cell counts, 2021 (scored / finite-target; runs' n):", ""]
    for p, res in results.items():
        if "2021" in res:
            for l in LEADS:
                b = res["2021"]["by_lead"][l]
                lines.append(f"- {NAMES[p]} lead {l}: " + "; ".join(f"{ch} {b[ch]['product']['n']:,} / {b[ch]['cells_finite_target']:,} (runs' n {v[l][ch]['n']:,}), dropped {b[ch]['cells_dropped_product_missing']:,}" for ch in CH))
    manifest["validation_2021"] = {"runs": SUMMARIES_2021, "rows": v,
                                   "products": {p: {l: {ch: res["2021"]["by_lead"][l][ch]["product"] | {"cells_finite_target": res["2021"]["by_lead"][l][ch]["cells_finite_target"],
                                                                                                        "dropped": res["2021"]["by_lead"][l][ch]["cells_dropped_product_missing"]}
                                                       for ch in CH} for l in LEADS}
                                                for p, res in results.items() if "2021" in res}}
    lines += ["", f"Generated by experiments/real_data/161_reference_rows.py on {manifest['created']}.", ""]
    (OUTPUT / "reference_rows.md").write_text("\n".join(lines))

    any_result = next(iter(next(iter(results.values())).values()))
    manifest["method"] = {k: any_result[k] for k in ("cells", "target", "product_anomaly", "leads", "levels_m", "normalization", "cohort")}
    manifest["method"]["interpolation"] = ("bilinear in latitude/longitude on the product's native grid (periodic in longitude), "
                                           "then linear in depth onto the cohort's 20 levels, no extrapolation; a missing corner or a level "
                                           "outside the finite part of the column leaves the cell missing (dropped)")
    manifest["method"]["units"] = {"TEMP": "degC", "SALT": "PSU (products: 1e-3)"}
    manifest["method"]["note_2024"] = ("the 2024 held-out values were read again: a declared extra pass on an already-opened test; "
                                       "no registration or results file of a sealed test was touched")
    manifest["method"]["validation"] = ("2021 (selection year): the product's RMSE against the runs' persisted OI and the climatology on "
                                        "the same cells; the sampler was checked against xarray's own interpolation (max |diff| 1.5e-5 degC, "
                                        "the int16 packing) and the z-space error against the absolute difference (1e-15)")
    for p, res in results.items():
        spec = PRODUCTS[p]
        months = {}
        for per, r in res.items():
            for d in r["downloads"]:
                months.setdefault(d["month"], {"dataset_id": d["dataset_id"], "file_bytes": d["file_bytes"],
                                               "copernicusmarine_data_transfer_size_mb": d["transfer_mb"], "periods": []})
                months[d["month"]]["periods"].append(per)
        sampled = {}
        for per in res:
            with np.load(OUTPUT / "cache" / f"{p}_{per}.npz") as z:
                for key in z.files:
                    src, tgt, _ = key.split(":")
                    sampled.setdefault(str(np.datetime64("2000-01", "M") + int(src)), set()).add(str(np.datetime64("2000-01", "M") + int(tgt)))
        sampled = {k: sorted(v) for k, v in sorted(sampled.items())}
        manifest["products"][p] = {"name": NAMES[p], "product_id": spec["product_id"], "dataset_id": spec["dataset_id"],
                                   "months_sampled": sampled, "n_months_sampled": len(sampled),
                                   "months_sampled_note": "source month -> target months served, from the sampled-column caches; "
                                                          "every one of these months was downloaded once (0-1100 m, T and S) and deleted; "
                                                          "months_downloaded lists only the downloads recorded by the last scoring run of each "
                                                          "period (a re-score from the cache records none)",
                                   "nrt_dataset_id": spec.get("nrt_dataset_id"), "nrt_used": any(m["dataset_id"] == spec.get("nrt_dataset_id") for m in months.values()),
                                   "variables": spec["variables"], "description": spec["description"],
                                   "depth_cut_m": 1100, "months_downloaded": dict(sorted(months.items())),
                                   "n_months_downloaded": len(months), "kept_on_disk": "nothing: each month was deleted after sampling; "
                                   "the sampled columns at the held-out profiles (~126 MB) are cached under outputs/reference_products_20261021/cache "
                                   "so a period can be re-scored without downloading; deletable",
                                   "score_files": {per: str(OUTPUT / f"{p}_{per}.json") for per in res},
                                   "caveat": ("GLORYS12: the dataset id served 1993-01..2026-08 (version 202311); the Copernicus catalogue "
                                              "documents the reanalysis proper to mid-2021 and an interim extension after it, which the "
                                              "metadata read here does not separate, so 2022-2024 may be the interim product"
                                              if p == "glorys12" else
                                              "ARMOR3D: multi-year (REP) dataset through 2024-12; the NRT continuation was not needed")}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float) + "\n")
    print(f"wrote {OUTPUT / 'reference_rows.md'} and {OUTPUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
