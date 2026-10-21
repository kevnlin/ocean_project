"""Queue of the iHESP 0.1-degree runs (2026-10-21; spec docs/superpowers/specs/2026-10-21-ihesp-sealed-test.md).

The CESM2 setup of ``c_y4`` on the iHESP cohort: 2000-2003 training, checkpoint chosen on
2004-2005, 70,000 steps, leads 0/1/3/6, 6,080 input profiles a month. Four read-outs, seeds
1234-1236: KA-Ocean (A), the ARMOR3D-style baseline, the Aardvark-style model and the softmax
read-out. No run reads 1997-1999 (the sealed test). Everything else is the queue of ``114``.

  .venv/bin/python experiments/real_data/137_ihesp_campaign.py --smoke --arms ih_ka --seeds 1234
  .venv/bin/python experiments/real_data/137_ihesp_campaign.py --gpus 6 --per-gpu 3
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("campaign114", ROOT / "experiments/real_data/114_method_selection_campaign.py")
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)

CACHES = ROOT / "outputs/ihesp_20261021/train_caches"
IHESP = ["--context-profiles", "6080", "--queries", "1024", "--amp", "--train-years", "4",
         "--cohort", "ihesp_hr_uniform", "--surface",
         "--satellite-cache", str(ROOT / "outputs/radius_scale_20261008/ihesp_hr_uniform/satellites.npz"),
         "--surface-grids", str(CACHES / "surface_grids.npz"), "--lagged-surface", str(CACHES / "lagged_surface.npz"),
         "--oi-summary", str(ROOT / "outputs/audit/synthetic/fixed_baselines_ihesp_hr_uniform/summary.json"),
         "--leads", "0", "1", "3", "6", "--val-years", "2004", "2005"]
Q.SETUPS["ihesp"] = (IHESP, 70000, ["--val-every", "2000", "--checkpoint-every", "500"])
Q.OUTPUT = ROOT / "outputs/ihesp_20261021"
#: arm -> (dataset, background, backbone, Poseidon-T initialisation (unused by the U-Net), kriging)
Q.ARMS = {"ih_ka": ("ihesp", "joint", "unet", "pretrained", "ka_flow"),
          "ih_armor": ("ihesp", "prior", "unet", "pretrained", "ka_oi"),
          "ih_none": ("ihesp", "joint", "unet", "pretrained", "none"),
          "ih_softmax": ("ihesp", "joint", "unet", "pretrained", "softmax")}
Q.DEFAULT_ARMS = tuple(Q.ARMS)

if __name__ == "__main__":
    Q.main()
