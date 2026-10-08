"""Prepare exact shared mmap scene arrays once before the full GPU campaign.

No profiles, levels or scoring points are reduced. This CPU process runs the
original loader and publishes its exact outputs under an atomic manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ocean_tokenizer.latent_scene_cache import prepare_scene_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--anchor-tag", default="anc_satday_dfs")
    parser.add_argument("--replace", action="store_true", help="publish a new generation after inputs change")
    args = parser.parse_args()
    started = time.time()
    print("Preparing exact CPU scenes and validating source SHA256 hashes", flush=True)
    manifest = prepare_scene_cache(args.root, args.anchor_tag, replace=args.replace)
    print(json.dumps({"cache_key": manifest["cache_key"], "n_profiles": manifest["n_profiles"],
                      "n_levels": manifest["n_levels"], "elapsed_seconds": time.time() - started,
                      "manifest": str(args.root / "outputs/latent_ocean/prepared_scenes" / args.anchor_tag / "manifest.json")}),
          flush=True)


if __name__ == "__main__":
    main()
