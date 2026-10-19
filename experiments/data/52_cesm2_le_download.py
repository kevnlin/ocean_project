"""The CESM2-LE 1-degree TEMP/SALT files the synthetic CESM2 store was cut from.

The Hugging Face dataset ``SingleBicycle/cesm2-le-ocean-sample`` holds one
member of CESM2-LE regridded to 1 degree, 60 levels, 1850-01 ... 2100-12, as
one NetCDF file per variable (TEMP 22 GB, SALT 18 GB) plus ``SHA256SUMS.txt``.
These are the files ``experiments/data/standardize.py --datasets cesm2_le_full``
reads (``CESM2_LE_DIR``); ``data/cesm2_le_full_standard.zarr`` is their
2000-2005 cut.

Both files go to ``data/cesm2_le_raw/``. An interrupted transfer continues from
its ``.part`` file; a file is renamed into place only after its SHA-256 matches
``SHA256SUMS.txt``, and ``manifest.json`` records the source revision.

The dataset is gated: the token of an approved account is read from
``HF_TOKEN`` or ``~/.cache/huggingface/token``.

  .venv/bin/python experiments/data/52_cesm2_le_download.py      # ~40 GB
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[2]
REPO = "SingleBicycle/cesm2-le-ocean-sample"
CHUNK = 8 << 20


def token():
    value = os.environ.get("HF_TOKEN")
    if not value:
        value = (Path.home() / ".cache/huggingface/token").read_text().strip()
    return value


def headers(**extra):
    return {"Authorization": f"Bearer {token()}", **extra}


def revision():
    reply = requests.get(f"https://huggingface.co/api/datasets/{REPO}", timeout=60, headers=headers())
    reply.raise_for_status()
    info = reply.json()
    return info["sha"], [s["rfilename"] for s in info["siblings"]]


def url(rev, name):
    return f"https://huggingface.co/datasets/{REPO}/resolve/{rev}/{name}"


def checksums(rev):
    reply = requests.get(url(rev, "SHA256SUMS.txt"), timeout=60, headers=headers())
    reply.raise_for_status()
    sums = {}
    for line in reply.text.splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[Path(name.strip().lstrip("*")).name] = digest.lower()
    return sums


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def fetch(rev, name, part, attempts=20):
    """Stream ``name`` into ``part``, resuming from its current length."""
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        try:
            with requests.get(url(rev, name), stream=True, timeout=(60, 300),
                              headers=headers(Range=f"bytes={have}-")) as reply:
                if reply.status_code == 416:            # already complete
                    return
                reply.raise_for_status()
                if have and reply.status_code != 206:
                    raise RuntimeError("the server ignored the resume offset")
                total = have + int(reply.headers.get("Content-Length", 0))
                started, done, mark = time.time(), have, have
                with open(part, "ab") as out:
                    for block in reply.iter_content(CHUNK):
                        out.write(block)
                        done += len(block)
                        if done - mark >= 2 << 30:
                            mark = done
                            rate = (done - have) / max(time.time() - started, 1e-6) / 2**20
                            print(f"  {name}: {done / 2**30:.1f} / {total / 2**30:.1f} GiB "
                                  f"({rate:.0f} MiB/s)", flush=True)
            if part.stat().st_size >= total:
                return
        except (requests.RequestException, RuntimeError) as error:
            print(f"  {name}: attempt {attempt} stopped at {part.stat().st_size if part.exists() else 0} "
                  f"bytes ({type(error).__name__}); resuming", flush=True)
            time.sleep(min(60, 5 * attempt))
    raise SystemExit(f"{name}: download did not complete after {attempts} attempts")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=ROOT / "data/cesm2_le_raw")
    p.add_argument("--variables", nargs="+", default=["TEMP", "SALT"])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    rev, names = revision()
    sums = checksums(rev)
    manifest.update(repository=REPO, revision=rev)
    for variable in args.variables:
        name = next(n for n in names if n.endswith(f".{variable}.185001-210012.nc"))
        final, part = args.output / name, args.output / (name + ".part")
        expected = sums[name]
        if final.exists() and manifest["files"].get(name, {}).get("sha256") == expected:
            print(f"{name}: present and verified", flush=True)
            continue
        print(f"{name}: downloading", flush=True)
        fetch(rev, name, part)
        print(f"{name}: checking SHA-256", flush=True)
        digest = sha256(part)
        if digest != expected:
            part.rename(part.with_suffix(".bad"))
            raise SystemExit(f"{name}: SHA-256 {digest} differs from SHA256SUMS.txt {expected}")
        os.replace(part, final)
        manifest["files"][name] = {"sha256": digest, "bytes": final.stat().st_size,
                                   "verified_unix": time.time()}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{name}: verified, {final.stat().st_size / 1e9:.2f} GB", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
