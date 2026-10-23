"""The public Poseidon-T processor (camlab-ethz/Poseidon-T) as a state encoder of the gridded model.

A baseline backbone for the comparison table: the author's multiscale, time-conditioned SwinV2
U-Net (ScOT) with its published PDE-pretrained weights, given the gridded model's inputs and asked
for its state channels. Only the processor's public tensors are reused; the input embedding and the
output recovery are new (the public ones map 4 PDE variables), initialised identically for the
pretrained and the random variant from the same seed.

The processor works on a square 128 x 128 image: the 1-degree input grid is area-averaged to
128 x 128 and the output interpolated back to the input grid, so the rest of the model (read-out,
kriging, scores) is unchanged. The lead conditions the processor through its time input
(lead / 3, the author's operator time), as in the earlier ``poseidon_ocean`` control: one direct
operator per lead, which replaces the gridded model's separate evolution network.

Assets and the isolated transformers 4.35.2 runtime the author's code needs are the ones fixed in
the anchored-fusion worktree (sha256-checked against their acquisition record). Its isolated runtime is
built for Python 3.11; the same package versions for this Python are in ``external/poseidon_deps``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ASSETS = Path("/DATA2/zihao/projects/ocean_project/.claude/worktrees/anchored-fusion")
BACKBONE = ASSETS / "outputs/foundation_v2_20261017/backbones/poseidon_t"
#: transformers 4.35.2, tokenizers 0.15.2, huggingface_hub 0.36.0 (the anchored-fusion runtime's versions, built
#: for this Python) plus regex and safetensors, installed with ``pip --target --no-deps``: the environment is not changed
RUNTIME = Path(__file__).resolve().parents[2] / "external/poseidon_deps"
IMAGE = 128
TIME_SCALE = 3.
RESET_KEYS = {"embeddings.patch_embeddings.projection.weight", "embeddings.patch_embeddings.projection.bias",
              "patch_recovery.projection.weight", "patch_recovery.projection.bias", "patch_recovery.mixup.weight"}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _authenticated():
    receipt = json.loads((BACKBONE / "acquisition.json").read_text())
    for name in ("config.json", "model.py", "model.safetensors"):
        if _sha256(BACKBONE / name) != receipt["files"][name]["sha256"]:
            raise ValueError(f"Poseidon-T asset differs from its acquisition record: {name}")
    return receipt


def _author_module():
    """The author's model.py under the isolated transformers 4.35.2 runtime."""
    if "transformers" not in sys.modules:
        sys.path.insert(0, str(RUNTIME))
    import transformers
    if transformers.__version__ != "4.35.2":
        raise RuntimeError("Poseidon-T needs the isolated transformers 4.35.2 imported before any other version")
    name = "poseidon_t_author"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, BACKBONE / "model.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def public_weights():
    _author_module()
    from safetensors.torch import load_file
    _authenticated()
    return load_file(str(BACKBONE / "model.safetensors"), device="cpu")


class PoseidonEncoder(nn.Module):
    """``x [1, C_in, H, W]`` and a lead in months -> ``[1, C_out, H, W]`` through the processor."""

    def __init__(self, c_in: int, c_out: int, initialization: str = "pretrained", seed: int = 1234):
        super().__init__()
        if initialization not in ("pretrained", "random"):
            raise ValueError("initialization must be pretrained or random")
        _authenticated()
        author = _author_module()
        native = json.loads((BACKBONE / "config.json").read_text())
        native.update(num_channels=int(c_in), num_out_channels=int(c_out),
                      channel_slice_list_normalized_loss=None, learn_residual=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.backbone = author.ScOT(author.ScOTConfig(**native))
            torch.manual_seed(seed + 104729)          # the same new adapters for both initialisations
            embedding = self.backbone.embeddings.patch_embeddings.projection
            nn.init.trunc_normal_(embedding.weight, std=.02)
            nn.init.zeros_(embedding.bias)
            recovery = self.backbone.patch_recovery
            nn.init.trunc_normal_(recovery.projection.weight, std=.02)
            nn.init.zeros_(recovery.projection.bias)
            with torch.no_grad():
                recovery.mixup.weight.zero_()
                index = torch.arange(int(c_out))
                recovery.mixup.weight[index, index, 2, 2] = 1.
        if initialization == "pretrained":
            public, state = public_weights(), self.backbone.state_dict()
            if set(public) != set(state):
                raise ValueError("the public state_dict names differ from the author model's")
            merged = {k: (state[k] if k in RESET_KEYS else public[k]) for k in state}
            self.backbone.load_state_dict(merged, strict=True)
        self.initialization = initialization

    def forward(self, x: torch.Tensor, lead: int) -> torch.Tensor:
        height, width = x.shape[-2:]
        image = F.adaptive_avg_pool2d(x, (IMAGE, IMAGE))
        time = image.new_full((image.shape[0],), float(lead) / TIME_SCALE)
        out = self.backbone(pixel_values=image, time=time, return_dict=True).output
        return F.interpolate(out, size=(height, width), mode="bilinear", align_corners=False)
