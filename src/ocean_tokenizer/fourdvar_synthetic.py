"""Pinned official 4DVarNet solver with a scatter-observation ocean adapter.

The author's GradSolver, ConvLstmGradModel and BilinAEPriorCost are executed
unchanged from their source AST. Only the observation operator and task interface
are adapted: channels represent depth/variable, rather than a time window, and
the training targets are heldout profile values rather than dense grid truth.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OFFICIAL_COMMIT = "20f1b5f34b201342cde6dd21a30419d07541db54"
OFFICIAL_MODEL_SHA256 = "0330c134f69e49dd16ddf3da16d0ea203606d54f7a10f6452a9dc2e29eb4e953"
OFFICIAL_CLASSES = ("GradSolver", "ConvLstmGradModel", "BilinAEPriorCost")


def official_components(repository):
    """Load only the three torch-only official classes, verifying the pin."""
    repository = Path(repository)
    revision = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    path = repository / "src/models.py"
    source = path.read_text()
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if revision != OFFICIAL_COMMIT or source_hash != OFFICIAL_MODEL_SHA256:
        raise ValueError("official 4DVarNet source differs from the registered commit/hash")
    parsed = ast.parse(source, filename=str(path))
    nodes = [node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name in OFFICIAL_CLASSES]
    if {node.name for node in nodes} != set(OFFICIAL_CLASSES):
        raise ValueError("official solver source lacks a required class")
    namespace = {"torch": torch, "nn": nn, "F": F, "__name__": __name__}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    metadata = {"repository": "https://github.com/CIA-Oceanix/4dvarnet-starter",
                "commit": revision, "source_path": str(path), "source_sha256": source_hash,
                "class_source_sha256": {node.name: hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
                                        for node in nodes},
                "extraction": "original AST class nodes, unchanged; torch-only imports supplied"}
    return {name: namespace[name] for name in OFFICIAL_CLASSES}, metadata


@dataclass(frozen=True)
class FourDVarConfig:
    levels: int = 20
    height: int = 180
    width: int = 360
    n_step: int = 10
    lr_grad: float = 1000.
    prior_hidden: int = 32
    gradient_hidden: int = 48
    prior_downsampling: int = 2
    gradient_downsampling: int | None = None
    dropout: float = .1
    bilin_quad: bool = False
    surface: bool = False
    auxiliary_weight: float = 1.

    @property
    def channels(self):
        return 2 * self.levels + (3 if self.surface else 0)

    def to_dict(self):
        return asdict(self)


def bilinear_geometry(lat, lon, height=180, width=360):
    """Continuous-position interpolation with periodic longitude and polar clamp.

    Grid centres are -90+(i+.5)*180/H and (j+.5)*360/W. Manual gather supports
    the second-order state gradients needed by the official unrolled solver.
    """
    lat, lon = torch.as_tensor(lat), torch.as_tensor(lon)
    if lat.ndim != 1 or lat.shape != lon.shape or lat.device != lon.device:
        raise ValueError("lat/lon must be matching vectors on the same device")
    if height < 2 or width < 2 or not torch.isfinite(lat).all() or not torch.isfinite(lon).all() or (lat.abs() > 90).any():
        raise ValueError("invalid grid or profile coordinates")
    y = ((lat.to(torch.float64) + 90.) / (180. / height) - .5).clamp(0., height - 1.)
    x = (lon.to(torch.float64).remainder(360.) / (360. / width) - .5).remainder(width)
    y0, x0 = y.floor().long(), x.floor().long()
    y1, x1 = (y0 + 1).clamp_max(height - 1), (x0 + 1).remainder(width)
    dy, dx = y - y0, x - x0
    index = torch.stack((y0 * width + x0, y0 * width + x1,
                         y1 * width + x0, y1 * width + x1), dim=-1)
    weight = torch.stack(((1 - dy) * (1 - dx), (1 - dy) * dx,
                          dy * (1 - dx), dy * dx), dim=-1)
    return index, weight


class ScatterBilinearOperator(nn.Module):
    def __init__(self, lat, lon, height=180, width=360):
        super().__init__()
        index, weight = bilinear_geometry(lat, lon, height, width)
        self.register_buffer("index", index)
        self.register_buffer("weight", weight)
        self.height, self.width = height, width

    def forward(self, state):
        if state.ndim != 4 or state.shape[-2:] != (self.height, self.width):
            raise ValueError("state must be [batch, channel, height, width]")
        gathered = state.flatten(2).index_select(2, self.index.reshape(-1))
        gathered = gathered.reshape(state.shape[0], state.shape[1], len(self.index), 4)
        result = (gathered * self.weight.to(state.dtype)[None, None]).sum(-1)
        return result.transpose(1, 2)


def profiles_to_grid(observed, lat, lon, height=180, width=360):
    """Input-only nearest-cell mean, preserving finite per-variable/depth masks.

    Observed has shape [profile, depth, T/S]. The official solver replaces missing
    initial cells by zero; the actual observation cost always uses original point
    values, independent of this compressed initialization.
    """
    observed, lat, lon = torch.as_tensor(observed), torch.as_tensor(lat), torch.as_tensor(lon)
    if observed.ndim != 3 or observed.shape[-1] != 2 or lat.shape != (len(observed),) or lon.shape != lat.shape:
        raise ValueError("observed must be [profile, depth, 2] with matching lat/lon")
    bilinear_geometry(lat, lon, height, width)  # validate coordinates
    if observed.device != lat.device or lat.device != lon.device:
        raise ValueError("observations and coordinates must share a device")
    row = torch.floor((lat + 90.) / (180. / height)).long().clamp(0, height - 1)
    col = torch.floor(lon.remainder(360.) / (360. / width)).long().remainder(width)
    index = (row * width + col)[None].expand(observed.shape[1] * 2, -1)
    values = observed.transpose(1, 2).reshape(len(observed), -1).T
    valid = torch.isfinite(values)
    total = values.new_zeros((len(values), height * width))
    count = torch.zeros_like(total)
    total.scatter_add_(1, index, torch.where(valid, values, 0.))
    count.scatter_add_(1, index, valid.to(values.dtype))
    mean = torch.where(count > 0, total / count.clamp_min(1), float("nan"))
    return mean.reshape(1, len(values), height, width)


@dataclass
class FourDVarBatch:
    input: torch.Tensor
    observed: torch.Tensor
    operator: ScatterBilinearOperator
    surface: torch.Tensor | None = None


class ScatterObservationCost(nn.Module):
    def __init__(self, channels, auxiliary_weight=1.):
        super().__init__()
        self.channels, self.auxiliary_weight = channels, auxiliary_weight

    def forward(self, state, batch):
        expected = batch.operator(state[:, :self.channels])
        target = batch.observed[None].expand(state.shape[0], -1, -1)
        valid = torch.isfinite(target)
        cost = F.mse_loss(expected[valid], target[valid]) if valid.any() else state.sum() * 0.
        if batch.surface is not None:
            surface = batch.surface.expand(state.shape[0], -1, -1, -1)
            active = torch.isfinite(surface)
            if active.any():
                cost = cost + self.auxiliary_weight * F.mse_loss(state[:, self.channels:][active], surface[active])
        return cost


def make_batch(observed, lat, lon, config, surface=None):
    initial = profiles_to_grid(observed, lat, lon, config.height, config.width)
    if bool(surface is not None) != config.surface:
        raise ValueError("surface state and configuration disagree")
    if surface is not None:
        surface = torch.as_tensor(surface, dtype=initial.dtype, device=initial.device)
        if surface.shape != (1, 3, config.height, config.width):
            raise ValueError("surface observations must be [1,3,height,width]")
        initial = torch.cat((initial, surface), dim=1)
    return FourDVarBatch(initial.detach(), observed.transpose(1, 2).reshape(len(observed), -1).detach(),
                        ScatterBilinearOperator(lat, lon, config.height, config.width), surface)


class OfficialFourDVarNet(nn.Module):
    def __init__(self, repository, config=FourDVarConfig()):
        super().__init__()
        if config.levels < 1 or config.n_step < 1 or config.height % config.prior_downsampling or config.width % config.prior_downsampling:
            raise ValueError("state dimensions must be positive and compatible with prior downsampling")
        if config.gradient_downsampling is not None and (config.height % config.gradient_downsampling or config.width % config.gradient_downsampling):
            raise ValueError("state dimensions incompatible with gradient downsampling")
        classes, metadata = official_components(repository)
        self.config, self.official_source = config, metadata
        prior = classes["BilinAEPriorCost"](dim_in=config.channels, dim_hidden=config.prior_hidden,
                  downsamp=config.prior_downsampling, bilin_quad=config.bilin_quad)
        gradient = classes["ConvLstmGradModel"](dim_in=config.channels, dim_hidden=config.gradient_hidden,
                  dropout=config.dropout, downsamp=config.gradient_downsampling)
        self.solver = classes["GradSolver"](prior_cost=prior,
            obs_cost=ScatterObservationCost(2 * config.levels, config.auxiliary_weight), grad_mod=gradient,
            n_step=config.n_step, lr_grad=config.lr_grad)

    def forward(self, batch):
        return self.solver(batch)

    def query(self, state, lat, lon, levels):
        values = ScatterBilinearOperator(lat, lon, self.config.height, self.config.width)(state)
        levels = torch.as_tensor(levels, dtype=torch.long, device=state.device)
        if levels.shape != (values.shape[1],) or (levels < 0).any() or (levels >= self.config.levels).any():
            raise ValueError("query level identities must match profiles and known depths")
        channels = torch.stack((levels, levels + self.config.levels), -1)
        return values.gather(2, channels[None].expand(state.shape[0], -1, -1))


def heldout_query_loss(prediction, target):
    """Same equal-variable MSE objective as the matched profile models."""
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 2:
        raise ValueError("prediction and target must be [query,2]")
    valid = torch.isfinite(target)
    errors = (prediction - torch.nan_to_num(target)).square()
    terms = [(errors[:, channel] * valid[:, channel]).sum() / valid[:, channel].sum().clamp_min(1)
             for channel in range(2)]
    return torch.stack(terms).mean()
