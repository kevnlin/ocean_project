"""A FuXi-Ocean-style state encoder (Huang et al., arXiv:2506.03210v2, section 3.2), re-implemented.

FuXi-Ocean's model code and weights are unreleased; this follows the paper's text. It reads the gridded
model's inputs (one month: rasterised profiles, surface grids, masks, position, season) and a lead, and
returns the state the shared read-out of ``grid_kriging_ocean`` reads:

  context  F_S = E + MLP(tau): a learned embedding per token position and the sinusoidal encoding of
           the input month's calendar month and of the lead (Eq. 1)
  encoder  two convolutions with kernel = stride = 2; the second's output is scaled by (1 + F_S) at
           every token (Eq. 2's W ⊙ F_S on the output channels); LayerNorm after each; a 1x1 fusion
  core     Swin-v2 blocks (scaled cosine attention, log-spaced continuous position bias, residual
           post-norm) in windows shifted every second block; windows wrap in longitude and are masked
           across the poles; both norms of a block are AdaLN, LayerNorm with a scale and shift linear
           in F_S at that token
  decoder  transposed convolutions mirroring the encoder, with skips from the encoder's half-resolution
           features and from the input, to ``c_out`` channels

Left out, for equal footing with KA-Ocean (spec ``docs/superpowers/specs/2026-10-26-fuxi-glorys-comparison-design.md``):
Mixture-of-Time (one input month), autoregression (one pass per lead), the Charbonnier loss and the two
training stages, physical-channel output.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channels of ``[B, C, H, W]``."""

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def sinusoidal(x: Tensor, n: int) -> Tensor:
    """``[..., 2n]``: sin and cos of the angles ``x`` at harmonics 1..n."""
    k = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    return torch.cat((torch.sin(x[..., None] * k), torch.cos(x[..., None] * k)), -1)


def window_partition(x: Tensor, window: int) -> Tensor:
    """``[B, H, W, C]`` -> ``[B * windows, window^2, C]``, windows row-major."""
    b, h, w, c = x.shape
    x = x.view(b, h // window, window, w // window, window, c).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, window * window, c)


def window_merge(x: Tensor, window: int, b: int, h: int, w: int) -> Tensor:
    c = x.shape[-1]
    x = x.view(b, h // window, w // window, window, window, c).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(b, h, w, c)


def polar_mask(h: int, w: int, window: int, shift: int) -> Tensor:
    """``[windows, window^2, window^2]``, True where two tokens of a shifted window must not attend:
    the shift rolled one of them across a pole. Longitude is periodic, so nothing is masked there."""
    region = torch.zeros(1, h, w, 1)
    region[:, h - shift:] = 1.                      # the rows the roll brought from the other pole
    r = window_partition(region, window)[..., 0]
    return r[:, :, None] != r[:, None, :]


class WindowAttention(nn.Module):
    """Swin-v2 window attention: cosine similarity times a learned per-head temperature, plus a
    relative position bias from an MLP of log-spaced offsets."""

    def __init__(self, dim: int, heads: int, window: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.logit_scale = nn.Parameter(torch.full((heads, 1, 1), math.log(10.)))
        self.cpb = nn.Sequential(nn.Linear(2, 512), nn.ReLU(), nn.Linear(512, heads, bias=False))
        r = torch.arange(1 - window, window, dtype=torch.float32) / max(window - 1, 1) * 8.
        table = torch.stack(torch.meshgrid(r, r, indexing="ij"), -1)
        self.register_buffer("table", torch.sign(table) * torch.log2(table.abs() + 1.) / 3., persistent=False)
        pos = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij")).flatten(1)
        rel = (pos[:, :, None] - pos[:, None, :]) + window - 1
        self.register_buffer("index", rel[0] * (2 * window - 1) + rel[1], persistent=False)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        n, t, c = x.shape
        q, k, v = self.qkv(x).view(n, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        attention = nn.functional.normalize(q, dim=-1) @ nn.functional.normalize(k, dim=-1).transpose(-2, -1)
        attention = attention * self.logit_scale.clamp(max=math.log(100.)).exp()
        bias = self.cpb(self.table).flatten(0, 1)[self.index.flatten()].view(t, t, self.heads).permute(2, 0, 1)
        attention = attention + 16. * torch.sigmoid(bias)
        if mask is not None:
            attention = attention.view(-1, mask.shape[0], self.heads, t, t).masked_fill(
                mask[None, :, None], float("-inf")).view(n, self.heads, t, t)
        return self.proj((attention.softmax(-1) @ v).transpose(1, 2).reshape(n, t, c))


class AdaLN(nn.Module):
    """LayerNorm whose scale and shift are linear in the context at each token; starts as LayerNorm."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        scale, shift = self.modulation(context).chunk(2, -1)
        return self.norm(x) * (1 + scale) + shift


class SwinBlock(nn.Module):
    """Swin-v2 residual post-norm with AdaLN: ``x + AdaLN(attention(x))``, then ``x + AdaLN(MLP(x))``."""

    def __init__(self, dim: int, heads: int, window: int, shift: int, tokens: tuple[int, int]):
        super().__init__()
        self.window, self.shift = window, shift
        self.attention = WindowAttention(dim, heads, window)
        self.norm1, self.norm2 = AdaLN(dim), AdaLN(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.register_buffer("mask", polar_mask(*tokens, window, shift) if shift else None, persistent=False)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        b, h, w, _ = x.shape
        s = self.shift
        y = torch.roll(x, (-s, -s), (1, 2)) if s else x
        y = window_merge(self.attention(window_partition(y, self.window), self.mask), self.window, b, h, w)
        y = torch.roll(y, (s, s), (1, 2)) if s else y
        x = x + self.norm1(y, context)
        return x + self.norm2(self.mlp(x), context)


class FuXiEncoder(nn.Module):
    """``x [1, C_in, H, W]``, a lead in months and the input month -> ``[1, C_out, H, W]``.
    H and W must be multiples of 4 and the token grid (H/4, W/4) a multiple of ``window``."""

    def __init__(self, c_in: int, c_out: int, grid_shape: tuple[int, int], width: int = 384, depth: int = 6,
                 heads: int = 12, window: int = 9, max_lead: int = 6):
        super().__init__()
        h, w = grid_shape
        if h % 4 or w % 4 or (h // 4) % window or (w // 4) % window:
            raise ValueError(f"grid {h}x{w}: H and W must be multiples of 4 and H/4, W/4 of the window {window}")
        if width % 2 or width % heads:
            raise ValueError("width must be even and a multiple of heads")
        half, tokens = width // 2, (h // 4, w // 4)
        self.max_lead = max_lead
        self.embed1, self.norm1 = nn.Conv2d(c_in, half, 2, stride=2), LayerNorm2d(half)
        self.embed2, self.norm2 = nn.Conv2d(half, width, 2, stride=2), LayerNorm2d(width)
        self.position = nn.Parameter(torch.zeros(width, *tokens))
        nn.init.trunc_normal_(self.position, std=.02)
        self.time = nn.Sequential(nn.Linear(16, width), nn.SiLU(), nn.Linear(width, width))
        nn.init.zeros_(self.time[-1].weight)
        nn.init.zeros_(self.time[-1].bias)
        self.fusion, self.fusion_norm = nn.Conv2d(width, width, 1), LayerNorm2d(width)
        self.blocks = nn.ModuleList(SwinBlock(width, heads, window, window // 2 if i % 2 else 0, tokens)
                                    for i in range(depth))
        self.up1, self.up1_norm = nn.ConvTranspose2d(width, half, 2, stride=2), LayerNorm2d(half)
        self.mix1 = nn.Conv2d(2 * half, half, 1)
        self.up2, self.up2_norm = nn.ConvTranspose2d(half, half, 2, stride=2), LayerNorm2d(half)
        self.head = nn.Conv2d(half + c_in, c_out, 1)

    def forward(self, x: Tensor, lead: int, month: float) -> Tensor:
        angles = x.new_tensor([2 * math.pi * (float(month) % 12) / 12, math.pi * float(lead) / self.max_lead])
        context = self.position[None] + self.time(sinusoidal(angles, 4).reshape(1, 16))[:, :, None, None]
        e1 = self.norm1(self.embed1(x))
        z = self.norm2(self.embed2(e1) * (1 + context))
        z = self.fusion_norm(self.fusion(z)).permute(0, 2, 3, 1)
        c = context.permute(0, 2, 3, 1)
        for block in self.blocks:
            z = block(z, c)
        d = self.mix1(torch.cat((self.up1_norm(self.up1(z.permute(0, 3, 1, 2))), e1), 1))
        return self.head(torch.cat((self.up2_norm(self.up2(d)), x), 1))
