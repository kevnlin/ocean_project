# FuXi comparison and GLORYS12 transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a FuXi-Ocean-style model exactly as KA-Ocean x1.375 was trained (100 CESM2 years, 200k steps, 3 seeds), then score both on CESM2 2005 and, in transfer, on the sealed GLORYS12 2019-2020 cells, and report both comparisons with each model's size in every table.

**Architecture:** A new backbone `fuxi` (`src/ocean_tokenizer/fuxi_backbone.py`) plugs into the gridded model the way Poseidon does (lead-conditioned, evolution off) and is read by the unchanged read-out with `--kriging none`. Queue `172` reuses `150`'s machinery with `169`'s options. Scorer `173` reuses `159`/`109`'s scoring with GLORYS's own statistics and `168`'s extra-pass identity check. Report `174` reuses `170`'s helpers. Spec: `docs/superpowers/specs/2026-10-26-fuxi-glorys-comparison-design.md`.

**Tech Stack:** Python 3.12 (`.venv`), PyTorch 2.13, NumPy, pytest; 8x 96 GB GPUs shared with other users.

---

Facts the tasks rely on (checked 2026-10-26):

* Branch `fuxi-transfer` from `size-study` ded2b90. Suite baseline: `4 failed, 1314 passed, 8 skipped` in 45 s. The 4 are `tests/test_poseidon_backbone.py` (they need the Poseidon runtime this checkout lacks) and are the only failures allowed throughout.
* KA x1.375 runs: `outputs/size_20261023/y100_x1375_s123{4,5,6}/` (a link into `/DATA2/linke/ocean_project_size`), `best_seed*.pt`, `summary_seed*.json`, `forecast_seed*.npz`; 17,730,974 parameters, of which 482,700 are the read-out shared with FuXi-style (decoder 230,208, mean head 124,962, std head 125,666, embeddings 1,824, decay constants 40). Checkpoint config grid `[-89.5, 1.0, 0.5, 1.0, 180, 360]`, `n_query_features` 58. The training record holds absolute cache paths (`satellite_cache`, `surface_grids`, `lagged_surface`, `oi_summary`).
* A prototype of the backbone below (width 384, depth 6, 12 heads, window 9) counted 17,508,596 parameters with the read-out (1.25 % below KA), ran a full-grid CPU forward in 0.25 s, gave every parameter a finite gradient, wrapped windows across longitude and not across the poles.
* `159_sealed_glorys_test.py` is untracked in `/DATA2/linke/ocean_project_ka_dev6/experiments/synthetic/`; it imports `109` and `121`, both present here (`109` identical to `ka-dev6`'s). Importing it only sets paths on `109`'s module: `T.DATA` (cohort, satellites, surface_grids, lagged_surface, oi_summary, damped_persistence_factors), `T.COHORT = "glorys12_test"`, `T.TEST_YEARS = (2006, 2007)`, `T.RESULTS` = `outputs/glorys_20261021/glorys12_test/sealed/results.json` (the main pass: keys `registration_sha256, started_unix, entries, damped_persistence, test_years_glorys`; entries carry `checkpoint, seed, tag, by_lead`).
* GLORYS files (all in `/DATA2/linke/ocean_project_ka_dev6/outputs/`): `glorys_20261021/glorys12_test/{satellites,surface_grids,lagged_surface}.npz` and `sealed/`, `audit/synthetic/fixed_baselines_glorys12_test/`. CESM2 caches: `cohort_cesm2_vlong_uniform/` and `audit/synthetic/fixed_baselines_cesm2_vlong_uniform/` there. `data/` is untracked (ignored), so a worktree can link it; `outputs/` has 815 tracked files and `.gitignore` names output folders one by one.
* Synthetic scenes have a zero first guess: the networks read no OI; the OI only makes the persisted-OI row (`scene["baseline"]`).
* `150_wang_campaign.queue` launches a job only on a place whose GPU has `min_free_mb` free and waits (20 s loop) indefinitely otherwise; a failed run is retried from `last_seed*.pt` after `retry_after`.
* "Run when GPUs are open" (Kevin): every launch and scoring waits for a GPU with at least 80,000 MiB free (`--min-free-mb 80000`, one run a GPU), i.e. a GPU no one else is really using.

`$SCRATCH` is the session's scratchpad directory. Every commit is authored by Kevin (repo git config) and ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. `PY` = `/DATA2/linke/projects/ocean_project/.venv/bin/python`. `SUITE` = `$PY -m pytest -q -p no:cacheprovider 2>&1 | tail -1`, expected `4 failed, N passed, 8 skipped` with only Poseidon failures (`$PY -m pytest -q -p no:cacheprovider 2>&1 | grep ^FAILED | grep -v poseidon_backbone` prints nothing). An open GPU is found with this shell function (define it in the command that uses it): `open_gpu() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 > 80000 {print $1; exit}'; }` (prints an open GPU's index, or nothing).

### Task 1: The FuXi-Ocean-style backbone

**Files:**
- Create: `src/ocean_tokenizer/fuxi_backbone.py`
- Test: `tests/test_fuxi_backbone.py`

- [ ] **Step 1: Failing tests** — create `tests/test_fuxi_backbone.py`:

```python
"""The FuXi-Ocean-style backbone: shapes, lead conditioning, windows that wrap in longitude only."""
import pytest
import torch

from ocean_tokenizer.fuxi_backbone import FuXiEncoder, SwinBlock, polar_mask


def _encoder(seed=0):
    torch.manual_seed(seed)
    return FuXiEncoder(93, 8, (12, 24), width=16, depth=2, heads=2, window=3).double()


def _x():
    return torch.randn(1, 93, 12, 24, dtype=torch.float64, generator=torch.Generator().manual_seed(5))


def test_the_state_has_the_grid_shape_and_is_finite_at_every_lead():
    model, x = _encoder(), _x()
    for lead in (0, 1, 3, 6):
        y = model(x, lead, 13.)
        assert y.shape == (1, 8, 12, 24) and torch.isfinite(y).all()


def test_the_lead_and_season_enter_through_a_time_context_that_starts_at_zero():
    model, x = _encoder(), _x()
    assert torch.equal(model(x, 0, 5.), model(x, 6, 5.))
    torch.nn.init.normal_(model.time[-1].weight, std=.1)
    assert not torch.allclose(model(x, 0, 5.), model(x, 6, 5.))
    assert not torch.allclose(model(x, 0, 5.), model(x, 0, 11.))


def test_a_shifted_window_wraps_in_longitude_and_not_across_the_poles():
    torch.manual_seed(1)
    block = SwinBlock(8, 2, 3, 1, (6, 6)).double()
    z, c = torch.randn(1, 6, 6, 8, dtype=torch.float64), torch.randn(1, 6, 6, 8, dtype=torch.float64)
    base = block(z, c)
    east, north = z.clone(), z.clone()
    east[0, 2, 0] += 1.                       # the first longitude column
    north[0, 0, 2] += 1.                      # the first latitude row
    assert not torch.allclose(block(east, c)[0, 2, 5], base[0, 2, 5])      # reaches the last column
    assert torch.allclose(block(north, c)[0, 5, 2], base[0, 5, 2])         # never the last row
    assert polar_mask(6, 6, 3, 1).flatten(1).any(1).tolist() == [False, False, True, True]


def test_every_parameter_learns():
    model = _encoder()
    model(_x(), 3, 5.).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_a_grid_the_windows_do_not_tile_is_refused():
    with pytest.raises(ValueError, match="multiples of 4"):
        FuXiEncoder(93, 8, (12, 24), width=16, depth=2, heads=2, window=9)
```

- [ ] **Step 2: Run** `$PY -m pytest -q -p no:cacheprovider tests/test_fuxi_backbone.py` — Expected: collection error, `No module named 'ocean_tokenizer.fuxi_backbone'`.

- [ ] **Step 3: Implement** — create `src/ocean_tokenizer/fuxi_backbone.py`:

```python
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
```

- [ ] **Step 4: Run** `$PY -m pytest -q -p no:cacheprovider tests/test_fuxi_backbone.py` — Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/ocean_tokenizer/fuxi_backbone.py tests/test_fuxi_backbone.py docs/superpowers/specs/2026-10-26-fuxi-glorys-comparison-design.md
git commit -q -F - <<'EOF'
fuxi_backbone: a FuXi-Ocean-style state encoder (context-modulated patch encoder, Swin-v2 with AdaLN, windows wrapping in longitude only, transposed-conv decoder with skips)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

### Task 2: Backbone `fuxi` in the gridded model

**Files:**
- Modify: `src/ocean_tokenizer/grid_kriging_ocean.py` (config fields and checks, `__init__`, `encode`)
- Test: `tests/test_grid_kriging_ocean.py` (append)

- [ ] **Step 1: Failing tests** — append to `tests/test_grid_kriging_ocean.py`:

```python
FUXI = dict(backbone="fuxi", evolution=False, kriging="none", fuxi_width=16, fuxi_depth=2, fuxi_heads=2, fuxi_window=3)


@pytest.mark.parametrize("lead", [0, 3])
def test_a_fuxi_model_decodes_finite_values_and_its_encoder_learns(lead):
    model = _model(**FUXI).train()
    with torch.no_grad():
        model.mean_head[-1].weight.normal_(0., .1)          # zero at initialisation: a first step's head
    out = model(_scene(lead))
    assert out["mean"].shape == (7, 2) and torch.isfinite(out["mean"]).all() and torch.isfinite(out["std"]).all()
    ((out["mean"] - .3).square().mean() + .01 * out["std"].mean()).backward()
    grads = [p.grad for p in model.encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert model.encoder.head.weight.grad.abs().sum() > 0


def test_the_fuxi_backbone_takes_the_lead_itself():
    with pytest.raises(ValueError, match="lead-conditioned"):
        _config(backbone="fuxi", kriging="none")                     # the evolution defaults to on
    with pytest.raises(ValueError, match="lead-conditioned"):
        _config(**{**FUXI, "background": "prior"})


def test_a_fuxi_config_replays_the_model():
    model = _model(**FUXI)
    again = GridKrigingModel(GridKrigingConfig(**model.config.to_dict())).double().eval()
    again.load_state_dict(model.state_dict(), strict=True)
    scene = _scene(1)
    assert torch.equal(model(scene)["mean"], again(scene)["mean"])


def test_at_the_study_size_the_fuxi_model_is_within_two_percent_of_ka_x1375():
    config = GridKrigingConfig(state_channels=176, hidden=352, backbone="fuxi", evolution=False, kriging="none")
    params = sum(p.numel() for p in GridKrigingModel(config).parameters())
    assert abs(params / 17_730_974 - 1) < .02                          # KA x1.375's runs
```

- [ ] **Step 2: Run** `$PY -m pytest -q -p no:cacheprovider tests/test_grid_kriging_ocean.py -k fuxi` — Expected: 5 failures (`unexpected keyword argument 'fuxi_width'` or `backbone must be unet or poseidon`).

- [ ] **Step 3: Implement** in `src/ocean_tokenizer/grid_kriging_ocean.py`.

Config fields — replace

```python
    #: the state encoder: our U-Net, or the public Poseidon-T processor (a baseline backbone, lead-conditioned)
    backbone: str = "unet"
    poseidon_init: str = "pretrained"
```

with

```python
    #: the state encoder: our U-Net, the public Poseidon-T processor or a FuXi-Ocean-style encoder
    #: (``fuxi_backbone``); the last two are baseline backbones, lead-conditioned themselves
    backbone: str = "unet"
    poseidon_init: str = "pretrained"
    #: the FuXi-Ocean-style encoder's sizes: 17.5M parameters with the x1.375 read-out (state 176, hidden 352)
    fuxi_width: int = 384
    fuxi_depth: int = 6
    fuxi_heads: int = 12
    fuxi_window: int = 9
```

Checks — replace

```python
        if self.backbone not in ("unet", "poseidon") or self.poseidon_init not in ("pretrained", "random"):
            raise ValueError("backbone must be unet or poseidon; poseidon_init pretrained or random")
        if self.backbone == "poseidon" and (self.evolution or self.background != "joint"):
            raise ValueError("the Poseidon backbone is lead-conditioned itself: evolution off, background joint")
```

with

```python
        if self.backbone not in ("unet", "poseidon", "fuxi") or self.poseidon_init not in ("pretrained", "random"):
            raise ValueError("backbone must be unet, poseidon or fuxi; poseidon_init pretrained or random")
        if self.backbone in ("poseidon", "fuxi") and (self.evolution or self.background != "joint"):
            raise ValueError(f"the {self.backbone} backbone is lead-conditioned itself: evolution off, background joint")
```

`__init__` — replace

```python
        elif c.backbone == "poseidon":
            from .poseidon_backbone import PoseidonEncoder
            self.encoder = PoseidonEncoder(c.input_channels, c.state_channels, c.poseidon_init)
```

with

```python
        elif c.backbone == "poseidon":
            from .poseidon_backbone import PoseidonEncoder
            self.encoder = PoseidonEncoder(c.input_channels, c.state_channels, c.poseidon_init)
        elif c.backbone == "fuxi":
            from .fuxi_backbone import FuXiEncoder
            self.encoder = FuXiEncoder(c.input_channels, c.state_channels, (grid.n_lat, grid.n_lon), c.fuxi_width,
                                       c.fuxi_depth, c.fuxi_heads, c.fuxi_window, c.max_lead)
```

`encode` — replace

```python
        elif self.config.backbone == "poseidon":
            # one direct operator per lead; the neighbours' background is the lead-0 state
            x = torch.cat((profiles[None], context), 1)
            grid_lead = self.encoder(x, lead)
            grid0 = grid_lead if not lead else (self.encoder(x, 0) if self.kriging is not None else grid_lead)
```

with

```python
        elif self.config.backbone in ("poseidon", "fuxi"):
            # one direct operator per lead; the neighbours' background is the lead-0 state
            x = torch.cat((profiles[None], context), 1)
            month = {"month": float(scene["input_month"])} if self.config.backbone == "fuxi" else {}
            grid_lead = self.encoder(x, lead, **month)
            grid0 = grid_lead if not lead else (self.encoder(x, 0, **month) if self.kriging is not None else grid_lead)
```

- [ ] **Step 4: Run** `$PY -m pytest -q -p no:cacheprovider tests/test_grid_kriging_ocean.py tests/test_fuxi_backbone.py` — Expected: all pass. Then `SUITE` — Expected: `4 failed, 1324 passed, 8 skipped`, Poseidon failures only.

- [ ] **Step 5: Commit**

```bash
git add src/ocean_tokenizer/grid_kriging_ocean.py tests/test_grid_kriging_ocean.py
git commit -q -F - <<'EOF'
grid_kriging_ocean: backbone "fuxi" (lead-conditioned like Poseidon; its sizes in the config, so earlier checkpoints load unchanged)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

### Task 3: `--backbone fuxi` in runner 101

**Files:**
- Modify: `experiments/synthetic/101_grid_ka_forecast.py` (docstring, `arguments()`, the source hashes in `main()`)
- Test: `tests/test_grid_runner_resume.py` (append)

- [ ] **Step 1: Failing test** — append to `tests/test_grid_runner_resume.py`:

```python
def test_the_runner_takes_the_fuxi_backbone_and_hashes_its_source(monkeypatch):
    monkeypatch.setattr("sys.argv", ["101", "--tag", "t", "--backbone", "fuxi", "--kriging", "none"])
    args = runner.arguments()
    assert args.backbone == "fuxi" and args.kriging == "none"
    assert 'ROOT/"src/ocean_tokenizer/fuxi_backbone.py"' in Path(runner.__file__).read_text()
```

- [ ] **Step 2: Run** `$PY -m pytest -q -p no:cacheprovider tests/test_grid_runner_resume.py -k fuxi` — Expected: FAIL, `invalid choice: 'fuxi'` (SystemExit 2).

- [ ] **Step 3: Implement.** In the docstring, after the two `--backbone poseidon` lines, add:

```
  --backbone fuxi     a FuXi-Ocean-style encoder (``fuxi_backbone``, re-implemented from the paper; sizes are
                      the config's defaults) as the state; lead-conditioned, so the evolution is off; with
                      ``--kriging none`` it is the FuXi comparison's arm (``172``)
```

In `arguments()` replace `p.add_argument("--backbone", choices=["unet", "poseidon"], default="unet")` with `p.add_argument("--backbone", choices=["unet", "poseidon", "fuxi"], default="unet")`.

In `main()`'s `source_hashes` tuple replace `ROOT/"src/ocean_tokenizer/poseidon_backbone.py")}` with `ROOT/"src/ocean_tokenizer/poseidon_backbone.py", ROOT/"src/ocean_tokenizer/fuxi_backbone.py")}`.

- [ ] **Step 4: Run** `$PY -m pytest -q -p no:cacheprovider tests/test_grid_runner_resume.py` — Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/synthetic/101_grid_ka_forecast.py tests/test_grid_runner_resume.py
git commit -q -F - <<'EOF'
101_grid_ka_forecast: --backbone fuxi; fuxi_backbone.py in the source hashes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

### Task 4: The queue `172_fuxi_campaign.py`

**Files:**
- Create: `experiments/synthetic/172_fuxi_campaign.py`
- Test: `tests/test_fuxi_transfer.py`

- [ ] **Step 1: Failing tests** — create `tests/test_fuxi_transfer.py`:

```python
"""The FuXi comparison: the queue's runs, the transfer scorer's refusals, the report's verdicts and labels."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/synthetic" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def options(command):
    """{flag: its values} of a runner command (``python -u runner ...``)."""
    out, flag = {}, None
    for word in command[3:]:
        if word.startswith("--"):
            flag = word
            out[flag] = []
        else:
            out[flag].append(word)
    return out


campaign = script("172_fuxi_campaign")
size = script("169_size_campaign")
MODEL = {"--backbone", "--kriging", "--encoder-channels", "--evolution-channels", "--keep-every", "--tag", "--output"}


def test_the_fuxi_runs_train_exactly_as_ka_x1375_but_for_the_model():
    fuxi, ka = options(campaign.job(1234)["command"]), options(size.job("y100_x1375", 1234)["command"])
    assert {k: v for k, v in fuxi.items() if k not in MODEL} == {k: v for k, v in ka.items() if k not in MODEL}
    assert fuxi["--backbone"] == ["fuxi"] and fuxi["--kriging"] == ["none"] and ka["--kriging"] == ["ka_flow"]
    assert fuxi["--state-channels"] == ["176"] and fuxi["--hidden"] == ["352"] and "--keep-every" not in fuxi
    assert fuxi["--steps"] == ["200000"] and fuxi["--train-years"] == ["100"] and fuxi["--test-years"] == ["2005"]


def test_a_smoke_run_is_short_and_kept_apart():
    item = campaign.job(1234, smoke=True)
    assert item["steps"] == 600 and Path(item["output"]).parent.name == "smoke"
    assert options(item["command"])["--val-every"] == ["300"] and item["tag"] == "y100_fuxi_s1234"
```

- [ ] **Step 2: Run** `$PY -m pytest -q -p no:cacheprovider tests/test_fuxi_transfer.py` — Expected: collection error (`No such file ... 172_fuxi_campaign.py`).

- [ ] **Step 3: Implement** — create `experiments/synthetic/172_fuxi_campaign.py`:

```python
"""Queue of the FuXi comparison (spec ``docs/superpowers/specs/2026-10-26-fuxi-glorys-comparison-design.md``):
a FuXi-Ocean-style model (``101 --backbone fuxi --kriging none``) trained as the size study trained KA-Ocean
x1.375 (``169``): ``cesm2_vlong_uniform`` with a 100-year window (1904-2003), 200,000 steps, leads 0/1/3/6,
the checkpoint chosen on 2004 and 2005 held out, seeds 1234-1236. The state (176 channels) and the read-out
(hidden 352) are x1.375's; the backbone's sizes are the config's defaults (17.5M parameters in all). No
snapshots are kept. The queue (a GPU memory check before each launch, retries from the last resume point,
finished runs skipped) is ``150``'s; with ``--min-free-mb 80000`` a run starts only on an open GPU.

  .venv/bin/python experiments/synthetic/172_fuxi_campaign.py --smoke --seeds 1234 --min-free-mb 80000
  nohup setsid .venv/bin/python experiments/synthetic/172_fuxi_campaign.py --min-free-mb 80000 \
      > outputs/fuxi_20261026/queue.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/fuxi_20261026"
RUNNER = ROOT / "experiments/synthetic/101_grid_ka_forecast.py"
STEPS, SMOKE_STEPS = 200000, 600
SEEDS = (1234, 1235, 1236)
ARM = "y100_fuxi"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/synthetic" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_queue = _script("150_wang_campaign")
VLONG, VLONG_NEEDS = _queue.cohort_options("cesm2_vlong_uniform", 100)
#: 169's options with the FuXi-style backbone in place of ka_flow, at x1.375's state and read-out
COMMON = ["--surface", "--queries", "1024", "--amp", "--leads", "0", "1", "3", "6",
          "--val-years", "2004", "--test-years", "2005", "--backbone", "fuxi", "--kriging", "none",
          "--state-channels", "176", "--hidden", "352"]


def job(seed, smoke=False):
    steps, every = (SMOKE_STEPS, 300) if smoke else (STEPS, 2000)
    folder = (OUTPUT / "smoke" if smoke else OUTPUT) / f"{ARM}_s{seed}"
    command = [sys.executable, "-u", str(RUNNER), *COMMON, *VLONG,
               "--steps", str(steps), "--val-every", str(every), "--checkpoint-every", str(min(every, 500)),
               "--seed", str(seed), "--tag", f"{ARM}_s{seed}", "--output", str(folder)]
    return {"tag": f"{ARM}_s{seed}", "arm": ARM, "seed": seed, "steps": steps, "output": str(folder),
            "needs": [str(p) for p in VLONG_NEEDS], "command": command}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="candidates; a launch waits for free memory")
    parser.add_argument("--per-gpu", type=int, default=1, help="workers sharing one GPU")
    parser.add_argument("--min-free-mb", type=int, default=80000, help="free GPU memory a launch needs")
    parser.add_argument("--retry-after", type=float, default=600.)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--stagger", type=float, default=120., help="seconds between two launches")
    args = parser.parse_args()
    name = "smoke" if args.smoke else "queue"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / f"{name}.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another queue is running")
    requested = [job(seed, args.smoke) for seed in args.seeds]
    places = [(int(gpu), worker) for gpu in args.gpus.split(",") for worker in range(args.per_gpu)]
    pending = [j for j in requested if not _queue.completed(j)]
    started = time.time()

    def status(extra):
        _queue.write_json(OUTPUT / f"status_{name}.json", {"requested": len(requested),
            "completed": sum(_queue.completed(j) for j in requested), "started_unix": started,
            "updated_unix": time.time(), **extra})

    if not args.smoke:
        _queue.write_json(OUTPUT / "campaign.json", {"seeds": args.seeds, "steps": STEPS, "options": COMMON + VLONG,
            "runner": str(RUNNER.relative_to(ROOT)), "jobs": [j["tag"] for j in requested]})
    print(json.dumps({"requested": len(requested), "to_run": len(pending),
                      "places": [f"{gpu}.{worker}" for gpu, worker in places]}), flush=True)
    try:
        _queue.queue(pending, places, args.stagger, status, min_free_mb=args.min_free_mb,
                     retry_after=args.retry_after, attempts_allowed=args.attempts)
        status({"running": {}, "pending": [], "finished": True, "runtime_s": time.time() - started})
    except Exception as error:
        status({"failure": repr(error)})
        raise


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run** `$PY -m pytest -q -p no:cacheprovider tests/test_fuxi_transfer.py` — Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add experiments/synthetic/172_fuxi_campaign.py tests/test_fuxi_transfer.py
git commit -q -F - <<'EOF'
172_fuxi_campaign: FuXi-Ocean-style on 100 CESM2 years with KA x1.375's options, state and read-out; launches only on an open GPU

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

### Task 5: The run worktree

The queue runs from a detached worktree so later commits do not change the code a running job reads.

**Files:** `.gitignore` (one line); nothing else tracked.

- [ ] **Step 1: Worktree, data, caches**

```bash
W=/DATA2/linke/ocean_project_fuxi
SRC=/DATA2/linke/ocean_project_ka_dev6/outputs
git worktree add --detach $W HEAD
ln -s /DATA2/linke/projects/ocean_project/data $W/data
mkdir -p $W/outputs/audit/synthetic $W/outputs/fuxi_20261026
cp -al $SRC/cohort_cesm2_vlong_uniform $W/outputs/cohort_cesm2_vlong_uniform
cp -al $SRC/audit/synthetic/fixed_baselines_cesm2_vlong_uniform $W/outputs/audit/synthetic/fixed_baselines_cesm2_vlong_uniform
ln -s $W/outputs/fuxi_20261026 outputs/fuxi_20261026
```

- [ ] **Step 2: Verify** `for f in satellites lagged_surface surface_grids; do stat -c '%h %i %n' $W/outputs/cohort_cesm2_vlong_uniform/$f.npz; done; git -C $W log -1 --oneline; df -h /DATA2 | tail -1` — Expected: link counts 3 or more (hard links, no copy), the worktree at Task 4's commit, free space unchanged (about 544 GB).

- [ ] **Step 3: Ignore the link** — append `/outputs/fuxi_20261026` to `.gitignore` (a symlink, so no trailing slash), then:

```bash
git add .gitignore
git commit -q -F - <<'EOF'
gitignore: the FuXi runs' outputs link

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

### Task 6: Smoke run (memory, speed, lr stability)

- [ ] **Step 1: Start it in the background** (Bash `run_in_background`; it waits for an open GPU, then trains 600 steps):

```bash
cd /DATA2/linke/ocean_project_fuxi && $PY experiments/synthetic/172_fuxi_campaign.py --smoke --seeds 1234 \
  --min-free-mb 80000 --stagger 10 > outputs/fuxi_20261026/smoke.log 2>&1
```

While it waits, carry on with Tasks 8, 9 and 11 (they need no GPU). When notified that it exited:

- [ ] **Step 2: Read the numbers**

```bash
cd /DATA2/linke/ocean_project_fuxi && tail -3 outputs/fuxi_20261026/smoke.log && $PY - <<'EOF'
import json
from pathlib import Path
fuxi = json.loads(Path("outputs/fuxi_20261026/smoke/y100_fuxi_s1234/summary_seed1234.json").read_text())
ka = json.loads(Path("/DATA2/linke/ocean_project_size/outputs/size_20261023/smoke/y100_x1375_s1234/summary_seed1234.json").read_text())
block = lambda s: s["history"][1]["elapsed_s"] - s["history"][0]["elapsed_s"]     # 300 steps and one validation
print(f"params {fuxi['params']:,}  peak {fuxi['peak_gpu_memory_mb'] / 1024:.1f} GB  "
      f"est. {6.6 * block(fuxi) / block(ka):.1f} h a run (KA x1.375: 6.6 h)")
print("2004 score", round(fuxi["initial"]["scores"]["macro_z"], 4), "->",
      [round(r["validation_mean_standardized_rmse"], 4) for r in fuxi["history"]])
EOF
```

Expected: `params 17,508,596`; a finite peak; the 2004 score at 600 steps below the initial one (training works at lr 3e-4); no `Traceback` in `outputs/fuxi_20261026/smoke/y100_fuxi_s1234/train.log`.

- [ ] **Step 3: Only if the log shows `nonfinite loss` or the score rises:** the spec's fallback, the paper's lr. In `172_fuxi_campaign.py` change the end of `COMMON` to `"--state-channels", "176", "--hidden", "352", "--lr", "2.5e-4"]`; in `tests/test_fuxi_transfer.py` add `"--lr"` to `MODEL` and `assert fuxi["--lr"] == ["2.5e-4"]` to the first test; run the tests; commit (`172_fuxi_campaign: lr 2.5e-4 (the paper's): 3e-4 diverged in the smoke`); move the worktree: `git -C /DATA2/linke/ocean_project_fuxi checkout --detach <that commit>`; delete `outputs/fuxi_20261026/smoke` and repeat Steps 1-2.

### Task 7: Launch the queue

- [ ] **Step 1: Start it detached, so it outlives this session**

```bash
cd /DATA2/linke/ocean_project_fuxi && nohup setsid $PY experiments/synthetic/172_fuxi_campaign.py \
  --min-free-mb 80000 > outputs/fuxi_20261026/queue.log 2>&1 < /dev/null &
```

- [ ] **Step 2: After a minute:** `tail -3 outputs/fuxi_20261026/queue.log; head -c 600 outputs/fuxi_20261026/status_queue.json` — Expected: `"requested": 3`; either `"started"` lines or all three in `pending` (no open GPU yet; the queue keeps waiting). No `failure` key.
- [ ] **Step 3:** Report to Kevin: launched or waiting for open GPUs, the per-run estimate from Task 6.

### Task 8: 159 into git, the GLORYS files linked

**Files:**
- Create: `experiments/synthetic/159_sealed_glorys_test.py` (copied unchanged)
- Modify: `.gitignore`

- [ ] **Step 1: Copy and link**

```bash
SRC=/DATA2/linke/ocean_project_ka_dev6
cp $SRC/experiments/synthetic/159_sealed_glorys_test.py experiments/synthetic/
cmp $SRC/experiments/synthetic/159_sealed_glorys_test.py experiments/synthetic/159_sealed_glorys_test.py && echo same
mkdir -p outputs/glorys_20261021 outputs/audit/synthetic
cp -al $SRC/outputs/glorys_20261021/glorys12_test outputs/glorys_20261021/glorys12_test
cp -al $SRC/outputs/audit/synthetic/fixed_baselines_glorys12_test outputs/audit/synthetic/fixed_baselines_glorys12_test
```

- [ ] **Step 2: Ignore them** — append to `.gitignore`:

```
/outputs/glorys_20261021/
/outputs/audit/synthetic/fixed_baselines_glorys12_test/
/outputs/transfer_20261026/
```

Run `git status --short` — Expected: only `.gitignore` modified and `experiments/synthetic/159_sealed_glorys_test.py` new.

- [ ] **Step 3: Commit**

```bash
git add .gitignore experiments/synthetic/159_sealed_glorys_test.py
git commit -q -F - <<'EOF'
159_sealed_glorys_test: into git unchanged from ka-dev6 (it produced the 2026-10-22 sealed GLORYS results); the GLORYS outputs ignored

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

### Task 9: The transfer scorer `173_glorys_transfer.py`

**Files:**
- Create: `experiments/synthetic/173_glorys_transfer.py`
- Test: `tests/test_fuxi_transfer.py` (append)

- [ ] **Step 1: Failing tests** — append to `tests/test_fuxi_transfer.py`:

```python
transfer = script("173_glorys_transfer")
LEVELS = [5, 15, 25, 35, 45, 55, 65, 85, 105, 125, 145, 165.1, 186.3, 222.6, 267.7, 326.9, 408.8, 527.7, 707.6, 984.7]


def test_the_transfer_scenes_are_159s():
    assert transfer.COHORT == "glorys12_test" and tuple(transfer.TEST_YEARS) == (2006, 2007)
    assert transfer.GLORYS_TRAIN_YEARS == 20 and "damped_persistence_factors" not in transfer.GLORYS
    assert transfer.MAIN.parent.name == "sealed" and transfer.MAIN.name == "results.json"


def test_a_checkpoint_on_another_grid_or_feature_set_is_refused():
    data = SimpleNamespace(grid_lat=np.arange(-89.5, 90.), grid_lon=np.arange(.5, 360.), x=torch.zeros(1, 56))
    checkpoint = {"config": {"grid": [-89.5, 1., .5, 1., 180, 360], "n_query_features": 58}, "training": {"tag": "t"}}
    transfer.compatible(checkpoint, data)
    with pytest.raises(SystemExit, match="query features"):
        transfer.compatible({**checkpoint, "config": {**checkpoint["config"], "grid": [-89., 2., 1., 2., 90, 180]}}, data)
    with pytest.raises(SystemExit, match="query features"):
        transfer.compatible({**checkpoint, "config": {**checkpoint["config"], "n_query_features": 60}}, data)


def test_the_score_is_101s_mean_over_variables_of_the_z_rmse():
    data = SimpleNamespace(levels=np.asarray(LEVELS), norm=SimpleNamespace(std={"TEMP": np.ones(20), "SALT": np.ones(20)}))
    target = np.array([[1., 2.], [1., 2.], [1., 2.], [1., np.nan]])
    collected = {0: {"mean": np.zeros((4, 2)), "target": target, "level": np.array([0, 1, 2, 3])}}
    assert transfer.macro_z(collected, data) == {"0": pytest.approx(1.5)}


def entry(tag, oi=.5, params=0):
    by_lead = {lead: {"all": {ch: {"model": {"rmse": .3}, "persisted_oi": {"rmse": oi}, "climatology": {"rmse": .8}}
                              for ch in ("TEMP", "SALT")}} for lead in ("0", "1", "3", "6")}
    return {"tag": tag, "params": params, "by_lead": by_lead, "macro_z": {lead: .5 for lead in ("0", "1", "3", "6")}}


def test_every_entry_must_have_the_main_passs_baselines():
    main = {"entries": [entry("g_ka_s1234")]}
    transfer.same_baselines([entry("y100_fuxi_s1234")], main)
    with pytest.raises(SystemExit, match="differs from the main pass"):
        transfer.same_baselines([entry("y100_fuxi_s1234", oi=.5001)], main)


def test_registration_needs_the_main_pass_and_cesm2_trained_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(transfer, "REGISTRATION", tmp_path / "registration.json")
    monkeypatch.setattr(transfer, "MAIN", tmp_path / "missing.json")
    with pytest.raises(SystemExit, match="main pass"):
        transfer.register([])
    main = tmp_path / "results.json"
    main.write_text("{}")
    monkeypatch.setattr(transfer, "MAIN", main)
    checkpoint = tmp_path / "best_seed1234.pt"
    torch.save({"training": {"cohort": "glorys12_uniform"}}, checkpoint)
    with pytest.raises(SystemExit, match="CESM2-trained"):
        transfer.register([str(checkpoint)])
    assert not (tmp_path / "registration.json").exists()


def test_the_transfer_is_scored_once(tmp_path, monkeypatch):
    done = tmp_path / "results.json"
    done.write_text("{}")
    monkeypatch.setattr(transfer, "RESULTS", done)
    with pytest.raises(SystemExit, match="scored once"):
        transfer.score("cpu")
```

- [ ] **Step 2: Run** `$PY -m pytest -q -p no:cacheprovider tests/test_fuxi_transfer.py` — Expected: collection error (`No such file ... 173_glorys_transfer.py`).

- [ ] **Step 3: Implement** — create `experiments/synthetic/173_glorys_transfer.py`:

```python
"""The GLORYS12 transfer test: CESM2-trained checkpoints scored once on the sealed GLORYS 2019-2020 cells.

Spec ``docs/superpowers/specs/2026-10-26-fuxi-glorys-comparison-design.md``. A declared extra pass on the
test ``159`` scored on 2026-10-22 (cohort ``glorys12_test``, cohort years 2006-2007 = GLORYS 2019-2020, every
held-out cell, leads 0, 1, 3, 6). The checkpoints were trained and chosen on CESM2 only. Everything
statistical is GLORYS's own (the climatology, the per-depth standardisation, the surface z-scores, the OI of
the persisted-OI row), built as ``159`` built them for ``g_ka`` (``train_years`` 20), so the persisted OI
and the climatology of every entry must equal ``159``'s main pass, or nothing is written. Each step refuses
to run twice:

  --check PATH          scores a checkpoint on its own CESM2 2004-2005 (the caches its run recorded); the
                        run's scores must come back (``109``'s check): this code still makes the model it trained
  --register PATH ...   the checkpoints, the GLORYS data files and the main pass's results, by sha256
  --score               verifies every file, scores every checkpoint, writes results.json once

  .venv/bin/python experiments/synthetic/173_glorys_transfer.py --check PATH
  .venv/bin/python experiments/synthetic/173_glorys_transfer.py --register PATH ...
  .venv/bin/python experiments/synthetic/173_glorys_transfer.py --score
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from ocean_tokenizer.point_baselines import CH, Scores      # noqa: E402


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


G = _module("glorys159", "experiments/synthetic/159_sealed_glorys_test.py")
T, SS = G.T, G.SS
SPEC = "docs/superpowers/specs/2026-10-26-fuxi-glorys-comparison-design.md"
COHORT, TEST_YEARS, LEADS = T.COHORT, T.TEST_YEARS, T.LEADS
GLORYS = {name: path for name, path in T.DATA.items() if name != "damped_persistence_factors"}
MAIN = T.RESULTS                                  # 159's main pass: read, never written
GLORYS_TRAIN_YEARS = 20                           # g_ka's --train-years: the scenes 159 built
OUTPUT = ROOT / "outputs/transfer_20261026"
REGISTRATION, RESULTS = OUTPUT / "registration.json", OUTPUT / "results.json"


def own_data(training):
    """``109``'s DATA for the cohort a checkpoint trained on: the caches its run recorded."""
    return {"cohort": ROOT / f"data/synthetic_argo/{training['cohort']}.nc",
            "satellites": Path(training["satellite_cache"]), "surface_grids": Path(training["surface_grids"]),
            "lagged_surface": Path(training["lagged_surface"]), "oi_summary": Path(training["oi_summary"])}


def check(path, device):
    """``109``'s check on the checkpoint's own cohort: its run's 2004-2005 scores must come back."""
    import torch
    training = torch.load(path, map_location="cpu", weights_only=True)["training"]
    T.DATA, T.COHORT = own_data(training), training["cohort"]
    T.check("grid", path, device)


def glorys_scenes(device, cache):
    """The GLORYS test cohort's scenes as ``159`` built them for ``g_ka`` (built once)."""
    if "glorys" not in cache:
        T.DATA, T.COHORT = GLORYS, COHORT
        cache["glorys"] = T.scenes("grid", {"train_years": GLORYS_TRAIN_YEARS, "scale": None}, device)
    return cache["glorys"]


def compatible(checkpoint, data):
    """The checkpoint's grid and query feature count must be the scenes'."""
    grid = (float(data.grid_lat[0]), float(data.grid_lat[1] - data.grid_lat[0]), float(data.grid_lon[0]),
            float(data.grid_lon[1] - data.grid_lon[0]), len(data.grid_lat), len(data.grid_lon))
    config, features = checkpoint["config"], data.x.shape[1] + 2        # + the two first-guess columns
    if not np.allclose(grid, config["grid"]) or config["n_query_features"] != features:
        raise SystemExit(f"{checkpoint['training']['tag']}: grid {config['grid']} and {config['n_query_features']} "
                         f"query features; the GLORYS scenes have {grid} and {features} query features")


def macro_z(collected, data):
    """{lead: the mean over T and S of the RMSE in the scored cohort's per-depth z units} (``101``'s score)."""
    out = {}
    for lead, a in sorted(collected.items()):
        scores = Scores(data.levels, data.norm.std)
        for j, ch in enumerate(CH):
            scores.add(ch, a["mean"][:, j], a["target"][:, j], a["level"])
        out[str(lead)] = scores.result()["macro_z"]
    return out


def evaluate(path, device, cache):
    """One CESM2-trained checkpoint on every cell of the GLORYS test years, with GLORYS's own statistics."""
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    training = checkpoint["training"]
    data = glorys_scenes(device, cache)
    compatible(checkpoint, data)
    model = T.model_of("grid", checkpoint, data.device)
    collected = SS.collect(model, data, data.analysis(), data.forecast_months(list(TEST_YEARS), 0),
                           training.get("eval_context_profiles") or training["context_profiles"], training["amp"])
    return checkpoint, SS.summarize(collected, data.norm, data.levels, TEST_YEARS), macro_z(collected, data)


def register(paths):
    import torch
    if not MAIN.exists():
        raise SystemExit(f"{MAIN} missing: the extra pass needs 159's main pass")
    for path in paths:
        cohort = torch.load(path, map_location="cpu", weights_only=True)["training"]["cohort"]
        if cohort.startswith("glorys"):
            raise SystemExit(f"{path}: trained on {cohort}; the transfer test scores CESM2-trained models only")
    record = SS.register(REGISTRATION, [("grid", Path(p)) for p in paths], {**GLORYS, "main_results": MAIN},
                         {"cohort": COHORT, "test_years": list(TEST_YEARS), "test_years_glorys": [2019, 2020],
                          "leads": list(LEADS), "cells": "every held-out cell", "spec": SPEC,
                          "declared_extra_pass": "the test 159 scored on 2026-10-22; nothing is selected on it",
                          "statistics": "the scored cohort's own (GLORYS); the checkpoints' CESM2 normalisation is recorded, not used"})
    print(f"registered {len(record['entries'])} checkpoints -> {REGISTRATION}")


def same_baselines(entries, main):
    """Every entry's persisted OI and climatology must be the main pass's: the same cells, the same OI."""
    first = main["entries"][0]["by_lead"]
    for e in entries:
        for lead in map(str, LEADS):
            for ch in CH:
                for row in ("persisted_oi", "climatology"):
                    a, b = e["by_lead"][lead]["all"][ch][row]["rmse"], first[lead]["all"][ch][row]["rmse"]
                    if abs(a - b) > 1e-9 * max(1., abs(b)):
                        raise SystemExit(f"{e['tag']} {row} lead {lead} {ch}: {a} differs from the main pass {b}")


def score(device):
    if RESULTS.exists():
        raise SystemExit(f"{RESULTS} exists: the transfer test is scored once")
    record = json.loads(REGISTRATION.read_text())
    SS.verify(record)
    results, cache = {"registration_sha256": SS.sha256(REGISTRATION), "started_unix": time.time(), "entries": []}, {}
    for entry in record["entries"]:
        started = time.time()
        checkpoint, outcome, macro = evaluate(entry["checkpoint"], device, cache)
        training = checkpoint["training"]
        results["entries"].append({**entry, "tag": training["tag"], "seed": checkpoint["seed"], "step": checkpoint["step"],
            "trained_on": training["cohort"], "backbone": checkpoint["config"]["backbone"],
            "kriging": checkpoint["config"]["kriging"], "params": int(sum(v.numel() for v in checkpoint["model"].values())),
            "normalization_trained": {k: {ch: checkpoint["normalization"][k][ch].tolist() for ch in CH} for k in ("mean", "std")},
            "seconds": time.time() - started, "by_lead": outcome, "macro_z": macro})
        print(json.dumps({"scored": training["tag"], "macro_z": {k: round(v, 4) for k, v in macro.items()}, "rmse": {
            lead: [round(outcome[lead]["all"][ch]["model"]["rmse"], 4) for ch in CH] for lead in map(str, LEADS)}}), flush=True)
    norm = cache["glorys"].norm
    results["normalization_used"] = {k: {ch: np.asarray(getattr(norm, k)[ch]).tolist() for ch in CH} for k in ("mean", "std")}
    same_baselines(results["entries"], json.loads(MAIN.read_text()))
    results["declared_extra_pass"] = {"main_results": str(MAIN), "main_results_sha256": SS.sha256(MAIN),
                                      "identity": "persisted OI and climatology of every entry equal the main pass's"}
    results["test_years_glorys"] = [2019, 2020]
    temporary = RESULTS.with_suffix(".tmp")
    temporary.write_text(json.dumps(results, indent=1, default=float) + "\n")
    os.replace(temporary, RESULTS)
    print(f"wrote {RESULTS}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", metavar="CHECKPOINT")
    p.add_argument("--register", nargs="+", metavar="CHECKPOINT")
    p.add_argument("--score", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if sum(map(bool, (args.check, args.register, args.score))) != 1:
        raise SystemExit("pass one of --check, --register, --score")
    if args.check:
        check(args.check, args.device)
    elif args.register:
        register(args.register)
    else:
        score(args.device)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run** `$PY -m pytest -q -p no:cacheprovider tests/test_fuxi_transfer.py` — Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add experiments/synthetic/173_glorys_transfer.py tests/test_fuxi_transfer.py
git commit -q -F - <<'EOF'
173_glorys_transfer: CESM2-trained checkpoints on the sealed GLORYS 2019-2020 cells (declared extra pass; GLORYS's own statistics; 159's baselines must come back)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

### Task 10: The KA check (on an open GPU)

Shows that Tasks 1-3 left KA's model and the scoring path unchanged. Needs one open GPU for about half an hour.

- [ ] **Step 1: Wait for an open GPU, then check** (Bash `run_in_background`):

```bash
open_gpu() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 > 80000 {print $1; exit}'; }
until G=$(open_gpu); [ -n "$G" ]; do sleep 600; done
CUDA_VISIBLE_DEVICES=$G $PY experiments/synthetic/173_glorys_transfer.py \
  --check outputs/size_20261023/y100_x1375_s1234/best_seed1234.pt > $SCRATCH/check_ka.log 2>&1
```

- [ ] **Step 2:** `tail -3 $SCRATCH/check_ka.log` — Expected: `check passed: 2004-2005 reproduced to a relative ...` (at most 1e-3). A failure here stops the plan: the FuXi changes touched KA's path; debug before anything else (superpowers:systematic-debugging).

### Task 11: The report `174_fuxi_transfer_report.py`

**Files:**
- Create: `experiments/synthetic/174_fuxi_transfer_report.py`
- Test: `tests/test_fuxi_transfer.py` (append)

- [ ] **Step 1: Failing tests** — append to `tests/test_fuxi_transfer.py`:

```python
report = script("174_fuxi_transfer_report")


def test_a_verdict_needs_a_difference_beyond_the_larger_seed_sd():
    assert report.verdict([.70, .71, .69], [.75, .76, .74])["verdict"] == "KA better"
    assert report.verdict([.75, .76, .74], [.70, .71, .69])["verdict"] == "FuXi-style better"
    assert report.verdict([.70, .72, .68], [.71, .73, .69])["verdict"] == "no difference beyond the seed spread"


def test_the_tables_name_each_model_with_its_size(monkeypatch):
    monkeypatch.setattr(report, "g_ka_params", lambda entries: 9_422_894)
    results = {"entries": [entry("y100_x1375_s1234", params=17_730_974), entry("y100_fuxi_s1234", params=17_508_596)]}
    glorys = report.transfer(results, {"entries": [entry("g_ka_s1234")]})
    text = report.markdown({}, glorys)
    assert "KA-Ocean (17.7M)" in text and "FuXi-Ocean-style (17.5M)" in text and "GLORYS-trained) (9.4M)" in text


def test_the_report_can_run_before_anything_finished():
    text = report.markdown({}, None)
    assert "Waits until both models have all three seeds" in text and "Waits for `173_glorys_transfer.py --score`" in text
```

- [ ] **Step 2: Run** `$PY -m pytest -q -p no:cacheprovider tests/test_fuxi_transfer.py` — Expected: collection error (`No such file ... 174_fuxi_transfer_report.py`).

- [ ] **Step 3: Implement** — create `experiments/synthetic/174_fuxi_transfer_report.py`:

```python
"""Report of the FuXi comparison and the GLORYS12 transfer
(spec ``docs/superpowers/specs/2026-10-26-fuxi-glorys-comparison-design.md``).

Reads KA-Ocean x1.375's summaries (the size study, ``outputs/size_20261023``), the FuXi-Ocean-style runs'
(``outputs/fuxi_20261026``), ``173``'s transfer results and, for ``g_ka``, ``159``'s main sealed results;
writes ``reports/synthetic/fuxi_transfer_20261026.md`` and ``.json``. Every table names each model with its
parameter count. A section whose inputs are not there yet says what it waits for.

  .venv/bin/python experiments/synthetic/174_fuxi_transfer_report.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/synthetic/fuxi_transfer_20261026"
TRANSFER = ROOT / "outputs/transfer_20261026/results.json"
MAIN = ROOT / "outputs/glorys_20261021/glorys12_test/sealed/results.json"
STEPS, SEEDS, HELD_OUT = 200000, 3, "2005"
CH, LEADS = ("TEMP", "SALT"), ("0", "1", "3", "6")
UNITS = {"TEMP": "°C", "SALT": "PSU"}
VARIABLE = {"TEMP": "temperature", "SALT": "salinity"}
#: CESM2-trained arm -> where its runs are
ARMS = {"y100_x1375": ROOT / "outputs/size_20261023", "y100_fuxi": ROOT / "outputs/fuxi_20261026"}
NAMES = {"y100_x1375": "KA-Ocean", "y100_fuxi": "FuXi-Ocean-style", "g_ka": "g_ka (KA-Ocean, GLORYS-trained)"}


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/synthetic" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


size_report = _script("170_size_report")          # score, by_lead, text
mean_sd, text = size_report.forecast_report.mean_sd, size_report.text


def runs():
    """{arm: its finished summaries}, each with the path of its saved forecast arrays."""
    found = {}
    for arm, folder in ARMS.items():
        for path in sorted(folder.glob(f"{arm}_s*/summary_seed*.json")):
            summary = json.loads(path.read_text())
            if summary.get("completed_steps") == STEPS and "forecast" in summary:
                summary["arrays"] = path.parent / f"forecast_seed{summary['seed']}.npz"
                found.setdefault(arm, []).append(summary)
    return found


def verdict(ka, fuxi):
    """The spec's rule on per-seed scores: a difference counts only beyond the larger seed sd."""
    (mk, sk), (mf, sf) = mean_sd(ka), mean_sd(fuxi)
    spread = max(sk or 0., sf or 0.)
    word = ("KA better" if mk < mf - spread else "FuXi-style better" if mf < mk - spread
            else "no difference beyond the seed spread")
    return {"verdict": word, "ka": mk, "fuxi": mf, "fuxi_minus_ka": mf - mk, "seed_spread": spread}


def g_ka_params(entries):
    """g_ka's parameter count, from its run's summary beside the registered checkpoint."""
    e = entries[0]
    return json.loads((Path(e["checkpoint"]).parent / f"summary_seed{e['seed']}.json").read_text())["params"]


def transfer(results, main):
    """GLORYS 2019-2020: RMSE by lead (mean, sd over seeds) per arm and variable, persisted OI and the
    climatology, the per-seed scores of the CESM2-trained arms, the parameter counts."""
    arms = {}
    for e in results["entries"]:
        arms.setdefault(e["tag"].rsplit("_s", 1)[0], []).append(e)
    rows = {**arms, "g_ka": [e for e in main["entries"] if e["tag"].startswith("g_ka_")]}
    first = results["entries"][0]["by_lead"]
    table = {lead: {**{f"oi_{ch}": first[lead]["all"][ch]["persisted_oi"]["rmse"] for ch in CH},
                    **{f"clim_{ch}": first[lead]["all"][ch]["climatology"]["rmse"] for ch in CH},
                    **{f"{arm}_{ch}": mean_sd([e["by_lead"][lead]["all"][ch]["model"]["rmse"] for e in entries])
                       for arm, entries in rows.items() for ch in CH}} for lead in LEADS}
    scores = {arm: [float(np.mean([e["macro_z"][lead] for lead in LEADS])) for e in entries] for arm, entries in arms.items()}
    return {"table": table, "scores": scores, "arms": list(rows),
            "params": {**{arm: entries[0]["params"] for arm, entries in arms.items()}, "g_ka": g_ka_params(rows["g_ka"])}}


def verdicts(found, glorys):
    if not (len(found) == len(ARMS) and all(len(s) == SEEDS for s in found.values())):
        return None
    out = {"cesm2_2005": verdict(*([size_report.score(s, HELD_OUT) for s in found[arm]] for arm in ARMS))}
    if glorys is not None:
        out["glorys_2019_2020"] = verdict(glorys["scores"]["y100_x1375"], glorys["scores"]["y100_fuxi"])
    return out


def markdown(found, glorys):
    """The report; ``found`` the finished CESM2 runs by arm, ``glorys`` ``transfer()``'s or None."""
    params = {**({} if glorys is None else glorys["params"]), **{arm: s[0]["params"] for arm, s in found.items()}}
    head = lambda arm: f"{NAMES[arm]} ({params[arm] / 1e6:.1f}M)"
    out = ["# KA-Ocean against a FuXi-Ocean-style model, and both in transfer to GLORYS12", "",
           "KA-Ocean x1.375 (the size study's choice) and a FuXi-Ocean-style model (Huang et al., arXiv:2506.03210, "
           "re-implemented: its code and weights are unreleased), trained alike on `cesm2_vlong_uniform` 1904-2003: "
           "200,000 steps of 1,024 queries, leads 0, 1, 3 and 6 months in one model, the checkpoint chosen on 2004. "
           "Spec: `docs/superpowers/specs/2026-10-26-fuxi-glorys-comparison-design.md`.", "",
           "## Models", "", "| model | parameters | trained on | seeds | hours a run | best steps |", "|---|---|---|---|---|---|"]
    for arm, summaries in found.items():
        hours = mean_sd([s["runtime_s"] / 3600 for s in summaries])
        out.append(f"| {NAMES[arm]} | {params[arm]:,} | 100 CESM2 years, 200k steps | {len(summaries)} | {text(hours, 1)} | "
                   f"{', '.join(str(s['best_step']) for s in summaries)} |")
    if glorys is not None:
        out.append(f"| {NAMES['g_ka']} | {params['g_ka']:,} | 20 GLORYS years, 100k steps | 3 | | |")
    out += ["", "## Verdicts", ""]
    v = verdicts(found, glorys)
    if v is None:
        out.append("Waits until both models have all three seeds: "
                   + ", ".join(f"{NAMES[a]} {len(found.get(a, []))}" for a in ARMS) + ".")
    else:
        for key, title in (("cesm2_2005", "CESM2 2005 (held out)"), ("glorys_2019_2020", "GLORYS12 2019-2020 (transfer)")):
            if key in v:
                out.append(f"* {title}: **{v[key]['verdict']}**; score KA-Ocean {v[key]['ka']:.4f}, FuXi-Ocean-style "
                           f"{v[key]['fuxi']:.4f}, larger seed sd {v[key]['seed_spread']:.4f}.")
    out += ["", "Score: mean over leads of the per-variable standardised RMSE (per-depth z units of the scored cohort), "
            "mean over seeds; a difference counts only beyond the larger seed sd."]
    leads = {arm: size_report.by_lead(s) for arm, s in found.items()}
    for ch in CH:
        out += ["", f"## CESM2 2005 (held out): {VARIABLE[ch]} RMSE ({UNITS[ch]})", ""]
        if not leads:
            out.append("Waits for finished runs.")
            continue
        first = next(iter(leads.values()))
        out += ["| lead | persisted OI | climatology | " + " | ".join(head(a) for a in leads) + " |",
                "|---" * (3 + len(leads)) + "|"]
        out += [f"| {lead} | {first[lead][f'oi_{ch}']:.4f} | {first[lead][f'clim_{ch}']:.4f} | "
                + " | ".join(text(leads[a][lead][ch]) for a in leads) + " |" for lead in LEADS]
    if glorys is None:
        out += ["", "## GLORYS12 2019-2020 (transfer)", "", "Waits for `173_glorys_transfer.py --score`."]
    else:
        order = [a for a in ("g_ka", "y100_x1375", "y100_fuxi") if a in glorys["arms"]]
        for ch in CH:
            out += ["", f"## GLORYS12 2019-2020 (transfer): {VARIABLE[ch]} RMSE ({UNITS[ch]})", "",
                    "| lead | persisted OI | climatology | " + " | ".join(head(a) for a in order) + " |",
                    "|---" * (3 + len(order)) + "|"]
            out += [f"| {lead} | {glorys['table'][lead][f'oi_{ch}']:.4f} | {glorys['table'][lead][f'clim_{ch}']:.4f} | "
                    + " | ".join(text(glorys["table"][lead][f"{a}_{ch}"]) for a in order) + " |" for lead in LEADS]
        transferred = [a for a in ("y100_x1375", "y100_fuxi") if a in glorys["arms"]]
        out += ["", "g_ka trained on 20 GLORYS years (the in-domain reference); KA-Ocean and FuXi-Ocean-style never saw GLORYS.",
                "", "## Transfer cost: GLORYS RMSE as a ratio to g_ka's", "",
                "| lead | " + " | ".join(f"{head(a)} {VARIABLE[ch]}" for a in transferred for ch in CH) + " |",
                "|---" * (1 + 2 * len(transferred)) + "|"]
        out += [f"| {lead} | " + " | ".join(f"{glorys['table'][lead][f'{a}_{ch}'][0] / glorys['table'][lead][f'g_ka_{ch}'][0]:.3f}"
                                           for a in transferred for ch in CH) + " |" for lead in LEADS]
    out += ["", "## Protocol and asymmetries", "",
            "* Equal: data, inputs (one month of profiles and surface grids), steps, queries, objective, optimiser, "
            "selection on 2004, the read-out after the state. FuXi-Ocean-style has no kriging read-out (KA's own).",
            "* Left out of FuXi-Ocean: Mixture-of-Time (one input month), autoregression (one pass per lead), the "
            "Charbonnier loss and its two training stages. Its own size is unpublished; it is matched to KA's.",
            "* KA-Ocean x1.375 won a three-size selection on 2004 (gain 0.0015, 0.2 %); FuXi-Ocean-style has one size.",
            "* GLORYS: a declared extra pass on the sealed 2019-2020 cells; GLORYS's own statistics; g_ka's rows are "
            "the 2026-10-22 main pass's, not re-scored. g_ka is smaller and trained half as long: a reference, not a competitor."]
    return "\n".join(out) + "\n"


def main():
    found = runs()
    glorys = (transfer(json.loads(TRANSFER.read_text()), json.loads(MAIN.read_text())) if TRANSFER.exists() else None)
    REPORT.with_suffix(".md").write_text(markdown(found, glorys))
    cesm2 = {arm: {"scores_2005": [size_report.score(s, HELD_OUT) for s in v], "by_lead": size_report.by_lead(v),
                   "params": v[0]["params"], "seeds": [s["seed"] for s in v]} for arm, v in found.items()}
    REPORT.with_suffix(".json").write_text(json.dumps({"cesm2_2005": cesm2, "glorys_2019_2020": glorys,
                                                       "verdicts": verdicts(found, glorys)}, indent=2) + "\n")
    print(REPORT.with_suffix(".md").relative_to(ROOT))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run** `$PY -m pytest -q -p no:cacheprovider tests/test_fuxi_transfer.py` — Expected: `11 passed`. Then `$PY experiments/synthetic/174_fuxi_transfer_report.py && head -30 reports/synthetic/fuxi_transfer_20261026.md` — Expected: the KA rows from the size study (`KA-Ocean (17.7M)` column in the CESM2 tables, KA's 2005 lead-0 T `0.0526 ± 0.0011` as in `size_20261023.md`), "Waits until both models have all three seeds", "Waits for `173_glorys_transfer.py --score`". Then `SUITE`.

- [ ] **Step 5: Commit** (the script only; the report is committed when the runs finish)

```bash
git add experiments/synthetic/174_fuxi_transfer_report.py tests/test_fuxi_transfer.py
git commit -q -F - <<'EOF'
174_fuxi_transfer_report: both comparisons, verdicts computed, every model named with its parameter count

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

### Task 12: When the FuXi runs finish

- [ ] **Step 1 (any time):** `head -c 800 outputs/fuxi_20261026/status_queue.json; grep -h '^validation' outputs/fuxi_20261026/y100_fuxi_s*/train.log | tail -3` — progress for Kevin.
- [ ] **Step 2 (`status_queue.json` has `"finished": true`): the FuXi check on an open GPU** (Bash `run_in_background`):

```bash
open_gpu() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 > 80000 {print $1; exit}'; }
until G=$(open_gpu); [ -n "$G" ]; do sleep 600; done
CUDA_VISIBLE_DEVICES=$G $PY experiments/synthetic/173_glorys_transfer.py \
  --check outputs/fuxi_20261026/y100_fuxi_s1234/best_seed1234.pt > $SCRATCH/check_fuxi.log 2>&1
```

Expected (`tail -1 $SCRATCH/check_fuxi.log`): `check passed: 2004-2005 reproduced to a relative ...`.
- [ ] **Step 3: Register the six checkpoints**

```bash
$PY experiments/synthetic/173_glorys_transfer.py --register $(for s in 1234 1235 1236; do
  echo outputs/size_20261023/y100_x1375_s$s/best_seed$s.pt; done) $(for s in 1234 1235 1236; do
  echo outputs/fuxi_20261026/y100_fuxi_s$s/best_seed$s.pt; done)
```

Expected: `registered 6 checkpoints -> .../outputs/transfer_20261026/registration.json`.
- [ ] **Step 4: Ask Kevin before scoring** (the spec: `--score` runs once, when Kevin asks). On his word, in the background:

```bash
open_gpu() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 > 80000 {print $1; exit}'; }
until G=$(open_gpu); [ -n "$G" ]; do sleep 600; done
CUDA_VISIBLE_DEVICES=$G $PY experiments/synthetic/173_glorys_transfer.py --score > $SCRATCH/score.log 2>&1
```

Expected (`tail -2 $SCRATCH/score.log`): six `"scored"` lines in all, then `wrote .../outputs/transfer_20261026/results.json`.
- [ ] **Step 5: Report:** `$PY experiments/synthetic/174_fuxi_transfer_report.py`; read `reports/synthetic/fuxi_transfer_20261026.md`; check one number by hand against its source (one arm, one lead, one variable, in a summary or in `results.json`).
- [ ] **Step 6: Commit the report**

```bash
git add reports/synthetic/fuxi_transfer_20261026.md reports/synthetic/fuxi_transfer_20261026.json
git commit -q -F - <<'EOF'
Reports: KA-Ocean against FuXi-Ocean-style on 100 CESM2 years, and both in transfer to GLORYS12 2019-2020

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

- [ ] **Step 7:** Report to Kevin: both verdicts, the tables' headline numbers with each model's size, the transfer cost. Merge `fuxi-transfer` only on his word.
