# D4RT query decoder (§2.3/§2.4) on the protocol_v1 line — design

Date: 2026-08-16
Branch: `intern/d4rt-query-decoder`
Source spec: `dfs_d4rt_intern_plan.md` §2.3, §2.4 (mentor-provided)
Reference implementation: <https://github.com/Lijiaxin0111/Open-d4rt> (read only;
no checkpoint or source file is vendored)

---

## 0. Provenance warning — read before citing anything

`dfs_d4rt_intern_plan.md` is written in the past tense as a completed audit, but
it was **drafted forward as a target and never run**. Confirmed with the author
2026-08-16. Nothing in it that resembles a measurement may be cited as a result:

- §7 validation table
- §8 frozen-2025 table and lead-wise table
- §9 missing-modality and duplicate-attack tables
- §6.3 nine checkpoint SHA-256 values
- §4 GODAS manifest SHA-256 and the 147,548,660-byte size
- §5.3 "106 passed in 3.08s"

Verified 2026-08-16 that the system those sections describe does not exist in
this repository: no `/DATA2`, no `src/dfs_attention/`, no `godas.py`,
`oi_residual.py`, `local_attention.py`, `objective_interpolation.py`, or
`losses.py`, no `experiments/13_download_godas.py` / `14_godas_dfs_d4rt.py` /
`16_summarize_refined_godas.py`, on any of the six branches, in any commit
(`git log --all -S"godas"` is empty), with zero GODAS files on the host.

Any number this work produces is a fresh measurement and must be reported as
such, never merged with or compared against the placeholder tables above.

## 1. Scope

**In scope** — mentor doc §2.3 and §2.4, built on the existing protocol_v1 /
CESM2-LE OSSE line:

- causal space-time query `(x, y, z, t_src, t_tgt)`
- shared-latent, independent-query decoder with no query-to-query attention
- query-local refiner attending directly to encoded observation tokens
- temperature/salinity channel experts
- the causality and invariance tests of §3

**Out of scope** — not built, not referenced in any result:

- §2.5 causal objective interpolation, §2.6 frozen OI residual and cBottle loss
- §§4–6 the GODAS experiment entirely (different dataset, region 25–50 N /
  280–330 E, 2000–2025, 16 levels, 38×26 grid, 2000-2018 / 2019-2021 /
  2022-2024 / 2025 splits)
- CNP baseline, superobbing, duplicate attack, raw-Argo phase (§11)

**Consequence to state plainly**: mentor §8 concludes *"the common
OI-residual/channel-expert design, rather than DFS specifically, explains the
large gain over the non-OI rows."* Since the OI residual is out of scope, this
work should not be expected to move RMSE much. Its defensible contribution is
the **new capability** — leads 1–3 existing at all — plus the query-independence
guarantees, not accuracy.

## 2. Data and protocol mapping

The mentor doc specifies GODAS. This work runs on protocol_v1
(`configs/protocol_v1.yaml`, `reports/protocol_v1.md`), which is frozen and not
modified here.

| Item | protocol_v1 value |
|---|---|
| Grid | global 1°, 180×360, 20 levels, 5–985 m |
| Train | 1985–2007, 276 months — climatology and z-stats from these ONLY |
| Validation | 2008–2010, 36 months — checkpoint selection |
| Test | 12 pinned months, zarr indices 1933, 1935, 1936, 1938, 1942, 1946, 1952, 1953, 1956, 1965, 1967, 1976 |
| Metric | unobserved-only anomaly RMSE in z-space |
| Floor | train-only climatology, 0.552 °C / 0.131 PSU on the pinned test months |

Never report ">1000 m" on this 20-level grid.

Mentor §2.1 token counts (24 profiles × 16 depths, 70 patches) are GODAS-sized.
The **structure** is followed; the **counts** come from protocol_v1: profile
count K is the existing augmentation axis, depths are 20, patch layout is the
existing `GridPatchEncoder` 10×12.

### 2.1 Split-boundary rule (mentor §3.7)

Mentor §3.7 requires all context and target months to stay inside their named
split. protocol_v1's test set is 12 *scattered* pinned months rather than a
contiguous era, so the rule is mapped as:

- `t_src` is the pinned test month; targets are `t_src + Δ` for `Δ ∈ 0..3`
- assert `t_src − 1` (the context month) and every `t_src + Δ` fall strictly
  after the validation period ends
- for train and validation, `t_src` is drawn such that `t_src − 1` and
  `t_src + 3` both remain inside that split; months within 1 of the lower
  bound or 3 of the upper bound are not eligible as `t_src`

Lead 0 therefore reproduces the protocol_v1 headline exactly, and because the
unobserved-cell mask is fixed by the `t_src` profile draw, leads 0–3 are scored
on identical cells and are directly comparable.

Baselines for the lead table: **persistence** (the model's own lead-0
reconstruction reused as the forecast) and **climatology** (zero anomaly, floor
recomputed per target month).

## 3. Architecture

All dimensions follow mentor §2.3 verbatim.

```
width d_model      = 64
heads              = 4
latent blocks      = 2
decoder blocks     = 2
global latents     = 32
observation slots  = 32
reference slots    = 8   (learned, availability-conditioned)
local refiner chunk= 128 (memory only; does not couple queries)
```

### 3.1 Risk recorded against the latent budget

32 unstructured global latents on the 180×360 grid is roughly one latent per
2000 ocean cells. `docs/` and the Week-4 notes record that an unstructured
global latent reached only ~6 % skill before month-memorisation overtook
learning, and that `anchor_grid` (one latent per coarse map cell) was introduced
to fix exactly this; see `AttnFusionModel` in `src/ocean_tokenizer/fusion.py`.
The mentor's 32 was sized for a 38×26 regional grid, roughly 1/60 the cells.

Decision: build as specified. `N_LATENT` is a single module constant so the
anchored alternative is a one-line change if the training curve reproduces the
Week-4 collapse. If it does, that is a finding to report, not a silent
deviation.

Second consequence: at `d_model=64` the existing checkpoints (`d_model=128`,
`Linear(68, 128)` projections) cannot warm-start. This build trains from
scratch.

### 3.2 Causal space-time query

Query is `(x, y, z, t_src, t_tgt)` per §2.3, represented as:

- `query_coord` `(..., 4)` = `(lat, lon, depth, month)` where `month` is the
  **target's** calendar month, consumed by the existing `coord_features`
- `lead` `(...,)` int64 = `t_tgt − t_src`, in months, range `0..3`

`coord_features` is **not modified**. It stays at `N_COORD_FEATS = 68` with
fourier_v2 intact. Rationale: `q_proj` and all three encoder `coord_proj` are
`Linear(68, ·)`; widening the feature vector breaks every saved checkpoint, and
target time is meaningless on the encoder side, where a token has a `t_obs` but
no `t_tgt`.

Lead enters additively:

```python
lead_embed = nn.Embedding(max_lead + 1, d_model, padding_idx=0)
q = q_proj(coord_features(query_coord)) + lead_embed(lead)
```

`padding_idx=0` pins row 0 at exactly zero and excludes it from gradients, so
**lead 0 contributes zero lead signal for the entire life of training**, not
merely at initialisation. This mirrors the precedent already set by the
zero-initialised `scale_proj` in `fusion.py`. Reconstruction is provably
unaffected by the forecasting machinery.

The encoder window stops at `t_src`. `t_tgt > t_src` is a genuine causal
forecast request; no input value is read from `t_tgt`.

### 3.3 Observation context window (mentor §2.1, §3.1)

Context is `[t_src − 1, t_src]`:

- profile point tokens at `t_src` only
- surface T/S patch tokens at both context months
- SSH patch tokens at both context months

Implemented by extending the observation assembly to stack two months of
surface and SSH fields; tokens already carry their own time via
`TokenBatch.time_offset`, so no encoder change is required.

### 3.4 Slots and shared latent

`32` observation slots plus `8` learned reference slots feed `32` global
latents through `2` latent blocks. The `8` reference slots are
availability-conditioned, generalising the single `null_token` currently in
`fusion.py` so that a dropped modality degrades gracefully rather than starving
cross-attention. Concretely: a learned `(8, d_model)` base table plus
`Linear(n_modalities, d_model)` applied to the binary modality-availability
vector and broadcast across the 8 slots. This is always-present key/value mass,
so attention never faces an empty or near-empty key set.

### 3.5 Independent-query decoder

`2` decoder blocks, each pre-norm cross-attention from queries to the shared
latent plus a feed-forward, with residuals — the Open-d4rt
`IndependentQueryDecoder` pattern. Every target query cross-attends the shared
latent and owns its MLP. **There is no self-attention between target queries.**

This is the load-bearing property of §2.3: adding, deleting, or permuting
queries cannot change any other query's prediction. It is enforced
architecturally and verified by test (§5).

### 3.6 Query-local refiner (§2.4)

The fixed global slot bottleneck can blur a nearby profile or patch, so each
decoded query additionally attends directly to all active encoded observation
tokens. Attention score:

```
content dot product
+ negative Gaussian distance in (dx, dy, dz, t_obs − t_tgt)
+ beta_head * log(evidence_mass)
```

with learned relative-value features `(dx, dy, dz, t_obs − t_src,
t_obs − t_tgt)`.

Frozen initial values from §2.4:

| parameter | value |
|---|---|
| local length scales | `(0.35, 0.35, 0.30, 3 months)` |
| mass exponent `p` | `1`, i.e. the bias term is `beta_head * log(tau**p)` |
| per-feature residual gate init | `0.05` |
| query chunk | `128` |

`beta_head` is one learned scalar per attention head, initialised to the mass
exponent's neutral value. The four length scales are learned, initialised to the
values above; the first three are in normalised coordinate units and the fourth
in months.

Two existing fields supply this with no new plumbing:

- `evidence_mass` = `DFSResult.tau`, the per-token degrees of freedom already
  produced by `DFSAttention.evidence()`
- the temporal term uses `TokenBatch.time_offset`, defined as signed offset in
  days from the analysis time

Using `time_offset` rather than `coord[..., 3]` is required for correctness:
`coord[..., 3]` is a calendar month in 1–12, so a December-to-January
difference computes as −11 instead of +1.

Chunking at 128 is a memory device only. Because queries never attend to one
another, chunk boundaries cannot change any prediction — asserted by the
extension-invariance test.

### 3.7 Channel experts (§2.4)

- **Temperature**: shared physical-state query → shared local refiner →
  temperature component of the shared head
- **Salinity**: learned salinity-variable embedding → separate local refiner →
  scalar variable head

### 3.8 Deliberately not touched

DFS evidence `τ_i` is **not** discounted by lead. An observation at `t_src` is
genuinely weaker evidence about `t_src + 3`, and making `τ` lead-aware is the
interesting mechanism — but evidence weighting belongs to the paper line and
editing `dfs.py` would tangle the two workstreams. Recorded as future work.

## 4. Module layout

| Path | Status | Contents |
|---|---|---|
| `src/ocean_tokenizer/query_decoder.py` | new | `LeadEmbedding`, `QueryCrossBlock`, `IndependentQueryDecoder`, `QueryLocalRefiner`, `ChannelExpertHead` |
| `src/ocean_tokenizer/fusion.py` | edited | reference slots, wiring of the decoder and refiner, `lead` threaded through `decode`/`forward` |
| `src/ocean_tokenizer/fullrun.py` | edited | two-month context assembly; lead-aware query/target packs |
| `tests/test_query_decoder.py` | new | the gates in §5 |
| `experiments/34_d4rt_lead_train.py` | new | training with the reconstruction/forecast query mix |
| `experiments/35_d4rt_lead_eval.py` | new | leads 0–3 vs persistence and climatology |
| `reports/d4rt_lead_probe.md` | new | results, written only from real runs |

Each unit is independently testable: the refiner is a pure function of
(query features, token features, τ, offsets); the decoder is a pure function of
(query features, latent); the lead embedding is a lookup.

## 5. Test gates

Following mentor §3, adapted to the pieces in scope:

1. **Query permutation invariance** — permuting the query batch permutes the
   output identically, exactly.
2. **Query extension invariance** — decoding 100 queries and decoding those
   same 100 inside a batch of 1000 gives bit-identical values for the shared
   100. Also run across a chunk boundary (chunk 128) to prove chunking does not
   couple queries.
3. **Token causality** — a token whose centre time is later than `t_src` raises.
4. **Support causality, checked independently** — a token whose *centre* is
   legal but whose temporal support (`time_offset ± support_t/2`) extends past
   `t_src` raises. Mentor §3.3 phrases this over quadrature-node times; this
   repo's `TokenBatch` carries support intervals rather than explicit quadrature
   nodes, so the support interval is the mapped equivalent. Tested separately
   from gate 3 so neither can mask the other.
5. **Reconstruction-path isolation** — with every query at lead 0, the forward
   pass is bit-identical whether or not `lead_embed` is present in the module.
   (This replaces a bit-identity check against the *current* `decode` path,
   which is not available: at `d_model=64` the dimensions no longer match the
   existing `d_model=128` implementation, so the two are not comparable.)
6. **Lead-0 zero-signal invariant** — `lead_embed.weight[0]` is exactly zero and
   has no gradient, after optimizer steps as well as at init.
7. **Integer lead indexing** — non-integer or out-of-range `lead` raises rather
   than silently rounding or wrapping.
8. **Masked NaN padding** — padded and all-NaN tokens contribute nothing to the
   refiner; the masked value never leaks through the `log(τ)` term.
9. **Empty evidence** — zero active observation tokens produces a finite
   prediction (background/reference-slot path) rather than NaN.
10. **Lead-0 target exclusion** (mentor §3.5) — profile columns supplied as
    inputs are excluded from lead-0 targets, so a point-copy cannot score.

## 6. Acceptance

Deliverable is code + tests + a smoke run proving the loop closes. The full
training run is launched by the user.

- all gates in §5 green
- smoke run: loss finite and decreasing, lead-0 score beating the protocol_v1
  climatology floor

Note the lead-0 score is **not** required to match the existing 0.149-class
depthwise U-Net or the `d_model=128` fusion numbers. This is a
from-scratch, narrower (`d_model=64`, 32-latent) architecture; the older
numbers are context, not a pass bar. Beating the floor is the bar; the §3.1
risk is the thing to watch.
- report written only from real runs, with no number traceable to §§7–9 of the
  mentor doc

A lead-1 forecast that beats persistence is the probe's positive result.
Whatever the outcome, the encoder window stays at `[t_src−1, t_src]`; widening
it further is a separate decision.
