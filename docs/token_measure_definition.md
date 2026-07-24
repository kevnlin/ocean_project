# Token Support-Mass & Provenance Definitions (Task 4)

*Single source of truth for what each token's `support_mass` means and how the
provenance fields identify "same evidence, different representation." The code
(`src/ocean_tokenizer/token_api.py`) implements exactly what is written here —
if they ever disagree, this document wins and the code is a bug.*

## Purpose

Standard attention weights evidence by **token count** — an arbitrary artifact
of tokenization. MBCA instead treats tokens as **quadrature elements of an
observation measure**: each token carries a nonnegative `support_mass` μ
describing how much physical observation support it represents. Masses are
normalized *within a modality* (μ̄ = μ/Σμ) and scaled by an equal modality
prior π_m over *present* modalities, so:

- refining a field into more tokens never increases its total attention mass;
- a modality's influence never depends on its raw token count;
- splitting one token into n identical children of mass μ/n leaves every
  attention output exactly unchanged (partition invariance).

Because of the within-modality normalization, each mass definition only needs
to be **proportionally correct within its own modality** — units cancel.

## Mass definitions

### 1 · Surface grid patch (`GridPatchEncoder`, 2-D input)

    μ_i = Σ_{valid cells in patch} cos(lat_cell) · Δlat · Δlon

`cos(lat)·Δlat·Δlon` is the spherical area of a cell (up to the constant R²,
which cancels in normalization); the grid spacings Δlat, Δlon are inferred
from the supplied coordinate vectors (`(max−min)/(n−1)`), so a genuinely
refined grid — 2× the cells at half the spacing — conserves every region's
mass exactly instead of quadrupling it. A cell is *valid* when any variable
is finite there (ocean, not land/missing). Consequences:

- a patch that is half land carries half the mass of a full-ocean patch;
- polar patches carry less mass than equatorial ones, matching their area;
- refining the grid 2× (4× the tokens) preserves the modality's total mass.

### 2 · Prior-volume patch (`GridPatchEncoder`, 3-D input, one token per
(level, patch))

    μ_i = [Σ_{valid cells} cos(lat_cell)] × Δz(level)

where Δz(level) is the **represented layer thickness**: the interval between
the midpoints to the adjacent levels (first/last levels extend half their
neighbor gap, clamped at 0 m). On the 20-level grid Δz grows from 10 m near
the surface to ~230 m at 985 m depth, so deep tokens correctly represent more
ocean volume than surface tokens.

**Depth-as-channels caveat (documented per the brief):** when a *different*
component (e.g. the joint-depth U-Net) encodes depth as channels, one spatial
token represents the **entire water column** over its patch: its 3-D support
is area × (full column span), not area × Δz. Any future channel-stacked
tokenizer must set `depth_lower/depth_upper` to the full column and use the
column span in μ.

### 3 · Profile segment token (`ProfileEncoder`, one token per physical depth
band per profile)

    μ_i = valid represented depth span within the band  [metres]

Each sampled level represents its inter-level interval (midpoint edges,
clamped to the band); μ sums those intervals over *finite* levels only. Hence:

- a band below a shallow float's max depth has μ = 0 (and is masked);
- resampling the same smooth profile at 2× the levels leaves μ ~unchanged
  (intervals halve, count doubles) — mass tracks span, not level count;
- a single-level profile represents a nominal 50 m interval.

**First-implementation rule (active by default,
`normalize_per_profile=True`):** band masses are normalized to sum to 1
*within each profile* **before** the modality-level normalization, so a
profile with more sampled depth levels does not automatically receive more
total mass than a sparsely sampled one. Set the flag to `False` to emit raw
metre spans (used by tests that check the span semantics).

### 4 · Point observation (`PointEncoder`) — excluded from the first MVP

Points currently carry **no mass** (`support_mass=None` → MBCA falls back to
uniform-within-modality). When added later, point mass must reflect
**reliability and local redundancy** (e.g. inverse local point density ×
reliability), *not* an unlimited physical support for one point — otherwise a
cluster of near-duplicate point sensors would dominate its modality.

## Provenance metadata (on every `TokenBatch`)

| field | type | meaning | current values |
|---|---|---|---|
| `parent_id` | int64 (B,N) | id of the source observation within its modality; tokens sharing `(modality, parent_id)` are one observation | profile index; 0 for a whole gridded field; point index |
| `family_id` | int64 (B,N) | tokenization-scheme id, so two *representations* of the same evidence are distinguishable from two *observations* | grid: `10000·ph + pw`; profile: `n_bands`; point: 0 |
| `variable_id` | int64 (B,N) | `VAR_IDS` (TEMP 0, SALT 1, SST 2, SSS 3); −1 = token carries multiple variables | grid/profile: −1; point: its variable |
| `depth_lower` / `depth_upper` | float (B,N) | represented depth interval (m) | band/level edges; surface: degenerate at the surface depth |
| `reliability` | float (B,N), optional | per-token quality in [0,1] | unset (None) — reserved for real-obs work |

Duplication semantics: *re-ingesting the same observation* (same modality,
same parent, same family) adds **no** evidence — the Task-6 controlled test
divides the original mass across copies and MBCA's output is exactly
unchanged. A *new* observation at the same location (different `parent_id`)
legitimately adds mass.

## What the fusion stage assumes

`mbca_weights(mass, modality, mask)` in `fusion.py` consumes only
`support_mass`, `modality`, `mask`. It clamps masses at 0, treats a missing
mass vector as uniform, normalizes within modality, and assigns equal π_m
over modalities present in the batch item (renormalized under missing
modalities). No learned quality gate and no uncertainty prediction in this
version — deliberately, to keep the first experiments interpretable.
