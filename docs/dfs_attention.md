# DFS-Attention — Scale-Aware Effective-Evidence Fusion

*Single source of truth for the method. The code (`src/ocean_tokenizer/dfs.py`,
the `DFSAttention` variant in `src/ocean_tokenizer/fusion.py`, and the metadata
stamped by `src/ocean_tokenizer/token_api.py`) implements exactly what is
written here — if they ever disagree, this document wins and the code is a bug.
Companion: [`token_measure_definition.md`](token_measure_definition.md), which
defines MBCA's hand-designed support masses and remains the baseline this
method is compared against.*

## The question

> How much **independent, useful** information does a heterogeneous observation
> set carry **at the target reconstruction scale**?

Not "how many tokens are there" (an artifact of tokenization, which standard
attention counts), and not "how much physical support did we assign by hand"
(MBCA, whose weights were designed rather than estimated, and whose evidence
was flattened to equal weights the moment the resampler ran).

## 1 · The estimator

Treat the token set as observations of a field with signal covariance `K` and
observation-error covariance `S`. The analysis is the kriging / optimal-
interpolation update against a climatological background,

    f̂ = K (K + S)⁻¹ y ,   H = K (K + S)⁻¹ ,   τ_i = H_ii ,   DFS = tr H = Σ_i τ_i

`τ_i ∈ [0,1]` is the fraction of observation *i* that the analysis takes from
the observation rather than from the background. It is simultaneously

* the **degrees of freedom for signal** of data assimilation, and
* the **ridge-leverage score** of observation *i* — the ridge parameter being
  the observation noise, and the prior being the background.

So "effective evidence" is not a new quantity invented for this paper; it is
the standard measure of how much an observing system actually adds, computed
per observation. The implementation uses the equivalent form

    τ_i = 1 − σ_i² [(K + S)⁻¹]_ii

Nothing about it is hand-designed: the only inputs are geometry, uncertainty,
provenance, and the target scale.

## 2 · The support kernel

`k_ij = k_h(Δx) · k_z(Δz) · k_t(Δt) · k_s(source_i, source_j)`

Each geometric leg is a **Gibbs (non-stationary) Gaussian** kernel, which stays
positive definite even though every token carries its own length scales:

    k_d(i,j) = √( 2 ℓ_i ℓ_j / (ℓ_i² + ℓ_j²) ) · exp( −Δ² / (ℓ_i² + ℓ_j²) )

Horizontal distance is the chordal distance between earth-centred coordinates
(km); vertical is the depth difference (m); temporal is the offset from the
analysis time (days).

### 2.1 Physical length scales — depth regimes

`DEPTH_REGIMES` in `dfs.py`, log-depth interpolated so the scales (and hence
the evidence) are continuous in depth:

| regime | from | ℓ_h | ℓ_z | why |
|---|---:|---:|---:|---|
| mixed layer | 0 m | 150 km | 75 m | vertically near-uniform: levels inside it are interchangeable |
| thermocline | 50 m | 150 km | 35 m | sharp structure: adjacent levels are *not* interchangeable |
| intermediate | 300 m | 250 km | 150 m | smoother water masses |
| deep ocean | 1000 m | 400 km | 400 m | slowly varying |

### 2.2 Stratification modifier

    ℓ_z ← ℓ_z(depth) / (1 + β · strat / strat_ref)

`strat` is stamped per token by the encoders: the magnitude of the vertical
gradient of the token's own content, in normalised units per 100 m. A token
sitting on a sharp gradient gets a short vertical scale, so its neighbours stop
being redundant with it. This is the mechanism behind "preserve fine levels
around strong gradients": at a 10 m target, the same 30 fine levels yield
**17.9 DFS** across a thermocline and **9.4 DFS** in a smooth column of equal
anomaly amplitude (`reports/dfs_evidence_probes.md` §3–4).

Because encoders see *anomaly* z-scores, `strat` measures anomaly structure —
which is the right thing: a vertically displaced thermocline appears in anomaly
space as exactly the sharp dipole that must not be smoothed away. The
climatological stratification enters through the depth-regime table.

### 2.3 Target-resolution awareness

    ℓ_eff = √( ℓ_phys² + Δ_target² )        for each of h, z, t

This is the representer argument: a target field at resolution Δ is the true
field convolved with an averaging footprint of that width, so its covariance is
the physical covariance convolved with the same footprint. Consequences, all
monotone by construction:

* coarse target → long ℓ → nearby observations become mutually redundant;
* fine target → ℓ falls back to the physical scale and they regain independence.

Measured on one 2 dbar profile (`reports/dfs_evidence_probes.md` §5): DFS rises
4.2 → 6.3 → 9.6 → 14.3 → 19.7 → 21.1 as Δz goes 250 → 100 → 50 → 25 → 10 → 5 m.
Two profiles 5 km apart score ×1.03 of a single profile at a 5° target and
×1.67 at a 5 km target — the plan's horizontal test case, reproduced.

### 2.4 Provenance

`k_s = 1` within a processing stream and `S_CROSS = 0.8` across streams:
observations sharing a retrieval / processing chain share error and are
therefore more mutually redundant than two independent instruments at the same
place. Two overlapping products of one field score 224 DFS where one alone
scores 128 — partially, not fully, redundant.

## 3 · Exact duplication

`record_id` identifies the **raw measurement** a token stands for; identity is
`(modality, record_id)`. Tokens sharing it are the same measurement re-ingested
— an exact duplicate, a retokenization, or a real-time / delayed-mode pair of
one float cycle. Such a group is perfectly correlated by definition, so:

* the kernel between its members is forced to exactly 1;
* only one member enters the linear system (a perfectly correlated block is
  singular, and solving it is unnecessary);
* the group's single degree of freedom is split evenly, `τ_i = τ_group / c`.

Total DFS is therefore **exactly** invariant to duplication — verified to
float precision for ×2/×4/×8 duplication of single tokens and whole profiles,
and for real-time + delayed-mode copies (`tests/test_dfs_evidence.py`).

A *new* float at the same site is a different record and is not merged; it adds
evidence sub-linearly, through noise averaging, as it should.

Stamped by the encoders as: profiles `(parent, band)` — where `parent`
defaults to the column index but can be supplied so two input columns declare
themselves one float cycle; grid patches `(family, slot)`; points
`(parent, variable)`.

## 4 · Conservative evidence transport

MBCA's first limitation: after the modality resampler, output tokens carried
*equal* weight, so the estimated evidence was discarded. `EvidenceResampler`
makes the resampler a transport plan instead. Each token distributes its whole
evidence over the slots — a softmax over **slots**, not over tokens (the
Slot-Attention normalisation):

    A_is ≥ 0 ,   Σ_s A_is = 1 ,   ν_s = Σ_i A_is τ_i    ⟹   Σ_s ν_s = Σ_i τ_i

exactly, to float precision (asserted in `tests/test_dfs_evidence.py`). Slot
content is the evidence-weighted mean of the tokens assigned to it, so a slot's
value and its outgoing mass describe the same evidence.

## 5 · Background-referenced latent fusion

The latent array first reads the climatological background (WOA), then attends
to the transported evidence with an additive `log ν_s` prior against a
background key held at `log λ_bg`:

    z = z_bg + CrossAttn( z, [resampled obs ; background] ;
                          bias = [log ν , log λ_bg] )

Because `ν` is an *absolute* number of observed degrees of freedom (not a
normalised weight), the comparison `ν` vs `λ_bg` is meaningful: where evidence
is thin the fusion falls back on climatology, where it is rich the observations
win. `λ_bg` is the ridge parameter of §1 wearing its other hat — the background
precision — and is learned.

The WOA modality is excluded from the evidence set (`DFSAttention.BACKGROUND`):
the background is what evidence is measured *against*, not evidence itself. Its
DFS is exactly 0 by construction.

## 6 · Scale-conditioned decoding

`decode(latent, query_coord, query_scale)` adds `(log Δx_q/Δx₀, log Δz_q/Δz₀)`
to the query embedding, so the decoder answers the same question the evidence
estimator was asked. Training at more than one scale is opt-in
(`experiments/18_full_train.py --scale-aug`), which box-averages the truth over
`fz` levels × `fx`×`fx` cells and asks for that field at that resolution.
With `--scale-aug 0` (the default) the run is a like-for-like protocol_v1
comparison against the other three variants and the scale-conditioning weights
stay at their identity initialisation.

## 7 · Cost and the localisation approximation

The exact `(K+S)⁻¹` is O(N³). The kernel decays, so each token's leverage is
solved against its `k` nearest tokens — the same locality argument that makes
the LETKF local. Cost O(N k³); ~60 ms for the ~6.5k observation tokens of one
analysis month on an A100, about 2.4× a standard fusion step.

**Regime of validity, measured.** The approximation is tight exactly when the
neighbourhood covers all the correlation there is, and truncation always biases
DFS *upward* (a token that cannot see its redundant partners believes it is
informative). `DFSResult.neighbour_cut` reports the mean kernel value at the
edge of the neighbourhood so this is observable rather than assumed.

| geometry | k=16 | k=32 | k=256 |
|---|---:|---:|---:|
| protocol_v1, 1500 profiles (6540 tokens) | 6355.6 | 6354.6 | 6354.5 |
| protocol_v1, 3000 profiles (12540 tokens) | 12045.5 | 12027.2 | 12021.8 |
| 150 profiles packed in a 2° box (600 tokens) | 106.5 | 80.9 | 48.8 *(exact)* |

On the deployed observing geometry k = 32 is converged to < 0.05 %. In a
pathologically dense cluster it is not, and the reported numbers there would be
optimistic — which is why `experiments/23_dfs_evidence_probes.py` solves those
small sets **exactly** (`k_neighbors=None`) rather than relying on the
approximation.

## 8 · What is claimed, and what is not

Claimed: the combination of (a) three-dimensional, stratification- and
target-resolution-aware support representation for grid, profile and point
observations, (b) set-level ridge-leverage evidence estimation over the whole
heterogeneous set, (c) exact consolidation of re-ingested measurements via
provenance, (d) conservative transport of that evidence through a fixed-size
resampler, and (e) background-referenced fusion in which the ridge parameter is
the background precision.

Not claimed as novel: Perceiver IO, shared latents, coordinate-query decoding,
log-weighted attention, ridge leverage itself, DFS itself, mass-conserving
attention itself, Gibbs kernels, or LETKF-style localisation.

Implemented but not exercised by protocol_v1: **irregular point observations**.
`PointEncoder` stamps the full DFS metadata and `dfs_scores` scores point
tokens like any other, but the protocol's input set is profiles + surface +
WOA, so `build_fusion_model` does not wire a point encoder into the variants
(adding one would change every variant's parameter count and break
comparability with the existing checkpoints). Point observations enter by
passing a `point` key in the observation dict of a model built with one.

Not demonstrated: any of this on **real** Argo. The store on this machine holds
CESM2-LE and WOA23 only; every profile here is synthetic (OSSE). The provenance
machinery (float id, cycle, real-time vs delayed mode, sensor, processing
version) is implemented and unit-tested, but has only been exercised on
constructed duplicates, not on a real duplicated Argo record.

## 9 · Success criterion

Per the plan, DFS-Attention must satisfy all three at once — low duplication
sensitivity is trivially achievable by ignoring the profiles:

    low duplication sensitivity + high observation retention + competitive accuracy

Reported together by `experiments/25_dfs_report.py`. Retention is measured two
ways: the share of estimated evidence that comes from profiles, and the RMSE
degradation when profiles are withheld from a trained model (a model that
ignores Argo degrades by nothing).
