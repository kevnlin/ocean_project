# Reading note — cBottle (*Climate in a Bottle*)

**Brenowitz, Ge, Subramaniam, Manshausen, Gupta, Hall, Mardani, Vahdat, Kashinath, Pritchard (NVIDIA).**
*Climate in a Bottle: Towards a Generative Foundation Model for the Kilometer-Scale Global Atmosphere*,
[arXiv:2505.06474](https://arxiv.org/abs/2505.06474).

Advisor-assigned reference. This note records what the paper actually claims,
where it differs from our task, and the four ideas worth importing. The design
implications live in [doc/architecture_spec.md](../doc/architecture_spec.md) §F.

---

## 1. What the paper does

cBottle is a **generative diffusion framework** that emulates global 5 km
atmospheric states — climate simulation *and* reanalysis — on the HEALPix grid,
at 12.5 M pixels per global field. Its central design choice is to **sample from
the full distribution of atmospheric states rather than roll forward
autoregressively**. The paper's stated motivation: autoregressive emulators
drift, go unstable over climate time horizons, are hard to scale to high
resolution, and force you to sift through enormous output to find rare extremes.

Two stages (per the abstract):

1. a **coarse-resolution generator**, conditioned on sea-surface temperature and
   solar position;
2. a **patch-based 16× super-resolution stage**.

Validation is distributional, not pointwise: diurnal-to-seasonal variability,
large-scale modes of variability, tropical-cyclone statistics, and trends in
climate change and weather extremes. The paper positions cBottle as *a step
toward* a foundation model on the strength of bridging data modalities
(reanalysis ↔ simulation) and supporting **zero-shot bias correction,
downscaling, and data infilling**. It also demonstrates guided diffusion: a
tropical-cyclone classifier trained alongside the generator lets them guide
sampling toward physically credible TC states.

> **Provenance caution.** The v1 abstract describes the *two* stages above. The
> intern plan refers to "three models (cBottle-3D / cBottle-Video / cBottle-SR)"
> and to masked-frame conditioning in a "cBottle-Video". I could not confirm a
> video variant from the abstract; treat that framing as unverified until
> someone reads the full paper body. The four ideas below do not depend on it,
> except idea 4, which is flagged accordingly.

---

## 2. Why nothing transplants directly

| | cBottle | OceanLatent (this repo) |
|---|---|---|
| Medium | Atmosphere | Ocean interior |
| Task | Generate plausible states from the climate distribution | Reconstruct **the one true state** consistent with the observations of a given month |
| Conditioning | SST + solar position (a handful of global fields) | Sparse in-situ profiles + dense surface fields + a WOA prior |
| Observation operator | none | central — the whole problem is that observations are sparse and heterogeneous |
| Representation | pixel-space diffusion on HEALPix | tokens → shared latent → coordinate query decoder |
| Metric | distributional (variability, extremes, modes) | pointwise unobserved-only anomaly RMSE against a known truth |
| Baseline to beat | prior generative emulators | **optimal interpolation** ([oi.py](../src/ocean_tokenizer/oi.py)) |

The gap that matters: cBottle answers *"what does a plausible atmosphere look
like?"*; we answer *"given these 1500 profiles, what is the ocean actually doing
at the 40 895 cells we did not observe?"*. A distributional sampler scores well
on cBottle's tests while being wrong at every individual unobserved cell, which
is exactly the quantity protocol_v1 measures. So cBottle is a source of
*mechanisms*, not of methodology.

---

## 3. The four transferable ideas

### 3.1 Multi-modal masked loss → **directly relevant, Phase 4.4**

Missing modalities/channels are zero-filled and the loss is renormalised by the
unmasked fraction, so each modality contributes equally regardless of how often
it is present.

*For us*: our sparse profiles are an extreme case of a heavily-masked channel
(~3.5 % coverage against dense SST/SSS and a dense WOA prior). The current loss
([baselines.py:361](../src/ocean_tokenizer/baselines.py#L361)) renormalises by
the spatial weight mask but does **not** renormalise per modality, and the
token-count imbalance (~10 k grid tokens vs ~6 k profile tokens) is a known open
problem. Per-modality renormalisation is a cheap, principled candidate — it is
row (c) of the Phase-4.4 loss ablation.

### 3.2 Zero-shot channel infilling → **our task in diffusion form**

Train with random channel dropout, then reconstruct unobserved channels from
observed ones at inference with no task-specific head.

*For us*: this is a genuine alternative route to the whole problem — a
diffusion model over (T, S, SST, SSS, SSH) with profiles as a partially observed
channel, sampled conditionally. Worth logging as a **future alternative
architecture**, not a change of course: it would forfeit the coordinate-query
interface that makes super-resolution and forecasting one operation (Phase 6),
and it optimises a distributional objective while we are scored pointwise.

### 3.3 Cascaded super-resolution → **the long-term Phase 6.2 reference**

Coarse generation followed by patch-based 16× SR with overlapping-patch
multi-diffusion blending, which is how they reach 12.5 M pixels without a
12.5 M-pixel generator.

*For us*: the advisor's "aiming for degree not just one degree, and super
resolution" ask. Note the contrast worth making explicit in the spec: cBottle
buys resolution with a **second cascaded model**, whereas a coordinate-query
decoder buys it by **evaluating queries off-grid** — no second model, no
blending seams. Phase 6.2 should measure both routes rather than assume ours
wins; the cBottle cascade is the reference design at km scale.

### 3.4 Masked-frame conditioning for forecasting → **the competing design to Phase 6.1**

*(Attributed in the plan to a cBottle-Video variant — unverified, see §1.)*
The idea itself is clear and stands on its own: instead of autoregressive
rollout, condition on lead frames and mask the future, letting the generative
model fill it in.

*For us*: this is the direct competitor to the D4RT-style latent query
(Phase 6.1). The two differ in where time enters — masked-frame conditioning
puts the future on the **input** side (a masked frame to be generated), the
D4RT format puts it on the **query** side (a target-time coordinate). Both
avoid rollout drift, which is the shared motivation. They belong side by side in
the Phase 6.1 discussion, and the choice is empirical.

---

## 4. What I would actually take

Ranked by expected value to this project:

1. **Per-modality loss renormalisation** (3.1) — testable now, in the existing
   Phase-4.4 ablation, at zero architectural cost.
2. **The framing of resolution as a cascade vs. a query** (3.3) — sharpens the
   Phase-6.2 experiment into a real comparison instead of a demo.
3. **Masked-future conditioning as the named alternative** (3.4) — gives
   Phase 6.1 a baseline to beat rather than a design asserted on its own.
4. **Channel-dropout infilling** (3.2) — interesting, but a different paper.

What I would *not* take: pixel-space diffusion, the HEALPix grid, and
distributional-only validation. All three are well matched to cBottle's problem
and poorly matched to ours, where a single true field exists and is scored
cell by cell.
