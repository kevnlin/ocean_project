# Replacing the Perceiver-IO backbone — the PhCA-style (LNO) fuse, and the result

*Kevin's action from the 2026-09-17 meeting, run on the global reconstruction
(every profile, satellite-era split, 2 seeds). Implementation:
`src/ocean_tokenizer/fusion.py::LNOFusion`; arms in
`reports/real_data/ablation_ladder.md` §1.*

## What was swapped, and what was kept

The Perceiver-IO trunk's fuse stage — 32 unaddressed resampler slots and 32 free
latent vectors — is replaced by a **PhCA-style** fuse after the Latent Neural
Operator (Wang & Wang, NeurIPS 2024), with a Set-Transformer / Slot-Attention
framing:

1. **Position-only weights.** Each observation token's logits over 32 latent
   slots come from an MLP on its coordinates alone; the measured value enters
   only as the value vector. Observation positions and query coordinates are
   decoupled.
2. **Softmax over the slots** (Slot Attention / Transolver normalisation): each
   token spreads exactly its DFS evidence τ across the slots, so slot mass sums
   to the total evidence — DFS's conservation property holds by construction.
   *This is the one deliberate departure from LNO*, whose PhCA encoder
   normalises over the input points (its Eq. 4); hence "PhCA-style".
3. **Per-slot competition** against the null key and the reference slots at
   `log λ_bg`, then the unchanged latent self-attention blocks.

Kept identical: the D4RT query decoder (query cross-attention, local refiner,
T/S heads), DFS evidence, data, split, seeds, 12 k steps. Size-matched:
**405,543 vs 405,063** parameters at 32 slots.

## Result (held-out 2022-23, TEMP z; J against climatology)

| arm | seed 1234 | seed 1235 | Δ vs its reference where neither collapsed |
|---|---:|---:|---:|
| Perceiver-IO + DFS (baseline) | 0.9264 | 1.0172 *collapsed* | — |
| Perceiver-IO + uniform mass | 0.9262 | 0.9584 | −0.0002 vs baseline (seed 1234 only) |
| **PhCA-style + DFS** | **0.9251** | 1.0365 *collapsed* | **−0.0013** vs baseline (seed 1234 only) |
| PhCA-style + uniform mass | 0.9756 | 0.9988 | +0.0505 vs PhCA + DFS (seed 1234 only) |

Climatology on the same held-out cells: 1.0638 z.

Reading it honestly:

* **The PhCA-style fuse ties the Perceiver fuse** and collapses on the same seed;
  swapping the fuse does not remove the collapse.
* **DFS beats uniform inside the PhCA-style fuse by 0.05 z** on the seed where
  the DFS run did not collapse, while the same contrast is worth nothing in the
  Perceiver fuse (−0.0002 z) — the one sign that slot competition gives the
  evidence estimate something to act on. It is a single seed and must be re-run
  with the refiner fixed before it is claimed.
* **No speed or memory gain here**: 4.1 h vs 3.9 h per run. The LNO paper's
  efficiency claims (30 % fewer parameters, 50 % less memory, 1.8× faster than
  Transolver) are about the attention stage; our wall time is the DFS neighbour
  search over ~30 k tokens.

## Recommendation

Adopt the PhCA-style fuse only if a *stable* comparison (refiner fixed, ≥ 3
seeds) shows it ahead; today it is a tie. The two changes with evidence behind
them are upstream of the fuse: make the refiner local, and give the model
information the profiles do not carry.
