# Prior-Art Overlap & Claim-Boundary Table (Task 2)

*Prepared for the Monday meeting. Sources: original papers (arXiv IDs in the
table). Purpose: state exactly which parts of our architecture already exist,
which claims are therefore blocked, and what remains defensible.*

## Positioning matrix

| Method | Main task | Input type | Internal representation | Output type | Handles sparse inputs? | Main overlap with us | Claim it blocks | Role in our paper |
|---|---|---|---|---|---|---|---|---|
| **Perceiver** (ICML'21, 2103.03206) | Any-modality classification/regression | Arbitrary token arrays (image, audio, points) | Fixed-size learned latent via iterative cross-attention | Global label/vector | Yes (any token set) | Cross-attention token→fixed-latent bottleneck | "Fixed latent bottleneck over heterogeneous tokens" | Architecture ancestor |
| **Perceiver IO** (ICLR'22, 2107.14795) | General structured input→output | Arbitrary token arrays | Fixed latent + **output query** cross-attention | Values at arbitrary query positions | Yes | Latent + coordinate-conditioned query decoding — *our whole trunk* | "Query-decoded shared latent" as novelty | **Variant-A architecture control** (identical trunk, standard attention) |
| **GraphDOP** (ECMWF, 2412.15687) | Medium-range weather forecasting **obs→obs** | Raw heterogeneous satellite + conventional observations | Graph encoder → latent gridded Earth-system state; transformer processor rollout | Predicted observations (any instrument/place/time) | Yes, natively | Shared latent learned *directly from heterogeneous Earth-system obs*, no (re)analysis input | "First shared-latent from heterogeneous Earth-system observations"; "decode anywhere" | Closest Earth-system precedent |
| **ADAF-Ocean** (2511.06041) | Ocean (surface) state estimation / DA | 6 obs types: sparse in-situ + 4 km satellite swaths, **no thinning/interp** + background field | Neural-process-style continuous mapping | Continuously queried ocean analyses | Yes, natively | Native multi-source *ocean* obs → queryable analysis; coupled to DL forecaster | "First native heterogeneous-obs ocean reconstruction with coordinate queries" | Closest ocean-specific precedent |
| **Set Transformer** (ICML'19, 1810.00825) / **ANP** (ICLR'19) | Set-valued regression / stochastic processes | Unordered variable-count sets; (ANP) context points | ISAB: attention through **fixed inducing points**; (ANP) latent + deterministic path | Per-element or per-query values | Yes, by construction | Permutation-invariant variable-count set encoding; ISAB *is* fixed-budget resampling | "Variable-length set input"; "query-conditioned decoding" | **Variant-B control** (fixed-budget resampler = learned inducing points) |
| **Samudra** (2412.03795) | Global ocean *emulation* (prognostic rollout) | Dense gridded model state + forcings | ConvNeXt U-Net | Dense gridded future state | No (dense grids only) | Full-depth global ocean T/S with ML | None of ours (different task) — but blocks "first ML full-depth global ocean T/S" | Related work (dense forecasting) |
| **Samudra 2** (2606.02610) | Ocean emulation across resolutions (1°→1/4°) | Dense gridded state | Wider U-Net, ConvNeXt blocks, dynamic per-channel loss reweighting | Dense rollout, mesoscale eddies | No | Depth/channel reweighting is a *loss-side* answer to imbalance (ours is attention-side) | — | Related work |
| **FuXi-Ocean** (2506.03210) | Global ocean forecasting, 6-hourly, 1/12°, to 1500 m | Dense gridded state | Stacked attention + Mixture-of-Time module | Dense gridded forecast | No | Attention over ocean fields at depth | "Attention for 3-D ocean fields" | Related work |

Also monitored (from the draft's risk list): **Appa** (latent-diffusion arbitrary-obs
assimilation), **Earth-o1** (multimodal latent obs space), **ESFM** (missing
variables / multi-resolution / station data in one backbone), **CLOINet /
ReconMOST** (sparse-to-dense 3-D ocean recovery). These reinforce, not change,
the boundary below.

## We do NOT claim as new

- heterogeneous observation input (GraphDOP, ADAF-Ocean, ESFM)
- variable-length / set-valued input (Set Transformer, ANP, Perceiver)
- a fixed latent bottleneck (Perceiver, Set Transformer ISAB)
- coordinate-conditioned output / continuous queries (Perceiver IO, ADAF-Ocean, ANP)
- missing-modality handling (ESFM, GraphDOP in practice)
- ocean-state reconstruction from sparse obs (CLOINet, ReconMOST, ADAF-Ocean)
- continuous depth queries (ADAF-Ocean's continuous mapping; implicit-neural lit.)
- "first shared-latent / coordinate-query / heterogeneous-obs ocean model" — **all blocked**

## Candidate contribution (defensible if the experiments support it)

1. **Identifying tokenization multiplicity as a bias source** in attention-based
   obs fusion: refining a dense field into more tokens, duplicating
   measurements, or sampling a profile at more levels changes attention mass
   with *no new physical information*. None of the systems above isolates or
   measures this.
2. **Treating tokens as quadrature elements of an observation measure**
   (support mass + within-modality normalization + modality prior).
3. **Measure-Balanced Cross-Attention (MBCA)** — log-mass attention prior;
   reduces exactly to mass-weighted attention.
4. **Exact invariance to equivalent token partitioning** (proposition + passing
   unit test; standard attention provably lacks it — demonstrated).
5. **Direct duplication / retokenization / profile-resampling evaluation
   protocol** — the sensitivity numbers themselves are a contribution.
6. **Structured vertical-profile tokenization** with physical depth bands and
   per-band represented-span mass (level-count invariant by construction).

## Consequences for the draft

- Introduction must say: shared latent + query decoder are *adopted*, from
  Perceiver IO/GraphDOP lineage — cited, not claimed.
- Perceiver IO (Variant A) and Set-Transformer-style resampling (Variant B) are
  *controls inside our experiments*, which is the strongest possible way to
  handle this prior art: we don't argue around them, we ablate against them.
- ADAF-Ocean is the paper we must be most careful with: it already does native
  multi-source ocean obs → continuous queries (surface). Our deltas: full-depth
  3-D anomaly reconstruction on a controlled OSSE, matched-input baselines, and
  the measure-invariance property. Do not overstate beyond that.
- Samudra 2's dynamic channel reweighting should be cited as a *loss-side*
  treatment of imbalance to contrast with our *attention-side* treatment.
