# Phase 4 — the pseudo-SSH ("satellite altimeter") modality

*Status 2026-08-08: **modality built, characterised, and the ablation run —
both pre-registered hypotheses confirmed** (§5). §3 states the hypothesis as it
was written before any model was trained, and has not been edited since.*

The advisor's tokenization list is "argo profile, satellite altimeter,
satellite images (2 dimension)… take sst ssh". The pipeline had SST and SSS but
**no SSH at all**: the standardized stores carry only TEMP/SALT/SST/SSS. This
phase adds one.

---

## 1. What was built, and why it is a *pseudo* SSH

Route B (download true CESM2-LE SSH from NCAR and reproduce the original 1°
regrid exactly) is on the backlog — it needs the regrid recipe, which nobody has
written down. Route A, implemented here, derives the **steric / baroclinic**
component of sea-surface height from the T/S fields already in the store, via
TEOS-10 dynamic height ([src/ocean_tokenizer/ssh.py](../src/ocean_tokenizer/ssh.py)):

```
p   = p_from_z(-z, lat)                        # dbar
SA  = SA_from_SP(SP, p, lon, lat)              # absolute salinity
CT  = CT_from_t(SA, t, p)                      # conservative temperature
Phi = geo_strf_dyn_height(SA, CT, p, p_ref)    # m^2/s^2
eta = Phi[surface] / g(lat)                    # metres
```

`p_ref = 990 dbar` sits below the deepest analysis level at every latitude
(985 m → 992.8 dbar at the equator, 998.1 dbar at the pole), so one fixed
reference is valid globally. Columns that do not reach it are NaN, exactly as
in real dynamic-height maps.

**Three limitations that must travel with every number derived from this field:**

1. **Baroclinic only.** No barotropic (bottom-pressure) term, no mass or
   freshwater term. Not comparable to an altimeter product in absolute terms.
2. **Derived, not independent.** It is computed *from* the same TEMP/SALT the
   model is asked to reconstruct. Inside the OSSE that is internally
   consistent, but it means this experiment tests *"does a vertically
   integrated surface constraint help?"* — **not** *"does real altimetry
   help?"*, which is a Phase-5 question against CMEMS L4 ADT.
3. **1° monthly.** The mesoscale eddy signal that makes altimetry valuable
   operationally is largely averaged away at this resolution.

## 2. The field, characterised

360 monthly (180, 360) fields covering 1985–2014 (every protocol_v1 split),
`outputs/cache/ssh_dyn.npz` (47 MB, gitignored — regenerate with
`experiments/28_make_ssh.py`, 13 min CPU). Diagnostics from
[ssh_dyn_meta.json](../outputs/cache/ssh_dyn_meta.json):

| diagnostic | value | reading |
|---|---:|---|
| coverage (fraction of ocean cells) | **0.867** | shelves and marginal seas do not reach 990 dbar; 13 % of the ocean has no pseudo-SSH |
| mean steric height | **1.395 m** | physically plausible relative to ~1000 dbar |
| **SSH anomaly std** | **2.89 cm** | realistic for steric height at 1° monthly |
| corr(SSH anomaly, **SST** anomaly) | **0.470** | |
| corr(SSH anomaly, **100–300 m T** anomaly) | **0.708** | |

**The last two rows are the reason this modality is worth a channel.** The
pseudo-SSH correlates substantially *more* with 100–300 m temperature than with
SST. If SSH were merely a restatement of the surface temperature the U-Net
already receives, the ordering would be reversed and the channel could only add
noise. Instead the field behaves as a **thermocline proxy**, which is precisely
the information the reconstruction is weakest at.

Note this is a property of the *modality alone* — no model is involved, so it
does not peek at the outcome of §3.

## 3. Pre-registered hypothesis

> **Written 2026-08-08, before the ablation was run.**
>
> **H1.** Adding the SSH channel improves full-column TEMP RMSE.
>
> **H2 (the sharp one).** The largest gain is in the **100–300 m band**,
> because sea-surface height integrates the density structure of the whole
> column and therefore encodes thermocline displacement. This band is currently
> the depthwise U-Net's weakest layer (0.1916 °C, vs 0.1589 at 0–100 m and
> 0.0837 at 300–max, `audit_depthwise_e40`), so it has the most room to move.
> The correlation structure in §2 (0.708 vs 0.470) is independent support for
> the mechanism.
>
> **Pre-registered null.** No significant change. Leading suspect would be that
> at 1° monthly the eddy signal is averaged away, leaving SSH close to a linear
> function of upper-ocean heat content the model can already infer from SST plus
> the WOA prior. **A negative result is a result** and will be reported as one,
> with this paragraph unedited.

## 4. Experimental design

[experiments/29_ssh_ablation.py](../experiments/29_ssh_ablation.py). Arms
identical in **every** respect except the cfg:

| arm | cfg | c_in | role |
|---|---|---|---|
| control | `profiles_woa_surf` | 10 | the established baseline |
| treatment | `profiles_woa_surf_ssh` | 12 | + SSH value and finite flag |
| `profiles_only` | `profiles` | 6 | not part of this ablation — the M1 like-for-like row, see [oi_baseline.md](oi_baseline.md) |

Same seed, same profile draws, same epochs, same validation-selected checkpoint
rule, same 12 pinned test months. The control is retrained inside this script
rather than reusing `audit_depthwise_e40`, so both arms share one training loop.

SSH enters as **two** channels — z-scored value plus a finite flag — following
the repo's missing-data convention (value → 0, flag → 0), so the 13 % of ocean
without a defined steric height is marked missing rather than asserted to be
zero anomaly. Its climatology and z-score statistics come from the **276
training months only** (`ssh.SSHAnom`), consistent with everything else in
protocol_v1.

The `ssh` cfg token is strictly additive:
[tests/test_unet_channels_ssh.py](../tests/test_unet_channels_ssh.py) pins that
every pre-existing config is **bit-identical** with the SSH code present, so the
certified checkpoints keep their c_in = 10 and every historical number stays
reproducible.

**The command actually used.** Every card on this box is held by other people's
jobs (~73 GB of 80 GB each), so the run had to fit in the remainder. The
training stack is ~17 GB at c_in = 12, which does not; `--cpu-tensors` keeps it
in host memory and ships one batch at a time, `--fwd-batch 16` caps the
inference peak (batch 64 reserves ~6.9 GB, batch 16 ~1.7 GB), and
`--mem-cap-gb` is a hard ceiling so a runaway allocation fails *this* process
rather than OOM-killing a co-tenant. Measured peak: **3.9 GB**.

```bash
CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python experiments/29_ssh_ablation.py --ssh-cache outputs/cache/ssh_dyn.npz \
  --cpu-tensors --fwd-batch 16 --mem-cap-gb 5.5 \
  --arms control_pws,treat_pws_ssh,profiles_only
```

(`expandable_segments` matters here: it brings reserved memory down from
4.83 GB to 3.62 GB by making it track allocated memory instead of over-reserving.)

## 5. Results

*Run 2026-08-08 on GPU 4 (shared card, `--cpu-tensors --fwd-batch 16
--mem-cap-gb 5.5`, 3.9 GB peak), 67.5 min for three arms. Seed 1234,
`outputs/cache/ssh_ablation.json`. **1 seed — not yet a 3-seed headline.***

TEMP, unobserved-only anomaly RMSE (°C), 12 pinned test months:

| | full column | 0–100 m | 100–300 m | 300–max | SALT full |
|---|---|---|---|---|---|
| control `profiles_woa_surf` | 0.1586 | 0.1582 | 0.1937 | 0.0838 | 0.0331 |
| treatment `profiles_woa_surf_ssh` | **0.1366** | **0.1401** | **0.1631** | **0.0718** | **0.0316** |
| **delta** (negative = SSH helps) | **−0.0220** | −0.0182 | **−0.0306** | −0.0120 | −0.0015 |
| **relative gain** | **+13.9 %** | +11.5 % | **+15.8 %** | +14.3 % | +4.5 % |

The control arm reproduces the certified checkpoint closely (0.1586 vs 0.1580
for `audit_depthwise_e40`), so the comparison is anchored to the established
baseline rather than to an idiosyncratic re-training.

### Verdict against §3 — both hypotheses confirmed

**H1 — confirmed.** Full-column TEMP improves by **13.9 %** (0.1586 → 0.1366).
For scale, that single channel buys about as much as **doubling the profile
count**: the density curve needs ~1500 → ~2800 profiles/month to achieve the
same drop. SALT improves too, by a smaller 4.5 %, which is consistent with
steric height being dominated by the thermal term.

**H2 — confirmed, on both readings.** The 100–300 m band gains the most in
absolute terms (−0.0306 °C, vs −0.0182 at 0–100 m and −0.0120 at 300–max) *and*
in relative terms (+15.8 %, vs +11.5 % and +14.3 %). This is the pre-registered
prediction and the mechanism behind it: SSH integrates the density structure of
the whole column, so it carries thermocline-displacement information that SST
and SSS — which see only the surface — cannot.

Two independent pieces of evidence now point the same way: the modality's own
correlation structure (§2: 0.708 with 100–300 m temperature vs 0.470 with SST,
computed before any model was trained) and the ablation's band ordering.

**A caveat that the numbers cannot settle.** §1.2 stands: the pseudo-SSH is
computed *from* the TEMP/SALT the model is asked to reconstruct. A 13.9 % gain
from a derived field is an upper bound on what a real altimeter would give,
because real ADT carries a barotropic component we omit, plus measurement and
representativeness error we do not simulate. The honest claim is **"a vertically
integrated surface constraint helps, and helps most in the thermocline"** — not
"altimetry buys 14 %". Phase 5 against CMEMS L4 ADT is what would test the
latter.

## 6. Next

* **3 seeds** (1235/1236) before this becomes a headline row.
* Add the channel to the shared-latent model as a third `GridPatchEncoder`
  stream (+540 tokens/month) and re-test there — on this evidence the fusion
  core is where a thermocline constraint should pay off most.
* The gain is large enough to change the Phase-2 story: at 1500 profiles the
  SSH-equipped U-Net (0.1366) already beats what the profile-only system needs
  ~2800 profiles to reach. Worth re-running the density curve *with* SSH to see
  whether the 0.1 °C crossing moves left of 4000.
* Route B (true CESM2-LE SSH) stays on the backlog and is now clearly worth it —
  it is the only way to separate "vertically integrated constraint" from
  "derived from the target". Ask how the original 1° regrid was done first.
