# Phase 4 — the pseudo-SSH ("satellite altimeter") modality

*Status 2026-08-08: **modality built and characterised; the U-Net ablation is
written and smoke-verified but not yet run** — every GPU on the box is at
73-75 GB / 80 GB. §3 states the hypothesis before any result exists; §4 gives
the exact launch command.*

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

## 4. Experimental design (written, smoke-verified, awaiting a GPU)

[experiments/29_ssh_ablation.py](../experiments/29_ssh_ablation.py). Two arms,
identical in **every** respect except the cfg:

| arm | cfg | c_in |
|---|---|---|
| control | `profiles_woa_surf` | 10 |
| treatment | `profiles_woa_surf_ssh` | 12 |

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

**Launch when a card frees up** (`nvidia-smi` first):

```bash
CUDA_VISIBLE_DEVICES=N python experiments/29_ssh_ablation.py
# or, on a partially-occupied card (~4 GB instead of ~17 GB):
CUDA_VISIBLE_DEVICES=N python experiments/29_ssh_ablation.py --cpu-tensors
```

Then paste the resulting table into §5 and answer H1/H2 against §3 **without
editing §3**.

## 5. Results

*Pending — no GPU was free on 2026-08-08. `outputs/cache/ssh_ablation.json` will
carry the numbers; the script prints the control/treatment delta per band
directly.*

| | TEMP full | 0–100 m | 100–300 m | 300–max | SALT full |
|---|---|---|---|---|---|
| control `profiles_woa_surf` | — | — | — | — | — |
| treatment `profiles_woa_surf_ssh` | — | — | — | — | — |
| **delta** (negative = SSH helps) | — | — | — | — | — |

## 6. Next

* Run the ablation (above), 3 seeds for the headline.
* If H2 holds, the same channel should go into the shared-latent model as a
  third `GridPatchEncoder` stream (540 extra tokens/month) and be re-tested
  there — the fusion core is where a thermocline constraint should pay off most.
* Route B (true CESM2-LE SSH) stays on the backlog. Ask how the original 1°
  regrid was done before attempting it; until then no claim about *real*
  altimetry may be made from this field.
