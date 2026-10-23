# Experiments after the 2026-10-21 group meeting (Wang's feedback)

Date: 2026-10-21. Branch `ka-dev6` (from `ka-dev4` 51a2f25). Approved by Zihao ("run these", every run to
100,000 steps). Written before any run started.

## What Wang asked, and what answers it

| Wang's point | Experiment | Where it goes |
|---|---|---|
| Kriging assumes a Gaussian decay; replace the covariance by a net | three PSD-safe learned covariances (deep, product, multiscale) | ablation table |
| Loss not converged, more data, random profile sampling | every run at 100k steps; 154 training years; 3 draws a month; random input density per step | ablation table (scale, augmentation) |
| Is KA multivariate? | KA with a linear model of coregionalisation (T and S solved jointly) | ablation table |
| A common benchmark (GLORYS12) | GLORYS12 1/12-degree as a third OSSE simulator with a sealed test; GLORYS12 and ARMOR3D products as reference rows at the held-out real floats | supplement (simulator), main real table (reference rows, not baselines) |
| Real data is data-limited | real training years 2011-2020 instead of 2016-2020 | declared extra pass on 2024 |
| The Gaussian assumption, argued | non-Gaussian truths in the controlled GP benchmark | supplement |

## Protocol (unchanged)

* Every new variant is chosen on the selection years (CESM2 2004-2005, real 2021) by the runner's
  criterion (mean standardised RMSE over leads on the fixed 8,000-cell draw), three seeds.
* The three opened tests (CESM2 2006-2007, real 2024, iHESP 1998-1999) receive at most one declared
  extra pass per variant, as the CLOINet-style and 4DVarNet rows did.
* Any change of the main method (a variant replacing A, or A at 100k steps replacing A at 70k)
  needs the new sealed CESM2 2008-2010 period, built only when that decision is taken.
* The comparison basis for every 100k variant is `c_flow100` (A's settings at 100k steps), not the
  70k A of the paper; the paper keeps 70k unless A moves to 100k through the sealed v2 test.
* New code lives in `ka-dev6`; the hashed runner sources are frozen once the queue starts.
  The real-data training-year change touches hashed modules and goes to its own worktree (`ka-dev7`).

## Kernels (`kriging_attention.py`, new modes)

All start as `ka_stationary` (zero-initialised heads) except `ka_multiscale`, whose second component
starts at amplitude e^-1.5; all are positive definite by construction:

* `ka_deep`: Gaussian kernel in a learned space, `C_ab = s_a s_b exp(-1/2 ||phi_a - phi_b||^2)` with
  `phi_a = [Delta_a / ell, psi_a]`, `psi` four learned coordinates from a head on the hidden vectors
  (bounded by 3 tanh). No shape matrices: the net replaces the Gaussian's geometry.
* `ka_product`: the flow kernel times the same feature Gaussian (Schur product).
* `ka_multiscale`: `s1_a s1_b PS(Sigma) + s2_a s2_b PS(9 Sigma)`, two length scales with per-point
  amplitudes (a sum of diagonally scaled PSD kernels).
* `--multivariate`: `Cov[(a,u),(b,v)] = B1_uv C^T_ab + B2_uv C^S_ab`, `B1, B2` 2x2 PSD per level
  (Cholesky factors), initialised to `diag(1,0)` and `diag(0,1)` so the start equals the per-variable
  model; one 64x64 solve per query.

## Augmentation and scale (runner options, cohorts)

* `--thin-inputs MIN MAX`: each training step keeps `n ~ U[MIN, MAX]` of the month's input profiles
  (state and neighbours alike); evaluation reads every input.
* `cesm2_long_aug3`: the 1964-1999 months with three draws of 7,600 profiles each (87's recipe,
  generators `[SEED, month + 10000 + 100000 d]`); 2000-2005 unchanged, so selection is unchanged.
* `cesm2_vlong_uniform`: 87's recipe from 1850-01 (raw member, `53` extended), 2000-2005 unchanged;
  the 2008+ months of the raw member are never read (reserved for the sealed v2 test).

## Arms (CESM2, 100k steps, leads 0/1/3/6, selection 2004-2005, seeds 1234/1235/1236)

| arm | options |
|---|---|
| c_flow100 | A's settings |
| c_deep, c_product, c_multiscale | `--kriging ka_deep` / `ka_product` / `ka_multiscale` |
| c_mv | `--kriging ka_flow --multivariate` |
| c_density | `--thin-inputs 1520 6080` |
| c_draws | cohort `cesm2_long_aug3`, `--thin-inputs 6080 6080` |
| c_both | cohort `cesm2_long_aug3`, `--thin-inputs 1520 6080` |
| c_y154 | cohort `cesm2_vlong_uniform`, `--train-years 154` |

Real (100k steps, selection 2021): `r_y10` training 2011-2020 (`ka-dev7`).

GLORYS12 OSSE: spec of its own once the download recipe is fixed (`2026-10-2x-glorys-osse.md`).
