# Baselines for sparse-profile ocean reconstruction, with code

Survey of 2026-10-06 for the multi-modal comparison: methods that reconstruct
subsurface temperature and salinity from in-situ profiles, satellite fields, or
both, and whether code exists. "Status here" says what this repository has.

| Method | Reference | Code | Inputs | Status here |
|---|---|---|---|---|
| Optimal interpolation | Bretherton, Davis and Fandry 1976 | `src/ocean_tokenizer/oi.py` | profiles | run on the synthetic cohort (`44_synth_argo_oi.py`) |
| Pointwise MLP | this repository's gridded line | `src/ocean_tokenizer/baselines.py` | profiles | run on the synthetic cohort (`45_synth_argo_mlp.py`) |
| 4DVarNet | Fablet et al. 2021 | https://github.com/CIA-Oceanix/4dvarnet-starter (CeCILL-C, last commit 2024-12); the older https://github.com/CIA-Oceanix/4dvarnet-core (last commit 2023-05, PyTorch 1.11) | gridded observations with gaps, optional second field | run on the synthetic cohort (`47_synth_argo_4dvarnet.py`) |
| ARMOR3D-style | Guinehut et al. 2012, https://os.copernicus.org/articles/8/845/2012/ | none public | regression of T/S on altimetry and SST, then OI with profiles | not built; both pieces exist here |
| ConvNP | DeepSensor, https://github.com/alan-turing-institute/deepsensor | pip package `deepsensor` | off-grid and gridded inputs, any target point | the repo's SetConv U-Net backbone is this family; not run on the synthetic cohort with fields |
| OSnet | Pauthenet et al. 2022, https://os.copernicus.org/articles/18/1221/2022/ | https://github.com/euroargodev/OSnet-GulfStream | satellite fields to T/S profiles | re-implemented without sea level for the gridded line (`src/baselines/osnet_mlp.py`) |
| NeSPReSO | Miranda et al., https://data.coaps.fsu.edu/eric_pub/papers_html/Miranda_et_al_24.pdf | no public repository found; a web API | satellite fields to PCA scores of profiles | re-implemented without sea level for the gridded line (`src/baselines/nesperso_pcamlp.py`) |
| Stacked LSTM | Buongiorno Nardelli 2020 | not checked | satellite fields to profiles | re-implemented without sea level for the gridded line (`src/baselines/nardelli_lstm.py`) |
| Senseiver | Santos et al. 2023 | public | sparse sensors to a field | reproduced on real data (`48_senseiver.py`) |

Related, not methods to run:

- OceanDepths (Donike et al., August 2026, https://arxiv.org/abs/2608.16373):
  a global dataset of paired surface and subsurface observations with a
  reconstruction baseline task, at https://huggingface.co/datasets/ESA-philab/OceanDepths.
- TS-Cast (Ocean Science 2026, https://os.copernicus.org/articles/22/2161/2026/):
  satellite fields to subsurface T/S in the northwestern Pacific; code
  availability not confirmed.

## Notes on 4DVarNet

- `4dvarnet-core` is the original research code: PyTorch 1.11, Lightning 1.6,
  Python 3.9, configurations tied to its authors' clusters. `4dvarnet-starter`
  is the same group's maintained, smaller version (PyTorch 2, about 1,000
  lines) and is what this repository uses.
- Both are CeCILL-C. The clone lives unmodified in the git-ignored `external/`
  folder; our driver imports its solver, prior and gradient model.
- It interpolates gridded fields, so profiles are binned to the 1° grid and its
  answer is sampled back at the query profiles.
- It is normally trained on a complete field with a gradient loss. On the
  synthetic cohort it is trained on Argo profiles only, like every other
  learned method there.
