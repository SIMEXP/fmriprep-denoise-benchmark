# Benchmark denoising strategies on fMRIPrep 

The project is a continuation of [load_confounds](https://github.com/SIMEXP/load_confounds).
The aim is to evaluate the impact of denoising strategy on functional connectivity data, using output processed by fMRIPrep LTS.

## Dataset structure

- All inputs (i.e. building blocks from other sources) are located in
  `inputs/`.
  Processed data for development can be found here:

  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5764254.svg)](https://doi.org/10.5281/zenodo.5764254)


- Custom code is located in `fmriprep_denoise/`. Installable through `pip install .`
- Preprocessing scripts are in `preprocess/`