# Benchmark denoising strategies on fMRIPrep 

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://simexp.github.io/fmriprep-denoise-benchmark/)

The project is a continuation of [load_confounds](https://github.com/SIMEXP/load_confounds).
The aim is to evaluate the impact of denoising strategy on functional connectivity data, using output processed by fMRIPrep LTS.

The preliminary results will be presented at OHBM 2022 as a poster. 
Please find poster number `WTh570`.

Find the presenter at the 
[virtual poster session](https://event.fourwaves.com/ohbm-2022/abstracts/d49d130b-7f83-4c87-92f4-e1a8e319502b)
on __Wednesday, June 8, 2022, 8:30 PM GMT + 1__.

At Glasgow, please contact the presenter on [Twitter](https://twitter.com/HaoTingW713) to schedule a time to chat,
or come to see the presenter on __Wednesday, June 22, 2022, 12:45 PM__ at the poster hall.

![spoiler](./content/images/ohbm2022_abstract_head.png)

## Dataset structure

- All inputs (i.e. building blocks from other sources) are located in
  `inputs/`.
  To build the book, one will need all the metrics from the study.
  The metrics are here:
  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5764254.svg)](https://doi.org/10.5281/zenodo.5764254)

- Custom code is located in `fmriprep_denoise/`. Installable through `pip install .`
- Preprocessing SLURM scripts are in `script/`
