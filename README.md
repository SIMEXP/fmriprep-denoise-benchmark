# Benchmark denoising strategies on fMRIPrep output

[![DOI](https://neurolibre.org/papers/10.55458/neurolibre.00012/status.svg)](https://doi.org/10.55458/neurolibre.00012)

The project is a continuation of [load_confounds](https://github.com/SIMEXP/load_confounds).
The aim is to evaluate the impact of denoising strategy on functional connectivity data, using output processed by fMRIPrep LTS in a reproducible workflow.

**Preprint of the manuscript is now on [biorxiv](https://www.biorxiv.org/content/10.1101/2023.04.18.537240).
The reporducible Jupyter Book preprint is on [NeuroLibre](https://neurolibre.org/papers/10.55458/neurolibre.00012).**

## Quick start

```bash
git clone --recurse-submodules https://github.com/SIMEXP/fmriprep-denoise-benchmark.git
cd fmriprep-denoise-benchmark
virtualenv env
source env/bin/activate
pip install -r binder/requirements.txt
pip install .
make data
make book
```

## Dataset structure

- `binder/` contains files to configure for neurolibre and/or binder hub.

- `content/` is the source of the JupyterBook.

- `data/` is reserved to store data for running analysis.
  To build the book, one will need all the metrics from the study.
  The metrics are here:
  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7764979.svg)](https://doi.org/10.5281/zenodo.7764979)
  The data will be automatically downloaded to `content/notebooks/data`.
  You can by pass this step through accessing the Neurolibre preprint [![DOI](https://neurolibre.org/papers/10.55458/neurolibre.00012/status.svg)](https://doi.org/10.55458/neurolibre.00012)!

- Custom code is located in `fmriprep_denoise/`. This project is installable.

- Preprocessing SLURM scripts, and scripts for creating figure for manuscript are in `scripts/`. 


## Poster and presentations

The results will be presented at QBIN science day 2023 as a flash talk, and OHBM 2023 Montreal as a poster :tada:. 

The preliminary results were presented at OHBM 2022 as a poster.

![spoiler](./content/images/ohbm2022_abstract_head.png)
