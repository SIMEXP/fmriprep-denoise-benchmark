# Benchmark denoising strategies on fMRIPrep output

[![DOI](https://neurolibre.org/papers/10.55458/neurolibre.00012/status.svg)](https://doi.org/10.55458/neurolibre.00012)

The project is a continuation of [load_confounds](https://github.com/SIMEXP/load_confounds).
The aim is to evaluate the impact of the denoising strategy on functional connectivity data, using output processed by fMRIPrep LTS in a reproducible workflow.

**The manuscript is now published in *PLOS Computational Biology* doi: [10.1371/journal.pcbi.1011942](http://dx.doi.org/10.1371/journal.pcbi.1011942).**

**The reproducible Jupyter Book preprint is on [NeuroLibre](https://neurolibre.org/papers/10.55458/neurolibre.00012).**

## Recommendations for those who thought this project is a software project

Bad news, this is not a software but a research project. 
It's more similar to your regular data science project. 
In other words, the code in this repository reflects the research done for the manuscript and is not suitable for a production-level application.

Some useful parts of the code have been extracted and further reviewed within the SIMEXP lab for deployment as Docker images for generic fMRIprep derivatives.

 - *time series and connectome workflow*: [`giga_connectome`](https://github.com/SIMEXP/giga_connectome).
 - *motion quality control metrics*: [`giga_auto_qc`](https://github.com/SIMEXP/giga_auto_qc).

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
  You can bypass this step by accessing the Neurolibre preprint [![DOI](https://neurolibre.org/papers/10.55458/neurolibre.00012/status.svg)](https://doi.org/10.55458/neurolibre.00012)!

- Custom code is located in `fmriprep_denoise/`. This project is installable.

- Preprocessing SLURM scripts, and scripts for creating a figure for the manuscript are in `scripts/`. 
