#**Update Feburary 2025**
This project is a continuation of **Benchmark denoising strategies on fMRIPrep output**. 
The aim is to adapt the original denoising workflow to utilize HALFpipe outputs.

<img width="561" alt="Screenshot 2025-04-30 at 10 30 14 PM" src="https://github.com/user-attachments/assets/5a848bf7-c2ee-4395-ae5d-c13ef543c5e5" />



# Benchmark denoising strategies on fMRIPrep output

[![DOI](https://neurolibre.org/papers/10.55458/neurolibre.00012/status.svg)](https://doi.org/10.55458/neurolibre.00012)

The project is a continuation of [load_confounds](https://github.com/SIMEXP/load_confounds).
The aim is to evaluate the impact of denoising strategy on functional connectivity data, using output processed by fMRIPrep LTS in a reproducible workflow.

**Preprint of the manuscript is now on [biorxiv](https://www.biorxiv.org/content/10.1101/2023.04.18.537240).
The reporducible Jupyter Book preprint is on [NeuroLibre](https://neurolibre.org/papers/10.55458/neurolibre.00012).**

## Recommandations for those who thought this project is a software

Bad news, this is not a software but a research project. 
It's more similar to your regular data science project. 
In other words, the code in this repository reflects the research done for the manuscript, and is not suitable for production level application.

Some useful part of the code has been extracted and further reviewed within SIMEXP lab for deplyment on generic fmriprep derivatives as docker images.

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
  You can by pass this step through accessing the Neurolibre preprint [![DOI](https://neurolibre.org/papers/10.55458/neurolibre.00012/status.svg)](https://doi.org/10.55458/neurolibre.00012)!

- Custom code is located in `fmriprep_denoise/`. This project is installable.

- Preprocessing SLURM scripts, and scripts for creating figure for manuscript are in `scripts/`. 
