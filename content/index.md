---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Abstract

Various fMRI denoising benchmark has been published in the past,
however the impact of the denoising strategies has yet to be evaluated on the popular minimal preprocessing tool fMRIPrep {cite:p}`fmriprep1`.
The confound output of fMRIPrep presented the users with a range of possible nusianse regressors.
While users are benifted from a wide selection of regressors,
without understanding of the literature one can pick a combination of regressors that reintroduce noise to the signal.
Current work aims to introduce an application programming interface (API) to standardise fMRIPrep confounds retrieval,
and provide benchmarks of different strategies using functional connectivity generated from resting state data.
The main tool is a part of
[`nilearn`](https://nilearn.github.io/stable/modules/generated/nilearn.interfaces.fmriprep.load_confounds.html#nilearn.interfaces.fmriprep.load_confounds) {cite:p}`nilearn`.

Keywords: fMRIPrep, nilearn, nuisance regressor, resting-state fMRI, functional connectivity, benchmark 