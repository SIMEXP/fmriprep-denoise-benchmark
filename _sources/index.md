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

Various fMRI denoising benchmarks have been published in the past, 
however the impact of the denoising strategies has yet to be evaluated on the popular minimal preprocessing tool fMRIPrep.
The confound outputs of fMRIPrep presented the users with a range of possible nuisance regressors.While users benefit from a wide selection of regressors,
without understanding the literature one can pick a combination of regressors that reintroduce noise to the signal.
Current work aims to introduce an application programming interface (API) implemented in
[`nilearn`](https://nilearn.github.io/stable/modules/generated/nilearn.interfaces.fmriprep.load_confounds.html#nilearn.interfaces.fmriprep.load_confounds)
to standardise fMRIPrep confounds retrieval from the confounds outputs,
and provide benchmarks of different strategies using functional connectivity generated from resting state data.
We compared the connectomes generated from a set of different atlases and compare 4 classes of denoising strategies against the high-pass filtered data: 
head motion and tissue signal (`simple`), scrubbing (`scrubbing`), CompCor (`compcor`), and ICA-AROMA (`ica_aroma`).
The benchmarks were performed on two datasets openly available on OpenNeuro, covering different age group and psychiatric conditions: 
`ds000228` (child and adult), and 
`ds000030` (healthy control, ADHD, bipolar, and schizophrenia). 
We investigated the loss of temporal degrees of freedom of each strategy, and three functional connectivity based metrics:
quality control / functional connectivity (QC-FC),
distance-dependent effects of motion on connectivity, and
denoised outcome on network modularity. 
After excluding subjects with excessive motion, 
all strategies can reduce the correlation of motion and functional connectivity measure, 
except ICA-AROMA combined with global signal regression.
Scrubbing based strategy is the best at eliminating distance dependency of motion.
For network modularity, strategies including global signal regression show more detectable networks.
Between different age groups, the number of ICA-AROMA and/or CompCor components varies, 
and will result in inconsistent loss in degrees of freedom and statistical power in the study.
A simple approach containing only head motion parameters, white matter and cerebrospinal fluid signal might provide consistency regardless of the sample.
In conclusion, with the maturity of fMRIPrep, the common denoising strategies are achieving the goal, 
and the choice of strategies are down to the researchers' needs. 
There is no definitive best option for denoising but methods that fit for different purposes.

**{fa}`key` Keywords:** 
`reproducibility`, 
`fMRIPrep`, 
`nilearn`, 
`nuisance regressor`, 
`resting-state fMRI`, 
`functional connectivity`