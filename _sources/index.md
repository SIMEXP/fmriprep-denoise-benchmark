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
Current work aims to introduce an application programming interface (API) implemented in
[`nilearn`](https://nilearn.github.io/stable/modules/generated/nilearn.interfaces.fmriprep.load_confounds.html#nilearn.interfaces.fmriprep.load_confounds) {cite:p}`nilearn`
to standardise fMRIPrep confounds retrieval from the confounds outputs,
and provide benchmarks of different strategies using functional connectivity generated from resting state data.
We compared the connectomes generated from a set of different atlases and compare 4 classes of denoising strategies against the high-pass filtered data:
head motion and tissue signal (`simple`), scrubbing (`scrubbing`), CompCor (`compcor`), and ICA-AROMA (`ica_aroma`).
Furthermore, the benchmarks were performed on two datasets openly available on OpenNeuro, covering different age group and psychiartic conditions: 
`ds000228` (child and adult), and 
`ds000030` (healthy control, ADHD, bipolar, and schizophrania). 
We investigated the loss of temporal degrees of freedom of each strategy, and three functional connectivity based metrics:
quality control / functional connectivity (QC-FC),
distance-dependent effects of motion on connectivity, and
denoised outcome on network modularity. 
After exluding subject with excessive motion, all strategies can reduce the correlation of motion and functional connectivity measure,
except ICA-AROMA combined with global signal regression. 
Scrubbing based strategy is the best at eliminating distance depenency of motion.
For network modularity, strategies including global signal regression shows more detectable networks. 
Between different age groups, the number of ICA-AROMA and/or CompCor components varies,
and will result in inconsistent loss in degrees of freedom and statistical power in the study.
A simple approach containing only head motion parameters, white matter and cerebrospinal fluid signal might provides consistency regardless of the sample. 
In conclusion, with the maturity of fMRIPrep, the common denoising strategies are achieving the goal,
and the choice of strategies are down to the researchers need. 
There is no definitive best option for denoising but methods that fits for different purpose.

**{fa}`key` Keywords:** 
`reproducibility`, 
`fMRIPrep`, 
`nilearn`, 
`nuisance regressor`, 
`resting-state fMRI`, 
`functional connectivity`