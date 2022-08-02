---
title: "Benchmark denoising strategies on fMRIPrep outputs"
tags:
  - fMRIPrep
  - denoising
  - fMRI
authors:
  - name: Hao-Ting Wang
    affiliation: "1, 7"
    orcid: 0000-0003-4078-2038
  - name: Steven L Meisler
    affiliation: "2, 3"
    orchid: 0000-0002-8888-1572
  - name: Hanad Sharmarke
    affiliation: 1
    orchid:
  - name: François Paugam
    affiliation: 4
    orchid:
  - name: Nicolas Gensollen
    affiliation: 5
    orchid: 0000-0001-7199-9753
  - name: Bertrand Thirion
    affiliation: 5
    orchid: 0000-0001-5018-7895
  - name: Christopher J Markiewicz
    affiliation: 6
    orchid: 0000-0002-6533-164X
  - name: Pierre Bellec
    affiliation: "1, 7"
    orchid: 0000-0002-9111-0699
affiliations:
- name: Centre de recherche de l'Institut universitaire de gériatrie de Montréal (CRIUGM), Montréal, Québec, Canada
  index: 1
- name: Harvard University, MA, USA
  index: 2
- name: Massachusetts Institute of Technology, MA, USA
  index: 3
- name: Computer Science and Operations Research Department, Université de Montréal, Montréal, Québec, Canada
  index: 4
- name: Inria, CEA, Université Paris-Saclay, Paris, France
  index: 5
- name: Department of Psychology, Stanford University, Stanford, United States
  index: 6
- name: Psychology Department, Université de Montréal, Montréal, Québec, Canada
  index: 7
date: 05 October 2021
bibliography: paper.bib
---

# Summary

Various fMRI denoising benchmark has been published in the past, 
however the impact of the denoising strategies has yet to be evaluated on the popular minimal preprocessing tool fMRIPrep [@fmriprep1].
The confound output of fMRIPrep presented the users with a range of possible nusianse regressors.
While users are benifted from a wide selection of regressors, 
without understanding of the literature one can pick a combination of regressors that reintroduce noise to the signal.
Current work aims to introduce an application programming interface (API) to standardise fMRIPrep confounds retrieval,
and provide benchmarks of different strategies using functional connectivity generated from resting state data.
The main tool is a part of 
[`nilearn`](https://nilearn.github.io/stable/modules/generated/nilearn.interfaces.fmriprep.load_confounds.html#nilearn.interfaces.fmriprep.load_confounds) [@nilearn].

![Overview of API.\label{top_level_fig}](./content/images/api_summary.png)

# Acknowledgements

The initial API was started by Hanad Sharmarke and Pierre Bellec.
The implementation was completed by Hao-Ting Wang, Steven Meisler, François Paugam, and Pierre Bellec.
Hao-Ting Wang migrated the code base to `nilearn`.
Nicolas Gensollen and Bertrand Thirion reviewed the code migrated to `nilearn`.
We thank Chris Markiewicz for feedbacks related to fMRIPrep.

Hao-Ting Wang and Pierre Bellec drafted the paper.

Please see the [original repository](https://github.com/SIMEXP/load_confounds#contributors-) for a full history of development and contributors.

<!-- Funding -->

# References