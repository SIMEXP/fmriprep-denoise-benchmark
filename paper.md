---
title: "A reproducible benchmark of resting-state fMRI denoising strategies using fMRIPrep and Nilearn"
tags:
  - reproducibility
  - nilearn
  - fMRIPrep
  - nuisance regressor
  - resting-state fMRI
  - functional connectivity
authors:
  - name: Hao-Ting Wang
    affiliation: "1, 7"
    orcid: 0000-0003-4078-2038
  - name: Steven L Meisler
    affiliation: "2, 3"
    orcid: 0000-0002-8888-1572
  - name: Hanad Sharmarke
    affiliation: 1
    orcid:
  - name: Natasha Clarke
    affiliation: 1
    orcid: 
  - name: François Paugam
    affiliation: 4
    orcid:
  - name: Nicolas Gensollen
    affiliation: 5
    orcid: 0000-0001-7199-9753
  - name: Christopher J Markiewicz
    affiliation: 6
    orcid: 0000-0002-6533-164X
  - name: Bertrand Thirion
    affiliation: 5
    orcid: 0000-0001-5018-7895
  - name: Pierre Bellec
    affiliation: "1, 7"
    orcid: 0000-0002-9111-0699
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
date: 11 May 2023
bibliography: paper.bib
---

# Summary

Reducing contributions from non-neuronal sources is a crucial step in functional magnetic resonance imaging (fMRI) connectivity analyses.
Many viable strategies for denoising fMRI are used in the literature, 
and practitioners rely on denoising benchmarks for guidance in the selection of an appropriate choice for their study.
However, fMRI denoising software is an ever-evolving field, and the benchmarks can quickly become obsolete as the techniques or implementations change.
In this work, we present a fully reproducible denoising benchmark featuring a range of denoising strategies and evaluation metrics for connectivity analyses,
built primarily on the fMRIPrep [@fmriprep1] and Nilearn [@nilearn] software packages.
We apply this reproducible benchmark to investigate the robustness of the conclusions across two different datasets and two versions of fMRIPrep.
The majority of benchmark results were consistent with prior literature.
Scrubbing, a technique which excludes time points with excessive motion,
combined with global signal regression, is generally effective at noise removal.
Scrubbing however disrupts the continuous sampling of brain images and is incompatible with some statistical analyses,
e.g. auto-regressive modeling. In this case, a simple strategy using motion parameters,
average activity in select brain compartments, and global signal regression should be preferred.
Importantly, we found that certain denoising strategies behave inconsistently across datasets and/or versions of fMRIPrep,
or had a different behavior than in previously published benchmarks.
These results demonstrate that a reproducible denoising benchmark can effectively assess the robustness of conclusions across multiple datasets and software versions.
In addition to reproducing core computations, interested readers can also reproduce or modify the figures of the article using the Jupyter Book project [@jupyter] and the Neurolibre [@neurolibre] reproducible preprint server.
With the denoising benchmark, we hope to provide useful guidelines for the community, 
and that our software infrastructure will facilitate continued development as the state-of-the-art advances. 

![Overview of API.\label{top_level_fig}](./content/images/api_summary.png)

# Acknowledgements

The initial API was started by Hanad Sharmarke and Pierre Bellec. 
The implementation was completed by Hao-Ting Wang, Steven Meisler, François Paugam, and Pierre Bellec. 
Hao-Ting Wang migrated the code base to Nilearn. 
Nicolas Gensollen and Bertrand Thirion reviewed the code migrated to Nilearn. 
We thank Chris Markiewicz for feedback related to fMRIPrep.

Hao-Ting Wang and Pierre Bellec drafted the initial version of the paper, with critical feedback from Natasha Clarke. 

Please see the original repository for a history of initial development and [contributors](https://github.com/SIMEXP/load_confounds#contributors-), 
and this [issue](https://github.com/nilearn/nilearn/issues/2777 ) for a history of the integration in Nilearn and all the linked Pull Requests.

The project is funded by IVADO PRF3, CCNA and Courtois Foundation, the neuromind collaboration. 
HTW and NC funded by Institut de valorisation des données (IVADO) postdoctoral research funding. 
SLM was funded by the National Institute on Deafness and Other Communication Disorders (NIDCD; Grant 5T32DC000038). 
CJM funded by NIMH 5R24MH117179. 
FP funded by Courtois Neuromod. 
PB funded by Fonds de Recherche du Québec - Santé (FRQ-S). 

# References