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

```{code-cell}
:tags: [hide-input, hide-output]

import warnings

warnings.filterwarnings('ignore')
from fmriprep_denoise.visualization import figures, utils
from myst_nb import glue

path_root = utils.get_data_root() / "denoise-metrics"

# Load metric data
dataset = 'ds000228'
fmriprep_version = 'fmriprep-20.2.1lts'

atlas_name = 'schaefer7networks'
dimension = 400
```

# OHBM 2022 abstract - submitted text

The preliminary results will be presented at OHBM 2022 as a poster.
Please find poster number `WTh570`.

Find the presenter at the
[virtual poster session](https://event.fourwaves.com/ohbm-2022/abstracts/d49d130b-7f83-4c87-92f4-e1a8e319502b)
on __Wednesday, June 8, 2022, 8:30 PM GMT + 1__.

At Glasgow, please contact the presenter on [Twitter](https://twitter.com/HaoTingW713) to schedule a time to chat,
or come to see the presenter on __Wednesday, June 22, 2022, 12:45 PM__ at the poster hall.

```{image} ../images/ohbm2022_abstract_head.png
:alt: poster
:align: center
```

## Impact of confound removal strategies on functional connectivity generated from fMRIPrep outputs

H-T Wang[^1], S L Meisler[^2][^3], H Shamarke, F Paugam[^1][^4], N Gensollen[^5], B Thirion[^5], C Markiewicz[^6], P Bellec[^1][^7]

[^1]: Centre de recherche de l'institut Universitaire de gériatrie de Montréal (CRIUGM), Montréal, Québec, Canada

[^2]: Harvard University, MA, USA

[^3]: Massachusetts Institute of Technology, MA, USA

[^4]: Computer Science and Operations Research Department, Université de Montréal, Montréal, Québec, Canada

[^5]: Inria, CEA, Université Paris-Saclay, Paris, France

[^6]: Department of Psychology, Stanford University, Stanford, United States

[^7]: Psychology Department, Université de Montréal, Montréal, Québec, Canada

### Introduction

Selecting an optimal denoising strategy is a key issue when processing fMRI data.
The popular software fMRIPrep {cite:p}`fmriprep1` aims to standardize fMRI preprocessing,
but users are still offered a wide range of confound regressors to choose from to denoise data.
Without a good understanding of the literature or the fMRIPrep documentation,
users can select suboptimal strategies.
Current work aims to provide a useful reference for fMRIPrep users by systematically evaluating the impact of different confound regression strategies,
and by contrasting the results with past literature based on alternative preprocessing software.

### Methods

We selected dataset ds000228 {cite:p}`richardson_development_2018` on OpenNeuro, which we preprocessed with fMRIPrep LTS20.2.1 using option `--use-aroma`.
Time series were extracted using the Schaefer 7 network atlas with 400 ROIs {cite:p}`schaefer_local-global_2017`.
We applied the denoising strategies listed in the table below using fMRIPrep-generated confounds.
Subjects with less than 80% of remaining volumes after scrubbing with a 0.5 mm threshold were excluded from all analysis.
We also calculated the connectome from high-pass filtered time series as a comparison baseline.


| strategy      | image                          | `high_pass` | `motion` | `wm_csf` | `global_signal` | `scrub` | `fd_thresh` | `compcor`     | `n_compcor` | `ica_aroma` | `demean` |
|---------------|--------------------------------|-------------|----------|----------|-----------------|---------|-------------|---------------|-------------|-------------|----------|
| baseline      | `desc-preproc_bold`            | `True`      | N/A      | N/A      | N/A             | N/A     | N/A         | N/A           | N/A         | N/A         | `True`   |
| simple        | `desc-preproc_bold`            | `True`      | full     | basic    | N/A             | N/A     | N/A         | N/A           | N/A         | N/A         | `True`   |
| simple+gsr    | `desc-preproc_bold`            | `True`      | full     | basic    | basic           | N/A     | N/A         | N/A           | N/A         | N/A         | `True`   |
| scrubbing     | `desc-preproc_bold`            | `True`      | full     | full     | N/A             | 5       | 0.5         | N/A           | N/A         | N/A         | `True`   |
| scrubbing+gsr | `desc-preproc_bold`            | `True`      | full     | full     | basic           | 5       | 0.5         | N/A           | N/A         | N/A         | `True`   |
| compcor       | `desc-preproc_bold`            | `True`      | full     | N/A      | N/A             | N/A     | N/A         | anat_combined | all         | N/A         | `True`   |
| compcor6      | `desc-preproc_bold`            | `True`      | full     | N/A      | N/A             | N/A     | N/A         | anat_combined | 6           | N/A         | `True`   |
| aroma         | `desc-smoothAROMAnonaggr_bold` | `True`      | N/A      | basic    | N/A             | N/A     | N/A         | N/A           | N/A         | full        | `True`   |
| aroma+gsr     | `desc-smoothAROMAnonaggr_bold` | `True`      | N/A      | basic    | basic           | N/A     | N/A         | N/A           | N/A         | full        | `True`   |

We used three metrics {cite:p}`ciric_benchmarking_2017`, {cite:p}`parkes_evaluation_2018` to evaluate the denoising results:
1. Quality control / functional connectivity (QCFC {cite:p}`power_recent_2015`): partial correlation between motion and connectivity with age and sex as covariates. We control for multiple comparisons with false positive rate correction.
2. Distance-dependent effects of motion on connectivity {cite:p}`power_scrubbing_2012`: correlation between node-wise Euclidean distance and QC-FC.
3. Network modularity {cite:p}`satterthwaite_impact_2012`: graph community detection based on Louvain method, implemented in the Brain Connectome Toolbox.

### Results

#### QC-FC and distance-dependent effect

No denoise strategy removed the correlation with motion captured by mean framewise displacement.
`aroma`, `compcor6`, and `simple` reduced the correlation between connectivity edges and mean framewise displacement.
`scrubbing` and `scrubbing+gsr` performed the best, as seen in previous work {cite:p}`power_recent_2015`.
`compcor`, which applies compcor-based regressors covering 50% of the variance, performs worse than the connectome baseline created with high-pass filtered time series.
Surprisingly, all strategies with global signal regression underperform, contradicting the existing literature {cite:p}`ciric_benchmarking_2017` {cite:p}`parkes_evaluation_2018`.

```{code-cell}
:tags: [hide-input, remove-output]

fig = figures.plot_motion_resid(dataset, fmriprep_version, path_root, atlas_name=atlas_name, dimension=dimension)
glue('ohbm-qcfc-fig', fig, display=False)
```

```{glue:figure} ohbm-qcfc-fig
:figwidth: 800px
:name: "ohbm-qcfc-fig"
```

#### Distance-dependent effects of motion on connectivity

Consistent with the literature, `aroma` reduces the distance dependency of motion on connectivity.

```{code-cell}
:tags: [hide-input, remove-output]

fig = figures.plot_distance_dependence(dataset, fmriprep_version, path_root, atlas_name=atlas_name, dimension=dimension)
glue('ohbm-dist-fig', fig, display=False)
```

```{glue:figure} ohbm-dist-fig
:figwidth: 800px
:name: "ohbm-dist-fig"
```

#### Network modularity

All strategies increased the overall network modularity compared to the `baseline`, with scrubbing based methods performing the best out of all.
GSR-based strategies improved the network modularity compared to their conunterparts.
The correlation between modularity quality and motion for each denoising approach shows that compcor-based and ICA-AROMA strategies are the best at eliminating correlations between motion and modularity.

```{code-cell}
:tags: [hide-input, remove-output]

fig = figures.plot_network_modularity(dataset, fmriprep_version, path_root, atlas_name=atlas_name, dimension=dimension)
glue('ohbm-network-fig', fig, display=False)
```

```{glue:figure} ohbm-network-fig
:figwidth: 800px
:name: "ohbm-network-fig"
```

### Conclusions

We replicated previous findings demonstrating the usefulness of standard denoising strategies (compcor, aroma, etc.).
However, results involving global signal regression methods systematically contradict the literature{cite:p}`ciric_benchmarking_2017` {cite:p}`parkes_evaluation_2018`.
This evaluation is implemented in a fully reproducible jupyter book framework, and it can be applied to evaluate denoising strategies for future fMRIPrep release.
This software may also be useful for researchers to select the most suitable strategy and produce denoising benchmarks for their own dataset.
