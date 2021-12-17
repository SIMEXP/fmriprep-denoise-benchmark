---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: [hide-input, hide-output]

import tarfile
import io
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, spearmanr
from fmriprep_denoise.metrics import partial_correlation, fdr, calculate_median_absolute


# Load metric data
path_root = Path.cwd().parents[0] / "inputs"
file_qcfc = "dataset-ds000228_atlas-schaefer7networks_nroi-400_qcfc.tsv"
file_dist = "atlas-schaefer7networks_nroi-400_desc-distance.tsv"
file_network = "dataset-ds000228_atlas-schaefer7networks_nroi-400_modularity.tsv"
# file_qcfc = "dataset-ds000228_atlas-gordon333_nroi-333_qcfc.tsv"
# file_dist = "atlas-gordon333_nroi-333_desc-distance.tsv"
# file_network = "dataset-ds000228_atlas-gordon333_nroi-333_modularity.tsv"
file_dataset = "dataset-ds000228_desc-movement_phenotype.tsv"
```

```{code-cell} ipython3
:tags: [hide-input, hide-output]

pairwise_distance = pd.read_csv(path_root / file_dist, sep='\t')
movement = pd.read_csv(path_root / file_dataset, sep='\t', index_col=0, header=0, encoding='utf8')
qcfc = pd.read_csv(path_root / file_qcfc, sep='\t', index_col=0)
modularity = pd.read_csv(path_root / file_network, sep='\t', index_col=0)


# separate correlation from siginficant value
sig_per_edge = qcfc.filter(regex="pvalue")
sig_per_edge.columns = [col.split('_')[0] for col in sig_per_edge.columns]
metric_per_edge = qcfc.filter(regex="correlation")
metric_per_edge.columns = [col.split('_')[0] for col in metric_per_edge.columns]
```

# OHBM 2022 abstract

## Impact of confound removal strategies on functional connectivity generated from fMRIPrep outputs

H-T Wang[^1], S L Meisler[^2][^3], H Shamarke, F Paugam[^1][^4], N Gensollen[^5], B Thirion[^5], C Markiewicz[^6], P Bellec[^1][^7]

[^1] Centre de recherche de l'institut Universitaire de gériatrie de Montréal (CRIUGM), Montréal, Québec, Canada
[^2] Harvard University, MA, USA
[^3] Massachusetts Institute of Technology, MA, USA
[^4] Computer Science and Operations Research Department, Université de Montréal, Montréal, Québec, Canada
[^5] Inria, CEA, Université Paris-Saclay, Paris, France
[^6] Department of Psychology, Stanford University, Stanford, United States
[^7] Psychology Department, Université de Montréal, Montréal, Québec, Canada

### Introduction

Selecting an optimal denoising strategy is a key issue when processing fMRI data. 
The popular software fMRIPrep {cite:p}`esteban_fmriprep_2020` aims to standardize fMRI preprocessing, 
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


| strategy   | image                          | `high_pass` | `motion` | `wm_csf` | `global_signal` | `scrub` | `fd_thresh` | `compcor`     | `n_compcor` | `ica_aroma` | `demean` |
|------------|--------------------------------|-------------|----------|----------|-----------------|---------|-------------|---------------|-------------|-------------|----------|
| baseline   | `desc-preproc_bold`            | `True`      | N/A      | N/A      | N/A             | N/A     | N/A         | N/A           | N/A         | N/A         | `True`   |
| simple     | `desc-preproc_bold`            | `True`      | full     | basic    | N/A             | N/A     | N/A         | N/A           | N/A         | N/A         | `True`   |
| simple+gsr | `desc-preproc_bold`            | `True`      | full     | basic    | basic           | N/A     | N/A         | N/A           | N/A         | N/A         | `True`   |
| scrubbing  | `desc-preproc_bold`            | `True`      | full     | full     | N/A             | 5       | 0.5         | N/A           | N/A         | N/A         | `True`   |
| simple+gsr | `desc-preproc_bold`            | `True`      | full     | full     | basic           | 5       | 0.5         | N/A           | N/A         | N/A         | `True`   |
| compcor    | `desc-preproc_bold`            | `True`      | full     | N/A      | N/A             | N/A     | N/A         | anat_combined | all         | N/A         | `True`   |
| compcor6   | `desc-preproc_bold`            | `True`      | full     | N/A      | N/A             | N/A     | N/A         | anat_combined | 6           | N/A         | `True`   |
| aroma      | `desc-smoothAROMAnonaggr_bold` | `True`      | N/A      | basic    | N/A             | N/A     | N/A         | N/A           | N/A         | full        | `True`   |
| aroma+gsr  | `desc-smoothAROMAnonaggr_bold` | `True`      | N/A      | basic    | basic           | N/A     | N/A         | N/A           | N/A         | full        | `True`   |

We used three metrics {cite:p}`ciric_benchmarking_2017`, {cite:p}`parkes_evaluation_2018` to evaluate the denoising results:
1. Quality control / functional connectivity (QCFC {cite:p}`power_recent_2015`): partial correlation between motion and connectivity with age and sex as covariates. We control for multiple comparisons with false positive rate correction.
2. Distance-dependent effects of motion on connectivity {cite:p}`power_spurious_2012`: correlation between node-wise Euclidean distance and QC-FC.
3. Network modularity {cite:p}`satterthwaite_impact_2012`: graph community detection based on Louvain method, implemented in the Brain Connectome Toolbox.

### Results

#### QC-FC

No denoise strategy removed the correlation with motion captured by mean framewise displacement. 
`aroma`, `compcor6`, and `simple` reduced the correlation between connectivity edges and mean framewise displacement. 
`scrubbing` and `scrubbing+gsr` performed the best, as seen in previous work {cite:p}`power_recent_2015`. 
`compcor`, which applies compcor-based regressors covering 50% of the variance, performs worse than the connectome baseline created with high-pass filtered time series. 
Surprisingly, all strategies with global signal regression underperform, contradicting the existing literature {cite:p}`ciric_benchmarking_2017` {cite:p}`parkes_evaluation_2018`.

```{code-cell} ipython3
:tags: [hide-input]

bar_color = sns.color_palette()[0]

# multiple comparision on qcfc
long_qcfc_sig= sig_per_edge.melt()
long_qcfc_sig['fdr'] = long_qcfc_sig.groupby('variable')['value'].transform(fdr)
long_qcfc_sig = long_qcfc_sig.groupby('variable').apply(lambda x: 100*x.fdr.sum()/x.fdr.shape[0])
long_qcfc_sig = pd.DataFrame(long_qcfc_sig, columns=["p_corrected"])

order = long_qcfc_sig.sort_values('p_corrected').index.tolist()
ax = sns.barplot(data=long_qcfc_sig.T, ci=None, order=order, color=bar_color)
ax.set_title("Percentage of edge significantly correlated with mean FD")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set(ylabel="Percentage %",
       xlabel="confound removal strategy")
plt.tight_layout()
plt.savefig("sig_qcfc.png", dpi=300)
```

```{code-cell} ipython3
:tags: [hide-input]

median_absolute = metric_per_edge.apply(calculate_median_absolute)
order = median_absolute.sort_values().index.tolist()

ax = sns.barplot(data=(pd.DataFrame(median_absolute).T), ci=None, order=order, color=bar_color)
ax.set_title("Median absolute deviation QC-FC")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set(ylabel="Median absolute deviation",
       xlabel="confound removal strategy")
plt.tight_layout()
plt.savefig("mad_qcfc.png", dpi=300)

def draw_absolute_median(data, **kws):
    ax = plt.gca()
    mad = calculate_median_absolute(data['qcfc'])
    ax.vlines(mad, ymin=0, ymax=0.5, color='r', linestyle=':')
    

long_qcfc = metric_per_edge.melt()
long_qcfc.columns = ["Strategy", "qcfc"]
long_qcfc["row"] = np.sort(np.tile(np.arange(3), int(long_qcfc.shape[0] / 3)))
long_qcfc["col"] = np.tile(np.sort(np.tile(np.arange(3), int(long_qcfc.shape[0] / 9))), 3)

g = sns.displot(
    long_qcfc, x="qcfc", col="col", row="row", kind='kde', fill=True, height=1.5, aspect=2
)
g.set(ylabel="Density")
g.map_dataframe(draw_absolute_median)
for i, name in zip(range(9), metric_per_edge.columns):
    axis_i = int(i / 3)
    axis_j = i % 3
    g.facet_axis(axis_i, axis_j).set(title=name)
    if axis_i == 2:
        g.facet_axis(axis_i, axis_j).set(xlabel="Pearson\'s correlation: \nmean FD and\nconnectome edges")
        
g.fig.subplots_adjust(top=0.9) 
g.fig.suptitle('Distribution of correlation between framewise distplacement and edge strength')
plt.tight_layout()
plt.savefig("dist_qcfc.png", dpi=300)
```

#### Distance-dependent effects of motion on connectivity

Consistent with the literature, functional connectivity strength reduced with the distance between the two nodes. 
The `simple` and `compcor` strategies reduced the distance dependence more than the baseline.

```{code-cell} ipython3
:tags: [hide-input]

corr_distance, p_val = spearmanr(pairwise_distance.iloc[:, -1], metric_per_edge)  

corr_distance = pd.DataFrame(corr_distance[1:, 0], index=metric_per_edge.columns)
long_qcfc['distance'] = np.tile(pairwise_distance.iloc[:, -1].values, 9)

order = corr_distance.sort_values(0).index.tolist()

ax = sns.barplot(data=corr_distance.T, ci=None, order=order, color=bar_color)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title("Distance-dependent effects of motion")
ax.set(ylim=(-0.5, 0.05))
ax.set(ylabel="Nodewise correlation between\nEuclidian distance and QC-FC metric",
        xlabel="confound removal strategy")
plt.tight_layout()
plt.savefig("corr_dist_qcfc_mean.png", dpi=300)

g = sns.FacetGrid(long_qcfc, col="col", row="row", height=1.7, aspect=1.5)
g.map(sns.regplot, 'distance', 'qcfc', fit_reg=True, ci=None, 
      line_kws={'color': 'red'}, scatter_kws={'s': 0.5, 'alpha': 0.15,})
g.refline(y=0)
for i, name in zip(range(9), metric_per_edge.columns):
    axis_i = int(i / 3)
    axis_j = i % 3
    g.facet_axis(axis_i, axis_j).set(title=name)
    if axis_i == 2:
        g.facet_axis(axis_i, axis_j).set(xlabel="Distance (mm)")
    if axis_j == 0:
        g.facet_axis(axis_i, axis_j).set(ylabel="QC-FC")
        
g.fig.subplots_adjust(top=0.9) 
g.fig.suptitle('Correlation between nodewise Euclidian distance and QC-FC')
plt.tight_layout()
plt.savefig("corr_dist_qcfc_dist.png", dpi=300)
```

#### Network modularity

All strategies increased the network modularity compared to the `baseline`. 
GSR-based strategies improved the network modularity compared to their conunterparts.
The correlation between modularity quality and motion for each denoising approach shows that compcor-based and ICA-AROMA strategies are the best at eliminating correlations between motion and modularity.

```{code-cell} ipython3
:tags: [hide-input]

corr_modularity = []
z_movement = movement.apply(zscore)
for column, values in modularity.iteritems():
    cur_data = pd.concat((modularity[column], 
                          movement[['mean_framewise_displacement']], 
                          z_movement[['Age', 'Gender']]), axis=1).dropna()
    current_strategy = partial_correlation(cur_data[column].values, 
                                           cur_data['mean_framewise_displacement'].values, 
                                           cur_data[['Age', 'Gender']].values)
    current_strategy['strategy'] = column
    corr_modularity.append(current_strategy)

plt.figure(figsize=(7, 5))
plt.subplot(1, 2, 1)
order = modularity.mean().sort_values().index.tolist()
ax = sns.barplot(data=modularity, order=order, color=bar_color)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title("Identifiability of network structure\nafter denoising")
ax.set(ylabel="Mean modularity quality (a.u.)",
       xlabel="confound removal strategy")
plt.subplot(1, 2, 2)

corr_modularity = pd.DataFrame(corr_modularity).sort_values('correlation')
ax = sns.barplot(data=corr_modularity, y='correlation', x='strategy', ci=None, color=bar_color)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title("Correlation between\nnetwork modularity and \nmean framewise displacement")
ax.set(ylabel="Pearson's correlation",
       xlabel="confound removal strategy")
plt.tight_layout()
plt.savefig("modularity.png", dpi=300)
```

### Conclusions


We replicated previous findings demonstrating the usefulness of standard denoising strategies (compcor, aroma, etc.).
However, results involving global signal regression methods systematically contradict the literature{cite:p}`ciric_benchmarking_2017` {cite:p}`parkes_evaluation_2018`. 
This evaluation is implemented in a fully reproducible jupyter book framework, and it can be applied to evaluate denoising strategies for future fMRIPrep release. 
This software may also be useful for researchers to select the most suitable strategy and produce denoising benchmarks for their own dataset.                 


### References
```{bibliography}
:filter: docname in docnames
```

```{code-cell} ipython3

```
