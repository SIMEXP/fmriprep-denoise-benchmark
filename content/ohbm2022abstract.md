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

# OHBM 2022 abstract

## Impact of confound removal strategies on functional connectivity generated from fMRIprep preprocessed data

Authors

### Introduction

Denoising strategy is an important topic in fMRI literature. 
The impact of the choice of confound regressor on functional connectivity has been a key debates in the field of fMRI (cf. global signal regression). 
Recent minimal preprocessing pipeline, fMRIPrep {cite:p}`esteban_fmriprep_2020`, aims to reduce the degree of freedom during the preprocessing step. 
However, a wide range of confound regressors can still introduce errors by the users. 
Without good understanding of the literature or the official documentation, users can still introduce error or unwanted noise while performing confound regressing. 
It’s difficult to navigate the confounds and implement the sensible subset of variables in downstream analysis. 
Lastly, recent literature has shown the tool-based variability and the potential impact on the results {cite:p}`li_moving_2021`. 
We hope to provide a useful reference for fMRIPrep users, and evaluate if the implementation in fMRIPrep provides consistent results as the past literature using other preprocessing procedures. 
Here we provide an uniformed API to retrieve fMRIPrep generated confounds implemented in NiLearn. 
Four predefined strategies are provided with the API. 
The current research provides a benchmarking on the strategies using the three from {cite:t}`ciric_benchmarking_2017`: 
the residual relationship between motion and connectivity, distance-dependent effects of motion on connectivity, network identifiability. 


### Methods

The dataset of choice is ds000228 {cite:p}`richardson_development_2018` on OpenNeuro, preprocessed with fMRIprep LTS20.2.1, using fMRIPrep-slurm wrapper with option `--use-aroma`. 
Time series are extracted using Schaefer 7 network atlas of 400 dimensions {cite:p}`schaefer_local-global_2017` and applied the following denoising strategies:

- `simple`: high pass filtering, motion (six base motion parameters and temporal derivatives, quadratic terms and their six temporal derivatives, 24 parameters in total), signal from tissue masks (white matter and  csf, 2 parameters), applied on output suffixed`desc-prepro_bold`.
- `simple+gsr`: strategy above, with basic global signal, applied on output suffixed`desc-prepro_bold`.
- `scrubbing`: high pass filtering, motion (six base motion parameters and temporal derivatives, quadratic terms and their six temporal derivatives, 24 parameters in total),  signal from tissue masks (white matter and csf, basic, the temporal derivative and quadratic, 8 parameters), motion outlier threshold set at 0.5 framewise displacement, segments with less than 5 consecutive volumes are removed, applied on output suffixed`desc-prepro_bold`.
- `scrubbing+gsr`: strategy above, with basic global signal, applied on output suffixed`desc-prepro_bold`.
- `acompcor`: high pass filtering, 24 motion parameters, compcor components explaining 50% of the variance with combined white matter and csf mask, applied on output suffixed`desc-prepro_bold`.
- `acompcor6`: high pass filtering, 24 motion parameters, top 6 compcor components with combined white matter and csf mask
- `aroma`: high pass filtering, signal from tissue masks (white matter and  csf, 2 parameters), applied on output suffixed `desc-smoothAROMAnonaggr_bold`
- `aroma+gsr`: high pass filtering, signal from tissue masks (white matter and  csf, 2 parameters), global signal, applied on output suffixed `desc-smoothAROMAnonaggr_bold`

The following metrics are used to assess quality of functional connectivity: 
the residual relationship between motion and connectivity, 
distance-dependent effects of motion on connectivity, 
network identifiability.

Code and processed data to reproduce the current analysis can be found on [github](https://github.com/SIMEXP/fmriprep-denoise-benchmark). 

### Results

```{code-cell} ipython3
:tags: [hide-input, hide-output]

import tarfile
import io
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from fmriprep_denoise.metrics import partial_correlation


# Load metric data
path_root = Path.cwd().parents[0] / "inputs"
file_qcfc = "dataset-ds000288_atlas-schaefer7networks_nroi-400_desc-qcfc.tsv"
file_dist = "atlas/schaefer7networks/atlas-schaefer7networks_nroi-400_desc-distance.tsv"
file_network = "dataset-ds000288_atlas-schaefer7networks_nroi-400_desc-modularity.tsv"
file_dataset = "dataset-ds000288.tar.gz"

with tarfile.open(path_root / file_dataset, 'r:gz') as tar:
    movement = tar.extractfile(
        "dataset-ds000288/dataset-ds000288_desc-movement_phenotype.tsv").read()
    movement = pd.read_csv(io.BytesIO(movement),
                        sep='\t', index_col=0, header=0, encoding='utf8')
    movement = movement.sort_index()


pairwise_distance = pd.read_csv(path_root / file_dist, sep='\t')
qcfc = pd.read_csv(path_root / file_qcfc, sep='\t', index_col=0)
modularity = pd.read_csv(path_root / file_network, sep='\t', index_col=0)

sig_per_edge = qcfc.filter(regex="pvalue")
sig_per_edge.columns = [col.split('_')[0] for col in sig_per_edge.columns]
metric_per_edge = qcfc.filter(regex="correlation")
metric_per_edge.columns = [col.split('_')[0] for col in metric_per_edge.columns]

long_qcfc = metric_per_edge.melt()
long_qcfc.columns = ["Strategy", "qcfc"]
long_qcfc["row"] = np.hstack((np.ones(int(long_qcfc.shape[0] / 3)), 
                              np.ones(int(long_qcfc.shape[0] / 3))* 2,
                              np.ones(int(long_qcfc.shape[0] / 3))* 3)
                             )
long_qcfc["col"] = np.tile(np.hstack((np.ones(metric_per_edge.shape[0]) * i for i in range(3))), 3)
```

```{code-cell} ipython3
:tags: [hide-input]

ax = sns.barplot(data=(sig_per_edge<0.05), ci=None)
ax.set_title("Proportion of edge significantly correlated with mean FD")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set(ylabel="Pearson's correlation",
       xlabel="confound removal strategy")
plt.tight_layout()
```

```{code-cell} ipython3
:tags: [hide-input]

def draw_absolute_median(data, **kws):
    ax = plt.gca()
    mad = (data['qcfc'] - data['qcfc'].median()).abs().median()
    ax.vlines(mad, ymin=0, ymax=0.4)

g = sns.displot(
    long_qcfc, x="qcfc", col="col", row="row", kind='kde', fill=True
)
g.set(ylabel="Density")
g.map_dataframe(draw_absolute_median)

for i, name in zip(range(9), metric_per_edge.columns):
    axis_i = int(i / 3)
    axis_j = i % 3
    g.facet_axis(axis_i, axis_j).set(title=name)
    if axis_i == 1:
        g.facet_axis(axis_i, axis_j).set(xlabel="Pearson\'s correlation: \nmean FD and\nconnectome edges")
```

```{code-cell} ipython3
:tags: [hide-input]

pairwise_distance = pairwise_distance.apply(zscore)
metric_per_edge = metric_per_edge.apply(zscore)
corr_distance = np.corrcoef(pairwise_distance.iloc[:, -1], metric_per_edge.T)[1:, 0]
corr_distance = pd.DataFrame(corr_distance, index=metric_per_edge.columns)
ax = sns.barplot(data=corr_distance.T, ci=None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title("Distance-dependent effects of motion")
ax.set(ylabel="Nodewise correlation between\nEuclidian distance and QC-FC metric",
        xlabel="confound removal strategy")
plt.tight_layout()


long_qcfc['distance'] = np.tile(pairwise_distance.iloc[:, -1].values, 9)
g = sns.FacetGrid(long_qcfc, col="col", row="row")
g.map(sns.regplot, 'distance', 'qcfc', fit_reg=True, ci=None, 
      line_kws={'color': 'red'}, scatter_kws={'s': 0.5, 'alpha': 0.15,})
g.refline(y=0)
for i, name in zip(range(9), metric_per_edge.columns):
    axis_i = int(i / 3)
    axis_j = i % 3
    g.facet_axis(axis_i, axis_j).set(title=name)
```

```{code-cell} ipython3
:tags: [hide-input]

plt.figure()
ax = sns.barplot(data=modularity)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title("Identifiability of network structure after de-noising")
ax.set(ylabel="Mean modularity quality (a.u.)",
       xlabel="confound removal strategy")
plt.tight_layout()

corr_modularity = []
z_movment = movement.apply(zscore)
for column, values in modularity.iteritems():
    current_strategy = partial_correlation(values.values, 
                                           z_movment['mean_framewise_displacement'].values, 
                                           z_movment[['Age', 'Gender']].values)
    current_strategy['strategy'] = column
    corr_modularity.append(current_strategy)
       
plt.figure()
corr_modularity = pd.DataFrame(corr_modularity)
ax = sns.barplot(data=corr_modularity, y='correlation', x='strategy', ci=None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title("Correlation between network modularity and motion")
ax.set(ylabel="Pearson's correlation",
       xlabel="confound removal strategy")
plt.tight_layout()
```

### Conclusions

We aim to run the same benchmark on different fMRIPrepLTS outputs and different type of parcelation scheme.


### References
```{bibliography}
:filter: docname in docnames
```
