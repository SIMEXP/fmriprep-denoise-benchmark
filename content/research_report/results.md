---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(launch:thebe)=
# Results

```{code-cell} ipython3
:tags: [hide-input, remove-output]

import warnings

warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns

from nilearn.plotting import plot_matrix
from nilearn.connectome import vec_to_sym_matrix

from statsmodels.stats.weightstats import ttest_ind

from fmriprep_denoise.visualization import figures, tables, utils
from fmriprep_denoise.features.derivatives import get_qc_criteria

from myst_nb import glue
import ipywidgets as widgets
from ipywidgets import interactive


path_root = utils.get_data_root() / "denoise-metrics"
strategy_order = list(utils.GRID_LOCATION.values())
group_order = {'ds000228': ['adult', 'child'], 'ds000030':['control', 'ADHD', 'bipolar', 'schizophrenia']}

fmriprep_version = 'fmriprep-20.2.1lts'
```

## Sample and subgroup size change based on quality control criteria

```{code-cell} ipython3
def demographic_table(criteria_name):
    criteria = get_qc_criteria(criteria_name)
    ds000228 = tables.lazy_demographic('ds000228', fmriprep_version, path_root, **criteria)
    ds000030 = tables.lazy_demographic('ds000030', fmriprep_version, path_root, **criteria)

    desc = pd.concat({'ds000228': ds000228, 'ds000030': ds000030}, axis=1, names=['dataset'])
    desc = desc.style.set_table_attributes('style="font-size: 12px"')
    display(desc)

criteria_name = widgets.Select(
    options=['stringent', 'minimal', None],
    value='stringent',
    description='Threshould: ',
    disabled=False
)

interactive(demographic_table, criteria_name=criteria_name)
```

To evaluate the impact of denoising strategy on connectomes in a practical scenario, 
we excluded subjects with high motion as such subjects would be normally excluded in the data quality control stage 
(see section {ref}`framewise-displacement`). 
We applied the stringent thresholds introduced in {cite:t}`parkes_evaluation_2018`.
{numref}`tbl:demographic_stringent` below shows the demographic information of the datasets after the automatic motion quality control.
For report in the full sample, please see [the supplemental material](../supplementary_materials/results_full-sample.md).

```{glue:figure} demographic_stringent
:name: "tbl:demographic_stringent"

Sample demographic information after removing subjects with high motion.
```

```{code-cell} ipython3
:tags: [hide-input, remove-output]

# statistic report
for_plotting = {}

datasets = ['ds000228', 'ds000030']
baseline_groups = ['adult', 'control']
for dataset, baseline_group in zip(datasets, baseline_groups):
    _, data, _ = tables.get_descriptive_data(dataset, fmriprep_version, path_root, **stringent)
    baseline = data[data['groups'] == baseline_group]
    for group in group_order[dataset]:
        compare = data[data['groups'] == group]
        glue(
            f'{dataset}_{group}_mean_qc',
            compare['mean_framewise_displacement'].mean(),
        )
        glue(
            f'{dataset}_{group}_sd_qc',
            compare['mean_framewise_displacement'].std(),
        )
        glue(
            f'{dataset}_{group}_n_qc',
            compare.shape[0],
        )
        if group != baseline_group:
            t_stats, pval, df = ttest_ind(
                baseline['mean_framewise_displacement'],
                compare['mean_framewise_displacement'],
                usevar='unequal',
            )
            glue(f'{dataset}_t_{group}_qc', t_stats)
            glue(f'{dataset}_p_{group}_qc', pval)
            glue(f'{dataset}_df_{group}_qc', df)
    for_plotting.update({dataset: data})
```

We checked the difference in the mean framewise displacement of each sample and the sub-groups {numref}`fig:meanFD-stringent`.
In `ds000228`, there was still a significant difference in motion during the scan captured by mean framewise displacement 
between the child 
(M = {glue:text}`ds000228_child_mean_qc:.2f`, SD = {glue:text}`ds000228_child_sd_qc:.2f`, n = {glue:text}`ds000228_child_n_qc:i`)
and adult sample
(M = {glue:text}`ds000228_adult_mean_qc:.2f`, SD = {glue:text}`ds000228_adult_sd_qc:.2f`, n = {glue:text}`ds000228_adult_n_qc:i`,
t({glue:text}`ds000228_df_child_qc:.2f`) = {glue:text}`ds000228_t_child_qc:.2f`, p = {glue:text}`ds000228_p_child_qc:.3f`).
In `ds000030`, the only patient group shows a difference compared to the control 
(M = {glue:text}`ds000030_control_mean_qc:.2f`, SD = {glue:text}`ds000030_control_sd_qc:.2f`, n = {glue:text}`ds000030_control_n_qc:i`)
is still the schizophrenia group 
(M = {glue:text}`ds000030_schizophrenia_mean_qc:.2f`, SD = {glue:text}`ds000030_schizophrenia_sd_qc:.2f`, n = {glue:text}`ds000030_schizophrenia_n_qc:i`;
t({glue:text}`ds000030_df_schizophrenia_qc:.2f`) = {glue:text}`ds000030_t_schizophrenia_qc:.2f`, p = {glue:text}`ds000030_p_schizophrenia_qc:.3f`).
There was no difference between the control and ADHD group
(M = {glue:text}`ds000030_ADHD_mean_qc:.2f`, SD = {glue:text}`ds000030_ADHD_sd_qc:.2f`, n = {glue:text}`ds000030_ADHD_n_qc:i`;
t({glue:text}`ds000030_df_ADHD_qc:.2f`) = {glue:text}`ds000030_t_ADHD_qc:.2f`, p = {glue:text}`ds000030_p_ADHD_qc:.3f`),
or the bipolar group 
(M = {glue:text}`ds000030_bipolar_mean_qc:.2f`, SD = {glue:text}`ds000030_bipolar_sd_qc:.2f`, n = {glue:text}`ds000030_bipolar_n_qc:i`;
t({glue:text}`ds000030_df_bipolar_qc:.2f`) = {glue:text}`ds000030_t_bipolar_qc:.2f`, p = {glue:text}`ds000030_p_bipolar_qc:.3f`).
In conclusion, adult samples have lower mean framewise displacement than youth samples.

```{code-cell} ipython3
:tags: [hide-input, remove-output]

def significant_notation(item_pairs, max_value, sig, ax):
    x1, x2 = item_pairs
    y, h, col = max_value + 0.01, 0.01, 'k'
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    ax.text((x1+x2)*.5, y+h, sig, ha='center', va='bottom', color=col)

datasets = ['ds000228', 'ds000030']
for dataset in datasets:
    _, data, _ = tables.get_descriptive_data(dataset, fmriprep_version, path_root, **stringent)
    for_plotting.update({dataset: data})

fig = plt.figure(figsize=(7, 5))
axs = fig.subplots(1, 2, sharey=True)
for dataset, ax in zip(for_plotting, axs):
    df = for_plotting[dataset]
    mean_fd = df['mean_framewise_displacement'].mean()
    sd_fd = df['mean_framewise_displacement'].std()
    df = df.rename(
        columns={
            'mean_framewise_displacement': 'Mean Framewise Displacement (mm)',
            'groups': 'Groups'
        }
    )
    sns.boxplot(
        y='Mean Framewise Displacement (mm)', x='Groups', data=df, ax=ax,
        order=group_order[dataset]
    )
    ax.set_xticklabels(group_order[dataset], rotation=45, ha='right', rotation_mode='anchor')
    ax.set_title(
        f'{dataset}\nMean\u00B1SD={mean_fd:.2f}\u00B1{sd_fd:.2f}\n$N={df.shape[0]}$'
    )

    # statistical annotation
    max_value = df['Mean Framewise Displacement (mm)'].max()
    if dataset == 'ds000228':
        significant_notation((0, 1), max_value, "*", ax)
    if dataset == 'ds000030':
        significant_notation((0, 3), max_value, "**", ax)

# fig.suptitle("Mean framewise displacement per sub-sample")

glue('meanFD-stringent', fig, display=False)
```

```{glue:figure} meanFD-stringent
:name: "fig:meanFD-stringent"

Mean framewise displacement of each dataset after excluding subjects with high motion.
```

```{code-cell} ipython3
:tags: [hide-input, remove-output]

fig = plt.figure(constrained_layout=True, figsize=(11, 5))
axs = fig.subplots(1, 2, sharey=True)
for ax, dataset in zip(axs, datasets):
    (
        confounds_phenotype,
        participant_groups,
        groups,
    ) = utils._get_participants_groups(
        dataset,
        fmriprep_version,
        path_root,
        gross_fd=stringent['gross_fd'],
        fd_thresh=stringent['fd_thresh'],
        proportion_thresh=stringent['proportion_thresh'],
    )

    # change up the data a bit for plotting
    confounds_phenotype.loc[:, ('aroma', 'aroma')] += confounds_phenotype.loc[:, ('aroma', 'fixed_regressors')]
    confounds_phenotype.loc[:, ('aroma+gsr', 'aroma')] += confounds_phenotype.loc[:, ('aroma+gsr', 'fixed_regressors')]
    confounds_phenotype.loc[:, ('compcor', 'compcor')] += confounds_phenotype.loc[:, ('compcor', 'fixed_regressors')]
    confounds_phenotype.loc[:, ('compcor6', 'compcor')] += confounds_phenotype.loc[:, ('compcor6', 'fixed_regressors')]

    confounds_phenotype = confounds_phenotype.reset_index()
    confounds_phenotype = confounds_phenotype.melt(
        id_vars=['index'],
        var_name=['strategy', 'type'],
    )
    sns.barplot(
        x='value',
        y='strategy',
        data=confounds_phenotype[confounds_phenotype['type'] == 'total'],
        ci=95,
        color='red',
        linewidth=1,
        ax=ax,
    )
    sns.barplot(
        x='value',
        y='strategy',
        data=confounds_phenotype[confounds_phenotype['type'] == 'compcor'],
        ci=95,
        color='blue',
        linewidth=1,
        ax=ax,
    )
    sns.barplot(
        x='value',
        y='strategy',
        data=confounds_phenotype[confounds_phenotype['type'] == 'aroma'],
        ci=95,
        color='orange',
        linewidth=1,
        ax=ax,
    )
    sns.barplot(
        x='value',
        y='strategy',
        data=confounds_phenotype[
            confounds_phenotype['type'] == 'fixed_regressors'
        ],
        ci=95,
        color='darkgrey',
        linewidth=1,
        ax=ax,
    )
    sns.barplot(
        x='value',
        y='strategy',
        data=confounds_phenotype[
            confounds_phenotype['type'] == 'high_pass'
        ],
        ci=95,
        color='grey',
        linewidth=1,
        ax=ax,
    )
    ax.set_xlim(0, 120)
    ax.set_xlabel('Degrees of freedom loss')
    ax.set_title(dataset)

colors = ['red', 'blue', 'orange', 'darkgrey', 'grey']
labels = [
    'Censored volumes',
    'CompCor \nregressors',
    'ICA-AROMA \npartial regressors',
    'Head motion and \ntissue signal',
    'Discrete cosine-basis \nregressors',
]
handles = [
    mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)
]
axs[1].legend(handles=handles, bbox_to_anchor=(1.7, 1))

glue(f'dof-fig_cleaned', fig, display=False)
```

```{glue:figure} dof-fig_cleaned
:name: "fig:dof-fig_cleaned"

Loss in temporal degrees of freedom break down by groups after quality control,
after applying the stringent quality control threshold.
```

The common analysis and denoising methods are based on linear regression.
Using more nuisance regressors can capture additional sources of noise-related variance in the data and thus improve denoising.
However, this comes at the expense of a loss of temporal degrees of freedom for statistical inference in further analysis.
This is an important point to consider alongside the denoising performance.

The average loss in temporal degrees of freedom by regressor number is summarised in {numref}`fig:dof-fig_cleaned`.
In fMRIPrep, high-pass filtering is done through discrete cosine-basis regressors, 
labeled as `cosine_*` in fMRIPrep confounds output.
In the following section, the number of discrete cosine-basis regressors will be denoted as $c$. 
Depending on the length of the scan, the number of discrete cosine-basis regressors can differ ($c_{ds000228}=4$, $c_{ds000030}=3$). 
The `simple` and `scrubbing`-based strategies are the strategy with a fixed number of degrees of freedom loss.
`compcor` and `aroma`-based strategies show variability depending on the number of noise components detected.
In theory, `compcor6` should also report a fixed number of degrees of freedom loss.
However, fMRIPrep outputs the compcor components based on the 50% variance cut-off.
For some subjects the number of components could be lower than 6, hence the variability seen in the supplemental figure {numref}`fig:dof-fig`.

In {cite:t}`ciric_benchmarking_2017`, the equivalent `aroma` and `aroma+gsr` strategies were reported with 
a lower magnitude of loss in temporal degrees of freedom than `scrubbing` or `simple` strategies.
However, we did not observe this advantage is limited to samples with relatively low motion (i.e. adults).
When selecting a denoising strategy, 
The two datasets used in the current benchmark both contained subjects with behaviours deviating from the healthy controls.
`ds000228` consists of adult healthy controls and children.
`ds000030` includes healthy controls and subjects with three different psychiatric conditions.
the loss in degrees of freedom `simple` ($26 + c$) and `simple+gsr` ($27 + c$) used the least amount of regressors in the general population.
Certain sub-sample uses fewer regressors with the `aroma` and `aroma+gsr` strategies.
The reason potentially lies in the implementation of ICA-AROMA. 
ICA-AROMA uses a pre-trained model on healthy subjects to select noise components {cite:p}`aroma`.


To compare the loss in the number of volumes from the scrubbing base strategy across datasets,
we calculate the proportion of volume loss to the number of volumes in a full scan ({numref}`fig:scrubbing-fig_cleaned`).
`ds000228` includes child subjects and shows a higher loss in volumes compared to `ds000030` with adult subjects only.
This is consistent with the trend in the difference in mean framewise displacement,
and it fits the observation shown in literature {cite:p}`satterthwaite_impact_2012`.
In `ds000030`, we see a similar trend mirroring the mean framewise displacement results.
The schizophrenia group shows the highest amount of volumes scrubbed,
followed by the bipolar group, and comparable results between the control group and ADHD group.
With a stringent 0.2 mm threshold, groups with high motion will lose on average close to half of the volumes. 

In the next section, we report the three functional connectivity-based metrics and break down the effect on each dataset.
We combined all atlases in the current report as the trends in each atlas are similar.
Due to the imbalanced samples per group and low number of subjects in certain groups after the automatic quality control, 
we also collasped all groups within each dataset to avoid speculation on underpowered sample.
For a breakdown of each metric by atlas, 
please see the supplemental material for 
[`ds000228`](../supplementary_materials/report_ds000228.md) and [`ds000030`](../supplementary_materials/report_ds000030.md).

### QC-FC

```{code-cell} ipython3
:tags: [hide-input, hide-output]

path_ds000228 = path_root / "ds000228_fmriprep-20-2-1lts_summary.tsv"
path_ds000030 =  path_root / "ds000030_fmriprep-20-2-1lts_summary.tsv"
ds000228 = pd.read_csv(path_ds000228, sep='\t', index_col=[0, 1], header=[0, 1])
ds000030  = pd.read_csv(path_ds000030, sep='\t', index_col=[0, 1], header=[0, 1])

data = pd.concat({'ds000228': ds000228, 'ds000030':ds000030}, names=['datasets'])
id_vars = data.index.names

# Plotting
data_long = data['qcfc_fdr_significant'].reset_index().melt(id_vars=id_vars, value_name='Percentage %')
data_long = data_long.set_index(keys=['datasets'])
fig = plt.figure(figsize=(11, 5))
axs = fig.subplots(1, 2, sharey=True)
for dataset, ax in zip(['ds000228', 'ds000030'], axs):
    df = data_long.loc[dataset, :]
    sns.barplot(
        y='Percentage %', x='strategy', data=df, ax=ax,
        order=strategy_order, ci=None,
        # hue_order=['full_sample']
        hue_order=group_order[dataset]
    )
    sns.stripplot(y='Percentage %', x='strategy', data=df, ax=ax, 
                  order=strategy_order, hue_order=group_order[dataset])
    ax.set_xticklabels(strategy_order, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_title(f'dataset-{dataset}')
    # Improve the legend
    # handles, labels = ax.get_legend_handles_labels()
    # lgd_idx = len(group_order[dataset])
    # ax.legend(handles[lgd_idx:], labels[lgd_idx:])

glue('qcfc_fdr_significant', fig, display=False)


data_long = data['qcfc_mad'].reset_index().melt(id_vars=id_vars, value_name='Median absolute deviation')
data_long = data_long.set_index(keys=['datasets'])
fig = plt.figure(figsize=(13, 5))
axs = fig.subplots(1, 2, sharey=True)
for dataset, ax in zip(['ds000228', 'ds000030'], axs):
    df = data_long.loc[dataset, :]
    sns.barplot(
        y='Median absolute deviation', x='strategy', data=df, ax=ax,
        order=strategy_order, ci='sd',
        # hue_order=['full_sample']
        hue_order=group_order[dataset]
    )
    ax.set_xticklabels(strategy_order, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_title(f'dataset-{dataset}')
glue('qcfc_mad', fig, display=False)
```

With good quality data, most denoising methods reduce the correlation between functional connectivity and mean framewise displacement,
accessed by the QC-FC measure.
`ds000030` consists of the adult sample only ({numref}`fig:qcfc_fdr_significant`).
All denoising strategies aside from `aroma+gsr` eliminate the impact of motion.
The variability in the healthy control is potentially driven by a larger sample than the rest.
When looking at the median absolute deviations, the schizophrenia group still retains a higher impact of motion than the remaining sample.
In `ds000228`, all strategies, including the baseline, 
show motion remains in close to 0% of the connectivity edges.
`aroma+gsr` performs worse than the baseline in the child sample.
The median absolute deviation of QC-FC is all similar to the baseline ({numref}`fig:qcfc_mad`). 

```{glue:figure} qcfc_fdr_significant
:name: "fig:qcfc_fdr_significant"

Percentage of edges significantly correlating with mean framewise displacement.

The siginificnant test results reported here are false-discovery-rate corrected, summarised across all atlas of choices.
The bar indicates the average percentage of nodes with significant QC-FC, 
the dots are the percentage of nodes with significant QC-FC of each parcellation scheme.
```

```{glue:figure} qcfc_mad
:name: "fig:qcfc_mad"

Median absolute deviation of the correlations between connectivity edges and mean framewise displacement, summarised across all atlas of choices.

A lower value indicates the less residual effect of motion after denoising.
The bar indicates the average median absolute deviation of QC-FC,
the error bars represent the standard deviations. 
```


### Distance-dependent of motion after denoising

```{code-cell} ipython3
:tags: [hide-input, remove-output]

data_long = data['corr_motion_distance'].reset_index().melt(id_vars=id_vars, value_name='Pearson\'s correlation')
data_long = data_long.set_index(keys=['datasets'])
fig = plt.figure(figsize=(13, 5))
axs = fig.subplots(1, 2, sharey=True)
for dataset, ax in zip(['ds000228', 'ds000030'], axs):
    df = data_long.loc[dataset, :]
    sns.barplot(
        y='Pearson\'s correlation', x='strategy', data=df, ax=ax,
        order=strategy_order, ci='sd', 
        # hue_order=['full_sample']
        hue_order=group_order[dataset]
    )
    ax.set_xticklabels(strategy_order, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_title(f'dataset-{dataset}')

glue('distance_qcfc', fig, display=False)
```

Ideally, a denoising strategy should leave no residual association between QC-FC and interregional distance ({numref}`fig:distance_qcfc`).
No strategy can eliminate the correlation between motion and short-range connectivity edges.
In both datasets, we see all strategies reduce the magnitude of the correlation between distance and motion compared to the baseline.
We observed a trend that strategies `scrubbing.2` and `scrubbing.2+gsr` are the closest in reducing the correlation to 0 between distance and motion, followed by `aroma` and `aroma+gsr`.
This trend is similar to the results reported in {cite:t}`ciric_benchmarking_2017`.

```{glue:figure} distance_qcfc
:name: "fig:distance_qcfc"

Distance-dependent of motion after denoising.

A value closer to zero indicates the less residual effect of motion after denoising.
The bar indicates the average Pearson's correlation between the Euclidean distance between node pairs and QC-FC,
the error bars represent its standard deviations. 
```

### Network modularity

```{code-cell} ipython3
:tags: [hide-input, remove-output]

data_long = data['modularity'].reset_index().melt(id_vars=id_vars, value_name='Mean modularity quality (a.u.)')
data_long = data_long.set_index(keys=['datasets'])
fig = plt.figure(figsize=(13, 5))
axs = fig.subplots(1, 2, sharey=True)
for dataset, ax in zip(['ds000228', 'ds000030'], axs):
    df = data_long.loc[dataset, :]
    sns.barplot(
        y='Mean modularity quality (a.u.)', x='strategy', data=df, ax=ax,
        order=strategy_order, ci='sd', 
        # hue_order=['full_sample']
        hue_order=group_order[dataset]
    )
    ax.set_xticklabels(strategy_order, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_title(f'dataset-{dataset}')

glue('modularity', fig, display=False)

data_long = data['corr_motion_modularity'].reset_index().melt(id_vars=id_vars, value_name='Pearson\'s correlation')
data_long = data_long.set_index(keys=['datasets'])
fig = plt.figure(figsize=(13, 5))
axs = fig.subplots(1, 2, sharey=True)
for dataset, ax in zip(['ds000228', 'ds000030'], axs):
    df = data_long.loc[dataset, :]
    sns.barplot(
        y='Pearson\'s correlation', x='strategy', data=df, ax=ax,
        order=strategy_order, ci='sd',
        # hue_order=['full_sample']
        hue_order=group_order[dataset]
    )
    ax.set_xticklabels(strategy_order, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_title(f'dataset-{dataset}')
glue('corr_motion_modularity', fig, display=False)
```

The average network modularity after denoising ({numref}`fig:modularity`) shows that the inclusion of global signal regressors increases the modularity in both datasets. 
The correlation between motion and network modularity is less conclusive ({numref}`fig:corr_motion_modularity`).
In `ds000228`, we first see the big differences between the adult and child samples.
Generally, the denoising strategies reduced the correlation motion and network modularity more in the adult sample than in the child sample.  
In both samples, `aroma` reduced the correlation the most, followed by the baseline and the `simple` strategy.
In `ds000030`, the schizophrenia sample still showed a high correlation between modularity and motion.
For the control group, `aroma`, `aroma+gsr`, `compcor6`, and `simple` all bring the correlation between modularity and motion close to 0.
The baseline along performs better than the remainders.
For ADHD and bipolar group, `compcor` was the best performing strategy and performed better than baseline.
`aroma` was the second best overall, however, it performed on a similar level compared to the baseline.


```{glue:figure} modularity
:name: "fig:modularity"

Average Louvain network modularity after denoising.

The bar indicates the average Louvain network modularity of all connectomes,
the error bars represent its standard deviations. 
In both datasets, strategies including the global signal regressor(s) have higher modularity values.
```

```{glue:figure} corr_motion_modularity
:name: "fig:corr_motion_modularity"

Correlation between mean framewise displacement and Louvain network modularity after denoising.

A value closer to zero indicates the less residual effect of motion after denoising.
The bar indicates the average Pearson's correlation between mean framewise displacement and Louvain network modularity,
the error bars represent its standard deviations. 
```

### Similarity of the denoised connectomes

```{code-cell} ipython3
:tags: [hide-input, remove-output]

import matplotlib as mpl

cmap = mpl.cm.viridis

fig = plt.figure(figsize=(11, 5))
subfigs = fig.subfigures(1, 2, width_ratios=[12, 1])
axes = subfigs[0].subplots(1, 2, sharex=False, sharey=False)
subfigs[0].subplots_adjust(wspace=0.3)
for ds, subfig in zip(datasets, axes):
    cc = pd.read_csv(path_root / ds / fmriprep_version / f'dataset-{ds}_atlas-mist_nroi-444_connectome.tsv', sep='\t', index_col=0)
    g = plot_matrix(cc.corr().values, labels=list(cc.columns), colorbar=False, axes=subfig, cmap=cmap,
                    title=ds, reorder='complete', vmax=1, vmin=0.7)
# make colorbar
ax = subfigs[1].subplots(1, 1, sharex=False, sharey=False)
norm = mpl.colors.Normalize(vmin=0.7, vmax=1)
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
cb1.set_label('Pearson\' correlation')
subfigs[-1].subplots_adjust(right=0.2)
glue('similarity_denoised_connectomes', fig, display=False)
```

```{glue:figure} similarity_denoised_connectomes
:name: "fig:similarity_denoised_connectomes"

Similarity of the connectome generated from different denoise strategies: Using MIST atlas as an example.
```

```{code-cell} ipython3
meta_2021 = pd.read_csv(path_root / 'ds000030_fmriprep-20-2-1lts_summary.tsv', sep='\t', index_col=[0, 1], header=[0, 1])
measures = meta_2021.columns.levels[0]
for m in measures:
    meta_2021 = pd.read_csv(path_root / 'ds000030_fmriprep-20-2-1lts_summary.tsv', sep='\t', index_col=[0, 1], header=[0, 1]).loc['full_sample', m]
    meta_2025 = pd.read_csv(path_root / 'ds000030_fmriprep-20-2-5lts_summary.tsv', sep='\t', index_col=[0, 1], header=[0, 1]).loc['full_sample', m]

    sorted_atlas = sorted(meta_2021.columns)
    meta_2021 = meta_2021.reindex(sorted_atlas, axis=1)
    meta_2025 = meta_2025.reindex(sorted_atlas, axis=1)

    corr = pd.DataFrame(np.corrcoef(meta_2021, meta_2025)[11:, :11], index=meta_2025.index, columns=meta_2021.index)
    plt.figure()
    plt.imshow(corr, vmax=1, vmin=0)
    plt.xlabel('20.2.1')
    plt.ylabel('20.2.5')
    plt.title(m)

    plt.xticks(range(11), corr.columns, rotation=90)
    plt.yticks(range(11), corr.index)
    plt.colorbar()
plt.show()
```
