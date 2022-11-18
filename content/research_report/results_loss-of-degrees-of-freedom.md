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

```{code-cell} ipython3
:tags: [hide-input]

import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns

from statsmodels.stats.weightstats import ttest_ind

from fmriprep_denoise.visualization import tables, utils
from fmriprep_denoise.features.derivatives import get_qc_criteria

import ipywidgets as widgets
from ipywidgets import interactive


path_root = utils.get_data_root() / "denoise-metrics"
strategy_order = list(utils.GRID_LOCATION.values())
group_order = {'ds000228': ['adult', 'child'], 'ds000030':['control', 'ADHD', 'bipolar', 'schizophrenia']}
datasets = ['ds000228', 'ds000030']
datasets_baseline = {'ds000228': 'adult', 'ds000030': 'control'}
```

# Results

## Sample and subgroup size change based on quality control criteria

```{code-cell} ipython3
def demographic_table(criteria_name, fmriprep_version):
    criteria = get_qc_criteria(criteria_name)
    ds000228 = tables.lazy_demographic('ds000228', fmriprep_version, path_root, **criteria)
    ds000030 = tables.lazy_demographic('ds000030', fmriprep_version, path_root, **criteria)

    desc = pd.concat({'ds000228': ds000228, 'ds000030': ds000030}, axis=1, names=['dataset'])
    desc = desc.style.set_table_attributes('style="font-size: 12px"')
    display(desc)


def statistic_report(criteria_name, fmriprep_version, dataset):
    criteria = get_qc_criteria(criteria_name)
    for_plotting = {}
    baseline_group = datasets_baseline[dataset]
    _, data, _ = tables.get_descriptive_data(dataset, fmriprep_version, path_root, **criteria)

    for_plotting.update({dataset: data})
    for_plotting.update({'stats': {}})
    baseline = data[data['groups'] == baseline_group]
    for i, group in enumerate(group_order[dataset]):
        compare = data[data['groups'] == group]
        if group != baseline_group:
            t_stats, pval, df = ttest_ind(
                baseline['mean_framewise_displacement'],
                compare['mean_framewise_displacement'],
                usevar='unequal',
            )
            for_plotting['stats'].update(
                {i: {
                    't_stats': t_stats,
                    'p_value': pval, 
                    'df': df}
                })
    return for_plotting


def significant_notation(item_pairs, max_value, sig, ax):
    x1, x2 = item_pairs
    y, h, col = max_value + 0.01, 0.01, 'k'
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    ax.text((x1+x2)*.5, y+h, sig, ha='center', va='bottom', color=col)


def plot_mean_fd(criteria_name, fmriprep_version):
    
    fig = plt.figure(figsize=(7, 5))
    axs = fig.subplots(1, 2, sharey=True)
    for ax, dataset in zip(axs, datasets):
        for_plotting = statistic_report(criteria_name, fmriprep_version, dataset)
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
        for i in for_plotting['stats']:
            if for_plotting['stats'][i]['p_value'] < 0.005:
                notation = "***"
            elif for_plotting['stats'][i]['p_value'] < 0.01:
                notation = "**"
            elif for_plotting['stats'][i]['p_value'] < 0.05:
                notation = "*"
            else:
                notation = None

            if for_plotting['stats'][i]['p_value'] < 0.05:
                significant_notation((0, i), max_value + 0.03 * (i - 1), notation, ax)
```

```{code-cell} ipython3
criteria_name = widgets.Dropdown(
    options=['stringent', 'minimal', None],
    value='stringent',
    description='Threshould: ',
    disabled=False
)
fmriprep_version = widgets.Dropdown(
    options=['fmriprep-20.2.1lts', 'fmriprep-20.2.5lts'],
    value='fmriprep-20.2.1lts',
    description='Preporcessing version : ',
    disabled=False
)
interactive(demographic_table, criteria_name=criteria_name, fmriprep_version=fmriprep_version)
```

```{code-cell} ipython3
criteria_name = widgets.Dropdown(
    options=['stringent', 'minimal', None],
    value='stringent',
    description='Threshould: ',
    disabled=False
)
fmriprep_version = widgets.Dropdown(
    options=['fmriprep-20.2.1lts', 'fmriprep-20.2.5lts'],
    value='fmriprep-20.2.1lts',
    description='Preporcessing version : ',
    disabled=False
)
interactive(plot_mean_fd, criteria_name=criteria_name, fmriprep_version=fmriprep_version)
```

## Loss of degrees of freedoms

```{code-cell} ipython3
:tags: [hide-input]

def loss_degree_of_freedom(criteria_name, fmriprep_version):
    criteria = get_qc_criteria(criteria_name)
    fig = plt.figure(constrained_layout=True, figsize=(11, 5))
    axs = fig.subplots(1, 2, sharey=True)
    print('Generating new graph')
    for ax, dataset in zip(axs, datasets):
        (
            confounds_phenotype,
            participant_groups,
            groups,
        ) = utils._get_participants_groups(
            dataset,
            fmriprep_version,
            path_root,
            gross_fd=criteria['gross_fd'],
            fd_thresh=criteria['fd_thresh'],
            proportion_thresh=criteria['proportion_thresh'],
        )

        # change up the data a bit for plotting
        full_length = confounds_phenotype.iloc[0, -1]
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
        ax.set_xlim(0, full_length)
        ax.set_xlabel(f'Degrees of freedom loss\n(Full length: {full_length})')
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
```

```{code-cell} ipython3
criteria_name = widgets.Dropdown(
    options=['stringent', 'minimal', None],
    value='stringent',
    description='Threshould: ',
    disabled=False
)

fmriprep_version = widgets.Dropdown(
    options=['fmriprep-20.2.1lts', 'fmriprep-20.2.5lts'],
    value='fmriprep-20.2.1lts',
    description='Preporcessing version : ',
    disabled=False
)

interactive(loss_degree_of_freedom, criteria_name=criteria_name, fmriprep_version=fmriprep_version)
```

## QC/FC

```{code-cell} ipython3
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
```

```{code-cell} ipython3

```
