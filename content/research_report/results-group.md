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
import matplotlib as mpl

from nilearn.plotting import plot_matrix

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
    print("Generating new tables...")

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
    print("Generating new graphs...")
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

## Benchmark metrics by fMRIPrep version and dataset

We filtered the preprocessed datasets with a stringent gross motion cut-off. 
Here you can brows the results by fMRIPrep version and dataset.

```{code-cell} ipython3
def qcfc(fmriprep_version, dataset):

    criteria_name = 'stringent'
    
    path_data = path_root / f"{dataset}_{fmriprep_version.replace('.', '-')}_desc-{criteria_name}_summary.tsv"
    data = pd.read_csv(path_data, sep='\t', index_col=[0, 1], header=[0, 1])
    id_vars = data.index.names

    # Plotting
    df = data['qcfc_fdr_significant'].reset_index().melt(id_vars=id_vars, value_name='Percentage %')
    fig = plt.figure(figsize=(17, 11))
    axs = fig.subplots(2, 3)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    sns.barplot(
        y='Percentage %', x='strategy', data=df, ax=axs[0, 0],
        order=strategy_order, ci=None,
        hue_order=group_order[dataset]
    )
    sns.stripplot(y='Percentage %', x='strategy', data=df, ax=axs[0, 0], 
                  order=strategy_order, hue_order=group_order[dataset])
    axs[0, 0].set_title('Significant QC/FC in connectomes')
    axs[0, 0].set_ylim(0, 100)


    df = data['qcfc_mad'].reset_index().melt(id_vars=id_vars, value_name='Median absolute deviation')

    sns.barplot(
        y='Median absolute deviation', x='strategy', data=df, ax=axs[0, 1],
        order=strategy_order, ci=95,
        # hue_order=['full_sample']
        hue_order=group_order[dataset]
    )

    axs[0, 1].set_title('Median absolute deviation of QC/FC')
    axs[0, 1].set_ylim(0, 0.25)
    

    df = data['corr_motion_distance'].reset_index().melt(id_vars=id_vars, value_name='Pearson\'s correlation')
    sns.barplot(
        y='Pearson\'s correlation', x='strategy', data=df, ax=axs[0, 2],
        order=strategy_order, ci=95, 
        # hue_order=['full_sample']
        hue_order=group_order[dataset]
    )
    axs[0, 2].set_title('Distance-dependent of motion')

    
    df = data['corr_motion_modularity'].reset_index().melt(id_vars=id_vars, value_name='Pearson\'s correlation')

    sns.barplot(
        y='Pearson\'s correlation', x='strategy', data=df, ax=axs[1, 0],
        order=strategy_order, ci=95, 
        # hue_order=['full_sample']
        hue_order=group_order[dataset]
    )
    axs[1, 0].set_title('Correlation between motion and network modularity')
    
    df = data['modularity'].reset_index().melt(id_vars=id_vars, value_name='Mean modularity quality (a.u.)')

    sns.barplot(
        y='Mean modularity quality (a.u.)', x='strategy', data=df, ax=axs[1, 1],
        order=strategy_order, ci=95, 
        # hue_order=['full_sample']
        hue_order=group_order[dataset]
    )
    axs[1, 1].set_title('Mean network modularity')
    
    cc = pd.read_csv(path_root / dataset / fmriprep_version / f'dataset-{dataset}_atlas-mist_nroi-444_connectome.tsv', 
                     sep='\t', index_col=0)
    plot_matrix(cc.corr().values, labels=list(cc.columns), colorbar=True, axes=axs[1, 2], cmap=mpl.cm.viridis,
                title="Connectome similarity", reorder='complete', vmax=1, vmin=0.7)
    for i in range(2):
        for j in range(2):
            axs[i, j].set_xticklabels(strategy_order, rotation=45, ha='right', rotation_mode='anchor')
    axs[0, 2].set_xticklabels(strategy_order, rotation=45, ha='right', rotation_mode='anchor')
    
    
fmriprep_version = widgets.Dropdown(
    options=['fmriprep-20.2.1lts', 'fmriprep-20.2.5lts'],
    value='fmriprep-20.2.1lts',
    description='Preporcessing version : ',
    disabled=False
)
dataset = widgets.Dropdown(
    options=['ds000228', 'ds000030'],
    value='ds000228',
    description='Dataset : ',
    disabled=False
)

interactive(qcfc, fmriprep_version=fmriprep_version, dataset=dataset)
```

```{code-cell} ipython3

```
