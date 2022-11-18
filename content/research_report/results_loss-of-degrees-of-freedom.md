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
# Loss of degrees of freedoms

```{code-cell} ipython3
:tags: [hide-input]

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns

from fmriprep_denoise.visualization import utils
from fmriprep_denoise.features.derivatives import get_qc_criteria

import ipywidgets as widgets
from ipywidgets import interactive


path_root = utils.get_data_root() / "denoise-metrics"
strategy_order = list(utils.GRID_LOCATION.values())
group_order = {'ds000228': ['adult', 'child'], 'ds000030':['control', 'ADHD', 'bipolar', 'schizophrenia']}
datasets = ['ds000228', 'ds000030']


def loss_degree_of_freedom(criteria_name, fmriprep_version):
    criteria = get_qc_criteria(criteria_name)
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
            gross_fd=criteria['gross_fd'],
            fd_thresh=criteria['fd_thresh'],
            proportion_thresh=criteria['proportion_thresh'],
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
    display(fig)
```

```{code-cell} ipython3

criteria_name = widgets.Select(
    options=['stringent', 'minimal', None],
    value='stringent',
    description='Threshould: ',
    disabled=False
)

fmriprep_version = widgets.Select(
    options=['fmriprep-20.2.1lts', 'fmriprep-20.2.5lts'],
    value='fmriprep-20.2.1lts',
    description='Preporcessing version : ',
    disabled=False
)

interactive(loss_degree_of_freedom, criteria_name=criteria_name, fmriprep_version=fmriprep_version)

```
