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
import numpy as np

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

```{code-cell} ipython3
import pandas as pd
from fmriprep_denoise.visualization import degrees_of_freedom_loss, motion_metrics
from fmriprep_denoise.visualization import utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

strategy_order = list(utils.GRID_LOCATION.values())
fmriprep_versions = ['fmriprep-20.2.1lts', 'fmriprep-20.2.5lts']


def load_data(path_root, datasets, criteria_name='stringent'):
    mean_total_ranking = []
    for v in fmriprep_versions:        
        dof = degrees_of_freedom_loss.load_data(
            path_root, datasets, criteria_name, v)
        for d in dof:
            df_data = pd.DataFrame()
            current_ranking = {}
            for s in strategy_order:
                m = dof[d].loc[:, (s, 'total')].mean()
                current_ranking[s] = [m]
            order = pd.DataFrame(current_ranking).T.sort_values(0)
            order.index = order.index.set_names(['strategy'])
            order = order.reset_index()
            order['version'] = v
            order['dataset'] = d
            order['loss_df'] = list(range(1, order.shape[0] + 1))
            order = order.drop(0, axis=1)
            df_data = pd.concat([df_data, order], axis=0)
            df_data = df_data.set_index(['strategy', 'version', 'dataset'])

            metrics = pd.DataFrame()
            for m in ['p_values', 'median', 'distance', 'modularity']:
                data, measure = motion_metrics.load_data(
                    path_root, datasets, criteria_name, v, measure_name=m)
                if m =='modularity':
                    ascending = False
                else:
                    ascending = True
                r = data[d].groupby('strategy')[measure['label']].describe()['mean'].sort_values(ascending=ascending)
                rk = pd.DataFrame(list(range(1, 12)), index=r.index, columns=[m])
                rk = rk.reset_index()
                rk['version'] = v
                rk['dataset'] = d
                rk = rk.set_index(['strategy', 'version', 'dataset'])
                metrics = pd.concat([metrics, rk], axis=1)
            df_data = pd.concat([df_data, metrics], axis=1)
            mean_total_ranking.append(df_data)
    mean_total_ranking = pd.concat(mean_total_ranking).reset_index()
    mean_total_ranking = mean_total_ranking.melt(
        id_vars=['strategy', 'version', 'dataset'], 
        var_name='metrics', 
        value_name='ranking')
    return pd.pivot_table(mean_total_ranking, columns='strategy', index=['dataset', 'version', 'metrics'], fill_value='ranking')
```

```{code-cell} ipython3
m = load_data(path_root, datasets)
```

```{code-cell} ipython3
fig, axs = plt.subplots(2, 2, figsize=(11, 4.8), sharex=True, sharey=True, constrained_layout=True)
fig.suptitle(
    "Ranking of all strategies per dataset per fMRIPrep version",
    weight="heavy",
    fontsize="x-large",
)
for i, d in enumerate(datasets):
    for j, v in enumerate(fmriprep_versions):
        mat = m.xs(d, level='dataset', drop_level=True)
        mat = mat.xs(v, level='version', drop_level=True)
        mat = mat.droplevel(None, axis=1)
        mat = mat.loc[['modularity', 'distance', 'median', 'p_values'], strategy_order]

        x, y = np.meshgrid(np.arange(mat.shape[1]), np.arange(mat.shape[0]))

        R = (12 - mat) / 12 / 2
        circles = [plt.Circle((j,i), radius=r) for r, j, i in zip(R.values.flat, x.flat, y.flat)]
        col = PatchCollection(circles, array=mat.values.flatten(), cmap="cividis_r")
        axs[i, j].add_collection(col)

        axs[i, j].set(xticks=np.arange(mat.shape[1]), yticks=np.arange(mat.shape[0]),
               xticklabels=mat.columns, 
               yticklabels=['Average network modularity', 'DM-FC', 'QC-FC: median', 'QC-FC: significant']
              )
        axs[i, j].set_xticklabels(mat.columns, rotation=45, ha='right')
        axs[i, j].set_xticks(np.arange(mat.shape[1] + 1)-0.5, minor=True)
        axs[i, j].set_yticks(np.arange(mat.shape[0] + 1)-0.5, minor=True)
        axs[i, j].grid(which='minor')
        axs[i, j].set_title(f"{d}: {v}")

fig.colorbar(col)
plt.show()
```

# Results: dataset level

Here we provides alternative visualisation of the benchmark results from the manuscript.
The Jupyter Book can only display static images. 
Please click on the launch botton to lunch the binder instance for interactive data viewing.

## Sample and subgroup size change based on quality control criteria

```{code-cell} ipython3
:tags: [hide-input]

def demographic_table(criteria_name, fmriprep_version):
    criteria = get_qc_criteria(criteria_name)
    ds000228 = tables.lazy_demographic('ds000228', fmriprep_version, path_root, **criteria)
    ds000030 = tables.lazy_demographic('ds000030', fmriprep_version, path_root, **criteria)

    desc = pd.concat({'ds000228': ds000228, 'ds000030': ds000030}, axis=1, names=['dataset'])
    desc = desc.style.set_table_attributes('style="font-size: 12px"')
    print("Generating new tables...")

    display(desc)
    

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
:tags: [hide-input]

from fmriprep_denoise.visualization import mean_framewise_displacement


def notebook_plot_mean_fd(criteria_name, fmriprep_version):
    stats = mean_framewise_displacement.load_data(path_root, criteria_name, fmriprep_version) 
    mean_framewise_displacement.plot_stats(stats)

    
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
interactive(notebook_plot_mean_fd, criteria_name=criteria_name, fmriprep_version=fmriprep_version)
```

```{code-cell} ipython3
from fmriprep_denoise.visualization import connectivity_similarity

def notebook_plot_connectomes(fmriprep_version):
    average_connectomes = connectivity_similarity.load_data(path_root, datasets, fmriprep_version) 
    connectivity_similarity.plot_stats(average_connectomes)


fmriprep_version = widgets.Dropdown(
    options=['fmriprep-20.2.1lts', 'fmriprep-20.2.5lts'],
    value='fmriprep-20.2.1lts',
    description='Preporcessing version : ',
    disabled=False
)
interactive(notebook_plot_connectomes, fmriprep_version=fmriprep_version)
```

## Loss of degrees of freedoms

```{code-cell} ipython3
:tags: [hide-input]

from fmriprep_denoise.visualization import degrees_of_freedom_loss

def notebook_plot_loss_degrees_of_freedom(criteria_name, fmriprep_version):
    data = degrees_of_freedom_loss.load_data(path_root, datasets, criteria_name, fmriprep_version) 
    degrees_of_freedom_loss.plot_stats(data)

    
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

interactive(notebook_plot_loss_degrees_of_freedom, criteria_name=criteria_name, fmriprep_version=fmriprep_version)
```

```{code-cell} ipython3
from fmriprep_denoise.visualization import motion_metrics

def notebook_plot_qcfc(criteria_name, fmriprep_version):
    data, measure = motion_metrics.load_data(path_root, datasets, criteria_name, fmriprep_version, 'p_values')
    motion_metrics.plot_stats(data, measure)
    
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

interactive(notebook_plot_qcfc, criteria_name=criteria_name, fmriprep_version=fmriprep_version)
```

```{code-cell} ipython3
from fmriprep_denoise.visualization import motion_metrics

def notebook_plot_qcfc_fdr(criteria_name, fmriprep_version):
    data, measure = motion_metrics.load_data(path_root, datasets, criteria_name, fmriprep_version, 'fdr_p_values')
    motion_metrics.plot_stats(data, measure)

    
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

interactive(notebook_plot_qcfc_fdr, criteria_name=criteria_name, fmriprep_version=fmriprep_version)
```

```{code-cell} ipython3
from fmriprep_denoise.visualization import motion_metrics

def notebook_plot_qcfc_median(criteria_name, fmriprep_version):
    data, measure = motion_metrics.load_data(path_root, datasets, criteria_name, fmriprep_version, 'median')
    motion_metrics.plot_stats(data, measure)

    
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

interactive(notebook_plot_qcfc_median, criteria_name=criteria_name, fmriprep_version=fmriprep_version)
```

```{code-cell} ipython3
from fmriprep_denoise.visualization import motion_metrics

def notebook_plot_distance(criteria_name, fmriprep_version):
    data, measure = motion_metrics.load_data(path_root, datasets, criteria_name, fmriprep_version, 'distance')
    motion_metrics.plot_stats(data, measure)

    
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

interactive(notebook_plot_distance, criteria_name=criteria_name, fmriprep_version=fmriprep_version)
```

```{code-cell} ipython3
from fmriprep_denoise.visualization import motion_metrics

def notebook_plot_modularity(criteria_name, fmriprep_version):
    data, measure = motion_metrics.load_data(path_root, datasets, criteria_name, fmriprep_version, 'modularity')
    motion_metrics.plot_stats(data, measure)

    
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

interactive(notebook_plot_modularity, criteria_name=criteria_name, fmriprep_version=fmriprep_version)
```

```{code-cell} ipython3
from fmriprep_denoise.visualization import motion_metrics

def notebook_plot_modularity_motion(criteria_name, fmriprep_version):
    data, measure = motion_metrics.load_data(path_root, datasets, criteria_name, fmriprep_version, 'modularity_motion')
    motion_metrics.plot_stats(data, measure)

    
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

interactive(notebook_plot_modularity_motion, criteria_name=criteria_name, fmriprep_version=fmriprep_version)
```

```{code-cell} ipython3
criteria_name = 'stringent'
dataset = 'ds000228'
fmriprep_version = 'fmriprep-20.2.1lts'
parcel = 'atlas-difumo_nroi-64'
path_data = path_root / dataset/ fmriprep_version/ f"dataset-{dataset}_{parcel}_modularity.tsv"
modularity = pd.read_csv(path_data, sep='\t', index_col=0)
path_data = path_root / dataset/ fmriprep_version/ f"dataset-{dataset}_desc-movement_phenotype.tsv"
motion = pd.read_csv(path_data, sep='\t', index_col=0)
data = pd.concat([modularity, motion.loc[modularity.index, :]], axis=1)

plt.figure()
sns.regplot('baseline', 'mean_framewise_displacement', data=data)
sns.regplot('scrubbing.2', 'mean_framewise_displacement', data=data)
sns.regplot('scrubbing.2+gsr', 'mean_framewise_displacement', data=data)
plt.xlabel('network modularity')
plt.title(dataset + ': scrubbing.2')
palette = sns.color_palette(n_colors=12)
colors = [palette[0], palette[1], palette[2]]
labels = [
    "baseline",
    "scrubbing.2",
    "scrubbing.2+gsr"
]
handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
plt.legend(handles=handles)

plt.figure()
sns.regplot('baseline', 'mean_framewise_displacement', data=data)
sns.regplot('simple', 'mean_framewise_displacement', data=data)
sns.regplot('simple+gsr', 'mean_framewise_displacement', data=data)
plt.xlabel('network modularity')
plt.title(dataset + ': simple')

palette = sns.color_palette(n_colors=12)
colors = [palette[0], palette[1], palette[2]]
labels = [
    "baseline",
    "scrubbing.2",
    "scrubbing.2+gsr"
]
handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
plt.legend(handles=handles)
```

```{code-cell} ipython3
criteria_name = 'stringent'
dataset = 'ds000030'
fmriprep_version = 'fmriprep-20.2.1lts'
parcel = 'atlas-difumo_nroi-64'
path_data = path_root / dataset/ fmriprep_version/ f"dataset-{dataset}_{parcel}_modularity.tsv"
modularity = pd.read_csv(path_data, sep='\t', index_col=0)
path_data = path_root / dataset/ fmriprep_version/ f"dataset-{dataset}_desc-movement_phenotype.tsv"
motion = pd.read_csv(path_data, sep='\t', index_col=0)
data = pd.concat([modularity, motion.loc[modularity.index, :]], axis=1)
plt.figure()
sns.regplot('baseline', 'mean_framewise_displacement', data=data)
sns.regplot('scrubbing.2', 'mean_framewise_displacement', data=data)
sns.regplot('scrubbing.2+gsr', 'mean_framewise_displacement', data=data)
plt.xlabel('network modularity')
plt.title(dataset + ': scrubbing.2')
plt.figure()
sns.regplot('baseline', 'mean_framewise_displacement', data=data)
sns.regplot('simple', 'mean_framewise_displacement', data=data)
sns.regplot('simple+gsr', 'mean_framewise_displacement', data=data)
plt.xlabel('network modularity')
plt.title(dataset + ': simple')
```

```{code-cell} ipython3

```
