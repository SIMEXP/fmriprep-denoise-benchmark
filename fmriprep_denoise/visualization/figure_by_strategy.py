from statistics import mean
from tkinter import Menu
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, spearmanr

from fmriprep_denoise.features import (fdr,
                                       calculate_median_absolute,
                                       get_atlas_pairwise_distance,
                                       partial_correlation)

from fmriprep_denoise.visualization import utils, tables
from fmriprep_denoise.features.derivatives import get_qc_criteria


dataset = 'ds000030'
atlas_name = None
dimension = None
path_root = utils.get_data_root()
qc = get_qc_criteria('stringent')


# QCFC derivatives

file_path, labels = utils._get_connectome_metric_paths(
    dataset, 'qcfc', atlas_name, dimension, path_root,
)

# significant correlation between motion and edges

long_qcfc_sig = []
metric = 'pvalue'
for p, label in zip(file_path, labels):

    qcfc_stats = pd.read_csv(p, sep='\t', index_col=0, header=[0, 1])
    groups = qcfc_stats.columns.levels[0].tolist()
    groups.remove('full_sample')
    qcfc_stats = qcfc_stats[groups]

    df = qcfc_stats.filter(regex=metric)
    new_col = pd.MultiIndex.from_tuples(
        [(group, strategy.replace(f'_{metric}', '')) for group, strategy in df.columns],
        names=['groups', 'strategy'])
    df.columns = new_col

    df = df.melt(var_name=['groups', 'strategy'])
    df['fdr'] = df.groupby(['groups', 'strategy'])['value'].transform(fdr)
    df = df.groupby(['groups', 'strategy']).apply(
        lambda x: 100 * x.fdr.sum() / x.fdr.shape[0]
    )
    df = pd.DataFrame(df, columns=[label])
    long_qcfc_sig.append(df)
long_qcfc_sig = pd.concat(long_qcfc_sig, axis=1).reset_index()
long_qcfc_sig = long_qcfc_sig.melt(id_vars=['groups', 'strategy'])


sns.barplot(x='value', y='strategy', hue='groups', data=long_qcfc_sig)
plt.show()


# median absolute
metric = 'correlation'

qcfc_median_absolute = []
for p, label in zip(file_path, labels):

    qcfc_stats = pd.read_csv(p, sep='\t', index_col=0, header=[0, 1])
    groups = qcfc_stats.columns.levels[0].tolist()
    groups.remove('full_sample')
    qcfc_stats = qcfc_stats[groups]

    df = qcfc_stats.filter(regex=metric)
    new_col = pd.MultiIndex.from_tuples(
        [(group, strategy.replace(f'_{metric}', '')) for group, strategy in df.columns],
        names=['groups', 'strategy'])
    df.columns = new_col

    df = df.apply(calculate_median_absolute)
    df.name = label
    qcfc_median_absolute.append(df)

qcfc_median_absolute = pd.concat(qcfc_median_absolute, axis=1).reset_index()
qcfc_median_absolute = qcfc_median_absolute.melt(id_vars=['groups', 'strategy'])

sns.boxplot(x='value', y='strategy', hue='groups', data=qcfc_median_absolute)
plt.show()

# distance dependency

corr_distance = []
for p, label in zip(file_path, labels):

    qcfc_stats = pd.read_csv(p, sep='\t', index_col=0, header=[0, 1])
    groups = qcfc_stats.columns.levels[0].tolist()
    groups.remove('full_sample')
    qcfc_stats = qcfc_stats[groups]

    df = qcfc_stats.filter(regex=metric)
    new_col = pd.MultiIndex.from_tuples(
        [(group, strategy.replace(f'_{metric}', '')) for group, strategy in df.columns],
        names=['groups', 'strategy'])
    df.columns = new_col

    atlas_name = label.split('atlas-')[-1].split('_')[0]
    dimension = label.split('nroi-')[-1].split('_')[0]
    pairwise_distance = get_atlas_pairwise_distance(atlas_name, dimension)
    cols = df.columns
    df, _ = spearmanr(pairwise_distance.iloc[:, -1], df)
    df = pd.DataFrame(df[1:, 0], index=cols, columns=[label])
    corr_distance.append(df)

corr_distance = pd.concat(corr_distance, axis=1).reset_index()
corr_distance = corr_distance.melt(id_vars=['groups', 'strategy'])

sns.boxplot(x='value', y='strategy', hue='groups', data=corr_distance)
plt.show()


# network modularity

files_network, labels = utils._get_connectome_metric_paths(
    dataset, 'modularity', atlas_name, dimension, path_root,
)
_, movement, _ = tables.get_descriptive_data(dataset, path_root, **qc)

mean_corr, mean_modularity = [], []
for file_network, label in zip(files_network, labels):

    # modularity
    modularity = pd.read_csv(file_network, sep='\t', index_col=0)
    modularity = pd.concat([movement['groups'], modularity],axis=1)
    mean_by_group = modularity.groupby(['groups']).mean().reset_index()
    mean_by_group = mean_by_group.melt(id_vars=['groups'], var_name="strategy", value_name=label)
    mean_by_group = mean_by_group.set_index(['groups', 'strategy'])
    mean_modularity.append(mean_by_group)

    # motion and modularity
    corr_modularity = []
    z_movement = movement[['mean_framewise_displacement', 'age', 'gender']].apply(zscore)
    modularity = modularity.reset_index().melt(id_vars=['index', 'groups'], var_name='strategy')

    for (group, strategy), df in modularity.groupby(['groups', 'strategy']):
        df = df.set_index('index')
        current_df = partial_correlation(
            df['value'],
            movement.loc[df.index, 'mean_framewise_displacement'].values,
            z_movement.loc[df.index, ['age', 'gender']].values,
        )
        current_df['strategy'] = strategy
        current_df['groups'] = group
        corr_modularity.append(current_df)

    corr_modularity = pd.DataFrame(corr_modularity).set_index(
        ['groups', 'strategy']
    )['correlation']
    corr_modularity.name = label
    mean_corr.append(corr_modularity)

# modularity
mean_modularity = pd.concat(mean_modularity, axis=1)

# motion and modularit
mean_corr = pd.concat(mean_corr, axis=1)

