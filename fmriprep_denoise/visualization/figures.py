from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, spearmanr

from fmriprep_denoise.features import (partial_correlation, fdr,
                                       calculate_median_absolute,
                                       get_atlas_pairwise_distance)


GRID_LOCATION = {
    (0, 0): 'baseline',
    (0, 2): 'simple',
    (0, 3): 'simple+gsr',
    (1, 0): 'scrubbing.5',
    (1, 1): 'scrubbing.5+gsr',
    (1, 2): 'scrubbing.2',
    (1, 3): 'scrubbing.2+gsr',
    (2, 0): 'compcor',
    (2, 1): 'compcor6',
    (2, 2): 'aroma',
    (2, 3): 'aroma+gsr',
}

path_root = Path.cwd().parents[0] / "inputs"
bar_color = sns.color_palette()[0]


def plot_motion_resid(dataset, atlas_name, dimension):
    file_qcfc = path_root/ f"metrics/dataset-{dataset}_atlas-{atlas_name}_nroi-{dimension}_qcfc.tsv"
    sig_per_edge = _get_qcfc_metric(file_qcfc, metric="pvalue")
    qcfc_per_edge = _get_qcfc_metric(file_qcfc, metric="correlation")
    qcfc_sig = _qcfc_fdr(sig_per_edge)
    qcfc_mad = _get_qcfc_median_absolute(qcfc_per_edge)
    long_qcfc = qcfc_per_edge.melt()
    long_qcfc.columns = ["Strategy", "qcfc"]

    # plotting
    fig = plt.figure(constrained_layout=True, figsize=(13, 5))
    fig.suptitle('Residual effect of motion on connectomes after de-noising', fontsize='xx-large')

    subfigs = fig.subfigures(1, 2)
    axs = subfigs[0].subplots(1, 2, sharey=False)
    for nn, (ax, figure_data) in enumerate(zip(axs, [qcfc_sig, qcfc_mad])):
        sns.barplot(data=figure_data['data'], orient='h',
                    ci=None, order=figure_data['order'],
                    color=bar_color, ax=ax)
        ax.set_title(figure_data['title'])
        ax.set(xlabel=figure_data['label'])
        if nn == 0:
            ax.set(ylabel="Confound removal strategy")

    axs = subfigs[1].subplots(3, 4, sharex=True, sharey=True)
    for i, row_axes in enumerate(axs):
        for j, ax in enumerate(row_axes):
            if cur_strategy := GRID_LOCATION.get((i, j), False):
                mask = long_qcfc["Strategy"] == cur_strategy
                g = sns.kdeplot(data=long_qcfc.loc[mask, :],
                                x='qcfc',
                                fill=True,
                                ax=ax)
                g.set_title(cur_strategy, fontsize='small')
                mad = qcfc_mad['data'][cur_strategy].values
                g.axvline(mad, linewidth=1, linestyle='--', color='r')
                xlabel = "Pearson\'s correlation" if i == 2 else None
                g.set(xlabel=xlabel)
            else:
                subfigs[1].delaxes(axs[i, j])
    subfigs[1].suptitle('Distribution of QC-FC')

    return fig


def plot_distance_dependence(dataset, atlas_name, dimension):
    file_qcfc = path_root / f"metrics/dataset-{dataset}_atlas-{atlas_name}_nroi-{dimension}_qcfc.tsv"

    pairwise_distance = get_atlas_pairwise_distance(atlas_name, dimension)
    qcfc_per_edge = _get_qcfc_metric(file_qcfc, metric="correlation")

    qcfc_dist = _get_corr_distance(pairwise_distance, qcfc_per_edge)
    corr_distance_long = qcfc_per_edge.melt()
    corr_distance_long.columns = ["Strategy", "qcfc"]
    corr_distance_long['distance'] = np.tile(pairwise_distance.iloc[:, -1].values, 11)


    fig = plt.figure(constrained_layout=True, figsize=(9, 5))
    subfigs = fig.subfigures(1, 2, width_ratios=[1, 2])
    fig.suptitle('Residual effect of motion on connectomes after de-noising', fontsize='x-large')

    ax = subfigs[0].subplots(1, 1, sharex=True, sharey=True)

    sns.barplot(data=qcfc_dist['data'], orient='h',
                ci=None, order=qcfc_dist['order'],
                color=bar_color, ax=ax)
    ax.set_title(qcfc_dist['title'])
    ax.set(xlabel=qcfc_dist['label'])
    ax.set(ylabel="Confound removal strategy")

    axs = subfigs[1].subplots(3, 4, sharex=True, sharey=True)
    for i, row_axes in enumerate(axs):
        for j, ax in enumerate(row_axes):
            if cur_strategy := GRID_LOCATION.get((i, j), False):
                mask = corr_distance_long["Strategy"] == cur_strategy
                g = sns.histplot(data=corr_distance_long.loc[mask, :],
                                    x='distance', y='qcfc',
                                    ax=ax)
                ax.set_title(cur_strategy, fontsize='small')
                g.axhline(0, linewidth=1, linestyle='--', alpha=0.5, color='k')
                sns.regplot(data=corr_distance_long.loc[mask, :],
                            x='distance', y='qcfc',
                            ci=None,
                            scatter=False,
                            line_kws={'color': 'r', 'linewidth': 0.5},
                            ax=ax)
                xlabel = "Distance (mm)" if i == 2 else None
                ylabel = "QC-FC" if j == 0 else None
                g.set(xlabel=xlabel, ylabel=ylabel)
            else:
                subfigs[1].delaxes(axs[i, j])
    subfigs[1].suptitle('Correlation between nodewise Euclidian distance and QC-FC')
    fig.suptitle('Distance-dependent effects of motion on connectivity¶', fontsize='xx-large')
    return fig


def plot_network_modularity(dataset, atlas_name, dimension):
    file_network = path_root / f"metrics/dataset-{dataset}_atlas-{atlas_name}_nroi-{dimension}_modularity.tsv"
    file_dataset = path_root / f"dataset-{dataset}/dataset-{dataset}_desc-movement_phenotype.tsv"

    modularity = pd.read_csv(file_network, sep='\t', index_col=0)
    movement = pd.read_csv(file_dataset, sep='\t', index_col=0, header=0, encoding='utf8')

    corr_mod = _corr_modularity_motion(modularity, movement)
    modularity_order = modularity.mean().sort_values(ascending=False).index.tolist()

    network_mod = {
        'data': modularity,
        'order': modularity_order,
        'title': "Identifiability of network structure\nafter denoising",
        'label': "Mean modularity quality (a.u.)",
    }

    # plotting
    fig = plt.figure(constrained_layout=True, figsize=(9, 5))
    axs = fig.subplots(1, 2, sharey=False)
    sns.barplot(data=network_mod['data'],
                orient='h',
                ci=None,
                order=network_mod['order'],
                color=bar_color, ax=axs[0])
    axs[0].set_title(network_mod['title'])
    axs[0].set(xlabel=network_mod['label'])
    axs[0].set(ylabel="Confound removal strategy")

    sns.barplot(data=corr_mod['data'], x='correlation', y='strategy',
                ci=None,
                order=None,
                color=bar_color, ax=axs[1])
    axs[1].set_title(corr_mod['title'])
    axs[1].set(xlabel=corr_mod['label'])

    fig.suptitle('Correlation between\nnetwork modularity and mean framewise displacement', fontsize='xx-large')
    return fig


def _get_qcfc_metric(file_path, metric):
    """ Get correlation or pvalue of QC-FC."""
    qcfc_stats = pd.read_csv(file_path, sep='\t', index_col=0)
    qcfc_per_edge = qcfc_stats.filter(regex=metric)
    qcfc_per_edge.columns = [col.split('_')[0] for col in qcfc_per_edge.columns]
    return qcfc_per_edge


def _get_corr_distance(pairwise_distance, qcfc_per_edge):
    corr_distance, _ = spearmanr(pairwise_distance.iloc[:, -1], qcfc_per_edge)
    corr_distance = pd.DataFrame(corr_distance[1:, 0], index=qcfc_per_edge.columns)
    corr_distance_order = corr_distance.sort_values(0, ascending=False).index.tolist() # needed
    corr_distance = corr_distance.T  # needed
    return {
        'data': corr_distance,
        'order': corr_distance_order,
        'title': "Correlation between\nnodewise distance and QC-FC",
        'label': "Pearson's correlation",
    }


def _corr_modularity_motion(modularity, movement):
    """Correlation between network modularity and  mean framewise displacement."""
    corr_modularity = []
    z_movement = movement.apply(zscore)
    for column, _ in modularity.iteritems():
        cur_data = pd.concat((modularity[column],
                              movement[['mean_framewise_displacement']],
                              z_movement[['age', 'gender']]), axis=1).dropna()
        current_strategy = partial_correlation(cur_data[column].values,
                                               cur_data['mean_framewise_displacement'].values,
                                               cur_data[['age', 'gender']].values)
        current_strategy['strategy'] = column
        corr_modularity.append(current_strategy)
    return {
        'data': pd.DataFrame(corr_modularity).sort_values('correlation'),
        'order': None,
        'title': "Correlation between\nnetwork modularity and motion",
        'label': "Pearson's correlation",
    }


def _qcfc_fdr(sig_per_edge):
    """Do FDR correction on qc-fc p-values."""
    long_qcfc_sig= sig_per_edge.melt()
    long_qcfc_sig['fdr'] = long_qcfc_sig.groupby('variable')['value'].transform(fdr)
    long_qcfc_sig = long_qcfc_sig.groupby('variable').apply(lambda x: 100*x.fdr.sum()/x.fdr.shape[0])
    long_qcfc_sig = pd.DataFrame(long_qcfc_sig, columns=["p_corrected"])
    long_qcfc_sig_order = long_qcfc_sig.sort_values('p_corrected').index.tolist()
    return {
        'data': long_qcfc_sig.T,
        'order': long_qcfc_sig_order,
        'title': "Percentage of significant QC-FC",
        'label': "Percentage %",
    }


def _get_qcfc_median_absolute(qcfc_per_edge):
    """Calculate absolute median and prepare for plotting."""
    qcfc_median_absolute = qcfc_per_edge.apply(calculate_median_absolute)
    qcfc_median_absolute_order = qcfc_median_absolute.sort_values().index.tolist()
    return {
        'data': pd.DataFrame(qcfc_median_absolute).T,
        'order': qcfc_median_absolute_order,
        'title': "Median absolute deviation\nof QC-FC",
        'label': "Median absolute deviation",
    }
