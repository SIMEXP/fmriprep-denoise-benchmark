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

path_root = Path(__file__).parents[2] / "inputs"
palette = sns.color_palette("Paired", n_colors=12)
palette_dict = {name: c for c, name in zip(palette[1:], GRID_LOCATION.values())}


def plot_motion_resid(dataset, atlas_name=None, dimension=None):
    # One cannot use specidic dimension but use wild card in atlas
    metric = "qcfc"
    files_qcfc, labels = _get_file_paths(dataset, metric, atlas_name, dimension)
    qcfc_sig = _qcfc_fdr(files_qcfc, labels)
    qcfc_mad = _get_qcfc_median_absolute(files_qcfc, labels)

    if len(files_qcfc) == 1 and not isinstance(dimension, type(None)):
        qcfc_per_edge = _get_qcfc_metric(files_qcfc, metric="correlation")[0]
        long_qcfc = qcfc_per_edge.melt()
        long_qcfc.columns = ["Strategy", "qcfc"]
        fig = _plot_single_motion_resid(qcfc_sig, qcfc_mad, long_qcfc)
    else:
        # plotting
        fig = plt.figure(constrained_layout=True, figsize=(11, 5))
        fig.suptitle('Residual effect of motion on connectomes', fontsize='xx-large')
        axs = fig.subplots(1, 2, sharey=False)
        for ax, figure_data in zip(axs, [qcfc_sig, qcfc_mad]):
            ax = _summary_plots(figure_data, ax)
            ax.set_title(figure_data['title'])
        axs[0].set(ylabel="Confound removal strategy")
    return fig


def _summary_plots(figure_data, ax):
    color_order = _get_palette(list(GRID_LOCATION.values()))
    if figure_data['data'].shape[0] != 1:
        ax = sns.boxplot(data=figure_data['data'], orient='h',
                         order=figure_data['order'],
                         width=.6, whis=0.65,
                         ax=ax, palette=color_order)
    else:
        sns.stripplot(data=figure_data['data'], orient='h',
                      order=figure_data['order'],
                      size=4, palette=color_order,
                      linewidth=1, alpha=1,
                      ax=ax)
    ax.set(xlabel=figure_data['label'])
    return ax


def plot_distance_dependence(dataset, atlas_name=None, dimension=None):
    metric = "qcfc"
    files_qcfc, labels = _get_file_paths(dataset, metric, atlas_name, dimension)
    qcfc_dist = _get_corr_distance(files_qcfc, labels)
    color_order = _get_palette(qcfc_dist['order'])
    if len(files_qcfc) == 1 and not isinstance(dimension, type(None)):

        fig = plt.figure(constrained_layout=True, figsize=(7, 5))
        subfigs = fig.subfigures(1, 2, width_ratios=[1, 2])
        ax = subfigs[0].subplots(1, 1, sharex=True, sharey=True)

        sns.barplot(data=qcfc_dist['data'], orient='h',
                    order=qcfc_dist['order'],
                    palette=color_order, ax=ax)
        ax.set_title(qcfc_dist['title'])
        ax.set(xlabel=qcfc_dist['label'])
        ax.set(ylabel="Confound removal strategy")

        axs = subfigs[1].subplots(3, 4, sharex=True, sharey=True)
        qcfc_per_edge = _get_qcfc_metric(files_qcfc, metric="correlation")[0]
        pairwise_distance = get_atlas_pairwise_distance(atlas_name, dimension)
        corr_distance_long = qcfc_per_edge.melt()
        corr_distance_long.columns = ["Strategy", "qcfc"]
        corr_distance_long['distance'] = np.tile(pairwise_distance.iloc[:, -1].values, 11)
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
        fig.suptitle('Distance-dependent effects of motion on connectivity', fontsize='xx-large')

    else:
        fig = plt.figure(constrained_layout=True, figsize=(7, 5))
        ax = fig.subplots(1, 1, sharex=True, sharey=True)
        ax = _summary_plots(qcfc_dist, ax)
        ax.set_title('Correlation between nodewise Euclidian distance and QC-FC', fontsize='xx-large')
        ax.set(xlabel=qcfc_dist['label'])
        ax.set(ylabel="Confound removal strategy")
        ax.set_xlim((-0.75, 0.05))
    return fig


def plot_network_modularity(dataset, atlas_name=None, dimension=None):
    metric = "modularity"
    files_network, labels = _get_file_paths(dataset, metric, atlas_name, dimension)

    file_dataset = path_root / f"dataset-{dataset}/dataset-{dataset}_desc-movement_phenotype.tsv"
    movement = pd.read_csv(file_dataset, sep='\t', index_col=0, header=0, encoding='utf8')
    network_mod, corr_mod = _corr_modularity_motion(movement, files_network, labels)
    color_order = _get_palette(list(GRID_LOCATION.values()))
    fig = plt.figure(constrained_layout=True, figsize=(11, 5))
    axs = fig.subplots(1, 2, sharey=False)
    if len(files_network) == 1 and not isinstance(dimension, type(None)):
        sns.barplot(data=network_mod['data'],
                    orient='h',
                    palette=color_order, ax=axs[0])
        sns.barplot(data=corr_mod['data'],
                    orient='h',
                    palette=color_order, ax=axs[1])
    else:
        axs[0] = _summary_plots(network_mod, axs[0])
        axs[1] = _summary_plots(corr_mod, axs[1])
    axs[0].set_title(network_mod['title'])
    axs[0].set(xlabel=network_mod['label'])
    axs[0].set(ylabel="Confound removal strategy")
    axs[0].set_xlim((-0.7, 0.65))

    axs[1].set_title(corr_mod['title'])
    axs[1].set(xlabel=corr_mod['label'])
    axs[1].set_xlim((0, 0.85))

    fig.suptitle('Network modularity', fontsize='xx-large')
    return fig


def plot_dof_overview(plot_info):
    datasets = ["ds000228", "ds000030"]
    fig = plt.figure(figsize=(11, 5))
    axs = fig.subplots(1, 2, sharey=True)
    for dataset, ax in zip(datasets, axs):
        file = f'dataset-{dataset}_desc-confounds_phenotype.tsv'
        path_dof = path_root / "metrics" / file
        df = pd.read_csv(path_dof, header=[0, 1], index_col=0, sep='\t')
        df = df.melt()
        _dof_report(dataset, ax, df, plot_info)
    axs[1].legend(bbox_to_anchor=(1.6, 1))
    return fig


def plot_dof_dataset(dataset, plot_info):
    group_info_column = "Child_Adult" if dataset == "ds000228" else "diagnosis"
    path_dof = path_root / "metrics" / f'dataset-{dataset}_desc-confounds_phenotype.tsv'
    path_participants = f'../inputs/{dataset}/participants.tsv'
    df = pd.read_csv(path_dof, header=[0, 1], index_col=0, sep='\t')
    df_participants = pd.read_csv(path_participants, index_col=0, sep='\t').loc[df.index, group_info_column]

    groups = df_participants.unique().tolist()

    # this is lazy
    if dataset == "ds000228":
        fig = plt.figure(figsize=(11, 5))
        axs = fig.subplots(1, 2, sharey=True)
        legend_loc = 1
    else:
        fig = plt.figure(figsize=(11, 11))
        axs = fig.subplots(2, 2, sharey=True, sharex=True)
        legend_loc = (0, 1)

    for group, ax in zip(groups, axs.flat):
        cur_dof = df[df_participants == group].melt()
        title = f"{dataset}-{group}"
        _dof_report(title, ax, cur_dof, plot_info)
    axs[legend_loc].legend(bbox_to_anchor=(1.6, 1))
    return fig


def _dof_report(title, ax, df, plot_info):

    if plot_info == 'dof':
        df.columns = ['Strategy', 'type', 'Number of regressors']
        sns.barplot(y='Strategy', x='Number of regressors',
                    data=df[df['type']=='compcor'], ci=95,
                    color='blue', ax=ax,
                    label='CompCor \nregressors')
        sns.barplot(y='Strategy', x='Number of regressors',
                    data=df[df['type']=='aroma'], ci=95,
                    color='orange', ax=ax,
                    label='ICA-AROMA \npartial regressors')
        sns.barplot(y='Strategy', x='Number of regressors',
                    data=df[df['type']=='fixed_regressors'],
                    color='darkgrey', ax=ax,
                    label='Head motion and \ntissue signal')
        sns.barplot(y='Strategy', x='Number of regressors',
                    data=df[df['type']=='high_pass'], ax=ax,
                    color='grey', label='Discrete cosine-basis \nregressors')
        ax.set_title(title)
        ax.set_xlim(0, 80)

    elif plot_info == 'scrubbing':
        df.columns = ['Strategy', 'type', 'Proportion of removed volumes to scan length']
        sns.barplot(y='Strategy', x='Proportion of removed volumes to scan length',
                    data=df[df['type']=='excised_vol_proportion'], ci=95,
                    palette=['darkgrey'] * 7+ ['orange', 'orange'] + ['blue', 'blue'],
                    ax=ax)
        ax.set_title(title)
        ax.set_xlim(0, 1)


def _get_file_paths(dataset, metric, atlas_name, dimension):
    atlas_name = "*" if isinstance(atlas_name, type(None)) else atlas_name
    dimension = "*" if isinstance(atlas_name, type(None)) or isinstance(dimension, type(None)) else dimension
    files = list(path_root.glob(f"metrics/dataset-{dataset}_atlas-{atlas_name}_nroi-{dimension}_{metric}.tsv"))
    if not files:
        raise FileNotFoundError("No file matching the supplied arguments:"
                                f"atlas_name={atlas_name}, "
                                f"dimension={dimension}, "
                                f"dataset={dataset}",
                                f"metric={metric}")
    labels = [file.name.split(f"_{metric}")[0] for file in files]
    return files, labels


def _plot_single_motion_resid(qcfc_sig, qcfc_mad, long_qcfc):
    fig = plt.figure(constrained_layout=True, figsize=(13, 5))
    fig.suptitle('Residual effect of motion on connectomes after de-noising', fontsize='xx-large')
    subfigs = fig.subfigures(1, 2)
    axs = subfigs[0].subplots(1, 2, sharey=False)
    for ax, figure_data in zip(axs, [qcfc_sig, qcfc_mad]):
        sns.barplot(data=figure_data['data'], orient='h',
                        order=figure_data['order'],
                        palette=_get_palette(figure_data['order']),
                        ax=ax)
        ax.set_title(figure_data['title'])
        ax.set(xlabel=figure_data['label'])
        ax.set(ylabel="Confound removal strategy")

    axs = subfigs[1].subplots(3, 4, sharex=True, sharey=True)
    for i, row_axes in enumerate(axs):
        for j, ax in enumerate(row_axes):
            if cur_strategy := GRID_LOCATION.get((i, j), False):
                mask = long_qcfc["Strategy"] == cur_strategy
                g = sns.kdeplot(data=long_qcfc.loc[mask, :],
                                x='qcfc',
                                fill=True,
                                color=palette_dict[cur_strategy],
                                ax=ax)
                g.set_title(cur_strategy, fontsize='small')
                mad = qcfc_mad['data'][cur_strategy].values
                g.axvline(mad, linewidth=1, linestyle='--', color='r')
                xlabel = "Pearson\'s correlation" if i == 2 else None
                g.set(xlabel=xlabel)
            else:
                subfigs[1].delaxes(axs[i, j])
    subfigs[1].set_facecolor('0.85')
    subfigs[1].suptitle('Distribution of QC-FC')
    return fig


def _get_qcfc_metric(file_path, metric):
    """ Get correlation or pvalue of QC-FC."""
    if not isinstance(file_path, list):
        file_path = [file_path]
    qcfc_per_edge = []
    for p in file_path:
        qcfc_stats = pd.read_csv(p, sep='\t', index_col=0)
        df = qcfc_stats.filter(regex=metric)
        df.columns = [col.split('_')[0] for col in df.columns]
        qcfc_per_edge.append(df)
    return qcfc_per_edge


def _get_corr_distance(files_qcfc, labels):

    qcfc_per_edge = _get_qcfc_metric(files_qcfc, metric="correlation")
    corr_distance = []
    for df, label in zip(qcfc_per_edge, labels):
        atlas_name = label.split("atlas-")[-1].split("_")[0]
        dimension = label.split("nroi-")[-1].split("_")[0]
        pairwise_distance = get_atlas_pairwise_distance(atlas_name, dimension)
        cols = df.columns
        df, _ = spearmanr(pairwise_distance.iloc[:, -1], df)
        df = pd.DataFrame(df[1:, 0], index=cols, columns=[label])
        corr_distance.append(df)

    if len(corr_distance) == 1:
        corr_distance = corr_distance[0]
    else:
        corr_distance = pd.concat(corr_distance, axis=1)

    return {
        'data': corr_distance.T,
        'order': list(GRID_LOCATION.values()),
        'title': "Correlation between\nnodewise distance and QC-FC",
        'label': "Pearson's correlation",
    }


def _corr_modularity_motion(movement, files_network, labels):
    mean_corr, mean_modularity = [], []
    for file_network, label in zip(files_network, labels):
        modularity = pd.read_csv(file_network, sep='\t', index_col=0)
        mean_modularity.append(modularity.mean())

        corr_modularity = []
        z_movement = movement.apply(zscore)
        for column, _ in modularity.iteritems():
            cur_data = pd.concat((modularity[column],
                                  movement[['mean_framewise_displacement']],
                                  z_movement[['age', 'gender']]), axis=1).dropna()
            current_strategy = partial_correlation(
                cur_data[column].values,
                cur_data['mean_framewise_displacement'].values,
                cur_data[['age', 'gender']].values
                )
            current_strategy['strategy'] = column
            corr_modularity.append(current_strategy)
        corr_modularity = pd.DataFrame(corr_modularity).set_index(['strategy'])['correlation']
        corr_modularity.columns = [label]
        mean_corr.append(corr_modularity)
    mean_corr = pd.concat(mean_corr, axis=1)
    mean_modularity = pd.concat(mean_modularity, axis=1)
    mean_modularity.columns = labels
    corr_modularity = {
        'data': mean_corr.T,
        'order': list(GRID_LOCATION.values()),
        'title': "Correlation between\nnetwork modularity and motion",
        'label': "Pearson's correlation",
    }
    network_mod = {
        'data': mean_modularity.T,
        'order': list(GRID_LOCATION.values()),
        'title': "Identifiability of network structure\nafter denoising",
        'label': "Mean modularity quality (a.u.)",
    }
    return corr_modularity, network_mod


def _qcfc_fdr(file_qcfc, labels):
    """Do FDR correction on qc-fc p-values."""
    sig_per_edge = _get_qcfc_metric(file_qcfc, metric="pvalue")
    long_qcfc_sig = []
    for df, label in zip(sig_per_edge, labels):
        df = df.melt()
        df['fdr'] = df.groupby('variable')['value'].transform(fdr)
        df = df.groupby('variable').apply(lambda x: 100*x.fdr.sum()/x.fdr.shape[0])
        df = pd.DataFrame(df, columns=[label])
        long_qcfc_sig.append(df)

    if len(long_qcfc_sig) == 1:
        long_qcfc_sig = long_qcfc_sig[0]
        long_qcfc_sig.columns = ['p_corrected']
    else:
        long_qcfc_sig = pd.concat(long_qcfc_sig, axis=1)

    return {
        'data': long_qcfc_sig.T,
        'order': list(GRID_LOCATION.values()),
        'title': "Percentage of significant QC-FC",
        'xlim': (-5, 105),
        'label': "Percentage %",
    }


def _get_qcfc_median_absolute(file_qcfc, labels):
    """Calculate absolute median and prepare for plotting."""
    qcfc_per_edge = _get_qcfc_metric(file_qcfc, metric="correlation")
    qcfc_median_absolute = []
    for df, label in zip(qcfc_per_edge, labels):
        df = df.apply(calculate_median_absolute)
        df.columns = [label]
        qcfc_median_absolute.append(df)

    if len(qcfc_median_absolute) == 1:
        qcfc_median_absolute = qcfc_median_absolute[0]
        title = "Median absolute deviation\nof QC-FC"
    else:
        qcfc_median_absolute = pd.concat(qcfc_median_absolute, axis=1)
        title = "Median absolute deviation of QC-FC"
    return {
        'data': pd.DataFrame(qcfc_median_absolute).T,
        'order': list(GRID_LOCATION.values()),
        'title': title,
        'xlim': (-0.02, 0.22),
        'label': "Median absolute deviation",
    }


def _get_palette(order):
    return [palette_dict[item] for item in order]
