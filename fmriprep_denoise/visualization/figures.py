from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from fmriprep_denoise.features import get_atlas_pairwise_distance

from fmriprep_denoise.visualization import utils


path_root = Path(__file__).parents[2] / "inputs"

def plot_motion_overview():
    datasets = ["ds000228", "ds000030"]
    fig = plt.figure(figsize=(7, 5))
    axs = fig.subplots(1, 2, sharey=True)
    for dataset, ax in zip(datasets, axs):
        file = f'dataset-{dataset}_desc-movement_phenotype.tsv'
        path_fd = path_root / f"dataset-{dataset}" / file
        df = pd.read_csv(path_fd, header=[0], index_col=0, sep='\t')
        _, participants_groups, _ = utils._get_participants_groups(dataset)
        participants_groups.name = 'group'
        df = pd.concat([df, participants_groups], axis=1, join='inner')
        sns.barplot(y='mean_framewise_displacement',
                    x='group',
                    data=df,
                    ax=ax)
        ax.set_title(f'{dataset} ($N={df.shape[0]}$)')
    fig.suptitle("Mean framewise displacement per sub-sample")
    return fig


def plot_motion_resid(dataset, atlas_name=None, dimension=None, group='full_sample'):
    # One cannot use specidic dimension but use wild card in atlas
    metric = "qcfc"
    files_qcfc, labels = utils._get_connectome_metric_paths(dataset, metric, atlas_name, dimension)
    qcfc_sig = utils._qcfc_fdr(files_qcfc, labels, group=group)
    qcfc_mad = utils._get_qcfc_median_absolute(files_qcfc, labels, group=group)

    if len(files_qcfc) == 1 and not isinstance(dimension, type(None)):
        qcfc_per_edge = utils._get_qcfc_metric(files_qcfc, metric="correlation", group=group)[0]
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
    color_order = utils._get_palette(list(utils.GRID_LOCATION.values()))
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


def plot_distance_dependence(dataset, atlas_name=None, dimension=None, group='full_sample'):
    metric = "qcfc"
    files_qcfc, labels = utils._get_connectome_metric_paths(dataset, metric, atlas_name, dimension)
    qcfc_dist = utils._get_corr_distance(files_qcfc, labels, group=group)
    color_order = utils._get_palette(qcfc_dist['order'])
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
        qcfc_per_edge = utils._get_qcfc_metric(files_qcfc, metric="correlation", group=group)[0]
        pairwise_distance = get_atlas_pairwise_distance(atlas_name, dimension)
        corr_distance_long = qcfc_per_edge.melt()
        corr_distance_long.columns = ["Strategy", "qcfc"]
        corr_distance_long['distance'] = np.tile(pairwise_distance.iloc[:, -1].values, 11)
        for i, row_axes in enumerate(axs):
            for j, ax in enumerate(row_axes):
                if cur_strategy := utils.GRID_LOCATION.get((i, j), False):
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


def plot_network_modularity(dataset, atlas_name=None, dimension=None, by_group=False):
    metric = "modularity"
    files_network, labels = utils._get_connectome_metric_paths(dataset, metric, atlas_name, dimension)

    file_dataset = path_root / f"dataset-{dataset}/dataset-{dataset}_desc-movement_phenotype.tsv"
    movement = pd.read_csv(file_dataset, sep='\t', index_col=0, header=0, encoding='utf8')
    if not by_group:
        return _plot_network_modularity(
            dimension, files_network, labels, dataset, movement
        )

    _, participant_groups, groups = utils._get_participants_groups(dataset)
    figs = []
    for group in groups:
        subgroup_movement = movement[participant_groups == group]
        fig = _plot_network_modularity(
            dimension, files_network, labels, group, subgroup_movement
        )
        figs.append(fig)
    return figs


def _plot_network_modularity(dimension, files_network, labels, group, movement):
    network_mod, corr_mod = utils._corr_modularity_motion(movement, files_network, labels)
    color_order = utils._get_palette(list(utils.GRID_LOCATION.values()))
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
    axs[0].set_xlim((-0.7, 1))

    axs[1].set_title(corr_mod['title'])
    axs[1].set(xlabel=corr_mod['label'])
    axs[1].set_xlim((0, 0.85))

    fig.suptitle(f'Network modularity - {group}', fontsize='xx-large')
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

    confounds_phenotype, participant_groups, groups = utils._get_participants_groups(dataset)

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
        cur_dof = confounds_phenotype[participant_groups == group].melt()
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


def _plot_single_motion_resid(qcfc_sig, qcfc_mad, long_qcfc):
    fig = plt.figure(constrained_layout=True, figsize=(13, 5))
    fig.suptitle('Residual effect of motion on connectomes after de-noising', fontsize='xx-large')
    subfigs = fig.subfigures(1, 2)
    axs = subfigs[0].subplots(1, 2, sharey=False)
    for ax, figure_data in zip(axs, [qcfc_sig, qcfc_mad]):
        sns.barplot(data=figure_data['data'], orient='h',
                    order=figure_data['order'],
                    palette=utils._get_palette(figure_data['order']),
                    ax=ax)
        ax.set_title(figure_data['title'])
        ax.set(xlabel=figure_data['label'])
        ax.set(ylabel="Confound removal strategy")

    axs = subfigs[1].subplots(3, 4, sharex=True, sharey=True)
    for i, row_axes in enumerate(axs):
        for j, ax in enumerate(row_axes):
            if cur_strategy := utils.GRID_LOCATION.get((i, j), False):
                mask = long_qcfc["Strategy"] == cur_strategy
                g = sns.kdeplot(data=long_qcfc.loc[mask, :],
                                x='qcfc',
                                fill=True,
                                color=utils.palette_dict[cur_strategy],
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
