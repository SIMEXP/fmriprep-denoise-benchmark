from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

from nilearn.plotting import plot_matrix

import seaborn as sns

from statsmodels.stats.weightstats import ttest_ind

from fmriprep_denoise.visualization import tables, utils
from fmriprep_denoise.features.derivatives import get_qc_criteria


group_order = {'ds000228': ['adult', 'child'], 'ds000030':['control', 'ADHD', 'bipolar', 'schizophrenia']}
datasets = ['ds000228', 'ds000030']
datasets_baseline = {'ds000228': 'adult', 'ds000030': 'control'}
criteria_name = 'stringent'
fmriprep_version = 'fmriprep-20.2.1lts'


if __name__ == "__main__":
    path_root = utils.get_data_root() / "denoise-metrics"
    strategy_order = list(utils.GRID_LOCATION.values())

    # Plotting
    fig_sig_qcfc, axs_sig_qcfc = plt.subplots(1, 2, sharey=True, constrained_layout=True)
    fig_sig_qcfc.suptitle(r'Significant QC/FC in connectomes (uncorrrected, $\alpha=0.05$)')

    fig_med_qcfc, axs_med_qcfc = plt.subplots(1, 2, sharey=True, constrained_layout=True)
    fig_med_qcfc.suptitle('Medians of absolute values of QC/FC')

    fig_dist, axs_dist = plt.subplots(1, 2, sharey=True, constrained_layout=True)
    fig_dist.suptitle('Distance-dependent of motion')

    fig_modularity_motion, axs_modularity_motion = plt.subplots(1, 2, sharey=True, constrained_layout=True)
    fig_modularity_motion.suptitle('Correlation between motion and network modularity')

    fig_modularity, axs_modularity = plt.subplots(1, 2, sharey=True, constrained_layout=True)
    fig_modularity.suptitle('Mean network modularity')

    for i, dataset in enumerate(datasets):
        path_data = path_root / f"{dataset}_{fmriprep_version.replace('.', '-')}_desc-{criteria_name}_summary.tsv"
        data = pd.read_csv(path_data, sep='\t', index_col=[0, 1], header=[0, 1])
        id_vars = data.index.names
        # df = data['qcfc_fdr_significant'].reset_index().melt(id_vars=id_vars, value_name='Percentage %')
        df = data['qcfc_significant'].reset_index().melt(id_vars=id_vars, value_name='Percentage %')

        sns.barplot(
            y='Percentage %', x='strategy', data=df, ax=axs_sig_qcfc[i],
            order=strategy_order, ci=95,
            hue_order=group_order[dataset]
        )
        # sns.stripplot(y='Percentage %', x='strategy', data=df, ax=axs_sig_qcfc[i],
        #             order=strategy_order, hue_order=group_order[dataset])
        axs_sig_qcfc[i].set_title(dataset)
        axs_sig_qcfc[i].set_ylim(0, 100)
        axs_sig_qcfc[i].set_xticklabels(strategy_order, rotation=45, ha='right', rotation_mode='anchor')

        # median value
        df = data['qcfc_mad'].reset_index().melt(id_vars=id_vars, value_name='Absolute median values')

        sns.barplot(
            y='Absolute median values', x='strategy', data=df, ax=axs_med_qcfc[i],
            order=strategy_order, ci=95,
            # hue_order=['full_sample']
            hue_order=group_order[dataset]
        )

        axs_med_qcfc[i].set_title(dataset)
        axs_med_qcfc[i].set_ylim(0, 0.25)
        axs_med_qcfc[i].set_xticklabels(strategy_order, rotation=45, ha='right', rotation_mode='anchor')

        #distance dependent
        df = data['corr_motion_distance'].reset_index().melt(id_vars=id_vars, value_name='Pearson\'s correlation')
        sns.barplot(
            y='Pearson\'s correlation', x='strategy', data=df, ax=axs_dist[i],
            order=strategy_order, ci=95,
            # hue_order=['full_sample']
            hue_order=group_order[dataset]
        )
        axs_dist[i].set_title(dataset)
        axs_dist[i].set_xticklabels(strategy_order, rotation=45, ha='right', rotation_mode='anchor')

        # correlation between motion and modularity
        df = data['corr_motion_modularity'].reset_index().melt(id_vars=id_vars, value_name='Pearson\'s correlation')

        sns.barplot(
            y='Pearson\'s correlation', x='strategy', data=df, ax=axs_modularity_motion[i],
            order=strategy_order, ci=95,
            # hue_order=['full_sample']
            hue_order=group_order[dataset]
        )
        axs_modularity_motion[i].set_title(dataset)
        axs_modularity_motion[i].set_xticklabels(strategy_order, rotation=45, ha='right', rotation_mode='anchor')

        # average modularity
        df = data['modularity'].reset_index().melt(id_vars=id_vars, value_name='Mean modularity quality (a.u.)')

        sns.barplot(
            y='Mean modularity quality (a.u.)', x='strategy', data=df, ax=axs_modularity[i],
            order=strategy_order, ci=95,
            # hue_order=['full_sample']
            hue_order=group_order[dataset]
        )
        axs_modularity[i].set_title(dataset)
        axs_modularity[i].set_xticklabels(strategy_order, rotation=45, ha='right', rotation_mode='anchor')

    # cc = pd.read_csv(path_root / dataset / fmriprep_version / f'dataset-{dataset}_atlas-mist_nroi-444_connectome.tsv',
    #                  sep='\t', index_col=0)
    # plot_matrix(cc.corr().values, labels=list(cc.columns), colorbar=True, axes=axs[1, 2], cmap=mpl.cm.viridis,
    #             title="Connectome similarity", reorder='complete', vmax=1, vmin=0.7)
    # for i in range(2):
    #     for j in range(2):
    #         axs[i, j].set_xticklabels(strategy_order, rotation=45, ha='right', rotation_mode='anchor')
    # axs[0, 2].set_xticklabels(strategy_order, rotation=45, ha='right', rotation_mode='anchor')
    fig_sig_qcfc.savefig(Path(__file__).parents[1] / 'outputs' / 'sig_qcfc.png')
    fig_med_qcfc.savefig(Path(__file__).parents[1] / 'outputs' / 'median_qcfc.png')
    fig_dist.savefig(Path(__file__).parents[1] / 'outputs' / 'distance_qcfc.png')
    fig_modularity_motion.savefig(Path(__file__).parents[1] / 'outputs' / 'modularity_qcfc.png')
    fig_modularity.savefig(Path(__file__).parents[1] / 'outputs' / 'modularity.png')