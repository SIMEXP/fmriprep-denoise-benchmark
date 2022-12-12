from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

from nilearn.plotting import plot_matrix
from nilearn.plotting.matrix_plotting import _reorder_matrix

import seaborn as sns

from statsmodels.stats.weightstats import ttest_ind

from fmriprep_denoise.visualization import tables, utils
from fmriprep_denoise.features.derivatives import get_qc_criteria


group_order = {
    "ds000228": ["adult", "child"],
    "ds000030": ["control", "ADHD", "bipolar", "schizophrenia"],
}
datasets = ["ds000228", "ds000030"]
datasets_baseline = {"ds000228": "adult", "ds000030": "control"}
criteria_name = "stringent"
fmriprep_version = "fmriprep-20.2.1lts"


if __name__ == "__main__":
    path_root = utils.get_data_root() / "denoise-metrics"
    strategy_order = list(utils.GRID_LOCATION.values())

    # connectome similarity needs a bit more work
    fig_similarity, axs_similarity = plt.subplots(
        1, 2, figsize=(9, 4.8), constrained_layout=True
    )
    fig_similarity.suptitle("Similarity of denoised connectome by strategy")

    average_connectomes = []
    for dataset in datasets:
        connectomes_path = path_root.glob(
            f"{dataset}/{fmriprep_version}/*connectome.tsv"
        )
        connectomes_correlations = []
        for p in connectomes_path:
            cc = pd.read_csv(p, sep="\t", index_col=0)
            connectomes_correlations.append(cc.corr().values)
        average_connectome = pd.DataFrame(
            np.mean(connectomes_correlations, axis=0),
            columns=cc.columns,
            index=cc.columns,
        )
        average_connectomes.append(average_connectome)

    # Average the two averages and cluster the correlation matrix
    _, labels = _reorder_matrix(
        np.mean(average_connectomes, axis=0), list(cc.columns), "complete"
    )

    for i, d in enumerate(average_connectomes):
        if i == 1:
            cbar = True
        else:
            cbar = False
        # reorder each matrix and plot
        current_mat = average_connectomes[i].loc[labels, labels]
        sns.heatmap(
            current_mat,
            square=True,
            ax=axs_similarity[i],
            vmin=0.6,
            vmax=1,
            linewidth=0.5,
            cbar=cbar,
        )
        axs_similarity[i].set_title(datasets[i])
        axs_similarity[i].set_xticklabels(
            labels, rotation=45, ha="right", rotation_mode="anchor"
        )
    fig_similarity.savefig(Path(__file__).parents[1] / "outputs" / "connectomes.png")

    # Plotting
    fig_sig_qcfc, axs_sig_qcfc = plt.subplots(
        1, 2, sharey=True, constrained_layout=True
    )
    fig_sig_qcfc.suptitle(
        r"Significant QC/FC in connectomes (uncorrrected, $\alpha=0.05$)"
    )

    fig_sig_qcfc_fdr, axs_sig_qcfc_fdr = plt.subplots(
        1, 2, sharey=True, constrained_layout=True
    )
    fig_sig_qcfc_fdr.suptitle(
        r"Significant QC/FC in connectomes (FDR corrected, $\alpha=0.05$)"
    )

    fig_med_qcfc, axs_med_qcfc = plt.subplots(
        1, 2, sharey=True, constrained_layout=True
    )
    fig_med_qcfc.suptitle("Medians of absolute values of QC/FC")

    fig_dist, axs_dist = plt.subplots(1, 2, sharey=True, constrained_layout=True)
    fig_dist.suptitle("Distance-dependent of motion")

    fig_modularity = plt.figure(constrained_layout=True, figsize=(6.4, 9.6))
    subfigs_modularity = fig_modularity.subfigures(2, 1, wspace=0.07)

    axs_modularity = subfigs_modularity[0].subplots(
        1, 2, sharey=True
    )
    subfigs_modularity[0].suptitle("Mean network modularity")


    axs_modularity_motion = subfigs_modularity[1].subplots(
        1, 2, sharey=True
    )
    subfigs_modularity[1].suptitle("Correlation between motion and network modularity")

    for i, dataset in enumerate(datasets):
        path_data = (
            path_root
            / f"{dataset}_{fmriprep_version.replace('.', '-')}_desc-{criteria_name}_summary.tsv"
        )
        data = pd.read_csv(path_data, sep="\t", index_col=[0, 1], header=[0, 1])
        id_vars = data.index.names

        # uncorrected qc-fc
        df = (
            data["qcfc_significant"]
            .reset_index()
            .melt(id_vars=id_vars, value_name="Percentage %")
        )

        sns.barplot(
            y="Percentage %",
            x="strategy",
            data=df,
            ax=axs_sig_qcfc[i],
            order=strategy_order,
            ci=95,
            hue_order=group_order[dataset],
        )
        axs_sig_qcfc[i].set_title(dataset)
        axs_sig_qcfc[i].set_ylim(0, 60)
        axs_sig_qcfc[i].set_xticklabels(
            strategy_order, rotation=45, ha="right", rotation_mode="anchor"
        )

        # fdr corrected
        df = (
            data["qcfc_fdr_significant"]
            .reset_index()
            .melt(id_vars=id_vars, value_name="Percentage %")
        )
        sns.barplot(
            y="Percentage %",
            x="strategy",
            data=df,
            ax=axs_sig_qcfc_fdr[i],
            order=strategy_order,
            ci=95,
            hue_order=group_order[dataset],
        )
        axs_sig_qcfc_fdr[i].set_title(dataset)
        axs_sig_qcfc_fdr[i].set_ylim(0, 60)
        axs_sig_qcfc_fdr[i].set_xticklabels(
            strategy_order, rotation=45, ha="right", rotation_mode="anchor"
        )

        # median value
        df = (
            data["qcfc_mad"]
            .reset_index()
            .melt(id_vars=id_vars, value_name="Absolute median values")
        )

        sns.barplot(
            y="Absolute median values",
            x="strategy",
            data=df,
            ax=axs_med_qcfc[i],
            order=strategy_order,
            ci=95,
            # hue_order=['full_sample']
            hue_order=group_order[dataset],
        )

        axs_med_qcfc[i].set_title(dataset)
        axs_med_qcfc[i].set_ylim(0, 0.25)
        axs_med_qcfc[i].set_xticklabels(
            strategy_order, rotation=45, ha="right", rotation_mode="anchor"
        )

        # distance dependent
        df = (
            data["corr_motion_distance"]
            .reset_index()
            .melt(id_vars=id_vars, value_name="Pearson's correlation")
        )
        sns.barplot(
            y="Pearson's correlation",
            x="strategy",
            data=df,
            ax=axs_dist[i],
            order=strategy_order,
            ci=95,
            # hue_order=['full_sample']
            hue_order=group_order[dataset],
        )
        axs_dist[i].set_title(dataset)
        axs_dist[i].set_xticklabels(
            strategy_order, rotation=45, ha="right", rotation_mode="anchor"
        )

        # correlation between motion and modularity
        df = (
            data["corr_motion_modularity"]
            .reset_index()
            .melt(id_vars=id_vars, value_name="Pearson's correlation")
        )

        sns.barplot(
            y="Pearson's correlation",
            x="strategy",
            data=df,
            ax=axs_modularity_motion[i],
            order=strategy_order,
            ci=95,
            # hue_order=['full_sample']
            hue_order=group_order[dataset],
        )
        axs_modularity_motion[i].set_title(dataset)
        axs_modularity_motion[i].set_xticklabels(
            strategy_order, rotation=45, ha="right", rotation_mode="anchor"
        )

        # average modularity
        df = (
            data["modularity"]
            .reset_index()
            .melt(id_vars=id_vars, value_name="Mean modularity quality (a.u.)")
        )

        sns.barplot(
            y="Mean modularity quality (a.u.)",
            x="strategy",
            data=df,
            ax=axs_modularity[i],
            order=strategy_order,
            ci=95,
            # hue_order=['full_sample']
            hue_order=group_order[dataset],
        )
        axs_modularity[i].set_title(dataset)
        axs_modularity[i].set_xticklabels(
            strategy_order, rotation=45, ha="right", rotation_mode="anchor"
        )

    fig_sig_qcfc.savefig(Path(__file__).parents[1] / "outputs" / "sig_qcfc.png")
    fig_sig_qcfc_fdr.savefig(Path(__file__).parents[1] / "outputs" / "sig_qcfc_fdr.png")
    fig_med_qcfc.savefig(Path(__file__).parents[1] / "outputs" / "median_qcfc.png")
    fig_dist.savefig(Path(__file__).parents[1] / "outputs" / "distance_qcfc.png")
    fig_modularity.savefig(Path(__file__).parents[1] / "outputs" / "modularity.png")
