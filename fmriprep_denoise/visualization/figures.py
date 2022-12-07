import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from fmriprep_denoise.features import get_atlas_pairwise_distance

from fmriprep_denoise.visualization import utils


def plot_motion_resid(
    dataset,
    fmriprep_version,
    path_root,
    atlas_name=None,
    dimension=None,
    group="full_sample",
    fdr=False,
):
    """
    Dirty and quick plot for residual motion impact on functional connectivity.

    Parameters
    ----------
    dataset : str
        Dataset ID.

    atlas_name : None or str
        Atlas name.

    dimension : None or str or int
        Dimension of the atlas.

    group : str
        Default to full sampe ('full_sample'), or string value under the
        dataset's groups column.

    fdr : boolean
        Default to False.
        Perform FDR correction or not on the qc/fc p values.
    """
    # One cannot use specific dimension but use wild card in atlas
    metric = "qcfc"
    files_qcfc, labels = utils._get_connectome_metric_paths(
        dataset,
        fmriprep_version,
        metric,
        atlas_name,
        dimension,
        path_root,
    )
    qcfc_sig = utils._qcfc_pvalue(files_qcfc, labels, group=group, fdr=fdr)
    qcfc_mad = utils._get_qcfc_absolute_median(files_qcfc, labels, group=group)

    if len(files_qcfc) == 1 and not isinstance(dimension, type(None)):
        qcfc_per_edge = utils._get_qcfc_metric(
            files_qcfc, metric="correlation", group=group
        )[0]
        long_qcfc = qcfc_per_edge.melt()
        long_qcfc.columns = ["Strategy", "qcfc"]
        fig = _plot_single_motion_resid(qcfc_sig, qcfc_mad, long_qcfc)
    else:
        # plotting
        fig = plt.figure(constrained_layout=True, figsize=(11, 5))
        fig.suptitle("Residual effect of motion on connectomes", fontsize="xx-large")
        axs = fig.subplots(1, 2, sharey=False)
        for ax, figure_data in zip(axs, [qcfc_sig, qcfc_mad]):
            ax = _summary_plots(figure_data, ax)
            ax.set_xlim(figure_data["xlim"])
            ax.set_title(figure_data["title"])
        axs[0].set(ylabel="Confound removal strategy")
    return fig


def _plot_single_motion_resid(qcfc_sig, qcfc_mad, long_qcfc):
    """Return motion metrics plot for one map."""
    fig = plt.figure(constrained_layout=True, figsize=(13, 5))
    fig.suptitle(
        "Residual effect of motion on connectomes after de-noising",
        fontsize="xx-large",
    )
    subfigs = fig.subfigures(1, 2)
    axs = subfigs[0].subplots(1, 2, sharey=False)
    for ax, figure_data in zip(axs, [qcfc_sig, qcfc_mad]):
        sns.barplot(
            data=figure_data["data"],
            orient="h",
            order=figure_data["order"],
            palette=utils._get_palette(figure_data["order"]),
            ax=ax,
        )
        ax.set_title(figure_data["title"])
        ax.set(xlabel=figure_data["label"])
        ax.set(ylabel="Confound removal strategy")

    axs = subfigs[1].subplots(3, 4, sharex=True, sharey=True)
    for i, row_axes in enumerate(axs):
        for j, ax in enumerate(row_axes):
            cur_strategy = utils.GRID_LOCATION.get((i, j), False)
            if cur_strategy:
                mask = long_qcfc["Strategy"] == cur_strategy
                g = sns.kdeplot(
                    data=long_qcfc.loc[mask, :],
                    x="qcfc",
                    fill=True,
                    color=utils.palette_dict[cur_strategy],
                    ax=ax,
                )
                g.set_title(cur_strategy, fontsize="small")
                g.axvline(0, linewidth=1, linestyle="--", color="r")
                xlabel = "Pearson's correlation" if i == 2 else None
                g.set(xlabel=xlabel)
            else:
                subfigs[1].delaxes(axs[i, j])
    subfigs[1].set_facecolor("0.85")
    subfigs[1].suptitle("Distribution of QC-FC")
    return fig


def _summary_plots(figure_data, ax):
    """Return motion metrics plot for a full set of atlas."""
    color_order = utils._get_palette(list(utils.GRID_LOCATION.values()))
    if figure_data["data"].shape[0] != 1:
        ax = sns.boxplot(
            data=figure_data["data"],
            orient="h",
            order=figure_data["order"],
            width=0.6,
            whis=0.65,
            ax=ax,
            palette=color_order,
        )
    else:
        ax = sns.stripplot(
            data=figure_data["data"],
            orient="h",
            order=figure_data["order"],
            size=4,
            palette=color_order,
            linewidth=1,
            alpha=1,
            ax=ax,
        )
    ax.set(xlabel=figure_data["label"])
    return ax


def plot_distance_dependence(
    dataset,
    fmriprep_version,
    path_root,
    atlas_name=None,
    dimension=None,
    group="full_sample",
):
    """
    Dirty and quick plot for motion distnace dependence.

    Parameters
    ----------
    dataset : str
        Dataset ID.

    atlas_name : None or str
        Atlas name.

    dimension : None or str or int
        Dimension of the atlas.

    group : str
        Default to full sampe ('full_sample'), or string value under the
        dataset's groups column.
    """
    metric = "qcfc"
    files_qcfc, labels = utils._get_connectome_metric_paths(
        dataset, fmriprep_version, metric, atlas_name, dimension, path_root
    )
    qcfc_dist = utils._get_corr_distance(files_qcfc, labels, group=group)
    color_order = utils._get_palette(qcfc_dist["order"])
    if len(files_qcfc) == 1 and not isinstance(dimension, type(None)):

        fig = plt.figure(constrained_layout=True, figsize=(7, 5))
        subfigs = fig.subfigures(1, 2, width_ratios=[1, 2])
        ax = subfigs[0].subplots(1, 1, sharex=True, sharey=True)

        sns.barplot(
            data=qcfc_dist["data"],
            orient="h",
            order=qcfc_dist["order"],
            palette=color_order,
            ax=ax,
        )
        ax.set_title(qcfc_dist["title"])
        ax.set(xlabel=qcfc_dist["label"])
        ax.set(ylabel="Confound removal strategy")

        axs = subfigs[1].subplots(3, 4, sharex=True, sharey=True)
        qcfc_per_edge = utils._get_qcfc_metric(
            files_qcfc, metric="correlation", group=group
        )[0]
        pairwise_distance = get_atlas_pairwise_distance(atlas_name, dimension)
        corr_distance_long = qcfc_per_edge.melt()
        corr_distance_long.columns = ["Strategy", "qcfc"]
        corr_distance_long["distance"] = np.tile(
            pairwise_distance.iloc[:, -1].values, 11
        )
        for i, row_axes in enumerate(axs):
            for j, ax in enumerate(row_axes):
                cur_strategy = utils.GRID_LOCATION.get((i, j), False)
                if cur_strategy:
                    mask = corr_distance_long["Strategy"] == cur_strategy
                    g = sns.histplot(
                        data=corr_distance_long.loc[mask, :],
                        x="distance",
                        y="qcfc",
                        ax=ax,
                    )
                    ax.set_title(cur_strategy, fontsize="small")
                    g.axhline(0, linewidth=1, linestyle="--", alpha=0.5, color="k")
                    sns.regplot(
                        data=corr_distance_long.loc[mask, :],
                        x="distance",
                        y="qcfc",
                        ci=None,
                        scatter=False,
                        line_kws={"color": "r", "linewidth": 0.5},
                        ax=ax,
                    )
                    xlabel = "Distance (mm)" if i == 2 else None
                    ylabel = "QC-FC" if j == 0 else None
                    g.set(xlabel=xlabel, ylabel=ylabel)
                else:
                    subfigs[1].delaxes(axs[i, j])
        subfigs[1].suptitle("Correlation between nodewise Euclidean distance and QC-FC")
        fig.suptitle(
            "Distance-dependent effects of motion on connectivity",
            fontsize="xx-large",
        )

    else:
        fig = plt.figure(constrained_layout=True, figsize=(7, 5))
        ax = fig.subplots(1, 1, sharex=True, sharey=True)
        ax = _summary_plots(qcfc_dist, ax)
        ax.set_title(
            "Correlation between nodewise Euclidean distance and QC-FC",
            fontsize="xx-large",
        )
        ax.set(xlabel=qcfc_dist["label"])
        ax.set(ylabel="Confound removal strategy")
        ax.set_xlim((-0.55, 0.5))
    return fig


def plot_network_modularity(
    dataset,
    fmriprep_version,
    path_root,
    atlas_name=None,
    dimension=None,
    by_group=False,
):
    """
    Dirty and quick plot for motion impact on modularity.

    Parameters
    ----------
    dataset : str
        Dataset ID.

    atlas_name : None or str
        Atlas name.

    dimension : None or str or int
        Dimension of the atlas.

    by_group : bool, default False
        Default to full sampe.
    """
    metric = "modularity"
    files_network, labels = utils._get_connectome_metric_paths(
        dataset,
        fmriprep_version,
        metric,
        atlas_name,
        dimension,
        path_root,
    )

    file_dataset = (
        path_root
        / dataset
        / fmriprep_version
        / f"dataset-{dataset}_desc-movement_phenotype.tsv"
    )
    movement = pd.read_csv(
        file_dataset, sep="\t", index_col=0, header=0, encoding="utf8"
    )
    movement = movement[["mean_framewise_displacement", "age", "gender"]]
    if not by_group:
        return _plot_network_modularity(
            dimension, files_network, labels, dataset, movement
        )

    _, participant_groups, groups = utils._get_participants_groups(
        dataset, fmriprep_version, path_root
    )
    figs = []
    for group in groups:
        subgroup_movement = movement[participant_groups == group]
        fig = _plot_network_modularity(
            dimension, files_network, labels, group, subgroup_movement
        )
        figs.append(fig)
    return figs


def _plot_network_modularity(dimension, files_network, labels, group, movement):
    """Motion impact on modularity for one map or a full set of atlas."""
    network_mod, corr_mod = utils._corr_modularity_motion(
        movement, files_network, labels
    )
    color_order = utils._get_palette(list(utils.GRID_LOCATION.values()))
    fig = plt.figure(constrained_layout=True, figsize=(11, 5))
    axs = fig.subplots(1, 2, sharey=False)
    if len(files_network) == 1 and not isinstance(dimension, type(None)):
        sns.barplot(
            data=network_mod["data"],
            orient="h",
            palette=color_order,
            ax=axs[0],
        )
        sns.barplot(data=corr_mod["data"], orient="h", palette=color_order, ax=axs[1])
    else:
        axs[0] = _summary_plots(network_mod, axs[0])
        axs[1] = _summary_plots(corr_mod, axs[1])
    axs[0].set_title(network_mod["title"])
    axs[0].set(xlabel=network_mod["label"])
    axs[0].set(ylabel="Confound removal strategy")
    axs[0].set_xlim((-0.7, 1))

    axs[1].set_title(corr_mod["title"])
    axs[1].set(xlabel=corr_mod["label"])
    axs[1].set_xlim((0, 0.85))

    fig.suptitle(f"Network modularity - {group}", fontsize="xx-large")
    return fig


def plot_dof_dataset(
    fmriprep_version, path_root, gross_fd=None, fd_thresh=None, proportion_thresh=None
):
    """
    Dirty and quick plot for loss of temporal degrees of freedom.

    Parameters
    ----------
    path_root : pathlib.Path
        Root of the metrics output.

    gross_fd : None or float
        Gross mean framewise dispancement threshold.

    fd_thresh : None or float
        Volume level framewise dispancement threshold.

    proportion_thresh : None or float
        Proportion of volumes scrubbed threshold.
    """
    datasets = ["ds000228", "ds000030"]

    fig = plt.figure(constrained_layout=True, figsize=(11, 5))
    axs = fig.subplots(1, 2, sharey=True)
    ds_groups = []
    for ax, dataset in zip(axs, datasets):
        (
            confounds_phenotype,
            participant_groups,
            groups,
        ) = utils._get_participants_groups(
            dataset,
            fmriprep_version,
            path_root,
            gross_fd=gross_fd,
            fd_thresh=fd_thresh,
            proportion_thresh=proportion_thresh,
        )
        ds_groups.append((dataset, groups))
        participant_groups = participant_groups.to_frame()
        participant_groups = participant_groups.reset_index(col_fill="participant_id")
        confounds_phenotype.index = pd.MultiIndex.from_frame(participant_groups)
        confounds_phenotype = confounds_phenotype.reset_index()
        confounds_phenotype = confounds_phenotype.melt(
            id_vars=["index", "groups"],
            var_name=["strategy", "type"],
        )
        sns.barplot(
            x="value",
            y="strategy",
            data=confounds_phenotype[confounds_phenotype["type"] == "compcor"],
            hue="groups",
            hue_order=groups,
            ci=95,
            color="blue",
            linewidth=1,
            edgecolor="blue",
            ax=ax,
        )
        sns.barplot(
            x="value",
            y="strategy",
            data=confounds_phenotype[confounds_phenotype["type"] == "aroma"],
            hue="groups",
            hue_order=groups,
            ci=95,
            color="orange",
            linewidth=1,
            edgecolor="orange",
            ax=ax,
        )
        sns.barplot(
            x="value",
            y="strategy",
            data=confounds_phenotype[confounds_phenotype["type"] == "fixed_regressors"],
            hue="groups",
            hue_order=groups,
            ci=95,
            palette=["darkgrey", "darkgrey"],
            linewidth=1,
            edgecolor="darkgrey",
            ax=ax,
        )
        sns.barplot(
            x="value",
            y="strategy",
            data=confounds_phenotype[confounds_phenotype["type"] == "high_pass"],
            hue="groups",
            hue_order=groups,
            ci=95,
            palette=["grey", "grey"],
            linewidth=1,
            edgecolor="grey",
            ax=ax,
        )
        ax.set_xlim(0, 80)
        ax.set_xlabel("Number of regressors")
        ax.set_title(dataset)
        # manually create legend
        ax.get_legend().remove()

    colors = ["blue", "orange", "darkgrey", "grey"]
    labels = [
        "CompCor \nregressors",
        "ICA-AROMA \npartial regressors",
        "Head motion and \ntissue signal",
        "Discrete cosine-basis \nregressors",
    ]
    handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    axs[1].legend(handles=handles, bbox_to_anchor=(1.7, 1))
    return fig, ds_groups


def plot_vol_scrubbed_dataset(
    fmriprep_version, path_root, gross_fd=None, fd_thresh=None, proportion_thresh=None
):
    """
    Dirty and quick plot for loss of temporal volumes in scrubbing based
    strategy.

    Parameters
    ----------
    path_root : pathlib.Path
        Root of the metrics output.

    gross_fd : None or float
        Gross mean framewise dispancement threshold.

    fd_thresh : None or float
        Volume level framewise dispancement threshold.

    proportion_thresh : None or float
        Proportion of volumes scrubbed threshold.
    """
    datasets = ["ds000228", "ds000030"]

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
            gross_fd=gross_fd,
            fd_thresh=fd_thresh,
            proportion_thresh=proportion_thresh,
        )
        participant_groups = participant_groups.to_frame()
        participant_groups = participant_groups.reset_index(col_fill="participant_id")
        confounds_phenotype.index = pd.MultiIndex.from_frame(participant_groups)
        selected = [
            col
            for col, strategy in zip(
                confounds_phenotype.columns,
                confounds_phenotype.columns.get_level_values(0),
            )
            if "scrub" in strategy
        ]
        confounds_phenotype = confounds_phenotype.loc[:, selected]
        confounds_phenotype = confounds_phenotype.reset_index()
        confounds_phenotype = confounds_phenotype.melt(
            id_vars=["index", "groups"],
            var_name=["strategy", "type"],
        )

        sns.boxplot(
            x="value",
            y="strategy",
            data=confounds_phenotype[
                confounds_phenotype["type"] == "excised_vol_proportion"
            ],
            hue="groups",
            hue_order=groups,
            ax=ax,
        )
        ax.set_xlabel("Proportion of removed volumes to scan length")
        ax.set_title(dataset)
        ax.set_xlim((-0.1, 1.1))
    return fig
