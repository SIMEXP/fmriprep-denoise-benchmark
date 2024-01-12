import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from fmriprep_denoise.features.derivatives import get_qc_criteria
from fmriprep_denoise.visualization import utils


strategy_order = list(utils.GRID_LOCATION.values())


def load_data(path_root, datasets, criteria_name, fmriprep_version):
    criteria = get_qc_criteria(criteria_name)
    confounds_phenotypes = {}
    for dataset in datasets:
        (
            confounds_phenotype,
            participant_groups,
            groups,
        ) = utils._get_participants_groups(
            dataset,
            fmriprep_version,
            path_root,
            gross_fd=criteria["gross_fd"],
            fd_thresh=criteria["fd_thresh"],
            proportion_thresh=criteria["proportion_thresh"],
        )
        df_plotting, full_length = _organise_data(confounds_phenotype[strategy_order])
        confounds_phenotypes[dataset] = {
            "group_values": groups,
            "participant_labels": participant_groups,
            "confounds_phenotype": df_plotting,
            "confounds_stats": confounds_phenotype[strategy_order],
            "full_length": full_length,
        }
    return confounds_phenotypes


def _plot_single_report(confounds_phenotype, full_length, palette, ax, title):
    # change up the data a bit for plotting
    sns.barplot(
        x="strategy",
        y="value",
        data=confounds_phenotype[confounds_phenotype["type"] == "total"],
        ci=95,
        color=palette[4],
        linewidth=1,
        ax=ax,
    )
    sns.barplot(
        x="strategy",
        y="value",
        data=confounds_phenotype[confounds_phenotype["type"] == "compcor"],
        ci=95,
        color=palette[0],
        linewidth=1,
        ax=ax,
    )
    sns.barplot(
        x="strategy",
        y="value",
        data=confounds_phenotype[confounds_phenotype["type"] == "aroma"],
        ci=95,
        color=palette[2],
        linewidth=1,
        ax=ax,
    )
    sns.barplot(
        x="strategy",
        y="value",
        data=confounds_phenotype[confounds_phenotype["type"] == "fixed_regressors"],
        ci=95,
        color=palette[1],
        linewidth=1,
        ax=ax,
    )
    sns.barplot(
        x="strategy",
        y="value",
        data=confounds_phenotype[confounds_phenotype["type"] == "high_pass"],
        ci=95,
        color=palette[3],
        linewidth=1,
        ax=ax,
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel(f"% Degrees of freedom loss\n(Full length: {full_length})")
    ax.set_title(title)
    ax.set_xticklabels(
        strategy_order, rotation=45, ha="right", rotation_mode="anchor"
    )

    return confounds_phenotype


def plot_stats(confounds_phenotypes, plot_subgroup=False):
    sns.set_palette("colorblind")
    palette = sns.color_palette(n_colors=5)
    figsize = (11, 13) if plot_subgroup else (11, 5)
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    axs = fig.subplots(3, 3, sharey=True) if plot_subgroup else fig.subplots(1, 2, sharey=True)
    fig.suptitle(
        "Loss of temporal degrees of freedom",
        weight="heavy",
        fontsize="x-large",
    )

    print("Generating new graph")
    key_axs = [axs[0, 0], axs[1, 0]] if plot_subgroup else [axs[0], axs[1]]
    for ax, dataset in zip(key_axs, confounds_phenotypes):
        confounds_phenotype = confounds_phenotypes[dataset]["confounds_phenotype"]
        _descriptive_stats(dataset, confounds_phenotypes[dataset]["confounds_stats"])
        n = confounds_phenotypes[dataset]["participant_labels"].shape[0]
        _plot_single_report(confounds_phenotype,
                            confounds_phenotypes[dataset]["full_length"],
                            palette, ax, f"{dataset}(N={n})")
    if plot_subgroup:
        starting_x = 0
        for dataset in confounds_phenotypes:
            subjects = confounds_phenotypes[dataset]["participant_labels"].index
            for i, group in enumerate(confounds_phenotypes[dataset]["group_values"]):
                # return index from participant_labels if group is in participant_labels
                selected = confounds_phenotypes[dataset]["participant_labels"] == group
                selected = subjects[selected]
                confounds_phenotype = confounds_phenotypes[dataset]["confounds_phenotype"].loc[selected, :]
                ax = axs[starting_x, (i % 2) + 1]
                if i % 2 == 1:
                    starting_x += 1
                # _descriptive_stats(dataset, confounds_phenotype)
                _plot_single_report(confounds_phenotype,
                                    confounds_phenotypes[dataset]["full_length"],
                                    palette, ax, f"{dataset}: {group} (N={selected.shape[0]})")
    colors = [palette[4], palette[0], palette[2], palette[1], palette[3]]
    labels = [
        "Censored volumes",
        "CompCor \nregressors",
        "ICA-AROMA \npartial regressors",
        "Head motion and \ntissue signal",
        "Discrete cosine-basis \nregressors",
    ]
    handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    if plot_subgroup:
        axs[2,0].legend(handles=handles, loc=0)
        axs[2,0].axis('off')
    else:
        axs[1].legend(handles=handles, loc=0)
    return fig


def _organise_data(confounds_phenotype):
    full_length = confounds_phenotype.iloc[0, -1]
    confounds_phenotype.loc[:, ("aroma", "aroma")] += confounds_phenotype.loc[
        :, ("aroma", "fixed_regressors")
    ]
    confounds_phenotype.loc[:, ("compcor", "compcor")] += confounds_phenotype.loc[
        :, ("compcor", "fixed_regressors")
    ]
    confounds_phenotype.loc[:, ("compcor6", "compcor")] += confounds_phenotype.loc[
        :, ("compcor6", "fixed_regressors")
    ]

    confounds_phenotype = confounds_phenotype.reset_index()
    confounds_phenotype = confounds_phenotype.melt(
        id_vars=["index"],
        var_name=["strategy", "type"],
    )
    confounds_phenotype["value"] /= full_length
    confounds_phenotype["value"] *= 100
    confounds_phenotype = confounds_phenotype.set_index("index")
    return confounds_phenotype, full_length


def _plot_single_report(confounds_phenotype, full_length, palette, ax, title):
    # change up the data a bit for plotting
    sns.barplot(
        x="strategy",
        y="value",
        data=confounds_phenotype[confounds_phenotype["type"] == "total"],
        ci=95,
        color=palette[4],
        linewidth=1,
        ax=ax,
    )
    sns.barplot(
        x="strategy",
        y="value",
        data=confounds_phenotype[confounds_phenotype["type"] == "compcor"],
        ci=95,
        color=palette[0],
        linewidth=1,
        ax=ax,
    )
    sns.barplot(
        x="strategy",
        y="value",
        data=confounds_phenotype[confounds_phenotype["type"] == "aroma"],
        ci=95,
        color=palette[2],
        linewidth=1,
        ax=ax,
    )
    sns.barplot(
        x="strategy",
        y="value",
        data=confounds_phenotype[confounds_phenotype["type"] == "fixed_regressors"],
        ci=95,
        color=palette[1],
        linewidth=1,
        ax=ax,
    )
    sns.barplot(
        x="strategy",
        y="value",
        data=confounds_phenotype[confounds_phenotype["type"] == "high_pass"],
        ci=95,
        color=palette[3],
        linewidth=1,
        ax=ax,
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel(f"% Degrees of freedom loss\n(Full length: {full_length})")
    ax.set_title(title)
    ax.set_xticklabels(
        strategy_order, rotation=45, ha="right", rotation_mode="anchor"
    )

    return confounds_phenotype


def _descriptive_stats(dataset, confounds_phenotype):
    print(dataset)
    _report_descriptive_stats(
        confounds_phenotype,
        "aroma",
        "aroma",
    )
    _report_descriptive_stats(
        confounds_phenotype,
        "compcor",
        "compcor",
    )
    _report_descriptive_stats(
        confounds_phenotype,
        "scrubbing.5",
        "excised_vol",
    )
    _report_descriptive_stats(
        confounds_phenotype,
        "scrubbing.2",
        "excised_vol",
    )


def _report_descriptive_stats(confounds_phenotype, strategy, variable):
    mean = confounds_phenotype.loc[:, (strategy, variable)].mean()
    cmax = confounds_phenotype.loc[:, (strategy, variable)].max()
    cmin = confounds_phenotype.loc[:, (strategy, variable)].min()
    std = confounds_phenotype.loc[:, (strategy, variable)].std()
    print(
        f"descriptive stats for {strategy} {variable}: {mean}({std}); range={cmin} - {cmax}"
    )
