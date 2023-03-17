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
        confounds_phenotypes[dataset] = confounds_phenotype
    return confounds_phenotypes


def plot_stats(confounds_phenotypes):

    fig = plt.figure(constrained_layout=True, figsize=(11, 5))
    axs = fig.subplots(1, 2, sharey=True)
    fig.suptitle(
        "Loss of temporal degrees of freedom",
        weight="heavy",
        fontsize="x-large",
    )

    print("Generating new graph")
    for ax, dataset in zip(axs, confounds_phenotypes):
        confounds_phenotype = confounds_phenotypes[dataset]
        _descriptive_stats(dataset, confounds_phenotype)

        # change up the data a bit for plotting
        full_length = confounds_phenotype.iloc[0, -1]
        confounds_phenotype.loc[:, ("aroma", "aroma")] += confounds_phenotype.loc[
            :, ("aroma", "fixed_regressors")
        ]
        confounds_phenotype.loc[:, ("aroma+gsr", "aroma")] += confounds_phenotype.loc[
            :, ("aroma+gsr", "fixed_regressors")
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
        sns.barplot(
            x="strategy",
            y="value",
            data=confounds_phenotype[confounds_phenotype["type"] == "total"],
            ci=95,
            color="red",
            linewidth=1,
            ax=ax,
        )
        sns.barplot(
            x="strategy",
            y="value",
            data=confounds_phenotype[confounds_phenotype["type"] == "compcor"],
            ci=95,
            color="blue",
            linewidth=1,
            ax=ax,
        )
        sns.barplot(
            x="strategy",
            y="value",
            data=confounds_phenotype[confounds_phenotype["type"] == "aroma"],
            ci=95,
            color="orange",
            linewidth=1,
            ax=ax,
        )
        sns.barplot(
            x="strategy",
            y="value",
            data=confounds_phenotype[confounds_phenotype["type"] == "fixed_regressors"],
            ci=95,
            color="darkgrey",
            linewidth=1,
            ax=ax,
        )
        sns.barplot(
            x="strategy",
            y="value",
            data=confounds_phenotype[confounds_phenotype["type"] == "high_pass"],
            ci=95,
            color="grey",
            linewidth=1,
            ax=ax,
        )
        ax.set_ylim(0, 100)
        ax.set_ylabel(f"% Degrees of freedom loss\n(Full length: {full_length})")
        ax.set_title(dataset)
        ax.set_xticklabels(
            strategy_order, rotation=45, ha="right", rotation_mode="anchor"
        )

    colors = ["red", "blue", "orange", "darkgrey", "grey"]
    labels = [
        "Censored volumes",
        "CompCor \nregressors",
        "ICA-AROMA \npartial regressors",
        "Head motion and \ntissue signal",
        "Discrete cosine-basis \nregressors",
    ]
    handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    axs[1].legend(handles=handles, bbox_to_anchor=(1.7, 1))
    return fig


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
