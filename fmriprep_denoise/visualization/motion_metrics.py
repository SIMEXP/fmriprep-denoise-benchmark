import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from fmriprep_denoise.visualization import utils


strategy_order = list(utils.GRID_LOCATION.values())

measures = {
    "p_values": {
        "var_name": "qcfc_significant",
        "label": "Percentage %",
        "title": "Significant QC-FC in connectomes\n"
        + r"(uncorrrected, $\alpha=0.05$)",
        "ylim": (0, 50),
    },
    "fdr_p_values": {
        "var_name": "qcfc_fdr_significant",
        "label": "Percentage %",
        "title": "Significant QC-FC in connectomes\n"
        + r"(FDR corrected, $\alpha=0.05$)",
        "ylim": (0, 25),
    },
    "median": {
        "var_name": "qcfc_mad",
        "label": "Absolute values of QC-FC",
        "title": "Medians of absolute values of QC-FC",
        "ylim": (0, 0.15),
    },
    "distance": {
        "var_name": "corr_motion_distance",
        "label": "Pearson's correlation, absolute value",
        "title": "DM-FC",
        "ylim": (0, 0.65),
    },
    "modularity": {
        "var_name": "modularity",
        "label": "Mean modularity quality (a.u.)",
        "title": "Mean network modularity",
        "ylim": (0, 0.63),
    },
    "modularity_motion": {
        "var_name": "corr_motion_modularity",
        "label": "Pearson's correlation, absolute value",
        "title": "Correlation between motion and network modularity",
        "ylim": (0, 0.42),
    },
}


def load_data(path_root, datasets, criteria_name, fmriprep_version, measure_name):
    measure = measures[measure_name]
    data = {}
    for dataset in datasets:
        path_data = (
            path_root
            / f"{dataset}_{fmriprep_version.replace('.', '-')}_desc-{criteria_name}_summary.tsv"
        )
        df = pd.read_csv(path_data, sep="\t", index_col=[0, 1], header=[0, 1])
        selected_strategy = pd.DataFrame()
        for strategy in strategy_order:
            selected_strategy = pd.concat(
                (selected_strategy, df.loc[(slice(None), strategy), :])
            )
        id_vars = selected_strategy.index.names
        if "absolute" in measure["label"]:
            selected_strategy = selected_strategy[measure["var_name"]].abs()
        else:
            selected_strategy = selected_strategy[measure["var_name"]]
        data[dataset] = selected_strategy.reset_index().melt(
            id_vars=id_vars, value_name=measure["label"]
        )
    return data, measure


def plot_stats(data, measure, group="full_sample"):
    palette = sns.color_palette("colorblind", n_colors=7)
    paired_palette = [palette[0]]
    for p in palette[1:4]:
        paired_palette.extend((p, p))
    paired_palette.extend((palette[-3], palette[-2], palette[-1], palette[-1]))

    fig, axs = plt.subplots(1, 2, sharey=True, constrained_layout=True)
    fig.suptitle(
        measure["title"],
        weight="heavy",
        fontsize="x-large",
    )
    for i, dataset in enumerate(data):
        df = data[dataset].query(f"groups=='{group}'")
        baseline_values = df.query("strategy=='baseline'")
        baseline_mean = baseline_values[measure["label"]].mean()
        sns.barplot(
            y=measure["label"],
            x="strategy",
            data=df,
            ax=axs[i],
            order=strategy_order,
            ci=95,
            palette=paired_palette,
        )
        # horizontal line baseline
        axs[i].axhline(baseline_mean, ls="-.", c=paired_palette[0])
        axs[i].set_title(dataset)
        axs[i].set_ylim(measure["ylim"])
        axs[i].set_xticklabels(
            strategy_order, rotation=45, ha="right", rotation_mode="anchor"
        )
        for i, bar in enumerate(axs[i].patches):
            if i > 0 and i % 2 == 0 and i != 8:  # only give gsr hatch
                bar.set_hatch("///")
    labels = ["No GSR", "With GSR"]
    hatches = ["", "///"]
    handles = [
        mpatches.Patch(edgecolor="black", facecolor="white", hatch=h, label=l)
        for h, l in zip(hatches, labels)
    ]
    axs[1].legend(handles=handles)
    return fig


def plot_joint_scatter(path_root, dataset, base_strategy, fmriprep_version):
    """Joint scatter plot for mean frame wise displacement against network modularity."""
    parcel = "atlas-difumo_nroi-64"
    path_data = (
        path_root
        / dataset
        / fmriprep_version
        / f"dataset-{dataset}_{parcel}_modularity.tsv"
    )
    modularity = pd.read_csv(path_data, sep="\t", index_col=0)
    path_data = (
        path_root
        / dataset
        / fmriprep_version
        / f"dataset-{dataset}_desc-movement_phenotype.tsv"
    )
    motion = pd.read_csv(path_data, sep="\t", index_col=0)
    data = pd.concat([modularity, motion.loc[modularity.index, :]], axis=1)
    data = data.drop("groups", axis=1)
    data.index.name = "participants"
    data = data.reset_index()
    data = data.loc[
        :,
        [
            "participants",
            "mean_framewise_displacement",
            "baseline",
            base_strategy,
            f"{base_strategy}+gsr",
        ],
    ]
    data = data.melt(
        id_vars=["participants", "mean_framewise_displacement"],
        var_name="Strategy",
        value_name="Modularity quality (a.u.)",
    )
    data = data.rename(
        columns={"mean_framewise_displacement": "Mean Framewise Displacement (mm)"}
    )

    palette = sns.color_palette("colorblind", n_colors=3)
    p = sns.jointplot(
        data=data,
        x="Modularity quality (a.u.)",
        y="Mean Framewise Displacement (mm)",
        hue="Strategy",
        palette=palette,
    )
    p.fig.suptitle(
        f"Distribution of mean frame wise displacement\nagainst network modularity:\n{dataset} / {fmriprep_version}",
        weight="heavy",
    )
    p.fig.tight_layout()
    return p
