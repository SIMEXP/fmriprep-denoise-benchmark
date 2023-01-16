import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fmriprep_denoise.visualization import utils

strategy_order = list(utils.GRID_LOCATION.values())

measures = {
    "p_values": {
        "var_name": "qcfc_significant",
        "label": "Percentage %",
        "title": "Significant QC-FC in connectomes\n"
        + r"(uncorrrected, $\alpha=0.05$)",
        "ylim": (0, 60),
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
        "ylim": (0, 0.25),
    },
    "distance": {
        "var_name": "corr_motion_distance",
        "label": "Pearson's correlation, absolute value",
        "title": "Distance-dependent of motion",
        "ylim": (0, 0.33),
    },
    "modularity": {
        "var_name": "modularity",
        "label": "Mean modularity quality (a.u.)",
        "title": "Mean network modularity",
        "ylim": (0, 0.52),
    },
    "modularity_motion": {
        "var_name": "corr_motion_modularity",
        "label": "Pearson's correlation, absolute value",
        "title": "Correlation between motion and network modularity",
        "ylim": (0, 0.62),
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
        id_vars = df.index.names
        if "absolute" in measure["label"]:
            df = df[measure["var_name"]].abs()
        else:
            df = df[measure["var_name"]]
        data[dataset] = df.reset_index().melt(
            id_vars=id_vars, value_name=measure["label"]
        )
    return data, measure


def plot_stats(data, measure):
    fig, axs = plt.subplots(1, 2, sharey=True, constrained_layout=True)
    fig.suptitle(
        measure["title"],
        weight="heavy",
        fontsize="x-large",
    )
    for i, dataset in enumerate(data):
        sns.barplot(
            y=measure["label"],
            x="strategy",
            data=data[dataset],
            ax=axs[i],
            order=strategy_order,
            ci=95,
        )
        axs[i].set_title(dataset)
        axs[i].set_ylim(measure["ylim"])
        axs[i].set_xticklabels(
            strategy_order, rotation=45, ha="right", rotation_mode="anchor"
        )
    return fig
