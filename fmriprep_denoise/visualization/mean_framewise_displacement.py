import matplotlib.pyplot as plt

import seaborn as sns

from statsmodels.stats.weightstats import ttest_ind

from fmriprep_denoise.visualization import tables
from fmriprep_denoise.features.derivatives import get_qc_criteria


group_order = {
    "ds000228": ["adult", "child"],
    "ds000030": ["control", "ADHD", "bipolar", "schizophrenia"],
}
datasets = ["ds000228", "ds000030"]
datasets_baseline = {"ds000228": "adult", "ds000030": "control"}


def load_data(path_root, criteria_name, fmriprep_version):
    criteria = get_qc_criteria(criteria_name)
    stats = {}
    for dataset in datasets:
        baseline_group = datasets_baseline[dataset]
        _, data, _ = tables.get_descriptive_data(
            dataset, fmriprep_version, path_root, **criteria
        )
        stats[dataset] = _statistic_report(dataset, baseline_group, data)
    return stats


def _statistic_report(dataset, baseline_group, data):
    """Mean framewise displacement t test between groups."""
    for_plotting = {"dataframe": data, "stats": {}}
    baseline = data[data["groups"] == baseline_group]
    for i, group in enumerate(group_order[dataset]):
        if group != baseline_group:
            compare = data[data["groups"] == group]
            t_stats, pval, df = ttest_ind(
                baseline["mean_framewise_displacement"],
                compare["mean_framewise_displacement"],
                usevar="unequal",
            )
            for_plotting["stats"].update(
                {i: {"t_stats": t_stats, "p_value": pval, "df": df}}
            )

    return for_plotting


def _significant_notation(item_pairs, max_value, sig, ax):
    """Plot significant notions."""
    x1, x2 = item_pairs
    y, h, col = max_value + 0.01, 0.01, "k"
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    ax.text((x1 + x2) * 0.5, y + h, sig, ha="center", va="bottom", color=col)


def plot_stats(stats):
    """Plot mean framewise displacement by dataset."""
    print("Generating new graphs...")
    fig = plt.figure(figsize=(7, 5))
    axs = fig.subplots(1, 2, sharey=True)
    for ax, dataset in zip(axs, datasets):
        df = stats[dataset]["dataframe"]
        mean_fd = df["mean_framewise_displacement"].mean()
        sd_fd = df["mean_framewise_displacement"].std()
        df = df.rename(
            columns={
                "mean_framewise_displacement": "Mean Framewise Displacement (mm)",
                "groups": "Groups",
            }
        )
        sns.boxplot(
            y="Mean Framewise Displacement (mm)",
            x="Groups",
            data=df,
            ax=ax,
            order=group_order[dataset],
        )
        ax.set_xticklabels(
            group_order[dataset], rotation=45, ha="right", rotation_mode="anchor"
        )
        ax.set_title(
            f"{dataset}\nMean\u00B1SD={mean_fd:.2f}\u00B1{sd_fd:.2f}\n$N={df.shape[0]}$"
        )

        # statistical annotation
        max_value = df["Mean Framewise Displacement (mm)"].max()
        for i in stats[dataset]["stats"]:
            if stats[dataset]["stats"][i]["p_value"] < 0.005:
                notation = "***"
            elif stats[dataset]["stats"][i]["p_value"] < 0.01:
                notation = "**"
            elif stats[dataset]["stats"][i]["p_value"] < 0.05:
                notation = "*"
            else:
                notation = None

            if stats[dataset]["stats"][i]["p_value"] < 0.05:
                _significant_notation((0, i), max_value + 0.03 * (i - 1), notation, ax)
    return fig
