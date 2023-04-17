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
        stats[dataset] = _statistic_report_group(dataset, baseline_group, data)
    return stats


def _statistic_report_group(dataset, baseline_group, data):
    """Mean framewise displacement t test between groups, baseline sex differences."""
    print(f"===={dataset}====")
    for_plotting = {"dataframe": data, "stats_group": {}}
    baseline = data[data["groups"] == baseline_group]
    for i, group in enumerate(group_order[dataset]):
        if group != baseline_group:
            compare = data[data["groups"] == group]
            t_stats, pval, df = ttest_ind(
                baseline["mean_framewise_displacement"],
                compare["mean_framewise_displacement"],
                usevar="pooled",
            )
            for_plotting["stats_group"].update(
                {i: {"t_stats": t_stats, "p_value": pval, "df": df}}
            )
            print(group, for_plotting["stats_group"][i])

    male = baseline[baseline["gender"] == 0]
    female = baseline[baseline["gender"] == 1]
    print(
        f"male M = {male['mean_framewise_displacement'].mean()}\n"
        f"\tSD = {male['mean_framewise_displacement'].std()}"
    )
    print(
        f"female M = {female['mean_framewise_displacement'].mean()}\n"
        f"\tSD = {female['mean_framewise_displacement'].std()}"
    )
    t_stats, pval, df = ttest_ind(
        male["mean_framewise_displacement"],
        female["mean_framewise_displacement"],
        usevar="pooled",
    )
    for_plotting["stats_sex"] = {"t_stats": t_stats, "p_value": pval, "df": df}
    print("male vs female: ", for_plotting["stats_sex"])
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
    fig = plt.figure(figsize=(7.5, 9))
    axs = fig.subplots(2, 2, sharey=True)
    # by group
    sns.set_palette("colorblind")
    palette = sns.color_palette(n_colors=7)
    colors_fd = {
        "ds000228": [palette[0], palette[1]],
        "ds000030": [palette[0]] + palette[2:5],
    }
    colors_sex = palette[5:]
    for ax, dataset in zip(axs[0], datasets):
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
            palette=colors_fd[dataset],
        )
        ax.set_xticklabels(
            group_order[dataset],
            # rotation=45, ha="right", rotation_mode="anchor"
        )
        ax.set_title(
            f"{dataset}\n"
            f"Motion in dataset subgroups\n"
            f"Mean\u00B1SD={mean_fd:.2f}\u00B1{sd_fd:.2f}; $N={df.shape[0]}$"
        )

        # statistical annotation
        max_value = df["Mean Framewise Displacement (mm)"].max()
        stacker = 0
        for i in stats[dataset]["stats_group"]:
            notation = _get_pvalue_star(stats[dataset]["stats_group"][i]["p_value"])
            if stats[dataset]["stats_group"][i]["p_value"] < 0.05:
                stacker += 1
                _significant_notation((0, i), max_value + 0.02 * stacker, notation, ax)

    # by sex in baseline only
    for ax, dataset in zip(axs[1], datasets):
        df = stats[dataset]["dataframe"]
        df = df[df["groups"] == datasets_baseline[dataset]]
        mean_fd = df["mean_framewise_displacement"].mean()
        sd_fd = df["mean_framewise_displacement"].std()

        df.loc[df["gender"] == 1, "gender"] = "Female"
        df.loc[df["gender"] == 0, "gender"] = "Male"

        df = df.rename(
            columns={
                "mean_framewise_displacement": "Mean Framewise Displacement (mm)",
                "gender": "Sex",
            }
        )
        sns.boxplot(
            y="Mean Framewise Displacement (mm)",
            x="Sex",
            data=df,
            ax=ax,
            order=["Male", "Female"],
            palette=colors_sex,
        )
        ax.set_xticklabels(
            ["Male", "Female"],
            # rotation=45, ha="right", rotation_mode="anchor"
        )
        ax.set_title(f"Sex difference in {datasets_baseline[dataset]}")
        # statistical annotation
        max_value = df["Mean Framewise Displacement (mm)"].max()
        notation = _get_pvalue_star(stats[dataset]["stats_sex"]["p_value"])

        if stats[dataset]["stats_sex"]["p_value"] < 0.05:
            _significant_notation((0, 1), max_value, notation, ax)
    fig.tight_layout()
    return fig


def _get_pvalue_star(p_value):
    if p_value < 0.005:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return None
