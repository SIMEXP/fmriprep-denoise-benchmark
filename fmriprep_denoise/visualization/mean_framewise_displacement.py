import logging
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.weightstats import ttest_ind
from fmriprep_denoise.visualization import tables
from fmriprep_denoise.features.derivatives import get_qc_criteria

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,  # Adjust the logging level as needed
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

group_order = {
    "ds000228": ["adult", "child"]
}
datasets = ["ds000228"]
datasets_baseline = {"ds000228": "adult"}


def load_data(path_root, criteria_name, fmriprep_version):
    logger.info("Starting load_data function")
    criteria = get_qc_criteria(criteria_name)
    logger.debug(f"QC criteria obtained: {criteria}")
    stats = {}
    for dataset in datasets:
        logger.info(f"Processing dataset: {dataset}")
        baseline_group = datasets_baseline[dataset]
        logger.debug(f"Baseline group for dataset {dataset}: {baseline_group}")
        _, data, _ = tables.get_descriptive_data(
            dataset, fmriprep_version, path_root, **criteria
        )
        logger.debug("Descriptive data obtained for dataset")
        stats[dataset] = _statistic_report_group(dataset, baseline_group, data)
        logger.info(f"Completed statistical report for dataset: {dataset}")
    logger.info("Completed load_data function")
    return stats


def _statistic_report_group(dataset, baseline_group, data):
    logger.info(f"Starting _statistic_report_group for dataset: {dataset}")
    print(f"===={dataset}====")
    for_plotting = {"dataframe": data, "stats_group": {}}
    baseline = data[data["groups"] == baseline_group]
    logger.debug(f"Baseline data extracted for group: {baseline_group}")
    for i, group in enumerate(group_order[dataset]):
        if group != baseline_group:
            logger.info(f"Performing t-test between baseline ({baseline_group}) and group: {group}")
            compare = data[data["groups"] == group]
            t_stats, pval, df = ttest_ind(
                baseline["mean_framewise_displacement"],
                compare["mean_framewise_displacement"],
                usevar="pooled",
            )
            for_plotting["stats_group"].update(
                {i: {"t_stats": t_stats, "p_value": pval, "df": df}}
            )
            logger.debug(f"t-test result for {group}: t={t_stats}, p={pval}, df={df}")
            print(group, for_plotting["stats_group"][i])
    # Report baseline sex differences
    male = baseline[baseline["gender"] == 0]
    female = baseline[baseline["gender"] == 1]
    male_mean = male["mean_framewise_displacement"].mean()
    male_std = male["mean_framewise_displacement"].std()
    female_mean = female["mean_framewise_displacement"].mean()
    female_std = female["mean_framewise_displacement"].std()
    logger.debug(f"Male mean: {male_mean}, SD: {male_std}")
    logger.debug(f"Female mean: {female_mean}, SD: {female_std}")
    print(
        f"male M = {male_mean}\n"
        f"\tSD = {male_std}"
    )
    print(
        f"female M = {female_mean}\n"
        f"\tSD = {female_std}"
    )
    t_stats, pval, df = ttest_ind(
        male["mean_framewise_displacement"],
        female["mean_framewise_displacement"],
        usevar="pooled",
    )
    for_plotting["stats_sex"] = {"t_stats": t_stats, "p_value": pval, "df": df}
    logger.debug(f"Sex differences t-test: t={t_stats}, p={pval}, df={df}")
    print("male vs female: ", for_plotting["stats_sex"])
    logger.info(f"Completed _statistic_report_group for dataset: {dataset}")
    return for_plotting


def _significant_notation(item_pairs, max_value, sig, ax):
    logger.debug(f"Adding significance notation: {sig} between {item_pairs} at height {max_value}")
    x1, x2 = item_pairs
    y, h, col = max_value + 0.01, 0.01, "k"
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    ax.text((x1 + x2) * 0.5, y + h, sig, ha="center", va="bottom", color=col)


def plot_stats(stats):
    logger.info("Starting plot_stats function")
    print("Generating new graphs...")
    fig = plt.figure(figsize=(7.5, 9))
    axs = fig.subplots(2, 2, sharey=True)
    sns.set_palette("colorblind")
    palette = sns.color_palette(n_colors=7)
    colors_fd = {
        "ds000228": [palette[0], palette[1]],
        "ds000030": [palette[0]] + palette[2:5],
    }
    colors_sex = palette[5:]
    # Boxplots by group
    for ax, dataset in zip(axs[0], datasets):
        logger.info(f"Plotting boxplot for dataset: {dataset} by group")
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
        ax.set_xticklabels(group_order[dataset])
        ax.set_title(
            f"{dataset}\n"
            f"Motion in dataset subgroups\n"
            f"Mean\u00B1SD={mean_fd:.2f}\u00B1{sd_fd:.2f}; $N={df.shape[0]}$"
        )
        # Statistical annotation for group comparisons
        max_value = df["Mean Framewise Displacement (mm)"].max()
        stacker = 0
        for i in stats[dataset]["stats_group"]:
            notation = _get_pvalue_star(stats[dataset]["stats_group"][i]["p_value"])
            if stats[dataset]["stats_group"][i]["p_value"] < 0.05:
                stacker += 1
                _significant_notation((0, i), max_value + 0.02 * stacker, notation, ax)
                logger.debug(f"Added significance notation for dataset {dataset} on group index {i}")
    # Boxplots by sex in baseline only
    for ax, dataset in zip(axs[1], datasets):
        logger.info(f"Plotting boxplot for dataset: {dataset} by sex")
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
        ax.set_xticklabels(["Male", "Female"])
        ax.set_title(f"Sex difference in {datasets_baseline[dataset]}")
        # Statistical annotation for sex comparisons
        max_value = df["Mean Framewise Displacement (mm)"].max()
        notation = _get_pvalue_star(stats[dataset]["stats_sex"]["p_value"])
        if stats[dataset]["stats_sex"]["p_value"] < 0.05:
            _significant_notation((0, 1), max_value, notation, ax)
            logger.debug(f"Added sex difference significance notation for dataset {dataset}")
    fig.tight_layout()
    logger.info("Completed plot_stats function")
    return fig


def _get_pvalue_star(p_value):
    logger.debug(f"Evaluating p-value: {p_value}")
    if p_value < 0.005:
        notation = "***"
    elif p_value < 0.01:
        notation = "**"
    elif p_value < 0.05:
        notation = "*"
    else:
        notation = None
    logger.debug(f"Significance notation determined: {notation}")
    return notation