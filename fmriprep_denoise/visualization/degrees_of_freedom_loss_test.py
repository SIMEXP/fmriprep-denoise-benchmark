import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from fmriprep_denoise.features.derivatives import get_qc_criteria
from fmriprep_denoise.visualization import utils

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,  # Adjust the logging level if needed
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define the fixed order for strategies based on your GRID_LOCATION_dof.
STRATEGIES = list(utils.GRID_LOCATION.values())


def load_data(path_root, datasets, criteria_name, fmriprep_version):
    """
    For each dataset, load the confounds data and participant information,
    then process the raw MultiIndex DataFrame (with new CSV structure) into
    a melted (long-format) DataFrame for plotting.
    """
    logger.info("Starting load_data function")
    criteria = get_qc_criteria(criteria_name)
    logger.debug(f"QC criteria obtained: {criteria}")
    confounds_phenotypes = {}
    for dataset in datasets:
        logger.info(f"Processing dataset: {dataset}")
        (confounds_phenotype,
         participant_groups,
         groups) = utils._get_participants_groups(
            dataset,
            fmriprep_version,
            path_root,
            gross_fd=criteria["gross_fd"],
            fd_thresh=criteria["fd_thresh"],
            proportion_thresh=criteria["proportion_thresh"],
        )
        logger.debug("Loaded confounds and participant groups for dataset")
        # Process the raw DataFrame into long form.
        df_plotting, full_length = _organise_data(confounds_phenotype)
        logger.debug(f"Data organized for dataset {dataset} (full_length: {full_length})")
        confounds_phenotypes[dataset] = {
            "group_values": groups,
            "participant_labels": participant_groups,
            "confounds_phenotype": df_plotting,   # melted DataFrame for plotting
            "confounds_stats": confounds_phenotype,  # raw MultiIndex DataFrame for stats
            "full_length": full_length,
        }
        logger.info(f"Completed processing for dataset: {dataset}")
    logger.info("Completed load_data function")
    return confounds_phenotypes


def _organise_data(confounds_phenotype):
    """
    Update the raw confounds DataFrame (with MultiIndex columns) from the new CSV structure.
    Expected level-0 (strategy) names: e.g. "Motion", "MotionGSR", "ScrubGSR", "Scrub", "CompCor"
    and level-1 (sub-column) names: 
        excised_vol, excised_vol_proportion, high_pass, fixed_regressors, compcor, aroma, total, full_length.
    
    For each strategy in STRATEGIES, add the 'fixed_regressors' value to the 'aroma' and 'compcor' columns.
    Then, use one strategy's 'full_length' as the denominator and melt the DataFrame into long format.
    Finally, convert values to percentages.
    """
    logger.info("Starting _organise_data function")
    # Use the first strategy as reference to get full_length.
    ref_strat = STRATEGIES[0]
    full_length = confounds_phenotype[(ref_strat, "full_length")].iloc[0]
    logger.debug(f"Reference strategy: {ref_strat}, full_length: {full_length}")
    
    # For each strategy, update 'aroma' and 'compcor' by adding 'fixed_regressors'
    for strat in STRATEGIES:
        if (strat, "aroma") in confounds_phenotype.columns and (strat, "fixed_regressors") in confounds_phenotype.columns:
            logger.debug(f"Updating 'aroma' for strategy: {strat}")
            confounds_phenotype[(strat, "aroma")] = confounds_phenotype[(strat, "aroma")] + confounds_phenotype[(strat, "fixed_regressors")]
        if (strat, "compcor") in confounds_phenotype.columns and (strat, "fixed_regressors") in confounds_phenotype.columns:
            logger.debug(f"Updating 'compcor' for strategy: {strat}")
            confounds_phenotype[(strat, "compcor")] = confounds_phenotype[(strat, "compcor")] + confounds_phenotype[(strat, "fixed_regressors")]
        # Add additional strategy-specific adjustments if needed.
    
    # Reset index and melt the DataFrame into long form.
    logger.debug("Resetting index and melting DataFrame into long form")
    confounds_reset = confounds_phenotype.reset_index()
    melted = confounds_reset.melt(id_vars=["index"], var_name=["strategy", "type"], value_name="value")
    # Convert values to percentages relative to full_length.
    logger.debug("Converting values to percentages relative to full_length")
    melted["value"] = (melted["value"] / full_length) * 100
    melted = melted.set_index("index")
    logger.info("Completed _organise_data function")
    return melted, full_length


def _plot_single_report(confounds_df, full_length, palette, ax, title):
    """
    Create a barplot report for a given melted DataFrame.
    Each call to sns.barplot uses the explicit order STRATEGIES so that the x-axis
    ticks always match.
    """
    logger.info(f"Starting _plot_single_report for report: {title}")
    
    # Plot "total" (censored volumes)
    logger.debug("Plotting 'total' barplot")
    sns.barplot(
        x="strategy",
        y="value",
        data=confounds_df[confounds_df["type"] == "total"],
        ci=95,
        color=palette[4],
        linewidth=1,
        ax=ax,
        order=STRATEGIES,
    )
    # Plot "compcor" (CompCor regressors)
    logger.debug("Plotting 'compcor' barplot")
    sns.barplot(
        x="strategy",
        y="value",
        data=confounds_df[confounds_df["type"] == "compcor"],
        ci=95,
        color=palette[0],
        linewidth=1,
        ax=ax,
        order=STRATEGIES,
    )
    # Plot "aroma" (ICA-AROMA partial regressors)
    logger.debug("Plotting 'aroma' barplot")
    sns.barplot(
        x="strategy",
        y="value",
        data=confounds_df[confounds_df["type"] == "aroma"],
        ci=95,
        color=palette[2],
        linewidth=1,
        ax=ax,
        order=STRATEGIES,
    )
    # Plot "fixed_regressors" (head motion and tissue signal)
    logger.debug("Plotting 'fixed_regressors' barplot")
    sns.barplot(
        x="strategy",
        y="value",
        data=confounds_df[confounds_df["type"] == "fixed_regressors"],
        ci=95,
        color=palette[1],
        linewidth=1,
        ax=ax,
        order=STRATEGIES,
    )
    # Plot "high_pass" (discrete cosine-basis regressors)
    logger.debug("Plotting 'high_pass' barplot")
    sns.barplot(
        x="strategy",
        y="value",
        data=confounds_df[confounds_df["type"] == "high_pass"],
        ci=95,
        color=palette[3],
        linewidth=1,
        ax=ax,
        order=STRATEGIES,
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel(f"% Degrees of freedom loss\n(Full length: {full_length})")
    ax.set_title(title)
    logger.info(f"Completed _plot_single_report for report: {title}")
    return confounds_df


def plot_stats(confounds_phenotypes, plot_subgroup=False):
    """
    Generate plots for the degrees-of-freedom loss.
    """
    logger.info("Starting plot_stats function")
    sns.set_palette("colorblind")
    palette = sns.color_palette(n_colors=5)
    figsize = (11, 13) if plot_subgroup else (11, 5)
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    
    if plot_subgroup:
        logger.debug("Using subplot layout for subgroup plots")
        axs = fig.subplots(3, 3, sharey=True)
    else:
        logger.debug("Using subplot layout for non-subgroup plots")
        axs = fig.subplots(1, 2, sharey=True)
    
    fig.suptitle("Loss of temporal degrees of freedom", weight="heavy", fontsize="x-large")
    logger.info("Generating new graph")
    print("Generating new graph")
    
    if not plot_subgroup:
        key_axs = [axs[0], axs[1]]
        for ax, dataset in zip(key_axs, confounds_phenotypes):
            logger.info(f"Plotting stats for dataset: {dataset}")
            confounds_df = confounds_phenotypes[dataset]["confounds_phenotype"]
            _descriptive_stats(dataset, confounds_phenotypes[dataset]["confounds_stats"])
            n = confounds_phenotypes[dataset]["participant_labels"].shape[0]
            _plot_single_report(
                confounds_df,
                confounds_phenotypes[dataset]["full_length"],
                palette,
                ax,
                f"{dataset} (N={n})"
            )
    else:
        key_axs = [axs[0, 0], axs[1, 0]]
        for ax, dataset in zip(key_axs, confounds_phenotypes):
            logger.info(f"Plotting subgroup stats for dataset: {dataset}")
            confounds_df = confounds_phenotypes[dataset]["confounds_phenotype"]
            _descriptive_stats(dataset, confounds_phenotypes[dataset]["confounds_stats"])
            n = confounds_phenotypes[dataset]["participant_labels"].shape[0]
            _plot_single_report(
                confounds_df,
                confounds_phenotypes[dataset]["full_length"],
                palette,
                ax,
                f"{dataset} (N={n})"
            )
        starting_x = 0
        for dataset in confounds_phenotypes:
            subjects = confounds_phenotypes[dataset]["participant_labels"].index
            for i, group in enumerate(confounds_phenotypes[dataset]["group_values"]):
                selected = confounds_phenotypes[dataset]["participant_labels"] == group
                selected = subjects[selected]
                confounds_df = confounds_phenotypes[dataset]["confounds_phenotype"].loc[selected, :]
                ax = axs[starting_x, (i % 2) + 1]
                if i % 2 == 1:
                    starting_x += 1
                logger.info(f"Plotting subgroup for dataset: {dataset}, group: {group} (N={selected.shape[0]})")
                _plot_single_report(
                    confounds_df,
                    confounds_phenotypes[dataset]["full_length"],
                    palette,
                    ax,
                    f"{dataset}: {group} (N={selected.shape[0]})"
                )
    
    colors = [palette[4], palette[0], palette[2], palette[1], palette[3]]
    labels = [
        "Censored volumes",
        "CompCor regressors",
        "ICA-AROMA partial regressors",
        "Head motion and tissue signal",
        "Discrete cosine-basis regressors",
    ]
    handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    if plot_subgroup:
        axs[2, 0].legend(handles=handles, loc=0)
        axs[2, 0].axis('off')
    else:
        axs[1].legend(handles=handles, loc=0)
    logger.info("Completed plot_stats function")
    return fig


def _report_descriptive_stats(confounds_stats, strategy, variable):
    """
    Report descriptive statistics for a given strategy and variable from the raw DataFrame.
    """
    logger.info(f"Reporting descriptive stats for {strategy} {variable}")
    if (strategy, variable) in confounds_stats.columns:
        mean_val = confounds_stats[(strategy, variable)].mean()
        std_val = confounds_stats[(strategy, variable)].std()
        min_val = confounds_stats[(strategy, variable)].min()
        max_val = confounds_stats[(strategy, variable)].max()
        logger.debug(f"Stats for {strategy} {variable}: mean {mean_val}, std {std_val}, range {min_val} - {max_val}")
        print(f"Descriptive stats for {strategy} {variable}: {mean_val} ({std_val}); range = {min_val} - {max_val}")
    else:
        logger.warning(f"Column ({strategy}, {variable}) not found in confounds_stats.")
        print(f"Column ({strategy}, {variable}) not found in confounds_stats.")


def _descriptive_stats(dataset, confounds_stats):
    logger.info(f"Descriptive statistics for dataset: {dataset}")
    print(dataset)
    for strat in STRATEGIES:
        _report_descriptive_stats(confounds_stats, strat, "aroma")
        _report_descriptive_stats(confounds_stats, strat, "compcor")
        _report_descriptive_stats(confounds_stats, strat, "excised_vol")
        # Add additional variables as needed.


def _report_descriptive_stats(confounds_phenotype, strategy, variable):
    logger.info(f"Reporting descriptive stats for {strategy} {variable} from confounds_phenotype")
    mean = confounds_phenotype.loc[:, (strategy, variable)].mean()
    cmax = confounds_phenotype.loc[:, (strategy, variable)].max()
    cmin = confounds_phenotype.loc[:, (strategy, variable)].min()
    std = confounds_phenotype.loc[:, (strategy, variable)].std()
    logger.debug(f"Stats for {strategy} {variable}: mean {mean}, std {std}, range {cmin} - {cmax}")
    print(
        f"descriptive stats for {strategy} {variable}: {mean}({std}); range={cmin} - {cmax}"
    )