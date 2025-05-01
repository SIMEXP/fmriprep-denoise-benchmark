import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from fmriprep_denoise.visualization import utils

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,  # Adjust the level if needed
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

strategy_order = list(utils.GRID_LOCATION.values())

measures = {
    "p_values": {
        "var_name": "qcfc_significant",
        "label": "Percentage %",
        "title": "Significant QC-FC in connectomes\n"
                 + r"(uncorrrected, $\alpha=0.05$)",
        "ylim": None,
    },
    "fdr_p_values": {
        "var_name": "qcfc_fdr_significant",
        "label": "Percentage %",
        "title": "Significant QC-FC in connectomes\n"
                 + r"(FDR corrected, $\alpha=0.05$)",
        "ylim": None,
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
        "title": "DM-FC",
        "ylim": None,
    },
    "modularity": {
        "var_name": "modularity",
        "label": "Mean modularity quality (a.u.)",
        "title": "Mean network modularity",
        "ylim": None,
    },
    "modularity_motion": {
        "var_name": "corr_motion_modularity",
        "label": "Pearson's correlation, absolute value",
        "title": "Correlation between motion and network modularity",
        "ylim": None,
    },
}


def load_data(path_root, datasets, criteria_name, fmriprep_version, measure_name):
    logger.info("Starting load_data function")
    measure = measures[measure_name]
    data = {}

    for dataset in datasets:
        logger.info(f"Processing dataset: {dataset}")
        path_data = (
            path_root
            / f"{dataset}_{fmriprep_version.replace('.', '-')}_desc-{criteria_name}_summary.tsv"
        )
        logger.debug(f"Loading data from {path_data}")
        
        try:
            df = pd.read_csv(path_data, sep="\t", index_col=[0, 1], header=[0, 1])
            df.index.names = ["group", "strategy"]  # 👈 manually set index names
        except Exception as e:
            logger.error(f"Failed to read data file for {dataset}: {e}")
            continue
        logger.debug(f"[{dataset}] df.index.names: {df.index.names}")
        logger.debug(f"[{dataset}] df.index example values: {df.index.tolist()[:5]}")        
        logger.debug(f"Data loaded for {dataset} with shape: {df.shape}")
        logger.debug(f"Available strategies in loaded data: {df.index.get_level_values(1).unique().tolist()}")
        logger.debug(f"strategy_order from GRID_LOCATION: {strategy_order}")

        available_strategies = df.index.get_level_values(1).unique().tolist()
        selected_strategy = pd.DataFrame()

        for strategy in strategy_order:
            matched_rows = df.loc[(slice(None), strategy), :]
            logger.debug(f"[{dataset}] Strategy '{strategy}' returned {matched_rows.shape[0]} rows")

        for strategy in strategy_order:
            if strategy in available_strategies:
                logger.debug(f"Selecting strategy: {strategy}")
                selected_strategy = pd.concat(
                    (selected_strategy, df.loc[(slice(None), strategy), :])
                )
            else:
                logger.warning(f"Strategy '{strategy}' not found in data index. Skipping.")

        # Validate index before melt
        id_vars = selected_strategy.index.names
        if selected_strategy.empty or None in id_vars:
            logger.warning(f"[{dataset}] Skipping due to empty or malformed selected_strategy.")
            continue

        logger.debug(f"Index variables for melt: {id_vars}")

        logger.debug(f"[{dataset}] Columns in selected_strategy: {selected_strategy.columns.tolist()}")


        # Find column matching the requested measure
        matching_columns = [col for col in selected_strategy.columns if col[0] == measure["var_name"]]

        if not matching_columns:
            logger.error(f"[{dataset}] No column found with measure name '{measure['var_name']}'")
            continue

        # Use the first matching column (there should only be one per measure)
        target_column = matching_columns[0]
        logger.debug(f"[{dataset}] Using target column: {target_column}")

        # Select and optionally apply abs
        if "absolute" in measure["label"]:
            selected_strategy = selected_strategy[target_column].abs()
        else:
            selected_strategy = selected_strategy[target_column]

        # 💥 This was missing: add the melted result to the data dict
        melted = selected_strategy.reset_index().melt(
            id_vars=id_vars, value_name=measure["label"]
        )

        melted["groups"] = melted["group"]

        data[dataset] = melted
        logger.debug(f"[{dataset}] Added melted data with shape {melted.shape}")

    logger.info("Completed load_data function")
    logger.debug(f"Returning data keys: {list(data.keys())}")
    return data, measure


def plot_stats(data, measure, group="full_sample"):
    logger.info("Starting plot_stats function")
    palette = sns.color_palette("colorblind", n_colors=7)
    paired_palette = [palette[0]]
    for p in palette[1:4]:
        paired_palette.extend((p, p))
    paired_palette.extend((palette[-3], palette[-2], palette[-1], palette[-1]))
    logger.debug(f"Paired palette: {paired_palette}")

    fig, axs = plt.subplots(1, 2, sharey=True, constrained_layout=True)
    fig.suptitle(
        measure["title"],
        weight="heavy",
        fontsize="x-large",
    )
    logger.debug("Created figure and subplots for plot_stats")

    for idx, dataset in enumerate(data):
        logger.info(f"Plotting statistics for dataset: {dataset}")
        df = data[dataset].query(f"groups=='{group}'")
        baseline_values = df.query("strategy=='baseline'")
        baseline_mean = baseline_values[measure["label"]].mean()
        logger.debug(f"Baseline mean for dataset {dataset}: {baseline_mean}")
        
        sns.barplot(
            y=measure["label"],
            x="strategy",
            data=df,
            ax=axs[idx],
            order=strategy_order,
            ci=95,
            palette=paired_palette,
        )
        axs[idx].axhline(baseline_mean, ls="-.", c=paired_palette[0])
        axs[idx].set_title(dataset)

        # Dynamic Y-axis scaling
        ymax = df[measure["label"]].max()
        axs[idx].set_ylim(0, ymax * 1.5)  # Add 10% headroom

        axs[idx].set_xticklabels(
            strategy_order, rotation=45, ha="right", rotation_mode="anchor"
        )
        logger.debug(f"Applied settings for axes of dataset {dataset}")

        for i, bar in enumerate(axs[idx].patches):
            if i > 0 and i % 2 == 0 and i != 8:
                logger.debug(f"Setting hatch for bar index {i} in dataset {dataset}")
                bar.set_hatch("///")

    labels = ["No GSR", "With GSR"]
    hatches = ["", "///"]
    handles = [
        mpatches.Patch(edgecolor="black", facecolor="white", hatch=h, label=l)
        for h, l in zip(hatches, labels)
    ]
    axs[1].legend(handles=handles)
    
    logger.info("Completed plot_stats function")
    return fig

def plot_joint_scatter(path_root, dataset, base_strategy, fmriprep_version, parcel=None):
    logger.info("Starting plot_joint_scatter function")
    
    if parcel is None:
        # Try to find the parcel automatically
        parcel_dir = path_root / dataset / fmriprep_version
        tsvs = list(parcel_dir.glob(f"dataset-{dataset}_atlas-*_modularity.tsv"))
        if len(tsvs) == 0:
            raise FileNotFoundError(f"No modularity TSV found in {parcel_dir}")
        # Extract parcel name from filename
        parcel = tsvs[0].name.split(f"dataset-{dataset}_")[1].split("_modularity.tsv")[0]
        logger.debug(f"Inferred parcel: {parcel}")
    else:
        logger.debug(f"Using provided parcel: {parcel}")

    # Build file path for modularity data.
    path_modularity = (
        path_root
        / dataset
        / fmriprep_version
        / f"dataset-{dataset}_{parcel}_modularity.tsv"
    )
    logger.debug(f"Reading modularity data from {path_modularity}")
    modularity = pd.read_csv(path_modularity, sep="\t", index_col=0)

    # Build file path for movement phenotype data.
    path_motion = (
        path_root
        / dataset
        / fmriprep_version
        / f"dataset-{dataset}_desc-movement_phenotype.tsv"
    )
    logger.debug(f"Reading movement phenotype data from {path_motion}")
    motion = pd.read_csv(path_motion, sep="\t", index_col=0)
    
    # Merge the two DataFrames (assumes they share the same index).
    data = pd.concat([modularity, motion.loc[modularity.index, :]], axis=1)
    logger.debug(f"Merged data shape: {data.shape}")
    
    # If there's a column "groups", drop it.
    if "groups" in data.columns:
        logger.debug("Column 'groups' found in data; dropping it")
        data = data.drop("groups", axis=1)
    
    data.index.name = "participants"
    data = data.reset_index()
    logger.debug("Reset index after merging data")

    # # Also define the GSR-related strategy (simply appending "GSR").
    gsr_strategy = base_strategy + "+gsr"
    logger.debug(f"Defined GSR strategy: {gsr_strategy}")
    
    # Select the relevant columns.
    data = data.loc[
        :,
        [
            "participants",
            "mean_framewise_displacement",
            "baseline",
            base_strategy,
            gsr_strategy,
        ],
    ]
    logger.debug("Selected relevant columns for joint scatter")
    
    # Melt the DataFrame so that we have a column 'Strategy' with appropriate values.
    data = data.melt(
        id_vars=["participants", "mean_framewise_displacement"],
        var_name="Strategy",
        value_name="Modularity quality (a.u.)",
    )
    logger.debug("Melted DataFrame for joint scatter")
    
    # Rename the displacement column for clarity.
    data = data.rename(
        columns={"mean_framewise_displacement": "Mean Framewise Displacement (mm)"}
    )
    logger.debug("Renamed displacement column")
    
    # Automatically choose the correct number of colors based on unique strategies
    n_strategies = data["Strategy"].nunique()
    palette = sns.color_palette("colorblind", n_colors=n_strategies)

    p = sns.jointplot(
        data=data,
        x="Modularity quality (a.u.)",
        y="Mean Framewise Displacement (mm)",
        hue="Strategy",
        palette=palette,
    )
    p.fig.suptitle(
        f"Distribution of mean framewise displacement\nagainst network modularity:\n{dataset} / {fmriprep_version}",
        weight="heavy",
    )
    p.fig.tight_layout()
    logger.info("Completed plot_joint_scatter function")
    return p