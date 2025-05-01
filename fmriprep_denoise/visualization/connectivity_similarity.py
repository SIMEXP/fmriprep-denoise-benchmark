import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nilearn.plotting.matrix_plotting import _reorder_matrix
from fmriprep_denoise.visualization import utils
import itertools
from pathlib import Path

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,  # Adjust level as needed
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

strategies = list(utils.GRID_LOCATION.values())


def load_data(path_root, datasets, fmriprep_version):
    """Average connectome similarity across different parcellations."""
    logger.info("Starting load_data function")
    average_connectomes = {}
    for dataset in datasets:
        logger.info(f"Processing dataset: {dataset}")
        connectomes_path = path_root.glob(
            f"{dataset}/{fmriprep_version}/*connectome.tsv"
        )
        connectomes_correlations = []
        for p in connectomes_path:
            logger.debug(f"Loading connectivity file from: {p}")
            df = pd.read_csv(p, sep="\t", index_col=0)
            logger.debug(f"Columns in loaded file: {df.columns.tolist()}")
            cc = df[strategies]
            corr_df = cc.corr()
            corr_matrix = cc.corr().values
            # 1. Mean and Std deviation per strategy
            mean_std_summary = cc.agg(['mean', 'std']).T
            logger.info(f"Mean and Std per strategy for {p.name}:\n{mean_std_summary.to_string(float_format=lambda x: f'{x:.6f}')}")

            # 2. Pairwise differences between strategies
            for s1 in strategies:
                for s2 in strategies:
                    if s1 != s2:
                        diff = cc[s1] - cc[s2]
                        logger.debug(f"Difference summary {s1} - {s2}: mean={diff.mean():.6f}, std={diff.std():.6f}")

            # 3. Save small scatterplot for visual inspection
            scatter_save_dir = Path("/home/seann/scratch/denoise/fmriprep-denoise-benchmark/outputs")  # <<< CHANGE THIS to your desired path
            scatter_save_dir.mkdir(parents=True, exist_ok=True)  # Create the folder if it doesn't exist

            for s1, s2 in itertools.combinations(strategies, 2):
                plt.figure()
                plt.scatter(cc[s1], cc[s2], alpha=0.5)
                plt.xlabel(s1)
                plt.ylabel(s2)
                plt.title(f"{s1} vs {s2}")
                
                save_filename = scatter_save_dir / f"{p.stem}_{s1}_vs_{s2}.png"
                plt.savefig(save_filename)
                plt.close()
                logger.info(f"Saved scatterplot: {save_filename}")

            logger.debug(f"Calculated correlation matrix with shape: {corr_df.shape}")
            logger.info(f"Correlations between strategies for {p.name}:\n{corr_df.to_string(float_format=lambda x: f'{x:.3f}')}")

            connectomes_correlations.append(corr_df.values)
            logger.debug(f"First few rows of input df:\n{df.head()}")
            logger.debug(f"First few of cc:\n{cc.head()}")
            logger.debug(f"Correlation matrix:\n{corr_matrix}")
        average_connectome_values = np.mean(connectomes_correlations, axis=0)
        logger.debug(f"Computed average connectome shape: {average_connectome_values.shape}")
        average_connectome = pd.DataFrame(
            average_connectome_values,
            columns=cc.columns,
            index=cc.columns,
        )
        average_connectomes[dataset] = average_connectome
        logger.info(f"Completed processing for dataset: {dataset}")

    # Average the computed connectomes and perform clustering
    logger.info("Computing overall average connectome for clustering")
    combined_average = np.mean(list(average_connectomes.values()), axis=0)
    logger.debug(f"Combined average matrix shape: {combined_average.shape}")
    _, labels = _reorder_matrix(
        combined_average,
        list(cc.columns),
        "complete",
    )
    logger.debug(f"Matrix reordered; labels after clustering: {labels}")

    reordered_connectomes = {d: average_connectomes[d].loc[labels, labels] for d in average_connectomes}
    logger.info("Completed load_data function")
    return reordered_connectomes


def plot_stats(average_connectomes, horizontal=False, strategy_order=None, show_colorbar=True):
    """
    Plot heatmaps for connectome similarity amongst denoising methods.
    Color scale is shared and automatically scaled to min/max values across all datasets.
    """
    logger.info("Starting plot_stats function")

    # Compute global vmin and vmax
    all_values = np.concatenate([
        df.values.flatten() for df in average_connectomes.values()
    ])
    vmin, vmax = np.min(all_values), np.max(all_values)
    logger.debug(f"Global vmin: {vmin:.3f}, vmax: {vmax:.3f}")
    # vmin, vmax = 0.5, 1.0
    logger.debug(f"Using fixed color scale: vmin={vmin}, vmax={vmax}")

    if horizontal:
        fig_similarity, axs_similarity = plt.subplots(
            2, 1, figsize=(4.8, 10.3), constrained_layout=True
        )
    else:
        fig_similarity, axs_similarity = plt.subplots(
            1, 2, figsize=(10.3, 4.8), constrained_layout=True
        )

    axs_similarity = np.atleast_1d(axs_similarity)

    fig_similarity.suptitle(
        "Similarity of denoised connectomes by strategy",
        weight="heavy",
        fontsize="x-large",
    )

    for i, dataset in enumerate(average_connectomes):
        logger.info(f"Plotting heatmap for dataset: {dataset}")
        df = average_connectomes[dataset]

        # Enforce strategy order if provided
        if strategy_order is not None:
            missing_strats = set(strategy_order) - set(df.columns)
            if missing_strats:
                logger.warning(f"Missing strategies in dataset {dataset}: {missing_strats}")
            common_order = [s for s in strategy_order if s in df.columns]
            df = df.loc[common_order, common_order]

        sns.heatmap(
            df,
            square=True,
            ax=axs_similarity[i],
            linewidth=0.5,
            cbar=show_colorbar,
            vmin=vmin,
            vmax=vmax,
        )

        axs_similarity[i].set_title(dataset)
        axs_similarity[i].set_xticklabels(
            df.columns, rotation=45, ha="right", rotation_mode="anchor"
        )
        axs_similarity[i].set_yticklabels(df.index, rotation=0)

        logger.debug(f"Heatmap plotted for dataset {dataset} with shape {df.shape}")

    logger.info("Completed plot_stats function")
    return fig_similarity