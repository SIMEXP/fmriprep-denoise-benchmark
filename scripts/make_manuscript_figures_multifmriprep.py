from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from adjustText import adjust_text
import logging
from fmriprep_denoise.visualization import utils
from fmriprep_denoise.visualization import (
    connectivity_similarity,
    degrees_of_freedom_loss,
    mean_framewise_displacement,
    motion_metrics,
    strategy_ranking,
)

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Settings
fixed_dataset = "ds000228"
datasets = [fixed_dataset]  # Keep as list for function calls
fmriprep_versions = ["fmriprep-20.2.7", "fmriprep-25.0.0"]
criteria_name = "minimal"
excluded_strategies = []

# Paths
path_root = Path("/home/seann/scratch/denoise/fmriprep-denoise-benchmark/outputs/denoise-metrics-atlas.5-4.24.25")
output_dir = Path(__file__).parents[1] / "outputs1"
output_dir.mkdir(parents=True, exist_ok=True)

# Utility
strategy_order = list(utils.GRID_LOCATION.values())
strategy_order = [s for s in strategy_order if s not in excluded_strategies]

# Plotting function for Loss vs Quality
def plot_loss_vs_quality(data, output_dir, version_filter):
    sns.set(style="whitegrid", context="talk")
    filtered_df = data.xs(version_filter, level='version')
    results = []

    for dataset in filtered_df.index.levels[0]:
        subset = filtered_df.loc[dataset]
        loss_df = subset.loc['loss_df']
        other_metrics = subset.drop('loss_df')

        if isinstance(other_metrics.index, pd.MultiIndex) and "strategy" in other_metrics.index.names:
            strategy_idx = other_metrics.index.get_level_values("strategy")
            other_metrics = other_metrics.loc[~strategy_idx.isin(excluded_strategies)]
        else:
            other_metrics = other_metrics.loc[~other_metrics.index.isin(excluded_strategies)]

        mean_other = other_metrics.mean()
        results.append((dataset, loss_df, mean_other))

    for dataset, loss_df, mean_other in results:
        loss_df = loss_df[[s for s in loss_df.index if (s[1] if isinstance(s, tuple) else s) not in excluded_strategies]]
        mean_other = mean_other[[s for s in mean_other.index if (s[1] if isinstance(s, tuple) else s) not in excluded_strategies]]

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(x=loss_df, y=mean_other, s=100, color="royalblue", ax=ax)

        texts = []
        for strategy in loss_df.index:
            label = strategy[1] if isinstance(strategy, tuple) else strategy
            texts.append(ax.text(loss_df[strategy], mean_other[strategy] + 0.1, label, fontsize=12, ha="center"))

        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))

        ax.set_title(f"{dataset} ({version_filter})\nLoss of DF vs Mean Rank of Other Metrics", fontsize=16)
        ax.set_xlabel("Loss of Degrees of Freedom (Rank)", fontsize=14)
        ax.set_ylabel("Mean Rank of Denoising Quality Metrics", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()

        fig.savefig(output_dir / f"{dataset}_loss_vs_quality_{version_filter}.png", transparent=True)
        plt.close(fig)

# MAIN
if __name__ == "__main__":

    for version in fmriprep_versions:

        # Mean Framewise Displacement
        stats = mean_framewise_displacement.load_data(path_root, criteria_name, version)
        fig_fd = mean_framewise_displacement.plot_stats(stats)
        fig_fd.savefig(output_dir / f"mean_fd_{version}.png", transparent=True)

        # Connectome Similarity
        average_connectomes = connectivity_similarity.load_data(path_root, datasets, version)
        fig_similarity = connectivity_similarity.plot_stats(
            average_connectomes, strategy_order=strategy_order, show_colorbar=True, horizontal=False
        )
        fig_similarity.savefig(output_dir / f"connectomes_{version}.png", transparent=True)

        # Loss of Degrees of Freedom
        data = degrees_of_freedom_loss.load_data(path_root, datasets, criteria_name, version)
        fig_dof = degrees_of_freedom_loss.plot_stats(data)
        fig_dof.savefig(output_dir / f"loss_degrees_of_freedom_{version}.png", transparent=True)

        fig_dof_subgroup = degrees_of_freedom_loss.plot_stats(data, plot_subgroup=True)
        fig_dof_subgroup.savefig(output_dir / f"loss_degrees_of_freedom_subgroups_{version}.png", transparent=True)

        # Motion Metrics
        metrics = {
            "p_values": "sig_qcfc",
            "fdr_p_values": "sig_qcfc_fdr",
            "median": "median_qcfc",
            "distance": "distance_qcfc",
        }
        for m in metrics:
            data_m, measure = motion_metrics.load_data(path_root, datasets, criteria_name, version, m)
            for dataset in data_m:
                data_m[dataset] = data_m[dataset][~data_m[dataset]["strategy"].isin(excluded_strategies)]
            fig = motion_metrics.plot_stats(data_m, measure)
            fig.savefig(output_dir / f"{metrics[m]}_{version}.png", transparent=True)

        # Modularity Special Combined Plot
        palette = sns.color_palette("colorblind", n_colors=7)
        paired_palette = [palette[0]]
        for p in palette[1:4]:
            paired_palette.extend((p, p))
        paired_palette.extend((palette[-3], palette[-2], palette[-1], palette[-1]))

        fig_modularity = plt.figure(constrained_layout=True, figsize=(6.4, 9.6))
        subfigs_modularity = fig_modularity.subfigures(2, 1, wspace=0.07)

        for j, m in enumerate(["modularity", "modularity_motion"]):
            data_m, measure = motion_metrics.load_data(path_root, datasets, criteria_name, version, m)
            for dataset in data_m:
                data_m[dataset] = data_m[dataset][~data_m[dataset]["strategy"].isin(excluded_strategies)]
            axs_modularity = subfigs_modularity[j].subplots(1, 2, sharey=True)
            for i, dataset in enumerate(data_m):
                df = data_m[dataset].query("groups=='full_sample'")
                baseline_values = df.query("strategy=='baseline'")
                baseline_mean = baseline_values[measure["label"]].mean()
                sns.barplot(
                    y=measure["label"],
                    x="strategy",
                    data=df,
                    ax=axs_modularity[i],
                    order=strategy_order,
                    ci=95,
                    palette=paired_palette,
                )
                axs_modularity[i].axhline(baseline_mean, ls="-.", c=paired_palette[0])
                axs_modularity[i].set_title(dataset)
                axs_modularity[i].set_ylim(measure["ylim"])
                axs_modularity[i].set_xticklabels(strategy_order, rotation=45, ha="right", rotation_mode="anchor")

            if j == 0:
                labels = ["No GSR", "With GSR"]
                hatches = ["", "///"]
                handles = [mpatches.Patch(edgecolor="black", facecolor="white", hatch=h, label=l) for h, l in zip(hatches, labels)]
                axs_modularity[1].legend(handles=handles)

        fig_modularity.savefig(output_dir / f"modularity_{version}.png", transparent=True)

        data_rank = strategy_ranking.load_data(path_root, datasets)

        # Flatten MultiIndex columns
        if isinstance(data_rank.columns, pd.MultiIndex):
            data_rank.columns = ['_'.join(filter(None, col)) for col in data_rank.columns.values]

        # Find only ranking columns
        ranking_cols = [col for col in data_rank.columns if col.startswith('ranking_')]

        # Remove excluded strategies
        for strategy in excluded_strategies:
            ranking_cols = [col for col in ranking_cols if not col.endswith(strategy)]

        # Subset to just those columns
        data_rank = data_rank[ranking_cols]

        # Rebuild MultiIndex with named levels
        data_rank.columns = pd.MultiIndex.from_tuples(
            [('ranking', col.replace('ranking_', '')) for col in data_rank.columns],
            names=["measure", "strategy"]
        )

        # Plot
        fig_ranking = strategy_ranking.plot_ranking(data_rank)
        fig_ranking.savefig(output_dir / f"strategy_ranking_{version}.png", transparent=True)

        # Loss of DOF vs Quality
        plot_loss_vs_quality(
            data=strategy_ranking.load_data(path_root, datasets),
            output_dir=output_dir,
            version_filter=version,
        )