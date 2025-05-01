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
logging.basicConfig(
    level=logging.DEBUG,  # Adjust the level if needed
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# group_order = {
#     "ds000228": ["adult", "child"],
#     "ds000030": ["control", "ADHD", "bipolar", "schizophrenia"],
# }
# datasets = ["ds000228", "ds000030"]
# datasets_baseline = {"ds000228": "adult", "ds000030": "control"}
# criteria_name = "stringent"
# fmriprep_version = "fmriprep-20.2.1lts"

group_order = {
    "ds000228": ["adult", "child"]
}
datasets = ["ds000228"]
datasets_baseline = {"ds000228": "adult"}
criteria_name = "minimal"
fmriprep_version = "fmriprep-25.0.0"
excluded_strategies = []#["compcor", "aroma"]

def plot_loss_vs_quality(data, output_dir, version_filter="fmriprep-25.0.0"):

    sns.set(style="whitegrid", context="talk")

    filtered_df = data.xs(version_filter, level='version')
    results = []

    for dataset in filtered_df.index.levels[0]:
        subset = filtered_df.loc[dataset]
        loss_df = subset.loc['loss_df']
        # other_metrics = subset.drop('loss_df')
        # mean_other = other_metrics.mean()
        other_metrics = subset.drop('loss_df')

        # Safe filtering regardless of index type
        if isinstance(other_metrics.index, pd.MultiIndex) and "strategy" in other_metrics.index.names:
            strategy_idx = other_metrics.index.get_level_values("strategy")
            other_metrics = other_metrics.loc[~strategy_idx.isin(excluded_strategies)]
        else:
            other_metrics = other_metrics.loc[~other_metrics.index.isin(excluded_strategies)]

        mean_other = other_metrics.mean()
        results.append((dataset, loss_df, mean_other))

    for dataset, loss_df, mean_other in results:
        # Drop by matching the second element of the index (strategy name)
        loss_df = loss_df[[s for s in loss_df.index if (s[1] if isinstance(s, tuple) else s) not in excluded_strategies]]
        mean_other = mean_other[[s for s in mean_other.index if (s[1] if isinstance(s, tuple) else s) not in excluded_strategies]]
        logger.debug(f"Filtered loss_df strategies: {[s[1] if isinstance(s, tuple) else s for s in loss_df.index]}")

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(x=loss_df, y=mean_other, s=100, color="royalblue", ax=ax)

        texts = []
        for strategy in loss_df.index:
            label = strategy[1] if isinstance(strategy, tuple) else strategy
            texts.append(
                ax.text(
                    loss_df[strategy],
                    mean_other[strategy] + 0.1,
                    label,
                    fontsize=12,
                    ha="center"
                )
            )
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))

        ax.set_title(f"{dataset} ({version_filter})\nLoss of DF vs Mean Rank of Other Metrics", fontsize=16)
        ax.set_xlabel("Loss of Degrees of Freedom (Rank)", fontsize=14)
        ax.set_ylabel("Mean Rank of Denoising Quality Metrics", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()

        # ✅ Save the plot
        fig.savefig(output_dir / f"{dataset}_loss_vs_quality.png", transparent=True)
        plt.close(fig)


if __name__ == "__main__":
    # path_root = Path(__file__).parents[1] / "data" / \
    #     "fmriprep-denoise-benchmark" / "denoise-metrics"

    path_root = Path("/home/seann/scratch/denoise/fmriprep-denoise-benchmark/outputs/denoise-metrics-atlas.5-4.27.25")
    strategy_order = list(utils.GRID_LOCATION.values())
    strategy_order = [s for s in strategy_order if s not in excluded_strategies]

    # mean fd
    stats = mean_framewise_displacement.load_data(
        path_root, criteria_name, fmriprep_version
    )
    fig_fd = mean_framewise_displacement.plot_stats(stats)
    fig_fd.savefig(
        Path(__file__).parents[1] / "outputs" / "mean_fd.png", transparent=True
    )

    # connectomes
    average_connectomes = connectivity_similarity.load_data(
        path_root, datasets, fmriprep_version
    )
    # strategy_order = ["baseline", "simple", "simple+gsr", "scrubbing.5", "scrubbing.5+gsr", "compcor", "aroma"]
    strategy_order = ["scrubbing.5+gsr","simple+gsr","compcor","scrubbing.5","simple","aroma","baseline"]

    fig_similarity = connectivity_similarity.plot_stats(
        average_connectomes,
        strategy_order=strategy_order,
        show_colorbar=True,
        horizontal=False,
    )

    fig_similarity.savefig(
        Path(__file__).parents[1] / "outputs" / "connectomes.png", transparent=True
    )

    # loss of degrees of freedom
    data = degrees_of_freedom_loss.load_data(
        path_root, datasets, criteria_name, fmriprep_version
    )
    fig_degrees_of_freedom = degrees_of_freedom_loss.plot_stats(data)
    fig_degrees_of_freedom.savefig(
        Path(__file__).parents[1] / "outputs" / "loss_degrees_of_freedom.png",
        transparent=True,
    )
    fig_degrees_of_freedom = degrees_of_freedom_loss.plot_stats(data, plot_subgroup=True)
    fig_degrees_of_freedom.savefig(
        Path(__file__).parents[1] / "outputs" / "loss_degrees_of_freedom_subgroups.png",
        transparent=True,
    )
    data = degrees_of_freedom_loss.load_data(
        path_root, datasets, "minimal", fmriprep_version
    )
    fig_degrees_of_freedom = degrees_of_freedom_loss.plot_stats(data, plot_subgroup=True)
    fig_degrees_of_freedom.savefig(
        Path(__file__).parents[1] / "outputs" / "loss_degrees_of_freedom_subgroups_qc-minimal.png",
        transparent=True,
    )
    data = degrees_of_freedom_loss.load_data(
        path_root, datasets, None, fmriprep_version
    )
    fig_degrees_of_freedom = degrees_of_freedom_loss.plot_stats(data, plot_subgroup=True)
    fig_degrees_of_freedom.savefig(
        Path(__file__).parents[1] / "outputs" / "loss_degrees_of_freedom_subgroups_qc-none.png",
        transparent=True,
    )
    # Plotting metrics
    metrics = {
        "p_values": "sig_qcfc",
        "fdr_p_values": "sig_qcfc_fdr",
        "median": "median_qcfc",
        "distance": "distance_qcfc",
    }

    for m in metrics:
        data, measure = motion_metrics.load_data(
            path_root, datasets, criteria_name, fmriprep_version, m
        )
        for dataset in data:
            data[dataset] = data[dataset][~data[dataset]["strategy"].isin(excluded_strategies)]
        fig = motion_metrics.plot_stats(data, measure)
        fig.savefig(
            Path(__file__).parents[1] / "outputs" / f"{metrics[m]}.png",
            transparent=True,
        )

    data, measure = motion_metrics.load_data(
        path_root, datasets, criteria_name, fmriprep_version, "p_values"
    )
    for dataset in data:
        data[dataset] = data[dataset][~data[dataset]["strategy"].isin(excluded_strategies)]
    fig = motion_metrics.plot_stats(data, measure)
    fig.savefig(
        Path(__file__).parents[1] / "outputs" / f"sig_qcfc_alt_fmriprep.png",
        transparent=True,
    )

    # customised code for mean netnor modularity as we are combining two figures
    palette = sns.color_palette("colorblind", n_colors=7)
    paired_palette = [palette[0]]
    for p in palette[1:4]:
        paired_palette.extend((p, p))
    paired_palette.extend((palette[-3], palette[-2], palette[-1], palette[-1]))

    fig_modularity = plt.figure(constrained_layout=True, figsize=(6.4, 9.6))
    subfigs_modularity = fig_modularity.subfigures(2, 1, wspace=0.07)
    for j, m in enumerate(["modularity", "modularity_motion"]):
        data, measure = motion_metrics.load_data(
            path_root, datasets, criteria_name, fmriprep_version, m
        )
        for dataset in data:
            data[dataset] = data[dataset][~data[dataset]["strategy"].isin(excluded_strategies)]
        subfigs_modularity[j].suptitle(
            measure["title"], weight="heavy", fontsize="x-large"
        )
        axs_modularity = subfigs_modularity[j].subplots(1, 2, sharey=True)
        for i, dataset in enumerate(data):
            df = data[dataset].query("groups=='full_sample'")
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
            axs_modularity[i].set_xticklabels(
                strategy_order, rotation=45, ha="right", rotation_mode="anchor"
            )
            for i, bar in enumerate(axs_modularity[i].patches):
                if i > 0 and i % 2 == 0 and i != 8:  # only give gsr hatch
                    bar.set_hatch("///")
        if j == 0:
            labels = ["No GSR", "With GSR"]
            hatches = ["", "///"]
            handles = [
                mpatches.Patch(edgecolor="black", facecolor="white", hatch=h, label=l)
                for h, l in zip(hatches, labels)
            ]
            axs_modularity[1].legend(handles=handles)

    fig_modularity.savefig(
        Path(__file__).parents[1] / "outputs" / "modularity.png", transparent=True
    )

    fig_joint = motion_metrics.plot_joint_scatter(
        path_root,
        dataset="ds000228",
        base_strategy="simple",
        fmriprep_version=fmriprep_version,
    )
    fig_joint.savefig(
        Path(__file__).parents[1]
        / "outputs"
        / "ds000228_simple_meanfd_modularity.png",
        transparent=True,
    )

    data = strategy_ranking.load_data(path_root, datasets)
    for dataset in data:
        df = data[dataset]

        # If strategy is still in the index, reset it
        if isinstance(df, pd.DataFrame):
            if isinstance(df.index, pd.MultiIndex) and "strategy" in df.index.names:
                df = df.reset_index()
            elif "strategy" not in df.columns:
                raise ValueError(f"'strategy' column not found in {dataset} even after reset.")

            df = df[~df["strategy"].isin(excluded_strategies)]
            data[dataset] = df
        else:
            logger.warning(f"Unexpected type for data[{dataset}]: {type(df)}")
    fig_ranking = strategy_ranking.plot_ranking(data)
    fig_ranking.savefig(
        Path(__file__).parents[1] / "outputs" / "strategy_ranking.png", transparent=True
    )

        # Plot loss of DOF vs mean rank of other metrics (scatter + adjustText)
    plot_loss_vs_quality(
        data=strategy_ranking.load_data(path_root, datasets),
        output_dir=Path(__file__).parents[1] / "outputs",
        version_filter=fmriprep_version
    )
