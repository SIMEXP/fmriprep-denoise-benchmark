from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from adjustText import adjust_text

from fmriprep_denoise.visualization import utils
from fmriprep_denoise.visualization import (
    connectivity_similarity,
    degrees_of_freedom_loss,
    mean_framewise_displacement,
    motion_metrics,
    strategy_ranking,
)

# Configuration parameters
group_order = {
    "ds000228": ["adult", "child"]
}
datasets = ["ds000228"]
datasets_baseline = {"ds000228": "adult"}
criteria_name = "minimal"
fmriprep_version = "fmriprep-25.0.0"

if __name__ == "__main__":
    # Define paths for data and outputs
    path_root = Path("/home/seann/scratch/denoise/fmriprep-denoise-benchmark/outputs/denoise-metrics-2")
    output_dir = Path(__file__).parents[1] / "outputs"  # Use same out path for all plots
    strategy_order = list(utils.GRID_LOCATION.values())

    # --------------------------
    # Generate standard plots
    # --------------------------
    # Mean Framewise Displacement
    stats = mean_framewise_displacement.load_data(path_root, criteria_name, fmriprep_version)
    fig_fd = mean_framewise_displacement.plot_stats(stats)
    fig_fd.savefig(output_dir / "mean_fd.png", transparent=True)

    # Connectomes similarity plot
    average_connectomes = connectivity_similarity.load_data(path_root, datasets, fmriprep_version)
    fig_similarity = connectivity_similarity.plot_stats(average_connectomes)
    fig_similarity.savefig(output_dir / "connectomes.png", transparent=True)

    # Loss of degrees of freedom plots
    data = degrees_of_freedom_loss.load_data(path_root, datasets, criteria_name, fmriprep_version)
    fig_degrees_of_freedom = degrees_of_freedom_loss.plot_stats(data)
    fig_degrees_of_freedom.savefig(output_dir / "loss_degrees_of_freedom.png", transparent=True)
    fig_degrees_of_freedom = degrees_of_freedom_loss.plot_stats(data, plot_subgroup=True)
    fig_degrees_of_freedom.savefig(output_dir / "loss_degrees_of_freedom_subgroups.png", transparent=True)
    data = degrees_of_freedom_loss.load_data(path_root, datasets, "minimal", fmriprep_version)
    fig_degrees_of_freedom = degrees_of_freedom_loss.plot_stats(data, plot_subgroup=True)
    fig_degrees_of_freedom.savefig(output_dir / "loss_degrees_of_freedom_subgroups_qc-minimal.png", transparent=True)
    data = degrees_of_freedom_loss.load_data(path_root, datasets, None, fmriprep_version)
    fig_degrees_of_freedom = degrees_of_freedom_loss.plot_stats(data, plot_subgroup=True)
    fig_degrees_of_freedom.savefig(output_dir / "loss_degrees_of_freedom_subgroups_qc-none.png", transparent=True)

    # Motion Metrics Plots
    metrics = {
        "p_values": "sig_qcfc",
        "fdr_p_values": "sig_qcfc_fdr",
        "median": "median_qcfc",
        "distance": "distance_qcfc",
    }
    for m in metrics:
        data, measure = motion_metrics.load_data(path_root, datasets, criteria_name, fmriprep_version, m)
        fig = motion_metrics.plot_stats(data, measure)
        fig.savefig(output_dir / f"{metrics[m]}.png", transparent=True)
    data, measure = motion_metrics.load_data(path_root, datasets, criteria_name, fmriprep_version, "p_values")
    fig = motion_metrics.plot_stats(data, measure)
    fig.savefig(output_dir / f"sig_qcfc_alt_fmriprep.png", transparent=True)

    # Customized modularity plot (combining two subfigures)
    palette = sns.color_palette("colorblind", n_colors=7)
    paired_palette = [palette[0]]
    for p in palette[1:4]:
        paired_palette.extend((p, p))
    paired_palette.extend((palette[-3], palette[-2], palette[-1], palette[-1]))

    fig_modularity = plt.figure(constrained_layout=True, figsize=(6.4, 9.6))
    subfigs_modularity = fig_modularity.subfigures(2, 1, wspace=0.07)
    for j, m in enumerate(["modularity", "modularity_motion"]):
        data, measure = motion_metrics.load_data(path_root, datasets, criteria_name, fmriprep_version, m)
        subfigs_modularity[j].suptitle(measure["title"], weight="heavy", fontsize="x-large")
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
            axs_modularity[i].set_xticklabels(strategy_order, rotation=45, ha="right", rotation_mode="anchor")
            for i, bar in enumerate(axs_modularity[i].patches):
                if i > 0 and i % 2 == 0 and i != 8:  # apply hatch for specific bars
                    bar.set_hatch("///")
        if j == 0:
            labels = ["No GSR", "With GSR"]
            hatches = ["", "///"]
            handles = [mpatches.Patch(edgecolor="black", facecolor="white", hatch=h, label=l)
                       for h, l in zip(hatches, labels)]
            axs_modularity[1].legend(handles=handles)
    fig_modularity.savefig(output_dir / "modularity.png", transparent=True)

    # Joint scatter plot of mean FD and modularity
    fig_joint = motion_metrics.plot_joint_scatter(
        path_root,
        dataset="ds000228",
        base_strategy="baseline",
        fmriprep_version=fmriprep_version,
    )
    fig_joint.savefig(output_dir / "ds000228_baseline_meanfd_modularity.png", transparent=True)

    data = strategy_ranking.load_data(path_root, datasets)
    fig_ranking = strategy_ranking.plot_ranking(data)
    fig_ranking.savefig(output_dir / "strategy_ranking.png", transparent=True)

    # --------------------------------------------------------------
    # Integrated Custom Scatter Plot (Loss DF vs Mean of Other Metrics)
    # --------------------------------------------------------------
    # Load the degrees_of_freedom_loss data (or reuse one of the earlier calls)
    custom_data = degrees_of_freedom_loss.load_data(path_root, datasets, criteria_name, fmriprep_version)
    # Ensure custom_data is a multi-indexed DataFrame with a "version" level
    filtered_df = custom_data.xs(fmriprep_version, level="version")
    results = []
    for dataset in filtered_df.index.levels[0]:
        subset = filtered_df.loc[dataset]
        loss_df = subset.loc["loss_df"]
        other_metrics = subset.drop("loss_df")
        mean_other = other_metrics.mean()
        results.append((dataset, loss_df, mean_other))

    # Generate the custom scatter plot(s) for each dataset
    for dataset, loss_df, mean_other in results:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=loss_df, y=mean_other, s=100, color="royalblue")
        texts = []
        for strategy in loss_df.index:
            label = strategy[1] if isinstance(strategy, tuple) else strategy
            texts.append(
                plt.text(
                    loss_df[strategy],
                    mean_other[strategy] + 0.1,  # slight offset for clarity
                    label,
                    fontsize=12,
                    ha="center",
                )
            )
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))
        plt.title(f"{dataset} ({fmriprep_version})\nLoss of DF vs Mean Rank of Other Metrics", fontsize=16)
        plt.xlabel("Loss of Degrees of Freedom (Rank)", fontsize=14)
        plt.ylabel("Mean Rank of Denoising Quality Metrics", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        # Save the custom scatter plot in the same output directory
        custom_plot_path = output_dir / f"custom_scatter_{dataset}.png"
        plt.savefig(custom_plot_path, transparent=True)
        plt.show()