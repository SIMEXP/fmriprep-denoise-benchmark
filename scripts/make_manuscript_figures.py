from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns

from fmriprep_denoise.visualization import utils
from fmriprep_denoise.visualization import (
    connectivity_similarity,
    degrees_of_freedom_loss,
    mean_framewise_displacement,
    motion_metrics,
)

group_order = {
    "ds000228": ["adult", "child"],
    "ds000030": ["control", "ADHD", "bipolar", "schizophrenia"],
}
datasets = ["ds000228", "ds000030"]
datasets_baseline = {"ds000228": "adult", "ds000030": "control"}
criteria_name = "stringent"
fmriprep_version = "fmriprep-20.2.1lts"


if __name__ == "__main__":
    path_root = utils.get_data_root() / "denoise-metrics"
    strategy_order = list(utils.GRID_LOCATION.values())

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
    fig_similarity = connectivity_similarity.plot_stats(average_connectomes)
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
        fig = motion_metrics.plot_stats(data, measure)
        fig.savefig(
            Path(__file__).parents[1] / "outputs" / f"{metrics[m]}.png",
            transparent=True,
        )

    data, measure = motion_metrics.load_data(
        path_root, datasets, criteria_name, "fmriprep-20.2.5lts", "p_values"
    )
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
        subfigs_modularity[j].suptitle(
            measure["title"], weight="heavy", fontsize="x-large"
        )
        axs_modularity = subfigs_modularity[j].subplots(1, 2, sharey=True)
        for i, dataset in enumerate(data):
            sns.barplot(
                y=measure["label"],
                x="strategy",
                data=data[dataset],
                ax=axs_modularity[i],
                order=strategy_order,
                ci=95,
                palette=paired_palette,
            )
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
        path_root, dataset="ds000228", fmriprep_version=fmriprep_version
    )
    fig_joint.savefig(
        Path(__file__).parents[1] / "outputs" / "meanfd_modularity.png",
        transparent=True,
    )
