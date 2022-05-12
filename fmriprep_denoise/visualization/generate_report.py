import argparse

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fmriprep_denoise.visualization import figures


grid_location = {
    (0, 0): 'baseline',
    (0, 2): 'simple',
    (0, 3): 'simple+gsr',
    (1, 0): 'scrubbing.5',
    (1, 1): 'scrubbing.5+gsr',
    (1, 2): 'scrubbing.2',
    (1, 3): 'scrubbing.2+gsr',
    (2, 0): 'compcor',
    (2, 1): 'compcor6',
    (2, 2): 'aroma',
    (2, 3): 'aroma+gsr',
}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Generate denoise metric based on denoising strategy for ds000228.",
    )
    parser.add_argument(
        "dataset",
        action="store",
        type=str,
        help="dataset"
    )
    parser.add_argument(
        "--atlas",
        action="store",
        type=str,
        help="Atlas name (schaefer7networks, mist, difumo, gordon333)"
    )
    parser.add_argument(
        "--dimension",
        action="store",
        help="Number of ROI. See meta data of each atlas to get valid inputs.",
    )
    return parser.parse_args()


def main():
    # Load metric data
    args = parse_args()
    print(vars(args))
    dataset = args.dataset
    atlas_name = args.atlas
    dimension = args.dimension

    path_root = Path(__file__).parents[1] / "inputs"
    output = Path(__file__).parents[1] / "results"

    # process data
    (qcfc_per_edge, sig_per_edge, modularity, movement, pairwise_distance) = figures.load_metrics(dataset, atlas_name, dimension, path_root)

    long_qcfc = qcfc_per_edge.melt()
    long_qcfc.columns = ["Strategy", "qcfc"]

    corr_distance_long = qcfc_per_edge.melt()
    corr_distance_long.columns = ["Strategy", "qcfc"]
    corr_distance_long['distance'] = np.tile(pairwise_distance.iloc[:, -1].values, 11)

    modularity_order = modularity.mean().sort_values().index.tolist()


    qcfc_sig = figures._qcfc_fdr(sig_per_edge)
    qcfc_mad = figures._get_qcfc_median_absolute(qcfc_per_edge)
    qcfc_dist = figures._get_corr_distance(pairwise_distance, qcfc_per_edge)
    corr_mod = figures._corr_modularity_motion(modularity, movement)
    network_mod = {
        'data': modularity,
        'order': modularity_order,
        'title': "Identifiability of network structure\nafter denoising",
        'label': "Mean modularity quality (a.u.)",
    }

    # strart plotting
    bar_color = sns.color_palette()[0]

    fig = plt.figure(constrained_layout=True, figsize=(23, 9))
    subfigs = fig.subfigures(2, 2, wspace=0.01)

    axsTopLeft = subfigs[0, 0].subplots(1, 3, sharey=False)
    for nn, (ax, figure_data) in enumerate(zip(axsTopLeft, [qcfc_sig, qcfc_mad, qcfc_dist])):
        sns.barplot(data=figure_data['data'], orient='h',
                    ci=None, order=figure_data['order'],
                    color=bar_color, ax=ax)
        ax.set_title(figure_data['title'])
        ax.set(xlabel=figure_data['label'])
        if nn == 0:
            ax.set(ylabel="Confound removal strategy")
    subfigs[0, 0].suptitle('Residual effect of motion on connectomes after de-noising')
    subfigs[0, 0].set_facecolor('0.75')

    axsBottomLeft = subfigs[1, 0].subplots(3, 4, sharex=True, sharey=True)
    for i, row_axes in enumerate(axsBottomLeft):
        for j, ax in enumerate(row_axes):
            if cur_strategy := grid_location.get((i, j), False):
                mask = corr_distance_long["Strategy"] == cur_strategy
                g = sns.histplot(data=corr_distance_long.loc[mask, :],
                                 x='distance', y='qcfc',
                                 ax=ax)
                ax.set_title(cur_strategy, fontsize='small')
                g.axhline(0, linewidth=1, linestyle='--', alpha=0.5, color='k')
                sns.regplot(data=corr_distance_long.loc[mask, :],
                            x='distance', y='qcfc',
                            ci=None,
                            scatter=False,
                            line_kws={'color': 'r', 'linewidth': 0.5},
                            ax=ax)
                xlabel = "Distance (mm)" if i == 2 else None
                ylabel = "QC-FC" if j == 0 else None
                g.set(xlabel=xlabel, ylabel=ylabel)
            else:
                subfigs[1, 0].delaxes(axsBottomLeft[i, j])
    subfigs[1, 0].suptitle('Correlation between nodewise Euclidian distance and QC-FC')
    subfigs[1, 0].set_facecolor('0.75')

    axsTopRight = subfigs[0, 1].subplots(1, 2, sharey=False)
    sns.barplot(data=network_mod['data'],
                orient='h',
                ci=None,
                order=network_mod['order'],
                color=bar_color, ax=axsTopRight[0])
    axsTopRight[0].set_title(network_mod['title'])
    axsTopRight[0].set(xlabel=network_mod['label'])
    axsTopRight[0].set(ylabel="Confound removal strategy")

    sns.barplot(data=corr_mod['data'], x='correlation', y='strategy',
                ci=None,
                order=None,
                color=bar_color, ax=axsTopRight[1])
    axsTopRight[1].set_title(corr_mod['title'])
    axsTopRight[1].set(xlabel=corr_mod['label'])

    subfigs[0, 1].suptitle('Correlation between\nnetwork modularity and mean framewise displacement')
    subfigs[0, 1].set_facecolor('0.75')

    axsBottomRight = subfigs[1, 1].subplots(3, 4, sharex=True, sharey=True)
    for i, row_axes in enumerate(axsBottomRight):
        for j, ax in enumerate(row_axes):
            if cur_strategy := grid_location.get((i, j), False):
                mask = long_qcfc["Strategy"] == cur_strategy
                g = sns.histplot(data=long_qcfc.loc[mask, :],
                                x='qcfc',
                                ax=ax)
                g.set_title(cur_strategy, fontsize='small')
                mad = qcfc_mad['data'][cur_strategy].values
                g.axvline(mad, linewidth=1, linestyle='--', color='r')
                xlabel = "Pearson\'s correlation" if i == 2 else None
                g.set(xlabel=xlabel)
            else:
                subfigs[1, 1].delaxes(axsBottomRight[i, j])
    subfigs[1, 1].suptitle('Distribution of correlation between framewise distplacement and edge strength')
    subfigs[1, 1].set_facecolor('0.75')

    fig.suptitle(f'atlas-{atlas_name}_nroi-{dimension}', fontsize='x-large')
    fig.savefig(output / f'atlas-{atlas_name}_nroi-{dimension}.png', dpi=300)


if __name__ == "__main__":
    main()
