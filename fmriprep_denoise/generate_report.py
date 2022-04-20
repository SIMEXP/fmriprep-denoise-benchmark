import argparse

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, spearmanr
from fmriprep_denoise.metrics import partial_correlation, fdr, calculate_median_absolute, get_atlas_pairwise_distance


new_loc = {
    'baseline': {'row': 0, 'col': 0},
    'simple': {'row': 0, 'col': 2},
    'simple+gsr': {'row': 0, 'col': 3},
    'scrubbing.5': {'row': 1, 'col': 0},
    'scrubbing.5+gsr': {'row': 1, 'col': 1},
    'scrubbing.2': {'row': 1, 'col': 2},
    'scrubbing.2+gsr': {'row': 1, 'col': 3},
    'compcor': {'row': 2, 'col': 0},
    'compcor6': {'row': 2, 'col': 1},
    'aroma': {'row': 2, 'col': 2},
    'aroma+gsr': {'row': 2, 'col': 3},
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
    file_qcfc = f"metrics/dataset-{dataset}_atlas-{atlas_name}_nroi-{dimension}_qcfc.tsv"
    file_network = f"metrics/dataset-{dataset}_atlas-{atlas_name}_nroi-{dimension}_modularity.tsv"
    file_dataset = f"dataset-{dataset}/dataset-{dataset}_desc-movement_phenotype.tsv"

    # calculate metrics
    pairwise_distance = get_atlas_pairwise_distance(atlas_name, dimension)
    movement = pd.read_csv(path_root / file_dataset, sep='\t', index_col=0, header=0, encoding='utf8')
    qcfc = pd.read_csv(path_root / file_qcfc, sep='\t', index_col=0)
    modularity = pd.read_csv(path_root / file_network, sep='\t', index_col=0)

    # separate correlation from siginficant value
    sig_per_edge = qcfc.filter(regex="pvalue")
    sig_per_edge.columns = [col.split('_')[0] for col in sig_per_edge.columns]
    metric_per_edge = qcfc.filter(regex="correlation")
    metric_per_edge.columns = [col.split('_')[0] for col in metric_per_edge.columns]

    bar_color = sns.color_palette()[0]

    # multiple comparision on qcfc
    long_qcfc_sig= sig_per_edge.melt()
    long_qcfc_sig['fdr'] = long_qcfc_sig.groupby('variable')['value'].transform(fdr)
    long_qcfc_sig = long_qcfc_sig.groupby('variable').apply(lambda x: 100*x.fdr.sum()/x.fdr.shape[0])
    long_qcfc_sig = pd.DataFrame(long_qcfc_sig, columns=["p_corrected"])

    order = long_qcfc_sig.sort_values('p_corrected').index.tolist()
    ax = sns.barplot(data=long_qcfc_sig.T, ci=None, order=order, color=bar_color)
    ax.set_title("Percentage of edge significantly correlated with mean FD")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set(ylabel="Percentage %",
        xlabel="confound removal strategy")
    plt.tight_layout()
    plt.savefig(output / f"atlas-{atlas_name}_nroi-{dimension}_sigqcfc.png", dpi=300)

    median_absolute = metric_per_edge.apply(calculate_median_absolute)
    order = median_absolute.sort_values().index.tolist()

    ax = sns.barplot(data=(pd.DataFrame(median_absolute).T), ci=None, order=order, color=bar_color)
    ax.set_title("Median absolute deviation QC-FC")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set(ylabel="Median absolute deviation",
        xlabel="confound removal strategy")
    plt.tight_layout()
    plt.savefig(output / f"atlas-{atlas_name}_nroi-{dimension}_mad_qcfc.png", dpi=300)

    def draw_absolute_median(data, **kws):
        ax = plt.gca()
        mad = calculate_median_absolute(data['qcfc'])
        ax.vlines(mad, ymin=0, ymax=0.5, color='r', linestyle=':')

    long_qcfc = metric_per_edge.melt()
    long_qcfc.columns = ["Strategy", "qcfc"]

    long_qcfc["row"] = long_qcfc.Strategy
    long_qcfc["col"] = long_qcfc.Strategy
    for name in new_loc:
        long_qcfc.loc[long_qcfc.Strategy == name, "row"] = new_loc[name]["row"]
        long_qcfc.loc[long_qcfc.Strategy == name, "col"] = new_loc[name]["col"]

    g = sns.displot(
        long_qcfc, x="qcfc", col="col", row="row", kind='kde', fill=True, height=1.5, aspect=2
    )
    g.fig.delaxes(g.axes[0, 1])

    g.set(ylabel="Density")
    g.map_dataframe(draw_absolute_median)
    for name, axis in new_loc.items():
        g.facet_axis(axis['row'], axis['col']).set(title=name)
        if axis['row'] == 2:
            g.facet_axis(axis['row'], axis['col']).set(xlabel="Pearson\'s correlation: \nmean FD and\nconnectome edges")

    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Distribution of correlation between framewise distplacement and edge strength')
    plt.tight_layout()
    plt.savefig(output / f"atlas-{atlas_name}_nroi-{dimension}_distqcfc.png", dpi=300)
    plt.close()

    corr_distance, p_val = spearmanr(pairwise_distance.iloc[:, -1], metric_per_edge)

    corr_distance = pd.DataFrame(corr_distance[1:, 0], index=metric_per_edge.columns)
    long_qcfc['distance'] = np.tile(pairwise_distance.iloc[:, -1].values, 11)

    order = corr_distance.sort_values(0).index.tolist()

    ax = sns.barplot(data=corr_distance.T, ci=None, order=order, color=bar_color)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title("Distance-dependent effects of motion")
    ax.set(ylim=(-0.5, 0.05))
    ax.set(ylabel="Nodewise correlation between\nEuclidian distance and QC-FC metric",
            xlabel="confound removal strategy")
    plt.tight_layout()
    plt.savefig(output / f"atlas-{atlas_name}_nroi-{dimension}_corr_dist_qcfc_mean.png", dpi=300)

    g = sns.FacetGrid(long_qcfc, col="col", row="row", height=1.7, aspect=1.5)
    g.map(sns.regplot, 'distance', 'qcfc', fit_reg=True, ci=None,
        line_kws={'color': 'red'}, scatter_kws={'s': 0.5, 'alpha': 0.15,})
    g.refline(y=0)
    g.fig.delaxes(g.axes[0, 1])
    for name, axis in new_loc.items():
        g.facet_axis(axis['row'], axis['col']).set(title=name)
        if axis['row'] == 2:
            g.facet_axis(axis['row'], axis['col']).set(xlabel="Distance (mm)")
        if axis['col'] == 0:
            g.facet_axis(axis['row'], axis['col']).set(ylabel="QC-FC")

    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Correlation between nodewise Euclidian distance and QC-FC')
    plt.tight_layout()
    plt.savefig(output / f"atlas-{atlas_name}_nroi-{dimension}_corr_dist_qcfc_dist.png", dpi=300)

    corr_modularity = []
    z_movement = movement.apply(zscore)
    for column, values in modularity.iteritems():
        cur_data = pd.concat((modularity[column],
                            movement[['mean_framewise_displacement']],
                            z_movement[['age', 'gender']]), axis=1).dropna()
        current_strategy = partial_correlation(cur_data[column].values,
                                            cur_data['mean_framewise_displacement'].values,
                                            cur_data[['age', 'gender']].values)
        current_strategy['strategy'] = column
        corr_modularity.append(current_strategy)

    plt.figure(figsize=(7, 5))
    plt.subplot(1, 2, 1)
    order = modularity.mean().sort_values().index.tolist()
    ax = sns.barplot(data=modularity, order=order, color=bar_color)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title("Identifiability of network structure\nafter denoising")
    ax.set(ylabel="Mean modularity quality (a.u.)",
        xlabel="confound removal strategy")
    plt.subplot(1, 2, 2)

    corr_modularity = pd.DataFrame(corr_modularity).sort_values('correlation')
    ax = sns.barplot(data=corr_modularity, y='correlation', x='strategy', ci=None, color=bar_color)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title("Correlation between\nnetwork modularity and \nmean framewise displacement")
    ax.set(ylabel="Pearson's correlation",
        xlabel="confound removal strategy")
    plt.tight_layout()
    plt.savefig(output / f"atlas-{atlas_name}_nroi-{dimension}_modularity.png", dpi=300)

if __name__ == "__main__":
    main()