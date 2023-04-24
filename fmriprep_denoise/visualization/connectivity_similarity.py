import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nilearn.plotting.matrix_plotting import _reorder_matrix

from fmriprep_denoise.visualization import utils


strategies= list(utils.GRID_LOCATION.values())


def load_data(path_root, datasets, fmriprep_version):
    """Average connetome similarity across different parcellations."""
    average_connectomes = {}
    for dataset in datasets:
        connectomes_path = path_root.glob(
            f"{dataset}/{fmriprep_version}/*connectome.tsv"
        )
        connectomes_correlations = []
        for p in connectomes_path:
            cc = pd.read_csv(p, sep="\t", index_col=0)[strategies]
            connectomes_correlations.append(cc.corr().values)
        average_connectome = pd.DataFrame(
            np.mean(connectomes_correlations, axis=0),
            columns=cc.columns,
            index=cc.columns,
        )
        average_connectomes[dataset] = average_connectome

    # Average the two averages and cluster the correlation matrix
    _, labels = _reorder_matrix(
        np.mean(list(average_connectomes.values()), axis=0),
        list(cc.columns),
        "complete",
    )

    # reorder by label
    return {d: average_connectomes[d].loc[labels, labels] for d in average_connectomes}


def plot_stats(
    average_connectomes,
    horizontal=False,
):
    """
    Plot heatmaps for connectome similarity amongst denoising methods.

    Parameters
    ----------
    connectomes :

    horizontal : bool
        Horizontal or verical layout.

    Return
    ------
    matplotlib.pyplot.figure

    """
    fig_similarity, axs_similarity = plt.subplots(
        1, 2, figsize=(10.3, 4.8), constrained_layout=True
    )
    if horizontal:
        fig_similarity, axs_similarity = plt.subplots(
            2, 1, figsize=(4.8, 10.3), constrained_layout=True
        )

    fig_similarity.suptitle(
        "Similarity of denoised connectomes by strategy",
        weight="heavy",
        fontsize="x-large",
    )

    for i, dataset in enumerate(average_connectomes):
        cbar = i == 1
        sns.heatmap(
            average_connectomes[dataset],
            square=True,
            ax=axs_similarity[i],
            vmin=0.6,
            vmax=1,
            linewidth=0.5,
            cbar=cbar,
        )
        axs_similarity[i].set_title(dataset)
        axs_similarity[i].set_xticklabels(
            average_connectomes[dataset].columns,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
    return fig_similarity
