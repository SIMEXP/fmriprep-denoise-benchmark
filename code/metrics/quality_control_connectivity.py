import pandas as pd
from scipy import stats, linalg


def _partial_correlation(x, y, cov):
    """A minimal implementation of partial correlation.

    x, y :
        Variable of interest.
    cov :
        Variable to be removed from variable of interest.
    """

    beta_cov_x = linalg.lstsq(cov, x)[0]
    beta_cov_y = linalg.lstsq(cov, y)[0]
    resid_x = x - cov.dot(beta_cov_x)
    resid_y = y - cov.dot(beta_cov_y)
    return stats.pearsonr(resid_x, resid_y)


def quality_control_connectivity(movement, dataset_connectomes):
    """
    metric calculation: quality control / functional connectivity

    For each edge, we then computed the correlation between the weight of
    that edge and the mean relative RMS motion.
    QC-FC relationships were calculated as partial correlations that
    accounted for participant age and sex

    Parameters
    ----------
    movement: pandas.DataFrame
        Containing header: ["Age", "Gender", "mean_framewise_displacement"]

    dataset_connectomes: pandas.DataFrame
        Flattened connectome of a whole dataset.
        Index: subjets
        Columns: ROI-ROI pairs
    """
    cur_qc_fc, cur_sig = [], []
    for edge_id, edge_val in dataset_connectomes.iteritems():
        # concatenate information to match by subject id
        current_edge = pd.concat((edge_val, movement), axis=1)
        # drop subject with no edge value
        current_edge = current_edge.dropna()
        # QC-FC
        r, p_val = _partial_correlation(
                current_edge[edge_id].values,
                current_edge["mean_framewise_displacement"].values,
                current_edge[["Age", "Gender"]].values)
        cur_qc_fc.append(r)
        cur_sig.append(p_val)
    return cur_qc_fc, cur_sig
