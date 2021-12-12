import pandas as pd
import numpy as np
from scipy import stats, linalg


def _partial_correlation(x, y, cov=None):
    """A minimal implementation of partial correlation.

    x, y :
        Variable of interest.
    cov :
        Variable to be removed from variable of interest.
    """
    if isinstance(cov, np.ndarray):
        beta_cov_x = linalg.lstsq(cov, x)[0]
        beta_cov_y = linalg.lstsq(cov, y)[0]
        resid_x = x - cov.dot(beta_cov_x)
        resid_y = y - cov.dot(beta_cov_y)
        r, p_val = stats.pearsonr(resid_x, resid_y)
    else:
        r, p_val = stats.pearsonr(x, y)
    return {'correlation': r, 'pvalue': p_val}


def qcfc(movement, connectomes, covarates=('Age', 'Gender')):
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

    connectomes: pandas.DataFrame
        Flattened connectome of a whole dataset.
        Index: subjets
        Columns: ROI-ROI pairs
    """
    # concatenate information to match by subject id
    edge_ids = connectomes.columns.tolist()
    connectomes = pd.concat((connectomes, movement), axis=1)
    # drop subject with no edge value
    connectomes = connectomes.dropna(axis=0)
    qcfc_edge = []
    if covarates is not None:
        covarates = connectomes[covarates].values
    for edge_id in edge_ids:
        # QC-FC
        metric = _partial_correlation(
            connectomes[edge_id].values,
            connectomes['mean_framewise_displacement'].values,
            covarates)
        qcfc_edge.append(metric)
    return qcfc_edge
