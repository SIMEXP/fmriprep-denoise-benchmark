import pandas as pd
import numpy as np
from scipy import stats, linalg

from statsmodels.stats import multitest


def calculate_median_absolute(x):
    """Calculate Absolute median value"""
    return x.abs().median()


def significant_level(x, alpha=0.05, correction=None):
    """
    Apply FDR correction to a pandas.Series p-value object.

    Parameters
    ----------

    x : pandas.Series
        Uncorrected p-values.

    alpha : float
        Alpha threshold.

    method : None or str
        Default as None for no multiple comparison
        Mutiple comparison methods.
        See statsmodels.stats.multitest.multipletests

    Returns
    -------
    ndarray, boolean
        Mask for data passing multiple comparison test.
    """
    if isinstance(correction, str):
        res, _, _, _ = multitest.multipletests(x, alpha=alpha, method=correction)
    else:
        res = x < 0.05
    return res


def partial_correlation(x, y, cov=None):
    """A minimal implementation of partial correlation.

    Parameters
    ----------
    x, y : np.ndarray
        Variable of interest.

    cov : None, np.ndarray
        Variable to be removed from variable of interest.
        If None, do a normal pearson's correlation.

    Returns
    -------
    dict
        Correlation and p-value.
    """
    if isinstance(cov, np.ndarray):
        beta_cov_x = linalg.lstsq(cov, x)[0]
        beta_cov_y = linalg.lstsq(cov, y)[0]
        resid_x = x - cov.dot(beta_cov_x)
        resid_y = y - cov.dot(beta_cov_y)
        r, p_val = stats.pearsonr(resid_x, resid_y)
    else:
        r, p_val = stats.pearsonr(x, y)
    return {"correlation": r, "pvalue": p_val}


def qcfc(movement, connectomes, covarates=None):
    """
    metric calculation: quality control / functional connectivity

    For each edge, we then computed the correlation between the weight of
    that edge and the mean relative RMS motion.
    QC-FC relationships were calculated as partial correlations that
    accounted for participant age and sex

    Parameters
    ----------

    movement: pandas.DataFrame
        Containing header: ["mean_framewise_displacement"]

    connectomes: pandas.DataFrame
        Flattened connectome of a whole dataset.
        Index: subjets
        Columns: ROI-ROI pairs

    covariates: pandas.DataFrame or None
        Age", Gender

    Returns
    -------

    List of dict
        QC/FC per connectome edge.
    """
    # concatenate information to match by subject id
    edge_ids = connectomes.columns.tolist()
    connectomes = pd.concat((connectomes, movement), axis=1)

    if covarates is not None:
        covarates = covarates.apply(stats.zscore)
        cov_names = covarates.columns
        connectomes = pd.concat((connectomes, covarates), axis=1)

    # drop subject with no edge value
    connectomes = connectomes.dropna(axis=0)

    qcfc_edge = []
    for edge_id in edge_ids:
        # QC-FC
        metric = partial_correlation(
            connectomes[edge_id].values,
            connectomes["mean_framewise_displacement"].values,
            connectomes[cov_names].values,
        )
        qcfc_edge.append(metric)

    return qcfc_edge
