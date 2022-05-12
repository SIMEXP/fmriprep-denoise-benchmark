import pandas as pd

from scipy.stats import zscore, spearmanr
from fmriprep_denoise.features import (partial_correlation, fdr,
                                       calculate_median_absolute,
                                       get_atlas_pairwise_distance)


def load_metrics(dataset, atlas_name, dimension, path_root):
    file_qcfc = path_root / f"metrics/dataset-{dataset}_atlas-{atlas_name}_nroi-{dimension}_qcfc.tsv"
    file_network = path_root / f"metrics/dataset-{dataset}_atlas-{atlas_name}_nroi-{dimension}_modularity.tsv"
    file_dataset = path_root / f"dataset-{dataset}/dataset-{dataset}_desc-movement_phenotype.tsv"
    pairwise_distance = get_atlas_pairwise_distance(atlas_name, dimension)
    qcfc_per_edge = _get_qcfc_metric(file_qcfc, metric="correlation")
    sig_per_edge = _get_qcfc_metric(file_qcfc, metric="pvalue")
    modularity = pd.read_csv(file_network, sep='\t', index_col=0)
    movement = pd.read_csv(file_dataset, sep='\t', index_col=0, header=0, encoding='utf8')
    return qcfc_per_edge, sig_per_edge, modularity, movement, pairwise_distance


def _get_qcfc_metric(file_path, metric):
    """ Get correlation or pvalue of QC-FC."""
    qcfc_stats = pd.read_csv(file_path, sep='\t', index_col=0)
    qcfc_per_edge = qcfc_stats.filter(regex=metric)
    qcfc_per_edge.columns = [col.split('_')[0] for col in qcfc_per_edge.columns]
    return qcfc_per_edge


def _get_corr_distance(pairwise_distance, qcfc_per_edge):
    corr_distance, _ = spearmanr(pairwise_distance.iloc[:, -1], qcfc_per_edge)
    corr_distance = pd.DataFrame(corr_distance[1:, 0], index=qcfc_per_edge.columns)
    corr_distance_order = corr_distance.sort_values(0).index.tolist() # needed
    corr_distance = corr_distance.T  # needed
    return {
        'data': corr_distance,
        'order': corr_distance_order,
        'title': "Correlation between\nnodewise distance and QC-FC",
        'label': "Pearson's correlation",
    }


def _corr_modularity_motion(modularity, movement):
    """Correlation between network modularity and  mean framewise displacement."""
    corr_modularity = []
    z_movement = movement.apply(zscore)
    for column, _ in modularity.iteritems():
        cur_data = pd.concat((modularity[column],
                              movement[['mean_framewise_displacement']],
                              z_movement[['age', 'gender']]), axis=1).dropna()
        current_strategy = partial_correlation(cur_data[column].values,
                                            cur_data['mean_framewise_displacement'].values,
                                            cur_data[['age', 'gender']].values)
        current_strategy['strategy'] = column
        corr_modularity.append(current_strategy)
    return {
        'data': pd.DataFrame(corr_modularity).sort_values('correlation'),
        'order': None,
        'title': "Correlation between\nnetwork modularity and motion",
        'label': "Pearson's correlation",
        }


def _qcfc_fdr(sig_per_edge):
    """Do FDR correction on qc-fc p-values."""
    long_qcfc_sig= sig_per_edge.melt()
    long_qcfc_sig['fdr'] = long_qcfc_sig.groupby('variable')['value'].transform(fdr)
    long_qcfc_sig = long_qcfc_sig.groupby('variable').apply(lambda x: 100*x.fdr.sum()/x.fdr.shape[0])
    long_qcfc_sig = pd.DataFrame(long_qcfc_sig, columns=["p_corrected"])
    long_qcfc_sig_order = long_qcfc_sig.sort_values('p_corrected').index.tolist()
    return {
        'data': long_qcfc_sig.T,
        'order': long_qcfc_sig_order,
        'title': "Percentage of significant QC-FC",
        'label': "Percentage %",
    }


def _get_qcfc_median_absolute(qcfc_per_edge):
    """Calculate absolute median and prepare for plotting."""
    qcfc_median_absolute = qcfc_per_edge.apply(calculate_median_absolute)
    qcfc_median_absolute_order = qcfc_median_absolute.sort_values().index.tolist()
    return {
        'data': pd.DataFrame(qcfc_median_absolute).T,
        'order': qcfc_median_absolute_order,
        'title': "Median absolute deviation\nof QC-FC",
        'label': "Median absolute deviation",
    }
