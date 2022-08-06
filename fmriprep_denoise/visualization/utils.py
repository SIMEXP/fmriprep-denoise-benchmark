from pathlib import Path

import pandas as pd
import seaborn as sns

from scipy.stats import zscore, spearmanr
from repo2data.repo2data import Repo2Data

from fmriprep_denoise.features import (
    partial_correlation,
    fdr,
    calculate_median_absolute,
    get_atlas_pairwise_distance,
)


GRID_LOCATION = {
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


palette = sns.color_palette('Paired', n_colors=12)
palette_dict = {
    name: c for c, name in zip(palette[1:], GRID_LOCATION.values())
}


# download data
def repo2data_path():
    data_req_path = (
        Path(__file__).parents[2] / 'binder' / 'data_requirement.json'
    )
    repo2data = Repo2Data(str(data_req_path))
    data_path = repo2data.install()
    return Path(data_path[0])


def get_data_root():
    default_path = (
        Path(__file__).parents[2] / 'inputs' / 'fmrieprep-denoise-metrics'
    )
    return default_path if default_path.exists() else repo2data_path()


def load_meanfd_groups(dataset, path_root):
    file = f'dataset-{dataset}_desc-movement_phenotype.tsv'
    path_fd = path_root / f'dataset-{dataset}' / file
    data = pd.read_csv(path_fd, header=[0], index_col=0, sep='\t')
    _, participants_groups, groups = _get_participants_groups(dataset)
    participants_groups.name = 'Groups'
    data = pd.concat(
        [data['mean_framewise_displacement'], participants_groups],
        axis=1,
        join='inner',
    )
    return data, groups


def _get_palette(order):
    return [palette_dict[item] for item in order]


def _get_participants_groups(dataset, path_root):

    # need more general solutions here, maybe as a user input?
    group_info_column = 'Child_Adult' if dataset == 'ds000228' else 'diagnosis'

    # read the degrees of freedom info as reference for subjects
    path_dof = (
        path_root
        / f'dataset-{dataset}'
        / f'dataset-{dataset}_desc-confounds_phenotype.tsv'
    )
    path_participants = (
        path_root
        / f'dataset-{dataset}'
        / f'dataset-{dataset}_desc-movement_phenotype.tsv'
    )

    confounds_phenotype = pd.read_csv(
        path_dof, header=[0, 1], index_col=0, sep='\t'
    )
    subjects = confounds_phenotype.index

    participant_groups = pd.read_csv(
        path_participants, index_col=0, sep='\t'
    ).loc[subjects, 'groups']
    groups = participant_groups.unique().tolist()
    return confounds_phenotype, participant_groups, groups


def _get_connectome_metric_paths(
    dataset, metric, atlas_name, dimension, path_root
):
    atlas_name = '*' if isinstance(atlas_name, type(None)) else atlas_name
    dimension = (
        '*'
        if isinstance(atlas_name, type(None))
        or isinstance(dimension, type(None))
        else dimension
    )
    files = list(
        path_root.glob(
            (
                f'fmrieprep-denoise-metrics/dataset-{dataset}/'
                f'dataset-{dataset}_atlas-{atlas_name}_nroi-{dimension}_'
                f'{metric}.tsv'
            )
        )
    )
    if not files:
        raise FileNotFoundError(
            'No file matching the supplied arguments:'
            f'atlas_name={atlas_name}, '
            f'dimension={dimension}, '
            f'dataset={dataset}',
            f'metric={metric}',
        )
    labels = [file.name.split(f'_{metric}')[0] for file in files]
    return files, labels


def _get_qcfc_metric(file_path, metric, group):
    """Get correlation or pvalue of QC-FC."""
    if not isinstance(file_path, list):
        file_path = [file_path]
    qcfc_per_edge = []
    # read subject information here
    for p in file_path:
        qcfc_stats = pd.read_csv(p, sep='\t', index_col=0, header=[0, 1])
        # deal with group info here
        qcfc_stats = qcfc_stats[group]
        df = qcfc_stats.filter(regex=metric)
        df.columns = [col.split('_')[0] for col in df.columns]
        qcfc_per_edge.append(df)
    return qcfc_per_edge


def _get_corr_distance(files_qcfc, labels, group):
    qcfc_per_edge = _get_qcfc_metric(
        files_qcfc, metric='correlation', group=group
    )
    corr_distance = []
    for df, label in zip(qcfc_per_edge, labels):
        atlas_name = label.split('atlas-')[-1].split('_')[0]
        dimension = label.split('nroi-')[-1].split('_')[0]
        pairwise_distance = get_atlas_pairwise_distance(atlas_name, dimension)
        cols = df.columns
        df, _ = spearmanr(pairwise_distance.iloc[:, -1], df)
        df = pd.DataFrame(df[1:, 0], index=cols, columns=[label])
        corr_distance.append(df)

    if len(corr_distance) == 1:
        corr_distance = corr_distance[0]
    else:
        corr_distance = pd.concat(corr_distance, axis=1)

    return {
        'data': corr_distance.T,
        'order': list(GRID_LOCATION.values()),
        'title': 'Correlation between\nnodewise distance and QC-FC',
        'label': "Pearson's correlation",
    }


def _corr_modularity_motion(movement, files_network, labels):
    mean_corr, mean_modularity = [], []
    for file_network, label in zip(files_network, labels):
        modularity = pd.read_csv(file_network, sep='\t', index_col=0)
        mean_modularity.append(modularity.mean())

        corr_modularity = []
        z_movement = movement.apply(zscore)
        for column, _ in modularity.iteritems():
            cur_data = pd.concat(
                (
                    modularity[column],
                    movement[['mean_framewise_displacement']],
                    z_movement[['age', 'gender']],
                ),
                axis=1,
            ).dropna()
            current_strategy = partial_correlation(
                cur_data[column].values,
                cur_data['mean_framewise_displacement'].values,
                cur_data[['age', 'gender']].values,
            )
            current_strategy['strategy'] = column
            corr_modularity.append(current_strategy)
        corr_modularity = pd.DataFrame(corr_modularity).set_index(
            ['strategy']
        )['correlation']
        corr_modularity.columns = [label]
        mean_corr.append(corr_modularity)
    mean_corr = pd.concat(mean_corr, axis=1)
    mean_modularity = pd.concat(mean_modularity, axis=1)
    mean_modularity.columns = labels
    corr_modularity = {
        'data': mean_corr.T,
        'order': list(GRID_LOCATION.values()),
        'title': 'Correlation between\nnetwork modularity and motion',
        'label': "Pearson's correlation",
    }
    network_mod = {
        'data': mean_modularity.T,
        'order': list(GRID_LOCATION.values()),
        'title': 'Identifiability of network structure\nafter denoising',
        'label': 'Mean modularity quality (a.u.)',
    }
    return corr_modularity, network_mod


def _qcfc_fdr(file_qcfc, labels, group):
    """Do FDR correction on qc-fc p-values."""
    sig_per_edge = _get_qcfc_metric(file_qcfc, metric='pvalue', group=group)

    long_qcfc_sig = []
    for df, label in zip(sig_per_edge, labels):
        df = df.melt()
        df['fdr'] = df.groupby('variable')['value'].transform(fdr)
        df = df.groupby('variable').apply(
            lambda x: 100 * x.fdr.sum() / x.fdr.shape[0]
        )
        df = pd.DataFrame(df, columns=[label])
        long_qcfc_sig.append(df)

    if len(long_qcfc_sig) == 1:
        long_qcfc_sig = long_qcfc_sig[0]
        long_qcfc_sig.columns = ['p_corrected']
    else:
        long_qcfc_sig = pd.concat(long_qcfc_sig, axis=1)

    return {
        'data': long_qcfc_sig.T,
        'order': list(GRID_LOCATION.values()),
        'title': 'Percentage of significant QC-FC',
        'xlim': (-5, 105),
        'label': 'Percentage %',
    }


def _get_qcfc_median_absolute(file_qcfc, labels, group):
    """Calculate absolute median and prepare for plotting."""
    qcfc_per_edge = _get_qcfc_metric(
        file_qcfc, metric='correlation', group=group
    )
    qcfc_median_absolute = []
    for df, label in zip(qcfc_per_edge, labels):
        df = df.apply(calculate_median_absolute)
        df.columns = [label]
        qcfc_median_absolute.append(df)

    if len(qcfc_median_absolute) == 1:
        qcfc_median_absolute = qcfc_median_absolute[0]
        title = 'Median absolute deviation\nof QC-FC'
    else:
        qcfc_median_absolute = pd.concat(qcfc_median_absolute, axis=1)
        title = 'Median absolute deviation of QC-FC'
    return {
        'data': pd.DataFrame(qcfc_median_absolute).T,
        'order': list(GRID_LOCATION.values()),
        'title': title,
        'xlim': (-0.02, 0.22),
        'label': 'Median absolute deviation',
    }
