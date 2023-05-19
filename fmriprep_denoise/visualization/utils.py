from pathlib import Path

import pandas as pd
import seaborn as sns

from scipy.stats import zscore, spearmanr
from repo2data.repo2data import Repo2Data

from fmriprep_denoise.features import (
    partial_correlation,
    significant_level,
    calculate_median_absolute,
    get_atlas_pairwise_distance,
)
from fmriprep_denoise.visualization.tables import (
    get_descriptive_data,
    group_name_rename,
)


GRID_LOCATION = {
    (0, 0): "baseline",
    (0, 2): "simple",
    (0, 3): "simple+gsr",
    (1, 0): "scrubbing.5",
    (1, 1): "scrubbing.5+gsr",
    (1, 2): "scrubbing.2",
    (1, 3): "scrubbing.2+gsr",
    (2, 0): "compcor",
    (2, 1): "compcor6",
    (2, 2): "aroma",
    # (2, 3): "aroma+gsr",
}

palette = sns.color_palette("Paired", n_colors=12)
palette_dict = {name: c for c, name in zip(palette[1:], GRID_LOCATION.values())}


# download data
def repo2data_path():
    """Download data using repo2data."""
    data_req_path = Path(__file__).parents[2] / "binder" / "data_requirement.json"
    repo2data = Repo2Data(str(data_req_path))
    data_path = repo2data.install()
    return Path(data_path[0])


def get_data_root():
    """Get motion metric data path root."""
    default_path = Path(__file__).parents[2] / \
        "data" / "fmriprep-denoise-benchmark"
    if not (default_path / "data_requirement.json").exists():
        default_path = repo2data_path()
    return default_path


def _get_palette(order):
    """Get colour palette for each strategy in a specific order."""
    return [palette_dict[item] for item in order]


def _get_connectome_metric_paths(
    dataset, fmriprep_version, metric, atlas_name, dimension, path_root
):
    """Load connectome motion metrics with some give labels."""
    atlas_name = "*" if isinstance(atlas_name, type(None)) else atlas_name
    dimension = (
        "*"
        if isinstance(atlas_name, type(None)) or isinstance(dimension, type(None))
        else dimension
    )
    files = list(
        path_root.glob(
            (
                f"{dataset}/{fmriprep_version}/"
                f"dataset-{dataset}_atlas-{atlas_name}_nroi-{dimension}_"
                f"{metric}.tsv"
            )
        )
    )
    if not files:
        raise FileNotFoundError(
            "No file matching the supplied arguments:"
            f"atlas_name={atlas_name}, "
            f"dimension={dimension}, "
            f"dataset={dataset}",
            f"metric={metric}",
        )
    labels = [file.name.split(f"_{metric}")[0] for file in files]
    return files, labels


def prepare_qcfc_plotting(dataset, fmriprep_version, atlas_name, dimension, path_root):
    """
    Generate three summary metrics for plotting:
        - significant correlation between motion and edges
        - median absolute of correlation between motion and edges
        - distance dependency of edge and motion

    Parameters
    ----------

    dataset : str
        Dataset name.

    fmriprep_version : str {fmrieprep-20.2.1lts, fmrieprep-20.2.5lts}
        fMRIPrep version used for preporcessin.

    atlas_name : None or str
        Atlas name. Default None to get all outputs.

    dimension : None or str
        Atlas dimension. Default None to get all outputs.

    path_root : pathlib.Path
        Path to the input data directory.

    Returns
    -------
    pandas.DataFrame
        Summary metrics by group, by strategy
    """
    ds_qcfc_sig, ds_qcfc_sig_fdr, ds_qcfc_median_absolute, ds_corr_distance = (
        [],
        [],
        [],
        [],
    )
    file_qcfc, qcfc_labels = _get_connectome_metric_paths(
        dataset, fmriprep_version, "qcfc", atlas_name, dimension, path_root
    )

    for p, label in zip(file_qcfc, qcfc_labels):
        label = label.replace(f"dataset-{dataset}_", "")
        # significant correlation between motion and edges
        qcfc_pvalue = _qcfc_bygroup("pvalue", p)
        qcfc_pvalue = qcfc_pvalue.melt(var_name=["groups", "strategy"])

        # fdr correction
        qcfc_pvalue["fdr"] = qcfc_pvalue.groupby(["groups", "strategy"])[
            "value"
        ].transform(significant_level, correction="fdr_bh")
        qcfc_sig = qcfc_pvalue.groupby(["groups", "strategy"]).apply(
            lambda x: 100 * x.fdr.sum() / x.fdr.shape[0]
        )
        qcfc_sig = pd.DataFrame(qcfc_sig, columns=[label])
        ds_qcfc_sig_fdr.append(qcfc_sig)

        # uncorrected p values
        qcfc_pvalue["p_value"] = qcfc_pvalue.groupby(["groups", "strategy"])[
            "value"
        ].transform(significant_level)
        qcfc_sig = qcfc_pvalue.groupby(["groups", "strategy"]).apply(
            lambda x: 100 * x.p_value.sum() / x.p_value.shape[0]
        )
        qcfc_sig = pd.DataFrame(qcfc_sig, columns=[label])
        ds_qcfc_sig.append(qcfc_sig)

        # median absolute of correlation between motion and edges
        qcfc = _qcfc_bygroup("correlation", p)
        mad_qcfc = qcfc.apply(calculate_median_absolute)
        mad_qcfc.name = label
        ds_qcfc_median_absolute.append(mad_qcfc)

        # distance dependency
        cur_atlas_name = label.split("atlas-")[-1].split("_")[0]
        cur_dimension = label.split("nroi-")[-1].split("_")[0]
        pairwise_distance = get_atlas_pairwise_distance(cur_atlas_name, cur_dimension)
        cols = qcfc.columns
        corr_distance_qcfc, _ = spearmanr(pairwise_distance.iloc[:, -1], qcfc)
        corr_distance_qcfc = pd.DataFrame(
            corr_distance_qcfc[1:, 0], index=cols, columns=[label]
        )
        ds_corr_distance.append(corr_distance_qcfc)

    ds_qcfc_sig = pd.concat(ds_qcfc_sig, axis=1)
    ds_qcfc_sig.columns = pd.MultiIndex.from_product(
        [["qcfc_significant"], ds_qcfc_sig.columns]
    )

    ds_qcfc_sig_fdr = pd.concat(ds_qcfc_sig_fdr, axis=1)
    ds_qcfc_sig_fdr.columns = pd.MultiIndex.from_product(
        [["qcfc_fdr_significant"], ds_qcfc_sig_fdr.columns]
    )

    ds_qcfc_median_absolute = pd.concat(ds_qcfc_median_absolute, axis=1)
    ds_qcfc_median_absolute.columns = pd.MultiIndex.from_product(
        [["qcfc_mad"], ds_qcfc_median_absolute.columns]
    )

    ds_corr_distance = pd.concat(ds_corr_distance, axis=1)
    ds_corr_distance.columns = pd.MultiIndex.from_product(
        [["corr_motion_distance"], ds_corr_distance.columns]
    )
    return pd.concat(
        [ds_qcfc_sig, ds_qcfc_sig_fdr, ds_qcfc_median_absolute, ds_corr_distance],
        axis=1,
    )


def prepare_modularity_plotting(
    dataset, fmriprep_version, atlas_name, dimension, path_root, qc
):
    """
    Generate two summary metrics for plotting:
        - Mean modularity
        - Correlation between motion and modularity

    Parameters
    ----------

    dataset : str
        Dataset name.

    fmriprep_version : str {fmrieprep-20.2.1lts, fmrieprep-20.2.5lts}
        fMRIPrep version used for preporcessin.

    atlas_name : None or str
        Atlas name. Default None to get all outputs.

    dimension : None or str
        Atlas dimension. Default None to get all outputs.

    path_root : pathlib.Path
        Path to the input data directory.

    qc : dict
        Movement quality control filter.

    Returns
    -------
    pandas.DataFrame
        Summary metrics by group, by strategy
    """
    files_network, modularity_labels = _get_connectome_metric_paths(
        dataset,
        fmriprep_version,
        "modularity",
        atlas_name,
        dimension,
        path_root,
    )
    _, movement, _ = get_descriptive_data(dataset, fmriprep_version, path_root, **qc)

    ds_mean_corr, ds_mean_modularity, ds_sd_modularity = [], [], []
    for file_network, label in zip(files_network, modularity_labels):
        label = label.replace(f"dataset-{dataset}_", "")

        modularity = pd.read_csv(file_network, sep="\t", index_col=0)
        modularity = pd.concat([movement["groups"], modularity], axis=1)

        mean_by_group, sd_by_group = _calculate_descriptive_modularity(
            modularity, label
        )
        corr_modularity = _calculate_corr_modularity(modularity, movement, label)
        ds_mean_modularity.append(mean_by_group)
        ds_sd_modularity.append(sd_by_group)
        ds_mean_corr.append(corr_modularity)

    # modularity
    ds_mean_modularity = pd.concat(ds_mean_modularity, axis=1)
    ds_mean_modularity.columns = pd.MultiIndex.from_product(
        [["modularity"], ds_mean_modularity.columns]
    )
    ds_sd_modularity = pd.concat(ds_sd_modularity, axis=1)
    ds_sd_modularity.columns = pd.MultiIndex.from_product(
        [["modularity_sd"], ds_sd_modularity.columns]
    )
    # motion and modularity
    ds_mean_corr = pd.concat(ds_mean_corr, axis=1)
    ds_mean_corr.columns = pd.MultiIndex.from_product(
        [["corr_motion_modularity"], ds_mean_corr.columns]
    )
    return pd.concat([ds_mean_modularity, ds_sd_modularity, ds_mean_corr], axis=1)


def _qcfc_bygroup(metric, p):
    """QC/FC statistics organised by groups."""
    qcfc_stats = pd.read_csv(p, sep="\t", index_col=0, header=[0, 1])
    qcfc_stats = qcfc_stats.rename(columns=group_name_rename)

    df = qcfc_stats.filter(regex=metric)
    new_col = pd.MultiIndex.from_tuples(
        [(group, strategy.replace(f"_{metric}", "")) for group, strategy in df.columns],
        names=["groups", "strategy"],
    )
    df.columns = new_col
    return df


def _calculate_corr_modularity(modularity, movement, label):
    """Calculate correlation between motion and network modularity by groups."""
    # motion and modularity
    corr_modularity = []
    z_movement = movement[["mean_framewise_displacement", "age", "gender"]].apply(
        zscore
    )

    # full sample
    for strategy, value in modularity.iloc[:, 1:].iteritems():
        current_df = partial_correlation(
            value,
            movement["mean_framewise_displacement"],
            z_movement[["age", "gender"]],
        )
        current_df["strategy"] = strategy
        current_df["groups"] = "full_sample"
        corr_modularity.append(current_df)

    # by group
    modularity_long = modularity.reset_index().melt(
        id_vars=["index", "groups"], var_name="strategy"
    )
    for (group, strategy), df in modularity_long.groupby(["groups", "strategy"]):
        df = df.set_index("index")
        current_df = partial_correlation(
            df["value"],
            movement.loc[df.index, "mean_framewise_displacement"].values,
            z_movement.loc[df.index, ["age", "gender"]].values,
        )
        current_df["strategy"] = strategy
        current_df["groups"] = group
        corr_modularity.append(current_df)

    corr_modularity = pd.DataFrame(corr_modularity).set_index(["groups", "strategy"])[
        "correlation"
    ]
    corr_modularity.name = label
    return corr_modularity


def _calculate_descriptive_modularity(modularity, label):
    """Calculate mean and sd of network modularity by groups."""

    # by group
    mean_by_group = modularity.groupby(["groups"]).mean()
    mean_by_group = mean_by_group.reset_index()
    mean_by_group = mean_by_group.melt(
        id_vars=["groups"], var_name="strategy", value_name=label
    )
    mean_by_group = mean_by_group.set_index(["groups", "strategy"])

    sd_by_group = modularity.groupby(["groups"]).std()
    sd_by_group = sd_by_group.reset_index()
    sd_by_group = sd_by_group.melt(
        id_vars=["groups"], var_name="strategy", value_name=label
    )
    sd_by_group = sd_by_group.set_index(["groups", "strategy"])

    # full sample
    mean_full_sample = modularity.iloc[:, 1:].mean()
    mean_full_sample.index = pd.MultiIndex.from_product(
        [["full_sample"], mean_full_sample.index], names=["groups", "strategy"]
    )
    mean_full_sample = mean_full_sample.to_frame()
    mean_full_sample.columns = [label]

    sd_full_sample = modularity.iloc[:, 1:].mean()
    sd_full_sample.index = pd.MultiIndex.from_product(
        [["full_sample"], sd_full_sample.index], names=["groups", "strategy"]
    )
    sd_full_sample = sd_full_sample.to_frame()
    sd_full_sample.columns = [label]
    return (
        pd.concat([mean_full_sample, mean_by_group]),
        pd.concat([sd_full_sample, sd_by_group]),
    )


def _get_participants_groups(
    dataset,
    fmriprep_version,
    path_root,
    gross_fd=None,
    fd_thresh=None,
    proportion_thresh=None,
):
    """Get subject group information."""

    # To me deleted if I ever refactor this code.
    confounds_phenotype, movements, groups = get_descriptive_data(
        dataset,
        fmriprep_version,
        path_root,
        gross_fd=gross_fd,
        fd_thresh=fd_thresh,
        proportion_thresh=proportion_thresh,
    )
    participant_groups = movements["groups"]
    return confounds_phenotype, participant_groups, groups


def _get_qcfc_metric(file_path, metric, group):
    """Get correlation or pvalue of QC-FC."""
    if not isinstance(file_path, list):
        file_path = [file_path]
    qcfc_per_edge = []
    # read subject information here
    for p in file_path:
        qcfc_stats = pd.read_csv(p, sep="\t", index_col=0, header=[0, 1])
        # deal with group info here
        qcfc_stats = qcfc_stats[group]
        df = qcfc_stats.filter(regex=metric)
        df.columns = [col.split("_")[0] for col in df.columns]
        qcfc_per_edge.append(df)
    return qcfc_per_edge


def _get_corr_distance(files_qcfc, labels, group):
    """Load correlation of QC/FC with node distances."""
    qcfc_per_edge = _get_qcfc_metric(files_qcfc, metric="correlation", group=group)
    corr_distance = []
    for df, label in zip(qcfc_per_edge, labels):
        atlas_name = label.split("atlas-")[-1].split("_")[0]
        dimension = label.split("nroi-")[-1].split("_")[0]
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
        "data": corr_distance.T,
        "order": list(GRID_LOCATION.values()),
        "title": "Correlation between\nnodewise distance and QC-FC",
        "label": "Pearson's correlation",
    }


def _corr_modularity_motion(movement, files_network, labels):
    """Load correlation of mean FD with network modularity."""
    mean_corr, mean_modularity = [], []
    for file_network, label in zip(files_network, labels):
        modularity = pd.read_csv(file_network, sep="\t", index_col=0)
        modularity = modularity[GRID_LOCATION.values()]  # select strategies
        mean_modularity.append(modularity.mean())

        corr_modularity = []
        z_movement = movement.apply(zscore)
        for strategy in GRID_LOCATION.values():
            cur_data = pd.concat(
                (
                    modularity[strategy],
                    movement[["mean_framewise_displacement"]],
                    z_movement[["age", "gender"]],
                ),
                axis=1,
            ).dropna()
            current_strategy = partial_correlation(
                cur_data[strategy].values,
                cur_data["mean_framewise_displacement"].values,
                cur_data[["age", "gender"]].values,
            )
            current_strategy["strategy"] = strategy
            corr_modularity.append(current_strategy)
        corr_modularity = pd.DataFrame(corr_modularity).set_index(["strategy"])[
            "correlation"
        ]
        corr_modularity.columns = [label]
        mean_corr.append(corr_modularity)
    mean_corr = pd.concat(mean_corr, axis=1)
    mean_modularity = pd.concat(mean_modularity, axis=1)
    mean_modularity.columns = labels
    corr_modularity = {
        "data": mean_corr.T,
        "order": list(GRID_LOCATION.values()),
        "title": "Correlation between\nnetwork modularity and motion",
        "label": "Pearson's correlation",
    }
    network_mod = {
        "data": mean_modularity.T,
        "order": list(GRID_LOCATION.values()),
        "title": "Identifiability of network structure\nafter denoising",
        "label": "Mean modularity quality (a.u.)",
    }
    return corr_modularity, network_mod


def _qcfc_pvalue(file_qcfc, labels, group, fdr):
    """Get qc-fc p-values."""
    sig_per_edge = _get_qcfc_metric(file_qcfc, metric="pvalue", group=group)

    long_qcfc_sig = []

    for df, label in zip(sig_per_edge, labels):
        df = df.melt()
        if fdr:
            df["fdr"] = df.groupby("variable")["value"].transform(
                significant_level, correction="fdr_bh"
            )
            df = df.groupby("variable").apply(
                lambda x: 100 * x.fdr.sum() / x.fdr.shape[0]
            )
        else:
            df["p_value"] = df.groupby("variable")["value"].transform(significant_level)
            df = df.groupby("variable").apply(
                lambda x: 100 * x.p_value.sum() / x.p_value.shape[0]
            )
        df = pd.DataFrame(df, columns=[label])
        long_qcfc_sig.append(df)

    if len(long_qcfc_sig) == 1:
        long_qcfc_sig = long_qcfc_sig[0]
        long_qcfc_sig.columns = ["p_value"]
    else:
        long_qcfc_sig = pd.concat(long_qcfc_sig, axis=1)

    return {
        "data": long_qcfc_sig.T,
        "order": list(GRID_LOCATION.values()),
        "title": "Percentage of significant QC-FC",
        "xlim": (-5, 40),
        "label": "Percentage %",
    }


def _get_qcfc_absolute_median(file_qcfc, labels, group):
    """Calculate absolute median value and prepare for plotting."""
    qcfc_per_edge = _get_qcfc_metric(file_qcfc, metric="correlation", group=group)
    qcfc_median_absolute = []
    for df, label in zip(qcfc_per_edge, labels):
        df = df.apply(calculate_median_absolute)
        df.columns = [label]
        qcfc_median_absolute.append(df)

    if len(qcfc_median_absolute) == 1:
        qcfc_median_absolute = qcfc_median_absolute[0]
        title = "Absolute median value \nof QC-FC"
    else:
        qcfc_median_absolute = pd.concat(qcfc_median_absolute, axis=1)
        title = "Absolute median value of QC-FC"
    return {
        "data": pd.DataFrame(qcfc_median_absolute).T,
        "order": list(GRID_LOCATION.values()),
        "title": title,
        "xlim": (0.00, 0.3),
        "label": "Absolute median value",
    }
