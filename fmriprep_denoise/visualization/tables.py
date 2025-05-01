import pandas as pd

# changed to match json 
fd2label = {0.5: "scrubbing.5", 0.2: "scrubbing.2"}

# fd2label = {0.1: "corrMatrixScrub", 0.2: "corrMatrixScrubGSR", 0.3: "corrMatrixMotion", 0.4: "corrMatrixMotionGSR", 0.5: "corrMatrixCompCor", 0.6: "corrMatrixICA"}


# fd2label = {0.1: "Scrub", 0.2: "ScrubGSR", 0.3: "Motion", 0.4: "MotionGSR", 0.5: "CompCor", 0.6: "ICA"} #needed to match confounds phenotype tsv when running make_maniscript_figures

group_name_rename = {
    "CONTROL": "control",
    "BIPOLAR": "bipolar",
    "SCHZ": "schizophrenia",
}
group_order = {
    "ds000228": ["adult", "child"],
    "ds000030": ["control", "ADHD", "bipolar", "schizophrenia"],
}


def lazy_demographic(
    dataset,
    fmriprep_version,
    path_root,
    gross_fd=None,
    fd_thresh=None,
    proportion_thresh=None,
):
    """
    Very lazy report of demographic information.

    Parameters
    ----------

    dataset : str
        Dataset name.

    fmriprep_version : str {fmrieprep-20.2.1lts, fmrieprep-20.2.5lts}
        fMRIPrep version used for preporcessin.

    path_root : pathlib.Path
        Root of the metrics output.

    gross_fd : None or float
        Gross mean framewise dispancement threshold.

    fd_thresh : None or float
        Volume level framewise dispancement threshold.

    proportion_thresh : None or float
        Proportion of volumes scrubbed threshold.


    Returns
    -------

    pandas.DataFrame
        Descriptive stats of age and gender.
    """
    _, df, groups = get_descriptive_data(
        dataset, fmriprep_version, path_root, gross_fd, fd_thresh, proportion_thresh
    )
    n_female = df["gender"].sum()
    n_female = pd.Series([n_female], index=["n_female"])
    full = df.describe()["age"]
    full = pd.concat([full, n_female])
    full.name = "full sample"

    desc = [full]
    for g in groups:
        sub_group = df[df["groups"] == g].describe()["age"]
        n_female = df.loc[df["groups"] == g, "gender"].sum()
        n_female = pd.Series([n_female], index=["n_female"])
        sub_group = pd.concat([sub_group, n_female])
        sub_group.name = g
        desc.append(sub_group)

    return pd.concat(desc, axis=1)


def get_descriptive_data(
    dataset,
    fmriprep_version,
    path_root,
    gross_fd=None,
    fd_thresh=None,
    proportion_thresh=None,
):
    """
    Get the data frame of all descriptive data needed for a dataset.

    Parameters
    ----------

    dataset : str
        Dataset name.

    fmriprep_version : str {fmrieprep-20.2.1lts, fmrieprep-20.2.5lts}
        fMRIPrep version used for preporcessin.

    path_root : pathlib.Path
        Root of the metrics output.

    gross_fd : None or float
        Gross mean framewise dispancement threshold.

    fd_thresh : None or float
        Volume level framewise dispancement threshold.

    proportion_thresh : None or float
        Proportion of volumes scrubbed threshold.


    Returns
    -------

    pandas.DataFrame, pandas.DataFrame, list
        confounds phenotype, movements,  groups
    """
    if not fd2label.get(fd_thresh, False) and fd_thresh is not None:
        raise ValueError(
            "We did not generate metric with scrubbing threshold set at "
            f"framewise displacement = {fd_thresh} mm."
        )
    # load basic data
    movements = (
        path_root
        / dataset
        / fmriprep_version
        / f"dataset-{dataset}_desc-movement_phenotype.tsv"
    )
    movements = pd.read_csv(movements, index_col=[0, -1], sep="\t")
    movements = movements.rename(index=group_name_rename)
    movements = movements.reset_index(level="groups")

    path_dof = (
        path_root
        / dataset
        / fmriprep_version
        / f"dataset-{dataset}_desc-confounds_phenotype.tsv"
    )
    confounds_phenotype = pd.read_csv(path_dof, header=[0, 1], index_col=0, sep="\t")

    # filter data by gross fd
    if gross_fd is not None:
        keep_gross_fd = movements["mean_framewise_displacement"] <= gross_fd
        keep_gross_fd = movements.index[keep_gross_fd]
    else:
        keep_gross_fd = movements.index

    # filter data by proportion vol scrubbed
    if fd_thresh is not None and proportion_thresh is not None:
        scrub_label = (fd2label[fd_thresh], "excised_vol_proportion")
        exclude_scrub = confounds_phenotype[scrub_label] > proportion_thresh
        keep_scrub = confounds_phenotype.index[~exclude_scrub]
    else:
        keep_scrub = confounds_phenotype.index
    mask_motion = keep_gross_fd.intersection(keep_scrub)

    if dataset not in group_order:
        groups = movements["groups"].unique().tolist()
    else:
        groups = group_order[dataset]
    movements = movements.loc[mask_motion, :]
    confounds_phenotype = confounds_phenotype.loc[mask_motion, :]
    return confounds_phenotype, movements, groups
