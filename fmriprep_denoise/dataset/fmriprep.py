from pathlib import Path

import json
import pandas as pd


from sklearn.utils import Bunch

STRATEGY_FILE = "benchmark_strategies.json"

PHENOTYPE_INFO = {
    "ds000228": {
        "columns": ["Age", "Gender", "Child_Adult"],
        "replace": {"Age": "age", "Gender": "gender", "Child_Adult": "groups"},
    },
    "ds000030": {
        "columns": ["age", "gender", "diagnosis"],
        "replace": {"diagnosis": "groups"},
    },
}


def fetch_fmriprep_derivative(
    dataset_name,
    participant_tsv_path,
    path_fmriprep_derivative,
    specifier,
    subject=None,
    space="MNI152NLin2009cAsym",
    aroma=False,
):
    """Fetch fmriprep derivative and return nilearn.dataset.fetch* like output.
    Load functional image, confounds, and participants.tsv only.

    Parameters
    ----------

    dataset_name : str
        Dataset name.

    participant_tsv_path : pathlib.Path
        A pathlib path point to the BIDS participants file.

    path_fmriprep_derivative : pathlib.Path
        A pathlib path point to the BIDS participants file.

    specifier : string
        Text in a fmriprep file name, in between sub-<subject>_ses-<session>_
        and `space-<template>`.

    subject : string, or list of string default None
        subject id. If none, return all results.

    space : string, default "MNI152NLin2009cAsym"
        Template flow template name in fmriprep output.

    aroma : boolean, default False
        Use ICA-AROMA processed data or not.

    Returns
    -------
    sklearn.utils.Bunch
        nilearn.dataset.fetch* like output.

    """

    # participants tsv from the main dataset
    if not participant_tsv_path.is_file():
        raise FileNotFoundError(f"Cannot find {participant_tsv_path}")
    if participant_tsv_path.name != "participants.tsv":
        raise FileNotFoundError(
            f"File {participant_tsv_path} is not a BIDS participant file."
        )
    participant_tsv = pd.read_csv(
        participant_tsv_path, index_col=["participant_id"], sep="\t"
    )
    # images and confound files
    if subject is None:
        subject_dirs = path_fmriprep_derivative.glob("sub-*/")
    elif isinstance(subject, str):
        subject_dirs = path_fmriprep_derivative.glob(f"sub-{subject}/")
    elif isinstance(subject, list):
        subject_dirs = []
        for s in subject:
            s_path = path_fmriprep_derivative / f"sub-{s}"
            if s_path.is_dir():
                subject_dirs.append(s_path)
    else:
        raise ValueError("Unsupported input for subject.")

    func_img_path, confounds_tsv_path, include_subjects = [], [], []
    for subject_dir in subject_dirs:
        subject = subject_dir.name
        desc = "smoothAROMAnonaggr" if aroma else "preproc"
        space = "MNI152NLin6Asym" if aroma else space
        cur_func = (
            subject_dir
            / "func"
            / f"{subject}_{specifier}_space-{space}_desc-{desc}_bold.nii.gz"
        )
        cur_confound = (
            subject_dir
            / "func"
            / f"{subject}_{specifier}_desc-confounds_timeseries.tsv"
        )

        if cur_func.is_file() and cur_confound.is_file():
            func_img_path.append(str(cur_func))
            confounds_tsv_path.append(str(cur_confound))
            include_subjects.append(subject)

    return Bunch(
        dataset_name=dataset_name,
        func=func_img_path,
        confounds=confounds_tsv_path,
        phenotypic=participant_tsv.loc[include_subjects, :],
    )


def get_prepro_strategy(strategy_name=None):
    """
    Select a single preprocessing strategy and associated parameters.

    Parameter
    ---------

    strategy_name : None or str
        Name of the denoising strategy. See benchmark_strategies.json.
        Default to None, returns all strategies.

    Return
    ------

    dict
        Denosing strategy parameter to pass to load_confounds.
    """
    strategy_file = Path(__file__).parent / STRATEGY_FILE
    with open(strategy_file, "r") as file:
        benchmark_strategies = json.load(file)

    if isinstance(strategy_name, str) and strategy_name not in benchmark_strategies:
        raise NotImplementedError(
            f"Strategy '{strategy_name}' is not implemented. Select from the"
            f"following: {[*benchmark_strategies]}"
        )

    if strategy_name is None:
        print("Process all strategies.")
        return benchmark_strategies
    (f"Process strategy '{strategy_name}'.")
    return {strategy_name: benchmark_strategies[strategy_name]}


def generate_movement_summary(data):
    """Generate and save movement stats and phenotype for openneuro datasets.

    Parameters
    ----------

    data : sklearn.utils.Bunch
        Dataset retrieved through fmriprep_denoise.fetch_fmriprep_derivative

    Returns
    -------
    pandas.DataFrame
        Participants phenotype reduced to: age, gender, group, motion.
        Motion is calculated through mean framewise displacement.

    """
    # get motion QC related metrics from confound files
    group_mean_fd = pd.DataFrame()
    group_mean_fd.index = group_mean_fd.index.set_names("participant_id")
    for confounds in data.confounds:
        subject_id = confounds.split("/")[-1].split("_")[0]
        confounds = pd.read_csv(confounds, sep="\t")
        mean_fd = confounds["framewise_displacement"].mean()
        group_mean_fd.loc[subject_id, "mean_framewise_displacement"] = mean_fd

    # load gender and age as confounds for the developmental dataset
    participants = data.phenotypic.copy()

    covar = participants.loc[:, PHENOTYPE_INFO[data.dataset_name]["columns"]]
    fix_col_name = PHENOTYPE_INFO[data.dataset_name].get("replace", False)
    if isinstance(fix_col_name, dict):
        covar = covar.rename(columns=fix_col_name)
    covar.loc[covar["gender"] == "F", "gender"] = 1
    covar.loc[covar["gender"] == "M", "gender"] = 0
    covar["gender"] = covar["gender"].astype("float")
    covar["age"] = covar["age"].astype("float")
    return pd.concat((group_mean_fd, covar), axis=1, join="inner")
