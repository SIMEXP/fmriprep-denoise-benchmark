import pandas as pd

from sklearn.utils import Bunch

from nilearn.interfaces.fmriprep import load_confounds_strategy, load_confounds


PHENOTYPE_INFO = {
    "ds000228":
        {"columns": ["Age", "Gender"],
         "replace": {'Age': 'age', 'Gender': 'gender'}},
    "ds000030":
        {"columns": ["age", "gender"]},
}


def fetch_fmriprep_derivative(dataset_name, participant_tsv_path, path_fmriprep_derivative,
                              specifier, subject=None, space="MNI152NLin2009cAsym", aroma=False):
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

    subject : string, default None
        subject id. If none, return all results.

    space : string, default "MNI152NLin2009cAsym"
        Template flow tempate name in fmriprep output.

    aroma : boolean, default False
        Use ICA-AROMA processed data or not.

    Returns
    -------
    sklearn.utils.Bunch
        nilearn.dataset.fetch* like output.

    """

    # participants tsv from the main dataset
    if not participant_tsv_path.is_file():
        raise(FileNotFoundError,
              f"Cannot find {participant_tsv_path}")
    if participant_tsv_path.name != "participants.tsv":
        raise(FileNotFoundError,
              f"File {participant_tsv_path} "
              "is not a BIDS participant file.")
    participant_tsv = pd.read_csv(participant_tsv_path,
                                  index_col=["participant_id"],
                                  sep="\t")
    # images and confound files
    if subject is None:
        subject_dirs = path_fmriprep_derivative.glob("sub-*/")
    else:
        subject_dirs = path_fmriprep_derivative.glob(f"sub-{subject}/")

    func_img_path, confounds_tsv_path, include_subjects = [], [], []
    for subject_dir in subject_dirs:
        subject = subject_dir.name
        desc = "smoothAROMAnonaggr" if aroma else "preproc"
        space = "MNI152NLin6Asym" if aroma else space
        cur_func = (subject_dir / "func" /
            f"{subject}_{specifier}_space-{space}_desc-{desc}_bold.nii.gz")
        cur_confound = (subject_dir / "func" /
            f"{subject}_{specifier}_desc-confounds_timeseries.tsv")

        if cur_func.is_file() and cur_confound.is_file():
            func_img_path.append(str(cur_func))
            confounds_tsv_path.append(str(cur_confound))
            include_subjects.append(subject)

    return Bunch(
        dataset_name=dataset_name,
        func=func_img_path,
        confounds=confounds_tsv_path,
        phenotypic=participant_tsv.loc[include_subjects, :]
        )


def phenotype_movement(data):
    """Retreive movement stats and phenotype for ds000228."""
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
    covar = participants.loc[:, PHENOTYPE_INFO[data.dataset_name]['columns']]
    fix_col_name = PHENOTYPE_INFO[data.dataset_name].get("replace", False)
    if isinstance(fix_col_name, dict):
        covar = covar.rename(columns=fix_col_name)
    covar.loc[covar['gender'] == 'F', 'gender'] = 1
    covar.loc[covar['gender'] == 'M', 'gender'] = 0
    covar['gender'] = covar['gender'].astype('float')

    return pd.concat((group_mean_fd, covar), axis=1)
