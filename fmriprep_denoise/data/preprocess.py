from pathlib import Path

import tarfile
import json
import pandas as pd

from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure

from sklearn.utils import Bunch

from fmriprep_denoise.data import fetch_atlas_path


PHENOTYPE_INFO = {
    "ds000228":
        {"columns": ["Age", "Gender"],
         "replace": {'Age': 'age', 'Gender': 'gender'}},
    "ds000030":
        {"columns": ["age", "gender"]},
}

STRATEGY_FILE = "benchmark_strategies.json"


def get_prepro_strategy(strategy_name=None):
    """Select a sinlge preprocessing strategy and associated parametes."""
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
    return benchmark_strategies[strategy_name]


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


def load_valid_timeseries(atlas, extracted_path, participant_id, file_pattern):
    valid_ids, valid_ts = [], []
    for subject in participant_id:
        file_path = list(
            (extracted_path / f"atlas-{atlas}" \
                / subject).glob(f"{subject}_*_{file_pattern}_timeseries.tsv")
        )[0]
        ts = _load_timeseries(file_path)
        if isinstance(ts, pd.DataFrame):
            valid_ids.append(subject)
            valid_ts.append(ts.values)
    return valid_ids, valid_ts


def compute_connectome(valid_subject_id, valid_subject_ts):
    correlation_measure = ConnectivityMeasure(kind='correlation',
                                              vectorize=True,
                                              discard_diagonal=True)
    subject_conn = correlation_measure.fit_transform(valid_subject_ts)
    subject_conn = pd.DataFrame(subject_conn, index=valid_subject_id)
    return subject_conn


def create_atlas_masker(atlas_name, dimension, subject_mask, detrend=True, nilearn_cache=""):
    """Create masker given metadata.
    Parameters
    ----------
    atlas_name : str
        Atlas name. Must be a key in ATLAS_METADATA.
    """
    atlas = fetch_atlas_path(atlas_name, dimension)
    labels = list(range(1, atlas.labels.shape[0] + 1))

    if atlas.type == 'dseg':
        masker = NiftiLabelsMasker(atlas.maps, labels=labels,
                                   mask_img=subject_mask, detrend=detrend)
    elif atlas.type == 'probseg':
        masker = NiftiMapsMasker(atlas.maps,
                                 mask_img=subject_mask, detrend=detrend)
    if nilearn_cache:
        masker = masker.set_params(memory=nilearn_cache, memory_level=1)

    return masker, labels


def check_extraction(input_path, extracted_path_root=None):
    dir_name = input_path.name.split('.tar')[0]
    extracted_path_root = Path(__file__).parents[2] / 'inputs'  \
        if extracted_path_root is None \
        else extracted_path_root

    extracted_path = extracted_path_root / dir_name

    if not extracted_path.is_dir() and input_path.is_file():
        print(
            f'Cannot file extracted file at {extracted_path}. '
            'Extracting...'
        )
        with tarfile.open(input_path, "r:gz") as tar:
            tar.extractall(extracted_path_root)
    return extracted_path


def load_phenotype(dataset="ds000228"):
    project_root = Path(__file__).parents[2]
    phenotype_path = project_root / f"inputs/dataset-{dataset}/" / \
        f"dataset-{dataset}_desc-movement_phenotype.tsv"
    phenotype = pd.read_csv(phenotype_path,
                            sep='\t', index_col=0, header=0)
    return phenotype.sort_index()


def _load_timeseries(file_path):
    if file_path.stat().st_size > 1:
        return pd.read_csv(file_path,sep='\t', header=0)
    return None

