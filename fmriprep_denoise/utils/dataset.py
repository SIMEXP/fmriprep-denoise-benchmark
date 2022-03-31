from pathlib import Path

from nilearn.connectome import ConnectivityMeasure

import tarfile
import pandas as pd


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
