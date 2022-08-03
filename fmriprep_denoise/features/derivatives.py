import tarfile
from pathlib import Path
import pandas as pd
from nilearn.connectome import ConnectivityMeasure

from fmriprep_denoise.data.fmriprep import fetch_fmriprep_derivative


project_root = Path(__file__).parents[2]
inputs = project_root / 'inputs'
group_info_column = {'ds000228': 'Child_Adult', 'ds000030': 'diagnosis'}


def compute_connectome(atlas, extracted_path, dataset, file_pattern):
    """Compute connectome of all valid data.

    Parameters
    ----------

    atlas : str
        Atlas name matching keys in fmriprep_denoise.data.atlas.ATLAS_METADATA.

    extracted_path : pathlib.Path
        Path object to where the time series were saved.

    dataset : str
        Name of the dataset.

    file_pattern : str
        Details about the atlas and description of the file.

    Returns
    -------
    pandas.DataFrame, pandas.DataFrame
        Flatten connectomes and phenotypes.
    """
    phenotype = _load_phenotype(dataset=dataset)
    participant_id = phenotype.index.to_list()
    valid_ids, valid_ts = _load_valid_timeseries(
        atlas, extracted_path, participant_id, file_pattern
    )
    correlation_measure = ConnectivityMeasure(
        kind='correlation', vectorize=True, discard_diagonal=True
    )
    subject_conn = correlation_measure.fit_transform(valid_ts)
    subject_conn = pd.DataFrame(subject_conn, index=valid_ids)
    if subject_conn.shape[0] != phenotype.shape[0]:
        print('take conjunction of the phenotype and connectome.')
        idx = subject_conn.index.intersection(phenotype.index)
        subject_conn, phenotype = (
            subject_conn.loc[idx, :],
            phenotype.loc[idx, :],
        )
    return subject_conn, phenotype


def check_extraction(input_path, extracted_path_root=None):
    """Check if the tar.gz of a fmriprep dataset has been extracted.

    Parameters
    ----------

    input_path : pathlib.Path
        Location of the tar.gz of the fMRIPrep output.

    extracted_path_root : None, pathlib.Path
        Destination of the extraction.

    Returns
    -------

    pathlib.Path
        Correct file path of the extracted dataset.
    """
    dir_name = input_path.name.split('.tar')[0]
    extracted_path_root = (
        inputs if extracted_path_root is None else extracted_path_root
    )

    extracted_path = extracted_path_root / dir_name

    if not extracted_path.is_dir() and input_path.is_file():
        print(
            f'Cannot file extracted file at {extracted_path}. ' 'Extracting...'
        )
        with tarfile.open(input_path, 'r:gz') as tar:
            tar.extractall(extracted_path_root)
    return extracted_path


def _load_phenotype(dataset):
    """Get subjects that were processed and passed quality controls."""
    if dataset not in group_info_column:
        raise KeyError(f'Unsupported dataset {dataset}')

    # read relevant files
    path_phenotype = (
        inputs
        / f'dataset-{dataset}/dataset-{dataset}_desc-movement_phenotype.tsv'
    )
    path_original_participants_info = inputs / f'{dataset}/participants.tsv'

    phenotype = pd.read_csv(path_phenotype, sep='\t', index_col=0)
    phenotype['age'] = phenotype['age'].astype('float')
    phenotype['gender'] = phenotype['gender'].astype('float')
    participants = pd.read_csv(
        path_original_participants_info, sep='\t', index_col=0
    )
    if dataset == 'ds000030':
        mask_quality = participants['ghost_NoGhost'] == 'No_ghost'
        participants = participants[mask_quality]
    # changen the header to something consistent
    participant_groups = group_info_column[dataset]
    participants = participants.rename(columns={participant_groups: 'groups'})

    phenotype = pd.concat(
        [phenotype, participants['groups']], axis=1, join='inner'
    )
    return phenotype.sort_index()


def _load_valid_timeseries(
    atlas, extracted_path, participant_id, file_pattern
):
    """Load time series from tsv file."""
    valid_ids, valid_ts = [], []
    for subject in participant_id:
        subject_path = extracted_path / f'atlas-{atlas}' / subject
        file_path = list(
            subject_path.glob(f'{subject}_*_{file_pattern}_timeseries.tsv')
        )
        if len(file_path) > 1:
            raise ValueError('Found more than one valid file.' f'{file_path}')
        file_path = file_path[0]
        if file_path.stat().st_size > 1:
            ts = pd.read_csv(file_path, sep='\t', header=0)
            valid_ids.append(subject)
            valid_ts.append(ts.values)
        else:
            continue
    return valid_ids, valid_ts
