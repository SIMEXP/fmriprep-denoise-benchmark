import json
import tarfile
from pathlib import Path
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np
from fmriprep_denoise.visualization import tables

MOTION_QC_FILE = "motion_qc.json"
project_root = Path(__file__).parents[2]
inputs = project_root / "data"
group_info_column = {"ds000228": "Child_Adult", "ds000030": "diagnosis"}

def get_qc_criteria(strategy_name=None):
    motion_qc_file = Path(__file__).parent / MOTION_QC_FILE
    with open(motion_qc_file, "r") as file:
        qc_strategies = json.load(file)

    if isinstance(strategy_name, str) and strategy_name not in qc_strategies:
        raise NotImplementedError(
            f"Strategy '{strategy_name}' is not implemented. Select from the following: None, {[*qc_strategies]}"
        )

    if strategy_name is None:
        print("No motion QC.")
        return {"gross_fd": None, "fd_thresh": None, "proportion_thresh": None}

    print(f"Process strategy '{strategy_name}'.")
    return qc_strategies[strategy_name]

def compute_connectome(
    atlas,
    extracted_path,
    dataset,
    fmriprep_version,
    path_root,
    file_pattern,
    full_roi_list,
    gross_fd=None,
    fd_thresh=None,
    proportion_thresh=None,
    impute_strategy="mean"
):
    _, phenotype, _ = tables.get_descriptive_data(
        dataset, fmriprep_version, path_root, gross_fd, fd_thresh, proportion_thresh
    )
    participant_id = phenotype.index.tolist()
    valid_ids, valid_ts = _load_valid_timeseries(
        atlas, extracted_path, participant_id, file_pattern, full_roi_list
    )

    missing_matrix = np.array([np.isnan(ts).any(axis=0) for ts in valid_ts])

    valid_ts, kept_rois, roi_mask = handle_missing_data_strategy_b(
        valid_ts, missing_matrix, full_roi_list, threshold=0.10, impute_strategy=impute_strategy
    )

    correlation_measure = ConnectivityMeasure(
        kind="correlation", vectorize=True, discard_diagonal=True
    )
    subject_conn = correlation_measure.fit_transform(valid_ts)
    subject_conn = pd.DataFrame(subject_conn, index=valid_ids)

    if subject_conn.shape[0] != phenotype.shape[0]:
        idx = subject_conn.index.intersection(phenotype.index)
        subject_conn = subject_conn.loc[idx]
        phenotype = phenotype.loc[idx]

    return subject_conn, phenotype

def handle_missing_data_strategy_b(subject_ts_list, missing_matrix, roi_labels, threshold=0.10, impute_strategy="mean"):
    roi_missing_rate = np.mean(missing_matrix, axis=0)
    roi_mask = roi_missing_rate < threshold
    kept_roi_labels = np.array(roi_labels)[roi_mask]

    cleaned_ts = []
    for subj_ts in subject_ts_list:
        ts_filtered = subj_ts[:, roi_mask]
        if np.isnan(ts_filtered).any():
            if impute_strategy == "mean":
                imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
                ts_filtered = imputer.fit_transform(ts_filtered)
            elif impute_strategy == "knn":
                imputer = KNNImputer(n_neighbors=5, weights="distance")
                ts_filtered = imputer.fit_transform(ts_filtered.T).T  # Transpose in and out for KNN across ROIs
            else:
                raise ValueError(f"Unsupported impute_strategy: {impute_strategy}")
        cleaned_ts.append(ts_filtered)

    return cleaned_ts, kept_roi_labels.tolist(), roi_mask

def check_extraction(input_path, extracted_path_root=None):
    dir_name = input_path.name.split(".tar")[0]
    extracted_path_root = inputs if extracted_path_root is None else extracted_path_root
    extracted_path = extracted_path_root / dir_name

    if not extracted_path.is_dir() and input_path.is_file():
        print(f"Cannot find extracted file at {extracted_path}. Extracting...")
        with tarfile.open(input_path, "r:gz") as tar:
            tar.extractall(extracted_path_root)
    return extracted_path

def _load_valid_timeseries(atlas, extracted_path, participant_id, file_pattern, full_roi_list):
    """Load time series from tsv file and align the columns to the full ROI list."""
    valid_ids, valid_ts = [], []
    for subject in participant_id:
        subject_path = extracted_path / subject  # <-- FIXED
        file_path = list(
            subject_path.glob(f"{subject}_*_{file_pattern}_timeseries.tsv")
        )
        print("Load_valid_timeseries from: " + str(file_path))
        if len(file_path) > 1:
            raise ValueError("Found more than one valid file: " + str(file_path))
        if not file_path:
            continue
        file_path = file_path[0]
        if file_path.stat().st_size > 1:
            df = pd.read_csv(file_path, sep="\t", header=None)
            df.columns = full_roi_list
            valid_ids.append(subject)
            valid_ts.append(df.values)
        else:
            continue
    return valid_ids, valid_ts
def load_full_roi_list(atlas_tsv_path):
    atlas_df = pd.read_csv(atlas_tsv_path, sep="\t", header=None)
    full_roi_list = atlas_df[1].tolist()
    return full_roi_list
