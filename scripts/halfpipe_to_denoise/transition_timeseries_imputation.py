#!/usr/bin/env python3
"""
Two-pass NaN handling and cleaning pipeline for Halfpipe-style fMRI timeseries.
Pass 1: Compute global ROI-wise missing data rate across all subjects.
Pass 2: Apply global ROI mask, impute, clean, and save reformatted timeseries.
Additionally, count (across all subjects) the number of times each ROI required
global mean imputation (i.e. the ROI was fully missing for that subject).
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from nilearn.signal import clean

FEATURE_RENAME_MAP = {
    "corrMatrixCompCor": "compcor",
    "corrMatrixICA": "aroma",
    "corrMatrixMotion": "simple",
    "corrMatrixMotionGSR": "simple+gsr",
    "corrMatrixScrubGSR": "scrubbing.5+gsr",
    "corrMatrixScrub": "scrubbing.5",
}

# FEATURE_RENAME_MAP = {
#     "baseline": "baseline",
#     "compcor": "compcor",
#     "aroma": "aroma",
#     "simple": "simple",
#     "simpleGsr": "simple+gsr",
#     "scrubbing5Gsr": "scrubbing.5+gsr",
#     "scrubbing5": "scrubbing.5",
# }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--space", default="MNI152NLin2009cAsym", type=str)
    parser.add_argument("--nroi", required=True, type=int,
                        help="Number of ROIs to expect (e.g., 434 for Schaefer2018Combined).")
    parser.add_argument("--atlas", type=str, required=True,
                        help="Name of the atlas TSV file (e.g., path/to/atlas_labels.tsv) with two columns: numeric and ROI name.")
    parser.add_argument("--nan_threshold", default=0.5, type=float,
                        help="Global ROI drop threshold based on NaN percentage across all subjects")
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        stream=sys.stdout)

def load_timeseries(file_path):
    return pd.read_csv(file_path, sep="\t", header=None)

def load_full_roi_list(atlas_tsv_path):
    atlas_df = pd.read_csv(atlas_tsv_path, sep="\t", header=None)
    full_roi_list = atlas_df[1].tolist()
    return full_roi_list

def parse_filename(filename):
    subject = feature = atlas = None
    parts = filename.split("_")
    for part in parts:
        if part.startswith("sub-"):
            subject = part
        elif part.startswith("feature-"):
            feature = part.replace("feature-", "")
        elif part.startswith("atlas-"):
            atlas = part.replace("atlas-", "")
    return subject, feature, atlas

def check_orientation_and_add_header(df, expected_nroi):
    rows, cols = df.shape
    if cols != expected_nroi and rows == expected_nroi:
        logging.info("Transposing DataFrame to match [time x ROI] format.")
        df = df.T
    df.columns = [str(i) for i in range(df.shape[1])]
    return df

def compute_global_nan_mask(ts_files, expected_nroi):
    nan_sum = pd.Series(0.0, index=[str(i) for i in range(expected_nroi)], dtype=float)
    num_subjects = 0

    for file_path in ts_files:
        try:
            df = load_timeseries(file_path)
            df = check_orientation_and_add_header(df, expected_nroi)
            for i in range(expected_nroi):
                col = str(i)
                if col not in df.columns:
                    df[col] = np.nan
            nan_frac = df.isna().mean(axis=0)
            nan_sum += nan_frac
            num_subjects += 1
        except Exception as e:
            logging.warning(f"Skipping {file_path} due to error: {e}")

    if num_subjects == 0:
        raise RuntimeError("No valid subjects found for NaN rate computation.")

    avg_nan_rate = nan_sum / num_subjects
    return avg_nan_rate

# def impute_and_clean(df, column_names):
#     placeholder = -9999

#     # Step 1: Pre-fill fully NaN columns with placeholder (we do this to allow imputation)
#     fully_nan_cols = df.columns[df.isna().all()]
#     if len(fully_nan_cols) > 0:
#         logging.warning(
#             f"{len(fully_nan_cols)} ROIs are fully NaN and will be pre-filled with placeholder ({placeholder}): {list(fully_nan_cols)}"
#         )
#         df[fully_nan_cols] = placeholder

#     # Step 2: Fit the imputer on partially missing data.
#     imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
#     imputer.fit(df)

#     # Step 3: Replace any remaining NaNs with placeholder.
#     df_copy = df.replace({np.nan: placeholder})

#     # Step 4: Transform using the fitted imputer.
#     imputed = imputer.transform(df_copy)

#     # Step 5: Build imputed DataFrame.
#     imputed_df = pd.DataFrame(imputed, columns=df.columns, index=df.index)

#     # Step 6: Identify any columns still containing NaNs (should be only columns that were completely missing)
#     if imputed_df.isna().any().any():
#         subject_global_mean = np.nanmean(imputed_df.values)
#         imputed_df = imputed_df.fillna(subject_global_mean)
#         logging.warning("Some ROIs were still NaN after imputation and were filled with subject-global mean.")
#         print(subject_global_mean)

#     # Step 7: Clean with nilearn.
#     cleaned = clean(imputed_df.values, detrend=True, standardize=True)
#     return pd.DataFrame(cleaned, columns=column_names, index=df.index)

def impute_and_clean(df, column_names):
    # Identify which columns are completely missing across time.
    fully_missing_cols = df.columns[df.isna().all()]
    # Identify columns that are not completely missing.
    partially_missing_cols = df.columns.difference(fully_missing_cols)
    
    # For columns that have at least some data, impute using the column mean.
    if len(partially_missing_cols) > 0:
        imputer = SimpleImputer(strategy="mean")
        df.loc[:, partially_missing_cols] = imputer.fit_transform(df[partially_missing_cols])
    
    # Now, for each row (i.e. for each timepoint), fill in any remaining NaNs
    # (which occur in fully missing columns) with the mean of the row
    # computed from the available (non-NaN) values for that row.
    # This ensures that each timepoint gets its own unique replacement value
    # rather than a single global number.
    df = df.apply(lambda row: row.fillna(row.mean()), axis=1)
    
    # Proceed to clean (detrend and standardize) the imputed data with nilearn.
    cleaned_values = clean(df.values, detrend=True, standardize=True)
    cleaned_df = pd.DataFrame(cleaned_values, columns=column_names, index=df.index)
    return cleaned_df

def main():
    setup_logging()
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    atlas_tsv = Path(args.atlas)
    full_roi_list = load_full_roi_list(atlas_tsv)
    logging.info("Loaded full ROI list with %d ROIs.", len(full_roi_list))

    ts_files = list(input_dir.rglob("*_timeseries.tsv"))
    if not ts_files:
        logging.error("No timeseries files found in input directory.")
        sys.exit(1)

    roi_nan_stats_path = output_dir / "roi_nan_stats.csv"
    if roi_nan_stats_path.exists():
        logging.info("Found cached ROI NaN stats at %s. Loading...", roi_nan_stats_path)
        roi_nan_df = pd.read_csv(roi_nan_stats_path)
        global_nan_rate = pd.Series(data=roi_nan_df["nan_rate"].values,
                                    index=roi_nan_df["roi_index"].astype(int))
    else:
        logging.info("PASS 1: Calculating global missing rate across %d files...", len(ts_files))
        global_nan_rate = compute_global_nan_mask(ts_files, args.nroi)
        roi_indices = [int(i) for i in global_nan_rate.index]
        roi_names = []
        for i in roi_indices:
            if i < len(full_roi_list):
                roi_names.append(full_roi_list[i])
            else:
                logging.error(f"ROI index {i} is out of bounds for ROI list of length {len(full_roi_list)}")
                roi_names.append("UNKNOWN")
        roi_nan_df = pd.DataFrame({
            "roi_index": roi_indices,
            "roi_name": roi_names,
            "nan_rate": global_nan_rate.values
        })
        roi_nan_df.to_csv(roi_nan_stats_path, index=False)

    rois_to_drop = global_nan_rate[global_nan_rate > args.nan_threshold].index.tolist()
    logging.info(f"Dropping {len(rois_to_drop)} ROIs with >{int(args.nan_threshold * 100)}%% missing data: {rois_to_drop}")
    dropped_roi_df = pd.DataFrame({
        "roi_index": [int(i) for i in rois_to_drop],
        "roi_name": [full_roi_list[int(i)] for i in rois_to_drop]
    })
    dropped_roi_df.to_csv(output_dir / "rois_dropped.csv", index=False)

    surviving_roi_indices = sorted(set(range(args.nroi)) - set(map(int, rois_to_drop)))
    surviving_roi_names = [full_roi_list[i] for i in surviving_roi_indices]
    surviving_roi_strs = [str(i) for i in surviving_roi_indices]

    final_roi_df = pd.DataFrame({
        "roi_index": surviving_roi_indices,
        "roi_name": surviving_roi_names
    })
    final_roi_df.to_csv(output_dir / "final_roi_labels.csv", index=False)
    logging.info(f"Saved surviving ROI labels to {output_dir / 'final_roi_labels.csv'}")

    # Initialize a dictionary to track global mean imputations per ROI.
    global_impute_counts = {roi: 0 for roi in surviving_roi_strs}

    logging.info("PASS 2: Cleaning and saving each subject's timeseries...")
    subject_shapes = []
    excluded_subjects = []
    subject_qc_records = []

    for file_path in ts_files:
        fname = file_path.name
        if "correlation_matrix" in fname or "covariance_matrix" in fname:
            continue

        subject, feature, atlas = parse_filename(fname)
        if not (subject and feature and atlas):
            logging.warning(f"Could not parse {fname}; skipping.")
            continue
        if feature not in FEATURE_RENAME_MAP:
            logging.warning(f"Feature '{feature}' not in rename map; skipping.")
            continue
        desc_value = FEATURE_RENAME_MAP[feature]

        try:
            df = load_timeseries(file_path)
            df = check_orientation_and_add_header(df, args.nroi)

            # Ensure that all expected ROI columns are present.
            for roi in map(str, range(args.nroi)):
                if roi not in df.columns:
                    df[roi] = np.nan

            df = df.drop(columns=[str(i) for i in rois_to_drop], errors="ignore")
            for roi in surviving_roi_strs:
                if roi not in df.columns:
                    df[roi] = np.nan

            df = df[surviving_roi_strs]

            valid_fractions = df.notna().mean(axis=0)
            n_valid = (valid_fractions >= 0.5).sum()
            coverage_ratio = n_valid / len(surviving_roi_strs)
            n_fully_nan = (df.isna().sum() == df.shape[0]).sum()
            n_nans_total = df.isna().sum().sum()
            # excluded = coverage_ratio < 0.9
            excluded = coverage_ratio < 0.8

            subject_qc_records.append({
                "subject_id": subject,
                "n_valid_rois": int(n_valid),
                "total_rois": len(surviving_roi_strs),
                "roi_coverage_ratio": round(coverage_ratio, 4),
                "n_fully_nan_rois": int(n_fully_nan),
                "total_n_nans": int(n_nans_total),
                "excluded_due_to_coverage": excluded
            })

            if excluded:
                logging.warning(f"{subject}: Only {coverage_ratio*100:.1f}% of ROIs have >=50% coverage; excluding subject.")
                excluded_subjects.append((subject, coverage_ratio))
                continue

            # --- NEW: Track which ROIs are fully missing for the subject ---
            subject_impute_mask = df.isna().all()  # Boolean Series: True for ROIs fully missing
            for col, impute_needed in subject_impute_mask.items():
                if impute_needed:
                    global_impute_counts[col] += 1

            # Now impute and clean the timeseries.
            df_clean = impute_and_clean(df, surviving_roi_strs)
            subject_shapes.append(df_clean.shape)

            n_nans_post = df_clean.isna().sum().sum()
            if n_nans_post > 0:
                logging.warning(f"{subject}: {n_nans_post} NaNs remain after cleaning!")

            final_name = (
                f"{subject}_task-{args.task}_space-{args.space}_"
                f"atlas-{atlas}_nroi-{df_clean.shape[1]}_"
                f"desc-{desc_value}_timeseries.tsv"
            )
            subject_folder = output_dir / subject
            subject_folder.mkdir(parents=True, exist_ok=True)
            out_path = subject_folder / final_name

            df_clean.to_csv(out_path, sep="\t", index=False)
            logging.info(f"Saved cleaned file: {out_path}")

        except Exception as e:
            logging.error(f"Failed to process {fname}: {e}")

    if excluded_subjects:
        pd.DataFrame(excluded_subjects, columns=["subject", "roi_coverage"]).to_csv(output_dir / "excluded_subjects.csv", index=False)

    pd.DataFrame(subject_qc_records).to_csv(output_dir / "subject_qc_report.csv", index=False)

    all_shapes = set(subject_shapes)
    if len(all_shapes) == 1:
        logging.info(f"All subjects have the same timeseries shape: {all_shapes.pop()}")
    else:
        logging.warning(f"Subjects have inconsistent shapes: {all_shapes}")

    # Produce CSV of global imputation counts by ROI.
    impute_counts_df = pd.DataFrame({
        "roi_index": [int(roi) for roi in surviving_roi_strs],
        "roi_name": surviving_roi_names,
        "global_impute_count": [global_impute_counts[roi] for roi in surviving_roi_strs]
    })
    impute_counts_csv_path = output_dir / "roi_global_impute_counts.csv"
    impute_counts_df.to_csv(impute_counts_csv_path, index=False)
    logging.info(f"Saved global imputation counts to {impute_counts_csv_path}")

    logging.info("All files processed.")

if __name__ == "__main__":
    main()