#!/usr/bin/env python3
"""
Transition Halfpipe timeseries files to match the denoising pipeline format.
This script:
  1. Recursively searches the input directory (e.g., the root of halfpipe derivatives)
     for timeseries files ending in "_timeseries.tsv".
  2. For each file, parses the filename to extract subject, feature, and atlas names.
  3. Maps the halfpipe feature name to the denoising pipeline format using FEATURE_RENAME_MAP.
  4. Uses the halfpipe atlas name directly.
  5. Normalizes (detrends & standardizes) the timeseries using nilearn.signal.clean.
  6. Creates a subject-specific folder in the output directory and writes the final file
     using a BIDS-like pattern, for example:
     sub-pixar001_task-pixar_space-MNI152NLin2009cAsym_atlas-Schaefer2018Combined_nroi-434_desc-simple_timeseries.tsv
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import nilearn
from nilearn.signal import clean

# Mapping Halfpipe feature names to standardized denoising strategy labels
FEATURE_RENAME_MAP = {
    "corrMatrixCompCor": "compcor",
    "corrMatrixICA": "aroma",
    "corrMatrixMotion": "simple",
    "corrMatrixMotionGSR": "simple+gsr",
    "corrMatrixScrubGSR": "scrubbing.5+gsr",
    "corrMatrixScrub": "scrubbing.5",
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Transition Halfpipe timeseries files to match denoising pipeline format."
    )
    parser.add_argument("--input_dir", required=True, type=str,
                        help="Root directory containing subject folders of halfpipe outputs (e.g., .../derivatives/halfpipe).")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Directory where reformatted timeseries files will be saved.")
    parser.add_argument("--task", required=True, type=str,
                        help="Task name (e.g., pixar or rest).")
    parser.add_argument("--space", default="MNI152NLin2009cAsym", type=str,
                        help="Template space (default: MNI152NLin2009cAsym).")
    parser.add_argument("--nroi", required=True, type=int,
                        help="Number of ROIs to expect (e.g., 434 for Schaefer2018Combined).")
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

def load_timeseries(file_path):
    """Load a halfpipe timeseries TSV with no header."""
    return pd.read_csv(file_path, sep="\t", header=None, na_values=["nan"])

def parse_filename(filename):
    """
    Parse a halfpipe filename.

    Example:
      sub-pixar001_task-pixar_feature-corrMatrixCompCor_atlas-Schaefer2018Combined_timeseries.tsv

    Returns:
      subject, feature (e.g., corrMatrixCompCor), atlas
    """
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
    """Ensure the DataFrame is [time x ROI] and add header labels."""
    rows, cols = df.shape
    logging.debug(f"Initial shape: {rows}x{cols} (expected {expected_nroi} columns)")
    if cols != expected_nroi and rows == expected_nroi:
        logging.info("Data appears transposed; transposing DataFrame.")
        df = df.T
    elif cols != expected_nroi:
        logging.warning("Data columns (%d) do not match expected nroi (%d).", cols, expected_nroi)
    df.columns = [str(i) for i in range(df.shape[1])]
    return df

def clean_timeseries(df):
    """Detrend and standardize the timeseries using nilearn.signal.clean."""
    logging.info("Cleaning timeseries: detrend=True, standardize=True")
    arr = df.values.astype(float)
    cleaned = clean(arr, detrend=True, standardize=True)
    return pd.DataFrame(cleaned, columns=df.columns)

def main():
    setup_logging()
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts_files = list(input_dir.rglob("*_timeseries.tsv"))
    if not ts_files:
        logging.error("No timeseries TSV files found in %s", input_dir)
        sys.exit(1)
    logging.info("Found %d timeseries files to process.", len(ts_files))

    for file_path in ts_files:
        fname = file_path.name
        logging.info("Processing file: %s", fname)

        if "task-" not in str(file_path.parent):
            logging.info("Skipping file not in a task folder: %s", fname)
            continue

        if "correlation_matrix" in fname or "covariance_matrix" in fname:
            logging.info("Skipping matrix file: %s", fname)
            continue

        subject, feature, atlas = parse_filename(fname)
        if not (subject and feature and atlas):
            logging.warning("Could not parse subject/feature/atlas from %s; skipping.", fname)
            continue

        # Apply renaming map
        if feature not in FEATURE_RENAME_MAP:
            logging.warning("Feature '%s' not found in renaming map; skipping.", feature)
            continue
        desc_value = FEATURE_RENAME_MAP[feature]

        try:
            df = load_timeseries(file_path)
        except Exception as e:
            logging.error("Failed reading %s: %s", fname, e)
            continue

        df = check_orientation_and_add_header(df, args.nroi)
        print("Sample values:\n", df.head())
        print("Data types:\n", df.dtypes)
        print("Any real NaNs?:", df.isna().any().any())
        df_clean = clean_timeseries(df)

        final_name = (
            f"{subject}_task-{args.task}_space-{args.space}_"
            f"atlas-{atlas}_nroi-{args.nroi}_"
            f"desc-{desc_value}_timeseries.tsv"
        )

        subject_folder = output_dir / subject
        subject_folder.mkdir(parents=True, exist_ok=True)
        out_path = subject_folder / final_name

        try:
            df_clean.to_csv(out_path, sep="\t", index=False)
            logging.info("Saved reformatted file to: %s", out_path)
        except Exception as e:
            logging.error("Failed to save %s: %s", final_name, e)

    logging.info("All files processed successfully.")

if __name__ == "__main__":
    main()