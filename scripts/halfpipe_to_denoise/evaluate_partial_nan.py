#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(description="Count fully and partially NaN ROI columns in fMRI timeseries.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory with *_timeseries.tsv files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save output CSVs.")
    return parser.parse_args()

def count_nan_columns(ts_file):
    subject_id = ts_file.name.split("_")[0]
    try:
        df = pd.read_csv(ts_file, sep="\t", header=None)
        # Assume [time x ROI]; transpose if needed
        if df.shape[0] < df.shape[1]:
            df = df.T

        fully_nan = (df.isna().sum() == df.shape[0]).sum()
        partially_nan = ((df.isna().sum() > 0) & (df.isna().sum() < df.shape[0])).sum()

        return {
            "subject_id": subject_id,
            "fully_nan_rois": fully_nan,
            "partially_nan_rois": partially_nan
        }

    except Exception as e:
        logging.warning(f"Error reading {ts_file.name}: {e}")
        return {
            "subject_id": subject_id,
            "fully_nan_rois": -1,
            "partially_nan_rois": -1
        }

def main():
    setup_logging()
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts_files = list(input_dir.rglob("*_timeseries.tsv"))
    if not ts_files:
        logging.error("No timeseries files found.")
        return

    logging.info(f"Found {len(ts_files)} timeseries files.")

    records = [count_nan_columns(f) for f in ts_files]

    df = pd.DataFrame(records)
    df[["subject_id", "fully_nan_rois"]].to_csv(output_dir / "fully_nan_rois_per_subject.csv", index=False)
    df[["subject_id", "partially_nan_rois"]].to_csv(output_dir / "partially_nan_rois_per_subject.csv", index=False)

    logging.info("Done. CSVs saved to output directory.")

if __name__ == "__main__":
    main()