import argparse
import logging
import pandas as pd
import numpy as np
import os  # Added for os.listdir
from pathlib import Path
import matplotlib.pyplot as plt

from fmriprep_denoise.dataset.fmriprep import get_prepro_strategy
from fmriprep_denoise.features.derivatives_test import (
    get_qc_criteria,
    load_full_roi_list,  # added for ROI list return
)

# Set up logging for debugging purposes.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Generate missing data statistics and plots across pipelines.",
    )
    parser.add_argument(
        "input_path",
        action="store",
        type=str,
        help="Input path to the timeseries collection.",
    )
    parser.add_argument(
        "output_path",
        action="store",
        type=str,
        help="Output directory where summary tables and plots will be saved.",
    )
    parser.add_argument(
        "--atlas",
        action="store",
        type=str,
        help="Atlas name (e.g., Schaefer2018)",
    )
    parser.add_argument(
        "--dimension",
        action="store",
        help="Number of ROIs. See meta data of each atlas for valid inputs.",
    )
    parser.add_argument(
        "--qc",
        action="store",
        default=None,
        help="Automatic motion QC thresholds.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset name (default: extracted from input_path)",
    )
    parser.add_argument(
        "--fmriprep_ver",
        type=str,
        default=None,
        help="Override fMRIPrep version (default: extracted from input_path)",
    )
    return parser.parse_args()


def load_subject_data(atlas, extracted_path, participant_ids, pipeline, full_roi_list):
    """
    Load each subject's time series DataFrame into a dictionary.
    The DataFrame is loaded without a header and then reindexed by assigning the full ROI list.
    """
    subject_data = {}
    for subject in participant_ids:
        subject_path = extracted_path / subject / "func" / "task-pixar"
        pattern = f"{subject}_task-pixar_feature-{pipeline}_atlas-{atlas}_timeseries.tsv"
        logging.debug("Checking %s with pattern: %s", subject_path, pattern)

        file_paths = list(subject_path.glob(pattern))
        if len(file_paths) != 1:
            logging.warning("Skipping subject %s for pipeline %s: expected one file, got %d", subject, pipeline, len(file_paths))
            continue
        file_path = file_paths[0]
        if file_path.stat().st_size <= 1:
            logging.warning("Skipping subject %s: file size too small", subject)
            continue
        try:
            df = pd.read_csv(file_path, sep="\t", header=None)
            if len(df.columns) != len(full_roi_list):
                logging.warning("Skipping subject %s: column count (%d) does not match expected (%d)",
                                subject, len(df.columns), len(full_roi_list))
                continue
            df.columns = full_roi_list
            subject_data[subject] = df
            logging.info("Loaded data for subject %s with shape %s", subject, df.shape)
        except Exception as e:
            logging.error("Error loading file for subject %s: %s", subject, e)
    return subject_data


def analyze_missing_data(subject_data):
    """
    Analyze missing data for a given pipeline.
    Returns:
      - subject_summary: dictionary with overall subject summary.
      - roi_summary: dictionary mapping ROI -> % missing.
      - subject_missing_distribution: dictionary mapping subject -> % missing.
      - rois_without_missing: list of ROIs with 0% missing.
    """
    total_subjects = len(subject_data)
    subjects_with_na = []
    subjects_without_na = []
    subject_missing_distribution = {}  # subject -> % missing
    for subj, df in subject_data.items():
        total_entries = df.shape[0] * df.shape[1]
        missing_entries = df.isna().sum().sum()
        pct_missing = (missing_entries / total_entries) * 100 if total_entries > 0 else np.nan
        subject_missing_distribution[subj] = pct_missing

        if df.isna().any().any():
            subjects_with_na.append(subj)
        else:
            subjects_without_na.append(subj)
    subject_summary = {
         "subjects_with_na": len(subjects_with_na),
         "subjects_without_na": len(subjects_without_na),
         "total_subjects": total_subjects,
         "percent_with_na": (len(subjects_with_na) / total_subjects) * 100 if total_subjects > 0 else np.nan
    }

    # ROI summary: for each ROI, percentage of subjects with at least one missing value.
    all_rois = next(iter(subject_data.values())).columns.tolist()
    roi_missing_counts = {roi: 0 for roi in all_rois}
    for subj, df in subject_data.items():
        for roi in all_rois:
            if df[roi].isna().any():
                roi_missing_counts[roi] += 1

    roi_summary = {roi: (count / total_subjects) * 100 for roi, count in roi_missing_counts.items()}
    rois_without_missing = [roi for roi, pct in roi_summary.items() if pct == 0]
    return subject_summary, roi_summary, subject_missing_distribution, rois_without_missing


def main():
    print("Logging main function start message", flush=True)
    args = parse_args()
    logging.debug("Parsed arguments: %s", vars(args))
    print(vars(args))

    input_path = Path(args.input_path)
    atlas = args.atlas
    dimension = args.dimension

    logging.debug("Input path: %s", input_path)
    print("Input path:", input_path)

    dataset = args.dataset if args.dataset else input_path.parents[0].name
    fmriprep_ver = args.fmriprep_ver if args.fmriprep_ver else input_path.name
    logging.debug("Dataset: %s, fMRIPrep version: %s", dataset, fmriprep_ver)
    print("Dataset:", dataset)
    print("fMRIPrep version:", fmriprep_ver)

    path_root = Path(args.output_path).absolute()
    output_path = path_root / dataset / fmriprep_ver
    output_path.mkdir(parents=True, exist_ok=True)
    logging.debug("Output path created: %s", output_path)
    print("Output path:", output_path)

    atlas_tsv = "/home/seann/projects/def-cmoreau/All_user_common_folder/atlas/atlas_enigma/atlas-Schaefer2018Combined_dseg.tsv"
    full_roi_list = load_full_roi_list(atlas_tsv)
    logging.debug("Loaded full ROI list with %d ROIs.", len(full_roi_list))

    strategy_names = get_prepro_strategy(None)
    logging.debug("Retrieved pre-processing strategies: %s", list(strategy_names.keys()))

    participants_file = Path("/home/seann/projects/def-cmoreau/All_user_common_folder/datasets/ds000228/participants_seann.tsv")
    participants_df = pd.read_csv(participants_file, sep="\t")
    participant_ids = participants_df["participant_id"].tolist()

    # Containers to collect statistics for each pipeline.
    subject_summary_dict = {}
    roi_summary_dict = {}
    subject_missing_distribution_dict = {}  # Now dictionary: pipeline -> {subject: % missing}
    rois_without_missing_dict = {}

    for pipeline in strategy_names.keys():
        logging.info("Evaluating missing data for pipeline: %s", pipeline)
        subject_data = load_subject_data(atlas, input_path, participant_ids, pipeline, full_roi_list)
        if not subject_data:
            logging.warning("No data loaded for pipeline: %s", pipeline)
            continue
        subj_summary, roi_summary, subj_missing_dist, rois_without_missing = analyze_missing_data(subject_data)
        subject_summary_dict[pipeline] = subj_summary
        roi_summary_dict[pipeline] = roi_summary
        subject_missing_distribution_dict[pipeline] = subj_missing_dist  # dictionary
        rois_without_missing_dict[pipeline] = rois_without_missing

        print(f"\n===== Pipeline: {pipeline} =====")
        print("Subject-level summary:")
        print(subj_summary)
        print("ROIs with NO missing data:")
        print(rois_without_missing)

    # Save subject-level and ROI-level summaries.
    df_subject = pd.DataFrame.from_dict(subject_summary_dict, orient="index")
    subject_summary_file = output_path / "subject_missing_summary.tsv"
    df_subject.to_csv(subject_summary_file, sep="\t")
    logging.info("Subject missing summary saved to %s", subject_summary_file)
    print("\nSubject-level missing summary:")
    print(df_subject)

    df_roi = pd.DataFrame(roi_summary_dict)
    roi_summary_file = output_path / "roi_missing_summary.tsv"
    df_roi.to_csv(roi_summary_file, sep="\t")
    logging.info("ROI missing summary saved to %s", roi_summary_file)
    print("\nROI-level missing summary:")
    print(df_roi)

    # Save Pipeline Summary Table (Subject-level summary + ROIs without missing).
    pipeline_summary = []
    for pipeline in subject_summary_dict.keys():
        summary = subject_summary_dict[pipeline].copy()
        summary["rois_without_missing"] = ", ".join(rois_without_missing_dict[pipeline])
        summary["pipeline"] = pipeline
        pipeline_summary.append(summary)
    df_pipeline_summary = pd.DataFrame(pipeline_summary).set_index("pipeline")
    pipeline_summary_file = output_path / "pipeline_summary.tsv"
    df_pipeline_summary.to_csv(pipeline_summary_file, sep="\t")
    logging.info("Pipeline summary saved to %s", pipeline_summary_file)
    print("\nPipeline summary:")
    print(df_pipeline_summary)

    # ---- Create ROI Intact Table Across Pipelines ----
    df_roi_intact = pd.DataFrame(index=full_roi_list)
    for pipeline, roi_dict in roi_summary_dict.items():
        intact = {roi: 1 if roi_dict.get(roi, 100) == 0 else 0 for roi in full_roi_list}
        df_roi_intact[pipeline] = pd.Series(intact)
    df_roi_intact_all = df_roi_intact[df_roi_intact.sum(axis=1) == len(df_roi_intact.columns)]
    roi_intact_file = output_path / "rois_intact_across_pipelines.tsv"
    df_roi_intact_all.to_csv(roi_intact_file, sep="\t")
    logging.info("ROI intact table (across all pipelines) saved to %s", roi_intact_file)
    print("\nROI intact table (all pipelines):")
    print(df_roi_intact_all)

    # ---- Plot Distribution Histograms ----
    # Define bins (20 equal-width bins from 0 to 100)
    bins = np.linspace(0, 100, 21)

    # Create a consistent color mapping.
    pipelines_sorted = sorted(subject_missing_distribution_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(pipelines_sorted)))
    color_dict = {pipeline: colors[i] for i, pipeline in enumerate(pipelines_sorted)}

    # (A) Individual Subject Integrity Histograms (Intact % = 100 - % missing)
    for pipeline in pipelines_sorted:
        intact = [100 - pct for pct in subject_missing_distribution_dict[pipeline].values()]
        plt.figure(figsize=(8, 6))
        n, b, patches = plt.hist(intact, bins=bins, alpha=0.8, edgecolor="black", color=color_dict[pipeline])
        for count, patch in zip(n, patches):
            if count > 0:
                plt.text(patch.get_x() + patch.get_width()/2, patch.get_height(), int(count),
                         ha='center', va='bottom', fontsize=8)
        plt.xlabel("% of Data Intact per Subject")
        plt.ylabel("Count of Subjects")
        plt.title(f"Subject Data Integrity Distribution: {pipeline}")
        plt.xticks(bins)
        plt.tight_layout()
        indiv_subject_file = output_path / f"subject_data_integrity_histogram_{pipeline}.png"
        plt.savefig(indiv_subject_file)
        logging.info("Subject integrity histogram for pipeline %s saved to %s", pipeline, indiv_subject_file)
        plt.close()

    # (B) Combined Subject Integrity Grouped Bar Chart
    num_bins = len(bins) - 1
    bar_width = 0.8 / len(pipelines_sorted)
    indices = np.arange(num_bins)
    subject_counts = {}
    for pipeline in pipelines_sorted:
        intact = [100 - pct for pct in subject_missing_distribution_dict[pipeline].values()]
        counts, _ = np.histogram(intact, bins=bins)
        subject_counts[pipeline] = counts

    plt.figure(figsize=(12, 7))
    for i, pipeline in enumerate(pipelines_sorted):
        counts = subject_counts[pipeline]
        plt.bar(indices + i * bar_width, counts, bar_width,
                label=pipeline, color=color_dict[pipeline], edgecolor="black")
        for j, count in enumerate(counts):
            plt.text(indices[j] + i * bar_width + bar_width/2, count, str(count),
                     ha='center', va='bottom', fontsize=8)
    plt.xlabel("% of Data Intact per Subject (Bin Range)")
    plt.ylabel("Count of Subjects")
    plt.title("Grouped Subject Data Integrity Distribution Across Pipelines")
    plt.xticks(indices + bar_width * (len(pipelines_sorted)-1)/2,
               [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(num_bins)])
    plt.legend(title="Pipeline")
    plt.tight_layout()
    combined_subject_file = output_path / "combined_subject_data_integrity_grouped.png"
    plt.savefig(combined_subject_file)
    logging.info("Combined grouped subject integrity plot saved to %s", combined_subject_file)
    plt.show()

    # (C) Individual Subject Missing Histograms (% missing per subject)
    for pipeline in pipelines_sorted:
        missing = list(subject_missing_distribution_dict[pipeline].values())
        plt.figure(figsize=(8, 6))
        n, b, patches = plt.hist(missing, bins=bins, alpha=0.8, edgecolor="black", color=color_dict[pipeline])
        for count, patch in zip(n, patches):
            if count > 0:
                plt.text(patch.get_x() + patch.get_width()/2, patch.get_height(), int(count),
                         ha='center', va='bottom', fontsize=8)
        plt.xlabel("% of Data Missing per Subject")
        plt.ylabel("Count of Subjects")
        plt.title(f"Subject Missing Data Distribution: {pipeline}")
        plt.xticks(bins)
        plt.tight_layout()
        indiv_missing_file = output_path / f"subject_missing_data_histogram_{pipeline}.png"
        plt.savefig(indiv_missing_file)
        logging.info("Subject missing histogram for pipeline %s saved to %s", pipeline, indiv_missing_file)
        plt.close()

    # (D) Combined Subject Missing Grouped Bar Chart
    subject_missing_counts = {}
    for pipeline in pipelines_sorted:
        missing = list(subject_missing_distribution_dict[pipeline].values())
        counts, _ = np.histogram(missing, bins=bins)
        subject_missing_counts[pipeline] = counts

    plt.figure(figsize=(12, 7))
    for i, pipeline in enumerate(pipelines_sorted):
        counts = subject_missing_counts[pipeline]
        plt.bar(indices + i * bar_width, counts, bar_width,
                label=pipeline, color=color_dict[pipeline], edgecolor="black")
        for j, count in enumerate(counts):
            plt.text(indices[j] + i * bar_width + bar_width/2, count, str(count),
                     ha='center', va='bottom', fontsize=8)
    plt.xlabel("% of Data Missing per Subject (Bin Range)")
    plt.ylabel("Count of Subjects")
    plt.title("Grouped Subject Missing Data Distribution Across Pipelines")
    plt.xticks(indices + bar_width * (len(pipelines_sorted)-1)/2,
               [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(num_bins)])
    plt.legend(title="Pipeline")
    plt.tight_layout()
    combined_missing_file = output_path / "combined_subject_missing_data_grouped.png"
    plt.savefig(combined_missing_file)
    logging.info("Combined grouped subject missing plot saved to %s", combined_missing_file)
    plt.show()

    # (E) Individual ROI Integrity Histograms (for each pipeline)
    for pipeline in sorted(roi_summary_dict.keys()):
        intact_roi = [100 - pct for pct in roi_summary_dict[pipeline].values()]
        plt.figure(figsize=(8, 6))
        n, b, patches = plt.hist(intact_roi, bins=bins, alpha=0.8, edgecolor="black", color=color_dict[pipeline])
        for count, patch in zip(n, patches):
            if count > 0:
                plt.text(patch.get_x() + patch.get_width()/2, patch.get_height(), int(count),
                         ha='center', va='bottom', fontsize=8)
        plt.xlabel("% of Subjects with Data per ROI")
        plt.ylabel("Count of ROIs")
        plt.title(f"ROI Data Integrity Distribution: {pipeline}")
        plt.xticks(bins)
        plt.tight_layout()
        indiv_roi_file = output_path / f"roi_data_integrity_histogram_{pipeline}.png"
        plt.savefig(indiv_roi_file)
        logging.info("ROI integrity histogram for pipeline %s saved to %s", pipeline, indiv_roi_file)
        plt.close()

    # (F) Combined ROI Integrity Grouped Bar Chart
    pipelines_roi = sorted(df_roi.columns)
    num_bins = len(bins) - 1
    bar_width = 0.8 / len(pipelines_roi)
    indices = np.arange(num_bins)
    roi_counts = {}
    for pipeline in pipelines_roi:
        intact_roi = 100 - df_roi[pipeline]
        counts, _ = np.histogram(intact_roi, bins=bins)
        roi_counts[pipeline] = counts

    plt.figure(figsize=(12, 7))
    for i, pipeline in enumerate(pipelines_roi):
        counts = roi_counts[pipeline]
        plt.bar(indices + i * bar_width, counts, bar_width,
                label=pipeline, color=color_dict[pipeline], edgecolor="black")
        for j, count in enumerate(counts):
            plt.text(indices[j] + i * bar_width + bar_width/2, count, str(count),
                     ha='center', va='bottom', fontsize=8)
    plt.xlabel("% of Subjects with Data per ROI (Bin Range)")
    plt.ylabel("Count of ROIs")
    plt.title("Grouped ROI Data Integrity Distribution Across Pipelines")
    plt.xticks(indices + bar_width * (len(pipelines_roi)-1)/2,
               [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(num_bins)])
    plt.legend(title="Pipeline")
    plt.tight_layout()
    combined_roi_file = output_path / "combined_roi_data_integrity_grouped.png"
    plt.savefig(combined_roi_file)
    logging.info("Combined grouped ROI integrity plot saved to %s", combined_roi_file)
    plt.show()


if __name__ == "__main__":
    print("Logging main function called", flush=True)
    main()