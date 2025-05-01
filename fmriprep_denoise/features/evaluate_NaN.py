#!/usr/bin/env python
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Import nilearn functions for plotting on a brain template.
from nilearn import datasets, plotting
import nibabel as nib
from scipy.ndimage import center_of_mass

# Import your existing data‐loading function.
from fmriprep_denoise.features.derivatives_test import load_full_roi_list

# Set up logging.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Visualize ROI missing data (NaN rates) on a brain template for halfpipe versions, "
                    "filtering subjects by FD and excluding specific subject IDs."
    )
    parser.add_argument("input_path", type=str,
                        help="Input path for halfpipe version A (timeseries collection).")
    parser.add_argument("output_path", type=str,
                        help="Output directory where plots and tables will be saved.")
    parser.add_argument("--atlas", type=str, required=True,
                        help="Atlas name (e.g., Schaefer2018).")
    parser.add_argument("--dimension", type=str, required=True,
                        help="Number of ROIs (as in atlas metadata).")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset name (default: extracted from input_path).")
    # Instead of a single fMRIPrep version, we now accept separate versions for A and B.
    parser.add_argument("--fmriprep_ver_a", type=str, default=None,
                        help="Override fMRIPrep version for halfpipe version A (default: extracted from input_path).")
    parser.add_argument("--version_b", type=str, default=None,
                        help="(Optional) Input path for halfpipe version B for comparison.")
    parser.add_argument("--fmriprep_ver_b", type=str, default=None,
                        help="Override fMRIPrep version for halfpipe version B (default: extracted from version_b path).")
    parser.add_argument("--pipeline", type=str, default="corrMatrixMotion",
                        help="Pipeline name (default: corrMatrixMotion).")
    # New argument: path to the atlas image (for computing centroids if coordinates are missing).
    parser.add_argument("--atlas_img", type=str, required=True,
                        help="Path to the atlas NIfTI image (used to compute ROI centroids if not provided in TSV).")
    parser.add_argument("--colormap", type=str, default="RdBu_r",
                        help="Colormap to use for brain plots (default: RdBu_r).")
    # New arguments for FD filtering.
    parser.add_argument("--confounds_root", type=str, required=True,
                        help="Root directory where confounds files are stored (e.g., path to fmriprep outputs).")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name to use in the confounds filename (e.g., 'task-pixar').")
    parser.add_argument("--fd_threshold", type=float, default=0.5,
                        help="Mean FD threshold; subjects with mean FD >= threshold are excluded (default: 0.5).")
    # New argument: subject exclusion file.
    parser.add_argument("--exclude_file", type=str,
                        default="/home/seann/projects/def-cmoreau/All_user_common_folder/datasets/ds000228/visual_qc_excude_list",
                        help="File containing subject IDs (one per line) to exclude from visualization.")
    # Optional: binary mode flag.
    parser.add_argument("--binary_mode", action="store_true",
                        help="If set, plot ROIs as binary (100 if any NaN, 0 if intact) instead of continuous percentages.")
    return parser.parse_args()


def filter_subjects_by_fd(participant_ids, confounds_root, task, fd_threshold):
    valid_subjects = []
    # Check if task already starts with "task-". If not, add it.
    if not task.startswith("task-"):
        task_str = f"task-{task}"
    else:
        task_str = task
    for subj in participant_ids:
        confounds_path = Path(confounds_root) / subj / "func" / f"{subj}_{task_str}_desc-confounds_timeseries.tsv"
        if not confounds_path.exists():
            logging.warning("Confounds file does not exist for subject %s at %s", subj, confounds_path)
            continue
        try:
            df_confounds = pd.read_csv(confounds_path, sep="\t")
            if "framewise_displacement" not in df_confounds.columns:
                logging.warning("Subject %s: 'framewise_displacement' column not found in %s", subj, confounds_path)
                continue
            mean_fd = np.nanmean(df_confounds["framewise_displacement"])
            logging.info("Subject %s mean FD: %.3f", subj, mean_fd)
            if mean_fd < fd_threshold:
                valid_subjects.append(subj)
            else:
                logging.info("Excluding subject %s due to high FD (%.3f)", subj, mean_fd)
        except Exception as e:
            logging.error("Error processing confounds for subject %s: %s", subj, e)
    return valid_subjects

def load_exclude_list(exclude_file):
    """
    Load a file with subject IDs to exclude.
    Returns a set of subject IDs.
    """
    try:
        with open(exclude_file, "r") as f:
            # Strip any whitespace and ignore empty lines.
            exclude_ids = {line.strip() for line in f if line.strip()}
        logging.info("Loaded %d subject IDs to exclude.", len(exclude_ids))
        return exclude_ids
    except Exception as e:
        logging.error("Error reading exclusion file %s: %s", exclude_file, e)
        return set()


def compute_roi_centroids(atlas_img_path, roi_labels):
    img = nib.load(str(atlas_img_path))
    data = img.get_fdata()
    affine = img.affine
    centroids = []
    for roi in roi_labels:
        try:
            roi_val = float(roi)
        except ValueError:
            continue
        mask = (data == roi_val)
        if np.sum(mask) == 0:
            continue
        com_voxel = center_of_mass(mask)
        com_world = nib.affines.apply_affine(affine, com_voxel)
        centroids.append({"roi": roi, "x": com_world[0], "y": com_world[1], "z": com_world[2]})
    return pd.DataFrame(centroids)


def load_subject_data(atlas, base_path, participant_ids, pipeline, full_roi_list):
    subject_data = {}
    for subject in participant_ids:
        subject_path = base_path / subject / "func" / "task-pixar"
        pattern = f"{subject}_task-pixar_feature-{pipeline}_atlas-{atlas}_timeseries.tsv"
        logging.debug("Checking %s with pattern: %s", subject_path, pattern)
        files = list(subject_path.glob(pattern))
        if len(files) != 1:
            logging.warning("Skipping subject %s: expected 1 file, got %d", subject, len(files))
            continue
        file_path = files[0]
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
            logging.error("Error loading subject %s: %s", subject, e)
    return subject_data


def build_missing_rate_matrix(subject_data, full_roi_list):
    subjects = list(subject_data.keys())
    mat = np.zeros((len(subjects), len(full_roi_list)))
    for i, subj in enumerate(subject_data.keys()):
        df = subject_data[subj]
        missing_fraction = df.isna().mean()
        mat[i, :] = missing_fraction.values
    return subjects, mat


def average_missing_rate(missing_matrix):
    return np.mean(missing_matrix, axis=0)


def plot_roi_missing_on_brain(roi_values, full_roi_list, atlas_coords_df, title, out_file, cmap="RdBu_r"):
    import matplotlib.pyplot as plt
    from nilearn import datasets, plotting

    print("Unique atlas ROI labels:", atlas_coords_df["roi"].unique()[:10])
    print("First 10 roi_values index (missing values):", roi_values.index[:10])
    
    merged = pd.merge(atlas_coords_df, roi_values.rename("missing_pct"), left_on="roi", right_index=True)
    if merged.empty:
        print("ERROR: Merged DataFrame is empty! No overlapping ROI labels found.")
        print("Atlas ROI labels:", atlas_coords_df["roi"].unique())
        print("roi_values index:", roi_values.index)
        return
    
    coords = merged[["x", "y", "z"]].values
    values = merged["missing_pct"].values

    vmin, vmax = 0.0, 1.0

    mni_template = datasets.load_mni152_template()
    display = plotting.plot_glass_brain(
        mni_template,
        colorbar=True,
        title=title,
        display_mode="ortho",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        black_bg=False
    )
    display.add_markers(coords, marker_color=values, marker_size=20, cmap=cmap)
    
    try:
        display.annotate_colorbar(label="Average missing rate", ticks=[0, 0.25, 0.5, 0.75, 1.0])
    except Exception as e:
        print("Could not annotate colorbar:", e)
    
    plt.savefig(out_file)
    plt.close()
    print(f"Brain plot saved to {out_file}")
    print(f"Data range => min: {values.min():.3f}, max: {values.max():.3f}")


def main():
    args = parse_args()
    logging.debug("Parsed arguments: %s", vars(args))
    print("Arguments:", vars(args))

    # Load participant phenotype.
    participants_file = Path("/home/seann/projects/def-cmoreau/All_user_common_folder/datasets/ds000228/participants_seann.tsv")
    pheno = pd.read_csv(participants_file, sep="\t")
    participant_ids = pheno["participant_id"].tolist()
    logging.info("Initial participant count: %d", len(participant_ids))
    print("Initial subject count:", len(participant_ids))

    # For "all subjects" processing (no FD filtering or exclusion)
    all_subjects = participant_ids.copy()
    logging.info("All subjects (unfiltered): %d", len(all_subjects))
    print("All subjects (unfiltered):", len(all_subjects))

    # Filter subjects based on FD.
    valid_subjects = filter_subjects_by_fd(participant_ids, args.confounds_root, args.task, args.fd_threshold)
    logging.info("Subjects after FD filtering: %d", len(valid_subjects))
    print("Subjects after FD filtering:", len(valid_subjects))

    # Load the exclusion list and remove those subjects.
    exclude_ids = load_exclude_list(args.exclude_file)
    final_subjects = [subj for subj in valid_subjects if subj not in exclude_ids]
    logging.info("Subjects after QC exclusion: %d", len(final_subjects))
    print("Subjects after QC exclusion:", len(final_subjects))

    # Set up Version A paths.
    version_A_path = Path(args.input_path)
    fmriprep_ver_a = args.fmriprep_ver_a if args.fmriprep_ver_a else version_A_path.name
    output_base = Path(args.output_path).absolute()
    dataset = args.dataset if args.dataset else version_A_path.parents[0].name
    colormap = args.colormap
    output_path_A = output_base / dataset / fmriprep_ver_a
    output_path_A.mkdir(parents=True, exist_ok=True)
    logging.info("Version A output path: %s", output_path_A)

    # Load full ROI list and atlas coordinates.
    atlas_tsv = f"/home/seann/projects/def-cmoreau/All_user_common_folder/atlas/atlas_enigma/atlas-{args.atlas}Combined_dseg.tsv"
    full_roi_list = load_full_roi_list(atlas_tsv)
    logging.info("Loaded full ROI list with %d ROIs.", len(full_roi_list))
    print("Number of ROIs:", len(full_roi_list))

    atlas_labels_df = pd.read_csv(atlas_tsv, sep="\t", header=None, names=["numeric", "roi_full"])
    atlas_labels_df["numeric"] = atlas_labels_df["numeric"].astype(str)
    atlas_labels_df["roi_full"] = atlas_labels_df["roi_full"].astype(str)

    roi_numeric_labels = atlas_labels_df["numeric"].tolist()
    logging.info("Computing centroids from atlas image using numeric ROI labels.")
    atlas_coords_df = compute_roi_centroids(args.atlas_img, roi_numeric_labels)
    atlas_coords_df["roi"] = atlas_coords_df["roi"].astype(str)
    atlas_coords_df = atlas_coords_df.merge(atlas_labels_df, left_on="roi", right_on="numeric", how="left")
    atlas_coords_df["roi"] = atlas_coords_df["roi_full"]
    atlas_coords_df.drop(columns=["numeric", "roi_full"], inplace=True)

    # Define a small epsilon to avoid division by zero.
    epsilon = 1e-6

    # --- Process Version A for ALL SUBJECTS (unfiltered) ---
    logging.info("Processing Version A (all subjects).")
    subject_data_A_all = load_subject_data(args.atlas, version_A_path, all_subjects, args.pipeline, full_roi_list)
    print("Version A (all subjects) loaded for", len(subject_data_A_all), "subjects.")
    if subject_data_A_all:
        subjects_A_all, missing_matrix_A_all = build_missing_rate_matrix(subject_data_A_all, full_roi_list)
        print("Number of Version A (all subjects) used:", len(subjects_A_all))
        avg_missing_A_all = average_missing_rate(missing_matrix_A_all)
        df_avg_A_all = pd.DataFrame({"roi": full_roi_list, "avg_missing_pct": avg_missing_A_all})
        df_avg_A_all.to_csv(output_path_A / "version_A_all_subjects_avg_missing_per_roi.tsv", sep="\t", index=False)
        out_file_A_all = output_path_A / "version_A_all_subjects_roi_missing_on_brain.png"
        roi_missing_series_A_all = pd.Series(avg_missing_A_all, index=full_roi_list)
        plot_roi_missing_on_brain(roi_missing_series_A_all, full_roi_list, atlas_coords_df,
                                  "Version A (All Subjects): Average ROI Missing Data", out_file_A_all, colormap)
    else:
        logging.warning("No subject data loaded for Version A (all subjects).")

    # --- Process Version A for FILTERED SUBJECTS ---
    logging.info("Processing Version A (filtered subjects).")
    subject_data_A = load_subject_data(args.atlas, version_A_path, final_subjects, args.pipeline, full_roi_list)
    print("Version A (filtered) loaded for", len(subject_data_A), "subjects.")
    if not subject_data_A:
        raise ValueError("No subject data loaded for Version A (filtered subjects).")
    subjects_A, missing_matrix_A = build_missing_rate_matrix(subject_data_A, full_roi_list)
    print("Number of Version A (filtered subjects) used:", len(subjects_A))
    avg_missing_A = average_missing_rate(missing_matrix_A)
    df_avg_A = pd.DataFrame({"roi": full_roi_list, "avg_missing_pct": avg_missing_A})
    df_avg_A.to_csv(output_path_A / "version_A_filtered_avg_missing_per_roi.tsv", sep="\t", index=False)
    out_file_A = output_path_A / "version_A_filtered_roi_missing_on_brain.png"
    roi_missing_series_A = pd.Series(avg_missing_A, index=full_roi_list)
    plot_roi_missing_on_brain(roi_missing_series_A, full_roi_list, atlas_coords_df,
                              "Version A (Filtered): Average ROI Missing Data", out_file_A, colormap)

    # --- Create a Relative Difference Plot within Version A (All vs. Filtered) ---
    # Compute relative difference (%) for each ROI.
    rel_diff_A = ((avg_missing_A - avg_missing_A_all) / (avg_missing_A_all + epsilon)) * 100
    df_rel_diff_A = pd.DataFrame({
        "roi": full_roi_list,
        "rel_diff_missing_pct": rel_diff_A,
        "all_subjects": avg_missing_A_all,
        "filtered": avg_missing_A
    })
    diff_file_A = output_path_A / "rel_diff_all_vs_filtered_avg_missing_per_roi.tsv"
    df_rel_diff_A.to_csv(diff_file_A, sep="\t", index=False)
    logging.info("Relative difference between unfiltered and filtered missing rates (Version A) saved to %s", diff_file_A)
    # Use a diverging colormap (here we use the common colormap for simplicity).
    out_file_diff_A = output_path_A / "rel_diff_all_vs_filtered_roi_missing_on_brain.png"
    roi_rel_diff_series_A = pd.Series(rel_diff_A, index=full_roi_list)
    plot_roi_missing_on_brain(roi_rel_diff_series_A, full_roi_list, atlas_coords_df,
                              "Relative Difference (All vs Filtered, Version A) [%]", out_file_diff_A, cmap=colormap)

    # --- Process Version B (if provided) ---
    if args.version_b:
        version_B_path = Path(args.version_b)
        fmriprep_ver_b = args.fmriprep_ver_b if args.fmriprep_ver_b else version_B_path.name
        output_path_B = output_base / dataset / fmriprep_ver_b
        output_path_B.mkdir(parents=True, exist_ok=True)
        logging.info("Version B output path: %s", output_path_B)
        
        # --- Version B: ALL SUBJECTS ---
        logging.info("Processing Version B (all subjects) from %s", version_B_path)
        subject_data_B_all = load_subject_data(args.atlas, version_B_path, all_subjects, args.pipeline, full_roi_list)
        print("Version B (all subjects) loaded for", len(subject_data_B_all), "subjects.")
        if subject_data_B_all:
            subjects_B_all, missing_matrix_B_all = build_missing_rate_matrix(subject_data_B_all, full_roi_list)
            print("Number of Version B (all subjects) used:", len(subjects_B_all))
            avg_missing_B_all = average_missing_rate(missing_matrix_B_all)
            df_avg_B_all = pd.DataFrame({"roi": full_roi_list, "avg_missing_pct": avg_missing_B_all})
            df_avg_B_all.to_csv(output_path_B / "version_B_all_subjects_avg_missing_per_roi.tsv", sep="\t", index=False)
            out_file_B_all = output_path_B / "version_B_all_subjects_roi_missing_on_brain.png"
            roi_missing_series_B_all = pd.Series(avg_missing_B_all, index=full_roi_list)
            plot_roi_missing_on_brain(roi_missing_series_B_all, full_roi_list, atlas_coords_df,
                                      "Version B (All Subjects): Average ROI Missing Data", out_file_B_all, colormap)
        else:
            logging.warning("No subject data loaded for Version B (all subjects).")
        
        # --- Version B: FILTERED SUBJECTS ---
        logging.info("Processing Version B (filtered subjects) from %s", version_B_path)
        subject_data_B = load_subject_data(args.atlas, version_B_path, final_subjects, args.pipeline, full_roi_list)
        print("Version B (filtered) loaded for", len(subject_data_B), "subjects.")
        if not subject_data_B:
            raise ValueError("No subject data loaded for Version B (filtered subjects).")
        subjects_B, missing_matrix_B = build_missing_rate_matrix(subject_data_B, full_roi_list)
        print("Number of Version B (filtered subjects) used:", len(subjects_B))
        avg_missing_B = average_missing_rate(missing_matrix_B)
        df_avg_B = pd.DataFrame({"roi": full_roi_list, "avg_missing_pct": avg_missing_B})
        df_avg_B.to_csv(output_path_B / "version_B_filtered_avg_missing_per_roi.tsv", sep="\t", index=False)
        out_file_B = output_path_B / "version_B_filtered_roi_missing_on_brain.png"
        roi_missing_series_B = pd.Series(avg_missing_B, index=full_roi_list)
        plot_roi_missing_on_brain(roi_missing_series_B, full_roi_list, atlas_coords_df,
                                  "Version B (Filtered): Average ROI Missing Data", out_file_B, colormap)
        
        # --- Difference Plot within Version B (All vs. Filtered) using relative difference ---
        rel_diff_B = ((avg_missing_B - avg_missing_B_all) / (avg_missing_B_all + epsilon)) * 100
        df_rel_diff_B = pd.DataFrame({"roi": full_roi_list,
                                      "rel_diff_missing_pct": rel_diff_B,
                                      "all_subjects": avg_missing_B_all,
                                      "filtered": avg_missing_B})
        diff_file_B = output_path_B / "rel_diff_all_vs_filtered_avg_missing_per_roi.tsv"
        df_rel_diff_B.to_csv(diff_file_B, sep="\t", index=False)
        logging.info("Relative difference between unfiltered and filtered missing rates (Version B) saved to %s", diff_file_B)
        out_file_diff_B = output_path_B / "rel_diff_all_vs_filtered_roi_missing_on_brain.png"
        roi_rel_diff_series_B = pd.Series(rel_diff_B, index=full_roi_list)
        plot_roi_missing_on_brain(roi_rel_diff_series_B, full_roi_list, atlas_coords_df,
                                  "Relative Difference (All vs Filtered, Version B) [%]", out_file_diff_B, cmap=colormap)
        
        # --- [New] Inter–Version Comparison for ALL SUBJECTS (relative difference) ---
        if subject_data_A_all and subject_data_B_all:
            rel_diff_inter_all = ((avg_missing_A_all - avg_missing_B_all) / (avg_missing_B_all + epsilon)) * 100
            df_diff_inter_all = pd.DataFrame({
                "roi": full_roi_list,
                "rel_diff_missing_pct": rel_diff_inter_all,
                "version_A_all": avg_missing_A_all,
                "version_B_all": avg_missing_B_all
            })
            inter_all_file = output_path_A / "rel_diff_inter_all_avg_missing_per_roi.tsv"
            df_diff_inter_all.to_csv(inter_all_file, sep="\t", index=False)
            logging.info("Inter–version relative difference (all subjects) saved to %s", inter_all_file)
            out_file_inter_all = output_path_A / "rel_diff_inter_all_roi_missing_on_brain.png"
            roi_diff_series_inter_all = pd.Series(rel_diff_inter_all, index=full_roi_list)
            plot_roi_missing_on_brain(roi_diff_series_inter_all, full_roi_list, atlas_coords_df,
                                      "Inter–Version Relative Difference (All Subjects): Version A vs Version B",
                                      out_file_inter_all, cmap=colormap)
            print("Inter–Version (All Subjects) relative difference calculated for", len(roi_diff_series_inter_all), "ROIs.")
        else:
            logging.warning("Cannot compute inter–version (all subjects) relative difference due to missing data.")
        
        # --- [New] Inter–Version Comparison for FILTERED SUBJECTS (relative difference) ---
        rel_diff_inter_filtered = ((avg_missing_A - avg_missing_B) / (avg_missing_B + epsilon)) * 100
        df_diff_inter_filtered = pd.DataFrame({
            "roi": full_roi_list,
            "rel_diff_missing_pct": rel_diff_inter_filtered,
            "version_A_filtered": avg_missing_A,
            "version_B_filtered": avg_missing_B
        })
        inter_filtered_file = output_path_A / "rel_diff_inter_filtered_avg_missing_per_roi.tsv"
        df_diff_inter_filtered.to_csv(inter_filtered_file, sep="\t", index=False)
        logging.info("Inter–version relative difference (filtered subjects) saved to %s", inter_filtered_file)
        out_file_inter_filtered = output_path_A / "rel_diff_inter_filtered_roi_missing_on_brain.png"
        roi_diff_series_inter_filtered = pd.Series(rel_diff_inter_filtered, index=full_roi_list)
        plot_roi_missing_on_brain(roi_diff_series_inter_filtered, full_roi_list, atlas_coords_df,
                                  "Inter–Version Relative Difference (Filtered Subjects): Version A vs Version B",
                                  out_file_inter_filtered, cmap=colormap)
        print("Inter–Version (Filtered Subjects) relative difference calculated for", len(roi_diff_series_inter_filtered), "ROIs.")
    else:
        print("Only one version provided; skipping Version B comparison.")
    
    # --- [New] Verify differences between Unfiltered and Filtered for each version ---
    # For Version A:
    diff_check_A = np.abs(avg_missing_A_all - avg_missing_A)
    print("=== Version A Difference Statistics (All vs Filtered) ===")
    print(f"Mean absolute difference: {np.mean(diff_check_A):.4f}")
    print(f"Median absolute difference: {np.median(diff_check_A):.4f}")
    corr_A = np.corrcoef(avg_missing_A_all, avg_missing_A)[0, 1]
    print(f"Correlation between unfiltered and filtered: {corr_A:.4f}")
    
    # For Version B, if available:
    if args.version_b and subject_data_B_all:
        diff_check_B = np.abs(avg_missing_B_all - avg_missing_B)
        print("=== Version B Difference Statistics (All vs Filtered) ===")
        print(f"Mean absolute difference: {np.mean(diff_check_B):.4f}")
        print(f"Median absolute difference: {np.median(diff_check_B):.4f}")
        corr_B = np.corrcoef(avg_missing_B_all, avg_missing_B)[0, 1]
        print(f"Correlation between unfiltered and filtered: {corr_B:.4f}")
    
if __name__ == "__main__":
    print("Starting missing data pattern evaluation on brain template...")
    main()