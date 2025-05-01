#!/usr/bin/env python3
"""
Visualization script for plotting atlas ROIs on a brain template.
- ROIs present in the provided "exclusion" CSV (the limited/included list) are colored green.
- ROIs not in the limited list are colored red.
- Additionally, any ROI that was imputed (tracked via a global imputation counts CSV)
  is plotted as a blue dot (with the color now transitioning from green for low counts
  to yellow for high counts). ROIs that were imputed will only be plotted once,
  not in the green/red groups.
  
Usage example:
    python visualize_roi_imputation.py \
      --atlas_img /path/to/atlas_image.nii.gz \
      --atlas_tsv /path/to/atlas_labels.tsv \
      --exclusion_csv /path/to/included_rois.csv \
      --global_impute_csv /path/to/roi_global_impute_counts.csv \
      --output roi_visualization.png \
      --title "ROI Visualization with Imputation Info"
"""

import argparse
import logging
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import datasets, plotting
from scipy.ndimage import center_of_mass
from matplotlib.colors import LinearSegmentedColormap

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot ROIs on a brain template with inclusion/exclusion and imputation counts."
    )
    parser.add_argument("--atlas_img", required=True,
                        help="Path to the atlas NIfTI image (for computing ROI centroids).")
    parser.add_argument("--atlas_tsv", required=True,
                        help="Path to the atlas TSV file mapping numeric labels to ROI names. "
                             "The file should contain two columns (numeric_label and roi_name).")
    parser.add_argument("--exclusion_csv", required=True,
                        help="CSV file with a column 'roi_name' listing the limited list of ROIs "
                             "that are included (these will be plotted as green).")
    parser.add_argument("--global_impute_csv", required=True,
                        help="CSV file with columns [roi_index, roi_name, global_impute_count] "
                             "tracking how many subjects had global imputation per ROI.")
    parser.add_argument("--output", default="roi_visualization.png",
                        help="Filename for the output brain plot image.")
    parser.add_argument("--title", default="ROI Visualization", help="Plot title")
    return parser.parse_args()

def compute_roi_centroids(atlas_img_path, roi_labels):
    """
    Compute centroids for each ROI given an atlas image and a list of ROI labels.
    Returns a DataFrame with columns: roi (numeric as string), x, y, and z (world coordinates).
    """
    img = nib.load(atlas_img_path)
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

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    # -------------------------------------------------------------------------
    # Load atlas label information.
    atlas_df = pd.read_csv(args.atlas_tsv, sep="\t", header=None, names=["numeric", "roi_name"])
    atlas_df["numeric"] = atlas_df["numeric"].astype(str)
    atlas_df["roi_name"] = atlas_df["roi_name"].astype(str)
    logging.info("Loaded %d atlas entries.", len(atlas_df))
    
    roi_numeric_labels = atlas_df["numeric"].tolist()
    atlas_coords_df = compute_roi_centroids(args.atlas_img, roi_numeric_labels)
    if atlas_coords_df.empty:
        logging.error("No ROI centroids computed. Check your atlas image and labels.")
        return

    # Merge atlas coordinates with ROI names.
    atlas_coords_df = atlas_coords_df.merge(atlas_df, left_on="roi", right_on="numeric", how="left")
    atlas_coords_df = atlas_coords_df[["roi_name", "x", "y", "z"]]
    
    # -------------------------------------------------------------------------
    # Load the limited/included ROIs from the exclusion CSV.
    exclusion_df = pd.read_csv(args.exclusion_csv)
    if "roi_name" not in exclusion_df.columns:
        logging.error("The exclusion CSV must contain a column named 'roi_name'.")
        return
    included_rois = set(exclusion_df["roi_name"].tolist())
    atlas_coords_df["included"] = atlas_coords_df["roi_name"].apply(lambda x: x in included_rois)
    
    # -------------------------------------------------------------------------
    # Load the global imputation counts CSV.
    impute_df = pd.read_csv(args.global_impute_csv)
    atlas_coords_df = atlas_coords_df.merge(impute_df[["roi_name", "global_impute_count"]],
                                            on="roi_name", how="left")
    atlas_coords_df["global_impute_count"] = atlas_coords_df["global_impute_count"].fillna(0).astype(int)
    logging.info("Found %d ROIs with nonzero global imputation counts.", 
                 (atlas_coords_df["global_impute_count"] > 0).sum())
    
    # Split data into two groups:
    # 1) ROIs that have been imputed (global_impute_count > 0) will be plotted in a custom color.
    # 2) ROIs that have NOT been imputed are further split into included (green) and excluded (red).
    imputed_df = atlas_coords_df[atlas_coords_df["global_impute_count"] > 0]
    non_imputed_df = atlas_coords_df[atlas_coords_df["global_impute_count"] == 0]
    non_imputed_included = non_imputed_df[non_imputed_df["included"]]
    non_imputed_excluded = non_imputed_df[~non_imputed_df["included"]]

    # -------------------------------------------------------------------------
    # Create a custom colormap that goes from green to yellow.
    # Lower normalized values will use the same green as the included group.
    green_yellow = LinearSegmentedColormap.from_list("green_yellow", ["green", "yellow"])

    # -------------------------------------------------------------------------
    # Create a glass brain display using the MNI152 template.
    mni_template = datasets.load_mni152_template(resolution=2)
    display = plotting.plot_glass_brain(mni_template,
                                        title=args.title,
                                        display_mode="ortho",
                                        cmap="gray",
                                        colorbar=False)
    
    # Plot non-imputed included ROIs (green).
    if not non_imputed_included.empty:
        display.add_markers(non_imputed_included[["x", "y", "z"]].values,
                            marker_color="green", marker_size=20)
    # Plot non-imputed excluded ROIs (red).
    if not non_imputed_excluded.empty:
        display.add_markers(non_imputed_excluded[["x", "y", "z"]].values,
                            marker_color="red", marker_size=20)
    
    # Plot imputed ROIs with a custom color from green (low count) to yellow (high count).
    if not imputed_df.empty:
        max_count = imputed_df["global_impute_count"].max()
        for _, row in imputed_df.iterrows():
            norm = row["global_impute_count"] / max_count if max_count > 0 else 0
            color = green_yellow(norm)
            display.add_markers(np.array([[row["x"], row["y"], row["z"]]]),
                                marker_color=color, marker_size=20)
    
    plt.savefig(args.output, dpi=300)
    plt.close()
    logging.info("Saved ROI visualization to %s", args.output)

if __name__ == "__main__":
    main()