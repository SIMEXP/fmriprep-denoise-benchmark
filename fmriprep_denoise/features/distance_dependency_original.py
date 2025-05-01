from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import distance

from fmriprep_denoise.dataset.atlas import fetch_atlas_path, ATLAS_METADATA
from nilearn.image import index_img
from nilearn.plotting import find_probabilistic_atlas_cut_coords

from scipy.ndimage import center_of_mass
import nibabel as nib



def get_atlas_pairwise_distance(atlas_name, dimension):
    """
    Compute pairwise distance of nodes in the atlas.

    Parameters
    ----------
    atlas_name : str
        Atlas name. Must be a key in ATLAS_METADATA.

    dimension : str or int
        Atlas dimension.

    Returns
    -------
    pandas.DataFrame
        Node ID pairs and the distance.

    """
    # Special case for gordon333 since it uses a precomputed distance file
    if atlas_name == "gordon333":
        file_dist = "atlas-gordon333_nroi-333_desc-distance.tsv"
        print(f"Loading precomputed distances for atlas '{atlas_name}' from {file_dist}")
        return pd.read_csv(Path(__file__).parent / "data" / file_dist, sep="\t")

    # Debug: Fetching centroids for atlas
    print(f"Fetching centroids for atlas: {atlas_name} with dimension: {dimension}")
    
    # Get centroids from get_centroid()
    centroids = get_centroid(atlas_name, dimension)

    # Check if centroids is None or empty
    if centroids is None:
        raise ValueError(f"Centroids for atlas '{atlas_name}' with dimension '{dimension}' returned None.")
    
    if isinstance(centroids, np.ndarray) and centroids.size == 0:
        raise ValueError(f"Centroids for atlas '{atlas_name}' returned an empty array. Check TSV format or path.")

    # Debug: Shape of centroids
    print(f"Centroids loaded successfully. Shape: {centroids.shape if isinstance(centroids, np.ndarray) else 'N/A'}")
    print(f"First 5 centroids:\n{centroids[:5]}")
    
    # Validate centroids shape before using cdist
    if not isinstance(centroids, np.ndarray):
        raise TypeError(f"Centroids must be a NumPy array, but got {type(centroids)}.")
    
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError(
            f"Centroids must be a 2D array with shape (n, 3). Got shape: {centroids.shape}"
        )

    # Compute pairwise distances using cdist
    print("Computing pairwise distances using cdist...")
    pairwise_distance = distance.cdist(centroids, centroids)

    # Create labels for each node
    labels = range(1, pairwise_distance.shape[0] + 1)

    # Transform into pandas dataframe
    pairwise_distance = pd.DataFrame(pairwise_distance, index=labels, columns=labels)
    
    # Keep only lower triangle and flatten to match nilearn.connectome.sym_matrix_to_vec
    lower_mask = np.tril(np.ones(pairwise_distance.shape), k=-1).astype(bool)
    pairwise_distance = pairwise_distance.where(lower_mask)
    pairwise_distance = pairwise_distance.stack().reset_index()
    pairwise_distance.columns = ["row", "column", "distance"]

    print(f"Pairwise distance computation complete. Returning distance dataframe with shape: {pairwise_distance.shape}")

    return pairwise_distance




def compute_roi_centroids(atlas_img_path, roi_labels):
    """
    Compute centroids for ROIs in an atlas using voxel-based center of mass.

    Parameters
    ----------
    atlas_img_path : str
        Path to the atlas NIfTI image.

    roi_labels : list
        List of numeric ROI labels to calculate centroids.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing centroids with columns: ["roi", "x", "y", "z"].
    """
    img = nib.load(str(atlas_img_path))
    data = img.get_fdata()
    affine = img.affine
    centroids = []

    # Iterate over all numeric ROI labels and compute centroids
    for roi in roi_labels:
        try:
            roi_val = float(roi)  # Ensure the value is numeric
        except ValueError:
            continue

        mask = (data == roi_val)  # Create a binary mask for the ROI
        if np.sum(mask) == 0:
            continue

        # Compute the center of mass in voxel space and convert to world space
        com_voxel = center_of_mass(mask)
        com_world = nib.affines.apply_affine(affine, com_voxel)

        # Append centroid coordinates to the list
        centroids.append({"roi": str(roi), "x": com_world[0], "y": com_world[1], "z": com_world[2]})

    return pd.DataFrame(centroids)


def get_centroid(atlas_name, dimension):
    """
    Load or generate centroids for the given atlas.

    Parameters
    ----------
    atlas_name : str
        Atlas name (e.g., "Schaefer2018Combined").

    dimension : str or int
        Atlas dimension (e.g., 200, 400, etc.).

    Returns
    -------
    np.ndarray
        Array with shape (n, 3) containing centroid coordinates.
    """
    if atlas_name not in ATLAS_METADATA:
        raise NotImplementedError(f"Atlas '{atlas_name}' is not supported.")

    # Define path to TSV and NIfTI files
    atlas_tsv_path = f"/home/seann/projects/def-cmoreau/All_user_common_folder/atlas/atlas_enigma/atlas-{atlas_name}Combined_dseg.tsv"
    atlas_img_path = f"/home/seann/projects/def-cmoreau/All_user_common_folder/atlas/atlas_enigma/atlas-{atlas_name}Combined_dseg.nii.gz"
    centroid_tsv_path = f"/home/seann/scratch/denoise/fmriprep-denoise-benchmark/outputs/denoise-metrics/ds000228/fmriprep-25.0.0/atlas-{atlas_name}Combined_centroids.tsv"

    # Check if centroids have already been computed
    if not Path(centroid_tsv_path).is_file():
        # Load ROI labels from TSV
        print(f"Loading ROI labels from TSV: {atlas_tsv_path}")
        atlas_labels_df = pd.read_csv(atlas_tsv_path, sep="\t", header=None, names=["numeric", "roi_full"])
        atlas_labels_df["numeric"] = atlas_labels_df["numeric"].astype(str)
        atlas_labels_df["roi_full"] = atlas_labels_df["roi_full"].astype(str)

        # Get numeric ROI labels as a list
        roi_numeric_labels = atlas_labels_df["numeric"].tolist()
        print(f"Found {len(roi_numeric_labels)} ROIs in the atlas TSV.")

        # Compute centroids and save to TSV
        centroids_df = compute_roi_centroids(atlas_img_path, roi_numeric_labels)
        centroids_df["roi"] = centroids_df["roi"].astype(str)

        # Merge centroids with ROI full names
        centroids_df = centroids_df.merge(atlas_labels_df, left_on="roi", right_on="numeric", how="left")
        centroids_df["roi"] = centroids_df["roi_full"]
        centroids_df.drop(columns=["numeric", "roi_full"], inplace=True)

        # Save computed centroids to TSV for future use
        centroids_df.to_csv(centroid_tsv_path, sep="\t", index=False)
        print(f"Centroid file generated and saved to: {centroid_tsv_path}")

    # Load precomputed centroids from TSV
    print(f"Loading centroids from precomputed TSV: {centroid_tsv_path}")
    centroids_df = pd.read_csv(centroid_tsv_path, sep="\t")
    
    if set(["x", "y", "z"]).issubset(centroids_df.columns):
        centroids = centroids_df.loc[:, ["x", "y", "z"]].values
    else:
        raise ValueError(f"Centroid file '{centroid_tsv_path}' does not contain expected columns.")
    
    print(f"Centroids successfully loaded. Shape: {centroids.shape}")
    return centroids


def get_difumo_centroids(d):
    """
    Compute difumo centroids.

    Parameters
    ----------

    d : int
        Atlas dimension.

    """
    current_atlas = fetch_atlas_path("difumo", d)
    if d > 256:
        # split the map and work on individual maps
        n_roi = current_atlas.labels.shape[0]
        centroid = []
        for i in range(0, n_roi, 200):
            if i == 0:
                start = i
                continue
            img = index_img(current_atlas.maps, slice(start, i))
            c = find_probabilistic_atlas_cut_coords(img)
            centroid.append(c)
            start = i
        img = index_img(current_atlas.maps, slice(start, n_roi))
        c = find_probabilistic_atlas_cut_coords(img)
        centroid.append(c)
        centroid = np.vstack(centroid)
    else:
        centroid = find_probabilistic_atlas_cut_coords(current_atlas.maps)
    centroid = pd.DataFrame(centroid, columns=["x", "y", "z"])
    centroid = pd.concat([current_atlas.labels, centroid], axis=1)
    output = Path(__file__).parent / "data" / f"atlas-DiFuMo_nroi-{d}_desc-distance.tsv"
    centroid.to_csv(output, sep="\t")
