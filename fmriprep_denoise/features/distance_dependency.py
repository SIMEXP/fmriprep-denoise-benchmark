from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import distance

from fmriprep_denoise.data.atlas import fetch_atlas_path, ATLAS_METADATA
from nilearn.image import index_img
from nilearn.plotting import find_probabilistic_atlas_cut_coords


def _compute_pairwise_distance(centroids):
    """Compute pairwise distance of a given set of centroids and flatten."""
    pairwise_distance = distance.cdist(centroids, centroids)
    lables = range(1, pairwise_distance.shape[0] + 1)

    # Transform into pandas dataframe
    pairwise_distance = pd.DataFrame(pairwise_distance, index=lables, columns=lables)
    # keep lower triangle and flatten match nilearn.connectome.sym_matrix_to_vec
    lower_mask = np.tril(np.ones(pairwise_distance.shape), k=-1).astype(np.bool)
    pairwise_distance = pairwise_distance.where(lower_mask)
    pairwise_distance = pairwise_distance.stack().reset_index()
    pairwise_distance.columns = ['row', 'column', 'distance']
    return pairwise_distance


def get_atlas_pairwise_distance(atlas_name, dimension):
    if atlas_name == 'gordon333':
        file_dist = "atlas-gordon333_nroi-333_desc-distance.tsv"
        return pd.read_csv(Path(__file__).parent / "data" / file_dist, sep='\t')
    return _compute_pairwise_distance(get_centroid(atlas_name, dimension))


def get_centroid(atlas_name, dimension):
    """Load parcel centroid for each atlas."""
    if atlas_name not in ATLAS_METADATA:
        raise NotImplementedError("Selected atlas is not supported.")

    if atlas_name == 'schaefer7networks':
        url = f"https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/Schaefer2018_{dimension}Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv"
        return pd.read_csv(url).loc[:, ['R', 'S', 'A']].values
    if atlas_name == 'gordon333':
        file_dist = "atlas-gordon333_nroi-333_desc-distance.tsv"
        return pd.read_csv(Path(__file__).parent / "data" / file_dist, sep='\t')

    current_atlas = fetch_atlas_path(atlas_name, dimension)
    if atlas_name == 'mist':
        return current_atlas.labels.loc[:, ['x', 'y', 'z']].values
    if atlas_name == 'difumo':
        # find files
        p = Path(__file__).parent / "data" / f"atlas-DiFuMo_nroi-{dimension}_desc-distance.tsv"
        if not p.is_file():
            get_difumo_centroids(dimension)
        return pd.read_csv(p, sep='\t')


def get_difumo_centroids(d):
    current_atlas = fetch_atlas_path('difumo', d)
    if d > 256:
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
    centroid = pd.DataFrame(centroid, columns=['x', 'y', 'z'])
    centroid = pd.concat([current_atlas.labels, centroid], axis=1)
    output = Path(__file__).parent / "data" / f"atlas-DiFuMo_nroi-{d}_desc-distance.tsv"
    centroid.to_csv(output, sep='\t')
