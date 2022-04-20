from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import distance

from fmriprep_denoise.utils.atlas import get_centroid


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

