import numpy as np
import pandas as pd
from scipy.spatial import distance
from nilearn.connectome import sym_matrix_to_vec

def compute_pairwise_distance(centroids):
    """Compute pairwise distance of a given set of centroids and flatten."""
    pairwise_distance = distance.cdist(centroids, centroids)
    lables = range(1, pairwise_distance.shape[0] + 1)
    # vec = sym_matrix_to_vec(pairwise_distance, discard_diagonal=True)

    # Transform into pandas dataframe
    pairwise_distance = pd.DataFrame(pairwise_distance, index=lables, columns=lables)
    # keep lower triangle and flatten
    lower_mask = np.tril(np.ones(pairwise_distance.shape), k=-1).astype(np.bool)
    pairwise_distance = pairwise_distance.where(lower_mask)
    pairwise_distance = pairwise_distance.stack().reset_index()
    pairwise_distance.columns = ['row', 'column', 'distance']
    return pairwise_distance
