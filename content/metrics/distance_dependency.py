import numpy as np
import pandas as pd
from scipy.spatial import distance


def compute_pairwise_distance(rsa_centroids):
    """Compute pairwise distance of a given set of centroids and flatten."""
    pairwise_distance = distance.cdist(rsa_centroids, rsa_centroids)
    lables = range(1, pairwise_distance.shape[0] + 1)
    # Transform into pandas dataframe
    pairwise_distance = pd.DataFrame(pairwise_distance, index=lables, columns=lables)
    # keep upper triangle and flatten
    upper_mask = np.triu(np.ones(pairwise_distance.shape), k=1).astype(np.bool)
    pairwise_distance = pairwise_distance.where(upper_mask)
    pairwise_distance = pairwise_distance.stack().reset_index()
    pairwise_distance.columns = ['row','column','distance']
    return pairwise_distance
