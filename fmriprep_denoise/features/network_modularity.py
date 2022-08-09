import numpy as np
from nilearn.connectome import vec_to_sym_matrix
from bct import modularity_louvain_und_sign
from math import sqrt


def louvain_modularity(vect):
    """
    Wrapper for `modularity_louvain_und_sign` from the Brain Connectivity
    tool box.

    Parameters
    ----------

    vect : np.ndarray
        Flatten connetome.

    Returns
    -------
    np.ndarray
        modularity (qtype dependent)

    """
    vect = np.array(vect)
    n = vect.shape[-1]
    n_columns = int((sqrt(8 * n + 1) - 1.0) / 2) + 1  # no diagnal

    full_graph = vec_to_sym_matrix(vect, diagonal=np.ones(n_columns))
    _, modularity = compute_commuity(full_graph, num_opt=100)
    return modularity


def compute_commuity(G, num_opt=100):
    """
    Compute community affiliation vector. Wrapper for
    `modularity_louvain_und_sign` from the Brain Connectivity tool box.

    Parameters
    ----------

    G : np.ndarray
        Symmetric Graph

    num_opt : int
        Number of Louvain optimizations to perform

    Return
    ------

    np.ndarray
        community affiliation vector

    np.ndarray
        modularity (qtype dependent)
    """
    CI = np.empty((G.shape[0], num_opt))
    Qs = np.empty((num_opt))
    for i in range(num_opt):
        P, Q = modularity_louvain_und_sign(G)
        CI[:, i] = P
        Qs[i] = Q
    return CI, Qs.mean()
