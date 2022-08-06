from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from fmriprep_denoise.visualization import utils


fd2label = {0.5: 'scrubbing.5', 0.2: 'scrubbing.2'}


def lazy_demographic(
    dataset, path_root, gross_fd=None, fd_thresh=None, proportion_thresh=None
):
    """
    Very lazy report of demographic information
    """
    if not fd2label.get(fd_thresh, False) and fd_thresh is not None:
        raise ValueError(
            'We did not generate metric with scrubbing threshold set at'
            f'framewise displacement = {fd_thresh} mm.'
        )
    df, groups = get_descriptive_data(
        dataset, path_root, gross_fd, fd_thresh, proportion_thresh
    )
    full = df.describe()['age']
    full.name = 'full sample'
    print(f"n female: {df['gender'].sum()}")

    desc = [full]
    for g in groups:
        sub_group = df[df['groups'] == g].describe()['age']
        sub_group.name = g
        print(f"n female in {g}: {df.loc[df['groups']==g,'gender'].sum()}")
        desc.append(sub_group)

    return pd.concat(desc, axis=1)


def get_descriptive_data(
    dataset, path_root, gross_fd=None, fd_thresh=None, proportion_thresh=None
):
    """Get the data frame of all descriptive data needed for a dataset."""
    if not fd2label.get(fd_thresh, False) and fd_thresh is not None:
        raise ValueError(
            'We did not generate metric with scrubbing threshold set at'
            f'framewise displacement = {fd_thresh} mm.'
        )
    # load basic data
    movements = (
        path_root
        / f'dataset-{dataset}'
        / f'dataset-{dataset}_desc-movement_phenotype.tsv'
    )
    movements = pd.read_csv(movements, index_col=0, sep='\t')

    (
        confounds_phenotype,
        participant_groups,
        groups,
    ) = utils._get_participants_groups(dataset, path_root)
    participant_groups.name = 'groups'

    if gross_fd is not None:
        keep_gross_fd = movements['mean_framewise_displacement'] <= gross_fd
        keep_gross_fd = movements.index[keep_gross_fd]
    else:
        keep_gross_fd = movements.index

    if fd_thresh is not None and proportion_thresh is not None:
        scrub_label = (fd2label[fd_thresh], 'excised_vol_proportion')
        keep_scrub = confounds_phenotype[scrub_label] <= proportion_thresh
        keep_scrub = confounds_phenotype.index[keep_scrub]
    else:
        keep_scrub = confounds_phenotype.index
    mask_motion = keep_gross_fd.intersection(keep_scrub)
    participant_groups = participant_groups.loc[mask_motion]
    df = movements.loc[participant_groups.index, :]
    return df, groups
