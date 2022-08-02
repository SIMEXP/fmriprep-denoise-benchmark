from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from fmriprep_denoise.visualization import utils


path_root = utils.repo2data_path()


def lazy_demographic(dataset):
    """
    Very lazy report of demographic information
    """
    df, groups = _get_descriptive_data(dataset)
    full = df.describe()['age']
    full.name = 'full sample'
    print(f"n female: {df['gender'].sum()}")

    desc = [full]
    for g in groups:
        sub_group = df[df['groups']==g].describe()['age']
        sub_group.name = g
        print(f"n female in {g}: {df.loc[df['groups']==g,'gender'].sum()}")
        desc.append(sub_group)

    return pd.concat(desc, axis=1)


def _get_descriptive_data(dataset):
    """Get the data frame of all descriptive data needed for a dataset."""
    # load basic data
    movements = path_root / f"dataset-{dataset}" / f'dataset-{dataset}_desc-movement_phenotype.tsv'
    movements = pd.read_csv(movements, index_col=0, sep='\t')
    _, participants_groups, groups = utils._get_participants_groups(dataset)
    participants_groups.name = 'groups'
    df = pd.concat([movements, participants_groups], axis=1)
    df = df.rename(columns={'mean_framewise_displacement': 'mean FD'})
    return df, groups
