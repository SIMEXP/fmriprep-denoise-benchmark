"""Prototype for the denoising project.

To-do:

- plotting and reports
- euclidian distance calculation for parcel centre of mass
- write dataset fetcher for fmriprep output and atlas (if not in nilearn)
  mimicing the nilearn API?
- consider other metrics - perhaps look at Parker 2018

"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import datasets
from nilearn.input_data import NiftiMapsMasker

from metrics import quality_control_connectivity
from utils.dataset import fetch_fmriprep_derivative, deconfound_connectome_single_strategy, ds000288_movement


# define path of input and output
STRATEGY = "code/benchmark_strategies.json"
OUTPUT = "results"
INPUT_FMRIPREP = "inputs/dev_data"
INPUT_BIDS_PARTICIPANTS = "inputs/dev_data/participants.tsv"
FMRIPREP_SPECIFIER = "task-pixar"


def main():
    """Main function."""
    strategy_file = Path(__file__).parents[1] / STRATEGY
    output = Path(__file__).parents[1] / OUTPUT
    input_fmriprep = Path(__file__).parents[1] / INPUT_FMRIPREP
    input_bids_participants = Path(__file__).parents[1] / INPUT_BIDS_PARTICIPANTS

    data = fetch_fmriprep_derivative(input_bids_participants, input_fmriprep,
                                     FMRIPREP_SPECIFIER)

    # read the strategy deining files
    with open(strategy_file, "r") as file:
        benchmark_strategies = json.load(file)

    movement = ds000288_movement(data)

    atlas = datasets.fetch_atlas_difumo(dimension=64).maps
    masker = NiftiMapsMasker(atlas, standardize=True, detrend=True)

    # deconfound connectome by all possible strategies
    # generate connectome for every strategy and every subject
    # Soon there will be a atlas level
    clean_correlation_matrix = {}
    for strategy_name, parameters in benchmark_strategies.items():
        strategy = {strategy_name: parameters}
        dataset_connectomes = deconfound_connectome_single_strategy(data.func, masker, strategy)
        clean_correlation_matrix[strategy_name] = dataset_connectomes.copy()

    # calculate QC metrices
    metric_per_edge, sig_per_edge = pd.DataFrame(), pd.DataFrame()
    for strategy_name, dataset_connectomes in clean_correlation_matrix.items():
        # QC-FC per edge
        cur_qc_fc, cur_sig = quality_control_connectivity(movement, dataset_connectomes)
        # dump the results
        metric_per_edge.loc[:, strategy_name] = cur_qc_fc
        sig_per_edge.loc[:, strategy_name] = cur_sig

    # plotting test
    ax = sns.barplot(data=(sig_per_edge<0.05), ci=None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.set(ylabel='Proportion of edge significantly correlated with mean FD',
           xlabel='confound removal strategy')
    plt.savefig(output / "percentage_sig_edge.png")
    plt.close()
    sns.stripplot(data=metric_per_edge, dodge=True, alpha=.01, zorder=1)
    sns.pointplot(data=metric_per_edge, dodge=.8 - .8 / 3,
                  join=False, palette="dark",
                  estimator=np.median,
                  markers="d", scale=.75, ci=None)
    plt.savefig(output / "dsitribution_edge.png")


if __name__ == "__main__":
    main()
