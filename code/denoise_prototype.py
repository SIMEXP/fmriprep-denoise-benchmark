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
from utils.dataset import fetch_fmriprep_derivative, deconfound_connectome_single_strategy


# define path of input and output
strategy_file = Path(__file__).parent / "benchmark_strategies.json"
# strategy_file = Path(__file__).parent / "test_strategies.json"
output = Path(__file__).parents[1] / "results"


def main():
    """Main function."""
    data = fetch_fmriprep_derivative(
        Path(__file__).parents[1] / "inputs/dev_data/participants.tsv",
        Path(__file__).parents[1] / "inputs/dev_data",
        "task-pixar")

    # read the strategy deining files
    with open(strategy_file, "r") as file:
        benchmark_strategies = json.load(file)

    # get motion QC related metrics from confound files
    group_mean_fd = pd.DataFrame()
    group_mean_fd.index = group_mean_fd.index.set_names("participant_id")
    for confounds in data.confounds:
        subject_id = confounds.split("/")[-1].split("_")[0]
        confounds = pd.read_csv(confounds, sep="\t")
        mean_fd = confounds["framewise_displacement"].mean()
        group_mean_fd.loc[subject_id, "mean_framewise_displacement"] = mean_fd

    # load gender and age as confounds for the developmental dataset
    participants = data.phenotypic.copy()
    covar = participants.loc[:, ["Age", "Gender"]]
    covar.loc[covar['Gender'] == 'F', 'Gender'] = 1
    covar.loc[covar['Gender'] == 'M', 'Gender'] = 0
    covar['Gender'] = covar['Gender'].astype('float')

    movement = pd.concat((group_mean_fd, covar), axis=1)


    atlas = datasets.fetch_atlas_difumo(dimension=64).maps
    masker = NiftiMapsMasker(atlas, standardize=True, detrend=True)

    # deconfound connectome by all possible strategies
    # generate connectome for every strategy and every subject
    # Soon there will be a atlas level
    clean_correlation_matrix = {}
    for strategy_name, parameters in benchmark_strategies.items():
        strategy = {strategy_name: parameters}
        dataset_connectomes = deconfound_connectome_single_strategy(data.func, masker, strategy)
        # dump the intrim results
        clean_correlation_matrix.update({
            strategy_name: dataset_connectomes.copy()})

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
