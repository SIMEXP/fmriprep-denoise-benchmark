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
from scipy import stats, linalg
from sklearn.utils import Bunch

from nilearn import datasets

from nilearn.connectome import ConnectivityMeasure
from nilearn.input_data import (fmriprep_confounds,
                                NiftiLabelsMasker, NiftiMapsMasker)


# define path of input and output
# strategy_file = Path(__file__).parent / "benchmark_strategies.json"
strategy_file = Path(__file__).parent / "test_strategies.json"
output = Path(__file__).parents[1] / "results"

def generate_dataset_connectomes(func_img, masker, strategy_name, parameters):
    """Create confound-removed subject-level connectomes for a dataset.

    Confound removal is defined by predefinied denoising strategy.

    Return
    ------
    pd.DataFrame :
        Shape: number of subject by number of edges
        Subjects ID should be the index and edge pair the header.
    """
    dataset_connectome = pd.DataFrame()
    for img in func_img:
        subject_id = img.split("/")[-1].split("_")[0]
        subject_timeseries = _get_timeseries(masker, strategy_name, parameters,
                                             img)
        dataset_connectome = _generate_subject_connectome(dataset_connectome,
                                                          subject_id,
                                                          subject_timeseries)
    return dataset_connectome


def _get_timeseries(masker, strategy_name, parameters, img):
    """Extract usable time series.

    Use masker to extract cleaned timeseries when the sample_mask is a valid
    value. With scrubbing, sample_mask can be an empty array if a subject has
    excessive motion.

    fmriprep_confounds need some warning when all voxels are scrubbed.
    """
    if strategy_name == "no_cleaning":
        return masker.fit_transform(img)

    reduced_confounds, sample_mask = fmriprep_confounds(img, **parameters)
    if sample_mask is None or len(sample_mask) != 0:
        return masker.fit_transform(img, confounds=reduced_confounds,
                                    sample_mask=sample_mask)
    else:
        return None


def _generate_subject_connectome(cleaned_connectome_collector, subject_id,
                                 subject_timeseries):
    """Calculate the connectome and append to the existing results.

    If the subject timeseries is invalid, append nan.
    """
    if isinstance(subject_timeseries, np.ndarray):
        correlation_measure = ConnectivityMeasure(kind='correlation',
                                                  vectorize=True,
                                                  discard_diagonal=True)
        flat_connectome = correlation_measure.fit_transform(
            [subject_timeseries])
        flat_connectome = pd.DataFrame(flat_connectome, index=[subject_id])
        cleaned_connectome_collector = pd.concat((cleaned_connectome_collector,
                                                  flat_connectome))
    else:
        cleaned_connectome_collector.loc[subject_id, :] = np.nan
    return cleaned_connectome_collector


def partial_correlation(x, y, cov):
    """A minimal implementation of partial correlation.

    x, y :
        Variable of interest.
    cov :
        Variable to be removed from variable of interest.
    """
    # matric calculation: QC-FC
    # For each edge, we then computed the correlation between the weight of
    # that edge and the mean relative RMS motion.
    # QC-FC relationships were calculated as partial correlations that
    # accounted for participant age and sex
    beta_cov_x = linalg.lstsq(cov, x)[0]
    beta_cov_y = linalg.lstsq(cov, y)[0]
    resid_x = x - cov.dot(beta_cov_x)
    resid_y = y - cov.dot(beta_cov_y)
    return stats.pearsonr(resid_x, resid_y)


def fetch_fmriprep_derivative(path_dataset, path_fmriprep_derivative,
                              specifier, space="MNI152NLin2009cAsym"):
    """Fetch fmriprep derivative and return nilearn.dataset.fetch* like output.
    Load functional image, confounds, and participants.tsv only.
    """
    participant_tsv_path = path_dataset / "participants.tsv"
    if not participant_tsv_path.is_file():
        raise(FileNotFoundError, f"Cannot find {participant_tsv_path}")
    participant_tsv = pd.read_csv(path_dataset / "participants.tsv",
                                  index_col="participant_id",
                                  sep="\t")

    subject_dirs = path_fmriprep_derivative.glob("sub-*/")
    func_img_path, confounds_tsv_path, include_subjects = [], [], []
    for subject_dir in subject_dirs:
        subject = subject_dir.name
        cur_func = (subject_dir / "func" /
            f"{subject}_{specifier}_space-{space}_desc-preproc_bold.nii.gz")
        cur_confound = (subject_dir / "func" /
            f"{subject}_{specifier}_desc-confounds_timeseries.tsv")

        if cur_func.is_file() and cur_confound.is_file():
            func_img_path.append(str(cur_func))
            confounds_tsv_path.append(str(cur_confound))
            include_subjects.append(subject)

    return Bunch(func=func_img_path,
                 confounds=confounds_tsv_path,
                 phenotypic=participant_tsv.iloc[include_subjects, :]
                 )


def main():
    """Main function."""
    # load 10 subjects for making the prototype
    # idea: write data fetcher for own data mimicing the nilearn API
    data = datasets.fetch_development_fmri(n_subjects=10,
                                           reduce_confounds=False)

    # get motion QC related metrics from confound files
    group_mean_fd = pd.DataFrame()
    group_mean_fd.index = group_mean_fd.index.set_names("participant_id")
    for confounds in data.confounds:
        subject_id = confounds.split("/")[-1].split("_")[0]
        confounds = pd.read_csv(confounds, sep="\t")
        mean_fd = confounds["framewise_displacement"].mean()
        group_mean_fd.loc[subject_id, "mean_framewise_displacement"] = mean_fd

    # load gender and age as confounds
    participants = pd.DataFrame(data.phenotypic).set_index("participant_id")
    covar = participants.loc[:, ["Age", "Gender"]]
    covar.loc[covar['Gender'] == 'F', 'Gender'] = 1
    covar.loc[covar['Gender'] == 'M', 'Gender'] = 0
    covar['Gender'] = covar['Gender'].astype('float')

    # read the strategy deining files
    with open(strategy_file, "r") as file:
        benchmark_strategies = json.load(file)

    # create a masker
    atlas = datasets.fetch_atlas_difumo(dimension=64)
    masker = NiftiMapsMasker(atlas.maps, standardize=True, detrend=True)

    # generate connectome for every strategy and every subject
    # Soon there will be a atlas level
    clean_correlation_matrix = {}
    metric_per_edge, sig_per_edge = pd.DataFrame(), pd.DataFrame()
    for strategy_name, parameters in benchmark_strategies.items():
        dataset_connectomes = generate_dataset_connectomes(data.func,
                                                           masker,
                                                           strategy_name,
                                                           parameters)
        # dump the intrim results
        clean_correlation_matrix.update({
            strategy_name: dataset_connectomes.copy()})
        # QC-FC per edge
        cur_qc_fc, cur_sig = [], []
        for edge_id, edge_val in dataset_connectomes.iteritems():
            # concatenate information to match by subject id
            current_edge = pd.concat((edge_val, group_mean_fd, covar), axis=1)
            # drop subject with no edge value
            current_edge = current_edge.dropna()
            # QC-FC
            r, p_val = partial_correlation(
                current_edge[edge_id].values,
                current_edge["mean_framewise_displacement"].values,
                current_edge[["Age", "Gender"]].values)
            cur_qc_fc.append(r)
            cur_sig.append(p_val)

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

import pytest


# Test of some customised function
def test_fetch_fmriprep_derivative(tmp_path):


def generate_dir(specifier, space):
    for subject in range(1, 11):
        template = f"sub-{subject:03d}/func/sub-{subject:03d}_{specifier}_space-{space}_desc-preproc.nii.gz"  # no qa


# def test_partial_correlation():
#     partial_correlation()