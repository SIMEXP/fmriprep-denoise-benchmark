import os
import argparse
from pathlib import Path
import json

import pandas as pd

from nilearn.connectome import ConnectivityMeasure
from nilearn.signal import clean
from nilearn.interfaces.fmriprep import load_confounds_strategy, load_confounds

from fmriprep_denoise.utils.dataset import fetch_fmriprep_derivative, phenotype_movement
from fmriprep_denoise.utils.atlas import create_atlas_masker, get_atlas_dimensions


# define path of input and output
# INPUT_FMRIPREP = "{home}/scratch/test_data/1637790137/fmriprep"
# INPUT_BIDS_PARTICIPANTS = "{home}/projects/rrg-pbellec/hwang1/test_data/participants.tsv"
ATLAS = 'schaefer7networks'


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Generate connectome based on denoising strategy for fmriprep processed dataset.",
    )
    parser.add_argument(
        "output_path",
        action="store",
        type=str,
        help="output path for connectome."
    )
    parser.add_argument(
        "--fmriprep_path",
        action="store",
        type=str,
        help="Path to a fmriprep dataset."
    )
    parser.add_argument(
        "--dataset_name",
        action="store",
        type=str,
        help="Dataset name."
    )
    parser.add_argument(
        "--subject",
        action="store",
        type=str,
        help="subject id."
    )
    parser.add_argument(
        "--specifier",
        action="store",
        type=str,
        help="Text in a fmriprep file name, in between sub-<subject>_ses-<session>_and `space-<template>`."
    )
    parser.add_argument(
        "--participants_tsv",
        action="store",
        type=str,
        help="Path to participants.tsv in the original BIDS dataset."
    )
    parser.add_argument(
        "--atlas",
        default=ATLAS,
        type=str,
        help="Atlas name (schaefer7networks, MIST, difumo, gordon333)"
    )
    parser.add_argument(
        "--strategy-name",
        action="store",
        default=None,
        help=("Denoise strategy name (see benchmark_strategies.json)."
              "Process all strategy if None.")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(vars(args))
    dataset_name = args.dataset_name
    subject = args.subject
    strategy_name = args.strategy_name
    atlas_name = args.atlas
    fmriprep_specifier = args.specifier
    fmriprep_path = Path(args.fmriprep_path)
    participant_tsv = Path(args.participants_tsv)
    output = Path(args.output_path)

    output.mkdir(exist_ok=True)
    strategy_file = Path(__file__).parent / "benchmark_strategies.json"

    _movement_summary(dataset_name, fmriprep_specifier, fmriprep_path, participant_tsv, output)

    data_aroma = fetch_fmriprep_derivative(dataset_name,
                                           participant_tsv, fmriprep_path,
                                           fmriprep_specifier, subject=subject, aroma=True)
    data = fetch_fmriprep_derivative(dataset_name,
                                     participant_tsv, fmriprep_path,
                                     fmriprep_specifier, subject=subject)
    benchmark_strategies, strategy_names = _get_prepro_strategy(strategy_name, strategy_file)
    subject_spec, subject_output, subject_mask, subject_mask_aroma = _get_subject_info(output, data_aroma, data)
    dimensions = get_atlas_dimensions(atlas_name)

    for dimension in dimensions:
        print(f"-- {atlas_name}: dimension {dimension} --")
        atlas_spec = f"atlas-{atlas_name}_nroi-{dimension}"
        print("raw time series")
        rawts_path = subject_output / f"{subject_spec}_{atlas_spec}_desc-raw_timeseries.tsv"
        raw_masker, _ = create_atlas_masker(atlas_name, dimension, subject_mask, detrend=False, nilearn_cache="")
        subject_timeseries = generate_raw_timeseries(rawts_path, data, raw_masker)

        for strategy_name in strategy_names:
            parameters = benchmark_strategies[strategy_name]
            print(f"Denoising: {strategy_name}")
            print(parameters)
            ts_path = subject_output / f"{subject_spec}_{atlas_spec}_desc-{strategy_name}_timeseries.tsv"
            img = data_aroma.func[0] if "aroma" in strategy_name else data.func[0]
            reduced_confounds, sample_mask = _get_confounds(strategy_name, parameters, img)

            if "aroma" in strategy_name:
                aroma_masker, _ = create_atlas_masker(atlas_name, dimension, subject_mask_aroma, nilearn_cache="")
                clean_timeseries = aroma_masker.fit_transform(
                    img, confounds=reduced_confounds, sample_mask=sample_mask)
            elif _check_exclusion(reduced_confounds, sample_mask):
                clean_timeseries = []
            else:
                clean_timeseries = clean(subject_timeseries,
                                        detrend=True, standardize=True,
                                        sample_mask=sample_mask,
                                        confounds=reduced_confounds)
            clean_timeseries = pd.DataFrame(clean_timeseries)

            clean_timeseries.to_csv(ts_path, sep='\t', index=False)


def _get_confounds(strategy_name, parameters, img):
    if strategy_name == 'baseline':
        reduced_confounds, sample_mask = load_confounds(img, **parameters)
    else:
        reduced_confounds, sample_mask = load_confounds_strategy(img, **parameters)
    return reduced_confounds, sample_mask


def _check_exclusion(reduced_confounds, sample_mask):
    if sample_mask is not None:
        kept_vol = len(sample_mask) / reduced_confounds.shape[0]
        remove = 1 - kept_vol
    else:
        remove = 0
    remove = remove > 0.2
    return remove


def generate_raw_timeseries(rawts_path, data, raw_masker):
    if not rawts_path.is_file():
        subject_timeseries = raw_masker.fit_transform(data.func[0])
        df = pd.DataFrame(subject_timeseries)
        df.to_csv(rawts_path, sep='\t', index=False)
    else:
        df = pd.read_csv(rawts_path, header=0, sep='\t')
        subject_timeseries = df.values
    del raw_masker
    return subject_timeseries


def _get_subject_info(output, data_aroma, data):
    img = data.func[0]

    subject_spec = data.func[0].split('/')[-1].split('_desc-')[0]
    subject_spec_aroma = data_aroma.func[0].split('/')[-1].split('_desc-')[0]

    subject_root = img.split(subject_spec)[0]
    subject_id = subject_spec.split('_')[0]

    subject_output = output / subject_id
    subject_output.mkdir(exist_ok=True)

    subject_mask = f"{subject_root}/{subject_spec}_desc-brain_mask.nii.gz"
    subject_mask_aroma = f"{subject_root}/{subject_spec_aroma}_desc-brain_mask.nii.gz"
    return subject_spec, subject_output, subject_mask, subject_mask_aroma


def _movement_summary(dataset_name, fmriprep_specifier, fmriprep_path, participant_tsv, output):
    if not Path(output / f"dataset-{dataset_name}_desc-movement_phenotype.tsv").is_file():
        data = fetch_fmriprep_derivative(dataset_name,
                                         participant_tsv, fmriprep_path,
                                         fmriprep_specifier)
        movement = phenotype_movement(data)
        movement = movement.sort_index()
        movement.to_csv( output / f"dataset-{dataset_name}_desc-movement_phenotype.tsv", sep='\t')
        print("Generate movement stats.")


def _get_prepro_strategy(strategy_name, strategy_file):
    # read the strategy deining files
    with open(strategy_file, "r") as file:
        benchmark_strategies = json.load(file)

    if strategy_name is None:
        print("Process all strategies.")
        strategy_names = [*benchmark_strategies]
    else:
        strategy_names = [strategy_name]
    return benchmark_strategies, strategy_names


if __name__ == "__main__":
    main()
