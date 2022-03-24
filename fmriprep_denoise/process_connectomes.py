import os
import argparse
from pathlib import Path
import json

import pandas as pd

from nilearn.connectome import ConnectivityMeasure

from fmriprep_denoise.utils.dataset import fetch_fmriprep_derivative, subject_timeseries, phenotype_movement
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


    if not Path(output / f"dataset-{dataset_name}_desc-movement_phenotype.tsv").is_file():
        data = fetch_fmriprep_derivative(dataset_name,
                                         participant_tsv, fmriprep_path,
                                         fmriprep_specifier)
        movement = phenotype_movement(data)
        movement = movement.sort_index()
        movement.to_csv( output / f"dataset-{dataset_name}_desc-movement_phenotype.tsv", sep='\t')
        print("Generate movement stats.")
    data_aroma = fetch_fmriprep_derivative(dataset_name,
                                           participant_tsv, fmriprep_path,
                                           fmriprep_specifier, subject=subject, aroma=True)
    data = fetch_fmriprep_derivative(dataset_name,
                                     participant_tsv, fmriprep_path,
                                     fmriprep_specifier, subject=subject)
    benchmark_strategies, strategy_names = _get_prepro_strategy(strategy_name, strategy_file)

    dimensions = get_atlas_dimensions(atlas_name)
    for dimension in dimensions:
        print(f"-- {atlas_name}: dimension {dimension} --")
        atlas_spec = f"atlas-{atlas_name}_nroi-{dimension}"
        for name in strategy_names:
            parameters = benchmark_strategies[name]
            print(f"Denoising: {name}")
            print(parameters)
            func_data = data_aroma.func if "aroma" in name else data.func
            if name == 'baseline':
                _dataset_timeseries(output, parameters, "raw", func_data, atlas_spec, atlas_name, dimension)
            _dataset_timeseries(output, parameters, name, func_data, atlas_spec, atlas_name, dimension)


def _dataset_timeseries(output, parameters, strategy, func_data, atlas_spec, atlas_name, dimension):
    for img in func_data:
        _, subject_mask, ts_path = _parse_subject_info(output, img, strategy, atlas_spec)
        cur_masker, _ = create_atlas_masker(atlas_name, dimension, subject_mask, nilearn_cache="")
        if not Path(ts_path).is_file():
            subject_ts = subject_timeseries(img, cur_masker, strategy, parameters)
            # save timeseries
            if subject_ts is not None:
                subject_ts.to_csv(ts_path, sep='\t', index=False)
            else:
                pd.DataFrame().to_csv(ts_path, sep='\t', index=False)


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


def _parse_subject_info(output, img, strategy, atlas_spec):
    subject_spec = img.split('/')[-1].split('_desc-')[0]
    subject_root = img.split(subject_spec)[0]
    subject_id = subject_spec.split('_')[0]
    subject_output = output / subject_id
    subject_output.mkdir(exist_ok=True)
    ts_path = subject_output / f"{subject_spec}_{atlas_spec}_desc-{strategy}_timeseries.tsv"
    subject_mask = f"{subject_root}/{subject_spec}_desc-brain_mask.nii.gz"
    return subject_id, subject_mask, ts_path


if __name__ == "__main__":
    main()
