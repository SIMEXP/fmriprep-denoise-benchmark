"""
Process fMRIPrep outputs to timeseries based on denoising strategy.
"""


import argparse
from pathlib import Path

from fmriprep_denoise.data.fmriprep import (get_prepro_strategy,
                                           fetch_fmriprep_derivative,
                                           generate_movement_summary)
from fmriprep_denoise.data.timeseries import generate_timeseries_per_dimension


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
        action="store",
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
    output_root = Path(args.output_path)

    ts_output = output_root / f"atlas-{atlas_name}"
    ts_output.mkdir(exist_ok=True, parents=True)

    if not Path(ts_output / f"dataset-{dataset_name}_desc-movement_phenotype.tsv").is_file():
        full_data = fetch_fmriprep_derivative(dataset_name,
                                              participant_tsv, fmriprep_path,
                                              fmriprep_specifier)
        generate_movement_summary(dataset_name, full_data, output_root)

    benchmark_strategies = get_prepro_strategy(strategy_name)
    data_aroma = fetch_fmriprep_derivative(dataset_name,
                                           participant_tsv, fmriprep_path,
                                           fmriprep_specifier, subject=subject,
                                           aroma=True)
    data = fetch_fmriprep_derivative(dataset_name,
                                     participant_tsv, fmriprep_path,
                                     fmriprep_specifier, subject=subject)

    generate_timeseries_per_dimension(atlas_name, ts_output,
                                      benchmark_strategies, data_aroma, data)


if __name__ == "__main__":
    main()
