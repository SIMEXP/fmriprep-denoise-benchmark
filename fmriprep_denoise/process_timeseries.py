import argparse
from pathlib import Path
import json

import pandas as pd

from nilearn.signal import clean
from nilearn.interfaces.fmriprep import load_confounds_strategy, load_confounds

from fmriprep_denoise.utils.preprocess import fetch_fmriprep_derivative, phenotype_movement, _get_prepro_strategy
from fmriprep_denoise.utils.atlas import create_atlas_masker, ATLAS_METADATA


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
    benchmark_strategies, strategy_names = _get_prepro_strategy(strategy_name, strategy_file)
    dimensions = _get_atlas_dimensions(atlas_name)
    _movement_summary(dataset_name, fmriprep_specifier, fmriprep_path, participant_tsv, output)

    data_aroma = fetch_fmriprep_derivative(dataset_name,
                                           participant_tsv, fmriprep_path,
                                           fmriprep_specifier, subject=subject, aroma=True)
    data = fetch_fmriprep_derivative(dataset_name,
                                     participant_tsv, fmriprep_path,
                                     fmriprep_specifier, subject=subject)
    output = output / f"atlas-{atlas_name}"
    output.mkdir(exist_ok=True)

    for dimension in dimensions:
        print(f"-- {atlas_name}: dimension {dimension} --")
        atlas_spec = f"atlas-{atlas_name}_nroi-{dimension}"
        print("raw time series")
        atlas_info = {"atlas_name":atlas_name,
                      "dimension":dimension}
        subject_timeseries = _generate_raw_timeseries(output, data, atlas_info)

        for strategy_name in strategy_names:
            parameters = benchmark_strategies[strategy_name]
            print(f"Denoising: {strategy_name}")
            print(parameters)
            if _is_aroma(strategy_name):
                subject_mask, img, ts_path = _get_output_info(strategy_name, output, data_aroma, atlas_spec)
            else:
                subject_mask, img, ts_path = _get_output_info(strategy_name, output, data, atlas_spec)

            if ts_path.is_file():
                continue

            reduced_confounds, sample_mask = _get_confounds(strategy_name, parameters, img)
            if _is_aroma(strategy_name):
                aroma_masker, _ = create_atlas_masker(atlas_name, dimension, subject_mask, nilearn_cache="")
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


def _generate_raw_timeseries(output, data, atlas_info):
    subject_spec, subject_output, subject_mask = _get_subject_info(output, data)
    rawts_path = subject_output / f"{subject_spec}_atlas-{atlas_info['atlas_name']}_nroi-{atlas_info['dimension']}_desc-raw_timeseries.tsv"
    raw_masker, atlas_labels = create_atlas_masker(atlas_info['atlas_name'],
                                                   atlas_info['dimension'],
                                                   subject_mask,
                                                   detrend=False,
                                                   nilearn_cache="")
    timeseries_labels = pd.DataFrame(columns=atlas_labels)
    if not rawts_path.is_file():
        subject_timeseries = raw_masker.fit_transform(data.func[0])
        df = pd.DataFrame(subject_timeseries, columns=raw_masker.labels_)
        # make sure missing label were put pack
        df = pd.concat([timeseries_labels, df])
        df.to_csv(rawts_path, sep='\t', index=False)
    else:
        df = pd.read_csv(rawts_path, header=0, sep='\t')
        subject_timeseries = df.values
    del raw_masker
    return subject_timeseries


def _is_aroma(strategy_name):
    return "aroma" in strategy_name


def _get_output_info(strategy_name, output, data, atlas_spec):
    subject_spec, subject_output, subject_mask = _get_subject_info(output, data)
    img = data.func[0]
    ts_path = subject_output / f"{subject_spec}_{atlas_spec}_desc-{strategy_name}_timeseries.tsv"
    return subject_mask,img,ts_path


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


def _get_subject_info(output, data):
    img = data.func[0]

    subject_spec = data.func[0].split('/')[-1].split('_desc-')[0]

    subject_root = img.split(subject_spec)[0]
    subject_id = subject_spec.split('_')[0]

    subject_output = output / subject_id
    subject_output.mkdir(exist_ok=True)

    subject_mask = f"{subject_root}/{subject_spec}_desc-brain_mask.nii.gz"
    return subject_spec, subject_output, subject_mask


def _movement_summary(dataset_name, fmriprep_specifier, fmriprep_path, participant_tsv, output):
    "Save mean FD, sex and age info."
    if not Path(output / f"dataset-{dataset_name}_desc-movement_phenotype.tsv").is_file():
        data = fetch_fmriprep_derivative(dataset_name,
                                         participant_tsv, fmriprep_path,
                                         fmriprep_specifier)
        movement = phenotype_movement(data)
        movement = movement.sort_index()
        movement.to_csv( output / f"dataset-{dataset_name}_desc-movement_phenotype.tsv", sep='\t')
        print("Generate movement stats.")


def _get_atlas_dimensions(atlas_name):
    return ATLAS_METADATA[atlas_name]['dimensions']


if __name__ == "__main__":
    main()
