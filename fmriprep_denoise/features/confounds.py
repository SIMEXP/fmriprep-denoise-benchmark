"""Calculate degree of freedom"""
import argparse
from pathlib import Path
import json

from fmriprep_denoise.data.timeseries import get_confounds
from fmriprep_denoise.data.fmriprep import (get_prepro_strategy,
                                            fetch_fmriprep_derivative)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Extract confound degree of freedome info.",
    )
    parser.add_argument(
        "output_path",
        action="store",
        type=str,
        help="output path data."
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
    return parser.parse_args()


def main():
    args = parse_args()
    print(vars(args))
    dataset_name = args.dataset_name
    fmriprep_specifier = args.specifier
    fmriprep_path = Path(args.fmriprep_path)
    participant_tsv = Path(args.participants_tsv)
    output_root = Path(args.output_path)

    path_dof = Path(output_root / f"dataset-{dataset_name}_desc-confounds_phenotype.tsv")
    if not path_dof.is_file():
        benchmark_strategies = get_prepro_strategy()
        data_aroma = fetch_fmriprep_derivative(dataset_name,
                                            participant_tsv, fmriprep_path,
                                            fmriprep_specifier,
                                            aroma=True)
        data = fetch_fmriprep_derivative(dataset_name,
                                        participant_tsv, fmriprep_path,
                                        fmriprep_specifier)
        info = {}
        for strategy_name, parameters in benchmark_strategies.items():
            print(f"Denoising: {strategy_name}")
            print(parameters)
            func_data = data_aroma.func if "aroma" in strategy_name else data.func
            for img in func_data:
                sub = img.split('/')[-1].split('_')[0]
                reduced_confounds, sample_mask = get_confounds(strategy_name,
                                                            parameters,
                                                            img)
                ts_length = reduced_confounds.shape[0] if sample_mask is None else len(sample_mask)
                excised_vol = reduced_confounds.shape[0] - ts_length
                aroma = 0
                compcor = 0
                if "aroma" in strategy_name:
                    path_aroma_ic = img.split('space-')[0] + 'AROMAnoiseICs.csv'
                    with open(path_aroma_ic, 'r') as f:
                        aroma = len(f.readline().split(','))
                compcor = sum('comp_cor' in i for i in regressors)
                regressors = reduced_confounds.columns.tolist()
                high_pass = sum('cosine' in i for i in regressors)
                partial = aroma + compcor
                if "compcor" not in strategy_name:
                    fixed = len(regressors)
                else:
                    fixed = len(regressors) - compcor

                stats = {
                    (strategy_name, 'excised_vol'): excised_vol,
                    (strategy_name, 'high_pass'): high_pass,
                    (strategy_name, 'fixed_regressors'): fixed,
                    (strategy_name, 'vary'): partial,
                    (strategy_name, 'total'): fixed + partial

                }
                if info.get(sub):
                    info[sub].update(stats)
                else:
                    info[sub] = stats
        import pandas as pd
        pd.DataFrame.from_dict(info, orient='index').to_csv(path_dof, sep='\t')

if __name__ == "__main__":
    main()