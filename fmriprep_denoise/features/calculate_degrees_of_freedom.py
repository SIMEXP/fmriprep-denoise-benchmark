"""Calculate degree of freedom"""
import argparse
from pathlib import Path
import pandas as pd


from fmriprep_denoise.data.timeseries import get_confounds
from fmriprep_denoise.data.fmriprep import (
    get_prepro_strategy,
    fetch_fmriprep_derivative,
    generate_movement_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Extract confound degree of freedome info.',
    )
    parser.add_argument(
        'output_path', action='store', type=str, help='output path data.'
    )
    parser.add_argument(
        '--fmriprep_path',
        action='store',
        type=str,
        help='Path to a fmriprep dataset.',
    )
    parser.add_argument(
        '--dataset_name', action='store', type=str, help='Dataset name.'
    )
    parser.add_argument(
        '--specifier',
        action='store',
        type=str,
        help=(
            'Text in a fmriprep file name, '
            'in between sub-<subject>_ses-<session>_and `space-<template>`.'
        ),
    )
    parser.add_argument(
        '--participants_tsv',
        action='store',
        type=str,
        help='Path to participants.tsv in the original BIDS dataset.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(vars(args))
    dataset_name = args.dataset_name
    fmriprep_specifier = args.specifier
    fmriprep_path = Path(args.fmriprep_path)
    participant_tsv = Path(args.participants_tsv)
    output_root = Path(args.output_path) / f'dataset-{dataset_name}'

    output_root.mkdir(exist_ok=True, parents=True)
    path_movement = Path(
        output_root
        / f'dataset-{dataset_name}_desc-movement_phenotype.tsv'
    )

    path_dof = Path(
        output_root
        / f'dataset-{dataset_name}_desc-confounds_phenotype.tsv'
    )

    if not path_movement.is_file():
        full_data = fetch_fmriprep_derivative(
            dataset_name, participant_tsv, fmriprep_path, fmriprep_specifier
        )
        generate_movement_summary(dataset_name, full_data, output_root)

    if not path_dof.is_file():
        benchmark_strategies = get_prepro_strategy()
        data_aroma = fetch_fmriprep_derivative(
            dataset_name,
            participant_tsv,
            fmriprep_path,
            fmriprep_specifier,
            aroma=True,
        )
        data = fetch_fmriprep_derivative(
            dataset_name, participant_tsv, fmriprep_path, fmriprep_specifier
        )
        info = {}
        for strategy_name, parameters in benchmark_strategies.items():
            print(f'Denoising: {strategy_name}')
            print(parameters)
            func_data = (
                data_aroma.func if 'aroma' in strategy_name else data.func
            )
            for img in func_data:
                sub = img.split('/')[-1].split('_')[0]
                reduced_confounds, sample_mask = get_confounds(
                    strategy_name, parameters, img
                )
                full_length = reduced_confounds.shape[0]
                ts_length = (
                    full_length if sample_mask is None else len(sample_mask)
                )
                excised_vol = full_length - ts_length
                excised_vol_pro = excised_vol / full_length
                regressors = reduced_confounds.columns.tolist()
                fixed = len(regressors)
                total = fixed
                aroma = 0
                compcor = 0
                if 'aroma' in strategy_name:
                    path_aroma_ic = (
                        img.split('space-')[0] + 'AROMAnoiseICs.csv'
                    )
                    with open(path_aroma_ic, 'r') as f:
                        aroma = len(f.readline().split(','))
                    aroma = fixed + aroma
                compcor = sum('comp_cor' in i for i in regressors)
                high_pass = sum('cosine' in i for i in regressors)
                if 'compcor' in strategy_name:
                    fixed = len(regressors) - compcor
                    compcor = fixed + compcor
                    total = compcor
                if 'aroma' in strategy_name:
                    aroma = fixed + aroma
                    total = aroma

                stats = {
                    (strategy_name, 'excised_vol'): excised_vol,
                    (strategy_name, 'excised_vol_proportion'): excised_vol_pro,
                    (strategy_name, 'high_pass'): high_pass,
                    (strategy_name, 'fixed_regressors'): fixed,
                    (strategy_name, 'compcor'): compcor,
                    (strategy_name, 'aroma'): aroma,
                    (strategy_name, 'total'): total,
                }
                if info.get(sub):
                    info[sub].update(stats)
                else:
                    info[sub] = stats
        confounds_stats = pd.DataFrame.from_dict(info, orient='index')
        confounds_stats = confounds_stats.sort_index()
        confounds_stats.to_csv(path_dof, sep='\t')


if __name__ == '__main__':
    main()
