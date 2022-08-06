import argparse

import pandas as pd

from pathlib import Path
from multiprocessing import Pool

from fmriprep_denoise.data.fmriprep import get_prepro_strategy
from fmriprep_denoise.features.derivatives import (
    compute_connectome,
    check_extraction,
    get_qc_criteria,
)
from fmriprep_denoise.features import qcfc


# another very bad special case handling
group_info_column = {'ds000228': 'Child_Adult', 'ds000030': 'diagnosis'}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Generate denoise metric based on denoising strategies.',
    )
    parser.add_argument(
        'input_path',
        action='store',
        type=str,
        help='input path for .gz dataset.',
    )
    parser.add_argument(
        'output_path',
        action='store',
        type=str,
        help='output path for metrics.',
    )
    parser.add_argument(
        '--atlas',
        action='store',
        type=str,
        help='Atlas name (schaefer7networks, mist, difumo, gordon333)',
    )
    parser.add_argument(
        '--dimension',
        action='store',
        help='Number of ROI. See meta data of each atlas to get valid inputs.',
    )
    parser.add_argument(
        '--qc',
        action='store',
        default=None,
        help='Automatic motion QC thresholds.',
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    print(vars(args))
    input_gz = Path(args.input_path)
    atlas = args.atlas
    dimension = args.dimension

    extracted_path = check_extraction(input_gz, extracted_path_root=None)
    print(extracted_path)
    dataset = extracted_path.name.split('-')[-1]
    path_root = Path(args.output_path).absolute()
    output_path = path_root / f'dataset-{dataset}'
    output_path.mkdir(exist_ok=True)

    strategy_names = get_prepro_strategy(None)
    motion_qc = get_qc_criteria(args.qc)

    metric_qcfc, metric_mod = [], []
    for strategy_name in strategy_names.keys():
        print(strategy_name)
        file_pattern = f'atlas-{atlas}_nroi-{dimension}_desc-{strategy_name}'

        connectome, phenotype = compute_connectome(
            atlas,
            extracted_path,
            dataset,
            path_root,
            file_pattern,
            gross_fd=motion_qc['gross_fd'],
            fd_thresh=motion_qc['fd_thresh'],
            proportion_thresh=motion_qc['proportion_thresh'],
        )
        print('\tLoaded connectome...')
        metric = qcfc(
            phenotype.loc[:, 'mean_framewise_displacement'],
            connectome,
            phenotype.loc[:, ['age', 'gender']],
        )
        metric = pd.DataFrame(metric)
        columns = [
            ('full_sample', f'{strategy_name}_{col}') for col in metric.columns
        ]
        columns = pd.MultiIndex.from_tuples(columns)
        metric.columns = columns
        metric_qcfc.append(metric)
        print('\tQC-FC...')

        # QC-FC by group
        groups = phenotype['groups'].unique()
        for group in groups:
            print(group)
            group_mask = phenotype['groups'] == group
            # make sure values are numerical
            subgroup = phenotype[group_mask].index
            metric = qcfc(
                phenotype.loc[subgroup, 'mean_framewise_displacement'],
                connectome.loc[subgroup, :],
                phenotype.loc[subgroup, ['age', 'gender']],
            )
            metric = pd.DataFrame(metric)
            metric.columns = [
                (group, f'{strategy_name}_{col}') for col in metric.columns
            ]
            metric_qcfc.append(metric)

    metric_qcfc = pd.concat(metric_qcfc, axis=1)
    metric_qcfc.to_csv(
        output_path
        / f'dataset-{dataset}_atlas-{atlas}_nroi-{dimension}_qcfc.tsv',
        sep='\t',
    )


if __name__ == '__main__':
    main()
