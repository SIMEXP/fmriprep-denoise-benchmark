import argparse

import pandas as pd

from pathlib import Path
from joblib import Parallel, delayed

from fmriprep_denoise.dataset.fmriprep import get_prepro_strategy
from fmriprep_denoise.features.derivatives import (
    compute_connectome,
    check_extraction,
    get_qc_criteria,
)
from fmriprep_denoise.features import louvain_modularity


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
    print(motion_qc)

    metric_mod = []
    for strategy_name in strategy_names.keys():
        print(strategy_name)
        file_pattern = f'atlas-{atlas}_nroi-{dimension}_desc-{strategy_name}'

        connectome, phenotype = compute_connectome(
            atlas,
            extracted_path,
            dataset,
            path_root,
            file_pattern,
            **motion_qc,
        )
        print('\tLoaded connectome...')

        qs = Parallel(n_jobs=4)(
            delayed(louvain_modularity)(vect)
            for vect in connectome.values.tolist())

        # if dataset == 'ds000228' and atlas in ['mist', 'schaefer7networks']:
        #     if dimension in ['800', '325', '444']:
        #         print('\t\tmultiprocesspr not working.')
        #         qs = [louvain_modularity(vect) for vect in connectome.values.tolist()]
        # else:
        #     with Pool(8) as pool:
        #         qs = pool.map(louvain_modularity, connectome.values.tolist())

        modularity = pd.DataFrame(
            qs, columns=[strategy_name], index=connectome.index
        )
        metric_mod.append(modularity)
        print('\tModularity...')

    metric_mod = pd.concat(metric_mod, axis=1)
    metric_mod.to_csv(
        output_path
        / f'dataset-{dataset}_atlas-{atlas}_nroi-{dimension}_modularity.tsv',
        sep='\t',
    )


if __name__ == '__main__':
    main()
