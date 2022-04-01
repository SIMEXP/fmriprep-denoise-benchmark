import argparse

import pandas as pd

from pathlib import Path
from multiprocessing import Pool

from fmriprep_denoise.metrics import qcfc, louvain_modularity
from fmriprep_denoise.utils.preprocess import _get_prepro_strategy
from fmriprep_denoise.utils.dataset import load_phenotype, load_valid_timeseries, compute_connectome, check_extraction


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Generate denoise metric based on denoising strategy for ds000228.",
    )
    parser.add_argument(
        "input_path",
        action="store",
        type=str,
        help="input path for .gz dataset."
    )
    parser.add_argument(
        "output_path",
        action="store",
        type=str,
        help="output path for metrics."
    )
    parser.add_argument(
        "--atlas",
        action="store",
        type=str,
        help="Atlas name (schaefer7networks, mist, difumo, gordon333)"
    )
    parser.add_argument(
        "--dimension",
        action="store",
        help="Number of ROI. See meta data of each atlas to get valid inputs.",
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
    """Main function."""
    args = parse_args()
    print(vars(args))
    input_gz = Path(args.input_path)
    strategy_name = args.strategy_name
    atlas = args.atlas
    dimension = args.dimension
    output_path = Path(args.output_path) / "metrics"
    output_path.mkdir(exist_ok=True)

    extracted_path = check_extraction(input_gz, extracted_path_root=None)
    dataset = extracted_path.name.split('-')[-1]
    phenotype = load_phenotype(dataset=dataset)
    participant_id = phenotype.index.to_list()

    strategy_file = Path(__file__).parent / "benchmark_strategies.json"
    _, strategy_names = _get_prepro_strategy(strategy_name, strategy_file)

    for strategy_name in strategy_names:
        print(strategy_name)
        file_pattern = f"atlas-{atlas}_nroi-{dimension}_desc-{strategy_name}"

        valid_ids, valid_ts = load_valid_timeseries(atlas, extracted_path,
                                                    participant_id, file_pattern)
        connectome = compute_connectome(valid_ids, valid_ts)
        print("Loaded connectome...")

        metric = qcfc(phenotype.loc[:, 'mean_framewise_displacement'],
                      connectome,
                      phenotype.loc[:, ['age', 'gender']])
        metric = pd.DataFrame(metric)
        metric.columns = [f'{strategy_name}_{col}' for col in metric.columns]
        metric.to_csv(
            output_path
            / f"dataset-{dataset}_atlas-{atlas}_nroi-{dimension}_desc-{strategy_name}_qcfc.tsv",
            sep='\t',
        )

        print("QC-FC...")
        with Pool(30) as pool:
            qs = pool.map(louvain_modularity, connectome.values.tolist())

        modularity = pd.DataFrame(qs,
                                  columns=[strategy_name],
                                  index=connectome.index)
        print("Modularity...")
        modularity.to_csv(
            output_path
            / f"dataset-{dataset}_atlas-{atlas}_nroi-{dimension}_desc-{strategy_name}_modularity.tsv",
            sep='\t',
        )

if __name__ == "__main__":
    main()
