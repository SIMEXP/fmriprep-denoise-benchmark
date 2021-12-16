"""Prototype for the denoising project.

The dataset used is hard coded for now.
"""
import argparse

import tarfile
import io
from pathlib import Path
import pandas as pd
from multiprocessing import Pool

from fmriprep_denoise.metrics import qcfc, louvain_modularity


# define path of input and output
INPUT = "inputs/dataset-ds000228.tar.gz"
ATLAS = "gordon333"
NROI = 333

STRATEGIES = ['baseline',
              'simple', 'simple+gsr',
              'scrubbing.2', 'scrubbing.2+gsr',
              'scrubbing.5', 'scrubbing.5+gsr',
              'compcor', 'compcor6',
              'aroma', 'aroma+gsr']

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Generate denoise metric based on denoising strategy for ds000228.",
    )
    parser.add_argument(
        "output_path",
        action="store",
        type=str,
        help="output path for metrics."
    )
    parser.add_argument(
        "--atlas",
        default=ATLAS,
        type=str,
        help="Atlas name (schaefer7networks, basc, difumo)"
    )
    parser.add_argument(
        "--dimension",
        default=NROI,
        help="Number of ROI. Process all resolution if None.",
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

    strategy_name = args.strategy_name
    atlas_name = args.atlas
    nroi = args.dimension

    output = Path(args.output_path)
    input_connectomes = Path(__file__).parents[1] / INPUT

    if strategy_name is None:
        print("Process all strategies.")
        strategy_names = STRATEGIES
    elif strategy_name in STRATEGIES:
        strategy_names = [strategy_name]
    else:
        raise ValueError(f"Unsupported input {strategy_name}")

    with tarfile.open(input_connectomes, 'r:gz') as tar:
        movement = tar.extractfile(
            "dataset-ds000228/dataset-ds000228_desc-movement_phenotype.tsv").read()
        movement = pd.read_csv(io.BytesIO(movement),
                            sep='\t', index_col=0, header=0, encoding='utf8')
        movement = movement.sort_index()
        for strategy_name in strategy_names:
            print(strategy_name)
            connectome = tar.extractfile(
                f"dataset-ds000228/atlas-{ATLAS}/dataset-ds000228_atlas-{atlas_name}_nroi-{nroi}_desc-{strategy_name}_data.tsv").read()
            dataset_connectomes = pd.read_csv(io.BytesIO(connectome),
                                              sep='\t',
                                              index_col=0,
                                              header=0).sort_index()
            print("Loaded connectome...")
            metric = qcfc(movement.loc[:, 'mean_framewise_displacement'],
                          dataset_connectomes,
                          movement.loc[:, ['Age', 'Gender']])
            metric = pd.DataFrame(metric)
            metric.columns = [f'{strategy_name}_{col}' for col in metric.columns]
            print("QC-FC...")
            with Pool(30) as pool:
                qs = pool.map(louvain_modularity, dataset_connectomes.values.tolist())
            modularity = pd.DataFrame(qs,
                                      columns=[strategy_name],
                                      index=dataset_connectomes.index)
            print("Modularity...")

            metric.to_csv(
                output
                / f"dataset-ds000228_atlas-{atlas_name}_nroi-{nroi}_desc-{strategy_name}_qcfc.tsv",
                sep='\t',
            )
            modularity.to_csv(
                output
                / f"dataset-ds000228_atlas-{atlas_name}_nroi-{nroi}_desc-{strategy_name}_modularity.tsv",
                sep='\t',
            )

if __name__ == "__main__":
    main()
