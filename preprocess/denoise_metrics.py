"""Prototype for the denoising project.

The dataset used is hard coded for now.
"""
import tarfile
import io
from pathlib import Path
import pandas as pd
from multiprocessing import Pool

from fmriprep_denoise.metrics import qcfc, louvain_modularity


# define path of input and output
OUTPUT = "inputs/"
INPUT = "inputs/dataset-ds000288_ohbm.tar.gz"
ATLAS = "gordon333"
RESOLUTION = 333
# ATLAS = "schaefer7networks"
# RESOLUTION = 400

def main():
    """Main function."""
    output = Path(__file__).parents[1] / OUTPUT
    input_connectomes = Path(__file__).parents[1] / INPUT

    metrics = pd.DataFrame()
    modularity = pd.DataFrame()
    # benchmark_strategies = ['baseline']
    benchmark_strategies = ['baseline',
                            'simple', 'simple+gsr',
                            'scrubbing.2', 'scrubbing.2+gsr',
                            'scrubbing.5', 'scrubbing.5+gsr',
                            'compcor', 'compcor6',
                            'aroma', 'aroma+gsr']

    with tarfile.open(input_connectomes, 'r:gz') as tar:
        movement = tar.extractfile(
            "dataset-ds000288/dataset-ds000288_desc-movement_phenotype.tsv").read()
        movement = pd.read_csv(io.BytesIO(movement),
                            sep='\t', index_col=0, header=0, encoding='utf8')
        movement = movement.sort_index()
        for strategy_name in benchmark_strategies:
            print(strategy_name)
            connectome = tar.extractfile(
                f"dataset-ds000288/atlas-{ATLAS}/dataset-ds000288_atlas-{ATLAS}_nroi-{RESOLUTION}_desc-{strategy_name}_data.tsv").read()
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
            metrics = pd.concat((metrics, metric), axis=1, join='outer')
            print("QC-FC...")
            with Pool(30) as pool:
                qs = pool.map(louvain_modularity, dataset_connectomes.values.tolist())
            modularity_index = pd.DataFrame(qs,
                                            columns=[strategy_name],
                                            index=dataset_connectomes.index)
            modularity = pd.concat((modularity, modularity_index), axis=1, join='outer')
            print("Modularity...")

    metrics.to_csv(
        output
        / f"dataset-ds000288_atlas-{ATLAS}_nroi-{RESOLUTION}_desc-qcfc_baseline.tsv",
        sep='\t',
    )
    modularity.to_csv(
        output
        / f"dataset-ds000288_atlas-{ATLAS}_nroi-{RESOLUTION}_desc-modularity_baseline.tsv",
        sep='\t',
    )

if __name__ == "__main__":
    main()
