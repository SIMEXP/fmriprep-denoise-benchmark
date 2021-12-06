"""Prototype for the denoising project.

To-do:

- plotting and reports
- euclidian distance calculation for parcel centre of mass
- write dataset fetcher for fmriprep output and atlas (if not in nilearn)
  mimicing the nilearn API?
- consider other metrics - perhaps look at Parker 2018

"""
import tarfile
import io
from os import sep
from pathlib import Path
import json
import pandas as pd

from metrics import quality_control_connectivity


# define path of input and output
STRATEGY = "code/benchmark_strategies.json"
OUTPUT = "inputs/interim"
INPUT = "inputs/dataset-ds000288.tar.gz"


def main():
    """Main function."""
    strategy_file = Path(__file__).parents[1] / STRATEGY
    output = Path(__file__).parents[1] / OUTPUT
    input_connectomes = Path(__file__).parents[1] / INPUT
    # read the strategy deining files
    with open(strategy_file, "r") as file:
        benchmark_strategies = json.load(file)

    with tarfile.open(input_connectomes, 'r:gz') as tar:
        movement = tar.extractfile('dataset-ds000288/dataset-ds000288_desc-movement_phenotype.tsv').read()
        movement = pd.read_csv(io.BytesIO(movement),
                               sep='\t', index_col=0, header=0, encoding='utf8')


    metric_per_edge, sig_per_edge = pd.DataFrame(), pd.DataFrame()

    for strategy_name in benchmark_strategies:
        print(strategy_name)
        with tarfile.open(input_connectomes, 'r:gz') as tar:
            connectome = tar.extractfile(f'dataset-ds000288/atlas-schaefer/dataset-ds000288_atlas-schaefer7networks_nroi-400_desc-{strategy_name}_data.tsv').read()
            dataset_connectomes = pd.read_csv(io.BytesIO(connectome), sep='\t', index_col=0, header=0)
        # QC-FC per edge
        cur_qc_fc, cur_sig = quality_control_connectivity(movement, dataset_connectomes)
        # dump the results
        metric_per_edge.loc[:, strategy_name] = cur_qc_fc
        sig_per_edge.loc[:, strategy_name] = cur_sig

    metric_per_edge.to_csv(output / "dataset-ds000288_qc-fc_metric_per_edge.tsv", sep='\t')
    sig_per_edge.to_csv(output / "dataset-ds000288_qc-fc_sig_per_edge.tsv", sep='\t')


if __name__ == "__main__":
    main()
