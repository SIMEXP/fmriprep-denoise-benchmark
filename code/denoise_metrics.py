"""Prototype for the denoising project.

To-do:

- plotting and reports
- euclidian distance calculation for parcel centre of mass
- write dataset fetcher for fmriprep output and atlas (if not in nilearn)
  mimicing the nilearn API?
- consider other metrics - perhaps look at Parker 2018

"""
from os import sep
from pathlib import Path
import json
import pandas as pd

from metrics import quality_control_connectivity


# define path of input and output
STRATEGY = "code/benchmark_strategies.json"
OUTPUT = "inputs/interim"
INPUT_CONNECTOMES = "inputs/"


def main():
    """Main function."""
    strategy_file = Path(__file__).parents[1] / STRATEGY
    output = Path(__file__).parents[1] / OUTPUT
    input_connectomes = Path(__file__).parents[1] / INPUT_CONNECTOMES
    # read the strategy deining files
    with open(strategy_file, "r") as file:
        benchmark_strategies = json.load(file)

    movement = pd.read_csv(input_connectomes / "dataset-ds000288_desc-movement_phenotype.tsv",
                           sep='\t', index_col=0, header=0)

    metric_per_edge, sig_per_edge = pd.DataFrame(), pd.DataFrame()

    for strategy_name in benchmark_strategies:
        print(strategy_name)

        dataset_connectomes = pd.read_csv(input_connectomes / f"dataset-ds000288_atlas-schaefer7networks_nroi-100_desc-{strategy_name}_data.tsv",
                                          sep='\t', index_col=0, header=0)
        # QC-FC per edge
        cur_qc_fc, cur_sig = quality_control_connectivity(movement, dataset_connectomes)
        # dump the results
        metric_per_edge.loc[:, strategy_name] = cur_qc_fc
        sig_per_edge.loc[:, strategy_name] = cur_sig

    metric_per_edge.to_csv(output / "dataset-ds000288_qc-fc_metric_per_edge.tsv", sep='\t')
    sig_per_edge.to_csv(output / "dataset-ds000288_qc-fc_sig_per_edge.tsv", sep='\t')


if __name__ == "__main__":
    main()
