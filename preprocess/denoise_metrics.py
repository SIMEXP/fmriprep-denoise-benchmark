"""Prototype for the denoising project.

The dataset used is hard coded for now.
"""
import tarfile
import io
from pathlib import Path
import pandas as pd

from metrics import qcfc, compute_pairwise_distance


# define path of input and output
OUTPUT = "inputs/"
INPUT = "inputs/dataset-ds000288.tar.gz"
CENTROIDS = "inputs/atlas/schaefer20187networks/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv"


def main():
    """Main function."""
    output = Path(__file__).parents[1] / OUTPUT
    input_connectomes = Path(__file__).parents[1] / INPUT
    input_centroids = Path(__file__).parents[1] / CENTROIDS

    metrics = pd.DataFrame()
    metrics_cov = pd.DataFrame()

    benchmark_strategies = ['raw',
                            'simple', 'simple+gsr',
                            'scrubbing', 'scrubbing+gsr',
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
                f"dataset-ds000288/atlas-schaefer7networks/dataset-ds000288_atlas-schaefer7networks_nroi-400_desc-{strategy_name}_data.tsv").read()
            dataset_connectomes = pd.read_csv(io.BytesIO(connectome),
                                            sep='\t',
                                            index_col=0, header=0).sort_index()

            metric = qcfc(movement, dataset_connectomes, None)
            metric_cov = qcfc(movement, dataset_connectomes, ['Age', 'Gender'])

            metric = pd.DataFrame(metric)
            metric.columns = [f'{strategy_name}_{col}' for col in metric.columns]
            metrics = pd.concat((metrics, metric), axis=1, join='outer')

            metric_cov = pd.DataFrame(metric_cov)
            metric_cov.columns = [f'{strategy_name}_{col}' for col in metric_cov.columns]
            metrics_cov = pd.concat((metrics_cov, metric_cov), axis=1, join='outer')

    labels = pd.read_csv(input_centroids)
    pairwise_distance = compute_pairwise_distance(labels.loc[:, ['R', 'S', 'A']])

    metrics.to_csv(
        output
        / "dataset-ds000288_atlas-schaefer7networks_nroi-400_desc-qcfc.tsv",
        sep='\t',
    )
    metrics_cov.to_csv(
        output
        / "dataset-ds000288_atlas-schaefer7networks_nroi-400_desc-covqcfc.tsv",
        sep='\t',
    )
    pairwise_distance.to_csv(
        output
        / "atlas/schaefer20187networks/atlas-schaefer7networks_nroi-400_desc-distance.tsv",
        sep='\t',
        index=False
    )


if __name__ == "__main__":
    main()
