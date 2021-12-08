"""Prototype for the denoising project.

The dataset used is hard coded for now.
To-do:

- plotting and reports
- euclidian distance calculation for parcel centre of mass
- write dataset fetcher for fmriprep output and atlas (if not in nilearn)
  mimicing the nilearn API?
- consider other metrics - perhaps look at Parker 2018

"""
import tarfile
import io
from pathlib import Path
import pandas as pd

from metrics import qcfc, compute_pairwise_distance


# define path of input and output
OUTPUT = "inputs/interim"
INPUT = "inputs/dataset-ds000288.tar.gz"
CENTROIDS = "inputs/atlas/schaefer20187networks/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv"


def main():
    """Main function."""
    output = Path(__file__).parents[1] / OUTPUT
    input_connectomes = Path(__file__).parents[1] / INPUT
    input_centroids = Path(__file__).parents[1] / CENTROIDS

    # pairwise distance
    rsa_centroids = pd.read_csv(input_centroids)
    rsa_centroids = rsa_centroids.loc[:, ['R', 'S', 'A']].values
    metrics = compute_pairwise_distance(rsa_centroids)

    with tarfile.open(input_connectomes, 'r:gz') as tar:
        movement = tar.extractfile("dataset-ds000288/dataset-ds000288_desc-movement_phenotype.tsv").read()
        movement = pd.read_csv(io.BytesIO(movement),
                               sep='\t', index_col=0, header=0, encoding='utf8')

        # find the strategies we need to iterate through.
        benchmark_strategies = []
        for member in tar.getmembers():
            filename = member.name.split('/')[-1]
            if "data.tsv" in filename:
                strategy = filename.split("desc-")[-1].split("_data")[0]
                benchmark_strategies.append(strategy)

        for strategy_name in benchmark_strategies:
            print(strategy_name)
            connectome = tar.extractfile(f"dataset-ds000288/atlas-schaefer7networks/dataset-ds000288_atlas-schaefer7networks_nroi-400_desc-{strategy_name}_data.tsv").read()
            dataset_connectomes = pd.read_csv(io.BytesIO(connectome), sep='\t', index_col=0, header=0)
            # QC-FC per edge
            metric = qcfc(movement, dataset_connectomes)
            metric = pd.DataFrame(metric)
            metric.columns = [f'{strategy_name}_{col}' for col in metric.columns]
            metrics = pd.concat((metrics, metric), axis=1)
    metrics.to_csv(
        output
        / "dataset-ds000288_atlas-schaefer7networks_nroi-400_desc-qcfc.tsv",
        sep='\t',
    )

if __name__ == "__main__":
    main()
