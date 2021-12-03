import tarfile
from pathlib import Path
import json

from utils.dataset import fetch_fmriprep_derivative, deconfound_connectome_single_strategy, ds000288_movement
from utils.atlas import create_atlas_masker

# define path of input and output
STRATEGY = "./code/benchmark_strategies.json"
OUTPUT = "inputs/interim/dataset-ds000288/"
INPUT_FMRIPREP = "{home}/scratch/test_data/1637790137/fmriprep"
INPUT_BIDS_PARTICIPANTS = "{home}/projects/rrg-pbellec/hwang1/test_data/participants.tsv"
FMRIPREP_SPECIFIER = "task-pixar"
ATLAS = 'difumo'


def main():
    strategy_file = Path(__file__).parents[1] / STRATEGY
    home = str(Path.home())
    input_fmriprep = Path(INPUT_FMRIPREP.format_map({'home': home}))
    input_bids_participants = Path(INPUT_BIDS_PARTICIPANTS.format_map({'home': home}))
    output = Path(__file__).parents[1] /OUTPUT
    output.mkdir(exist_ok=True)

    data = fetch_fmriprep_derivative(input_bids_participants, input_fmriprep,
                                     FMRIPREP_SPECIFIER)

    data_aroma = fetch_fmriprep_derivative(input_bids_participants, input_fmriprep,
                                           FMRIPREP_SPECIFIER, aroma=True)
    # read the strategy deining files
    with open(strategy_file, "r") as file:
        benchmark_strategies = json.load(file)

    movement = ds000288_movement(data)
    movement.to_csv( output / "dataset-ds000288_desc-movement_phenotype.tsv", sep='\t')

    atlas = create_atlas_masker(ATLAS)
    for nroi in atlas:
        print(f"-- {ATLAS}: dimension {nroi} --")
        for strategy_name, parameters in benchmark_strategies.items():
            print(f"Denoising: {strategy_name}")
            strategy = {strategy_name: parameters}
            func_data = data_aroma.func[:2] if "aroma" in strategy_name else data.func[:2]
            dataset_connectomes = deconfound_connectome_single_strategy(func_data, atlas[nroi]['masker'], strategy)
            dataset_connectomes.to_csv(output / f"dataset-ds000288_atlas-{ATLAS}_nroi-{nroi}_desc-{strategy_name}_data.tsv", sep='\t')

    with tarfile.open(output / f"dataset-ds000288_atlas-{ATLAS}.tar.gz", "w:gz") as tar:
        tar.add(output, arcname=output.name)


if __name__ == "__main__":
    main()
