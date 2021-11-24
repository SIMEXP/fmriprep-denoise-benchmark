from pathlib import Path
import json

from utils.dataset import fetch_fmriprep_derivative, deconfound_connectome_single_strategy, ds000288_movement
from utils.atlas import create_atlas_masker

# define path of input and output
STRATEGY = "code/benchmark_strategies.json"
OUTPUT = "results"
INPUT_FMRIPREP = "inputs/dev_data"
INPUT_BIDS_PARTICIPANTS = "inputs/dev_data/participants.tsv"
FMRIPREP_SPECIFIER = "task-pixar"


def main():
    strategy_file = Path(__file__).parents[1] / STRATEGY
    input_fmriprep = Path(__file__).parents[1] / INPUT_FMRIPREP
    input_bids_participants = Path(__file__).parents[1] / INPUT_BIDS_PARTICIPANTS

    data = fetch_fmriprep_derivative(input_bids_participants, input_fmriprep,
                                     FMRIPREP_SPECIFIER)

    data_aroma = fetch_fmriprep_derivative(input_bids_participants, input_fmriprep,
                                           FMRIPREP_SPECIFIER, aroma=True)
    # read the strategy deining files
    with open(strategy_file, "r") as file:
        benchmark_strategies = json.load(file)

    movement = ds000288_movement(data)
    movement.to_csv("inputs/interim/dataset-ds000288_desc-movement_phenotype.tsv", sep='\t')

    atlas = create_atlas_masker("basc")
    for strategy_name, parameters in benchmark_strategies.items():
        strategy = {strategy_name: parameters}
        func_data = data_aroma.func if "aroma" in strategy_name else data.func
        output = Path(__file__).parents[1] / f"inputs/interim/{strategy_name}"
        output.mkdir(exist_ok=True)
        dataset_connectomes = deconfound_connectome_single_strategy(func_data, atlas[444]['masker'], strategy)
        dataset_connectomes.to_csv(output / "dataset-ds000288_atlas-BASC_nroi-444_data.tsv", sep='\t')

if __name__ == "__main__":
    main()
