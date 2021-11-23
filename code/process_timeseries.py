from pathlib import Path
import json
import pandas as pd
from nilearn import datasets
from nilearn.input_data import NiftiMapsMasker
from utils.dataset import fetch_fmriprep_derivative, deconfound_connectome_single_strategy, ds000288_movement
from utils.atlas import create_atlas_masker

# define path of input and output
STRATEGY = "code/benchmark_strategies.json"
ATLAS = "code/atlas_metadata.json"
OUTPUT = "results"
INPUT_FMRIPREP = "inputs/dev_data"
INPUT_BIDS_PARTICIPANTS = "inputs/dev_data/participants.tsv"
FMRIPREP_SPECIFIER = "task-pixar"


atlas_metadata = Path(__file__).parents[1] / ATLAS
strategy_file = Path(__file__).parents[1] / STRATEGY
input_fmriprep = Path(__file__).parents[1] / INPUT_FMRIPREP
input_bids_participants = Path(__file__).parents[1] / INPUT_BIDS_PARTICIPANTS

data = fetch_fmriprep_derivative(input_bids_participants, input_fmriprep,
                                    FMRIPREP_SPECIFIER)


# read the strategy deining files
with open(strategy_file, "r") as file:
    benchmark_strategies = json.load(file)

movement = ds000288_movement(data)
movement.to_csv("inputs/interim/dataset-ds000288_desc-movement_phenotype.tsv", sep='\t')

atlas = create_atlas_masker("mist")
print(atlas[444]['masker'])
for strategy_name, parameters in benchmark_strategies.items():
    strategy = {strategy_name: parameters}
    print(strategy_name)
    dataset_connectomes = deconfound_connectome_single_strategy(data.func, atlas[444]['masker'], strategy)
    print(dataset_connectomes.shape)