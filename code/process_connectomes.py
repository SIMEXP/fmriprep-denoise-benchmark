import argparse
from pathlib import Path
import json

from utils.dataset import fetch_fmriprep_derivative, deconfound_connectome_single_strategy, ds000288_movement
from utils.atlas import create_atlas_masker


# define path of input and output
STRATEGY = "{home}/projects/rrg-pbellec/hwang1/fmriprep-denoise-benchmark/code/benchmark_strategies.json"
INPUT_FMRIPREP = "{home}/scratch/test_data/1637790137/fmriprep"
INPUT_BIDS_PARTICIPANTS = "{home}/projects/rrg-pbellec/hwang1/test_data/participants.tsv"
ATLAS = 'difumo'
NROI = None


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Generate connectome based on denoising strategy for ds000288.",
    )
    parser.add_argument(
        "output_path",
        action="store",
        type=str,
        help="output path for connectome."
    )
    parser.add_argument(
        "--atlas",
        default=ATLAS,
        type=str,
        help="Atlas name (currently only support difumo)"
    )
    parser.add_argument(
        "--dimension",
        default=NROI,
        help="Number of ROI (currently only support difumo). Process all resolution if None.",
    )
    parser.add_argument(
        "--strategy-name",
        action="store",
        default=None,
        help="Denoise strategy name (see benchmark_strategies.json). Process all strategy if None."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(vars(args))
    strategy_name = args.strategy_name
    atlas_name = args.atlas
    nroi = args.dimension

    fmriprep_specifier = "task-pixar"

    home = str(Path.home())
    strategy_file = Path(STRATEGY.format_map({'home': home}))
    input_fmriprep = Path(INPUT_FMRIPREP.format_map({'home': home}))
    input_bids_participants = Path(INPUT_BIDS_PARTICIPANTS.format_map({'home': home}))

    output = Path(args.output_path)
    output.mkdir(exist_ok=True)

    data = fetch_fmriprep_derivative(input_bids_participants, input_fmriprep,
                                     fmriprep_specifier)

    data_aroma = fetch_fmriprep_derivative(input_bids_participants, input_fmriprep,
                                           fmriprep_specifier, aroma=True)
    if not Path(output / "dataset-ds000288_desc-movement_phenotype.tsv").is_file():
        movement = ds000288_movement(data)
        movement.to_csv( output / "dataset-ds000288_desc-movement_phenotype.tsv", sep='\t')

    # read the strategy deining files
    with open(strategy_file, "r") as file:
        benchmark_strategies = json.load(file)
    if strategy_name is None:
        print("Process all strategies.")
        strategy_names = [*benchmark_strategies]
    else:
        strategy_names = [strategy_name]

    atlas = create_atlas_masker(atlas_name)
    resolutions = atlas['resolutions'] if nroi is None else [int(nroi)]
    for nroi in resolutions:
        print(f"-- {atlas_name}: dimension {nroi} --")
        for name in strategy_names:
            parameters = benchmark_strategies[name]
            print(f"Denoising: {name}")
            strategy = {name: parameters}
            func_data = data_aroma.func if "aroma" in name else data.func
            dataset_connectomes = deconfound_connectome_single_strategy(func_data, atlas[nroi]['masker'], strategy)
            dataset_connectomes.to_csv(output / f"dataset-ds000288_atlas-{atlas_name}_nroi-{nroi}_desc-{name}_data.tsv", sep='\t')


if __name__ == "__main__":
    main()
