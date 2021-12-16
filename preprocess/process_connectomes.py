import argparse
from pathlib import Path
import json

import pandas as pd

from nilearn.connectome import ConnectivityMeasure

from fmriprep_denoise.utils.dataset import fetch_fmriprep_derivative, subject_timeseries, ds000228_movement
from fmriprep_denoise.utils.atlas import create_atlas_masker


# define path of input and output
STRATEGY = "{home}/projects/rrg-pbellec/hwang1/fmriprep-denoise-benchmark/preprocess/benchmark_strategies.json"
INPUT_FMRIPREP = "{home}/scratch/test_data/1637790137/fmriprep"
INPUT_BIDS_PARTICIPANTS = "{home}/projects/rrg-pbellec/hwang1/test_data/participants.tsv"
ATLAS = 'schaefer7networks'
NROI = None


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Generate connectome based on denoising strategy for ds000228.",
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
    args = parse_args()
    print(vars(args))
    strategy_name = args.strategy_name
    atlas_name = args.atlas
    nroi = args.dimension

    fmriprep_specifier = "task-pixar"  # ds000228 specific; will have to remove when refactoring

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
    _calculate_movement_stats(output, data)
    benchmark_strategies, strategy_names = _get_prepro_strategy(strategy_name, strategy_file)

    atlas = create_atlas_masker(atlas_name)
    output = output / f"atlas-{atlas_name}"
    output.mkdir(exist_ok=True)
    resolutions = atlas['resolutions'] if nroi is None else [int(nroi)]
    for nroi in resolutions:
        print(f"-- {atlas_name}: dimension {nroi} --")
        for name in strategy_names:
            parameters = benchmark_strategies[name]
            print(f"Denoising: {name}")
            print(parameters)
            func_data = data_aroma.func if "aroma" in name else data.func
            valid_subject_ts, valid_subject_id = _dataset_timeseries(nroi, output, atlas, name, parameters, name, func_data)
            dataset_connectomes = _compute_connectome(valid_subject_ts, valid_subject_id)
            dataset_connectomes = dataset_connectomes.sort_index()
            output_connectome = output / f"dataset-ds000228_atlas-{atlas_name}_nroi-{nroi}_desc-{name}_data.tsv"
            dataset_connectomes.to_csv(output_connectome, sep='\t')


def _dataset_timeseries(nroi, output, atlas, name, parameters, strategy, func_data):
    valid_subject_ts = []
    valid_subject_id = []
    for img in func_data:
        subject_id, subject_mask, ts_path = _parse_subject_info(output, img, name)
        subject_ts = _get_timeseries(nroi, atlas, parameters, strategy, img, subject_mask, ts_path)
        if isinstance(subject_ts, pd.DataFrame):
            valid_subject_ts.append(subject_ts.values)
            valid_subject_id.append(subject_id)
    return valid_subject_ts,valid_subject_id


def _calculate_movement_stats(output, data):
    if not Path(output / "dataset-ds000228_desc-movement_phenotype.tsv").is_file():
        movement = ds000228_movement(data)
        movement = movement.sort_index()
        movement.to_csv( output / "dataset-ds000228_desc-movement_phenotype.tsv", sep='\t')
        print("Generate movement stats.")


def _get_prepro_strategy(strategy_name, strategy_file):
    # read the strategy deining files
    with open(strategy_file, "r") as file:
        benchmark_strategies = json.load(file)

    if strategy_name is None:
        print("Process all strategies.")
        strategy_names = [*benchmark_strategies]
    else:
        strategy_names = [strategy_name]
    return benchmark_strategies, strategy_names


def _get_timeseries(nroi, atlas, parameters, strategy, img, subject_mask, ts_path):
    if not Path(ts_path).is_file():
        masker = atlas[nroi]['masker']
        masker = masker.set_params(mask_img=subject_mask)
        subject_ts = subject_timeseries(img, masker, strategy, parameters)
        # save timeseries
        if subject_ts is not None:
            subject_ts.to_csv(ts_path, sep='\t', index=False)
        else:
            pd.DataFrame().to_csv(ts_path, sep='\t', index=False)
        return subject_ts
    else:
        return pd.read_csv(ts_path, header=0, sep='\t')


def _compute_connectome(valid_subject_ts, valid_subject_id):
    correlation_measure = ConnectivityMeasure(kind='correlation',
                                              vectorize=True,
                                              discard_diagonal=True)
    subject_conn = correlation_measure.fit_transform(valid_subject_ts)
    subject_conn = pd.DataFrame(subject_conn, index=valid_subject_id)
    return subject_conn


def _parse_subject_info(output, img, name):
    subject_spec = img.split('/')[-1].split('_desc-')[0]
    subject_root = img.split(subject_spec)[0]
    subject_id = subject_spec.split('_')[0]
    subject_output = output / subject_id
    subject_output.mkdir(exist_ok=True)
    ts_path = subject_output / f"{subject_spec}_desc-{name}_timeseries.tsv"
    subject_mask = f"{subject_root}/{subject_spec}_desc-brain_mask.nii.gz"
    return subject_id, subject_mask, ts_path


if __name__ == "__main__":
    main()
