import argparse
from pathlib import Path
import json

import pandas as pd
import numpy as np

from nilearn.connectome import ConnectivityMeasure

from utils.dataset import fetch_fmriprep_derivative, subject_timeseries, ds000288_movement
from utils.atlas import create_atlas_masker


# define path of input and output
STRATEGY = "{home}/projects/rrg-pbellec/hwang1/fmriprep-denoise-benchmark/preprocess/benchmark_strategies.json"
INPUT_FMRIPREP = "{home}/scratch/test_data/1637790137/fmriprep"
INPUT_BIDS_PARTICIPANTS = "{home}/projects/rrg-pbellec/hwang1/test_data/participants.tsv"
ATLAS = 'schaefer7networks'
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
        print("Generate movement stats.")

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
            dataset_connectomes = pd.DataFrame()
            for img in func_data:
                subject_id, subject_mask, ts_path = _parse_subject_info(atlas_name, nroi, output, img)
                masker = atlas[nroi]['masker']
                masker = masker.set_params(mask_img=subject_mask)
                subject_ts = subject_timeseries(img, masker, strategy, parameters)

                print(subject_ts.shape)
                if isinstance(subject_ts, pd.DataFrame):  # save time series
                    subject_conn = _compute_connectome(subject_id, subject_ts)
                    dataset_connectomes = pd.concat((dataset_connectomes, subject_conn), axis=0)
                else:
                    subject_ts = pd.DataFame()
                    dataset_connectomes.loc[subject_id, :] = np.nan
                # save timeseries
                subject_ts.to_csv(ts_path, sep='\t', index=False)
            output_connectome = output / f"dataset-ds000288_atlas-{atlas_name}_nroi-{nroi}_desc-{name}_data.tsv"
            dataset_connectomes.to_csv(output_connectome, sep='\t')


def _compute_connectome(subject_id, subject_ts):
    correlation_measure = ConnectivityMeasure(kind='correlation',
                                              vectorize=True,
                                              discard_diagonal=True)
    subject_conn = correlation_measure.fit_transform([subject_ts])
    subject_conn = pd.DataFrame(subject_conn, index=[subject_id])
    return subject_conn


def _parse_subject_info(atlas_name, nroi, output, img):
    subject_spec = img.split('/')[-1].split('_desc-')[0]
    subject_root = img.split(subject_spec)[0]
    subject_id = subject_spec.split('_')[0]
    subject_output = output / subject_id
    subject_output.mkdir(exist_ok=True)
    ts_path = subject_output / f"{subject_spec}_desc-{atlas_name}{nroi}_timeseries.tsv"
    subject_mask = f"{subject_root}/{subject_spec}_desc-brain_mask.nii.gz"
    return subject_id, subject_mask, ts_path


if __name__ == "__main__":
    main()