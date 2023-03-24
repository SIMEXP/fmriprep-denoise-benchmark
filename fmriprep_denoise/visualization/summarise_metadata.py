"""
Summarise data for visualisation.
To retrievel atlas information, you need internet connection.
"""
import argparse
from pathlib import Path
import pandas as pd
from fmriprep_denoise.features.derivatives import get_qc_criteria
from fmriprep_denoise.visualization import utils
import itertools


qc_names = ["stringent", "minimal", None]
datasets = ["ds000228", "ds000030"]
fmriprep_versions = ["fmriprep-20.2.1lts", "fmriprep-20.2.5lts"]


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Summarise denoising metrics for visualization and save at the top level of the denoise metric outputs directory.",
    )
    parser.add_argument(
        "output_root", action="store", default=None, help="Output root path data."
    )
    parser.add_argument(
        "--fmriprep_version",
        action="store",
        choices=fmriprep_versions,
        type=str,
        help="Path to a fmriprep dataset.",
    )
    parser.add_argument(
        "--dataset_name",
        action="store",
        choices=datasets,
        type=str,
        help="Dataset name.",
    )
    parser.add_argument(
        "--qc",
        action="store",
        choices=qc_names,
        default=None,
        help="Automatic motion QC thresholds.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(vars(args))
    input_root = (
        utils.get_data_root() / "denoise-metrics"
        if args.output_root is None
        else Path(args.output_root)
    )
    output_root = (
        utils.get_data_root() / "denoise-metrics"
        if args.output_root is None
        else Path(output_root)
    )
    qc_name = args.qc
    dataset = args.dataset_name
    fmriprep_version = args.fmriprep_version

    for qc_name, dataset, fmriprep_version in itertools.product(
        *[qc_names, datasets, fmriprep_versions]
    ):
        qc = get_qc_criteria(qc_name)
        if qc_name is None:
            qc_name = "None"

        print(f"Processing {dataset}, {qc_name}, {fmriprep_version}...")
        ds_modularity = utils.prepare_modularity_plotting(
            dataset, fmriprep_version, None, None, input_root, qc
        )
        ds_qcfc = utils.prepare_qcfc_plotting(
            dataset, fmriprep_version, None, None, input_root
        )
        data = pd.concat([ds_qcfc, ds_modularity], axis=1)
        data.to_csv(
            output_root
            / f"{dataset}_{fmriprep_version.replace('.', '-')}_desc-{qc_name}_summary.tsv",
            sep="\t",
        )


if __name__ == "__main__":
    main()
