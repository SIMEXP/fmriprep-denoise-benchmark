import argparse
import logging
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

from fmriprep_denoise.dataset.fmriprep import get_prepro_strategy
from fmriprep_denoise.features.derivatives_test import (
    compute_connectome,
    get_qc_criteria,
    load_full_roi_list, #added for roi list return
)
from fmriprep_denoise.features import qcfc, louvain_modularity

import glob  # put this at the top of your script if not already

# another very bad special case handling
group_info_column = {"ds000228": "Child_Adult", "ds000030": "diagnosis"}

# Set up logging for debugging purposes.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Generate denoise metric based on denoising strategies.",
    )
    parser.add_argument(
        "input_path",
        action="store",
        type=str,
        help="Input path to the timeseries collection.",
    )
    parser.add_argument(
        "output_path",
        action="store",
        type=str,
        help="output path for metrics.",
    )
    parser.add_argument(
        "--atlas",
        action="store",
        type=str,
        help="Atlas name (schaefer7networks, mist, difumo, gordon333)",
    )
    parser.add_argument(
        "--dimension",
        action="store",
        help="Number of ROI. See meta data of each atlas to get valid inputs.",
    )
    parser.add_argument(
        "--qc",
        action="store",
        default=None,
        help="Automatic motion QC thresholds.",
    )
    parser.add_argument(
        "--metric",
        action="store",
        default="connectomes",
        help="Metric to build {connectomes, qcfc, modularity}",
    ) 
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset name (default: extracted from input_path)",
    )
    parser.add_argument(
        "--fmriprep_ver",
        type=str,
        default=None,
        help="Override fMRIPrep version (default: extracted from input_path)",
    )
    return parser.parse_args()


def main():
    """Main function."""

    print("Logging main function start message", flush=True)

    args = parse_args()
    logging.debug("Parsed arguments: %s", vars(args))
    print(vars(args))

    input_path = Path(args.input_path)
    atlas = args.atlas
    dimension = args.dimension

    logging.debug("Input path: %s", input_path)
    print(input_path)

    # Use overrides if provided, otherwise extract from input_path
    dataset = args.dataset if args.dataset else input_path.parents[0].name
    fmriprep_ver = args.fmriprep_ver if args.fmriprep_ver else input_path.name
    logging.debug("Dataset: %s, fMRIPrep version: %s", dataset, fmriprep_ver)
    print(dataset)
    print(fmriprep_ver)

    path_root = Path(args.output_path).absolute()
    output_path = path_root / dataset / fmriprep_ver
    output_path.mkdir(parents=True, exist_ok=True)
    logging.debug("Output path created: %s", output_path)
    print(output_path)

    # Load the full ROI list from the atlas TSV file.
    # Get the "denoise" directory from the input path
    denoise_dir = input_path

    # # In case the input is a deeper file or folder (e.g. sub-XXX), move up until we hit "denoise"
    # while denoise_dir.name != "denoise" and denoise_dir != denoise_dir.parent:
    #     denoise_dir = denoise_dir.parent

    # Now build the full path to final_roi_labels.csv
    roi_label_csv = denoise_dir / "final_roi_labels.csv"
    roi_df = pd.read_csv(roi_label_csv)
    full_roi_list = roi_df["roi_name"].tolist()
    logging.debug("Loaded full ROI list with %d ROIs.", len(full_roi_list))

    strategy_names = get_prepro_strategy(None)
    logging.debug("Retrieved pre-processing strategies: %s", list(strategy_names.keys()))
    motion_qc = get_qc_criteria(args.qc)
    logging.debug("Motion QC criteria: %s", motion_qc)
    metric_option = str(args.metric)
    logging.debug("Metric option selected: %s", metric_option)

    collection_metric = []
    used_strategy_names = []  # track only those that succeed (for 'connectome' case)

    for strategy_name in strategy_names.keys():
        file_pattern = f"atlas-{atlas}_nroi-{dimension}_desc-{strategy_name}"
        logging.info("Processing strategy: %s with file pattern: %s", strategy_name, file_pattern)
        print(strategy_name)

        # Corrected glob pattern: look inside subject folders
        glob_pattern = str(input_path / "sub-*" / f"*{file_pattern}_timeseries.tsv")
        matching_files = glob.glob(glob_pattern)

        if not matching_files:
            logging.warning(f"No time series files found for strategy '{strategy_name}', skipping.")
            continue

        connectome, phenotype = compute_connectome(
            atlas,
            input_path,
            dataset,
            fmriprep_ver,
            path_root,
            file_pattern,
            full_roi_list,
            gross_fd=motion_qc["gross_fd"],
            fd_thresh=motion_qc["fd_thresh"],
            proportion_thresh=motion_qc["proportion_thresh"],
        )
        logging.debug("Connectome computed for strategy %s", strategy_name)
        print("\tLoaded connectomes...")

        if metric_option == "connectome":
            cur_strategy_average = connectome.mean(axis=0)
            collection_metric.append(cur_strategy_average)
            used_strategy_names.append(strategy_name)
            logging.debug("Average connectome computed for strategy %s", strategy_name)
            print("\tAverage connectomes...")

        elif metric_option == "modularity":
            logging.info("Computing modularity using louvain_modularity for strategy %s", strategy_name)
            qs = Parallel(n_jobs=4)(
                delayed(louvain_modularity)(vect) for vect in connectome.values.tolist()
            )
            modularity = pd.DataFrame(
                qs, columns=[strategy_name], index=connectome.index
            )
            collection_metric.append(modularity)
            logging.debug("Modularity computed for strategy %s", strategy_name)
            print("\tModularity...")

        elif metric_option == "qcfc":
            logging.info("Computing QC-FC for strategy %s", strategy_name)
            metric = qcfc(
                phenotype.loc[:, "mean_framewise_displacement"],
                connectome,
                phenotype.loc[:, ["age", "gender"]],
            )
            metric = pd.DataFrame(metric)
            columns = [
                ("full_sample", f"{strategy_name}_{col}") for col in metric.columns
            ]
            columns = pd.MultiIndex.from_tuples(columns)
            metric.columns = columns
            collection_metric.append(metric)
            logging.debug("QC-FC metric computed for strategy %s", strategy_name)
            print("\tQC-FC...")

            # QC-FC by group
            groups = phenotype["groups"].unique()
            logging.debug("Processing QC-FC by group for strategy %s, groups: %s", strategy_name, groups)
            for group in groups:
                group_mask = phenotype["groups"] == group
                subgroup = phenotype[group_mask].index
                metric = qcfc(
                    phenotype.loc[subgroup, "mean_framewise_displacement"],
                    connectome.loc[subgroup, :],
                    phenotype.loc[subgroup, ["age", "gender"]],
                )
                metric = pd.DataFrame(metric)
                metric.columns = [
                    (group, f"{strategy_name}_{col}") for col in metric.columns
                ]
                collection_metric.append(metric)
                logging.debug("QC-FC metric computed for group %s under strategy %s", group, strategy_name)

        else:
            logging.error("Invalid metric option: %s", metric_option)
            raise ValueError

    collection_metric = pd.concat(collection_metric, axis=1)
    logging.debug("Concatenated collection_metric with shape: %s", collection_metric.shape)

    if metric_option == "connectome":
        collection_metric.columns = used_strategy_names
        logging.debug("Columns set to used strategy names for connectome metric.")

    output_file = output_path / f"dataset-{dataset}_atlas-{atlas}_nroi-{dimension}_{metric_option}.tsv"
    collection_metric.to_csv(output_file, sep="\t")
    logging.info("Final metrics saved to %s", output_file)

if __name__ == "__main__":
    print("Logging main function called", flush=True)
    main()