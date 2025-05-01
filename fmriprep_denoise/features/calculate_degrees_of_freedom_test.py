"""Calculate degree of freedom"""
import argparse
import logging
from pathlib import Path
import pandas as pd

from fmriprep_denoise.dataset.timeseries import get_confounds
from fmriprep_denoise.dataset.fmriprep import (
    get_prepro_strategy,
    fetch_fmriprep_derivative,
    generate_movement_summary,
)

# Set up logging for debugging purposes.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Extract confound degree of freedom info.",
    )
    parser.add_argument(
        "output_path", action="store", type=str, help="Output path data."
    )
    parser.add_argument(
        "--fmriprep_path",
        action="store",
        type=str,
        help="Path to a fmriprep dataset.",
    )
    parser.add_argument(
        "--dataset_name", action="store", type=str, help="Dataset name."
    )
    parser.add_argument(
        "--specifier",
        action="store",
        type=str,
        help=(
            "Text in a fmriprep file name, "
            "in between sub-<subject>_ses-<session>_and `space-<template>`."
        ),
    )
    parser.add_argument(
        "--participants_tsv",
        action="store",
        type=str,
        help="Path to participants.tsv in the original BIDS dataset.",
    )
    return parser.parse_args()


def main():
    print("Logging main function started", flush=True)
    args = parse_args()
    logging.debug("Parsed arguments: %s", vars(args))
    dataset_name = args.dataset_name
    fmriprep_specifier = args.specifier
    fmriprep_path = Path(args.fmriprep_path)
    participant_tsv = Path(args.participants_tsv)
    output_root = Path(args.output_path)

    # Ensure output directory exists
    output_root.mkdir(exist_ok=True, parents=True)
    logging.debug("Output directory created or already exists: %s", output_root)

    path_movement = Path(
        output_root / f"dataset-{dataset_name}_desc-movement_phenotype.tsv"
    )
    path_dof = Path(
        output_root / f"dataset-{dataset_name}_desc-confounds_phenotype.tsv"
    )

    # Fetch the fmriprep derivative data.
    logging.info("Fetching fmriprep derivative data.")
    full_data = fetch_fmriprep_derivative(
        dataset_name, participant_tsv, fmriprep_path, fmriprep_specifier
    )
    logging.debug("Initial fetch complete with keys: %s", full_data.__dict__.keys())

    if dataset_name == "ds000030":
        logging.info("Processing dataset ds000030 specifics.")
        participants = full_data.phenotypic.copy()
        mask_quality = participants["ghost_NoGhost"] == "No_ghost"
        participants = participants[mask_quality].index.tolist()
        subjects = [p.split("-")[-1] for p in participants]
        logging.debug("Filtered subjects: %s", subjects)
        full_data = fetch_fmriprep_derivative(
            dataset_name,
            participant_tsv,
            fmriprep_path,
            fmriprep_specifier,
            subject=subjects,
        )
        logging.debug("Refetch complete for specific subjects.")

    # Generate movement summary.
    logging.info("Generating movement summary.")
    movement = generate_movement_summary(full_data)
    movement = movement.sort_index()
    movement.to_csv(path_movement, sep="\t")
    logging.info("Movement stats generated and saved to: %s", path_movement)

    subjects = [p.split("-")[-1] for p in movement.index]
    logging.debug("Subjects extracted for further processing: %s", subjects)

    # Fetch benchmark strategies and aroma data.
    benchmark_strategies = get_prepro_strategy()
    logging.info("Benchmark strategies obtained.")
    data_aroma = fetch_fmriprep_derivative(
        dataset_name,
        participant_tsv,
        fmriprep_path,
        fmriprep_specifier,
        aroma=True,
        subject=subjects,
    )
    data = fetch_fmriprep_derivative(
        dataset_name,
        participant_tsv,
        fmriprep_path,
        fmriprep_specifier,
        subject=subjects,
    )
    logging.debug("Fetched aroma and standard fmriprep derivative data.")

    # Process confounds for each strategy.
    info = {}
    for strategy_name, parameters in benchmark_strategies.items():
        logging.info("Processing denoising strategy: %s", strategy_name)
        logging.debug("Parameters: %s", parameters)
        func_data = data_aroma.func if "aroma" in strategy_name else data.func
        for img in func_data:
            sub = img.split("/")[-1].split("_")[0]
            logging.debug("Processing image for subject: %s", sub)
            reduced_confounds, sample_mask = get_confounds(
                strategy_name, parameters, img
            )
            full_length = reduced_confounds.shape[0]
            logging.info("Full_length: %s", full_length)
            mask_length = len(sample_mask) if sample_mask is not None else 0
            logging.info("sample_mask_length: %s", mask_length)
            ts_length = full_length if sample_mask is None else len(sample_mask)
            logging.info("ts_length: %s", ts_length)
            excised_vol = full_length - ts_length
            excised_vol_pro = excised_vol / full_length
            regressors = reduced_confounds.columns.tolist()
            compcor = sum("comp_cor" in i for i in regressors)
            high_pass = sum("cosine" in i for i in regressors)
            total = len(regressors)
            fixed = total - compcor if "compcor" in strategy_name else len(regressors)

            # if "aroma" in strategy_name:
            #     path_aroma_ic = img.split("space-")[0] + "AROMAnoiseICs.csv"
            #     with open(path_aroma_ic, "r") as f:
            #         aroma = len(f.readline().split(","))
            #     total = fixed + aroma
            # else:
            #     aroma = 0
            #needed to replace because "AROMAnoiseICs.csv" when running halfpipe
            if "aroma" in strategy_name:
                # Try to infer AROMA components from confounds columns
                aroma_cols = [col for col in reduced_confounds.columns if "aroma" in col.lower()]
                aroma = len(aroma_cols)
            else:
                aroma = 0
            
            total = fixed + aroma 

            if "scrub" in strategy_name:
                total += excised_vol

            stats = {
                (strategy_name, "excised_vol"): excised_vol,
                (strategy_name, "excised_vol_proportion"): excised_vol_pro,
                (strategy_name, "high_pass"): high_pass,
                (strategy_name, "fixed_regressors"): fixed,
                (strategy_name, "compcor"): compcor,
                (strategy_name, "aroma"): aroma,
                (strategy_name, "total"): total,
                (strategy_name, "full_length"): full_length,
            }
            logging.debug("Stats for subject %s under strategy %s: %s", sub, strategy_name, stats)
            if info.get(sub):
                info[sub].update(stats)
            else:
                info[sub] = stats

    confounds_stats = pd.DataFrame.from_dict(info, orient="index")
    confounds_stats = confounds_stats.sort_index()
    confounds_stats.to_csv(path_dof, sep="\t")
    logging.info("Confounds stats generated and saved to: %s", path_dof)


if __name__ == "__main__":
    print("Logging main function called", flush=True)
    main()