"""
Process fMRIPrep outputs to timeseries based on denoising strategy.
"""
try:
    import argparse
    print("argparse imported successfully")
except Exception as e:
    print("Failed to import argparse:", e)

try:
    import logging
    print("logging imported successfully")
except Exception as e:
    print("Failed to import logging:", e)

try:
    import sys
    print("sys imported successfully")
except Exception as e:
    print("Failed to import sys:", e)

try:
    from pathlib import Path
    print("Path imported successfully")
except Exception as e:
    print("Failed to import Path:", e)

try:
    from fmriprep_denoise.dataset.fmriprep import get_prepro_strategy, fetch_fmriprep_derivative
    print("fmriprep_denoise.dataset.fmriprep imported successfully")
except Exception as e:
    print("Failed to import from fmriprep_denoise.dataset.fmriprep:", e)

try:
    from fmriprep_denoise.dataset.timeseries import generate_timeseries_per_dimension
    print("fmriprep_denoise.dataset.timeseries imported successfully")
except Exception as e:
    print("Failed to import from fmriprep_denoise.dataset.timeseries:", e)

print("Finished imports")

# Configure logging to output detailed debug information to stdout
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Generate connectome based on denoising strategy for "
            "fmriprep processed dataset."
        ),
    )
    parser.add_argument(
        "output_path",
        action="store",
        type=str,
        help="Output path for connectome.",
    )
    parser.add_argument(
        "--fmriprep_path",
        action="store",
        type=str,
        help="Path to a fMRIPrep dataset.",
    )
    parser.add_argument(
        "--dataset_name", action="store", type=str, help="Dataset name."
    )
    parser.add_argument("--subject", action="store", type=str, help="Subject ID.")
    parser.add_argument(
        "--specifier",
        action="store",
        type=str,
        help=(
            "Text in a fMRIPrep file name, in between "
            "sub-<subject>_ses-<session>_and `space-<template>`."
        ),
    )
    parser.add_argument(
        "--participants_tsv",
        action="store",
        type=str,
        help="Path to participants.tsv in the original BIDS dataset.",
    )
    parser.add_argument(
        "--atlas",
        action="store",
        type=str,
        help="Atlas name (schaefer7networks, MIST, difumo, gordon333).",
    )
    parser.add_argument(
        "--strategy-name",
        action="store",
        default=None,
        help=(
            "Denoise strategy name (see benchmark_strategies.json). "
            "Process all strategy if None."
        ),
    )
    return parser.parse_args()


def main():
    try:
        logging.info("Starting make_timeseries script")
        args = parse_args()
        logging.debug("Parsed arguments: %s", vars(args))
        
        dataset_name = args.dataset_name
        subject = args.subject
        strategy_name = args.strategy_name
        atlas_name = args.atlas
        fmriprep_specifier = args.specifier
        fmriprep_path = Path(args.fmriprep_path)
        participant_tsv = Path(args.participants_tsv)
        output_root = Path(args.output_path)
        
        # Log the computed output path
        logging.info("Output root: %s", output_root)
        ts_output = output_root / f"atlas-{atlas_name}"
        logging.info("Creating timeseries output directory at: %s", ts_output)
        ts_output.mkdir(exist_ok=True, parents=True)
        
        logging.info("Fetching benchmark strategies for strategy: %s", strategy_name)
        benchmark_strategies = get_prepro_strategy(strategy_name)
        benchmark_strategies = {
            k: v for k, v in benchmark_strategies.items() if "aroma" not in k
        }
        data_aroma = None
        logging.debug("Benchmark strategies: %s", benchmark_strategies)

        
        
        
        # logging.info("Fetching fMRIPrep derivative (ARoma) for subject: %s", subject)
        # data_aroma = fetch_fmriprep_derivative(
        #     dataset_name,
        #     participant_tsv,
        #     fmriprep_path,
        #     fmriprep_specifier,
        #     subject=subject,
        #     aroma=True,
        # )
        # logging.debug("Fetched ARoma data: %s", data_aroma)
        
        logging.info("Fetching fMRIPrep derivative (non-ARoma) for subject: %s", subject)
        data = fetch_fmriprep_derivative(
            dataset_name,
            participant_tsv,
            fmriprep_path,
            fmriprep_specifier,
            subject=subject,
        )
        logging.debug("Fetched non-ARoma data: %s", data)
        
        logging.info("Calling generate_timeseries_per_dimension with atlas: %s", atlas_name)
        generate_timeseries_per_dimension(
            atlas_name, ts_output, benchmark_strategies, data_aroma, data
        )
        logging.info("Timeseries generation completed successfully for subject: %s", subject)
    except Exception as e:
        logging.exception("An error occurred during timeseries generation: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()