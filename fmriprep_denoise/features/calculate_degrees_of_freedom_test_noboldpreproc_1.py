#!/usr/bin/env python
"""
This script calculates degrees of freedom and extracts movement/denoising metrics
from fMRIPrep confounds timeseries files only. All helper functions are defined
within this script.
"""

import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.utils import Bunch

from nilearn.interfaces.fmriprep import load_confounds_strategy, load_confounds



STRATEGY_FILE = "benchmark_strategies.json"


PHENOTYPE_INFO = {
    "ds000228": {
        "columns": ["Age", "Gender", "Child_Adult"],
        "replace": {"Age": "age", "Gender": "gender", "Child_Adult": "groups"},
    },
    "ds000030": {
        "columns": ["age", "gender", "diagnosis"],
        "replace": {"diagnosis": "groups"},
    },
}

# -------------------------------
# Helper
# -------------------------------

def expand_strategy_columns(strategy_name, df, parameters):
    """
    Expand placeholders into actual columns for each denoise strategy.
    """
    cols = []

    # Get denoise strategy if defined
    strat = parameters.get("denoise_strategy", None)
    include_gsr = "global_signal" in parameters

    if strat == "simple":
        motion_bases = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
        motion_cols = [col for col in df.columns if any(col.startswith(base) for base in motion_bases)]
        wmcsf_cols = []  # Exclude WM/CSF
        cosine_cols = []  # Gaussian convolution used

        cols = motion_cols + wmcsf_cols + cosine_cols
        if include_gsr and "global_signal" in df.columns:
            cols.append("global_signal")

    elif strat == "scrubbing":
        motion_bases = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
        motion_cols = [col for col in df.columns if any(col.startswith(base) for base in motion_bases)]
        wmcsf_cols = []  
        cosine_cols = []  

        scrub_cols = []  

        cols = motion_cols + wmcsf_cols + cosine_cols + scrub_cols
        if include_gsr and "global_signal" in df.columns:
            cols.append("global_signal")

    elif strat == "compcor":
      
        compcor_cols = [col for col in df.columns if col.startswith("c_comp_cor_0")]
        cols = compcor_cols[:5]  # (c_comp_cor_00 to c_comp_cor_04)

    elif strat == "ica_aroma":
        logging.warning(f"[{strategy_name}] No explicit AROMA regressors in TSV — using high-pass + wm/csf averages.")
        cosine_cols = [] 
        wmcsf_cols = []  
        cols = cosine_cols + wmcsf_cols
        if include_gsr and "global_signal" in df.columns:
            cols.append("global_signal")

    elif strat == "baseline":
        cosine_cols = []  
        cols = cosine_cols

    # If explicit "strategy" is defined in JSON, override all above
    if "strategy" in parameters:
        cols = []
        for item in parameters["strategy"]:
            if item == "high_pass":
                logging.debug(f"[{strategy_name}] Ignoring 'high_pass' since Gaussian filtering was used.")
                continue
            elif item in df.columns:
                cols.append(item)
            else:
                logging.warning(f"[{strategy_name}] Column '{item}' not found in TSV.")

    if not cols:
        logging.warning(f"[{strategy_name}] No valid regressors found in file.")
    else:
        logging.debug(f"[{strategy_name}] Final columns: {cols}")

    return cols

def _load_confounds_from_tsv(tsv_path, strategy_name, parameters):
    df = pd.read_csv(tsv_path, sep="\t")
    n_timepoints = df.shape[0]

    cols = expand_strategy_columns(strategy_name, df, parameters)
    if not cols:
        logging.warning(f"[{strategy_name}] No valid regressors found in TSV. Proceeding with empty confounds.")
        reduced_confounds = pd.DataFrame(index=df.index)  # Return an empty DataFrame with correct number of rows
        return reduced_confounds, [True] * len(df)  # All timepoints retained

    logging.debug(f"[{strategy_name}] Selected columns: {cols}")

    reduced_confounds = df[cols].copy()
    sample_mask = [True] * n_timepoints

    logging.debug(
        f"TSV loader ({strategy_name}): keeping {len(cols)} cols, "
        f"{n_timepoints} timepoints → mask of all True"
    )
    return reduced_confounds, sample_mask

def get_confounds(strategy_name, parameters, img):
    logging.debug(f"get_confounds  img={img}  strat={strategy_name}  params={parameters}")

    if img.lower().endswith(".tsv") and "scrub" in strategy_name.lower():
        tsv_path = Path(img)
        func_dir = tsv_path.parent
        subject_id = tsv_path.name.split("_")[0]
        specifier = "_".join(tsv_path.name.split("_")[1:tsv_path.name.split("_").index("desc-confounds")])

        bold_candidates = sorted(func_dir.glob(f"{subject_id}_{specifier}_space-*_res-*_desc-preproc_bold.nii.gz"))
        if not bold_candidates:
            bold_candidates = sorted(func_dir.glob(f"{subject_id}_{specifier}_space-*_desc-preproc_bold.nii.gz"))

        if bold_candidates:
            bold_file = bold_candidates[0]
            logging.warning(f"[{strategy_name}] Found BOLD image: {bold_file} — using NIfTI-based sample_mask.")

            # Determine the actual confound strategy
            strategy = []
            if "motion" in parameters or parameters.get("denoise_strategy") == "scrubbing":
                strategy.append("motion")
            if "global_signal" in parameters:
                strategy.append("global_signal")
            strategy.append("scrub")  # Required for scrubbing mask

            clean_parameters = {k: v for k, v in parameters.items()
                                if k in {"motion", "global_signal", "scrub", "fd_threshold", "std_dvars_threshold"}}

            logging.debug(f"[{strategy_name}] Cleaned load_confounds parameters: {clean_parameters}")

            return load_confounds(str(bold_file), strategy=tuple(strategy), **clean_parameters)

        else:
            logging.warning(f"[{strategy_name}] BOLD file not found — falling back to TSV-based loader.")

    # Fallback to TSV-based or other strategy
    if img.lower().endswith(".tsv"):
        return _load_confounds_from_tsv(img, strategy_name, parameters)

    if "aroma" in strategy_name.lower():
        valid_keys = {"global_signal", "motion", "wm_csf", "compcor", "high_pass", "scrub"}
        clean = {k: v for k, v in parameters.items() if k in valid_keys}
        return load_confounds(img, strategy=("motion", "ica_aroma", "global_signal"), **clean)

    return load_confounds_strategy(img, denoise_strategy=parameters.get("denoise_strategy", "simple"), **parameters)


def get_prepro_strategy(strategy_name=None):
    """
    Return a dictionary of denoising strategies and their parameters by reading
    them from the benchmark_strategies.json file at a fixed absolute path.

    Parameters
    ----------
    strategy_name : None or str
        If provided, returns only that strategy’s parameters.

    Returns
    -------
    dict
        A dictionary mapping strategy names to their parameters.
    """
    # Use the provided absolute path to the JSON file.
    strategy_file_path = Path("/home/seann/scratch/denoise/fmriprep-denoise-benchmark/fmriprep_denoise/dataset/benchmark_strategies.json") #change this to your benchmark.json
    if not strategy_file_path.is_file():
        raise FileNotFoundError(f"Strategy file not found at {strategy_file_path}")
    
    with open(strategy_file_path, "r") as f:
        benchmark_strategies = json.load(f)
    
    if strategy_name is None:
        logging.info("Processing all strategies.")
        return benchmark_strategies
    
    if strategy_name not in benchmark_strategies:
        raise NotImplementedError(
            f"Strategy '{strategy_name}' is not implemented. Choose from: {list(benchmark_strategies.keys())}"
        )
    
    logging.info(f"Processing strategy '{strategy_name}'.")
    return {strategy_name: benchmark_strategies[strategy_name]}


def fetch_fmriprep_derivative(dataset_name, participant_tsv_path, path_fmriprep_derivative, specifier,
                              subject=None, space="MNI152NLin2009cAsym", aroma=False):
    """
    Fetch fMRIPrep derivatives—here, confounds files only.
    This function searches the fMRIPrep derivative folder for confounds timeseries files.
    
    Parameters
    ----------
    dataset_name : str
        Dataset name.
    
    participant_tsv_path : pathlib.Path
        Path to the BIDS participants.tsv file.
    
    path_fmriprep_derivative : pathlib.Path
        Path to the fMRIPrep derivative directory.
    
    specifier : str
        A substring present in the confounds file name that identifies the processing.
    
    subject : str or list, optional
        Subject(s) to include; if None, all subjects are processed.
    
    space : str, default "MNI152NLin2009cAsym"
        Template space (used if needed in file naming).
    
    aroma : bool, default False
        If True, select only AROMA confounds files.
    
    Returns
    -------
    Bunch
        An object with attributes:
            - dataset_name: the dataset name.
            - confounds: list of confounds file paths (as strings).
            - phenotypic: DataFrame of participants phenotypic information.
    """
    if not participant_tsv_path.is_file():
        raise FileNotFoundError(f"Cannot find {participant_tsv_path}")
    if participant_tsv_path.name != "participants.tsv":
        raise FileNotFoundError(f"File {participant_tsv_path} is not a BIDS participant file.")
    participant_tsv = pd.read_csv(participant_tsv_path, sep="\t", index_col=["participant_id"])
    logging.debug("Loaded participants.tsv with shape: %s", participant_tsv.shape)
    
    if subject is None:
        subject_dirs = list(path_fmriprep_derivative.glob("sub-*/"))
    elif isinstance(subject, str):
        subject_dirs = list(path_fmriprep_derivative.glob(f"sub-{subject}/"))
    elif isinstance(subject, list):
        subject_dirs = []
        for s in subject:
            s_path = path_fmriprep_derivative / f"sub-{s}"
            if s_path.is_dir():
                subject_dirs.append(s_path)
    else:
        raise ValueError("Unsupported input for subject.")
    
    confounds_tsv_path = []
    include_subjects = []
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        logging.debug("Processing subject directory: %s", subject_id)
        # Construct  expected confounds file path
        cur_confound = subject_dir / "func" / f"{subject_id}_{specifier}_desc-confounds_timeseries.tsv"
        logging.debug("Looking for confounds file: %s", cur_confound)
        if cur_confound.is_file():
            logging.info("Found confounds for %s", subject_id)
            confounds_tsv_path.append(str(cur_confound))
            include_subjects.append(subject_id)
        else:
            logging.warning("Missing confounds file for %s", subject_id)
    
    logging.debug("Subjects included: %s", include_subjects)
    return Bunch(
        dataset_name=dataset_name,
        confounds=confounds_tsv_path,
        phenotypic=participant_tsv.loc[include_subjects, :]
    )


def generate_movement_summary(data):
    """
    Generate movement statistics for each subject based on the confounds files.
    
    Parameters
    ----------
    data : Bunch
        The object returned by fetch_fmriprep_derivative, containing:
            - confounds: list of confounds file paths.
            - phenotypic: DataFrame of participant information.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame with subjects as rows and movement metrics (e.g. mean framewise displacement)
        along with phenotypic data.
    """
    group_mean_fd = pd.DataFrame()
    group_mean_fd.index.name = "participant_id"
    for confounds in data.confounds:
        subject_id = confounds.split("/")[-1].split("_")[0]
        logging.debug("Reading confounds file for subject %s: %s", subject_id, confounds)
        try:
            confounds_df = pd.read_csv(confounds, sep="\t")
        except Exception as e:
            logging.error("Error reading confounds file %s: %s", confounds, e)
            continue
        logging.debug("Confounds shape for %s: %s", subject_id, confounds_df.shape)
        if "framewise_displacement" in confounds_df.columns:
            mean_fd = confounds_df["framewise_displacement"].mean()
            logging.debug("Mean FD for subject %s: %f", subject_id, mean_fd)
            group_mean_fd.loc[subject_id, "mean_framewise_displacement"] = mean_fd
        else:
            logging.warning("Framewise displacement column missing for subject %s", subject_id)

    participants = data.phenotypic.copy()
   
    covar = participants.loc[:, PHENOTYPE_INFO[data.dataset_name]["columns"]]
    fix_col_name = PHENOTYPE_INFO[data.dataset_name].get("replace", False)
    if isinstance(fix_col_name, dict):
        covar = covar.rename(columns=fix_col_name)
    
    # Convert gender column from string to numeric
    if covar["gender"].dtype == object:
        covar["gender"] = covar["gender"].map({"M": 0, "F": 1})
    
    covar["gender"] = covar["gender"].astype("float")
    covar["age"] = covar["age"].astype("float")
    combined = pd.concat((group_mean_fd, covar), axis=1, join="inner")
    logging.debug("Combined movement and phenotype data head:\n%s", combined.head())
    return combined

# -------------------------------
# Main Processing Function
# -------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Extract confound degree of freedom info from confounds files only."
    )
    parser.add_argument("output_path", type=str, help="Output directory for result TSV files.")
    parser.add_argument("--fmriprep_path", type=str, help="Path to the fMRIPrep dataset directory.")
    parser.add_argument("--dataset_name", type=str, help="Dataset name (used in directory structure).")
    parser.add_argument("--specifier", type=str, help="Substring to filter confounds files (e.g., 'task-pixar').")
    parser.add_argument("--participants_tsv", type=str, help="Path to participants.tsv file (BIDS).")
    return parser.parse_args()

def main():
    print("Logging main function started", flush=True)
    args = parse_args()
    logging.debug("Parsed arguments: %s", vars(args))
    dataset_name = args.dataset_name
    specifier = args.specifier
    fmriprep_path = Path(args.fmriprep_path)
    participant_tsv = Path(args.participants_tsv) if args.participants_tsv else None
    output_root = Path(args.output_path)


    output_root.mkdir(exist_ok=True, parents=True)
    logging.debug("Output directory exists: %s", output_root)

    path_movement = output_root / f"dataset-{dataset_name}_desc-movement_phenotype_HALFpipe.tsv"
    path_dof = output_root / f"dataset-{dataset_name}_desc-confounds_phenotype.HALFpipe.tsv"

    # Fetch confounds derivative data
    logging.info("Fetching confounds derivative data.")
    data = fetch_fmriprep_derivative(dataset_name, participant_tsv, fmriprep_path, specifier)
    data_aroma = fetch_fmriprep_derivative(dataset_name, participant_tsv, fmriprep_path, specifier, aroma=True)

    # Generate movement summary from confounds files
    logging.info("Generating movement summary.")
    movement = generate_movement_summary(data)
    movement = movement.sort_index()
    movement.to_csv(path_movement, sep="\t")
    logging.info("Movement summary saved to: %s", path_movement)

    subjects = list(movement.index)
    logging.debug("Subjects extracted: %s", subjects)

    benchmark_strategies = get_prepro_strategy()
    logging.info("Benchmark strategies obtained: %s", list(benchmark_strategies.keys()))

    info = {}
    for strategy_name, parameters in benchmark_strategies.items():
        logging.info("Processing strategy: %s", strategy_name)
        confound_files = data_aroma.confounds if "aroma" in strategy_name.lower() else data.confounds

        for cf_path in confound_files:
            parts = Path(cf_path).name.split("_")
            sub = parts[0]
            logging.debug("Processing confounds for subject: %s", sub)
            reduced_confounds, sample_mask = get_confounds(strategy_name, parameters, cf_path)

            full_length = reduced_confounds.shape[0]

            regressors = reduced_confounds.columns.tolist()

            # High-pass filtering was done via Gaussian convolution, not cosine regressors
            high_pass = 0 # high_pass = sum(col.startswith("cosine") for col in regressors)
            compcor   = sum("comp_cor" in col for col in regressors)
            aroma     = sum("aroma" in col.lower() for col in regressors)
            scrub     = sum(col.startswith("motion_outlier") for col in regressors)

         
            if sample_mask is None:
                excised_vol = 0
            elif isinstance(sample_mask, (np.ndarray, list)) and np.array(sample_mask).dtype == np.bool_:
                excised_vol = np.sum(~np.array(sample_mask)) 
            else:
                excised_vol = full_length - len(sample_mask) 
            excised_vol_pro = excised_vol / full_length if full_length > 0 else 0
            logging.debug(f"[{strategy_name}] sample_mask type: {type(sample_mask)}, dtype: {np.array(sample_mask).dtype}, shape: {np.array(sample_mask).shape}")

            fixed = len(regressors) - compcor
            
            if "scrub" in strategy_name.lower():
                total = len(regressors) + excised_vol
            else:
                total = len(regressors)

            stats = {
                (strategy_name, "excised_vol"): excised_vol,
                (strategy_name, "excised_vol_proportion"): excised_vol_pro,
                (strategy_name, "high_pass"): high_pass,
                (strategy_name, "fixed_regressors"): fixed,
                (strategy_name, "compcor"): compcor,
                (strategy_name, "aroma"): aroma,
                (strategy_name, "scrub"): scrub,
                (strategy_name, "total"): total,
                (strategy_name, "full_length"): full_length,
            }
            logging.debug("Stats for subject %s, strategy %s: %s", sub, strategy_name, stats)
            if sub in info:
                info[sub].update(stats)
            else:
                info[sub] = stats

    confounds_stats = pd.DataFrame.from_dict(info, orient="index")
    confounds_stats = confounds_stats.sort_index()
    confounds_stats.to_csv(path_dof, sep="\t")
    logging.info("Confounds stats saved to: %s", path_dof)
# -------------------------------
# Main Entry Point
# -------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    print("Logging main function called", flush=True)
    main()