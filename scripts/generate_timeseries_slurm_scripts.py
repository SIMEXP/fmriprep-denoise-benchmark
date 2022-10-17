#!/usr/bin/python3

"""
Generate timeseries slurm script.

Run this after preprocessing the fMRI data with fmriprep-slurm.
"""

from pathlib import Path

import sys
import argparse
import re
import json
import time


SLURM_ACCOUNT_DEFAULT = "rrg-pbellec"
SLURM_JOB_DIR = ".slurm"
SLURM_LOGS_DIR = ".logs"
FMRIPREP_SPECIFIER_PATTERN = r"sub-[A-Za-z0-9]*_(ses-[A-Za-z0-9]*_)?([A-Za-z0-9_-]*)_space"

ATLAS_COLLECTIONS = {
    'dseg': ["mist", "gordon333", "schaefer7networks"],
    'probseg': ["difumo"]
}


RESOURCE = {
    'dseg': {
        'time': "1:00:00",
        'cpus': 1,
        'mem_per_cpu': 8,
    },
    'probseg':{
        'time': "1:00:00",
        'cpus': 1,
        'mem_per_cpu': 24,
    }
}

slurm_preamble = """#!/bin/bash
#SBATCH --account={slurm_account}
#SBATCH --job-name={jobname}
#SBATCH --output={log_output}/{jobname}.%a.out
#SBATCH --error={log_output}/{jobname}.%a.err
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem-per-cpu={mem_per_cpu}G
#SBATCH --array=1-{n_subjects}

# Activate environment
source {virtualenv}/bin/activate

# All subjects
SUBJECTS=({subjects})
"""

slurm_task_array = """
subject=${SUBJECTS[${SLURM_ARRAY_TASK_ID} - 1 ]}
echo $subject

# Run the script
"""

cmd_template = """make_timeseries \
--fmriprep_path {fmriprep_path} \
--dataset_name {dataset} \
--specifier {specifier} \
--participants_tsv {participants_tsv} \
--atlas {atlas} \
--subject {subject} \
{timeseires_output}
"""


def create_slurm_scripts(args):
    subjects = list(p.name.split("sub-")[1]
        for p in Path(args.fmriprep_output).glob("sub-*/") if p.is_dir())
    subjects = sorted(subjects)
    dataset = Path(args.fmriprep_output).parents[1].name
    fmriprep_version = get_fmriprep_version(args.fmriprep_output)
    # timestamp = int(time.time())
    timeseires_output = f"{args.scratch_path}/fmriprep-denoise-benchmark/giga_timeseries/{dataset}/{fmriprep_version}"
    (Path(timeseires_output) / SLURM_JOB_DIR).mkdir(parents=True, exist_ok=True)
    (Path(timeseires_output) / SLURM_LOGS_DIR).mkdir(parents=True, exist_ok=True)
    print("Writing slurm scripts:")

    for atlas_type in ATLAS_COLLECTIONS:
        job_spec = {
            'slurm_account': args.slurm_account,
            'jobname': f"timeseries_{dataset}_{fmriprep_version}_{atlas_type}",
            'log_output': str(Path(timeseires_output) / SLURM_LOGS_DIR),
            'participants_tsv': args.participants_tsv,
            'virtualenv': args.virtualenv,
            'n_subjects': len(subjects),
            'subjects': " ".join(subjects),
        }
        job_spec.update(RESOURCE[atlas_type].copy())
        script_path = Path(timeseires_output) / SLURM_JOB_DIR / f"{job_spec['jobname']}.sh"
        header = [slurm_preamble.format(**job_spec), slurm_task_array]
        cmd_atlas = create_cmd_inputs(args, '${subject}', dataset, atlas_type, timeseires_output)
        script = ("\n").join(header + cmd_atlas)
        with open(script_path, "w") as f:
            f.write(script)
    print("Find outputs in " + timeseires_output + "/" + SLURM_JOB_DIR)
    print("Run jobs:")
    print(f"find {timeseires_output}/{SLURM_JOB_DIR}/timeseries_{dataset}_{fmriprep_version}_*.sh -type f | while read file; do sbatch \"$file\"; done")


def create_cmd_inputs(args, subject, dataset, atlas_type, timeseires_output):
    cmd_atlas = []
    for atlas in ATLAS_COLLECTIONS[atlas_type]:
        cur_spec = {
            'fmriprep_path': args.fmriprep_output,
            'dataset': dataset,
            'specifier': find_specifier(args.fmriprep_output),
            'participants_tsv': args.participants_tsv,
            'atlas': atlas,
            'subject': subject,
            'timeseires_output': timeseires_output
        }
        cmd = cmd_template.format(**cur_spec)
        cmd_atlas.append(cmd)
    return cmd_atlas


def get_fmriprep_version(fmriprep_output):
    desc_file = f"{fmriprep_output}/dataset_description.json"
    with open(desc_file, 'r') as file:
        version = json.load(file)['GeneratedBy'][0]['Version']
    if (".").join(version.split(".")[:2]) == "20.2":
        return f"fmriprep-{version}lts"
    else:
        return f"fmriprep-{version}"


def find_specifier(fmriprep_output):
    "Text in between sub-<subject>_ses-<session>_and space-<template>"
    filename = list(
        Path(fmriprep_output).glob("sub-*/func/*_desc-preproc_bold.nii.gz")
    )[0].name
    spec_finder = re.compile(FMRIPREP_SPECIFIER_PATTERN)
    return spec_finder.search(filename).groups()[-1]


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="create timeseries extraction scripts",
    )
    parser.add_argument(
        "scratch_path",
        help="Path to scratch space."
    )
    parser.add_argument(
        "fmriprep_output",
        help="Path to fMRIPrep output directory."
    )
    parser.add_argument(
        "participants_tsv",
        help="Path to participants.tsv in the original BIDS dataset."
    )
    parser.add_argument(
        "virtualenv",
        help="Path to virtual environment of this project."
    ),
    parser.add_argument(
        "--slurm-account",
        action="store",
        default=SLURM_ACCOUNT_DEFAULT,
        help="SLURM account for job submission (default: rrg-pbellec)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(vars(args))
    create_slurm_scripts(args)


if __name__ == "__main__":
    main()
