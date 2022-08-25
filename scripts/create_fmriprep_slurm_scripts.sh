#!/bin/bash

CONTAINER_PATH="/lustre03/project/6003287/containers"
VERSION=("20.2.1" "20.2.7")
DATASET_PATH="/lustre04/scratch/${USER}/openneuro"
FMRIPREP_SLURM="/lustre03/project/6003287/fmriprep-slurm"
EMAIL=${SLACK_EMAIL_BOT}

echo "Create fmriprep-slurm scripts"

module load singularity/3.8

echo "BIDS validators"
for dataset in ${DATASETS[*]}; do
    # run BIDS validator on the dataset
    # you only need this done once
    singularity exec -B ${DATASET_PATH}/${dataset}:/DATA \
        ${CONTAINER_PATH}/fmriprep-20.2.1lts.sif bids-validator /DATA
done

echo ds000228
for v in ${VERSION[*]}; do
    echo "Slurm files for fmriprep-${v}"
    ${FMRIPREP_SLURM}/singularity_run.bash \
        ${DATASET_PATH}/ds000228 \
        fmriprep-${v}lts \
        --fmriprep-args=\"--use-aroma\" \
        --email=${EMAIL} \
        --container ${CONTAINER_PATH}/fmriprep-${v}lts.sif
done

echo ds000030
subjects=$(cat inputs/ds000030_valid-subjects.txt)
for v in ${VERSION[*]}; do
    echo "Slurm files for fmriprep-${v}"
    ${FMRIPREP_SLURM}/singularity_run.bash \
        ${DATASET_PATH}/ds000030 \
        fmriprep-${v}lts \
        --participant-label $subjects \
        --fmriprep-args=\"-t rest --use-aroma\" \
        --email=${EMAIL} \
        --container ${CONTAINER_PATH}/fmriprep-${v}lts.sif
done
