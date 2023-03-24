# create fMRIPrep SLURM scripts, run from project root
#!/bin/bash

CONTAINER_PATH="/lustre03/project/6003287/containers"
VERSION=("20.2.1" "20.2.5")
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
        --time=24:00:00 \
        --mem-per-cpu=16384 \
        --cpus=1 \
        --container fmriprep-${v}lts
done

echo ds000030
subjects=$(cat scripts/preprocessing/ds000030_valid-subjects.txt)
for v in ${VERSION[*]}; do
    echo "Slurm files for fmriprep-${v}"
    ${FMRIPREP_SLURM}/singularity_run.bash \
        ${DATASET_PATH}/ds000030 \
        fmriprep-${v}lts \
        --participant-label $subjects \
        --fmriprep-args=\"-t rest --use-aroma\" \
        --time=24:00:00 \
        --mem-per-cpu=16384 \
        --cpus=1 \
        --email=${EMAIL} \
        --container fmriprep-${v}lts
done
