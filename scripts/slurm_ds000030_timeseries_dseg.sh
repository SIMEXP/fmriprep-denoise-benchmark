#!/bin/bash
#SBATCH --job-name=ds000030dseg
#SBATCH --time=12:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/ds000030dseg.%a.out
#SBATCH --error=logs/ds000030dseg.%a.err
#SBATCH --array=1-259
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G 


OUTPUT="/home/${USER}/scratch/giga_timeseries/dataset-ds000030"
fmriprep_path="/home/${USER}/scratch/ds000030/1651688951/fmriprep"
participants_tsv="/home/${USER}/scratch/ds000030/participants.tsv"
valid_id="/home/${USER}/scratch/ds000030/valid_subjects.txt"
source /home/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate
cd /home/${USER}/projects/def-pbellec/${USER}/fmriprep-denoise-benchmark/


# subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" ${participants_tsv} )
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID}))s/(\S*)\>.*/\1/gp" ${valid_id} )
echo $subject

python ./fmriprep_denoise/data/make_dataset.py \
    --fmriprep_path ${fmriprep_path} \
    --dataset_name ds000030 \
    --specifier task-rest \
    --participants_tsv ${participants_tsv} \
    --atlas mist \
    --subject ${subject} \
    ${OUTPUT}

python ./fmriprep_denoise/data/make_dataset.py \
    --fmriprep_path ${fmriprep_path} \
    --dataset_name ds000030 \
    --specifier task-rest \
    --participants_tsv ${participants_tsv} \
    --atlas gordon333 \
    --subject ${subject} \
    ${OUTPUT}
