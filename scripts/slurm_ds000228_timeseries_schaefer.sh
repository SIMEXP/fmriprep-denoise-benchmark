#!/bin/bash
#SBATCH --job-name=ds000228schaefer
#SBATCH --time=12:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/ds000228schaefer.%a.out
#SBATCH --error=logs/ds000228schaefer.%a.err
#SBATCH --array=1-155
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G 


OUTPUT="/home/${USER}/scratch/giga_timeseries/dataset-ds000228"
fmriprep_path="/home/${USER}/scratch/ds000228/1643916303/fmriprep"
participants_tsv="/home/${USER}/scratch/ds000228/participants.tsv"
source /home/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate
cd /home/${USER}/projects/def-pbellec/${USER}/fmriprep-denoise-benchmark/


subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" ${participants_tsv} )
echo $subject

python ./fmriprep_denoise/data/make_dataset.py \
    --fmriprep_path ${fmriprep_path} \
    --dataset_name ds000228 \
    --specifier task-pixar \
    --participants_tsv ${participants_tsv} \
    --atlas schaefer7networks \
    --subject ${subject} \
    ${OUTPUT}
