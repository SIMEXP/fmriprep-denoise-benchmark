#!/bin/bash
#SBATCH --job-name=ds000228dseg
#SBATCH --time=2:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/ds000228dseg.%a.out
#SBATCH --error=logs/ds000228dseg.%a.err
#SBATCH --array=1-155
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G 


OUTPUT="/home/${USER}/scratch/giga_timeseries/dataset-ds000228"
fmriprep_path="/home/${USER}/scratch/ds000228/1643916303/fmriprep"
participants_tsv="/home/${USER}/scratch/ds000228/participants.tsv"
source /lustre03/project/6003287/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate
cd /home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/


subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" ${participants_tsv} )
echo $subject

python ./fmriprep_denoise/dataset/make_dataset.py \
    --fmriprep_path ${fmriprep_path} \
    --dataset_name ds000228 \
    --specifier task-pixar \
    --participants_tsv ${participants_tsv} \
    --atlas mist \
    --subject ${subject} \
    ${OUTPUT}

python ./fmriprep_denoise/dataset/make_dataset.py \
    --fmriprep_path ${fmriprep_path} \
    --dataset_name ds000228 \
    --specifier task-pixar \
    --participants_tsv ${participants_tsv} \
    --atlas gordon333 \
    --subject ${subject} \
    ${OUTPUT}

python ./fmriprep_denoise/dataset/make_dataset.py \
    --fmriprep_path ${fmriprep_path} \
    --dataset_name ds000228 \
    --specifier task-pixar \
    --participants_tsv ${participants_tsv} \
    --atlas schaefer7networks \
    --subject ${subject} \
    ${OUTPUT}
