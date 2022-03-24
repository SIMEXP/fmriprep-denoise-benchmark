#!/bin/bash
#SBATCH --job-name=dseg
#SBATCH --time=12:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/dseg.%a.out
#SBATCH --error=logs/dseg.%a.err
#SBATCH --array=0-10 
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G 


OUTPUT="/home/${USER}/scratch/giga_timeseries/ds000228"
fmriprep_path="/home/${USER}/scratch/ds000228/1643916303/fmriprep"
participants_tsv="/home/${USER}/scratch/ds000228/participants.tsv"
source /home/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate
cd /home/${USER}/projects/def-pbellec/${USER}/fmriprep-denoise-benchmark/

mapfile -t arr < <(jq -r 'keys[]' fmriprep_denoise/benchmark_strategies.json)
STRATEGY=${arr[${SLURM_ARRAY_TASK_ID}]}
echo $STRATEGY

python ./fmriprep_denoise/process_connectomes.py \
    --fmriprep_path ${fmriprep_path} \
    --dataset_name ds000228 \
    --specifier task-pixar \
    --participants_tsv ${participants_tsv} \
    --atlas mist \
    --strategy-name ${STRATEGY} \
    ${OUTPUT}

python ./fmriprep_denoise/process_connectomes.py \
    --fmriprep_path ${fmriprep_path} \
    --dataset_name ds000228 \
    --specifier task-pixar \
    --participants_tsv ${participants_tsv} \
    --atlas gordon333 \
    --strategy-name ${STRATEGY} \
    ${OUTPUT}

python ./fmriprep_denoise/process_connectomes.py \
    --fmriprep_path ${fmriprep_path} \
    --dataset_name ds000228 \
    --specifier task-pixar \
    --participants_tsv ${participants_tsv} \
    --atlas schaefer7networks \
    --strategy-name ${STRATEGY} \
    ${OUTPUT}

