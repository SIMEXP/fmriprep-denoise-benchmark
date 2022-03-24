#!/bin/bash
#SBATCH --job-name=probseg
#SBATCH --time=12:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/probseg.%a.out
#SBATCH --error=logs/probseg.%a.err
#SBATCH --array=0-10 
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G 


OUTPUT="/home/${USER}/scratch/giga_timeseries"
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
    --atlas difumo \
    --strategy-name ${STRATEGY} \
    ${OUTPUT}
