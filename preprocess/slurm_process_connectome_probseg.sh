#!/bin/bash
#SBATCH --job-name=probseg
#SBATCH --time=12:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/dseg.%a.out
#SBATCH --error=logs/dseg.%a.err
#SBATCH --array=0-10 
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G 


OUTPUT="/home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/"
participants_tsv="fmriprep_denoise/tests/data/participants.tsv"
source /lustre03/project/6003287/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate
cd /home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/

mapfile -t arr < <(jq -r 'keys[]' preprocess/benchmark_strategies.json)
STRATEGY=${arr[${SLURM_ARRAY_TASK_ID}]}
echo $STRATEGY

python ./preprocess/process_connectomes.py \
    --fmriprep_path inputs/raw/ds000228/derivatives/fmriprep/ \
    --dataset_name ds000228 \
    --specifier task-pixar \
    --participants_tsv ${participants_tsv} \
    --atlas difumo \
    --strategy-name ${STRATEGY} \
    ${OUTPUT}
