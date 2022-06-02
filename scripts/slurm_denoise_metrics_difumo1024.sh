#!/bin/bash
#SBATCH --job-name=metric_difumo1024
#SBATCH --time=24:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/metric_difumo1024.%a.out
#SBATCH --error=logs/metric_difumo1024.%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G 
#SBATCH --array=0-1


OUTPUT="/home/${USER}/scratch/fmriprep-denoise-benchmark"
source /home/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate
DATASET=(ds000030 ds000228)

cd /home/${USER}/projects/def-pbellec/${USER}/fmriprep-denoise-benchmark/

python ./fmriprep_denoise/features/build_features.py \
    "/home/${USER}/projects/def-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-${DATASET[${SLURM_ARRAY_TASK_ID}]}.tar.gz" \
    ${OUTPUT} \
    --atlas difumo \
    --dimension 1024
