#!/bin/bash
#SBATCH --job-name=metric_highdim
#SBATCH --time=24:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/metric_highdim.%a.out
#SBATCH --error=logs/metric_highdim.%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G 
#SBATCH --array=0-1


OUTPUT="/home/${USER}/scratch/fmriprep-denoise-benchmark"
source /lustre03/project/6003287/${USER}/.virtualenvs//fmriprep-denoise-benchmark/bin/activate
DATASET=(ds000030 ds000228)

cd /home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/

python ./fmriprep_denoise/features/build_features.py \
    "/home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-${DATASET[${SLURM_ARRAY_TASK_ID}]}.tar.gz" \
    ${OUTPUT} \
    --atlas schaefer7networks \
    --dimension 800

python ./fmriprep_denoise/features/build_features.py \
    "/home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-${DATASET[${SLURM_ARRAY_TASK_ID}]}.tar.gz" \
    ${OUTPUT} \
    --atlas schaefer7networks \
    --dimension 1000
