#!/bin/bash
#SBATCH --job-name=metric_mist
#SBATCH --time=12:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/metric_mist.%a.out
#SBATCH --error=logs/metric_mist.%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G 
#SBATCH --array=0-9


OUTPUT="/home/${USER}/scratch/fmriprep-denoise-benchmark"
DIMENSIONS=(7 12 20 36 64 122 197 325 444 ROI)

source /lustre03/project/6003287/${USER}/.virtualenvs//fmriprep-denoise-benchmark/bin/activate

cd /home/${USER}/projects/def-pbellec/${USER}/fmriprep-denoise-benchmark/

python ./fmriprep_denoise/features/build_features.py \
    "/home/${USER}/projects/def-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-ds000030.tar.gz" \
    ${OUTPUT} \
    --atlas mist \
    --dimension ${DIMENSIONS[${SLURM_ARRAY_TASK_ID}]}
    
python ./fmriprep_denoise/features/build_features.py \
    "/home/${USER}/projects/def-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-ds000228.tar.gz" \
    ${OUTPUT} \
    --atlas mist \
    --dimension ${DIMENSIONS[${SLURM_ARRAY_TASK_ID}]}
