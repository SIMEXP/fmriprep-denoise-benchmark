#!/bin/bash
#SBATCH --job-name=metric_difumo
#SBATCH --time=24:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/metric_difumo.%a.out
#SBATCH --error=logs/metric_difumo.%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G 
#SBATCH --array=0-3


OUTPUT="/home/${USER}/scratch/fmriprep-denoise-benchmark"
DIMENSIONS=(64 128 256 512)
source /lustre03/project/6003287/${USER}/.virtualenvs//fmriprep-denoise-benchmark/bin/activate

cd /home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/

python ./fmriprep_denoise/features/build_features.py \
    "/home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-ds000030.tar.gz" \
    ${OUTPUT} \
    --atlas difumo \
    --dimension ${DIMENSIONS[${SLURM_ARRAY_TASK_ID}]}
    
python ./fmriprep_denoise/features/build_features.py \
    "/home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-ds000228.tar.gz" \
    ${OUTPUT} \
    --atlas difumo \
    --dimension ${DIMENSIONS[${SLURM_ARRAY_TASK_ID}]}
