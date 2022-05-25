#!/bin/bash
#SBATCH --job-name=metric_difumo
#SBATCH --time=12:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/metric_mem_heavy.out
#SBATCH --error=logs/metric_memh_eavy.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G 


OUTPUT="/home/${USER}/scratch/fmriprep-denoise-benchmark"
source /home/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate

cd /home/${USER}/projects/def-pbellec/${USER}/fmriprep-denoise-benchmark/

python ./fmriprep_denoise/features/build_features.py \
    "/home/${USER}/projects/def-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-ds000228.tar.gz" \
    ${OUTPUT} \
    --atlas difumo \
    --dimension 1024

python ./fmriprep_denoise/features/build_features.py \
    "/home/${USER}/projects/def-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-ds000030.tar.gz" \
    ${OUTPUT} \
    --atlas difumo \
    --dimension 1024

python ./fmriprep_denoise/features/build_features.py \
    "/home/${USER}/projects/def-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-ds000228.tar.gz" \
    ${OUTPUT} \
    --atlas schaefer7networks \
    --dimension 1000

python ./fmriprep_denoise/features/build_features.py \
    "/home/${USER}/projects/def-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-ds000030.tar.gz" \
    ${OUTPUT} \
    --atlas schaefer7networks \
    --dimension 1000
