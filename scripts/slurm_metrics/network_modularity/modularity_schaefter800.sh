#!/bin/bash
#SBATCH --job-name=netmod_schaefer800
#SBATCH --time=24:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/netmod_schaefer800.%a.out
#SBATCH --error=logs/netmod_schaefer800.%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G 
#SBATCH --array=0-1


OUTPUT="inputs/fmrieprep-denoise-metrics"
source /lustre03/project/6003287/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate
DATASET=(ds000030 ds000228)

cd /home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/

build_features_modularity \
    "/home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-${DATASET[${SLURM_ARRAY_TASK_ID}]}.tar.gz" \
    ${OUTPUT} \
    --atlas schaefer7networks \
    --dimension 800 \
    --qc stringent

