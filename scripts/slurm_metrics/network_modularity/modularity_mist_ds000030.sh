#!/bin/bash
#SBATCH --job-name=netmod_mist_ds000030
#SBATCH --time=24:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/netmod_mist_ds000030.%a.out
#SBATCH --error=logs/netmod_mist_ds000030.%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G 
#SBATCH --array=0-9


OUTPUT="inputs/fmrieprep-denoise-metrics"
DIMENSIONS=(7 12 20 36 64 122 197 325 444 ROI)
DATASET=ds000030

source /lustre03/project/6003287/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate

cd /home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/

build_features_modularity \
    "/home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-${DATASET}.tar.gz" \
    ${OUTPUT} \
    --atlas mist \
    --dimension ${DIMENSIONS[${SLURM_ARRAY_TASK_ID}]} \
    --qc stringent


    
