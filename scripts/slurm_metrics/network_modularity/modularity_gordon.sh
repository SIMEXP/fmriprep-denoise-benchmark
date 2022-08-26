#!/bin/bash
#SBATCH --job-name=netmod_gordon
#SBATCH --time=12:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/netmod_gordon.out
#SBATCH --error=logs/netmod_gordon.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G 


OUTPUT="inputs/fmrieprep-denoise-metrics"

source /lustre03/project/6003287/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate

cd /home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/

echo "gordon333"

build_features_modularity \
    "/home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-ds000228.tar.gz" \
    ${OUTPUT} \
    --atlas gordon333 \
    --dimension 333 \
    --qc stringent


build_features_modularity \
    "/home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-ds000030.tar.gz" \
    ${OUTPUT} \
    --atlas gordon333 \
    --dimension 333 \
    --qc stringent

