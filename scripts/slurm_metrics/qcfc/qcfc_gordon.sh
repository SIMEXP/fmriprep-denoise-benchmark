#!/bin/bash
#SBATCH --job-name=qcfc_gordon
#SBATCH --time=12:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/qcfc_gordon.out
#SBATCH --error=logs/qcfc_gordon.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G 


OUTPUT="/home/${USER}/scratch/fmriprep-denoise-benchmark"

source /lustre03/project/6003287/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate

cd /home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/

echo "gordon333"

python ./fmriprep_denoise/features/build_features_qcfc.py \
    "/home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-ds000228.tar.gz" \
    ${OUTPUT} \
    --atlas gordon333 \
    --dimension 333 \
    --qc stringent


python ./fmriprep_denoise/features/build_features_qcfc.py \
    "/home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-ds000030.tar.gz" \
    ${OUTPUT} \
    --atlas gordon333 \
    --dimension 333 \
    --qc stringent

