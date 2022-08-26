#!/bin/bash
#SBATCH --job-name=qcfc_schaefer_ds000030
#SBATCH --time=12:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/qcfc_schaefer_ds000030.%a.out
#SBATCH --error=logs/qcfc_schaefer_ds000030.%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G 
#SBATCH --array=0-5


OUTPUT="inputs/fmrieprep-denoise-metrics"
DIMENSIONS=(100 200 300 400 500 600 800)
DATASET=ds000030

source /lustre03/project/6003287/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate

cd /home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/
    
build_features_qcfc \
    "/home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-${DATASET}.tar.gz" \
    ${OUTPUT} \
    --atlas schaefer7networks \
    --dimension ${DIMENSIONS[${SLURM_ARRAY_TASK_ID}]} \
    --qc stringent


