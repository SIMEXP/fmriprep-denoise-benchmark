#!/bin/bash
#SBATCH --job-name=qcfc_difumo_ds000030
#SBATCH --time=24:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/qcfc_difumo_ds000030.%a.out
#SBATCH --error=logs/qcfc_difumo_ds000030.%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G 
#SBATCH --array=0-3


OUTPUT="/home/${USER}/scratch/fmriprep-denoise-benchmark"
DIMENSIONS=(64 128 256 512)
source /lustre03/project/6003287/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate

cd /home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/
DATASET=ds000030

python ./fmriprep_denoise/features/build_features_qcfc.py \
    "/home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-${DATASET}.tar.gz" \
    ${OUTPUT} \
    --atlas difumo \
    --dimension ${DIMENSIONS[${SLURM_ARRAY_TASK_ID}]} \
    --qc stringent
