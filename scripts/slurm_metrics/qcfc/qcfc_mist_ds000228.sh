#!/bin/bash
#SBATCH --job-name=qcfc_mist_ds000228
#SBATCH --time=24:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/qcfc_mist_ds000228.%a.out
#SBATCH --error=logs/qcfc_mist_ds000228.%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G 
#SBATCH --array=0-9


OUTPUT="/home/${USER}/scratch/fmriprep-denoise-benchmark"
DIMENSIONS=(7 12 20 36 64 122 197 325 444 ROI)
DATASET=ds000228

source /lustre03/project/6003287/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate

cd /home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/
    
python ./fmriprep_denoise/features/build_features_qcfc.py \
    "/home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-${DATASET}.tar.gz" \
    ${OUTPUT} \
    --atlas mist \
    --dimension ${DIMENSIONS[${SLURM_ARRAY_TASK_ID}]} \
    --qc stringent

