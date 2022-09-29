#!/bin/bash
#SBATCH --job-name=qcfc_highdem
#SBATCH --time=24:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/qcfc_highdem.%a.out
#SBATCH --error=logs/qcfc_highdem.%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G 
#SBATCH --array=0-1


source /lustre03/project/6003287/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate
DATASET=(ds000030 ds000228)

cd /home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/

build_features inputs/giga_timeseries/${DATASET[${SLURM_ARRAY_TASK_ID}]}/fmriprep-20.2.1lts/ \
    inputs/denoise-metrics/ \
    --atlas difumo --dimension 1024 --qc stringent


build_features inputs/giga_timeseries/${DATASET[${SLURM_ARRAY_TASK_ID}]}/fmriprep-20.2.1lts/ \
    inputs/denoise-metrics/ \
    --atlas schaefer7networks \
    --dimension 800 \
    --qc stringent
    