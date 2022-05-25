#!/bin/bash
#SBATCH --job-name=package
#SBATCH --time=24:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/archive.%a.out
#SBATCH --error=logs/archive.%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G 
#SBATCH --array=0-3


SOURCE=(
    "/scratch/hwang1/giga_timeseries/dataset-ds000228/" \
    "/scratch/hwang1/giga_timeseries/dataset-ds000030/" \
    "/scratch/hwang1/ds000228/" \
    "/scratch/hwang1/ds000030/"
)

TARGET=(
    "/lustre06/nearline/6035398/hwang1/fmriprep-denoise-benchmark/giga_timeseries/dataset-ds000228.tar.gz" \
    "/lustre06/nearline/6035398/hwang1/fmriprep-denoise-benchmark/giga_timeseries/dataset-ds000030.tar.gz" \
    "/lustre06/nearline/6035398/hwang1/ds000228_1643916303.tar.gz" \
    "/lustre06/nearline/6035398/hwang1/ds000030_1651688951.tar.gz"
)

cd ${SOURCE[${SLURM_ARRAY_TASK_ID}]}
tar -czf ${TARGET[${SLURM_ARRAY_TASK_ID}]} .
