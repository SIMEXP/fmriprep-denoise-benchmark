#!/bin/bash
#SBATCH --job-name=metric_gordon
#SBATCH --time=12:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/metric_gordon.out
#SBATCH --error=logs/metric_gordon.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G 


INPUTS="/home/${USER}/projects/def-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-ds000228.tar.gz"
OUTPUT="/home/${USER}/scratch/fmriprep-denoise-benchmark"

source /home/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate

cd /home/${USER}/projects/def-pbellec/${USER}/fmriprep-denoise-benchmark/

echo "gordon333"
python ./fmriprep_denoise/process_denoise_metrics.py ${INPUTS} ${OUTPUT} --atlas gordon333 --dimension 333
