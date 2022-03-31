#!/bin/bash
#SBATCH --job-name=metric
#SBATCH --time=12:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/metric.out
#SBATCH --error=logs/metric.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G 


INPUTS="/home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-ds000228.tar.gz"
OUTPUT="/home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/"

source /home/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate

cd /home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/

python ./preprocess/process_denoise_metrics.py ${INPUTS} ${OUTPUT} --atlas schaefer7networks --dimension 1000
python ./preprocess/process_denoise_metrics.py ${INPUTS} ${OUTPUT} --atlas gordon333 --dimension 333
python ./preprocess/process_denoise_metrics.py ${INPUTS} ${OUTPUT} --atlas mist --dimension ROI
python ./preprocess/process_denoise_metrics.py ${INPUTS} ${OUTPUT} --atlas difumo --dimension 1024