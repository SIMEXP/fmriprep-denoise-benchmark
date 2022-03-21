#!/bin/bash
#SBATCH --job-name=metric
#SBATCH --time=12:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/metric.out
#SBATCH --error=logs/metric.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G 


OUTPUT="/home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/"
source /home/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate
cd /home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/
mkdir ${OUTPUT}

cd inputs/
tar -czvf dataset-ds000228.tar.gz dataset-ds000228/
cd ..
python ./preprocess/denoise_metrics.py ${OUTPUT} --atlas schaefer7networks --dimension 400
python ./preprocess/denoise_metrics.py ${OUTPUT} --atlas gordon333 --dimension 333
