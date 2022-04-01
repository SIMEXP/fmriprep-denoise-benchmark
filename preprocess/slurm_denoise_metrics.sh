#!/bin/bash
#SBATCH --job-name=metric
#SBATCH --time=12:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/metric.out
#SBATCH --error=logs/metric.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G 


INPUTS="/home/${USER}/projects/def-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-ds000228.tar.gz"
OUTPUT="/home/${USER}/scratch/fmriprep-denoise-benchmark"

source /home/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate

cd /home/${USER}/projects/def-pbellec/${USER}/fmriprep-denoise-benchmark/

echo "gordon333"
python ./fmriprep_denoise/process_denoise_metrics.py ${INPUTS} ${OUTPUT} --atlas gordon333 --dimension 333

# echo "schaefer7networks"
# for n in 100 200 300 400 500 600 800 1000; do
#     python ./fmriprep_denoise/process_denoise_metrics.py ${INPUTS} ${OUTPUT} --atlas schaefer7networks --dimension $n
# done 

# echo "mist"
# for n in 7 12 20 36 64 122 197 325 444 ROI; do
#     python ./fmriprep_denoise/process_denoise_metrics.py ${INPUTS} ${OUTPUT} --atlas mist --dimension $n
# done 

# echo "difumo"
# for n in 64 128 256 512 1024; do
#     python ./fmriprep_denoise/process_denoise_metrics.py ${INPUTS} ${OUTPUT} --atlas difumo --dimension $n
# done 
