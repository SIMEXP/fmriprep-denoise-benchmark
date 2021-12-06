#!/bin/bash
#SBATCH --job-name=schaefer
#SBATCH --time=12:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/schaefer.%a.out
#SBATCH --error=logs/schaefer.%a.err
#SBATCH --array=0-15 
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G 


OUTPUT="/home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/"
source /lustre03/project/6003287/hwang1/.virtualenvs/fmriprep-denoise-benchmark/bin/activate
cd /home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/
mkdir ${OUTPUT}

mapfile -t arr < <(jq -r 'keys[]' code/benchmark_strategies.json)
STRATEGY=${arr[${SLURM_ARRAY_TASK_ID}]}
echo $STRATEGY
python ./code/process_connectomes.py ${OUTPUT} --atlas schaefer7networks --strategy-name ${STRATEGY}

