#!/bin/bash
#SBATCH --job-name=connectome
#SBATCH --time=12:00:00
#SBATCH --mem=4G 
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/connectome.out
#SBATCH --error=logs/connectome.err
#SBATCH --array=0-15 

OUTPUT="/home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/"
source /lustre03/project/6003287/hwang1/.virtualenvs/fmriprep-denoise-benchmark/bin/activate
cd /home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/
mkdir ${OUTPUT}

mapfile -t arr < <(jq -r 'keys[]' code/benchmark_strategies.json)
STRATEGY=${arr[${SLURM_ARRAY_TASK_ID}]}
python ./code/process_connectomes.py ${OUTPUT} --atlas difumo --strategy-name ${STRATEGY}

