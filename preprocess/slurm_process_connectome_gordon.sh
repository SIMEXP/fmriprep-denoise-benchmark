#!/bin/bash
#SBATCH --job-name=gordon
#SBATCH --time=12:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/gordon.%a.out
#SBATCH --error=logs/gordon.%a.err
#SBATCH --array=0-10 
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G 


OUTPUT="/home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/inputs/dataset-ds000228/"
source /lustre03/project/6003287/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate
cd /home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/
mkdir ${OUTPUT}

mapfile -t arr < <(jq -r 'keys[]' preprocess/benchmark_strategies.json)
STRATEGY=${arr[${SLURM_ARRAY_TASK_ID}]}
echo $STRATEGY

python ./preprocess/process_connectomes.py ${OUTPUT} --atlas gordon333 --dimension 333 --strategy-name ${STRATEGY}
