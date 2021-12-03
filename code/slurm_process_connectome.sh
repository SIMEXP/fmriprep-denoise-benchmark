#!/bin/bash
#SBATCH --job-name=connectome
#SBATCH --time=12:00:00
#SBATCH --mem=4G 
#SBATCH --account=rrg-pbellec
#SBATCH --output={jobname}.out
#SBATCH --error={jobname}.err


OUTPUT="/${HOME}/scratch/test"

workon fmriprep-denoise-benchmark

python code/process_connectome.py ${OUTPUT} --atlas difumo 

