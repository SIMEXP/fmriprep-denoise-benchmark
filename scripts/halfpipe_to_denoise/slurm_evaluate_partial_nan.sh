#!/bin/bash
# Get some confounds metadata, such as numver of volumes scrubbed
# Run after generating fMRIPRep dataset
#SBATCH --job-name=evaluate_partial_nan
#SBATCH --time=2:00:00
#SBATCH --account=def-cmoreau
#SBATCH --output=logs/evaluate_partial_nan.out
#SBATCH --error=logs/evaluate_partial_nan.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G 

source /home/seann/scratch/denoise/fmriprep-denoise-benchmark/denoise/bin/activate

cd /home/seann/scratch/denoise/fmriprep-denoise-benchmark/scripts/halfpipe_to_denoise


python evaluate_partial_nan.py \
    --input_dir /home/seann/scratch/halfpipe_test/test15/derivatives_3.24.2025/halfpipe \
    --output_dir /home/seann/scratch/denoise/fmriprep-denoise-benchmark/scripts/halfpipe_to_denoise