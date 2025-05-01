#!/bin/bash
# Get some confounds metadata, such as numver of volumes scrubbed
# Run after generating fMRIPRep dataset
#SBATCH --job-name=summarise_metadata
#SBATCH --time=1:00:00
#SBATCH --account=def-cmoreau
#SBATCH --output=logs/summarise_metadata.out
#SBATCH --error=logs/summarise_metadata.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G 

source /home/seann/scratch/denoise/fmriprep-denoise-benchmark/denoise/bin/activate

cd /home/seann/scratch/denoise/fmriprep-denoise-benchmark/fmriprep_denoise/visualization

# python summarise_metadata.py /home/seann/scratch/denoise/fmriprep-denoise-benchmark/outputs/denoise-metrics --fmriprep_version fmriprep-20.2.7 --dataset_name ds000228 --qc minimal

python summarise_metadata.py /home/seann/scratch/denoise/fmriprep-denoise-benchmark/outputs/denoise-metrics-atlas.5-4.27.25 --fmriprep_version fmriprep-20.2.7  --dataset_name ds000228 --qc minimal
