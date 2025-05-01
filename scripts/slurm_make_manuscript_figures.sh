#!/bin/bash
# Get some confounds metadata, such as numver of volumes scrubbed
# Run after generating fMRIPRep dataset
#SBATCH --job-name=make_manuscript_figures
#SBATCH --time=1:00:00
#SBATCH --account=def-cmoreau
#SBATCH --output=logs/make_manuscript_figures.out
#SBATCH --error=logs/make_manuscript_figures.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G 

source /home/seann/scratch/denoise/fmriprep-denoise-benchmark/denoise/bin/activate

cd /home/seann/scratch/denoise/fmriprep-denoise-benchmark/scripts

python make_manuscript_figures.py

# python make_manuscript_figures_multifmriprep.py

# python plot.py