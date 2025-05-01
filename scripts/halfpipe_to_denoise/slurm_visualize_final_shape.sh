#!/bin/bash
# Get some confounds metadata, such as the number of volumes scrubbed
# Run after generating the fMRIPrep dataset

#SBATCH --job-name=visualize_final_shape
#SBATCH --time=1:00:00
#SBATCH --account=def-cmoreau
#SBATCH --output=logs/visualize_final_shape.out
#SBATCH --error=logs/visualize_final_shape.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G 

# Activate the Python virtual environment.
source /home/seann/scratch/denoise/fmriprep-denoise-benchmark/denoise/bin/activate

# Change to the directory containing the visualization script.
cd /home/seann/scratch/denoise/fmriprep-denoise-benchmark/scripts/halfpipe_to_denoise

# Run the visualization script with all required arguments.
python visualize_final_shape.py \
    --atlas_img /home/seann/projects/def-cmoreau/All_user_common_folder/atlas/atlas_enigma/atlas-Schaefer2018Combined_dseg.nii.gz \
    --atlas_tsv /home/seann/projects/def-cmoreau/All_user_common_folder/atlas/atlas_enigma/atlas-Schaefer2018Combined_dseg.tsv \
    --exclusion_csv /home/seann/scratch/halfpipe_test/test15/derivatives/denoise_0.8subjectthreshold/final_roi_labels.csv \
    --global_impute_csv /home/seann/scratch/halfpipe_test/test15/derivatives/denoise_0.8subjectthreshold/roi_global_impute_counts.csv \
    --output /home/seann/scratch/denoise/fmriprep-denoise-benchmark/scripts/halfpipe_to_denoise/visualize_final_shape.png \
    --title "Final ROIs for 1.2.3lts"