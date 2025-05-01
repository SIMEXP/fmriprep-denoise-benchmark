#!/bin/bash
# Get some confounds metadata, such as numver of volumes scrubbed
# Run after generating fMRIPRep dataset
#SBATCH --job-name=9transition_timeseries
#SBATCH --time=1:00:00
#SBATCH --account=def-cmoreau
#SBATCH --output=logs/9transition_timeseries.out
#SBATCH --error=logs/9transition_timeseries.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G 

source /home/seann/scratch/denoise/fmriprep-denoise-benchmark/denoise/bin/activate

cd /home/seann/scratch/denoise/fmriprep-denoise-benchmark/scripts/halfpipe_to_denoise

# python transition_timeseries.py \
#     --input_dir /home/seann/scratch/halfpipe_test/test14/derivatives_3.25.2025/halfpipe \
#     --output_dir /home/seann/scratch/halfpipe_test/test14/derivatives_3.25.2025/denoise \
#     --task pixar \
#     --space MNI152NLin2009cAsym \
#     --nroi 434 

python transition_timeseries_imputation.py \
    --input_dir /home/seann/scratch/halfpipe_test/test15/derivatives_3.24.2025/halfpipe \
    --output_dir /home/seann/scratch/halfpipe_test/test15/derivatives_3.24.2025/denoise_0.8subjectthreshold \
    --task pixar \
    --space MNI152NLin2009cAsym \
    --nroi 434 \
    --atlas /home/seann/projects/def-cmoreau/All_user_common_folder/atlas/atlas_enigma/atlas-Schaefer2018Combined_dseg.tsv \
    --nan_threshold 0.5