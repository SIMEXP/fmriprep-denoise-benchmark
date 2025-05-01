#!/bin/bash
# Evaluate imputation methods across all subject timeseries
#SBATCH --job-name=impute_eval
#SBATCH --time=1:00:00
#SBATCH --account=def-cmoreau
#SBATCH --output=logs/impute_eval.out
#SBATCH --error=logs/impute_eval.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Activate environment
source /home/seann/scratch/denoise/fmriprep-denoise-benchmark/denoise/bin/activate

# Navigate to script location
cd /home/seann/scratch/denoise/fmriprep-denoise-benchmark/scripts/imputation_testing

# # Run the evaluation script
# python test_imputation_methods.py \
#     --base_folder /home/seann/scratch/denoise/fmriprep-denoise-benchmark/giga_timeseries/brain_canada_fmriprep-20.2.7lts_1732217118/fmriprep-20.2.7lts/atlas-schaefer7networks \
#     --missing_rate 0.1 \
#     --missing_profile_tsv /home/seann/scratch/denoise/fmriprep-denoise-benchmark/outputs/denoise-metrics/ds000228_04.03.25/brain_visualization/ds000228/fmriprep-25.0.0/version_A_all_subjects_avg_missing_per_roi.tsv \
#     --output_file /home/seann/scratch/denoise/fmriprep-denoise-benchmark/scripts/imputation_testing/imputation_metrics_comparison.csv \
#     --visualize_output /home/seann/scratch/denoise/fmriprep-denoise-benchmark/scripts/imputation_testing/visualization



# python test_imputation_methods.py \
#     --base_folder /home/seann/scratch/halfpipe_test/test15/derivatives_3.24.2025/denoise \
#     --missing_rate 0.1 \
#     --missing_profile_tsv /home/seann/scratch/denoise/fmriprep-denoise-benchmark/outputs/denoise-metrics/ds000228_04.03.25/brain_visualization/ds000228/fmriprep-25.0.0/version_A_all_subjects_avg_missing_per_roi.tsv \
#     --output_file /home/seann/scratch/denoise/fmriprep-denoise-benchmark/scripts/imputation_testing/imputation_metrics_comparison.csv \
#     --visualize_output /home/seann/scratch/denoise/fmriprep-denoise-benchmark/scripts/imputation_testing/visualization

python test_imputation_methods.py \
    --base_folder /home/seann/scratch/halfpipe_test/test15/derivatives_3.24.2025/denoise \
    --missing_rate 0.1 \
    --output_file /home/seann/scratch/denoise/fmriprep-denoise-benchmark/scripts/imputation_testing/imputation_metrics_comparison.csv \
    --visualize_output /home/seann/scratch/denoise/fmriprep-denoise-benchmark/scripts/imputation_testing/visualization