#!/bin/bash
# Get some confounds metadata, such as numver of volumes scrubbed
# Run after generating fMRIPRep dataset
#SBATCH --job-name=meta_confounds
#SBATCH --time=1:00:00
#SBATCH --account=def-cmoreau
#SBATCH --output=logs/meta_confounds.out
#SBATCH --error=logs/meta_confounds.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G 

# source /lustre03/project/6003287/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate

# # calculate_degrees_of_freedom inputs/denoise-metrics/ds000228/fmriprep-20.2.1lts \
# # 	--fmriprep_path=/scratch/${USER}/ds000228/1643916303/fmriprep-20.2.1lts \
# # 	--dataset_name=ds000228 \
# # 	--specifier=task-pixar \
# # 	--participants_tsv /scratch/${USER}/ds000228/participants.tsv

# # calculate_degrees_of_freedom inputs/denoise-metrics/ds000030/fmriprep-20.2.1lts \
# # 	--fmriprep_path=/scratch/${USER}/ds000030/1651688951/fmriprep-20.2.1lts \
# # 	--dataset_name=ds000030 \
# # 	--specifier=task-rest \
# # 	--participants_tsv /scratch/${USER}/ds000030/participants.tsv

# # # 20.2.5
# # calculate_degrees_of_freedom inputs/denoise-metrics/ds000228/fmriprep-20.2.5lts \
# # 	--fmriprep_path=/scratch/${USER}/ds000228/1664056904/fmriprep-20.2.5lts \
# # 	--dataset_name=ds000228 \
# # 	--specifier=task-pixar \
# # 	--participants_tsv /scratch/${USER}/ds000228/participants.tsv

# # calculate_degrees_of_freedom inputs/denoise-metrics/ds000030/fmriprep-20.2.5lts \
# # 	--fmriprep_path=/scratch/${USER}/ds000030/1664058034/fmriprep-20.2.5lts \
# # 	--dataset_name=ds000030 \
# # 	--specifier=task-rest \
# # 	--participants_tsv /scratch/${USER}/ds000030/participants.tsv

source /home/seann/scratch/denoise/fmriprep-denoise-benchmark/denoise/bin/activate

# calculate_degrees_of_freedom /home/seann/scratch/denoise/fmriprep-denoise-benchmark/outputs/denoise-metrics/brain_canada_fmriprep-20.2.7lts_1732217118/fmriprep-20.2.5lts \
# 	--fmriprep_path=/home/seann/projects/def-cmoreau/All_user_common_folder/datasets/brain_canada/preprocessed_fMRI_brain\ canada/brain_canada_fmriprep-20.2.7lts_1732217118/BIDS_brain_canada/20.2.7 \
# 	--dataset_name=brain_canada_fmriprep-20.2.7lts_1732217118 \
# 	--specifier=task-rest \
# 	--participants_tsv /home/seann/scratch/denoise/fmriprep-denoise-benchmark/participants.tsv


# find /home/seann/scratch/halfpipe_test/test13/derivatives_atlascoverage0.8/fmriprep -type f -name "*_res-2_desc-preproc_bold.nii.gz" | while read file; do
#     # Remove _res-2 from the file name to construct the expected name
#     target=$(echo "$file" | sed 's/_res-2//')
#     # Create the symlink only if the target doesn't already exist
#     if [ ! -e "$target" ]; then
#         ln -s "$file" "$target"
#         echo "Created symlink: $target -> $file"
#     fi
# done

# calculate_degrees_of_freedom_test /home/seann/scratch/denoise/fmriprep-denoise-benchmark/outputs/denoise-metrics/ds000228/fmriprep-25.0.0 \
#     --fmriprep_path=/home/seann/scratch/halfpipe_test/test13/derivatives_atlascoverage0.8/fmriprep \
#     --dataset_name=ds000228 \
#     --specifier=task-pixar \
#     --participants_tsv /home/seann/projects/def-cmoreau/All_user_common_folder/datasets/ds000228/participants.tsv \


# find /home/seann/scratch/halfpipe_test/test15/derivatives_3.24.2025/fmriprep -type f -name "*_res-2_desc-preproc_bold.nii.gz" | while read file; do
#     # Remove _res-2 from the file name to construct the expected name
#     target=$(echo "$file" | sed 's/_res-2//')
#     # Create the symlink only if the target doesn't already exist
#     if [ ! -e "$target" ]; then
#         ln -s "$file" "$target"
#         echo "Created symlink: $target -> $file"
#     fi
# done

# calculate_degrees_of_freedom_test /home/seann/scratch/denoise/fmriprep-denoise-benchmark/outputs/denoise-metrics/ds000228/fmriprep-20.2.7 \
#     --fmriprep_path=/home/seann/scratch/halfpipe_test/test15/derivatives/fmriprep \
#     --dataset_name=ds000228 \
#     --specifier=task-pixar \
#     --participants_tsv /home/seann/projects/def-cmoreau/All_user_common_folder/datasets/ds000228/participants.tsv \

cd /home/seann/scratch/denoise/fmriprep-denoise-benchmark/fmriprep_denoise/features

# # Call the updated script that relies only on confounds files.
# python calculate_degrees_of_freedom_test_noboldpreproc_1.py \
#     /home/seann/scratch/denoise/fmriprep-denoise-benchmark/outputs/denoise-metrics-atlas.8-4.17.25/ds000228/fmriprep-20.2.7 \
#     --fmriprep_path=/home/seann/scratch/halfpipe_test/test15/derivatives_3.24.2025/fmriprep \
#     --dataset_name=ds000228 \
#     --specifier=task-pixar \
#     --participants_tsv /home/seann/projects/def-cmoreau/All_user_common_folder/datasets/ds000228/participants.tsv

# # Call the updated script that relies only on confounds files.
# python calculate_degrees_of_freedom_test_noboldpreproc.py \
#     /home/seann/scratch/denoise/fmriprep-denoise-benchmark/outputs/denoise-metrics/ds000228/fmriprep-25.0.0 \
#     --fmriprep_path=/home/seann/scratch/halfpipe_test/test14/derivatives/fmriprep \
#     --dataset_name=ds000228 \
#     --specifier=task-pixar \
#     --participants_tsv /home/seann/projects/def-cmoreau/All_user_common_folder/datasets/ds000228/participants.tsv

# find /home/seann/scratch/halfpipe_test/test14/derivatives/fmriprep -type f -name "*_res-2_desc-preproc_bold.nii.gz" | while read file; do
#     # Remove _res-2 from the file name to construct the expected name
#     target=$(echo "$file" | sed 's/_res-2//')
#     # Create the symlink only if the target doesn't already exist
#     if [ ! -e "$target" ]; then
#         ln -s "$file" "$target"
#         echo "Created symlink: $target -> $file"
#     fi
# done


find /home/seann/scratch/halfpipe_test/test15/derivatives/fmriprep -type f -name "*_res-2_desc-preproc_bold.nii.gz" | while read file; do
    # Remove _res-2 from the file name to construct the expected name
    target=$(echo "$file" | sed 's/_res-2//')
    # Create the symlink only if the target doesn't already exist
    if [ ! -e "$target" ]; then
        ln -s "$file" "$target"
        echo "Created symlink: $target -> $file"
    fi
done


# Call the updated script that relies only on confounds files.
python calculate_degrees_of_freedom_test_noboldpreproc_1.py \
    /home/seann/scratch/denoise/fmriprep-denoise-benchmark/outputs/denoise-metrics-atlas.5-4.27.25/ds000228/fmriprep-20.2.7 \
    --fmriprep_path=/home/seann/scratch/halfpipe_test/test15/derivatives/fmriprep \
    --dataset_name=ds000228 \
    --specifier=task-pixar \
    --participants_tsv /home/seann/projects/def-cmoreau/All_user_common_folder/datasets/ds000228/participants.tsv

# # Call the updated script that relies only on confounds files.
# python calculate_degrees_of_freedom_test_noboldpreproc_1.py \
#     /home/seann/scratch/denoise/fmriprep-denoise-benchmark/outputs/denoise-metrics-atlas.5-4.24.25/ds000228/fmriprep-25.0.0 \
#     --fmriprep_path=/home/seann/scratch/halfpipe_test/test14/derivatives/fmriprep \
#     --dataset_name=ds000228 \
#     --specifier=task-pixar \
#     --participants_tsv /home/seann/projects/def-cmoreau/All_user_common_folder/datasets/ds000228/participants.tsv