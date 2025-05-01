#!/bin/bash
#SBATCH --job-name=evaluate_NaN
#SBATCH --time=4:00:00
#SBATCH --account=def-cmoreau
#SBATCH --output=logs/evaluate_NaN.%a.out
#SBATCH --error=logs/evaluate_NaN.%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G 
#SBATCH --array=0


source /home/seann/scratch/denoise/fmriprep-denoise-benchmark/denoise/bin/activate

# DATASETS=(ds000228)
# METRICS=(modularity qcfc connectome)
# ds=${DATASETS[${SLURM_ARRAY_TASK_ID}]}
# met=${METRICS[${SLURM_ARRAY_TASK_ID}]}

# echo $ds
# echo $met

# cd /home/seann/scratch/denoise/fmriprep-denoise-benchmark

# evaluate_NaN /home/seann/scratch/halfpipe_test/test13/derivatives_atlascoverage0.8/halfpipe \
#     outputs/denoise-metrics \
#     --dataset "ds000228" \
#     --fmriprep_ver fmriprep-25.0.0 \
#     --atlas Schaefer2018 \
#     --dimension 434 \
#     --qc minimal 

# evaluate_NaN /home/seann/scratch/halfpipe_test/test10_v2/derivatives_atlascoverage0.8/halfpipe \
#     outputs/denoise-metrics \
#     --dataset "ds000228" \
#     --fmriprep_ver fmriprep-20.2.7 \
#     --atlas Schaefer2018 \
#     --dimension 434 \
#     --qc minimal 


# Halfpipe version A path and its fMRIPrep version.
VERSION_A="/home/seann/scratch/halfpipe_test/test14/derivatives/halfpipe"
FMRIPREP_VER_A="fmriprep-25.0.0"

# Halfpipe version B path and its fMRIPrep version.
VERSION_B="/home/seann/scratch/halfpipe_test/test15/derivatives/halfpipe"
FMRIPREP_VER_B="fmriprep-20.2.7"

# Set the output directory.
OUTPUT_DIR="/home/seann/scratch/denoise/fmriprep-denoise-benchmark/outputs/denoise-metrics/ds000228/brain_visualization"

# Set atlas, dimension, dataset, and pipeline.
ATLAS="Schaefer2018"
DIMENSION="434"
DATASET="ds000228"
PIPELINE="corrMatrixMotion"

# Path to the atlas image (NIfTI) to compute centroids.
ATLAS_IMG="/home/seann/projects/def-cmoreau/All_user_common_folder/atlas/atlas_enigma/atlas-Schaefer2018Combined_dseg.nii.gz"

# Optional: specify the colormap for brain plots.
COLORMAP="Reds"

evaluate_NaN \
    "$VERSION_A" \
    "$OUTPUT_DIR" \
    --atlas "$ATLAS" \
    --dimension "$DIMENSION" \
    --dataset "$DATASET" \
    --fmriprep_ver_a "$FMRIPREP_VER_A" \
    --version_b "$VERSION_B" \
    --fmriprep_ver_b "$FMRIPREP_VER_B" \
    --pipeline "$PIPELINE" \
    --atlas_img "$ATLAS_IMG" \
    --colormap "$COLORMAP" \
    --confounds_root "/home/seann/scratch/halfpipe_test/test14/derivatives/fmriprep" \
    --task "task-pixar" \
    --fd_threshold 0.5 \
    --exclude_file "/home/seann/projects/def-cmoreau/All_user_common_folder/datasets/ds000228/visual_qc_excude_list"