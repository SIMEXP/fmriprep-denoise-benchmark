#!/bin/bash
#SBATCH --job-name=metrics_modularity
#SBATCH --time=36:00:00
#SBATCH --account=def-cmoreau
#SBATCH --output=logs/metrics_modularity.%a.out
#SBATCH --error=logs/metrics_modularity.%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G 

OUTPUT="outputs/denoise-metrics/"

# echo "gordon333"

# for ds in ds000030 ds000228; do
#     build_features inputs/giga_timeseries/${ds}/fmriprep-20.2.1lts/ \
#         ${OUTPUT} \
#         --atlas gordon333 --dimension 333 --qc stringent --metric modularity
# done

# echo "mist"
# for N in 7 12 20 36 64 122 197 325 444 ROI; do 
#     for ds in ds000030 ds000228; do
#         build_features inputs/giga_timeseries/${ds}/fmriprep-20.2.1lts/ \
#             ${OUTPUT} \
#             --atlas mist --dimension $N --qc stringent --metric modularity
#     done
# done 

# echo "schaefer7networks"
# for N in 100 200 300 400 500 600 800; do 
#     for ds in ds000030 ds000228; do
#         build_features inputs/giga_timeseries/${ds}/fmriprep-20.2.1lts/ \
#             ${OUTPUT} \
#             --atlas schaefer7networks --dimension $N --qc stringent --metric modularity
#     done
# done 

# echo "difumo"
# for N in 64 128 256 512 1024; do 
#     for ds in ds000030 ds000228; do
#         build_features inputs/giga_timeseries/${ds}/fmriprep-20.2.1lts/ \
#             ${OUTPUT} \
#             --atlas difumo --dimension $N --qc stringent --metric modularity
#     done
# done 

source /home/seann/scratch/denoise/fmriprep-denoise-benchmark/denoise/bin/activate


echo "Schaefer2018Combined"

build_features_test /home/seann/scratch/halfpipe_test/test14/derivatives_3.24.2025/denoise/ \
    outputs/denoise-metrics/ \
    --dataset "ds000228" \
    --fmriprep_ver fmriprep-25.0.0 \
    --atlas Schaefer2018 \
    --dimension 434 \
    --qc minimal --metric modularity