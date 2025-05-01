#!/bin/bash
#SBATCH --job-name=0.5_0.8_lts_metrics_highdem
#SBATCH --time=36:00:00
#SBATCH --account=def-cmoreau
#SBATCH --output=logs/0.5_0.8_lts_metrics_highdem.%a.out
#SBATCH --error=logs/0.5_0.8_lts_metrics_highdem.%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G 
#SBATCH --array=0-2

# source /lustre03/project/6003287/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate

# DATASETS=(ds000030 ds000228 ds000030 ds000228)
# METRICS=(modularity qcfc qcfc modularity)
# ds=${DATASETS[${SLURM_ARRAY_TASK_ID}]}
# met=${METRICS[${SLURM_ARRAY_TASK_ID}]}

# echo $ds
# echo $met

# cd /home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/

# build_features inputs/giga_timeseries/${ds}/fmriprep-20.2.5lts/ \
#     inputs/denoise-metrics/ \
#     --atlas difumo --dimension 1024 --qc stringent --metric ${met}

# build_features inputs/giga_timeseries/${ds}/fmriprep-20.2.5lts/ \
#     inputs/denoise-metrics/ \
#     --atlas schaefer7networks \
#     --dimension 800 \
#     --qc stringent --metric ${met}


source /home/seann/scratch/denoise/fmriprep-denoise-benchmark/denoise/bin/activate

DATASETS=(ds000228 ds000228 ds000228)
METRICS=(qcfc modularity connectome)
ds=${DATASETS[${SLURM_ARRAY_TASK_ID}]}
met=${METRICS[${SLURM_ARRAY_TASK_ID}]}

echo $ds
echo $met

cd /home/seann/scratch/denoise/fmriprep-denoise-benchmark



# build_features_test /home/seann/scratch/halfpipe_test/test14/derivatives/denoise_0.8subjectthreshold/ \
#     outputs/denoise-metrics-atlas.5-4.27.25 \
#     --dataset "ds000228" \
#     --fmriprep_ver fmriprep-25.0.0 \
#     --atlas Schaefer2018 \
#     --dimension 424 \
#     --qc minimal --metric ${met}

# build_features_test /home/seann/scratch/halfpipe_test/test14/derivatives_3.25.2025/denoise_0.8subjectthreshold \
#     outputs/denoise-metrics-atlas.8-4.17.25 \
#     --dataset "ds000228" \
#     --fmriprep_ver fmriprep-25.0.0 \
#     --atlas Schaefer2018 \
#     --dimension 400 \
#     --qc minimal --metric ${met}

# build_features_test /home/seann/scratch/halfpipe_test/test15/derivatives_3.24.2025/denoise_0.8subjectthreshold \
#     outputs/denoise-metrics-atlas.8-4.17.25 \
#     --dataset "ds000228" \
#     --fmriprep_ver fmriprep-20.2.7 \
#     --atlas Schaefer2018 \
#     --dimension 412 \
#     --qc minimal --metric ${met}

build_features_test /home/seann/scratch/halfpipe_test/test15/derivatives/denoise_0.8subjectthreshold \
    outputs/denoise-metrics-atlas.5-4.24.25 \
    --dataset "ds000228" \
    --fmriprep_ver fmriprep-20.2.7 \
    --atlas Schaefer2018 \
    --dimension 426 \
    --qc minimal --metric ${met}

