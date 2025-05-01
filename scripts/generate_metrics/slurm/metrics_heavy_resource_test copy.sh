#!/bin/bash
#SBATCH --job-name=metrics_highdem
#SBATCH --time=24:00:00
#SBATCH --account=def-cmoreau
#SBATCH --output=logs/metrics_highdem.%a.out
#SBATCH --error=logs/metrics_highdem.%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G 
#SBATCH --array=0-3

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

DATASETS=(brain_canada_fmriprep-20.2.7lts_1732217118)
METRICS=(modularity qcfc connectome modularity)
ds=${DATASETS[${SLURM_ARRAY_TASK_ID}]}
met=${METRICS[${SLURM_ARRAY_TASK_ID}]}

echo $ds
echo $met

cd /home/seann/scratch/denoise/fmriprep-denoise-benchmark

build_features /home/seann/scratch/denoise/fmriprep-denoise-benchmark/giga_timeseries/${ds}/fmriprep-20.2.7lts/ \
    outputs/denoise-metrics/ \
    --dataset "${ds}" \
    --fmriprep_ver fmriprep-20.2.7lts \
    --atlas schaefer7networks \
    --dimension 800 \
    --qc stringent --metric ${met}