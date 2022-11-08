#!/bin/bash
#SBATCH --job-name=metrics_lowdim
#SBATCH --time=24:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/metrics_lowdim.%a.out
#SBATCH --error=logs/metrics_lowdim.%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G 
#SBATCH --array=0-3


source /lustre03/project/6003287/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate
cd /home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/
OUTPUT=inputs/denoise-metrics/

DATASETS=(ds000030 ds000228 ds000030 ds000228)
METRICS=(modularity qcfc qcfc modularity)
ds=${DATASETS[${SLURM_ARRAY_TASK_ID}]}
met=${METRICS[${SLURM_ARRAY_TASK_ID}]}

echo $ds
echo $met

echo "schaefer7networks"
for N in 600 100 200 300 400 500; do 
    build_features inputs/giga_timeseries/${ds}/fmriprep-20.2.5lts/ \
        ${OUTPUT} \
        --atlas schaefer7networks --dimension $N --qc stringent --metric ${met}
done 

echo "difumo"
for N in 512 64 128 256 512; do 
    build_features inputs/giga_timeseries/${ds}/fmriprep-20.2.5lts/ \
        ${OUTPUT} \
        --atlas difumo --dimension $N --qc stringent --metric ${met}
done 

echo "gordon333"
build_features inputs/giga_timeseries/${ds}/fmriprep-20.2.5lts/ \
    ${OUTPUT} \
    --atlas gordon333 --dimension 333 --qc stringent --metric ${met}
echo "mist"

for N in 325 444 ROI 7 12 20 36 64 122 197; do 
    build_features inputs/giga_timeseries/${ds}/fmriprep-20.2.5lts/ \
        ${OUTPUT} \
        --atlas mist --dimension $N --qc stringent --metric ${met}
done 