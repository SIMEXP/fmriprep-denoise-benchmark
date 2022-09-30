#!/bin/bash
#SBATCH --job-name=qcfc_gordon
#SBATCH --time=24:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/qcfc_gordon.out
#SBATCH --error=logs/qcfc_gordon.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G 


source /lustre03/project/6003287/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate
cd /home/${USER}/projects/rrg-pbellec/${USER}/fmriprep-denoise-benchmark/
OUTPUT=inputs/denoise-metrics/

echo "gordon333"
build_features inputs/giga_timeseries/ds000030/fmriprep-20.2.1lts/ \
    ${OUTPUT} \
    --atlas gordon333 --dimension 333 --qc stringent
build_features inputs/giga_timeseries/ds000228/fmriprep-20.2.1lts/ \
    ${OUTPUT} \
    --atlas gordon333 --dimension 333 --qc stringent

echo "mist"
for N in 7 12 20 36 64 122 197 325 444 ROI; do 
    build_features inputs/giga_timeseries/ds000228/fmriprep-20.2.1lts/ \
        ${OUTPUT} \
        --atlas mist --dimension $N --qc stringent
    build_features inputs/giga_timeseries/ds000030/fmriprep-20.2.1lts/ \
        ${OUTPUT} \
        --atlas mist --dimension $N --qc stringent
done 

echo "schaefer7networks"
for N in 100 200 300 400 500 600; do 
    build_features inputs/giga_timeseries/ds000228/fmriprep-20.2.1lts/ \
        ${OUTPUT} \
        --atlas schaefer7networks --dimension $N --qc stringent
    build_features inputs/giga_timeseries/ds000030/fmriprep-20.2.1lts/ \
        ${OUTPUT} \
        --atlas schaefer7networks --dimension $N --qc stringent
done 

echo "difumo"
for N in 64 128 256 512; do 
    build_features inputs/giga_timeseries/ds000228/fmriprep-20.2.1lts/ \
        ${OUTPUT} \
        --atlas difumo --dimension $N --qc stringent
    build_features inputs/giga_timeseries/ds000030/fmriprep-20.2.1lts/ \
        ${OUTPUT} \
        --atlas difumo --dimension $N --qc stringent
done 
