#!/bin/bash
# Get some confounds metadata, such as numver of volumes scrubbed
# Run after generating fMRIPRep dataset
#SBATCH --job-name=meta_confounds
#SBATCH --time=2:00:00
#SBATCH --account=rrg-pbellec
#SBATCH --output=logs/meta_confounds.out
#SBATCH --error=logs/meta_confounds.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G 

source /lustre03/project/6003287/${USER}/.virtualenvs/fmriprep-denoise-benchmark/bin/activate

calculate_degrees_of_freedom inputs/denoise-metrics/ds000228/fmrieprep-20.2.1lts \
	--fmriprep_path=/scratch/${USER}/ds000228/1643916303/fmriprep \
	--dataset_name=ds000228 \
	--specifier=task-pixar \
	--participants_tsv /scratch/${USER}/ds000228/participants.tsv

# calculate_degrees_of_freedom inputs/denoise-metrics/ds000030/fmrieprep-20.2.1lts \
# 	--fmriprep_path=/scratch/${USER}/ds000030/1651688951/fmriprep \
# 	--dataset_name=ds000030 \
# 	--specifier=task-rest \
# 	--participants_tsv /scratch/${USER}/ds000030/participants.tsv

# 20.2.5
calculate_degrees_of_freedom inputs/denoise-metrics/ds000228/fmrieprep-20.2.5lts \
	--fmriprep_path=/scratch/${USER}/ds000228/1663958770/fmriprep \
	--dataset_name=ds000228 \
	--specifier=task-pixar \
	--participants_tsv /scratch/${USER}/ds000228/participants.tsv

# calculate_degrees_of_freedom inputs/denoise-metrics/ds000030/fmrieprep-20.2.5lts \
# 	--fmriprep_path=/scratch/${USER}/ds000030/1663959923/fmriprep \
# 	--dataset_name=ds000030 \
# 	--specifier=task-rest \
# 	--participants_tsv /scratch/${USER}/ds000030/participants.tsv
