# Customised scripts for preprocessing 


The dataset and features requires HPC for the computation. Here are scripts we
used to process the current project presented in the study.

- `download_atlases.sh`:
    Download atlases and save them in templateflow competible format.

- `slurm_[dataset name]_timesereis_*.sh`: 
    Extract time series with different atlases. 
    The scripts were separated based on different memory requirements.
- `slurm_[dataset name]_metric_[atlas].sh`: 
    Caculate metrics on denoising quality per atlas.
