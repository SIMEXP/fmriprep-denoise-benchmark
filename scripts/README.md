# Customised scripts for preprocessing 


The dataset and features requires HPC for the computation. Here are scripts we
used to process the current project presented in the study.

- `setup.sh`:
    Download atlases and save them in templateflow competible format.

- `slurm_timesereis/*.sh`: 
    Extract time series with different atlases. 
    The scripts were separated based on different memory requirements.
    
- `slurm_metric/*/*.sh`: 
    Caculate metrics on denoising quality per atlas.
