# Customised scripts for preprocessing 


The dataset and features requires HPC for the computation. Here are scripts we
used to process the current project presented in the study.

- `setup.sh`:
    Install all dependencies.
    Download atlases and save them in templateflow competible format.
    
- `slurm_timesereis/*.sh`: 
    Extract time series with different atlases. 
    The scripts were separated based on different memory requirements.
    Use this line to submit all jobs at once.
    ```bash
    find scripts/slurm_timeseries/*/*.sh -type f | while read file; do sbatch $file; done
    ```

- `slurm_metric/slurm_meta_confounds.sh`:
    Create files to determine which subject will enter the next stage for metric generation.

- `slurm_metric/*/*.sh`: 
    Caculate metrics on denoising quality per atlas. 
    Use this line to submit all jobs at once.
    ```bash
    find scripts/slurm_metrics/*/*.sh -type f | while read file; do sbatch $file; done
    ```