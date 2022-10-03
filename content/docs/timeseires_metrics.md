# Generate the reports

## timeseries and metrics 

1. Create a symbolic link of the metrics to the `inputs/` directory

    ```
    mkdir $SCRATCH/fmriprep-denoise-benchmark/ 
    rm -rf inputs/  # when you clone the project, this should be empty
    ln -s $SCRATCH/fmriprep-denoise-benchmark/ inputs
    ```
    This way when the book built is triggered, it will not download data from zenodo.

2. `generate_timeseries_slurm_scripts.py`: 

    Extract time series with different atlases. 
    The scripts generates slurm to extract time series with different atlases.
    We created two separate scripts for descrete and probability atlas due to different memory requrement.
    You will find the output under:
    ```
    /scratch/${USER}/fmriprep-denoise-benchmark/giga_timeseries/{DATASET_NAME}/{FMRIPREP_VERSION}/{UNIXTIME}/.slurm
    ```
    Similar to fmriprep-slurm, it will give you the exact commands you need to run.
    It should be something looking like this:
    ```
    find /scratch/${USER}/ds000228/UNIXTIME/.slurm/smriprep_sub-*.sh -type f | while read file; do sbatch "$file"; done
    ```
    This process will take a few hours.

3. `slurm_meta_confounds.sh`:

    Create files to determine which subject will enter the next stage for metric generation.

    ```bash
    sbatch slurm_metric/slurm_meta_confounds.sh
    ```

4. `slurm_metric/*/*.sh`: 

    Caculate metrics on denoising quality per atlas. 
    Use this line to submit all jobs at once.

    ```bash
    find scripts/slurm_metrics/metrics*.sh -type f | while read file; do sbatch $file; done
    ```

## Build the book

```
make book
```