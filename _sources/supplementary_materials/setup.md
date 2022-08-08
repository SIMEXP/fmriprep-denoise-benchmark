# Setup to run all customised scripts for preprocessing 

The dataset and features requires HPC for the computation. 
Here are scripts we used to process the current project presented in the study.
If not otherwise specified, the scripts were executive from the root of the project.

0. Preprocess data with fMRIPrep.
    We generate the slurm files using [fMRIPrep-slurm](https://simexp-documentation.readthedocs.io/en/latest/giga_preprocessing/preprocessing.html).
    
    ```bash
    fmriprep-slurm/singularity_run.bash \
        PATH/TO/BIDS/DATASET \
        fmriprep \
        --fmriprep-args=\"--use-aroma\" \
        --email=$SLACK_EMAIL_BOT
    ```
    See here to learn about [`$SLACK_EMAIL_BOT`](https://simexp-documentation.readthedocs.io/en/latest/alliance_canada/hpc.html?highlight=slack#slurm-notifications-on-slack)

1. `setup.sh`:
    Install all dependencies.
    Download atlases and save them in templateflow competible format.
    
2. `slurm_timesereis/*.sh`: 
    Extract time series with different atlases. 
    The scripts were separated based on different memory requirements.
    Use this line to submit all jobs at once.

    ```bash
    find scripts/slurm_timeseries/*.sh -type f | while read file; do sbatch $file; done
    ```
    If you get an error along the line of `Job violates accounting/QOS policy`,
    Submit batches by dataset.

3. `slurm_metric/slurm_meta_confounds.sh`:
    Create files to determine which subject will enter the next stage for metric generation.

    ```bash
    sbatch scripts/slurm_metric/slurm_meta_confounds.sh
    ```

4. `slurm_metric/*/*.sh`: 
    Caculate metrics on denoising quality per atlas. 
    Use this line to submit all jobs at once.

    ```bash
    find scripts/slurm_metrics/*/*.sh -type f | while read file; do sbatch $file; done
    ```
