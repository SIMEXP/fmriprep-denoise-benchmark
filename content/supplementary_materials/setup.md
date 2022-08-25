# Setup to run all customised scripts for preprocessing 

The dataset and features requires HPC for the computation. 
Here are scripts we used to process the current project presented in the study.
If not otherwise specified, the scripts were executive from the root of the project.
If you are a member of the SIMEXP team, you can run the script as is;
otherwise, please install other software and modify the path and variables accordingly. 

We recommand to understand how to create and detach a tmux session before further reading.

- [tmux cheat sheet](https://tmuxcheatsheet.com/)
- [How to leave your script running in a tmux session](https://stackoverflow.com/questions/25331471/tmux-detach-while-running-script)

In most of the slurm scripts, we send SLURM job status email to slack bot, 
please see 
[here](https://simexp-documentation.readthedocs.io/en/latest/alliance_canada/hpc.html?highlight=slack#slurm-notifications-on-slack) 
to set-up your slack bot email address. 
Please modify the scripts accordingly.

## Atlases

After cloning and setting up the project, we will need to get all the atlases for the analysis.
We will need to download from `templateflow` and `nilearn`, and organise them in `templateflow` standard.

1. Download all the atlases (require network access):

    ```bash
    python scripts/fetch_templates.py
    ```
    Or you can use the make command:
    
    ```bash
    make atlas
    ```

2. Process the atlases and organise the files to templateflow standard.
    One of the measure require the node-wise distances, so we need to separate the Difumo atlas dimensions into parcels.

    ```bash
    bash scripts/setup_templateflow.sh
    ```
    Or the Make command:
    
    ```bash
    make templateflow
    ```
    If this step is computational too intensive, you can run it in on a computing node.


## Preprocessing fMRI data

The steps here are required if you want to fully rerun the analysis.
If you are not a SIMEXP member on Aliance Canada, 
you will need to clone [fMRIPrep-slurm](https://github.com/SIMEXP/fmriprep-slurm) and modify paths accordingly.
All scripts are under `scripts/preprocessing`.
If you wish to use the produced metrics to generate the book, feel free to skip this section.

1. `preprocessing/get_fmriprep.sh`:

    We need to build the fMRIPrep singularity containers version `20.2.1` (fMRIPrep-slurm default) and `20.2.7`.
    For details, please find the documentation on [niprep](https://www.nipreps.org/apps/singularity/).
    Alternatively, you can execute the script prepared `get_fmriprep.sh`. 
    We recommand building the container in a tmux session.
    This will take a few hours!
    Run it before going home.

2. `preprocessing/get_datasets.sh`:

    While runing Step 1 you can get the OpenNeuro datasets in parallel.
    Run the script in a tmux session.
    This will take an hour or so.

3. `preprocessing/create_fmriprep_slurm_scripts.sh`:
    
    After the containers are built and datasets downloaded, we generate the slurm files using [fMRIPrep-slurm](https://simexp-documentation.readthedocs.io/en/latest/giga_preprocessing/preprocessing.html).
    
    Remember to check your script to make sure everything runs correctly.
    This will take the time of a coffee break.

## Generate timeseries and metrics 
    
3. `slurm_timesereis/*.sh`: 

    Extract time series with different atlases. 
    The scripts were separated based on different memory requirements.
    Use this line to submit all jobs at once.

    ```bash
    find scripts/slurm_timeseries/*.sh -type f | while read file; do sbatch $file; done
    ```
    If you get an error along the line of `Job violates accounting/QOS policy`,
    Submit batches by dataset.

4. `slurm_metric/slurm_meta_confounds.sh`:

    Create files to determine which subject will enter the next stage for metric generation.

    ```bash
    sbatch scripts/slurm_metric/slurm_meta_confounds.sh
    ```

5. `slurm_metric/*/*.sh`: 
    Caculate metrics on denoising quality per atlas. 
    Use this line to submit all jobs at once.

    ```bash
    find scripts/slurm_metrics/*/*.sh -type f | while read file; do sbatch $file; done
    ```
