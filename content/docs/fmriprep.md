# Preprocessing fMRI data

The steps here are required if you want to fully rerun the analysis.
If you are not a SIMEXP member on Aliance Canada, 
you will need to clone [fMRIPrep-slurm](https://github.com/SIMEXP/fmriprep-slurm) and modify paths accordingly.
All scripts are under `scripts/preprocessing`.
If you wish to use the produced metrics to generate the book, feel free to skip this section.

1. `preprocessing/get_fmriprep.sh`:

    We need to build the fMRIPrep singularity containers version `20.2.1` (fMRIPrep-slurm default) and `20.2.5`.
    For details, please find the documentation on [niprep](https://www.nipreps.org/apps/singularity/).
    Alternatively, you can execute the script prepared `get_fmriprep.sh`. 
    We recommend building the container in a tmux session.
    This will take a few hours!
    Run it before going home.

2. `preprocessing/get_datasets.sh`:

    While running Step 1 you can get the OpenNeuro datasets in parallel.
    Run the script in a tmux session.
    This will take an hour or so.

3. `preprocessing/create_fmriprep_slurm_scripts.sh`:
    
    After the containers are built and datasets downloaded, we generate the slurm files using [fMRIPrep-slurm](https://simexp-documentation.readthedocs.io/en/latest/giga_preprocessing/preprocessing.html).
    
    Remember to check your script to make sure everything runs correctly.
    This will take the time of a coffee break.

4. Run fmriprep
    
    fMRIPrep-slurm will give you the exact commands you need to run.
    It should be something looking like this:
    ```
    find /scratch/${USER}/ds000228/UNIXTIME/.slurm/smriprep_sub-*.sh -type f | while read file; do sbatch "$file"; done
    ```
    This process will take up a day.
    