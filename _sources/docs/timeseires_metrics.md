# Generate the reports

## timeseries and metrics 

1. Create a symbolic link of the metrics to the `inputs/` directory

    ```
    mkdir $SCRATCH/fmriprep-denoise-benchmark/ 
    ```

2. `generate_timeseries_slurm_scripts.py`: 

    Extract time series with different atlases. 
    The scripts generates slurm to extract time series with different atlases. Here's the docs.

    ```
    usage: generate_timeseries_slurm_scripts.py [-h]
                                                [--slurm-account SLURM_ACCOUNT]
                                                scratch_path fmriprep_output
                                                participants_tsv virtualenv

    create timeseries extraction scripts

    positional arguments:
    scratch_path          Path to scratch space.
    fmriprep_output       Path to fMRIPrep output directory.
    participants_tsv      Path to participants.tsv in the original BIDS dataset.
    virtualenv            Path to virtual environment of this project.

    optional arguments:
    -h, --help            show this help message and exit
    --slurm-account SLURM_ACCOUNT
                            SLURM account for job submission (default: rrg-pbellec)
    ```

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

4. `generate_metrics/*/slurm/metrics*.sh`: 

    Caculate metrics on denoising quality per atlas. 
    Use this line to submit all jobs at once.

    ```bash
    find scripts/generate_metrics/slurm/metrics*.sh -type f | while read file; do sbatch $file; done
    ```

    The extra scripts in `generate_metrics/` are for running directly on a computing node.

## Build the book

To improve build time, we need to summarise the metrics further. 
If you generated the data from scratch, you will need to run the following command.

```
usage: summarise_metadata [-h]
                          [--fmriprep_version {fmriprep-20.2.1lts,fmriprep-20.2.5lts}]
                          [--dataset_name {ds000228,ds000030}]
                          [--qc {stringent,minimal,None}]
                          output_root

Summarise denoising metrics for visualization and save at the top level of the denoise metric outputs directory.

positional arguments:
  output_root           Output root path data.

optional arguments:
  -h, --help            show this help message and exit
  --fmriprep_version {fmriprep-20.2.1lts,fmriprep-20.2.5lts}
                        Path to a fmriprep dataset.
  --dataset_name {ds000228,ds000030}
                        Dataset name.
  --qc {stringent,minimal,None}
                        Automatic motion QC thresholds.
```

Now you can build the book:

```
make book
```