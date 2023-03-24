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


### Option 1: Download the atlas

You can download a customised tempalteflow directory, untar into `inputs/`.

```bash
make atlas
```

### Option 2: Generate the atlas

Alternatively, you can re-run the scripts.
We will need to download from `templateflow` and `nilearn`, and organise them in `templateflow` standard.
This is a shared step for those who wants to rerun the whole workflow.
I separate the template fetching from template flowset up as the computing node of the HCP doesn't have network access.

1. Download all the atlases (require network access):

    ```bash
    python scripts/fetch_templates.py
    ```

2. Process the atlases and organise the files to templateflow standard.
    One of the measure require the node-wise distances, so we need to separate the Difumo atlas dimensions into parcels.

    ```bash
    bash scripts/setup_templateflow.sh
    ```
    If this step is computational too intensive, you can run it in on a computing node.

