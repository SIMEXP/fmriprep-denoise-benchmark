# Move templates to locations you need them to be
#!/bin/bash


cd /home/${USER}/projects/def-pbellec/${USER}/fmriprep-denoise-benchmark

python scripts/fetch_templates.py


python scripts/mist2templateflow/mist2templateflow.py -d -o inputs/custome_templateflow

python scripts/difumo_segmentation/difumo_segmentation/main.py  -o inputs/custome_templateflow
mv inputs/custome_templateflow/segmented_difumo_atlases/tpl-MNI152NLin2009cAsym/* inputs/custome_templateflow/tpl-MNI152NLin2009cAsym

# clean up
rm -rf inputs/custome_templateflow/segmented_difumo_atlases/
rm -rf difumo_segmentation/data/raw/segmented_difumo_atlases
rm -rf difumo_segmentation/data/raw/icbm152_2009
