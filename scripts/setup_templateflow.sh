# set up the project after cloneing
# Move templates to locations you need them to be
#!/bin/bash

# mist
python scripts/mist2templateflow/mist2templateflow.py -d -o inputs/custome_templateflow

# difumo
python scripts/difumo_segmentation/difumo_segmentation/main.py  -o inputs/custome_templateflow
mv inputs/custome_templateflow/segmented_difumo_atlases/tpl-MNI152NLin2009cAsym/* inputs/custome_templateflow/tpl-MNI152NLin2009cAsym
python scripts/caculate_centroids.py

# clean up
rm -rf inputs/custome_templateflow/segmented_difumo_atlases/
rm -rf difumo_segmentation/data/raw/segmented_difumo_atlases
rm -rf difumo_segmentation/data/raw/icbm152_2009
