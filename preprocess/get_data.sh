#!/bin/bash

SCRIPT_DIR=$(readlink -e $(dirname $0))
PROJECT_DIR=$(dirname $SCRIPT_DIR)
DATASET_DIR=$PROJECT_DIR/inputs/openneuro/ds000228
N_SUBJ=20

# get probably 20 subjects for testing 
for i in $(seq -f "%03g" 1 $N_SUBJ); do 
    echo "sub-pixar$i"
    datalad get -d $DATASET_DIR -r $DATASET_DIR/sub-pixar$i
done

# preprocess with fMRIprep
