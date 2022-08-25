#!/bin/bash

DATASETS=("ds000030" "ds000228")
DATASET_PATH="/lustre04/scratch/${USER}/openneuro"

mkdir -p ${DATASET_PATH}
for dataset in ${DATASETS[*]}; do
    echo "Get ${dataset}"
    cd ${DATASET_PATH}
    datalad install https://github.com/OpenNeuroDatasets/${dataset}.git
    cd ${dataset}
    datalad get . -r
done
