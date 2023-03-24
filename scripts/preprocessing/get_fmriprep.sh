#!/bin/bash

CONTAINER_PATH="/lustre03/project/6003287/containers"
VERSION=("20.2.1" "20.2.5")

module load singularity/3.8

echo "Check signularity containers"
for v in ${VERSION[*]}; do
    echo "fmriprep-${v}"
    current_container=${CONTAINER_PATH}/fmriprep-${v}lts.sif
    if [ -f "${current_container}" ]; then
        echo "$current_container exists."
    else
        echo "Building ${current_container}"
        singularity build ${current_container} docker://nipreps/fmriprep:${v}
    fi
done

echo "All done!"
