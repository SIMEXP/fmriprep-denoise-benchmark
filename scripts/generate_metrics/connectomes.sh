# Generate connectomes. Run on interactive nodes.
#!/bin/bash

OUTPUT="inputs/denoise-metrics/"

echo "gordon333"

for ds in ds000030 ds000228; do
    build_features inputs/giga_timeseries/${ds}/fmriprep-20.2.1lts/ \
        ${OUTPUT} \
        --atlas gordon333 --dimension 333 --qc stringent --metric connectome
done

echo "mist"
for N in 7 12 20 36 64 122 197 325 444 ROI; do 
    for ds in ds000030 ds000228; do
        build_features inputs/giga_timeseries/${ds}/fmriprep-20.2.1lts/ \
            ${OUTPUT} \
            --atlas mist --dimension $N --qc stringent --metric connectome
    done
done 

echo "schaefer7networks"
for N in 100 200 300 400 500 600 800; do 
    for ds in ds000030 ds000228; do
        build_features inputs/giga_timeseries/${ds}/fmriprep-20.2.1lts/ \
            ${OUTPUT} \
            --atlas schaefer7networks --dimension $N --qc stringent --metric connectome
    done
done 

echo "difumo"
for N in 64 128 256 512 1024; do 
    for ds in ds000030 ds000228; do
        build_features inputs/giga_timeseries/${ds}/fmriprep-20.2.1lts/ \
            ${OUTPUT} \
            --atlas difumo --dimension $N --qc stringent --metric connectome
    done
done 