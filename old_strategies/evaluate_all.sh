#!/bin/bash

# Check number of command line arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <checkpoint> <model_type>"
    exit 1
fi

# Model checkpoint and type
checkpoint=$1
model_type=$2

# List of datasets, comment any out to exclude them
datapath="/pub/kgreif/samples/h5dat"
datasets=(
    "${datapath}/public_test_nominal"
    "${datapath}/public_esup"
    "${datapath}/public_esdown"
    "${datapath}/public_cer"
    "${datapath}/public_cpos"
    "${datapath}/public_teg"
    "${datapath}/public_tej"
    "${datapath}/public_tfl"
    "${datapath}/public_tfj"
    "${datapath}/public_bias"
    "${datapath}/public_ttbar_pythia"
    "${datapath}/public_ttbar_herwig"
    "${datapath}/public_cluster"
    "${datapath}/public_string"
    "${datapath}/public_angular"
    "${datapath}/public_dipole"
)

# Loop over datasets
for ((i=0; i<${#datasets[@]}; i+=1))
do
    dataset="${datasets[i]}"
    echo -e "\nEvaluating $dataset..."
    python evaluate.py --data "$dataset" --checkpoint "$checkpoint" --type "$model_type"
done
