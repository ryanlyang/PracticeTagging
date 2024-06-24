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
datapath="/DFS-L/DATA/whiteson/kgreif/JetTaggingH5"
datasets=(
    "${datapath}/public_test_nominal" "--store_numbers --store_weights"
    "${datapath}/public_esup" "--store_numbers"
    "${datapath}/public_esdown" "--store_numbers"
    "${datapath}/public_cer" "--store_numbers"
    "${datapath}/public_cpos" "--store_numbers"
    "${datapath}/public_eff" "--store_numbers"
    "${datapath}/public_teg" "--store_numbers"
    "${datapath}/public_tej" "--store_numbers"
    "${datapath}/public_tfl" "--store_numbers"
    "${datapath}/public_tfj" "--store_numbers"
    "${datapath}/public_tbias" "--store_numbers"
    "${datapath}/public_ttbar_pythia" ""
    "${datapath}/public_ttbar_herwig" ""
    "${datapath}/public_cluster" ""
    "${datapath}/public_string" ""
    "${datapath}/public_angular" ""
    "${datapath}/public_dipole" ""
)

# Loop over datasets
for ((i=0; i<${#datasets[@]}; i+=2))
do
    dataset="${datasets[i]}"
    flags="${datasets[i+1]}"
    echo -e "\nEvaluating $dataset..."
    python evaluate.py --data "$dataset" --checkpoint "$checkpoint" --type "$model_type" $flags
done
