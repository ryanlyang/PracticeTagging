#!/usr/bin/env bash
set -euo pipefail

mkdir -p unmerge_new_physics_logs

# Increase dataset/constituents for this sweep
N_TRAIN_JETS=${N_TRAIN_JETS:-1000000}
MAX_CONSTITS=${MAX_CONSTITS:-100}
MAX_MERGE_COUNT=${MAX_MERGE_COUNT:-10}

declare -a RUNS=(
  "physics02_relpos_new1m|0.2|1000000|attn"
  "physics05|0.5|1000000|none"
  "physics03|0.3|1000000|none"
  "physics10|1.0|1000000|none"
  "physics05_relpos|0.5|1000000|attn"
  "physics03_relpos|0.3|1000000|attn"
  "physics10_relpos|1.0|1000000|attn"
)

for entry in "${RUNS[@]}"; do
  IFS="|" read -r RUN_NAME PHYSICS_WEIGHT OFFSET_JETS RELPOS_MODE <<< "${entry}"
  echo "Submitting ${RUN_NAME} (physics_weight=${PHYSICS_WEIGHT}, offset=${OFFSET_JETS}, relpos=${RELPOS_MODE})"
  sbatch --export=ALL,RUN_NAME="${RUN_NAME}",PHYSICS_WEIGHT="${PHYSICS_WEIGHT}",OFFSET_JETS="${OFFSET_JETS}",RELPOS_MODE="${RELPOS_MODE}",\
N_TRAIN_JETS="${N_TRAIN_JETS}",MAX_CONSTITS="${MAX_CONSTITS}",MAX_MERGE_COUNT="${MAX_MERGE_COUNT}" \
    run_unmerge_new_physics_relpos_tier3.sh
done
