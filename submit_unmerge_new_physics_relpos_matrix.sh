#!/usr/bin/env bash
set -euo pipefail

mkdir -p unmerge_new_physics_logs

declare -a RUNS=(
  "physics02_relpos_new200k|0.2|200000|attn"
  "physics05|0.5|0|none"
  "physics03|0.3|0|none"
  "physics10|1.0|0|none"
  "physics05_relpos|0.5|0|attn"
  "physics03_relpos|0.3|0|attn"
  "physics10_relpos|1.0|0|attn"
)

for entry in "${RUNS[@]}"; do
  IFS="|" read -r RUN_NAME PHYSICS_WEIGHT OFFSET_JETS RELPOS_MODE <<< "${entry}"
  echo "Submitting ${RUN_NAME} (physics_weight=${PHYSICS_WEIGHT}, offset=${OFFSET_JETS}, relpos=${RELPOS_MODE})"
  sbatch --export=ALL,RUN_NAME="${RUN_NAME}",PHYSICS_WEIGHT="${PHYSICS_WEIGHT}",OFFSET_JETS="${OFFSET_JETS}",RELPOS_MODE="${RELPOS_MODE}" \
    run_unmerge_new_physics_relpos_tier3.sh
done
