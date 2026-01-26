#!/bin/bash

# Submit a small sweep for full_unmerge_unsmear.py

RUN_SCRIPT="run_full_unmerge_unsmear_tier3.sh"
SAVE_DIR=${SAVE_DIR:-"checkpoints/full_unmerge_unsmear_sweep"}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
MAX_MERGE_COUNT=${MAX_MERGE_COUNT:-10}

mkdir -p full_unmerge_logs

declare -a RUN_NAMES=(
  "best_all_on"
  "no_distributional"
  "no_curriculum"
  "no_true_count"
  "unmerge_chamfer"
  "no_snr_weight"
  "no_self_cond"
  "no_cfg"
)

declare -a EXTRA_ARGS=(
  ""
  "--no_distributional"
  "--no_curriculum"
  "--no_true_count"
  "--unmerge_loss chamfer"
  "--no_snr_weight"
  "--self_cond_prob 0.0"
  "--guidance_scale 1.0 --cond_drop_prob 0.0"
)

count=0
for i in "${!RUN_NAMES[@]}"; do
  RUN_NAME="${RUN_NAMES[$i]}"
  EX_ARGS="${EXTRA_ARGS[$i]}"
  echo "Submitting: $RUN_NAME"
  EXPORTS="ALL,"
  EXPORTS+="RUN_NAME=$RUN_NAME,"
  EXPORTS+="SAVE_DIR=$SAVE_DIR,"
  EXPORTS+="N_TRAIN_JETS=$N_TRAIN_JETS,"
  EXPORTS+="MAX_CONSTITS=$MAX_CONSTITS,"
  EXPORTS+="MAX_MERGE_COUNT=$MAX_MERGE_COUNT,"
  EXPORTS+="EXTRA_ARGS=$(printf '%q' "$EX_ARGS")"
  sbatch --export="$EXPORTS" "$RUN_SCRIPT"
  count=$((count + 1))
done

echo "Total jobs submitted: $count"
