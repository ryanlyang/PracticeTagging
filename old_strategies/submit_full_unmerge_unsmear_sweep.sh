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
  "best_all_on_ftoff"
  "no_distributional_ftoff"
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
  "--no_unsmear_finetune"
  "--no_distributional --no_unsmear_finetune"
)

count=0
for i in "${!RUN_NAMES[@]}"; do
  RUN_NAME="${RUN_NAMES[$i]}"
  EX_ARGS="${EXTRA_ARGS[$i]}"
  echo "Submitting: $RUN_NAME"
  if [ -n "$EX_ARGS" ]; then
    sbatch --export="ALL,RUN_NAME=$RUN_NAME,SAVE_DIR=$SAVE_DIR,N_TRAIN_JETS=$N_TRAIN_JETS,MAX_CONSTITS=$MAX_CONSTITS,MAX_MERGE_COUNT=$MAX_MERGE_COUNT,EXTRA_ARGS=$EX_ARGS" "$RUN_SCRIPT"
  else
    sbatch --export="ALL,RUN_NAME=$RUN_NAME,SAVE_DIR=$SAVE_DIR,N_TRAIN_JETS=$N_TRAIN_JETS,MAX_CONSTITS=$MAX_CONSTITS,MAX_MERGE_COUNT=$MAX_MERGE_COUNT,EXTRA_ARGS=" "$RUN_SCRIPT"
  fi
  count=$((count + 1))
done

echo "Total jobs submitted: $count"
