#!/bin/bash

# Submit a 12-run sweep for three_steps_un.py

RUN_SCRIPT="run_three_steps_un_tier3.sh"
SAVE_DIR=${SAVE_DIR:-"checkpoints/three_steps_un_sweep"}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
MAX_MERGE_COUNT=${MAX_MERGE_COUNT:-10}

mkdir -p three_steps_un_logs

declare -a RUN_NAMES=(
  "best_all_on"
  "no_distributional"
  "no_curriculum"
  "no_true_count"
  "unmerge_chamfer"
  "no_snr_weight"
  "no_self_cond"
  "no_cfg"
  "pred_eps"
  "pred_x0"
  "unmerge_ft_off"
  "no_distributional_ft_on"
)

declare -a EXTRA_ARGS=(
  "--unmerge_finetune"
  "--no_distributional --unmerge_finetune"
  "--no_curriculum --unmerge_finetune"
  "--no_true_count --unmerge_finetune"
  "--unmerge_loss chamfer --unmerge_finetune"
  "--no_snr_weight --unmerge_finetune"
  "--self_cond_prob 0.0 --unmerge_finetune"
  "--guidance_scale 1.0 --cond_drop_prob 0.0 --unmerge_finetune"
  "--pred_type eps --unmerge_finetune"
  "--pred_type x0 --unmerge_finetune"
  ""  # unmerge finetune off
  "--no_distributional --unmerge_finetune"
)

count=0
for i in "${!RUN_NAMES[@]}"; do
  RUN_NAME="${RUN_NAMES[$i]}"
  EX_ARGS="${EXTRA_ARGS[$i]}"
  echo "Submitting: $RUN_NAME"
  sbatch --export="ALL,RUN_NAME=$RUN_NAME,SAVE_DIR=$SAVE_DIR,N_TRAIN_JETS=$N_TRAIN_JETS,MAX_CONSTITS=$MAX_CONSTITS,MAX_MERGE_COUNT=$MAX_MERGE_COUNT,EXTRA_ARGS=$EX_ARGS" "$RUN_SCRIPT"
  count=$((count + 1))
done

echo "Total jobs submitted: $count"
