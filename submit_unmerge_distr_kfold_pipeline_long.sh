#!/bin/bash

# K-fold pipeline with chained long fold jobs (resume via checkpoints).
# Uses best_all_on settings (hungarian + distributional + physics + curriculum).

BASE_DIR=${BASE_DIR:-"checkpoints/unmerge_distr_kfold_sweep"}
RUN_NAME=${RUN_NAME:-"best_all_on_long"}
K_FOLDS=${K_FOLDS:-5}
KFOLD_ENSEMBLE=${KFOLD_ENSEMBLE:-1}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
MAX_MERGE_COUNT=${MAX_MERGE_COUNT:-10}
TRAIN_PATH=${TRAIN_PATH:-""}
CHAIN_LEN=${CHAIN_LEN:-2}

# Best-all-on settings (match first matrix run)
UNMERGE_LOSS=${UNMERGE_LOSS:-"hungarian"}
PHYSICS_WEIGHT=${PHYSICS_WEIGHT:-0.2}
NLL_WEIGHT=${NLL_WEIGHT:-1.0}
NO_DISTRIBUTIONAL=${NO_DISTRIBUTIONAL:-0}
NO_CURRICULUM=${NO_CURRICULUM:-0}
CURR_START=${CURR_START:-2}
CURR_EPOCHS=${CURR_EPOCHS:-20}
USE_TRUE_COUNT=${USE_TRUE_COUNT:-0}
NUM_WORKERS=${NUM_WORKERS:-6}
MAX_HOURS=${MAX_HOURS:-104}
TIME_MARGIN_MIN=${TIME_MARGIN_MIN:-30}

mkdir -p "$BASE_DIR"

submit_chain_for_fold() {
  local fold_id="$1"
  local prev=""
  local jid=""
  for i in $(seq 1 "$CHAIN_LEN"); do
    if [ -z "$prev" ]; then
      jid=$(env \
        SAVE_DIR="$BASE_DIR" RUN_NAME="$RUN_NAME" \
        K_FOLDS="$K_FOLDS" \
        N_TRAIN_JETS="$N_TRAIN_JETS" MAX_CONSTITS="$MAX_CONSTITS" MAX_MERGE_COUNT="$MAX_MERGE_COUNT" \
        TRAIN_PATH="$TRAIN_PATH" \
        UNMERGE_LOSS="$UNMERGE_LOSS" PHYSICS_WEIGHT="$PHYSICS_WEIGHT" NLL_WEIGHT="$NLL_WEIGHT" \
        NO_DISTRIBUTIONAL="$NO_DISTRIBUTIONAL" NO_CURRICULUM="$NO_CURRICULUM" CURR_START="$CURR_START" CURR_EPOCHS="$CURR_EPOCHS" \
        USE_TRUE_COUNT="$USE_TRUE_COUNT" NUM_WORKERS="$NUM_WORKERS" \
        MAX_HOURS="$MAX_HOURS" TIME_MARGIN_MIN="$TIME_MARGIN_MIN" \
        FOLD_ID="$fold_id" \
        sbatch --parsable run_unmerge_distr_fold_long.sh)
    else
      jid=$(env \
        SAVE_DIR="$BASE_DIR" RUN_NAME="$RUN_NAME" \
        K_FOLDS="$K_FOLDS" \
        N_TRAIN_JETS="$N_TRAIN_JETS" MAX_CONSTITS="$MAX_CONSTITS" MAX_MERGE_COUNT="$MAX_MERGE_COUNT" \
        TRAIN_PATH="$TRAIN_PATH" \
        UNMERGE_LOSS="$UNMERGE_LOSS" PHYSICS_WEIGHT="$PHYSICS_WEIGHT" NLL_WEIGHT="$NLL_WEIGHT" \
        NO_DISTRIBUTIONAL="$NO_DISTRIBUTIONAL" NO_CURRICULUM="$NO_CURRICULUM" CURR_START="$CURR_START" CURR_EPOCHS="$CURR_EPOCHS" \
        USE_TRUE_COUNT="$USE_TRUE_COUNT" NUM_WORKERS="$NUM_WORKERS" \
        MAX_HOURS="$MAX_HOURS" TIME_MARGIN_MIN="$TIME_MARGIN_MIN" \
        FOLD_ID="$fold_id" \
        sbatch --parsable --dependency=afterok:$prev run_unmerge_distr_fold_long.sh)
    fi
    prev="$jid"
  done
  echo "$prev"
}

echo "Submitting long k-fold chains (len=$CHAIN_LEN) for $RUN_NAME..."
dep_list=""
for fold in $(seq 0 $((K_FOLDS-1))); do
  last_jid=$(submit_chain_for_fold "$fold")
  echo "  Fold $fold chain last job: $last_jid"
  if [ -z "$dep_list" ]; then
    dep_list="$last_jid"
  else
    dep_list="${dep_list}:$last_jid"
  fi
done

echo "Submitting final job afterok:$dep_list"
env \
  SAVE_DIR="$BASE_DIR" RUN_NAME="$RUN_NAME" \
  K_FOLDS="$K_FOLDS" KFOLD_ENSEMBLE="$KFOLD_ENSEMBLE" \
  N_TRAIN_JETS="$N_TRAIN_JETS" MAX_CONSTITS="$MAX_CONSTITS" MAX_MERGE_COUNT="$MAX_MERGE_COUNT" \
  TRAIN_PATH="$TRAIN_PATH" \
  UNMERGE_LOSS="$UNMERGE_LOSS" PHYSICS_WEIGHT="$PHYSICS_WEIGHT" NLL_WEIGHT="$NLL_WEIGHT" \
  NO_DISTRIBUTIONAL="$NO_DISTRIBUTIONAL" NO_CURRICULUM="$NO_CURRICULUM" CURR_START="$CURR_START" CURR_EPOCHS="$CURR_EPOCHS" \
  USE_TRUE_COUNT="$USE_TRUE_COUNT" NUM_WORKERS="$NUM_WORKERS" \
  sbatch --dependency=afterok:$dep_list run_unmerge_distr_kfold_final.sh
