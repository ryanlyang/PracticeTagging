#!/bin/bash
set -euo pipefail

# Queue ONLY the pre-unsmear pipelines (no base/mergeflag/twohead from the main matrix).
# This mirrors the "pre-unsmear singletons" block in submit_unmerge_distr_kfold_matrix_unsmear.sh

BASE_DIR=${BASE_DIR:-"checkpoints/unmerge_distr_kfold_unsmear_sweep"}
K_FOLDS=${K_FOLDS:-5}
KFOLD_ENSEMBLE=${KFOLD_ENSEMBLE:-1}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
MAX_MERGE_COUNT=${MAX_MERGE_COUNT:-10}
TRAIN_PATH=${TRAIN_PATH:-""}

mkdir -p "$BASE_DIR"

# Pre-unsmear pipeline run name
RUN_NAME=${RUN_NAME:-"kfold_base_det_unsmear_preunsmear_singletons"}

# Unmerger knobs (baseline-like)
UNMERGE_LOSS=${UNMERGE_LOSS:-"chamfer"}
PHYSICS_WEIGHT=${PHYSICS_WEIGHT:-0.0}
NLL_WEIGHT=${NLL_WEIGHT:-1.0}
NO_DISTRIBUTIONAL=${NO_DISTRIBUTIONAL:-1}
NO_CURRICULUM=${NO_CURRICULUM:-1}
CURR_START=${CURR_START:-2}
CURR_EPOCHS=${CURR_EPOCHS:-20}
USE_TRUE_COUNT=${USE_TRUE_COUNT:-0}
PRE_UNSMEAR_WEIGHT=${PRE_UNSMEAR_WEIGHT:-2.0}

# 1) Full pre-unsmear singletons pipeline (pre-unsmear kfold -> unmerger kfold -> cache -> unsmear kfold -> final)
PREUNSMEAR_RUN_NAME="$RUN_NAME"
echo "Submitting pre-unsmear singletons run (reruns unmerger + double kfold unsmear)"
pre_out=$(env \
  SAVE_DIR="$BASE_DIR" RUN_NAME="$PREUNSMEAR_RUN_NAME" \
  K_FOLDS="$K_FOLDS" KFOLD_ENSEMBLE="$KFOLD_ENSEMBLE" \
  N_TRAIN_JETS="$N_TRAIN_JETS" MAX_CONSTITS="$MAX_CONSTITS" MAX_MERGE_COUNT="$MAX_MERGE_COUNT" \
  TRAIN_PATH="$TRAIN_PATH" \
  UNMERGE_LOSS="$UNMERGE_LOSS" \
  PHYSICS_WEIGHT="$PHYSICS_WEIGHT" \
  NLL_WEIGHT="$NLL_WEIGHT" \
  NO_DISTRIBUTIONAL="$NO_DISTRIBUTIONAL" \
  NO_CURRICULUM="$NO_CURRICULUM" \
  CURR_START="$CURR_START" \
  CURR_EPOCHS="$CURR_EPOCHS" \
  USE_TRUE_COUNT="$USE_TRUE_COUNT" \
  PRE_UNSMEAR_WEIGHT="$PRE_UNSMEAR_WEIGHT" \
  bash submit_unmerge_preunsmear_doublekfold_pipeline_unsmear.sh)

echo "$pre_out"
pre_final_jid=$(echo "$pre_out" | awk '/Final job submitted as Job/{print $NF}' | tail -n 1)
if [ -z "$pre_final_jid" ]; then
  echo "WARNING: Could not determine pre-unsmear final job id; downstream runs will start immediately."
fi

PRE_UNSMEAR_CACHE="$BASE_DIR/$PREUNSMEAR_RUN_NAME/unmerged_cache.npz"

# 2) Follow-up: base unsmear kfold on pre-unsmear unmerger outputs
RUN_PRE_BASE="${PREUNSMEAR_RUN_NAME}_unsmear_kfold"
echo "Submitting pre-unsmear unmerger -> kfold unsmear run (depends on $pre_final_jid)"
env \
  SAVE_DIR="$BASE_DIR" RUN_NAME="$RUN_PRE_BASE" \
  K_FOLDS="$K_FOLDS" KFOLD_ENSEMBLE="$KFOLD_ENSEMBLE" \
  N_TRAIN_JETS="$N_TRAIN_JETS" MAX_CONSTITS="$MAX_CONSTITS" MAX_MERGE_COUNT="$MAX_MERGE_COUNT" \
  TRAIN_PATH="$TRAIN_PATH" \
  UNMERGED_CACHE="$PRE_UNSMEAR_CACHE" \
  DEPENDS_ON="$pre_final_jid" \
  bash submit_unmerge_doublekfold_pipeline_unsmear.sh

# 3) Follow-up: merge-flag unsmear kfold on pre-unsmear unmerger outputs
RUN_PRE_MERGEFLAG="${PREUNSMEAR_RUN_NAME}_unsmear_kfold_mergeflag"
echo "Submitting pre-unsmear unmerger -> merge-flag unsmear run (depends on $pre_final_jid)"
env \
  SAVE_DIR="$BASE_DIR" RUN_NAME="$RUN_PRE_MERGEFLAG" \
  K_FOLDS="$K_FOLDS" KFOLD_ENSEMBLE="$KFOLD_ENSEMBLE" \
  N_TRAIN_JETS="$N_TRAIN_JETS" MAX_CONSTITS="$MAX_CONSTITS" MAX_MERGE_COUNT="$MAX_MERGE_COUNT" \
  TRAIN_PATH="$TRAIN_PATH" \
  UNMERGED_CACHE="$PRE_UNSMEAR_CACHE" \
  DEPENDS_ON="$pre_final_jid" \
  UNSMEAR_MERGE_FLAG=1 \
  UNSMEAR_MERGE_WEIGHT=2.0 \
  bash submit_unmerge_doublekfold_pipeline_unsmear.sh

# 4) Follow-up: two-head unsmear kfold on pre-unsmear unmerger outputs
RUN_PRE_TWOHEAD="${PREUNSMEAR_RUN_NAME}_unsmear_kfold_twohead"
echo "Submitting pre-unsmear unmerger -> two-head unsmear run (depends on $pre_final_jid)"
env \
  SAVE_DIR="$BASE_DIR" RUN_NAME="$RUN_PRE_TWOHEAD" \
  K_FOLDS="$K_FOLDS" KFOLD_ENSEMBLE="$KFOLD_ENSEMBLE" \
  N_TRAIN_JETS="$N_TRAIN_JETS" MAX_CONSTITS="$MAX_CONSTITS" MAX_MERGE_COUNT="$MAX_MERGE_COUNT" \
  TRAIN_PATH="$TRAIN_PATH" \
  UNMERGED_CACHE="$PRE_UNSMEAR_CACHE" \
  DEPENDS_ON="$pre_final_jid" \
  UNSMEAR_TWO_HEAD=1 \
  bash submit_unmerge_doublekfold_pipeline_unsmear.sh

