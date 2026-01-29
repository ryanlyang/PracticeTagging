#!/bin/bash

# Submit a single k-fold pipeline run for the unsmear variant.
# This is the baseline-like OOF setup (closest to unmerge_model.py) with
# deterministic unmerger + chamfer loss, predicted counts only, no physics,
# no curriculum, no distributional. Ensemble is used for val/test.

BASE_DIR=${BASE_DIR:-"checkpoints/unmerge_distr_kfold_unsmear_sweep"}
K_FOLDS=${K_FOLDS:-5}
KFOLD_ENSEMBLE=${KFOLD_ENSEMBLE:-1}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
MAX_MERGE_COUNT=${MAX_MERGE_COUNT:-10}
TRAIN_PATH=${TRAIN_PATH:-""}

mkdir -p "$BASE_DIR"

RUN_NAME=${RUN_NAME:-"kfold_base_det_unsmear"}

echo "Submitting config: $RUN_NAME (base unmerger + unsmear)"
base_out=$(env \
  SAVE_DIR="$BASE_DIR" RUN_NAME="$RUN_NAME" \
  K_FOLDS="$K_FOLDS" KFOLD_ENSEMBLE="$KFOLD_ENSEMBLE" \
  N_TRAIN_JETS="$N_TRAIN_JETS" MAX_CONSTITS="$MAX_CONSTITS" MAX_MERGE_COUNT="$MAX_MERGE_COUNT" \
  TRAIN_PATH="$TRAIN_PATH" \
  UNMERGE_LOSS="chamfer" \
  PHYSICS_WEIGHT=0.0 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=1 \
  NO_CURRICULUM=1 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0 \
  MC_SWEEP=0 \
  bash submit_unmerge_distr_kfold_pipeline_unsmear.sh)
echo "$base_out"
base_final_jid=$(echo "$base_out" | awk '/Final job submitted as Job/{print $NF}' | tail -n 1)
if [ -z "$base_final_jid" ]; then
  echo "ERROR: Could not determine base final job id."
  exit 1
fi

# Second run: unsmearer trained with K-fold on cached unmerged dataset
UNMERGED_CACHE="$BASE_DIR/$RUN_NAME/unmerged_cache.npz"
DOUBLE_RUN_NAME="${RUN_NAME}_unsmear_kfold"
echo "Submitting double-kfold unsmear run (depends on base final job $base_final_jid)"
env \
  SAVE_DIR="$BASE_DIR" RUN_NAME="$DOUBLE_RUN_NAME" \
  K_FOLDS="$K_FOLDS" KFOLD_ENSEMBLE="$KFOLD_ENSEMBLE" \
  N_TRAIN_JETS="$N_TRAIN_JETS" MAX_CONSTITS="$MAX_CONSTITS" MAX_MERGE_COUNT="$MAX_MERGE_COUNT" \
  TRAIN_PATH="$TRAIN_PATH" \
  UNMERGED_CACHE="$UNMERGED_CACHE" \
  DEPENDS_ON="$base_final_jid" \
  bash submit_unmerge_doublekfold_pipeline_unsmear.sh

# Third run: double-kfold unsmear + merge-flag conditioning/weighting
MERGEFLAG_RUN_NAME="${RUN_NAME}_unsmear_kfold_mergeflag"
echo "Submitting merge-flag unsmear run (depends on base final job $base_final_jid)"
env \
  SAVE_DIR="$BASE_DIR" RUN_NAME="$MERGEFLAG_RUN_NAME" \
  K_FOLDS="$K_FOLDS" KFOLD_ENSEMBLE="$KFOLD_ENSEMBLE" \
  N_TRAIN_JETS="$N_TRAIN_JETS" MAX_CONSTITS="$MAX_CONSTITS" MAX_MERGE_COUNT="$MAX_MERGE_COUNT" \
  TRAIN_PATH="$TRAIN_PATH" \
  UNMERGED_CACHE="$UNMERGED_CACHE" \
  DEPENDS_ON="$base_final_jid" \
  UNSMEAR_MERGE_FLAG=1 \
  UNSMEAR_MERGE_WEIGHT=2.0 \
  bash submit_unmerge_doublekfold_pipeline_unsmear.sh

# Fourth run: double-kfold unsmear + two-head routing (no merge-flag conditioning)
TWOHEAD_RUN_NAME="${RUN_NAME}_unsmear_kfold_twohead"
echo "Submitting two-head unsmear run (depends on base final job $base_final_jid)"
env \
  SAVE_DIR="$BASE_DIR" RUN_NAME="$TWOHEAD_RUN_NAME" \
  K_FOLDS="$K_FOLDS" KFOLD_ENSEMBLE="$KFOLD_ENSEMBLE" \
  N_TRAIN_JETS="$N_TRAIN_JETS" MAX_CONSTITS="$MAX_CONSTITS" MAX_MERGE_COUNT="$MAX_MERGE_COUNT" \
  TRAIN_PATH="$TRAIN_PATH" \
  UNMERGED_CACHE="$UNMERGED_CACHE" \
  DEPENDS_ON="$base_final_jid" \
  UNSMEAR_TWO_HEAD=1 \
  bash submit_unmerge_doublekfold_pipeline_unsmear.sh

# Fifth run: pre-unsmear singletons BEFORE unmerger (reruns unmerger + double kfold unsmear)
PREUNSMEAR_RUN_NAME="${RUN_NAME}_preunsmear_singletons"
echo "Submitting pre-unsmear singletons run (reruns unmerger + double kfold unsmear)"
pre_out=$(env \
  SAVE_DIR="$BASE_DIR" RUN_NAME="$PREUNSMEAR_RUN_NAME" \
  K_FOLDS="$K_FOLDS" KFOLD_ENSEMBLE="$KFOLD_ENSEMBLE" \
  N_TRAIN_JETS="$N_TRAIN_JETS" MAX_CONSTITS="$MAX_CONSTITS" MAX_MERGE_COUNT="$MAX_MERGE_COUNT" \
  TRAIN_PATH="$TRAIN_PATH" \
  UNMERGE_LOSS="chamfer" \
  PHYSICS_WEIGHT=0.0 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=1 \
  NO_CURRICULUM=1 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0 \
  PRE_UNSMEAR_WEIGHT=2.0 \
  bash submit_unmerge_preunsmear_doublekfold_pipeline_unsmear.sh)
echo "$pre_out"
pre_final_jid=$(echo "$pre_out" | awk '/Final job submitted as Job/{print $NF}' | tail -n 1)
if [ -z "$pre_final_jid" ]; then
  echo "WARNING: Could not determine pre-unsmear final job id; downstream runs will start immediately."
fi

# Additional runs using the pre-unsmear unmerger outputs
PRE_UNSMEAR_CACHE="$BASE_DIR/$PREUNSMEAR_RUN_NAME/unmerged_cache.npz"

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
