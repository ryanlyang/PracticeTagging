#!/bin/bash

# Single-run pre-unsmear pipeline (no k-fold) with multiple unsmear variants.
# Uses a single pre-unsmearer + unmerger to build cache, then runs multiple
# unsmearer configurations off that cache.

BASE_DIR=${BASE_DIR:-"checkpoints/unmerge_preunsmear_sweep"}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
MAX_MERGE_COUNT=${MAX_MERGE_COUNT:-10}
TRAIN_PATH=${TRAIN_PATH:-""}

mkdir -p "$BASE_DIR"
mkdir -p unmerge_preunsmear_logs

RUN_NAME=${RUN_NAME:-"preunsmear_base_relpos"}

echo "Submitting base pre-unsmear + unmerge job: $RUN_NAME"
base_jid=$(sbatch --time=5-00:00:00 --export=ALL,\
SAVE_DIR="$BASE_DIR",\
RUN_NAME="$RUN_NAME",\
N_TRAIN_JETS="$N_TRAIN_JETS",\
MAX_CONSTITS="$MAX_CONSTITS",\
MAX_MERGE_COUNT="$MAX_MERGE_COUNT",\
TRAIN_PATH="$TRAIN_PATH",\
K_FOLDS=1,\
KFOLD_ENSEMBLE=0,\
KFOLD_USE_PRETRAINED=0,\
KFOLD_FIXED_FOLD=-1,\
UNMERGE_LOSS="chamfer",\
PHYSICS_WEIGHT=0.0,\
NLL_WEIGHT=1.0,\
NO_DISTRIBUTIONAL=1,\
NO_CURRICULUM=1,\
CURR_START=2,\
CURR_EPOCHS=20,\
USE_TRUE_COUNT=0,\
PRE_UNSMEAR=1,\
PRE_UNSMEAR_WEIGHT=2.0,\
PRE_UNSMEAR_K_FOLDS=1,\
PRE_UNSMEAR_KFOLD_USE_PRETRAINED=0,\
PRE_UNSMEAR_KFOLD_ENSEMBLE=0,\
UNSMEAR_K_FOLDS=1,\
UNSMEAR_KFOLD_USE_PRETRAINED=0,\
UNSMEAR_KFOLD_ENSEMBLE=0,\
STOP_AFTER_UNMERGE=1,\
UNMERGE_RELPOS_MODE="attn" \
run_unmerge_preunsmear_final.sh | awk '{print $4}')

echo "Base job submitted as Job $base_jid"
if [ -z "$base_jid" ]; then
  echo "ERROR: Could not determine base job id."
  exit 1
fi

UNMERGED_CACHE="$BASE_DIR/$RUN_NAME/unmerged_cache.npz"

submit_variant () {
  local name="$1"
  local extra_export="${2:-}"
  echo "Submitting unsmear variant: $name (depends on $base_jid)"
  if [ -n "$extra_export" ]; then
    extra_export=",$extra_export"
  fi
  sbatch --time=4-00:00:00 --dependency=afterok:$base_jid --export=ALL,\
SAVE_DIR="$BASE_DIR",\
RUN_NAME="$name",\
N_TRAIN_JETS="$N_TRAIN_JETS",\
MAX_CONSTITS="$MAX_CONSTITS",\
MAX_MERGE_COUNT="$MAX_MERGE_COUNT",\
TRAIN_PATH="$TRAIN_PATH",\
K_FOLDS=1,\
KFOLD_ENSEMBLE=0,\
KFOLD_USE_PRETRAINED=0,\
KFOLD_FIXED_FOLD=-1,\
UNMERGE_LOSS="chamfer",\
PHYSICS_WEIGHT=0.0,\
NLL_WEIGHT=1.0,\
NO_DISTRIBUTIONAL=1,\
NO_CURRICULUM=1,\
CURR_START=2,\
CURR_EPOCHS=20,\
USE_TRUE_COUNT=0,\
UNMERGED_CACHE="$UNMERGED_CACHE",\
UNSMEAR_K_FOLDS=1,\
UNSMEAR_KFOLD_USE_PRETRAINED=0,\
UNSMEAR_KFOLD_ENSEMBLE=0,\
PRE_UNSMEAR=0,\
STOP_AFTER_UNMERGE=0,\
UNMERGE_RELPOS_MODE="attn"$extra_export \
run_unmerge_preunsmear_final.sh
}

# 1) Base unsmearer (no extra knobs)
submit_variant "${RUN_NAME}_unsmear_base"

# 2) Merge-flag conditioning/weighting
submit_variant "${RUN_NAME}_unsmear_mergeflag" \
  "UNSMEAR_MERGE_FLAG=1,UNSMEAR_MERGE_WEIGHT=2.0"

# 3) Two-head routing for merged vs untouched constituents
submit_variant "${RUN_NAME}_unsmear_twohead" \
  "UNSMEAR_TWO_HEAD=1"

# 4) Local neighborhood attention unsmearing
submit_variant "${RUN_NAME}_unsmear_localattn" \
  "UNSMEAR_LOCAL_ATTN=1,UNSMEAR_LOCAL_ATTN_RADIUS=0.2,UNSMEAR_LOCAL_ATTN_SCALE=2.0"

# 5) Separate pT/E vs eta/phi heads
submit_variant "${RUN_NAME}_unsmear_ptetaph" \
  "UNSMEAR_FEAT_HEAD_MODE=ptetaph"
