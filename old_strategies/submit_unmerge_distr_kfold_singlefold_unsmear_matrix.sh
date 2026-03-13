#!/bin/bash

# Run single-fold unmerge + single unsmearer variants using a fixed k-fold model.
# Uses fold_0 from the existing kfold_base_det_unsmear run.

set -euo pipefail

BASE_DIR=${BASE_DIR:-"checkpoints/unmerge_distr_singlefold_unsmear_sweep"}
K_FOLDS=${K_FOLDS:-5}
KFOLD_FIXED_FOLD=${KFOLD_FIXED_FOLD:-0}
KFOLD_MODEL_DIR=${KFOLD_MODEL_DIR:-"checkpoints/unmerge_distr_kfold_unsmear_sweep/kfold_base_det_unsmear/kfold_models"}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
MAX_MERGE_COUNT=${MAX_MERGE_COUNT:-10}
TRAIN_PATH=${TRAIN_PATH:-""}
LOG_DIR=${LOG_DIR:-"unmerge_distr_singlefold_unsmear_logs"}

mkdir -p "$BASE_DIR"
mkdir -p "$LOG_DIR"

submit_run() {
  local name="$1"; shift
  local extra="$1"
  echo "Submitting singlefold unsmear run: $name"
  local export_list="SAVE_DIR=$BASE_DIR,RUN_NAME=$name,K_FOLDS=$K_FOLDS,KFOLD_ENSEMBLE=0,KFOLD_RANDOM_VALTEST=0,KFOLD_FIXED_FOLD=$KFOLD_FIXED_FOLD,KFOLD_MODEL_DIR=$KFOLD_MODEL_DIR,N_TRAIN_JETS=$N_TRAIN_JETS,MAX_CONSTITS=$MAX_CONSTITS,MAX_MERGE_COUNT=$MAX_MERGE_COUNT,TRAIN_PATH=$TRAIN_PATH,UNSMEAR_K_FOLDS=1,LOG_DIR=$LOG_DIR,NO_DISTRIBUTIONAL=1,UNMERGE_LOSS=chamfer"
  if [ -n "$extra" ]; then
    export_list="$export_list,$extra"
  fi
  sbatch --output="$LOG_DIR/unmerge_distr_final_%j.out" --error="$LOG_DIR/unmerge_distr_final_%j.err" --export=ALL,$export_list run_unmerge_distr_kfold_final_unsmear.sh
  sleep 0.2
}

# 1) Base: fixed fold unmerge + single unsmearer (no special flags)
submit_run "singlefold_base" ""

# 2) Merge-flag conditioning/weighting for unsmearer
submit_run "singlefold_mergeflag" "UNSMEAR_MERGE_FLAG=1,UNSMEAR_MERGE_WEIGHT=2.0"

# 3) Two-head routing for unmerged vs untouched constituents
submit_run "singlefold_twohead" "UNSMEAR_TWO_HEAD=1"

# 4) Local neighborhood attention unsmearing
submit_run "singlefold_localattn" "UNSMEAR_LOCAL_ATTN=1,UNSMEAR_LOCAL_ATTN_RADIUS=0.2,UNSMEAR_LOCAL_ATTN_SCALE=2.0"

# 5) Separate pT/E vs eta/phi heads for unsmearer
submit_run "singlefold_featheads" "UNSMEAR_FEAT_HEAD_MODE=ptetaph"

echo "Queued 5 singlefold unsmear runs."
