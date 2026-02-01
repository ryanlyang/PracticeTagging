#!/bin/bash

# Re-run ONLY the kfold final jobs using random-per-jet unmerge selection
# and updated classifier runs (no retraining of folds).

set -euo pipefail

SAVE_DIR=${SAVE_DIR:-"checkpoints/unmerge_distr_kfold_sweep"}
K_FOLDS=${K_FOLDS:-5}
KFOLD_ENSEMBLE=${KFOLD_ENSEMBLE:-1}
KFOLD_RANDOM_VALTEST=${KFOLD_RANDOM_VALTEST:-1}
KFOLD_RANDOM_MODE=${KFOLD_RANDOM_MODE:-"jet"}
KFOLD_RANDOM_SEED=${KFOLD_RANDOM_SEED:-42}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
MAX_MERGE_COUNT=${MAX_MERGE_COUNT:-10}
TRAIN_PATH=${TRAIN_PATH:-""}

submit_final() {
  local name="$1"; shift
  echo "Submitting random-per-jet final for: $name"
  env \
    SAVE_DIR="$SAVE_DIR" RUN_NAME="$name" \
    K_FOLDS="$K_FOLDS" KFOLD_ENSEMBLE="$KFOLD_ENSEMBLE" \
    KFOLD_RANDOM_VALTEST="$KFOLD_RANDOM_VALTEST" \
    KFOLD_RANDOM_MODE="$KFOLD_RANDOM_MODE" \
    KFOLD_RANDOM_SEED="$KFOLD_RANDOM_SEED" \
    N_TRAIN_JETS="$N_TRAIN_JETS" MAX_CONSTITS="$MAX_CONSTITS" MAX_MERGE_COUNT="$MAX_MERGE_COUNT" \
    TRAIN_PATH="$TRAIN_PATH" \
    "$@" \
    bash submit_unmerge_distr_kfold_final_random.sh
  sleep 0.2
}

# Match the configs used in submit_unmerge_distr_kfold_matrix.sh

# 1) kfold_base_det
submit_final "kfold_base_det" \
  UNMERGE_LOSS="chamfer" \
  PHYSICS_WEIGHT=0.0 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=1 \
  NO_CURRICULUM=1 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0

# 2) kfold_base_hungarian
submit_final "kfold_base_hungarian" \
  UNMERGE_LOSS="hungarian" \
  PHYSICS_WEIGHT=0.0 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=1 \
  NO_CURRICULUM=1 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0

# 3) kfold_dist
submit_final "kfold_dist" \
  UNMERGE_LOSS="hungarian" \
  PHYSICS_WEIGHT=0.0 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=0 \
  NO_CURRICULUM=1 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0

# 4) kfold_base_physics
submit_final "kfold_base_physics" \
  UNMERGE_LOSS="chamfer" \
  PHYSICS_WEIGHT=0.2 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=1 \
  NO_CURRICULUM=1 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0

# 5) kfold_base_curriculum
submit_final "kfold_base_curriculum" \
  UNMERGE_LOSS="chamfer" \
  PHYSICS_WEIGHT=0.0 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=1 \
  NO_CURRICULUM=0 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0

# 6) kfold_all_on
submit_final "kfold_all_on" \
  UNMERGE_LOSS="hungarian" \
  PHYSICS_WEIGHT=0.2 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=0 \
  NO_CURRICULUM=0 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0

echo "Queued random-per-jet final jobs for all kfold configs."
