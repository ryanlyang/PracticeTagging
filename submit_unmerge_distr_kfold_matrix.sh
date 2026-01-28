#!/bin/bash

# Submit 8 unmerger-configuration sweeps (each uses k-fold pipeline).
# Order is intentional: start with baseline-like OOF (closest to unmerge_model.py),
# then loss variants, then distributional/MC, then physics/curriculum, then all-on.

BASE_DIR=${BASE_DIR:-"checkpoints/unmerge_distr_kfold_sweep"}
K_FOLDS=${K_FOLDS:-5}
KFOLD_ENSEMBLE=${KFOLD_ENSEMBLE:-1}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
MAX_MERGE_COUNT=${MAX_MERGE_COUNT:-10}
TRAIN_PATH=${TRAIN_PATH:-""}

mkdir -p "$BASE_DIR"

submit_cfg() {
  local name="$1"
  shift
  echo "Submitting config: $name"
  env \
    SAVE_DIR="$BASE_DIR" RUN_NAME="$name" \
    K_FOLDS="$K_FOLDS" KFOLD_ENSEMBLE="$KFOLD_ENSEMBLE" \
    N_TRAIN_JETS="$N_TRAIN_JETS" MAX_CONSTITS="$MAX_CONSTITS" MAX_MERGE_COUNT="$MAX_MERGE_COUNT" \
    TRAIN_PATH="$TRAIN_PATH" \
    "$@" \
    bash submit_unmerge_distr_kfold_pipeline.sh
  sleep 0.3
}

# 1) Baseline-like OOF (closest to unmerge_model.py): deterministic + chamfer,
#    predicted counts only, no physics, no curriculum, no distributional.
submit_cfg "kfold_base_det" \
  UNMERGE_LOSS="chamfer" \
  PHYSICS_WEIGHT=0.0 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=1 \
  NO_CURRICULUM=1 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0

# 2) Same as baseline but Hungarian loss
submit_cfg "kfold_base_hungarian" \
  UNMERGE_LOSS="hungarian" \
  PHYSICS_WEIGHT=0.0 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=1 \
  NO_CURRICULUM=1 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0

# 3) Distributional unmerger (no physics, no curriculum)
submit_cfg "kfold_dist" \
  UNMERGE_LOSS="hungarian" \
  PHYSICS_WEIGHT=0.0 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=0 \
  NO_CURRICULUM=1 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0 \
  MC_SWEEP=1

# 4) Baseline-like with physics loss
submit_cfg "kfold_base_physics" \
  UNMERGE_LOSS="chamfer" \
  PHYSICS_WEIGHT=0.2 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=1 \
  NO_CURRICULUM=1 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0

# 5) Baseline-like with curriculum learning
submit_cfg "kfold_base_curriculum" \
  UNMERGE_LOSS="chamfer" \
  PHYSICS_WEIGHT=0.0 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=1 \
  NO_CURRICULUM=0 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0

# 6) All-on (distributional + Hungarian + physics + curriculum), no true-count
submit_cfg "kfold_all_on" \
  UNMERGE_LOSS="hungarian" \
  PHYSICS_WEIGHT=0.2 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=0 \
  NO_CURRICULUM=0 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0
