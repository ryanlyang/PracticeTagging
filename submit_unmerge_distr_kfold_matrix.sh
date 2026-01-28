#!/bin/bash

# Submit 8 unmerger-configuration sweeps (each uses k-fold pipeline with restart chains).

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
    bash submit_unmerge_distr_kfold_pipeline_long.sh
  sleep 0.3
}

# 1) Best guess (all on)
submit_cfg "best_all_on" \
  UNMERGE_LOSS="hungarian" \
  PHYSICS_WEIGHT=0.2 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=0 \
  NO_CURRICULUM=0 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0

# 2) Chamfer loss
submit_cfg "chamfer" \
  UNMERGE_LOSS="chamfer" \
  PHYSICS_WEIGHT=0.2 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=0 \
  NO_CURRICULUM=0 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0

# 3) No distributional output
submit_cfg "no_distributional" \
  UNMERGE_LOSS="hungarian" \
  PHYSICS_WEIGHT=0.2 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=1 \
  NO_CURRICULUM=0 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0

# 4) No physics loss
submit_cfg "no_physics" \
  UNMERGE_LOSS="hungarian" \
  PHYSICS_WEIGHT=0.0 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=0 \
  NO_CURRICULUM=0 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0

# 5) NLL weight high
submit_cfg "nll_hi" \
  UNMERGE_LOSS="hungarian" \
  PHYSICS_WEIGHT=0.2 \
  NLL_WEIGHT=2.0 \
  NO_DISTRIBUTIONAL=0 \
  NO_CURRICULUM=0 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0

# 6) NLL weight low
submit_cfg "nll_lo" \
  UNMERGE_LOSS="hungarian" \
  PHYSICS_WEIGHT=0.2 \
  NLL_WEIGHT=0.3 \
  NO_DISTRIBUTIONAL=0 \
  NO_CURRICULUM=0 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0

# 7) No curriculum
submit_cfg "no_curriculum" \
  UNMERGE_LOSS="hungarian" \
  PHYSICS_WEIGHT=0.2 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=0 \
  NO_CURRICULUM=1 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0

# 8) Train with true counts (upper bound)
submit_cfg "true_count" \
  UNMERGE_LOSS="hungarian" \
  PHYSICS_WEIGHT=0.2 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=0 \
  NO_CURRICULUM=0 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=1
