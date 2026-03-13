#!/bin/bash
# Re-run kfold final classifiers (with merge-flag runs) using existing kfold models.
# This submits only the FINAL jobs and depends on a list of prior job IDs.

BASE_DIR=${BASE_DIR:-"checkpoints/unmerge_distr_kfold_sweep"}
K_FOLDS=${K_FOLDS:-5}
KFOLD_ENSEMBLE=${KFOLD_ENSEMBLE:-1}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
MAX_MERGE_COUNT=${MAX_MERGE_COUNT:-10}
TRAIN_PATH=${TRAIN_PATH:-""}

# Job IDs to wait for (space-separated). Override by setting DEP_IDS in the environment.
DEP_IDS_DEFAULT="21034612 21034613 21034614 21034615 21034616 21034618 21034619 21034620 21034621 21034622 21034624 21034625 21034626 21034627 21034628 21034629 21034630 21034631 21034632 21034633 21034634 21034636 21034637 21034638 21034639 21034640 21034642 21034643 21034644 21034645 21034646"
DEP_IDS=${DEP_IDS:-$DEP_IDS_DEFAULT}

# Build a dependency list of job IDs that are still in the queue (pending/running)
valid_ids=()
for id in $DEP_IDS; do
  if squeue -j "$id" -h 2>/dev/null | grep -q .; then
    valid_ids+=("$id")
    continue
  fi
  echo "Info: job id $id not in squeue; skipping from dependency list"
done

DEP_OPT=""
if [ ${#valid_ids[@]} -gt 0 ]; then
  DEP_STR=$(IFS=:; echo "${valid_ids[*]}")
  DEP_OPT="--dependency=afterok:$DEP_STR"
fi

submit_final_cfg() {
  local name="$1"; shift
  local extra_exports="$*"
  echo "Submitting final classifier job for: $name"
  local export_list="SAVE_DIR=$BASE_DIR,RUN_NAME=$name,K_FOLDS=$K_FOLDS,KFOLD_ENSEMBLE=$KFOLD_ENSEMBLE,N_TRAIN_JETS=$N_TRAIN_JETS,MAX_CONSTITS=$MAX_CONSTITS,MAX_MERGE_COUNT=$MAX_MERGE_COUNT,TRAIN_PATH=$TRAIN_PATH"
  if [ -n "$extra_exports" ]; then
    export_list="$export_list,$extra_exports"
  fi
  sbatch $DEP_OPT \
    --export="$export_list" \
    run_unmerge_distr_kfold_final.sh
}

# Match the configs used in submit_unmerge_distr_kfold_matrix.sh
submit_final_cfg "kfold_base_det" \
  UNMERGE_LOSS="chamfer" \
  PHYSICS_WEIGHT=0.0 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=1 \
  NO_CURRICULUM=1 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0

submit_final_cfg "kfold_base_hungarian" \
  UNMERGE_LOSS="hungarian" \
  PHYSICS_WEIGHT=0.0 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=1 \
  NO_CURRICULUM=1 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0

submit_final_cfg "kfold_dist" \
  UNMERGE_LOSS="hungarian" \
  PHYSICS_WEIGHT=0.0 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=0 \
  NO_CURRICULUM=1 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0 \
  MC_SWEEP=1

submit_final_cfg "kfold_base_physics" \
  UNMERGE_LOSS="chamfer" \
  PHYSICS_WEIGHT=0.2 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=1 \
  NO_CURRICULUM=1 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0

submit_final_cfg "kfold_base_curriculum" \
  UNMERGE_LOSS="chamfer" \
  PHYSICS_WEIGHT=0.0 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=1 \
  NO_CURRICULUM=0 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0

submit_final_cfg "kfold_all_on" \
  UNMERGE_LOSS="hungarian" \
  PHYSICS_WEIGHT=0.2 \
  NLL_WEIGHT=1.0 \
  NO_DISTRIBUTIONAL=0 \
  NO_CURRICULUM=0 \
  CURR_START=2 \
  CURR_EPOCHS=20 \
  USE_TRUE_COUNT=0
