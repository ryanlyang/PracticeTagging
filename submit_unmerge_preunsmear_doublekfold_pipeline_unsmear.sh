#!/bin/bash

# Pipeline: pre-unsmear singletons (k-fold) -> unmerger k-fold -> cache-only -> unsmearer k-fold -> final classifiers

K_FOLDS=${K_FOLDS:-5}
SAVE_DIR=${SAVE_DIR:-"checkpoints/unmerge_distr_kfold_unsmear_sweep"}
RUN_NAME=${RUN_NAME:-"kfold_preunsmear_singletons"}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
MAX_MERGE_COUNT=${MAX_MERGE_COUNT:-10}
NUM_WORKERS=${NUM_WORKERS:-6}
TRAIN_PATH=${TRAIN_PATH:-""}
KFOLD_ENSEMBLE=${KFOLD_ENSEMBLE:-1}

# Unmerger knobs
UNMERGE_LOSS=${UNMERGE_LOSS:-"chamfer"}
NO_DISTRIBUTIONAL=${NO_DISTRIBUTIONAL:-1}
NO_CURRICULUM=${NO_CURRICULUM:-1}
CURR_START=${CURR_START:-2}
CURR_EPOCHS=${CURR_EPOCHS:-20}
PHYSICS_WEIGHT=${PHYSICS_WEIGHT:-0.0}
NLL_WEIGHT=${NLL_WEIGHT:-1.0}
USE_TRUE_COUNT=${USE_TRUE_COUNT:-0}

# Pre-unsmear knobs
PRE_UNSMEAR_WEIGHT=${PRE_UNSMEAR_WEIGHT:-2.0}
PRE_UNSMEAR_KFOLD_MODEL_DIR=${PRE_UNSMEAR_KFOLD_MODEL_DIR:-"$SAVE_DIR/$RUN_NAME/pre_unsmear_kfold_models"}

# Unsmearer k-fold (post-unmerge)
UNMERGED_CACHE=${UNMERGED_CACHE:-"$SAVE_DIR/$RUN_NAME/unmerged_cache.npz"}
UNSMEAR_KFOLD_MODEL_DIR=${UNSMEAR_KFOLD_MODEL_DIR:-"$SAVE_DIR/$RUN_NAME/unsmear_kfold_models"}
UNSMEAR_MERGE_FLAG=${UNSMEAR_MERGE_FLAG:-0}
UNSMEAR_MERGE_WEIGHT=${UNSMEAR_MERGE_WEIGHT:-2.0}
UNSMEAR_TWO_HEAD=${UNSMEAR_TWO_HEAD:-0}

DEPENDS_ON=${DEPENDS_ON:-""}

mkdir -p unmerge_distr_kfold_logs

dep_opt=""
if [ -n "$DEPENDS_ON" ]; then
  dep_opt="--dependency=afterok:$DEPENDS_ON"
fi

echo "Submitting pre-unsmearer $K_FOLDS fold jobs (after dependency: ${DEPENDS_ON:-none})..."
pre_ids=()
for ((i=0; i<K_FOLDS; i++)); do
  jid=$(sbatch $dep_opt --export=ALL,\
FOLD_ID=$i,\
PRE_UNSMEAR_K_FOLDS=$K_FOLDS,\
PRE_UNSMEAR_KFOLD_MODEL_DIR=$PRE_UNSMEAR_KFOLD_MODEL_DIR,\
PRE_UNSMEAR_WEIGHT=$PRE_UNSMEAR_WEIGHT,\
SAVE_DIR=$SAVE_DIR,\
RUN_NAME=$RUN_NAME,\
N_TRAIN_JETS=$N_TRAIN_JETS,\
MAX_CONSTITS=$MAX_CONSTITS,\
MAX_MERGE_COUNT=$MAX_MERGE_COUNT,\
NUM_WORKERS=$NUM_WORKERS,\
TRAIN_PATH=$TRAIN_PATH \
  run_pre_unsmear_kfold_fold_unsmear.sh | awk '{print $4}')
  echo "  Pre-unsmear fold $i submitted as Job $jid"
  pre_ids+=("$jid")
  sleep 0.2
done

dep_pre=$(IFS=:; echo "${pre_ids[*]}")

echo "Submitting unmerger $K_FOLDS fold jobs (after pre-unsmear folds)..."
unmerge_ids=()
for ((i=0; i<K_FOLDS; i++)); do
  jid=$(sbatch --dependency=afterok:$dep_pre --export=ALL,\
FOLD_ID=$i,\
SAVE_DIR=$SAVE_DIR,\
RUN_NAME=$RUN_NAME,\
K_FOLDS=$K_FOLDS,\
KFOLD_MODEL_DIR=$SAVE_DIR/$RUN_NAME/kfold_models,\
N_TRAIN_JETS=$N_TRAIN_JETS,\
MAX_CONSTITS=$MAX_CONSTITS,\
MAX_MERGE_COUNT=$MAX_MERGE_COUNT,\
NUM_WORKERS=$NUM_WORKERS,\
TRAIN_PATH=$TRAIN_PATH,\
UNMERGE_LOSS=$UNMERGE_LOSS,\
NO_DISTRIBUTIONAL=$NO_DISTRIBUTIONAL,\
NO_CURRICULUM=$NO_CURRICULUM,\
CURR_START=$CURR_START,\
CURR_EPOCHS=$CURR_EPOCHS,\
PHYSICS_WEIGHT=$PHYSICS_WEIGHT,\
NLL_WEIGHT=$NLL_WEIGHT,\
USE_TRUE_COUNT=$USE_TRUE_COUNT,\
PRE_UNSMEAR=1,\
PRE_UNSMEAR_K_FOLDS=$K_FOLDS,\
PRE_UNSMEAR_KFOLD_MODEL_DIR=$PRE_UNSMEAR_KFOLD_MODEL_DIR,\
PRE_UNSMEAR_KFOLD_USE_PRETRAINED=1,\
PRE_UNSMEAR_KFOLD_ENSEMBLE=1,\
PRE_UNSMEAR_WEIGHT=$PRE_UNSMEAR_WEIGHT \
  run_unmerge_distr_fold_unsmear.sh | awk '{print $4}')
  echo "  Unmerge fold $i submitted as Job $jid"
  unmerge_ids+=("$jid")
  sleep 0.2
done

dep_unmerge=$(IFS=:; echo "${unmerge_ids[*]}")

echo "Submitting cache-only job (after unmerger folds)..."
cache_jid=$(sbatch --dependency=afterok:$dep_unmerge --export=ALL,\
SAVE_DIR=$SAVE_DIR,\
RUN_NAME=$RUN_NAME,\
K_FOLDS=$K_FOLDS,\
KFOLD_ENSEMBLE=$KFOLD_ENSEMBLE,\
N_TRAIN_JETS=$N_TRAIN_JETS,\
MAX_CONSTITS=$MAX_CONSTITS,\
MAX_MERGE_COUNT=$MAX_MERGE_COUNT,\
NUM_WORKERS=$NUM_WORKERS,\
TRAIN_PATH=$TRAIN_PATH,\
UNMERGE_LOSS=$UNMERGE_LOSS,\
NO_DISTRIBUTIONAL=$NO_DISTRIBUTIONAL,\
NO_CURRICULUM=$NO_CURRICULUM,\
CURR_START=$CURR_START,\
CURR_EPOCHS=$CURR_EPOCHS,\
PHYSICS_WEIGHT=$PHYSICS_WEIGHT,\
NLL_WEIGHT=$NLL_WEIGHT,\
USE_TRUE_COUNT=$USE_TRUE_COUNT,\
STOP_AFTER_UNMERGE=1,\
PRE_UNSMEAR=1,\
PRE_UNSMEAR_K_FOLDS=$K_FOLDS,\
PRE_UNSMEAR_KFOLD_MODEL_DIR=$PRE_UNSMEAR_KFOLD_MODEL_DIR,\
PRE_UNSMEAR_KFOLD_USE_PRETRAINED=1,\
PRE_UNSMEAR_KFOLD_ENSEMBLE=1,\
PRE_UNSMEAR_WEIGHT=$PRE_UNSMEAR_WEIGHT \
  run_unmerge_distr_kfold_final_unsmear.sh | awk '{print $4}')
echo "Cache job submitted as Job $cache_jid"

echo "Submitting unsmearer k-fold + final classifiers (after cache job)..."
env \
  SAVE_DIR="$SAVE_DIR" RUN_NAME="$RUN_NAME" \
  K_FOLDS="$K_FOLDS" KFOLD_ENSEMBLE="$KFOLD_ENSEMBLE" \
  N_TRAIN_JETS="$N_TRAIN_JETS" MAX_CONSTITS="$MAX_CONSTITS" MAX_MERGE_COUNT="$MAX_MERGE_COUNT" \
  NUM_WORKERS="$NUM_WORKERS" TRAIN_PATH="$TRAIN_PATH" \
  UNMERGED_CACHE="$UNMERGED_CACHE" \
  UNSMEAR_KFOLD_MODEL_DIR="$UNSMEAR_KFOLD_MODEL_DIR" \
  UNSMEAR_MERGE_FLAG="$UNSMEAR_MERGE_FLAG" \
  UNSMEAR_MERGE_WEIGHT="$UNSMEAR_MERGE_WEIGHT" \
  UNSMEAR_TWO_HEAD="$UNSMEAR_TWO_HEAD" \
  DEPENDS_ON="$cache_jid" \
  bash submit_unmerge_doublekfold_pipeline_unsmear.sh
