#!/bin/bash

# Submit K-fold training jobs + a full-train unmerger, then a dependent final job
# that uses the full-train unmerger for val/test generation (Option C).

K_FOLDS=${K_FOLDS:-5}
SAVE_DIR=${SAVE_DIR:-"checkpoints/unmerge_distr_kfold"}
RUN_NAME=${RUN_NAME:-"kfold_run"}
KFOLD_MODEL_DIR=${KFOLD_MODEL_DIR:-"$SAVE_DIR/$RUN_NAME/kfold_models"}

# Optional overrides
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
MAX_MERGE_COUNT=${MAX_MERGE_COUNT:-10}
UNMERGE_LOSS=${UNMERGE_LOSS:-"hungarian"}
USE_TRUE_COUNT=${USE_TRUE_COUNT:-0}
NO_CURRICULUM=${NO_CURRICULUM:-0}
CURR_START=${CURR_START:-2}
CURR_EPOCHS=${CURR_EPOCHS:-20}
PHYSICS_WEIGHT=${PHYSICS_WEIGHT:-0.2}
NLL_WEIGHT=${NLL_WEIGHT:-1.0}
NO_DISTRIBUTIONAL=${NO_DISTRIBUTIONAL:-0}
KFOLD_ENSEMBLE=${KFOLD_ENSEMBLE:-1}
TRAIN_PATH=${TRAIN_PATH:-""}
MC_SWEEP=${MC_SWEEP:-0}
NUM_WORKERS=${NUM_WORKERS:-6}

mkdir -p unmerge_distr_kfold_logs

job_ids=()

echo "Submitting $K_FOLDS fold jobs..."
for ((i=0; i<K_FOLDS; i++)); do
  jid=$(sbatch --export=ALL,\
FOLD_ID=$i,\
K_FOLDS=$K_FOLDS,\
SAVE_DIR=$SAVE_DIR,\
RUN_NAME=$RUN_NAME,\
KFOLD_MODEL_DIR=$KFOLD_MODEL_DIR,\
N_TRAIN_JETS=$N_TRAIN_JETS,\
MAX_CONSTITS=$MAX_CONSTITS,\
MAX_MERGE_COUNT=$MAX_MERGE_COUNT,\
UNMERGE_LOSS=$UNMERGE_LOSS,\
USE_TRUE_COUNT=$USE_TRUE_COUNT,\
NO_CURRICULUM=$NO_CURRICULUM,\
CURR_START=$CURR_START,\
CURR_EPOCHS=$CURR_EPOCHS,\
PHYSICS_WEIGHT=$PHYSICS_WEIGHT,\
NLL_WEIGHT=$NLL_WEIGHT,\
NO_DISTRIBUTIONAL=$NO_DISTRIBUTIONAL,\
MC_SWEEP=$MC_SWEEP,\
TRAIN_PATH=$TRAIN_PATH,\
NUM_WORKERS=$NUM_WORKERS \
  run_unmerge_distr_fold.sh | awk '{print $4}')
  echo "  Fold $i submitted as Job $jid"
  job_ids+=("$jid")
  sleep 0.2
 done

dep_folds=$(IFS=:; echo "${job_ids[*]}")

# Full-train unmerger job (Option C)
FULL_RUN_NAME="${RUN_NAME}_full"
FULL_DIR="$SAVE_DIR/$FULL_RUN_NAME"

echo "Submitting full-train unmerger job (runs alongside folds)..."
full_jid=$(sbatch --export=ALL,\
SAVE_DIR=$SAVE_DIR,\
RUN_NAME=$FULL_RUN_NAME,\
N_TRAIN_JETS=$N_TRAIN_JETS,\
MAX_CONSTITS=$MAX_CONSTITS,\
MAX_MERGE_COUNT=$MAX_MERGE_COUNT,\
UNMERGE_LOSS=$UNMERGE_LOSS,\
USE_TRUE_COUNT=$USE_TRUE_COUNT,\
NO_CURRICULUM=$NO_CURRICULUM,\
CURR_START=$CURR_START,\
CURR_EPOCHS=$CURR_EPOCHS,\
PHYSICS_WEIGHT=$PHYSICS_WEIGHT,\
NLL_WEIGHT=$NLL_WEIGHT,\
NO_DISTRIBUTIONAL=$NO_DISTRIBUTIONAL,\
TRAIN_PATH=$TRAIN_PATH,\
NUM_WORKERS=$NUM_WORKERS \
  run_unmerge_distr_full.sh | awk '{print $4}')

echo "  Full-train unmerger submitted as Job $full_jid"

# Final job depends on folds + full-train unmerger
final_dep="$dep_folds:$full_jid"

echo "Submitting final job with dependency afterok:$final_dep"
final_jid=$(sbatch --dependency=afterok:$final_dep --export=ALL,\
K_FOLDS=$K_FOLDS,\
SAVE_DIR=$SAVE_DIR,\
RUN_NAME=$RUN_NAME,\
KFOLD_MODEL_DIR=$KFOLD_MODEL_DIR,\
N_TRAIN_JETS=$N_TRAIN_JETS,\
MAX_CONSTITS=$MAX_CONSTITS,\
MAX_MERGE_COUNT=$MAX_MERGE_COUNT,\
UNMERGE_LOSS=$UNMERGE_LOSS,\
USE_TRUE_COUNT=$USE_TRUE_COUNT,\
NO_CURRICULUM=$NO_CURRICULUM,\
CURR_START=$CURR_START,\
CURR_EPOCHS=$CURR_EPOCHS,\
PHYSICS_WEIGHT=$PHYSICS_WEIGHT,\
NLL_WEIGHT=$NLL_WEIGHT,\
NO_DISTRIBUTIONAL=$NO_DISTRIBUTIONAL,\
KFOLD_ENSEMBLE=$KFOLD_ENSEMBLE,\
MC_SWEEP=$MC_SWEEP,\
TRAIN_PATH=$TRAIN_PATH,\
NUM_WORKERS=$NUM_WORKERS,\
KFOLD_VALTEST_FULL_DIR=$FULL_DIR \
  run_unmerge_distr_kfold_final.sh | awk '{print $4}')

echo "Final job submitted as Job $final_jid"

echo "Classifier-only sweeps will run inside the final job."
