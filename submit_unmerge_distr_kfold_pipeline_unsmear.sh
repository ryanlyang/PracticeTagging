#!/bin/bash

# Submit K-fold training jobs, then a dependent final job.

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
TRAIN_PATH=$TRAIN_PATH \
  run_unmerge_distr_fold_unsmear.sh | awk '{print $4}')
  echo "  Fold $i submitted as Job $jid"
  job_ids+=("$jid")
  sleep 0.2
 done

dep=$(IFS=:; echo "${job_ids[*]}")

echo "Submitting final job with dependency afterok:$dep"
final_jid=$(sbatch --dependency=afterok:$dep --export=ALL,\
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
TRAIN_PATH=$TRAIN_PATH \
  run_unmerge_distr_kfold_final_unsmear.sh | awk '{print $4}')

echo "Final job submitted as Job $final_jid"

echo "Classifier-only sweeps will run inside the final job."
