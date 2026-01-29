#!/bin/bash

# Submit unsmearer K-fold training jobs AFTER a base unmerger pipeline is done,
# then a dependent final job that uses the pretrained unsmearer folds + cached unmerged data.

K_FOLDS=${K_FOLDS:-5}
SAVE_DIR=${SAVE_DIR:-"checkpoints/unmerge_distr_kfold_unsmear_sweep"}
RUN_NAME=${RUN_NAME:-"kfold_base_det_unsmear"}
UNMERGED_CACHE=${UNMERGED_CACHE:-"$SAVE_DIR/$RUN_NAME/unmerged_cache.npz"}
UNSMEAR_KFOLD_MODEL_DIR=${UNSMEAR_KFOLD_MODEL_DIR:-"$SAVE_DIR/$RUN_NAME/unsmear_kfold_models"}

# Optional overrides
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
MAX_MERGE_COUNT=${MAX_MERGE_COUNT:-10}
NUM_WORKERS=${NUM_WORKERS:-6}
KFOLD_ENSEMBLE=${KFOLD_ENSEMBLE:-1}
TRAIN_PATH=${TRAIN_PATH:-""}
DEPENDS_ON=${DEPENDS_ON:-""}
TEACHER_CKPT=${TEACHER_CKPT:-"$SAVE_DIR/$RUN_NAME/teacher.pt"}
BASELINE_CKPT=${BASELINE_CKPT:-"$SAVE_DIR/$RUN_NAME/baseline.pt"}
UNSMEAR_MERGE_FLAG=${UNSMEAR_MERGE_FLAG:-0}
UNSMEAR_MERGE_WEIGHT=${UNSMEAR_MERGE_WEIGHT:-2.0}
UNSMEAR_TWO_HEAD=${UNSMEAR_TWO_HEAD:-0}

mkdir -p unmerge_distr_kfold_logs

job_ids=()
dep_opt=""
if [ -n "$DEPENDS_ON" ]; then
  dep_opt="--dependency=afterok:$DEPENDS_ON"
fi

echo "Submitting unsmearer $K_FOLDS fold jobs (after dependency: ${DEPENDS_ON:-none})..."
for ((i=0; i<K_FOLDS; i++)); do
  jid=$(sbatch $dep_opt --export=ALL,\
FOLD_ID=$i,\
UNSMEAR_K_FOLDS=$K_FOLDS,\
SAVE_DIR=$SAVE_DIR,\
RUN_NAME=$RUN_NAME,\
UNSMEAR_KFOLD_MODEL_DIR=$UNSMEAR_KFOLD_MODEL_DIR,\
UNMERGED_CACHE=$UNMERGED_CACHE,\
N_TRAIN_JETS=$N_TRAIN_JETS,\
MAX_CONSTITS=$MAX_CONSTITS,\
MAX_MERGE_COUNT=$MAX_MERGE_COUNT,\
NUM_WORKERS=$NUM_WORKERS,\
UNSMEAR_MERGE_FLAG=$UNSMEAR_MERGE_FLAG,\
UNSMEAR_MERGE_WEIGHT=$UNSMEAR_MERGE_WEIGHT,\
UNSMEAR_TWO_HEAD=$UNSMEAR_TWO_HEAD \
  run_unsmear_kfold_fold_unsmear.sh | awk '{print $4}')
  echo "  Unsmear fold $i submitted as Job $jid"
  job_ids+=("$jid")
  sleep 0.2
done

dep=$(IFS=:; echo "${job_ids[*]}")

echo "Submitting final job with dependency afterok:$dep"
final_jid=$(sbatch --dependency=afterok:$dep --export=ALL,\
K_FOLDS=$K_FOLDS,\
SAVE_DIR=$SAVE_DIR,\
RUN_NAME=$RUN_NAME,\
N_TRAIN_JETS=$N_TRAIN_JETS,\
MAX_CONSTITS=$MAX_CONSTITS,\
MAX_MERGE_COUNT=$MAX_MERGE_COUNT,\
NUM_WORKERS=$NUM_WORKERS,\
KFOLD_ENSEMBLE=$KFOLD_ENSEMBLE,\
TRAIN_PATH=$TRAIN_PATH,\
CLASSIFIER_ONLY=1,\
UNMERGED_CACHE=$UNMERGED_CACHE,\
UNSMEAR_K_FOLDS=$K_FOLDS,\
UNSMEAR_KFOLD_MODEL_DIR=$UNSMEAR_KFOLD_MODEL_DIR,\
UNSMEAR_KFOLD_USE_PRETRAINED=1,\
UNSMEAR_KFOLD_ENSEMBLE=1,\
TEACHER_CKPT=$TEACHER_CKPT,\
BASELINE_CKPT=$BASELINE_CKPT,\
UNSMEAR_MERGE_FLAG=$UNSMEAR_MERGE_FLAG,\
UNSMEAR_MERGE_WEIGHT=$UNSMEAR_MERGE_WEIGHT,\
UNSMEAR_TWO_HEAD=$UNSMEAR_TWO_HEAD \
  run_unmerge_distr_kfold_final_unsmear.sh | awk '{print $4}')

echo "Final job submitted as Job $final_jid"
echo "Classifier-only sweeps will run inside the final job."
