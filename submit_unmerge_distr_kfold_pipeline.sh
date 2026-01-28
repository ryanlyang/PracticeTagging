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
TRAIN_PATH=$TRAIN_PATH \
  run_unmerge_distr_fold.sh | awk '{print $4}')
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
TRAIN_PATH=$TRAIN_PATH \
  run_unmerge_distr_kfold_final.sh | awk '{print $4}')

echo "Final job submitted as Job $final_jid"

echo "Submitting classifier-only sweeps (after final job)..."
# 12 runs: mix KD details, MC on/off, self-train on/off, + KD-off baselines
declare -a TAGS=(\"mc0_st1\" \"mc4_st1\" \"mc4_hiCons\" \"mc0_nost\" \"kd0_mc0\" \"kd0_mc4\" \"rep_hi\" \"nce_hi\" \"attn_hi\" \"temp_lo\" \"temp_hi\" \"conf_off\")
declare -a MC_S=(1 4 4 1 1 4 1 1 1 1 1 1)
declare -a MC_W=(0.1 0.1 0.3 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1)
declare -a NO_ST=(0 0 0 1 1 1 0 0 0 0 0 0)
declare -a ALPHA_KD=(\"\" \"\" \"\" \"\" \"0.0\" \"0.0\" \"\" \"\" \"\" \"\" \"\" \"\")
declare -a ALPHA_REP=(\"\" \"\" \"\" \"\" \"\" \"\" \"0.30\" \"\" \"\" \"\" \"\" \"\")
declare -a ALPHA_NCE=(\"\" \"\" \"\" \"\" \"\" \"\" \"\" \"0.30\" \"\" \"\" \"\" \"\")
declare -a ALPHA_ATTN=(\"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" \"0.15\" \"\" \"\" \"\")
declare -a KD_TEMP=(\"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" \"5.0\" \"9.0\" \"\")
declare -a NO_CONF=(0 0 0 0 0 0 0 0 0 0 0 1)

for i in "${!TAGS[@]}"; do
  tag=${TAGS[$i]}
  ms=${MC_S[$i]}
  mw=${MC_W[$i]}
  ns=${NO_ST[$i]}
  akd=${ALPHA_KD[$i]}
  arep=${ALPHA_REP[$i]}
  ance=${ALPHA_NCE[$i]}
  aattn=${ALPHA_ATTN[$i]}
  kdt=${KD_TEMP[$i]}
  nconf=${NO_CONF[$i]}
  jid=$(sbatch --dependency=afterok:$final_jid --export=ALL,\
TAG=$tag,\
MC_SAMPLES=$ms,\
MC_CONS_W=$mw,\
NO_SELF_TRAIN=$ns,\
ALPHA_KD=$akd,\
ALPHA_REP=$arep,\
ALPHA_NCE=$ance,\
ALPHA_ATTN=$aattn,\
KD_TEMP=$kdt,\
NO_CONF_KD=$nconf,\
SAVE_DIR=$SAVE_DIR,\
RUN_NAME=$RUN_NAME,\
N_TRAIN_JETS=$N_TRAIN_JETS,\
MAX_CONSTITS=$MAX_CONSTITS,\
TRAIN_PATH=$TRAIN_PATH \
  run_unmerge_distr_classifier_only.sh | awk '{print $4}')\n  echo \"  Classifier $tag submitted as Job $jid\"\n+  sleep 0.2\n+done
