#!/bin/bash

#SBATCH --job-name=unmerge_distr_final
#SBATCH --output=unmerge_distr_kfold_logs/unmerge_distr_final_%j.out
#SBATCH --error=unmerge_distr_kfold_logs/unmerge_distr_final_%j.err
#SBATCH --time=19-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

mkdir -p unmerge_distr_kfold_logs

echo "=========================================="
echo "Unmerge Distributional Model (K-fold final)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

source ~/.bashrc
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi
echo ""

TRAIN_PATH=${TRAIN_PATH:-""}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
MAX_MERGE_COUNT=${MAX_MERGE_COUNT:-10}
SAVE_DIR=${SAVE_DIR:-"checkpoints/unmerge_distr_kfold"}
RUN_NAME=${RUN_NAME:-"kfold_run"}
UNMERGE_LOSS=${UNMERGE_LOSS:-"hungarian"}
NUM_WORKERS=${NUM_WORKERS:-6}
USE_TRUE_COUNT=${USE_TRUE_COUNT:-0}
NO_CURRICULUM=${NO_CURRICULUM:-0}
CURR_START=${CURR_START:-2}
CURR_EPOCHS=${CURR_EPOCHS:-20}
PHYSICS_WEIGHT=${PHYSICS_WEIGHT:-0.2}
NLL_WEIGHT=${NLL_WEIGHT:-1.0}
NO_DISTRIBUTIONAL=${NO_DISTRIBUTIONAL:-0}
K_FOLDS=${K_FOLDS:-5}
KFOLD_ENSEMBLE=${KFOLD_ENSEMBLE:-1}
KFOLD_MODEL_DIR=${KFOLD_MODEL_DIR:-"$SAVE_DIR/$RUN_NAME/kfold_models"}
MC_SWEEP=${MC_SWEEP:-0}

CMD="python unmerge_distr_model.py \
  --save_dir $SAVE_DIR \
  --run_name $RUN_NAME \
  --n_train_jets $N_TRAIN_JETS \
  --max_constits $MAX_CONSTITS \
  --max_merge_count $MAX_MERGE_COUNT \
  --unmerge_loss $UNMERGE_LOSS \
  --num_workers $NUM_WORKERS \
  --curriculum_start $CURR_START \
  --curriculum_epochs $CURR_EPOCHS \
  --physics_weight $PHYSICS_WEIGHT \
  --nll_weight $NLL_WEIGHT \
  --k_folds $K_FOLDS \
  --kfold_model_dir $KFOLD_MODEL_DIR \
  --kfold_use_pretrained \
  --save_unmerged_cache"

if [ -n "$TRAIN_PATH" ]; then
  CMD="$CMD --train_path $TRAIN_PATH"
fi

if [ "$USE_TRUE_COUNT" -eq 1 ]; then
  CMD="$CMD --use_true_count"
fi

if [ "$KFOLD_ENSEMBLE" -eq 1 ]; then
  CMD="$CMD --kfold_ensemble_valtest"
fi

if [ "$NO_CURRICULUM" -eq 1 ]; then
  CMD="$CMD --no_curriculum"
fi

if [ "$NO_DISTRIBUTIONAL" -eq 1 ]; then
  CMD="$CMD --no_distributional"
fi

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  CMD="$CMD --device cuda"
else
  CMD="$CMD --device cpu"
fi

echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "Classifier-only sweeps (sequential)"
  echo "=========================================="

  CACHE_PATH="$SAVE_DIR/$RUN_NAME/unmerged_cache.npz"
  MC_CACHE_PATH="$SAVE_DIR/$RUN_NAME/unmerged_mc_cache.npz"
  TEACHER_CKPT="$SAVE_DIR/$RUN_NAME/teacher.pt"
  BASELINE_CKPT="$SAVE_DIR/$RUN_NAME/baseline.pt"

  if [ "$MC_SWEEP" -eq 1 ]; then
    declare -a TAGS=("mc1" "mc2" "mc4" "mc8" "mc16" "mc32")
    declare -a MC_S=(1 2 4 8 16 32)
    declare -a MC_W=(0.1 0.1 0.1 0.08 0.05 0.03)
    declare -a NO_ST=(0 0 0 0 0 0)
    declare -a ALPHA_KD=("" "" "" "" "" "")
    declare -a ALPHA_REP=("" "" "" "" "" "")
    declare -a ALPHA_NCE=("" "" "" "" "" "")
    declare -a ALPHA_ATTN=("" "" "" "" "" "")
    declare -a KD_TEMP=("" "" "" "" "" "")
    declare -a NO_CONF=(0 0 0 0 0 0)
  else
    declare -a TAGS=("mc0_st1" "mc4_st1" "mc4_hiCons" "mc0_nost" "kd0_mc0" "kd0_mc4" "rep_hi" "nce_hi" "attn_hi" "temp_lo" "temp_hi" "conf_off")
    declare -a MC_S=(1 4 4 1 1 4 1 1 1 1 1 1)
    declare -a MC_W=(0.1 0.1 0.3 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1)
    declare -a NO_ST=(0 0 0 1 1 1 0 0 0 0 0 0)
    declare -a ALPHA_KD=("" "" "" "" "0.0" "0.0" "" "" "" "" "" "")
    declare -a ALPHA_REP=("" "" "" "" "" "" "0.30" "" "" "" "" "")
    declare -a ALPHA_NCE=("" "" "" "" "" "" "" "0.30" "" "" "" "")
    declare -a ALPHA_ATTN=("" "" "" "" "" "" "" "" "0.15" "" "" "")
    declare -a KD_TEMP=("" "" "" "" "" "" "" "" "" "5.0" "9.0" "")
    declare -a NO_CONF=(0 0 0 0 0 0 0 0 0 0 0 1)
  fi

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

    CMD_CLS="python unmerge_distr_model.py \
      --save_dir $SAVE_DIR \
      --run_name ${RUN_NAME}_cls_${tag} \
      --n_train_jets $N_TRAIN_JETS \
      --max_constits $MAX_CONSTITS \
      --num_workers $NUM_WORKERS \
      --classifier_only \
      --load_unmerged_cache $CACHE_PATH \
      --teacher_checkpoint $TEACHER_CKPT \
      --baseline_checkpoint $BASELINE_CKPT \
      --k_folds 1 \
      --mc_samples $ms \
      --mc_consistency_weight $mw"

    if [ "$ns" -eq 1 ]; then
      CMD_CLS="$CMD_CLS --no_self_train"
    fi
    if [ -n "$akd" ]; then
      CMD_CLS="$CMD_CLS --alpha_kd $akd"
    fi
    if [ -n "$arep" ]; then
      CMD_CLS="$CMD_CLS --alpha_rep $arep"
    fi
    if [ -n "$ance" ]; then
      CMD_CLS="$CMD_CLS --alpha_nce $ance"
    fi
    if [ -n "$aattn" ]; then
      CMD_CLS="$CMD_CLS --alpha_attn $aattn"
    fi
    if [ -n "$kdt" ]; then
      CMD_CLS="$CMD_CLS --kd_temp $kdt"
    fi
    if [ "$nconf" -eq 1 ]; then
      CMD_CLS="$CMD_CLS --no_conf_kd"
    fi
    if [ "$ms" -gt 1 ] && [ -f "$MC_CACHE_PATH" ]; then
      CMD_CLS="$CMD_CLS --load_mc_cache $MC_CACHE_PATH"
    fi
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
      CMD_CLS="$CMD_CLS --device cuda"
    else
      CMD_CLS="$CMD_CLS --device cpu"
    fi

    echo ""
    echo "Classifier run: $tag"
    echo "$CMD_CLS"
    eval $CMD_CLS
  done
fi

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
  echo "Final run completed successfully"
  echo "Results saved to: $SAVE_DIR/$RUN_NAME"
else
  echo "Final run failed with exit code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
