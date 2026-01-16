#!/bin/bash
set -euo pipefail

# SuperRes sweep for SuperRes.py (20 runs)
#
# Block A (16 runs): SR + KD core hyperparameters
#   SR_SAMPLES: 2, 4
#   KD_TEMP: 2.5, 3.5
#   LOSS_KD: 0.3, 0.6
#   LOSS_EMB: 0.1, 0.3
#
# Block B (4 runs): generator regularization
#   GEN_LAMBDA_MASK: 0.5, 1.0
#   GEN_LAMBDA_STATS: 0.05, 0.1

SAVE_DIR=${SAVE_DIR:-"checkpoints/superres_sweep"}
TRAIN_PATH=${TRAIN_PATH:-""}
N_TRAIN_JETS=${N_TRAIN_JETS:-100000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
TEACHER_CKPT=${TEACHER_CKPT:-""}
BASELINE_CKPT=${BASELINE_CKPT:-""}
SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS:-0}
SR_MODE=${SR_MODE:-"both"}

# Fixed defaults (can be overridden via environment)
SR_EPOCHS=${SR_EPOCHS:-50}
SR_LR=${SR_LR:-5e-4}
SR_MASK_THRESHOLD=${SR_MASK_THRESHOLD:-0.5}

GEN_EPOCHS=${GEN_EPOCHS:-30}
GEN_LR=${GEN_LR:-5e-4}
GEN_LAMBDA_FEAT=${GEN_LAMBDA_FEAT:-1.0}
GEN_LAMBDA_MULT=${GEN_LAMBDA_MULT:-0.1}
GEN_LAMBDA_KL=${GEN_LAMBDA_KL:-0.01}

LOSS_SUP=${LOSS_SUP:-1.0}
LOSS_CONS_PROB=${LOSS_CONS_PROB:-0.1}
LOSS_CONS_EMB=${LOSS_CONS_EMB:-0.05}
LOSS_ATTN=${LOSS_ATTN:-0.0}

mkdir -p superres_logs

run_count=0

submit_job() {
    local run_name="$1"
    local sr_samples="$2"
    local kd_temp="$3"
    local loss_kd="$4"
    local loss_emb="$5"
    local gen_lambda_mask="$6"
    local gen_lambda_stats="$7"

    local exports="ALL"
    exports+=",RUN_NAME=${run_name}"
    exports+=",SR_MODE=${SR_MODE}"
    exports+=",SR_SAMPLES=${sr_samples}"
    exports+=",SR_EVAL_SAMPLES=${sr_samples}"
    exports+=",SR_EPOCHS=${SR_EPOCHS}"
    exports+=",SR_LR=${SR_LR}"
    exports+=",SR_MASK_THRESHOLD=${SR_MASK_THRESHOLD}"
    exports+=",KD_TEMP=${kd_temp}"
    exports+=",LOSS_SUP=${LOSS_SUP}"
    exports+=",LOSS_CONS_PROB=${LOSS_CONS_PROB}"
    exports+=",LOSS_CONS_EMB=${LOSS_CONS_EMB}"
    exports+=",LOSS_KD=${loss_kd}"
    exports+=",LOSS_EMB=${loss_emb}"
    exports+=",LOSS_ATTN=${LOSS_ATTN}"
    exports+=",GEN_EPOCHS=${GEN_EPOCHS}"
    exports+=",GEN_LR=${GEN_LR}"
    exports+=",GEN_LAMBDA_FEAT=${GEN_LAMBDA_FEAT}"
    exports+=",GEN_LAMBDA_MASK=${gen_lambda_mask}"
    exports+=",GEN_LAMBDA_MULT=${GEN_LAMBDA_MULT}"
    exports+=",GEN_LAMBDA_STATS=${gen_lambda_stats}"
    exports+=",GEN_LAMBDA_KL=${GEN_LAMBDA_KL}"
    exports+=",SAVE_DIR=${SAVE_DIR}"
    exports+=",SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS}"

    if [ -n "$TRAIN_PATH" ]; then
        exports+=",TRAIN_PATH=${TRAIN_PATH}"
    fi
    if [ -n "$N_TRAIN_JETS" ]; then
        exports+=",N_TRAIN_JETS=${N_TRAIN_JETS}"
    fi
    if [ -n "$MAX_CONSTITS" ]; then
        exports+=",MAX_CONSTITS=${MAX_CONSTITS}"
    fi
    if [ -n "$TEACHER_CKPT" ]; then
        exports+=",TEACHER_CKPT=${TEACHER_CKPT}"
    fi
    if [ -n "$BASELINE_CKPT" ]; then
        exports+=",BASELINE_CKPT=${BASELINE_CKPT}"
    fi

    sbatch --export="$exports" run_superres_single.sh
    run_count=$((run_count + 1))
}

echo "Block A: SR+KD core hyperparameters (16 runs)"
SR_SAMPLES_LIST=(2 4)
KD_TEMP_LIST=(2.5 3.5)
LOSS_KD_LIST=(0.3 0.6)
LOSS_EMB_LIST=(0.1 0.3)

GEN_LAMBDA_MASK_A=0.5
GEN_LAMBDA_STATS_A=0.1

for K in "${SR_SAMPLES_LIST[@]}"; do
  for T in "${KD_TEMP_LIST[@]}"; do
    for LK in "${LOSS_KD_LIST[@]}"; do
      for LE in "${LOSS_EMB_LIST[@]}"; do
        t_tag=${T//./p}
        lk_tag=${LK//./p}
        le_tag=${LE//./p}
        RUN_NAME="A_K${K}_T${t_tag}_lk${lk_tag}_le${le_tag}"
        submit_job "$RUN_NAME" "$K" "$T" "$LK" "$LE" \
          "$GEN_LAMBDA_MASK_A" "$GEN_LAMBDA_STATS_A"
      done
    done
  done
done

echo "Block B: generator regularization (4 runs)"
GEN_LAMBDA_MASK_LIST=(0.5 1.0)
GEN_LAMBDA_STATS_LIST=(0.05 0.1)

K_FIXED=4
T_FIXED=3.0
LK_FIXED=0.6
LE_FIXED=0.2

for LM in "${GEN_LAMBDA_MASK_LIST[@]}"; do
  for LS in "${GEN_LAMBDA_STATS_LIST[@]}"; do
    lm_tag=${LM//./p}
    ls_tag=${LS//./p}
    RUN_NAME="B_lm${lm_tag}_ls${ls_tag}"
    submit_job "$RUN_NAME" "$K_FIXED" "$T_FIXED" "$LK_FIXED" "$LE_FIXED" \
      "$LM" "$LS"
  done
done

echo "Submitted ${run_count} jobs to Slurm."
