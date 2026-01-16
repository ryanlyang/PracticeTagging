#!/bin/bash
set -euo pipefail

# SuperResDiff sweep (30 runs)
#
# Block A (18 runs): SR samples + generator regularization
#   SR_SAMPLES: 2, 4, 6
#   GEN_LAMBDA_MASK: 0.25, 0.5, 1.0
#   GEN_LAMBDA_STATS: 0.02, 0.05
#
# Block B (6 runs): generator min_sigma
#   GEN_MIN_SIGMA: 0.01, 0.02, 0.04
#   SR_SAMPLES: 4, 6
#
# Block C (6 runs): SR learning rate
#   SR_LR: 3e-4, 5e-4, 8e-4
#   SR_SAMPLES: 4, 6

SAVE_DIR=${SAVE_DIR:-"checkpoints/superres_diff_sweep"}
TRAIN_PATH=${TRAIN_PATH:-""}
N_TRAIN_JETS=${N_TRAIN_JETS:-100000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
TEACHER_CKPT=${TEACHER_CKPT:-""}
BASELINE_CKPT=${BASELINE_CKPT:-""}
SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS:-0}
SR_MODE=${SR_MODE:-"both"}

# Fixed defaults (can be overridden via environment)
SR_EPOCHS=${SR_EPOCHS:-50}
SR_MASK_THRESHOLD=${SR_MASK_THRESHOLD:-0.5}

GEN_EPOCHS=${GEN_EPOCHS:-30}
GEN_LR=${GEN_LR:-5e-4}
GEN_LAMBDA_FEAT=${GEN_LAMBDA_FEAT:-1.0}
GEN_LAMBDA_MULT=${GEN_LAMBDA_MULT:-0.1}
GEN_LAMBDA_KL=${GEN_LAMBDA_KL:-0.01}

LOSS_SUP=${LOSS_SUP:-1.0}
LOSS_CONS_PROB=${LOSS_CONS_PROB:-0.1}
LOSS_CONS_EMB=${LOSS_CONS_EMB:-0.05}

# Fixed KD distribution hyperparameters
CONF_POWER=${CONF_POWER:-2.0}
CONF_MIN=${CONF_MIN:-0.0}
TEMP_INIT=${TEMP_INIT:-7.0}
TEMP_FINAL=${TEMP_FINAL:-""}
ALPHA_INIT=${ALPHA_INIT:-0.5}
ALPHA_FINAL=${ALPHA_FINAL:-""}
ALPHA_ATTN=${ALPHA_ATTN:-0.05}
ALPHA_REP=${ALPHA_REP:-0.10}
ALPHA_NCE=${ALPHA_NCE:-0.10}
TAU_NCE=${TAU_NCE:-0.10}
CONF_WEIGHTED_KD=${CONF_WEIGHTED_KD:-1}

mkdir -p superres_diff_logs

run_count=0

submit_job() {
    local run_name="$1"
    local sr_samples="$2"
    local sr_lr="$3"
    local gen_min_sigma="$4"
    local gen_lambda_mask="$5"
    local gen_lambda_stats="$6"

    local exports="ALL"
    exports+=",RUN_NAME=${run_name}"
    exports+=",SR_MODE=${SR_MODE}"
    exports+=",SR_SAMPLES=${sr_samples}"
    exports+=",SR_EVAL_SAMPLES=${sr_samples}"
    exports+=",SR_EPOCHS=${SR_EPOCHS}"
    exports+=",SR_LR=${sr_lr}"
    exports+=",SR_MASK_THRESHOLD=${SR_MASK_THRESHOLD}"
    exports+=",GEN_MIN_SIGMA=${gen_min_sigma}"
    exports+=",GEN_EPOCHS=${GEN_EPOCHS}"
    exports+=",GEN_LR=${GEN_LR}"
    exports+=",GEN_LAMBDA_FEAT=${GEN_LAMBDA_FEAT}"
    exports+=",GEN_LAMBDA_MASK=${gen_lambda_mask}"
    exports+=",GEN_LAMBDA_MULT=${GEN_LAMBDA_MULT}"
    exports+=",GEN_LAMBDA_STATS=${gen_lambda_stats}"
    exports+=",GEN_LAMBDA_KL=${GEN_LAMBDA_KL}"
    exports+=",LOSS_SUP=${LOSS_SUP}"
    exports+=",LOSS_CONS_PROB=${LOSS_CONS_PROB}"
    exports+=",LOSS_CONS_EMB=${LOSS_CONS_EMB}"
    exports+=",CONF_POWER=${CONF_POWER}"
    exports+=",CONF_MIN=${CONF_MIN}"
    exports+=",TEMP_INIT=${TEMP_INIT}"
    exports+=",TEMP_FINAL=${TEMP_FINAL}"
    exports+=",ALPHA_INIT=${ALPHA_INIT}"
    exports+=",ALPHA_FINAL=${ALPHA_FINAL}"
    exports+=",ALPHA_ATTN=${ALPHA_ATTN}"
    exports+=",ALPHA_REP=${ALPHA_REP}"
    exports+=",ALPHA_NCE=${ALPHA_NCE}"
    exports+=",TAU_NCE=${TAU_NCE}"
    exports+=",CONF_WEIGHTED_KD=${CONF_WEIGHTED_KD}"
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

    sbatch --export="$exports" run_superres_diff_single.sh
    run_count=$((run_count + 1))
}

echo "Block A: SR samples + generator regularization (18 runs)"
SR_SAMPLES_LIST_A=(2 4 6)
GEN_LAMBDA_MASK_LIST_A=(0.25 0.5 1.0)
GEN_LAMBDA_STATS_LIST_A=(0.02 0.05)

SR_LR_A=${SR_LR_A:-5e-4}
GEN_MIN_SIGMA_A=${GEN_MIN_SIGMA_A:-0.02}

for K in "${SR_SAMPLES_LIST_A[@]}"; do
  for LM in "${GEN_LAMBDA_MASK_LIST_A[@]}"; do
    for LS in "${GEN_LAMBDA_STATS_LIST_A[@]}"; do
      lm_tag=${LM//./p}
      ls_tag=${LS//./p}
      RUN_NAME="A_K${K}_lm${lm_tag}_ls${ls_tag}"
      submit_job "$RUN_NAME" "$K" "$SR_LR_A" "$GEN_MIN_SIGMA_A" "$LM" "$LS"
    done
  done
done

echo "Block B: generator min_sigma (6 runs)"
GEN_MIN_SIGMA_LIST_B=(0.01 0.02 0.04)
SR_SAMPLES_LIST_B=(4 6)

GEN_LAMBDA_MASK_B=0.5
GEN_LAMBDA_STATS_B=0.05
SR_LR_B=5e-4

for SIG in "${GEN_MIN_SIGMA_LIST_B[@]}"; do
  for K in "${SR_SAMPLES_LIST_B[@]}"; do
    s_tag=${SIG//./p}
    RUN_NAME="B_sig${s_tag}_K${K}"
    submit_job "$RUN_NAME" "$K" "$SR_LR_B" "$SIG" "$GEN_LAMBDA_MASK_B" "$GEN_LAMBDA_STATS_B"
  done
done

echo "Block C: SR learning rate (6 runs)"
SR_LR_LIST_C=(3e-4 5e-4 8e-4)
SR_SAMPLES_LIST_C=(4 6)

GEN_MIN_SIGMA_C=0.02
GEN_LAMBDA_MASK_C=0.5
GEN_LAMBDA_STATS_C=0.05

for LR in "${SR_LR_LIST_C[@]}"; do
  for K in "${SR_SAMPLES_LIST_C[@]}"; do
    lr_tag=${LR//./p}
    RUN_NAME="C_lr${lr_tag}_K${K}"
    submit_job "$RUN_NAME" "$K" "$LR" "$GEN_MIN_SIGMA_C" "$GEN_LAMBDA_MASK_C" "$GEN_LAMBDA_STATS_C"
  done
done

echo "Submitted ${run_count} jobs to Slurm."
