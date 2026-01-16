#!/bin/bash
set -euo pipefail

# A2 sweep for Offline_multi.py (18 runs)
#
# Block A (16 runs): student/KD hyperparameters
#   K_SAMPLES: 8, 16
#   KD_TEMP: 3.0, 4.0
#   A_KD: 0.7, 1.0
#   BETA_ALL: 0.0, 0.1
#
# Block B (2 runs): generator hyperparameters (focused)
#   GEN_MIN_SIGMA: 0.01, 0.02
#   GEN_LAMBDA_MASK: 1.0

SAVE_DIR=${SAVE_DIR:-"checkpoints/offline_multi_sweep"}
TRAIN_PATH=${TRAIN_PATH:-""}
N_TRAIN_JETS=${N_TRAIN_JETS:-100000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
TEACHER_CKPT=${TEACHER_CKPT:-""}
BASELINE_CKPT=${BASELINE_CKPT:-""}
GENERATOR_CKPT=${GENERATOR_CKPT:-""}
SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS:-0}

# Fixed weights (can be overridden via environment)
A_SUP=${A_SUP:-1.0}
A_EMB=${A_EMB:-1.0}
TAU_U=${TAU_U:-0.05}
W_MIN=${W_MIN:-0.1}
GEN_EPOCHS=${GEN_EPOCHS:-30}
GEN_LR=${GEN_LR:-1e-3}
GEN_GLOBAL_NOISE=${GEN_GLOBAL_NOISE:-0.05}

mkdir -p offline_multi_logs

run_count=0

submit_job() {
    local run_name="$1"
    local kd_temp="$2"
    local a_kd="$3"
    local k_samples="$4"
    local beta_all="$5"
    local gen_min_sigma="$6"
    local gen_lambda_mask="$7"
    local gen_lambda_perc="$8"
    local gen_lambda_logit="$9"

    local exports="ALL"
    exports+=",RUN_NAME=${run_name}"
    exports+=",KD_TEMP=${kd_temp}"
    exports+=",A_SUP=${A_SUP}"
    exports+=",A_KD=${a_kd}"
    exports+=",A_EMB=${A_EMB}"
    exports+=",BETA_ALL=${beta_all}"
    exports+=",TAU_U=${TAU_U}"
    exports+=",W_MIN=${W_MIN}"
    exports+=",K_SAMPLES=${k_samples}"
    exports+=",GEN_EPOCHS=${GEN_EPOCHS}"
    exports+=",GEN_LR=${GEN_LR}"
    exports+=",GEN_MIN_SIGMA=${gen_min_sigma}"
    exports+=",GEN_LAMBDA_MASK=${gen_lambda_mask}"
    exports+=",GEN_LAMBDA_PERC=${gen_lambda_perc}"
    exports+=",GEN_LAMBDA_LOGIT=${gen_lambda_logit}"
    exports+=",GEN_GLOBAL_NOISE=${GEN_GLOBAL_NOISE}"
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
    if [ -n "$GENERATOR_CKPT" ]; then
        exports+=",GENERATOR_CKPT=${GENERATOR_CKPT}"
    fi

    sbatch --export="$exports" run_offline_multi_single.sh
    run_count=$((run_count + 1))
}

echo "Block A: student/KD hyperparameters (16 runs)"
K_SAMPLES_LIST=(8 16)
KD_TEMP_LIST=(3.0 4.0)
A_KD_LIST=(0.7 1.0)
BETA_ALL_LIST=(0.0 0.1)

# Use default generator settings for Block A
GEN_MIN_SIGMA_A=0.02
GEN_LAMBDA_MASK_A=1.0
GEN_LAMBDA_PERC_A=1.0
GEN_LAMBDA_LOGIT_A=0.5

for K in "${K_SAMPLES_LIST[@]}"; do
  for T in "${KD_TEMP_LIST[@]}"; do
    for A in "${A_KD_LIST[@]}"; do
      for B in "${BETA_ALL_LIST[@]}"; do
        t_tag=${T//./p}
        a_tag=${A//./p}
        b_tag=${B//./p}
        RUN_NAME="A_K${K}_T${t_tag}_A${a_tag}_B${b_tag}"
        submit_job "$RUN_NAME" "$T" "$A" "$K" "$B" \
          "$GEN_MIN_SIGMA_A" "$GEN_LAMBDA_MASK_A" "$GEN_LAMBDA_PERC_A" "$GEN_LAMBDA_LOGIT_A"
      done
    done
  done
done

echo "Block B: generator hyperparameters (2 runs)"
GEN_MIN_SIGMA_LIST=(0.01 0.02)
GEN_LAMBDA_MASK_LIST=(1.0)
GEN_LAMBDA_PERC_LIST=(1.0)
GEN_LAMBDA_LOGIT_LIST=(0.5)

# Fix student/KD for Block B
K_FIXED=16
T_FIXED=4.0
A_FIXED=1.0
B_FIXED=0.1

for SIG in "${GEN_MIN_SIGMA_LIST[@]}"; do
  for LMASK in "${GEN_LAMBDA_MASK_LIST[@]}"; do
    for LPERC in "${GEN_LAMBDA_PERC_LIST[@]}"; do
      for LLOG in "${GEN_LAMBDA_LOGIT_LIST[@]}"; do
        s_tag=${SIG//./p}
        lm_tag=${LMASK//./p}
        lp_tag=${LPERC//./p}
        ll_tag=${LLOG//./p}
        RUN_NAME="B_sig${s_tag}_lm${lm_tag}_lp${lp_tag}_ll${ll_tag}"
        submit_job "$RUN_NAME" "$T_FIXED" "$A_FIXED" "$K_FIXED" "$B_FIXED" \
          "$SIG" "$LMASK" "$LPERC" "$LLOG"
      done
    done
  done
done

echo "Submitted ${run_count} jobs to Slurm."
