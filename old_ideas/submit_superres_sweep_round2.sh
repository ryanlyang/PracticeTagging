#!/bin/bash
set -euo pipefail

# SuperRes sweep round 2 (80 runs)
#
# Block A (24 runs): KD core around best region
#   SR_SAMPLES: 4, 6
#   KD_TEMP: 3.0, 3.5, 4.0
#   LOSS_KD: 0.6, 0.8
#   LOSS_EMB: 0.2, 0.3
#
# Block B (16 runs): high-KD / higher-temp edge cases
#   SR_SAMPLES: 4, 6
#   KD_TEMP: 3.5, 4.5
#   LOSS_KD: 0.9, 1.1
#   LOSS_EMB: 0.3, 0.5
#
# Block C (12 runs): generator regularization sweep
#   GEN_LAMBDA_MASK: 0.25, 0.5, 0.75, 1.0
#   GEN_LAMBDA_STATS: 0.02, 0.05, 0.10
#
# Block D (4 runs): sample count sweep
#   SR_SAMPLES: 2, 4, 6, 8
#
# Block E (24 runs): KD temp vs KD/emb balance
#   KD_TEMP: 2.5, 3.0, 3.5, 4.0
#   LOSS_KD: 0.5, 0.7, 0.9
#   LOSS_EMB: 0.1, 0.3

SAVE_DIR=${SAVE_DIR:-"checkpoints/superres_sweep_round2"}
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

GEN_LAMBDA_MASK_A=0.5
GEN_LAMBDA_STATS_A=0.05

echo "Block A: KD core around best region (24 runs)"
SR_SAMPLES_LIST_A=(4 6)
KD_TEMP_LIST_A=(3.0 3.5 4.0)
LOSS_KD_LIST_A=(0.6 0.8)
LOSS_EMB_LIST_A=(0.2 0.3)

for K in "${SR_SAMPLES_LIST_A[@]}"; do
  for T in "${KD_TEMP_LIST_A[@]}"; do
    for LK in "${LOSS_KD_LIST_A[@]}"; do
      for LE in "${LOSS_EMB_LIST_A[@]}"; do
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

echo "Block B: high-KD / higher-temp edge cases (16 runs)"
SR_SAMPLES_LIST_B=(4 6)
KD_TEMP_LIST_B=(3.5 4.5)
LOSS_KD_LIST_B=(0.9 1.1)
LOSS_EMB_LIST_B=(0.3 0.5)

for K in "${SR_SAMPLES_LIST_B[@]}"; do
  for T in "${KD_TEMP_LIST_B[@]}"; do
    for LK in "${LOSS_KD_LIST_B[@]}"; do
      for LE in "${LOSS_EMB_LIST_B[@]}"; do
        t_tag=${T//./p}
        lk_tag=${LK//./p}
        le_tag=${LE//./p}
        RUN_NAME="B_K${K}_T${t_tag}_lk${lk_tag}_le${le_tag}"
        submit_job "$RUN_NAME" "$K" "$T" "$LK" "$LE" \
          "$GEN_LAMBDA_MASK_A" "$GEN_LAMBDA_STATS_A"
      done
    done
  done
done

echo "Block C: generator regularization sweep (12 runs)"
GEN_LAMBDA_MASK_LIST_C=(0.25 0.5 0.75 1.0)
GEN_LAMBDA_STATS_LIST_C=(0.02 0.05 0.10)

K_FIXED_C=4
T_FIXED_C=3.5
LK_FIXED_C=0.7
LE_FIXED_C=0.3

for LM in "${GEN_LAMBDA_MASK_LIST_C[@]}"; do
  for LS in "${GEN_LAMBDA_STATS_LIST_C[@]}"; do
    lm_tag=${LM//./p}
    ls_tag=${LS//./p}
    RUN_NAME="C_lm${lm_tag}_ls${ls_tag}"
    submit_job "$RUN_NAME" "$K_FIXED_C" "$T_FIXED_C" "$LK_FIXED_C" "$LE_FIXED_C" \
      "$LM" "$LS"
  done
done

echo "Block D: sample count sweep (4 runs)"
SR_SAMPLES_LIST_D=(2 4 6 8)

K_FIXED_D=4
T_FIXED_D=3.5
LK_FIXED_D=0.7
LE_FIXED_D=0.3

for K in "${SR_SAMPLES_LIST_D[@]}"; do
  RUN_NAME="D_K${K}"
  submit_job "$RUN_NAME" "$K" "$T_FIXED_D" "$LK_FIXED_D" "$LE_FIXED_D" \
    "$GEN_LAMBDA_MASK_A" "$GEN_LAMBDA_STATS_A"
done

echo "Block E: KD temp vs KD/emb balance (24 runs)"
KD_TEMP_LIST_E=(2.5 3.0 3.5 4.0)
LOSS_KD_LIST_E=(0.5 0.7 0.9)
LOSS_EMB_LIST_E=(0.1 0.3)

K_FIXED_E=4

for T in "${KD_TEMP_LIST_E[@]}"; do
  for LK in "${LOSS_KD_LIST_E[@]}"; do
    for LE in "${LOSS_EMB_LIST_E[@]}"; do
      t_tag=${T//./p}
      lk_tag=${LK//./p}
      le_tag=${LE//./p}
      RUN_NAME="E_T${t_tag}_lk${lk_tag}_le${le_tag}"
      submit_job "$RUN_NAME" "$K_FIXED_E" "$T" "$LK" "$LE" \
        "$GEN_LAMBDA_MASK_A" "$GEN_LAMBDA_STATS_A"
    done
  done
done

echo "Submitted ${run_count} jobs to Slurm."
