#!/bin/bash
set -euo pipefail

# SuperRes mixture sweep (120 runs)
#
# Block A (60 runs): main grid
#   GEN_MIX_COMPONENTS: 4, 8, 16, 32, 64
#   GEN_MIN_SIGMA: 1e-3, 5e-3, 1e-2
#   SR_SAMPLES: 4, 6, 8, 10
#
# Block B (20 runs): high-component stress test
#   GEN_MIX_COMPONENTS: 64, 128
#   GEN_MIN_SIGMA: 1e-3, 1e-2, 5e-2, 1e-1, 2e-1
#   SR_SAMPLES: 12, 16
#
# Block C (12 runs): generator regularization sweep
#   GEN_LAMBDA_MASK: 0.25, 0.5, 0.75
#   GEN_LAMBDA_STATS: 0.02, 0.05, 0.10, 0.20
#
# Block D (16 runs): NLL weight vs min_sigma
#   GEN_LAMBDA_FEAT: 0.5, 1.0, 2.0, 4.0
#   GEN_MIN_SIGMA: 1e-4, 1e-3, 1e-2, 1e-1
#
# Block E (12 runs): SR sample count sweep
#   SR_SAMPLES: 2, 4, 6, 8, 12, 16
#   GEN_MIX_COMPONENTS: 8, 32

SAVE_DIR=${SAVE_DIR:-"checkpoints/superres_mix_sweep"}
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
GEN_LAMBDA_MASK=${GEN_LAMBDA_MASK:-0.5}
GEN_LAMBDA_MULT=${GEN_LAMBDA_MULT:-0.1}
GEN_LAMBDA_STATS=${GEN_LAMBDA_STATS:-0.1}

LOSS_SUP=${LOSS_SUP:-1.0}
LOSS_CONS_PROB=${LOSS_CONS_PROB:-0.1}
LOSS_CONS_EMB=${LOSS_CONS_EMB:-0.05}

# Locked KD/consistency hyperparameters
CONF_POWER=${CONF_POWER:-2.0}
CONF_MIN=${CONF_MIN:-0.0}
TEMP_INIT=${TEMP_INIT:-7.0}
ALPHA_INIT=${ALPHA_INIT:-0.5}
ALPHA_ATTN=${ALPHA_ATTN:-0.05}
ALPHA_REP=${ALPHA_REP:-0.10}
ALPHA_NCE=${ALPHA_NCE:-0.10}
TAU_NCE=${TAU_NCE:-0.10}
CONF_WEIGHTED_KD=${CONF_WEIGHTED_KD:-1}

mkdir -p superres_mix_logs

run_count=0

submit_job() {
    local run_name="$1"
    local sr_samples="$2"
    local mix_components="$3"
    local min_sigma="$4"
    local gen_lambda_mask="$5"
    local gen_lambda_stats="$6"
    local gen_lambda_feat="$7"

    local exports="ALL"
    exports+=",RUN_NAME=${run_name}"
    exports+=",SR_MODE=${SR_MODE}"
    exports+=",SR_SAMPLES=${sr_samples}"
    exports+=",SR_EVAL_SAMPLES=${sr_samples}"
    exports+=",SR_EPOCHS=${SR_EPOCHS}"
    exports+=",SR_LR=${SR_LR}"
    exports+=",SR_MASK_THRESHOLD=${SR_MASK_THRESHOLD}"
    exports+=",GEN_EPOCHS=${GEN_EPOCHS}"
    exports+=",GEN_LR=${GEN_LR}"
    exports+=",GEN_MIX_COMPONENTS=${mix_components}"
    exports+=",GEN_MIN_SIGMA=${min_sigma}"
    exports+=",GEN_LAMBDA_FEAT=${gen_lambda_feat}"
    exports+=",GEN_LAMBDA_MASK=${gen_lambda_mask}"
    exports+=",GEN_LAMBDA_MULT=${GEN_LAMBDA_MULT}"
    exports+=",GEN_LAMBDA_STATS=${gen_lambda_stats}"
    exports+=",LOSS_SUP=${LOSS_SUP}"
    exports+=",LOSS_CONS_PROB=${LOSS_CONS_PROB}"
    exports+=",LOSS_CONS_EMB=${LOSS_CONS_EMB}"
    exports+=",CONF_POWER=${CONF_POWER}"
    exports+=",CONF_MIN=${CONF_MIN}"
    exports+=",TEMP_INIT=${TEMP_INIT}"
    exports+=",ALPHA_INIT=${ALPHA_INIT}"
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

    sbatch --export="$exports" run_superres_mix_single.sh
    run_count=$((run_count + 1))
}

GEN_LAMBDA_MASK_BASE=0.5
GEN_LAMBDA_STATS_BASE=0.05
GEN_LAMBDA_FEAT_BASE=1.0

echo "Block A: main grid (60 runs)"
MIX_LIST_A=(4 8 16 32 64)
SIGMA_LIST_A=(1e-3 5e-3 1e-2)
SR_SAMPLES_LIST_A=(4 6 8 10)

for K in "${SR_SAMPLES_LIST_A[@]}"; do
  for M in "${MIX_LIST_A[@]}"; do
    for S in "${SIGMA_LIST_A[@]}"; do
      s_tag=${S//./p}
      RUN_NAME="A_K${K}_M${M}_s${s_tag}"
      submit_job "$RUN_NAME" "$K" "$M" "$S" \
        "$GEN_LAMBDA_MASK_BASE" "$GEN_LAMBDA_STATS_BASE" "$GEN_LAMBDA_FEAT_BASE"
    done
  done
done

echo "Block B: high-component stress test (20 runs)"
MIX_LIST_B=(64 128)
SIGMA_LIST_B=(1e-3 1e-2 5e-2 1e-1 2e-1)
SR_SAMPLES_LIST_B=(12 16)

for K in "${SR_SAMPLES_LIST_B[@]}"; do
  for M in "${MIX_LIST_B[@]}"; do
    for S in "${SIGMA_LIST_B[@]}"; do
      s_tag=${S//./p}
      RUN_NAME="B_K${K}_M${M}_s${s_tag}"
      submit_job "$RUN_NAME" "$K" "$M" "$S" \
        "$GEN_LAMBDA_MASK_BASE" "$GEN_LAMBDA_STATS_BASE" "$GEN_LAMBDA_FEAT_BASE"
    done
  done
done

echo "Block C: generator regularization sweep (12 runs)"
GEN_LAMBDA_MASK_LIST_C=(0.25 0.5 0.75)
GEN_LAMBDA_STATS_LIST_C=(0.02 0.05 0.10 0.20)

K_FIXED_C=6
M_FIXED_C=16
S_FIXED_C=1e-2

for LM in "${GEN_LAMBDA_MASK_LIST_C[@]}"; do
  for LS in "${GEN_LAMBDA_STATS_LIST_C[@]}"; do
    lm_tag=${LM//./p}
    ls_tag=${LS//./p}
    RUN_NAME="C_lm${lm_tag}_ls${ls_tag}"
    submit_job "$RUN_NAME" "$K_FIXED_C" "$M_FIXED_C" "$S_FIXED_C" \
      "$LM" "$LS" "$GEN_LAMBDA_FEAT_BASE"
  done
done

echo "Block D: NLL weight vs min_sigma (16 runs)"
GEN_LAMBDA_FEAT_LIST_D=(0.5 1.0 2.0 4.0)
GEN_MIN_SIGMA_LIST_D=(1e-4 1e-3 1e-2 1e-1)

K_FIXED_D=6
M_FIXED_D=32

for LF in "${GEN_LAMBDA_FEAT_LIST_D[@]}"; do
  for S in "${GEN_MIN_SIGMA_LIST_D[@]}"; do
    lf_tag=${LF//./p}
    s_tag=${S//./p}
    RUN_NAME="D_lf${lf_tag}_s${s_tag}"
    submit_job "$RUN_NAME" "$K_FIXED_D" "$M_FIXED_D" "$S" \
      "$GEN_LAMBDA_MASK_BASE" "$GEN_LAMBDA_STATS_BASE" "$LF"
  done
done

echo "Block E: SR sample count sweep (12 runs)"
SR_SAMPLES_LIST_E=(2 4 6 8 12 16)
MIX_LIST_E=(8 32)

S_FIXED_E=1e-2

for K in "${SR_SAMPLES_LIST_E[@]}"; do
  for M in "${MIX_LIST_E[@]}"; do
    RUN_NAME="E_K${K}_M${M}"
    submit_job "$RUN_NAME" "$K" "$M" "$S_FIXED_E" \
      "$GEN_LAMBDA_MASK_BASE" "$GEN_LAMBDA_STATS_BASE" "$GEN_LAMBDA_FEAT_BASE"
  done
done

echo "Submitted ${run_count} SuperRes mixture runs."
