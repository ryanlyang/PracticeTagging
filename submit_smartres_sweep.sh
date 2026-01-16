#!/bin/bash
set -euo pipefail

# SmartRes flow sweep (50 runs)
#
# Block A (18 runs): base grid over SR samples + flow depth/width
#   SR_SAMPLES: 4, 6, 8
#   FLOW_LAYERS: 4, 6, 8
#   FLOW_HIDDEN: 96, 128
#
# Block B (12 runs): flow scale sweep
#   SR_SAMPLES: 6, 8
#   FLOW_LAYERS: 6, 8
#   FLOW_SCALE: 0.6, 0.8, 1.0
#
# Block C (9 runs): generator regularization sweep
#   GEN_LAMBDA_MASK: 0.25, 0.5, 0.75
#   GEN_LAMBDA_STATS: 0.02, 0.05, 0.10
#
# Block D (4 runs): high-compute extremes
#   SR_SAMPLES: 12, 16
#   FLOW_LAYERS: 10, 12
#   FLOW_HIDDEN: 256
#
# Block E (3 runs): flow NLL weight
#   GEN_LAMBDA_FEAT: 0.5, 1.0, 2.0
#
# Block F (4 runs): extreme flow scale
#   FLOW_SCALE: 0.4, 0.6, 1.0, 1.2

SAVE_DIR=${SAVE_DIR:-"checkpoints/smartres_sweep"}
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
GEN_LAMBDA_MULT=${GEN_LAMBDA_MULT:-0.1}

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

mkdir -p smartres_logs

run_count=0

submit_job() {
    local run_name="$1"
    local sr_samples="$2"
    local flow_layers="$3"
    local flow_hidden="$4"
    local flow_scale="$5"
    local gen_lambda_mask="$6"
    local gen_lambda_stats="$7"
    local gen_lambda_feat="$8"

    local exports="ALL"
    exports+=",RUN_NAME=${run_name}"
    exports+=",SR_MODE=${SR_MODE}"
    exports+=",SR_SAMPLES=${sr_samples}"
    exports+=",SR_EVAL_SAMPLES=${sr_samples}"
    exports+=",SR_EPOCHS=${SR_EPOCHS}"
    exports+=",SR_LR=${SR_LR}"
    exports+=",SR_MASK_THRESHOLD=${SR_MASK_THRESHOLD}"
    exports+=",GEN_FLOW_LAYERS=${flow_layers}"
    exports+=",GEN_FLOW_HIDDEN=${flow_hidden}"
    exports+=",GEN_FLOW_SCALE=${flow_scale}"
    exports+=",GEN_EPOCHS=${GEN_EPOCHS}"
    exports+=",GEN_LR=${GEN_LR}"
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

    sbatch --export="$exports" run_smartres_single.sh
    run_count=$((run_count + 1))
}

GEN_LAMBDA_MASK_BASE=0.5
GEN_LAMBDA_STATS_BASE=0.05
GEN_LAMBDA_FEAT_BASE=1.0

echo "Block A: base grid (18 runs)"
SR_SAMPLES_LIST_A=(4 6 8)
FLOW_LAYERS_LIST_A=(4 6 8)
FLOW_HIDDEN_LIST_A=(96 128)
FLOW_SCALE_A=0.8

for K in "${SR_SAMPLES_LIST_A[@]}"; do
  for L in "${FLOW_LAYERS_LIST_A[@]}"; do
    for H in "${FLOW_HIDDEN_LIST_A[@]}"; do
      RUN_NAME="A_K${K}_L${L}_H${H}"
      submit_job "$RUN_NAME" "$K" "$L" "$H" "$FLOW_SCALE_A" \
        "$GEN_LAMBDA_MASK_BASE" "$GEN_LAMBDA_STATS_BASE" "$GEN_LAMBDA_FEAT_BASE"
    done
  done
done

echo "Block B: flow scale sweep (12 runs)"
SR_SAMPLES_LIST_B=(6 8)
FLOW_LAYERS_LIST_B=(6 8)
FLOW_SCALE_LIST_B=(0.6 0.8 1.0)
FLOW_HIDDEN_B=128

for K in "${SR_SAMPLES_LIST_B[@]}"; do
  for L in "${FLOW_LAYERS_LIST_B[@]}"; do
    for S in "${FLOW_SCALE_LIST_B[@]}"; do
      s_tag=${S//./p}
      RUN_NAME="B_K${K}_L${L}_S${s_tag}"
      submit_job "$RUN_NAME" "$K" "$L" "$FLOW_HIDDEN_B" "$S" \
        "$GEN_LAMBDA_MASK_BASE" "$GEN_LAMBDA_STATS_BASE" "$GEN_LAMBDA_FEAT_BASE"
    done
  done
done

echo "Block C: generator regularization sweep (9 runs)"
GEN_LAMBDA_MASK_LIST_C=(0.25 0.5 0.75)
GEN_LAMBDA_STATS_LIST_C=(0.02 0.05 0.10)

K_FIXED_C=8
L_FIXED_C=8
H_FIXED_C=192
S_FIXED_C=0.8

for LM in "${GEN_LAMBDA_MASK_LIST_C[@]}"; do
  for LS in "${GEN_LAMBDA_STATS_LIST_C[@]}"; do
    lm_tag=${LM//./p}
    ls_tag=${LS//./p}
    RUN_NAME="C_lm${lm_tag}_ls${ls_tag}"
    submit_job "$RUN_NAME" "$K_FIXED_C" "$L_FIXED_C" "$H_FIXED_C" "$S_FIXED_C" \
      "$LM" "$LS" "$GEN_LAMBDA_FEAT_BASE"
  done
done

echo "Block D: high-compute extremes (4 runs)"
SR_SAMPLES_LIST_D=(12 16)
FLOW_LAYERS_LIST_D=(10 12)
FLOW_HIDDEN_D=256
FLOW_SCALE_D=1.0

for K in "${SR_SAMPLES_LIST_D[@]}"; do
  for L in "${FLOW_LAYERS_LIST_D[@]}"; do
    RUN_NAME="D_K${K}_L${L}_H256"
    submit_job "$RUN_NAME" "$K" "$L" "$FLOW_HIDDEN_D" "$FLOW_SCALE_D" \
      "$GEN_LAMBDA_MASK_BASE" "$GEN_LAMBDA_STATS_BASE" "$GEN_LAMBDA_FEAT_BASE"
  done
done

echo "Block E: flow NLL weight sweep (3 runs)"
GEN_LAMBDA_FEAT_LIST_E=(0.5 1.0 2.0)

K_FIXED_E=6
L_FIXED_E=6
H_FIXED_E=128
S_FIXED_E=0.8

for LF in "${GEN_LAMBDA_FEAT_LIST_E[@]}"; do
  lf_tag=${LF//./p}
  RUN_NAME="E_lf${lf_tag}"
  submit_job "$RUN_NAME" "$K_FIXED_E" "$L_FIXED_E" "$H_FIXED_E" "$S_FIXED_E" \
    "$GEN_LAMBDA_MASK_BASE" "$GEN_LAMBDA_STATS_BASE" "$LF"
done

echo "Block F: extreme flow scale (4 runs)"
FLOW_SCALE_LIST_F=(0.4 0.6 1.0 1.2)

K_FIXED_F=8
L_FIXED_F=8
H_FIXED_F=256

for S in "${FLOW_SCALE_LIST_F[@]}"; do
  s_tag=${S//./p}
  RUN_NAME="F_S${s_tag}"
  submit_job "$RUN_NAME" "$K_FIXED_F" "$L_FIXED_F" "$H_FIXED_F" "$S" \
    "$GEN_LAMBDA_MASK_BASE" "$GEN_LAMBDA_STATS_BASE" "$GEN_LAMBDA_FEAT_BASE"
done

echo "Submitted ${run_count} SmartRes runs."
