#!/bin/bash
set -euo pipefail

# SuperResStrong sweep (120 runs)
#
# Block A (60 runs): main grid on mix components + min_sigma + sample count + embed
#   GEN_MIX_COMPONENTS: 4, 8, 16, 32, 64
#   GEN_MIN_SIGMA: 1e-3, 1e-2, 1e-1
#   SR_SAMPLES: 4, 8
#   GEN_EMBED_DIM: 96, 128
#
# Block B (24 runs): depth/decoder/ff sweep
#   GEN_NUM_LAYERS: 4, 6, 8
#   GEN_DEC_LAYERS: 2, 3, 4
#   GEN_FF_DIM: 512, 768
#   (Fixed SR_SAMPLES=6, MIX=16, MIN_SIGMA=1e-2)
#
# Block C (18 runs): perceptual/logit loss sweep
#   GEN_LAMBDA_PERC: 0.1, 0.2, 0.4
#   GEN_LAMBDA_LOGIT: 0.05, 0.1, 0.2
#   GEN_MIN_SIGMA: 1e-3, 1e-2
#
# Block D (12 runs): high-component stress test
#   GEN_MIX_COMPONENTS: 64, 128, 256
#   GEN_MIN_SIGMA: 1e-2, 2e-1
#   SR_SAMPLES: 12, 16
#
# Block E (6 runs): NLL weight vs stats
#   GEN_LAMBDA_FEAT: 0.5, 1.0, 2.0
#   GEN_LAMBDA_STATS: 0.05, 0.2
#
# Total: 120 runs

SAVE_DIR=${SAVE_DIR:-"checkpoints/superres_strong_sweep"}
TRAIN_PATH=${TRAIN_PATH:-""}
N_TRAIN_JETS=${N_TRAIN_JETS:-100000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
TEACHER_CKPT=${TEACHER_CKPT:-""}
BASELINE_CKPT=${BASELINE_CKPT:-""}
SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS:-0}
SR_MODE=${SR_MODE:-"both"}

# Fixed defaults (can be overridden via environment)
SR_EPOCHS=${SR_EPOCHS:-100}
SR_LR=${SR_LR:-5e-4}
SR_MASK_THRESHOLD=${SR_MASK_THRESHOLD:-0.5}

GEN_NUM_HEADS=${GEN_NUM_HEADS:-8}
GEN_DROPOUT=${GEN_DROPOUT:-0.1}
GEN_LATENT_DIM=${GEN_LATENT_DIM:-32}
GEN_LR=${GEN_LR:-3e-4}
GEN_EPOCHS=${GEN_EPOCHS:-60}
GEN_PATIENCE=${GEN_PATIENCE:-20}
GEN_LAMBDA_FEAT=${GEN_LAMBDA_FEAT:-1.0}
GEN_LAMBDA_MASK=${GEN_LAMBDA_MASK:-0.5}
GEN_LAMBDA_MULT=${GEN_LAMBDA_MULT:-0.1}
GEN_LAMBDA_STATS=${GEN_LAMBDA_STATS:-0.1}
GEN_LAMBDA_PERC=${GEN_LAMBDA_PERC:-0.2}
GEN_LAMBDA_LOGIT=${GEN_LAMBDA_LOGIT:-0.1}

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

mkdir -p superres_strong_logs

run_count=0

submit_job() {
    local run_name="$1"
    local sr_samples="$2"
    local mix_components="$3"
    local min_sigma="$4"
    local gen_embed_dim="$5"
    local gen_num_layers="$6"
    local gen_dec_layers="$7"
    local gen_ff_dim="$8"
    local gen_lambda_feat="$9"
    local gen_lambda_mask="${10}"
    local gen_lambda_stats="${11}"
    local gen_lambda_perc="${12}"
    local gen_lambda_logit="${13}"

    local exports="ALL"
    exports+=",RUN_NAME=${run_name}"
    exports+=",SR_MODE=${SR_MODE}"
    exports+=",SR_SAMPLES=${sr_samples}"
    exports+=",SR_EVAL_SAMPLES=${sr_samples}"
    exports+=",SR_EPOCHS=${SR_EPOCHS}"
    exports+=",SR_LR=${SR_LR}"
    exports+=",SR_MASK_THRESHOLD=${SR_MASK_THRESHOLD}"
    exports+=",GEN_EMBED_DIM=${gen_embed_dim}"
    exports+=",GEN_NUM_HEADS=${GEN_NUM_HEADS}"
    exports+=",GEN_NUM_LAYERS=${gen_num_layers}"
    exports+=",GEN_DEC_LAYERS=${gen_dec_layers}"
    exports+=",GEN_FF_DIM=${gen_ff_dim}"
    exports+=",GEN_DROPOUT=${GEN_DROPOUT}"
    exports+=",GEN_LATENT_DIM=${GEN_LATENT_DIM}"
    exports+=",GEN_MIX_COMPONENTS=${mix_components}"
    exports+=",GEN_MIN_SIGMA=${min_sigma}"
    exports+=",GEN_LR=${GEN_LR}"
    exports+=",GEN_EPOCHS=${GEN_EPOCHS}"
    exports+=",GEN_PATIENCE=${GEN_PATIENCE}"
    exports+=",GEN_LAMBDA_FEAT=${gen_lambda_feat}"
    exports+=",GEN_LAMBDA_MASK=${gen_lambda_mask}"
    exports+=",GEN_LAMBDA_MULT=${GEN_LAMBDA_MULT}"
    exports+=",GEN_LAMBDA_STATS=${gen_lambda_stats}"
    exports+=",GEN_LAMBDA_PERC=${gen_lambda_perc}"
    exports+=",GEN_LAMBDA_LOGIT=${gen_lambda_logit}"
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

    sbatch --export="$exports" run_superres_strong_single.sh
    run_count=$((run_count + 1))
}

GEN_LAMBDA_MASK_BASE=0.5
GEN_LAMBDA_STATS_BASE=0.05
GEN_LAMBDA_FEAT_BASE=1.0
GEN_LAMBDA_PERC_BASE=0.2
GEN_LAMBDA_LOGIT_BASE=0.1

echo "Block A: main grid (60 runs)"
MIX_LIST_A=(64 32 16 8 4)
SIGMA_LIST_A=(1e-3 1e-2 1e-1)
SR_SAMPLES_LIST_A=(4 8)
EMBED_LIST_A=(96 128)

for M in "${MIX_LIST_A[@]}"; do
  for K in "${SR_SAMPLES_LIST_A[@]}"; do
    for S in "${SIGMA_LIST_A[@]}"; do
      for E in "${EMBED_LIST_A[@]}"; do
        s_tag=${S//./p}
        RUN_NAME="A_K${K}_M${M}_s${s_tag}_E${E}"
        submit_job "$RUN_NAME" "$K" "$M" "$S" "$E" 4 2 512 \
          "$GEN_LAMBDA_FEAT_BASE" "$GEN_LAMBDA_MASK_BASE" "$GEN_LAMBDA_STATS_BASE" \
          "$GEN_LAMBDA_PERC_BASE" "$GEN_LAMBDA_LOGIT_BASE"
      done
    done
  done
done

echo "Block B: depth/decoder/ff sweep (24 runs)"
LAYERS_LIST_B=(4 6 8)
DEC_LIST_B=(2 3 4)
FF_LIST_B=(512 768)

K_FIXED_B=6
M_FIXED_B=16
S_FIXED_B=1e-2
E_FIXED_B=128

for L in "${LAYERS_LIST_B[@]}"; do
  for D in "${DEC_LIST_B[@]}"; do
    for F in "${FF_LIST_B[@]}"; do
      RUN_NAME="B_L${L}_D${D}_F${F}"
      submit_job "$RUN_NAME" "$K_FIXED_B" "$M_FIXED_B" "$S_FIXED_B" "$E_FIXED_B" "$L" "$D" "$F" \
        "$GEN_LAMBDA_FEAT_BASE" "$GEN_LAMBDA_MASK_BASE" "$GEN_LAMBDA_STATS_BASE" \
        "$GEN_LAMBDA_PERC_BASE" "$GEN_LAMBDA_LOGIT_BASE"
    done
  done
done

echo "Block C: perceptual/logit loss sweep (18 runs)"
PERC_LIST_C=(0.1 0.2 0.4)
LOGIT_LIST_C=(0.05 0.1 0.2)
SIGMA_LIST_C=(1e-3 1e-2)

K_FIXED_C=6
M_FIXED_C=16
E_FIXED_C=128
L_FIXED_C=4
D_FIXED_C=2
F_FIXED_C=512

for P in "${PERC_LIST_C[@]}"; do
  for Lg in "${LOGIT_LIST_C[@]}"; do
    for S in "${SIGMA_LIST_C[@]}"; do
      p_tag=${P//./p}
      lg_tag=${Lg//./p}
      s_tag=${S//./p}
      RUN_NAME="C_lp${p_tag}_ll${lg_tag}_s${s_tag}"
      submit_job "$RUN_NAME" "$K_FIXED_C" "$M_FIXED_C" "$S" "$E_FIXED_C" "$L_FIXED_C" "$D_FIXED_C" "$F_FIXED_C" \
        "$GEN_LAMBDA_FEAT_BASE" "$GEN_LAMBDA_MASK_BASE" "$GEN_LAMBDA_STATS_BASE" \
        "$P" "$Lg"
    done
  done
done

echo "Block D: high-component stress test (12 runs)"
MIX_LIST_D=(64 128 256)
SIGMA_LIST_D=(1e-2 2e-1)
SR_SAMPLES_LIST_D=(12 16)

E_FIXED_D=128
L_FIXED_D=6
D_FIXED_D=3
F_FIXED_D=768

for K in "${SR_SAMPLES_LIST_D[@]}"; do
  for M in "${MIX_LIST_D[@]}"; do
    for S in "${SIGMA_LIST_D[@]}"; do
      s_tag=${S//./p}
      RUN_NAME="D_K${K}_M${M}_s${s_tag}"
      submit_job "$RUN_NAME" "$K" "$M" "$S" "$E_FIXED_D" "$L_FIXED_D" "$D_FIXED_D" "$F_FIXED_D" \
        "$GEN_LAMBDA_FEAT_BASE" "$GEN_LAMBDA_MASK_BASE" "$GEN_LAMBDA_STATS_BASE" \
        "$GEN_LAMBDA_PERC_BASE" "$GEN_LAMBDA_LOGIT_BASE"
    done
  done
done

echo "Block E: NLL weight vs stats (6 runs)"
FEAT_LIST_E=(0.5 1.0 2.0)
STATS_LIST_E=(0.05 0.2)

K_FIXED_E=6
M_FIXED_E=16
S_FIXED_E=1e-2
E_FIXED_E=128
L_FIXED_E=4
D_FIXED_E=2
F_FIXED_E=512

for LF in "${FEAT_LIST_E[@]}"; do
  for LS in "${STATS_LIST_E[@]}"; do
    lf_tag=${LF//./p}
    ls_tag=${LS//./p}
    RUN_NAME="E_lf${lf_tag}_ls${ls_tag}"
    submit_job "$RUN_NAME" "$K_FIXED_E" "$M_FIXED_E" "$S_FIXED_E" "$E_FIXED_E" "$L_FIXED_E" "$D_FIXED_E" "$F_FIXED_E" \
      "$LF" "$GEN_LAMBDA_MASK_BASE" "$LS" "$GEN_LAMBDA_PERC_BASE" "$GEN_LAMBDA_LOGIT_BASE"
  done
done

echo "Submitted ${run_count} SuperResStrong runs."
