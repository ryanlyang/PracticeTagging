#!/bin/bash
set -euo pipefail

# KnowledgeRes sweep (~30 runs)
# Focus on the inverse sampler hyperparameters.
#
# Grid:
#   KNOW_SAMPLES: 4, 8
#   INV_NOISE_SCALE: 0.7, 1.0, 1.3
#   EXTRA_COUNT_SCALE: 0.5, 1.0, 1.5
#   SPLIT_FRAC: 0.3, 0.5
#
# Total: 2 * 3 * 3 * 2 = 36
# We'll prune to 30 by skipping 6 high-noise/high-extra combos,
# then add a couple high-K stress runs (K=128).

SAVE_DIR=${SAVE_DIR:-"checkpoints/knowledge_res_sweep"}
TRAIN_PATH=${TRAIN_PATH:-""}
N_TRAIN_JETS=${N_TRAIN_JETS:-100000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS:-0}

# Only run the knowledge-res student
RUN_TEACHER=0
RUN_BASELINE=0
RUN_STUDENT=1
USE_KD=${USE_KD:-0}

# Defaults
SR_MODE=${SR_MODE:-"both"}

KNOW_EVAL_SAMPLES=${KNOW_EVAL_SAMPLES:-4}
HIGH_K_LIST=(${HIGH_K_LIST:-16 32 64 128})
HIGH_K_EVAL_SAMPLES=${HIGH_K_EVAL_SAMPLES:-}
SPLIT_RADIUS=${SPLIT_RADIUS:-0.01}
SPLIT_MIN_FRAC=${SPLIT_MIN_FRAC:-0.05}
SPLIT_MAX_FRAC=${SPLIT_MAX_FRAC:-0.3}
CONSERVE_PT=${CONSERVE_PT:-1}

# Locked KD/consistency hyperparameters (leave as defaults unless overridden)
CONF_POWER=${CONF_POWER:-2.0}
CONF_MIN=${CONF_MIN:-0.0}
TEMP_INIT=${TEMP_INIT:-7.0}
ALPHA_INIT=${ALPHA_INIT:-0.5}
ALPHA_ATTN=${ALPHA_ATTN:-0.05}
ALPHA_REP=${ALPHA_REP:-0.10}
ALPHA_NCE=${ALPHA_NCE:-0.10}
TAU_NCE=${TAU_NCE:-0.10}
CONF_WEIGHTED_KD=${CONF_WEIGHTED_KD:-1}

mkdir -p knowledge_res_logs

run_count=0

submit_job() {
    local run_name="$1"
    local know_samples="$2"
    local inv_noise="$3"
    local extra_scale="$4"
    local split_frac="$5"
    local know_eval_samples="${6:-$KNOW_EVAL_SAMPLES}"

    local exports="ALL"
    exports+=",RUN_NAME=${run_name}"
    exports+=",SAVE_DIR=${SAVE_DIR}"
    exports+=",RUN_TEACHER=${RUN_TEACHER}"
    exports+=",RUN_BASELINE=${RUN_BASELINE}"
    exports+=",RUN_STUDENT=${RUN_STUDENT}"
    exports+=",USE_KD=${USE_KD}"
    exports+=",KNOW_SAMPLES=${know_samples}"
    exports+=",KNOW_EVAL_SAMPLES=${know_eval_samples}"
    exports+=",INV_NOISE_SCALE=${inv_noise}"
    exports+=",EXTRA_COUNT_SCALE=${extra_scale}"
    exports+=",SPLIT_FRAC=${split_frac}"
    exports+=",SPLIT_RADIUS=${SPLIT_RADIUS}"
    exports+=",SPLIT_MIN_FRAC=${SPLIT_MIN_FRAC}"
    exports+=",SPLIT_MAX_FRAC=${SPLIT_MAX_FRAC}"
    exports+=",CONSERVE_PT=${CONSERVE_PT}"
    exports+=",CONF_POWER=${CONF_POWER}"
    exports+=",CONF_MIN=${CONF_MIN}"
    exports+=",TEMP_INIT=${TEMP_INIT}"
    exports+=",ALPHA_INIT=${ALPHA_INIT}"
    exports+=",ALPHA_ATTN=${ALPHA_ATTN}"
    exports+=",ALPHA_REP=${ALPHA_REP}"
    exports+=",ALPHA_NCE=${ALPHA_NCE}"
    exports+=",TAU_NCE=${TAU_NCE}"
    exports+=",CONF_WEIGHTED_KD=${CONF_WEIGHTED_KD}"
    exports+=",SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS}"
    exports+=",COMPARE_TEACHER_CKPT="
    exports+=",COMPARE_BASELINE_CKPT="

    if [ -n "$TRAIN_PATH" ]; then
        exports+=",TRAIN_PATH=${TRAIN_PATH}"
    fi
    if [ -n "$N_TRAIN_JETS" ]; then
        exports+=",N_TRAIN_JETS=${N_TRAIN_JETS}"
    fi
    if [ -n "$MAX_CONSTITS" ]; then
        exports+=",MAX_CONSTITS=${MAX_CONSTITS}"
    fi

    sbatch --export="$exports" run_knowledge_res_single.sh
    run_count=$((run_count + 1))
}

KNOW_SAMPLES_LIST=(4 8)
INV_NOISE_LIST=(0.7 1.0 1.3)
EXTRA_COUNT_LIST=(0.5 1.0 1.5)
SPLIT_FRAC_LIST=(0.3 0.5)

echo "Submitting KnowledgeRes sweep..."
for K in "${KNOW_SAMPLES_LIST[@]}"; do
  for N in "${INV_NOISE_LIST[@]}"; do
    for E in "${EXTRA_COUNT_LIST[@]}"; do
      for S in "${SPLIT_FRAC_LIST[@]}"; do
        # Skip 6 overly aggressive combos to land at ~30 runs:
        # (inv_noise=1.3, extra=1.5) across all splits (2) and K (2) => 4
        # plus (inv_noise=1.3, extra=1.0, K=8) across splits (2) => 2
        if [ "$N" = "1.3" ] && [ "$E" = "1.5" ]; then
            continue
        fi
        if [ "$N" = "1.3" ] && [ "$E" = "1.0" ] && [ "$K" = "8" ]; then
            continue
        fi

        n_tag=${N//./p}
        e_tag=${E//./p}
        s_tag=${S//./p}
        RUN_NAME="KR_K${K}_N${n_tag}_E${e_tag}_S${s_tag}"
        submit_job "$RUN_NAME" "$K" "$N" "$E" "$S"
      done
    done
  done
done

# Extra high-K runs to probe large-sample behavior without exploding the grid.
HIGH_K_CONFIGS=(
  "1.0 1.0 0.5"
  "0.7 0.5 0.3"
)
for K in "${HIGH_K_LIST[@]}"; do
  for cfg in "${HIGH_K_CONFIGS[@]}"; do
    read -r N E S <<< "$cfg"
    n_tag=${N//./p}
    e_tag=${E//./p}
    s_tag=${S//./p}
    RUN_NAME="KR_K${K}_N${n_tag}_E${e_tag}_S${s_tag}"
    eval_samples="${HIGH_K_EVAL_SAMPLES:-$K}"
    submit_job "$RUN_NAME" "$K" "$N" "$E" "$S" "$eval_samples"
  done
done

echo "Submitted ${run_count} KnowledgeRes runs."
