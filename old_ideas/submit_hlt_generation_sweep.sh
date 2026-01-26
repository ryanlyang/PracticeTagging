#!/bin/bash
set -euo pipefail

# Sweep for generator hyperparameters in HLT_generation.py
# 60 runs = 3 (min_sigma) * 4 (lambda_mask) * 5 (lambda_stats)

GEN_MIN_SIGMA_LIST=(0.01 0.02 0.04)
GEN_LAMBDA_MASK_LIST=(0.5 1.0 2.0 3.0)
GEN_LAMBDA_STATS_LIST=(0.0 0.05 0.10 0.20 0.30)

# Fixed KD/consistency settings (from mvkd_v3_lp0p5_le0p5_cp2p0_t5p0_a0p5)
TEMP_INIT=${TEMP_INIT:-5.0}
TEMP_FINAL=${TEMP_FINAL:-""}
ALPHA_INIT=${ALPHA_INIT:-0.5}
ALPHA_FINAL=${ALPHA_FINAL:-""}
ALPHA_CONS_PROB=${ALPHA_CONS_PROB:-0.5}
ALPHA_CONS_EMB=${ALPHA_CONS_EMB:-0.5}
CONS_CONF_POWER=${CONS_CONF_POWER:-2.0}
CONS_CONF_MIN=${CONS_CONF_MIN:-0.0}
CONS_RAMPUP_FRAC=${CONS_RAMPUP_FRAC:-0.2}
GEN_VIEWS=${GEN_VIEWS:-3}
ALPHA_ATTN=${ALPHA_ATTN:-0.05}
ALPHA_REP=${ALPHA_REP:-0.10}
ALPHA_NCE=${ALPHA_NCE:-0.10}
TAU_NCE=${TAU_NCE:-0.10}

SAVE_DIR=${SAVE_DIR:-"checkpoints/hlt_generation_sweep"}
TRAIN_PATH=${TRAIN_PATH:-""}
N_TRAIN_JETS=${N_TRAIN_JETS:-100000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
TEACHER_CKPT=${TEACHER_CKPT:-""}
BASELINE_CKPT=${BASELINE_CKPT:-""}
GENERATOR_CKPT=${GENERATOR_CKPT:-""}
SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS:-0}
NO_CONF_KD=${NO_CONF_KD:-0}

mkdir -p hlt_gen_logs

run_count=0

for SIGMA in "${GEN_MIN_SIGMA_LIST[@]}"; do
  for LMASK in "${GEN_LAMBDA_MASK_LIST[@]}"; do
    for LSTAT in "${GEN_LAMBDA_STATS_LIST[@]}"; do
      sigma_tag=${SIGMA//./p}
      lmask_tag=${LMASK//./p}
      lstat_tag=${LSTAT//./p}
      RUN_NAME="gen_sigma${sigma_tag}_lmask${lmask_tag}_lstat${lstat_tag}"

      EXPORTS="ALL"
      EXPORTS="${EXPORTS},RUN_NAME=${RUN_NAME}"
      EXPORTS="${EXPORTS},TEMP_INIT=${TEMP_INIT}"
      EXPORTS="${EXPORTS},ALPHA_INIT=${ALPHA_INIT}"
      EXPORTS="${EXPORTS},ALPHA_CONS_PROB=${ALPHA_CONS_PROB}"
      EXPORTS="${EXPORTS},ALPHA_CONS_EMB=${ALPHA_CONS_EMB}"
      EXPORTS="${EXPORTS},CONS_CONF_POWER=${CONS_CONF_POWER}"
      EXPORTS="${EXPORTS},CONS_CONF_MIN=${CONS_CONF_MIN}"
      EXPORTS="${EXPORTS},CONS_RAMPUP_FRAC=${CONS_RAMPUP_FRAC}"
      EXPORTS="${EXPORTS},GEN_VIEWS=${GEN_VIEWS}"
      EXPORTS="${EXPORTS},ALPHA_ATTN=${ALPHA_ATTN}"
      EXPORTS="${EXPORTS},ALPHA_REP=${ALPHA_REP}"
      EXPORTS="${EXPORTS},ALPHA_NCE=${ALPHA_NCE}"
      EXPORTS="${EXPORTS},TAU_NCE=${TAU_NCE}"
      EXPORTS="${EXPORTS},GEN_MIN_SIGMA=${SIGMA}"
      EXPORTS="${EXPORTS},GEN_LAMBDA_MASK=${LMASK}"
      EXPORTS="${EXPORTS},GEN_LAMBDA_STATS=${LSTAT}"
      EXPORTS="${EXPORTS},SAVE_DIR=${SAVE_DIR}"
      EXPORTS="${EXPORTS},SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS}"
      EXPORTS="${EXPORTS},NO_CONF_KD=${NO_CONF_KD}"

      if [ -n "$TEMP_FINAL" ]; then
        EXPORTS="${EXPORTS},TEMP_FINAL=${TEMP_FINAL}"
      fi
      if [ -n "$ALPHA_FINAL" ]; then
        EXPORTS="${EXPORTS},ALPHA_FINAL=${ALPHA_FINAL}"
      fi
      if [ -n "$TRAIN_PATH" ]; then
        EXPORTS="${EXPORTS},TRAIN_PATH=${TRAIN_PATH}"
      fi
      if [ -n "$N_TRAIN_JETS" ]; then
        EXPORTS="${EXPORTS},N_TRAIN_JETS=${N_TRAIN_JETS}"
      fi
      if [ -n "$MAX_CONSTITS" ]; then
        EXPORTS="${EXPORTS},MAX_CONSTITS=${MAX_CONSTITS}"
      fi
      if [ -n "$TEACHER_CKPT" ]; then
        EXPORTS="${EXPORTS},TEACHER_CKPT=${TEACHER_CKPT}"
      fi
      if [ -n "$BASELINE_CKPT" ]; then
        EXPORTS="${EXPORTS},BASELINE_CKPT=${BASELINE_CKPT}"
      fi
      if [ -n "$GENERATOR_CKPT" ]; then
        EXPORTS="${EXPORTS},GENERATOR_CKPT=${GENERATOR_CKPT}"
      fi

      sbatch --export="${EXPORTS}" run_hlt_generation_single.sh
      run_count=$((run_count + 1))
    done
  done
done

echo "Submitted ${run_count} jobs to Slurm."
