#!/bin/bash
set -euo pipefail

# Queue 3 promising unsmear runs using run_unsmear_single.sh
# Each run sets env vars consumed by run_unsmear_single.sh

queue() {
  local name="$1"; shift
  echo "Submitting $name"
  sbatch --export=ALL,${*} run_unsmear_single.sh
}

# 1) Current best-ish baseline (v-pred, snr-weight, cross-attn, moderate guidance)
queue unsmear_v_snr_xattn \
  RUN_NAME=unsmear_v_snr_xattn \
  PRED_TYPE=v \
  SNR_WEIGHT=1 \
  SNR_GAMMA=5.0 \
  SELF_COND_PROB=0.5 \
  COND_DROP_PROB=0.1 \
  JET_LOSS_WEIGHT=0.1 \
  USE_CROSS_ATTN=1 \
  SAMPLING_METHOD=ddim \
  GUIDANCE_SCALE=1.5 \
  SAMPLE_STEPS=100 \
  N_SAMPLES_EVAL=2

# 2) Eps-pred with lower dropout and lighter guidance (often more stable)
queue unsmear_eps_lowdrop \
  RUN_NAME=unsmear_eps_lowdrop \
  PRED_TYPE=eps \
  SNR_WEIGHT=1 \
  SNR_GAMMA=5.0 \
  SELF_COND_PROB=0.5 \
  COND_DROP_PROB=0.0 \
  JET_LOSS_WEIGHT=0.1 \
  USE_CROSS_ATTN=1 \
  SAMPLING_METHOD=ddim \
  GUIDANCE_SCALE=1.0 \
  SAMPLE_STEPS=100 \
  N_SAMPLES_EVAL=2

# 3) x0-pred with stronger jet loss (encourages global consistency)
queue unsmear_x0_jetstrong \
  RUN_NAME=unsmear_x0_jetstrong \
  PRED_TYPE=x0 \
  SNR_WEIGHT=0 \
  SELF_COND_PROB=0.5 \
  COND_DROP_PROB=0.1 \
  JET_LOSS_WEIGHT=0.2 \
  USE_CROSS_ATTN=1 \
  SAMPLING_METHOD=ddim \
  GUIDANCE_SCALE=1.5 \
  SAMPLE_STEPS=100 \
  N_SAMPLES_EVAL=2

echo "Queued 3 unsmear runs."
