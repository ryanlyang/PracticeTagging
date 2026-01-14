#!/bin/bash
set -euo pipefail

# Sweep configuration (144 runs total)
N_HLT_VIEWS_LIST=(2 3 4)
LAMBDA_PROB_LIST=(0.5 1.0 2.0)
LAMBDA_EMB_LIST=(0.1 0.25 0.5)
CONF_POWER_LIST=(1.0 2.0)
TEMP_INIT_LIST=(5.0 7.0)
ALPHA_INIT_LIST=(0.3 0.5)

# Defaults for other knobs (override via environment if desired)
SAVE_DIR=${SAVE_DIR:-"checkpoints/transformer_twohlt_sweep"}
RAMPUP_FRAC=${RAMPUP_FRAC:-0.2}
CONF_MIN=${CONF_MIN:-0.0}
ALPHA_ATTN=${ALPHA_ATTN:-0.05}
ALPHA_REP=${ALPHA_REP:-0.10}
ALPHA_NCE=${ALPHA_NCE:-0.10}
TAU_NCE=${TAU_NCE:-0.10}
NO_CONF_KD=${NO_CONF_KD:-0}
HLT_SEED_BASE=${HLT_SEED_BASE:-123}
HLT_SEED_STEP=${HLT_SEED_STEP:-333}
TRAIN_PATH=${TRAIN_PATH:-""}
N_TRAIN_JETS=${N_TRAIN_JETS:-""}
MAX_CONSTITS=${MAX_CONSTITS:-""}
TEACHER_CKPT=${TEACHER_CKPT:-""}
SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS:-0}

mkdir -p transformer_logs

run_count=0

for n_views in "${N_HLT_VIEWS_LIST[@]}"; do
  for lp in "${LAMBDA_PROB_LIST[@]}"; do
    for le in "${LAMBDA_EMB_LIST[@]}"; do
      for cp in "${CONF_POWER_LIST[@]}"; do
        for t in "${TEMP_INIT_LIST[@]}"; do
          for a in "${ALPHA_INIT_LIST[@]}"; do
            lp_tag=${lp//./p}
            le_tag=${le//./p}
            cp_tag=${cp//./p}
            t_tag=${t//./p}
            a_tag=${a//./p}
            RUN_NAME="mvkd_v${n_views}_lp${lp_tag}_le${le_tag}_cp${cp_tag}_t${t_tag}_a${a_tag}"

            EXPORTS="ALL"
            EXPORTS="${EXPORTS},N_HLT_VIEWS=${n_views}"
            EXPORTS="${EXPORTS},LAMBDA_PROB=${lp}"
            EXPORTS="${EXPORTS},LAMBDA_EMB=${le}"
            EXPORTS="${EXPORTS},CONF_POWER=${cp}"
            EXPORTS="${EXPORTS},TEMP_INIT=${t}"
            EXPORTS="${EXPORTS},ALPHA_INIT=${a}"
            EXPORTS="${EXPORTS},RUN_NAME=${RUN_NAME}"
            EXPORTS="${EXPORTS},SAVE_DIR=${SAVE_DIR}"
            EXPORTS="${EXPORTS},RAMPUP_FRAC=${RAMPUP_FRAC}"
            EXPORTS="${EXPORTS},CONF_MIN=${CONF_MIN}"
            EXPORTS="${EXPORTS},ALPHA_ATTN=${ALPHA_ATTN}"
            EXPORTS="${EXPORTS},ALPHA_REP=${ALPHA_REP}"
            EXPORTS="${EXPORTS},ALPHA_NCE=${ALPHA_NCE}"
            EXPORTS="${EXPORTS},TAU_NCE=${TAU_NCE}"
            EXPORTS="${EXPORTS},NO_CONF_KD=${NO_CONF_KD}"
            EXPORTS="${EXPORTS},HLT_SEED_BASE=${HLT_SEED_BASE}"
            EXPORTS="${EXPORTS},HLT_SEED_STEP=${HLT_SEED_STEP}"
            EXPORTS="${EXPORTS},SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS}"

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

            sbatch --export="${EXPORTS}" run_more_const_teach_single.sh
            run_count=$((run_count + 1))
          done
        done
      done
    done
  done
done

echo "Submitted ${run_count} jobs to Slurm."
