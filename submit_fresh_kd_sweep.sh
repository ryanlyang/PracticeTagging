#!/bin/bash
set -euo pipefail

# fresh_KD sweep with ~500 runs across new KD options.
# Trains shared teacher/baseline once, then queues dependent runs.

MAX_RUNS=${MAX_RUNS:-500}
SAVE_DIR=${SAVE_DIR:-"checkpoints/fresh_kd_sweep"}
TRAIN_PATH=${TRAIN_PATH:-""}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS:-1}
GPU_GRES=${GPU_GRES:-"gpu:1"}

SHARED_DIR="checkpoints/fresh_kd/shared_models"
TEACHER_CKPT="$SHARED_DIR/teacher.pt"
BASELINE_CKPT="$SHARED_DIR/baseline.pt"

TEACHER_ENSEMBLE_CKPTS=${TEACHER_ENSEMBLE_CKPTS:-""}

mkdir -p fresh_kd_logs
mkdir -p "$SAVE_DIR"

submit_job() {
    local run_name="$1"
    shift
    local extra_env="$1"

    if [ "$run_count" -ge "$MAX_RUNS" ]; then
        return
    fi

    local exports="ALL"
    exports+=",RUN_NAME=${run_name}"
    exports+=",SAVE_DIR=${SAVE_DIR}"
    exports+=",SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS}"
    exports+=",TEACHER_CKPT=${TEACHER_CKPT}"
    exports+=",BASELINE_CKPT=${BASELINE_CKPT}"
    exports+=",N_TRAIN_JETS=${N_TRAIN_JETS}"
    exports+=",MAX_CONSTITS=${MAX_CONSTITS}"

    if [ -n "$TRAIN_PATH" ]; then
        exports+=",TRAIN_PATH=${TRAIN_PATH}"
    fi
    if [ -n "$TEACHER_ENSEMBLE_CKPTS" ]; then
        exports+=",TEACHER_ENSEMBLE_CKPTS=${TEACHER_ENSEMBLE_CKPTS}"
    fi
    if [ -n "$extra_env" ]; then
        exports+=",$extra_env"
    fi

    if [ -n "$DEP_JOB" ]; then
        sbatch --dependency=afterok:$DEP_JOB --gres="$GPU_GRES" --export="$exports" run_fresh_kd_single.sh
    else
        sbatch --gres="$GPU_GRES" --export="$exports" run_fresh_kd_single.sh
    fi
    run_count=$((run_count + 1))
}

echo "=========================================="
echo "fresh_KD Sweep"
echo "Target runs: $MAX_RUNS"
echo "Save dir: $SAVE_DIR"
echo "=========================================="

DEP_JOB=""
if [ ! -f "$TEACHER_CKPT" ] || [ ! -f "$BASELINE_CKPT" ]; then
    echo "Shared models not found. Submitting shared training job..."
    SHARED_JOB_ID=$(sbatch --parsable --gres="$GPU_GRES" train_fresh_kd_shared.sh)
    echo "Shared training job ID: $SHARED_JOB_ID"
    DEP_JOB="$SHARED_JOB_ID"
else
    echo "Found shared models:"
    echo "  $TEACHER_CKPT"
    echo "  $BASELINE_CKPT"
fi

run_count=0

# Base grid (no self-train): calibration, EMA, adaptive alpha, relational KD
if [ -n "$TEACHER_ENSEMBLE_CKPTS" ]; then
    ALPHA_REL_VALUES=(0.0 0.03 0.06 0.09 0.12)
else
    ALPHA_REL_VALUES=(0.0 0.02 0.04 0.06 0.08 0.10)
fi

CAL_FLAGS=(0 1)
EMA_DECAYS=(0.99 0.995 0.999)
ADAPT_WARMUPS=(0.0 0.1)
ADAPT_PATIENCE=(2 4)

for CAL in "${CAL_FLAGS[@]}"; do
  for EMA in 0 1; do
    for REL in "${ALPHA_REL_VALUES[@]}"; do
      # ADAPT off
      extra_env="CALIBRATE_TEACHER=$CAL,EMA_TEACHER=$EMA,ALPHA_REL=$REL,USE_ENSEMBLE=0"
      if [ "$EMA" -eq 1 ]; then
          for DECAY in "${EMA_DECAYS[@]}"; do
              extra_env="CALIBRATE_TEACHER=$CAL,EMA_TEACHER=1,EMA_DECAY=$DECAY,ALPHA_REL=$REL,USE_ENSEMBLE=0"
              RUN_NAME="A_cal${CAL}_ema${DECAY}_adapt0_rel${REL}"
              submit_job "$RUN_NAME" "$extra_env"
          done
      else
          RUN_NAME="A_cal${CAL}_ema0_adapt0_rel${REL}"
          submit_job "$RUN_NAME" "$extra_env"
      fi

      # ADAPT on (4 combos)
      for W in "${ADAPT_WARMUPS[@]}"; do
        for P in "${ADAPT_PATIENCE[@]}"; do
          if [ "$EMA" -eq 1 ]; then
            for DECAY in "${EMA_DECAYS[@]}"; do
              extra_env="CALIBRATE_TEACHER=$CAL,EMA_TEACHER=1,EMA_DECAY=$DECAY,ADAPTIVE_ALPHA=1,ALPHA_WARMUP=$W,ALPHA_STABLE_PATIENCE=$P,ALPHA_REL=$REL"
              RUN_NAME="A_cal${CAL}_ema${DECAY}_adapt1_w${W}_p${P}_rel${REL}"
              submit_job "$RUN_NAME" "$extra_env"
            done
          else
            extra_env="CALIBRATE_TEACHER=$CAL,EMA_TEACHER=0,ADAPTIVE_ALPHA=1,ALPHA_WARMUP=$W,ALPHA_STABLE_PATIENCE=$P,ALPHA_REL=$REL,USE_ENSEMBLE=0"
            RUN_NAME="A_cal${CAL}_ema0_adapt1_w${W}_p${P}_rel${REL}"
            submit_job "$RUN_NAME" "$extra_env"
          fi
        done
      done
    done
  done
done

# Self-train block (single-teacher only): 16 base combos * 16 self-train combos = 256
ST_REL_VALUES=(0.0 0.06)
ST_CAL_FLAGS=(0 1)
ST_EMA=(0 1)
ST_WARMUPS=(0.0 0.1)
ST_PATIENCE=(2 4)
ST_SOURCES=("teacher" "student")
ST_EPOCHS=(3 5)
ST_CONF_MIN=(0.0 0.2)
ST_CONF_POWER=(1.0 2.0)

for CAL in "${ST_CAL_FLAGS[@]}"; do
  for EMA in "${ST_EMA[@]}"; do
    for REL in "${ST_REL_VALUES[@]}"; do
      for W in "${ST_WARMUPS[@]}"; do
        for P in "${ST_PATIENCE[@]}"; do
          base_env="CALIBRATE_TEACHER=$CAL,EMA_TEACHER=$EMA,ADAPTIVE_ALPHA=1,ALPHA_WARMUP=$W,ALPHA_STABLE_PATIENCE=$P,ALPHA_REL=$REL,SELF_TRAIN=1,USE_ENSEMBLE=0"
          if [ "$EMA" -eq 1 ]; then
              base_env="$base_env,EMA_DECAY=0.995"
          fi
          for SRC in "${ST_SOURCES[@]}"; do
            for E in "${ST_EPOCHS[@]}"; do
              for CM in "${ST_CONF_MIN[@]}"; do
                for CP in "${ST_CONF_POWER[@]}"; do
                  RUN_NAME="B_cal${CAL}_ema${EMA}_st${SRC}_e${E}_cm${CM}_cp${CP}_rel${REL}"
                  extra_env="$base_env,SELF_TRAIN_SOURCE=$SRC,SELF_TRAIN_EPOCHS=$E,SELF_TRAIN_CONF_MIN=$CM,SELF_TRAIN_CONF_POWER=$CP"
                  submit_job "$RUN_NAME" "$extra_env"
                done
              done
            done
          done
        done
      done
    done
  done
done

# Optional ensemble block (only if TEACHER_ENSEMBLE_CKPTS is provided).
if [ -n "$TEACHER_ENSEMBLE_CKPTS" ]; then
  ENSEMBLE_REL_VALUES=(0.0 0.06 0.12)
  ENSEMBLE_ADAPT_WARMUPS=(0.0 0.1)
  ENSEMBLE_ADAPT_PATIENCE=(2 4)
  ENSEMBLE_CAL_FLAGS=(0 1)
  ENSEMBLE_EMA_DECAYS=(0.995 0.999)

  for CAL in "${ENSEMBLE_CAL_FLAGS[@]}"; do
    for REL in "${ENSEMBLE_REL_VALUES[@]}"; do
      # ADAPT off, EMA off
      extra_env="CALIBRATE_TEACHER=$CAL,EMA_TEACHER=0,ADAPTIVE_ALPHA=0,ALPHA_REL=$REL,USE_ENSEMBLE=1"
      RUN_NAME="E_cal${CAL}_ema0_adapt0_rel${REL}"
      submit_job "$RUN_NAME" "$extra_env"

      # ADAPT on, EMA off
      for W in "${ENSEMBLE_ADAPT_WARMUPS[@]}"; do
        for P in "${ENSEMBLE_ADAPT_PATIENCE[@]}"; do
          extra_env="CALIBRATE_TEACHER=$CAL,EMA_TEACHER=0,ADAPTIVE_ALPHA=1,ALPHA_WARMUP=$W,ALPHA_STABLE_PATIENCE=$P,ALPHA_REL=$REL,USE_ENSEMBLE=1"
          RUN_NAME="E_cal${CAL}_ema0_adapt1_w${W}_p${P}_rel${REL}"
          submit_job "$RUN_NAME" "$extra_env"
        done
      done

      # EMA on with two decays
      for DECAY in "${ENSEMBLE_EMA_DECAYS[@]}"; do
        extra_env="CALIBRATE_TEACHER=$CAL,EMA_TEACHER=1,EMA_DECAY=$DECAY,ADAPTIVE_ALPHA=0,ALPHA_REL=$REL,USE_ENSEMBLE=1"
        RUN_NAME="E_cal${CAL}_ema${DECAY}_adapt0_rel${REL}"
        submit_job "$RUN_NAME" "$extra_env"
      done
    done
  done
fi

echo ""
echo "Submitted $run_count fresh_KD runs."
