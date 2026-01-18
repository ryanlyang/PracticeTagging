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

# Strategy-first sweep: emphasize combinations of new options.
CAL_FLAGS=(0 1)
REL_VALUES=(0.0 0.04 0.08)
EMA_DECAYS_CORE=(0.99 0.995)
ADAPT_WARMUPS=(0.0 0.1)
ADAPT_PATIENCE=(2 4)

# Block A: Core strategy matrix (no self-train).
for CAL in "${CAL_FLAGS[@]}"; do
  for REL in "${REL_VALUES[@]}"; do
    # No EMA, no adaptive alpha
    extra_env="CALIBRATE_TEACHER=$CAL,EMA_TEACHER=0,ADAPTIVE_ALPHA=0,ALPHA_REL=$REL,USE_ENSEMBLE=0"
    RUN_NAME="A_cal${CAL}_ema0_adapt0_rel${REL}"
    submit_job "$RUN_NAME" "$extra_env"

    # No EMA, adaptive alpha on
    for W in "${ADAPT_WARMUPS[@]}"; do
      for P in "${ADAPT_PATIENCE[@]}"; do
        extra_env="CALIBRATE_TEACHER=$CAL,EMA_TEACHER=0,ADAPTIVE_ALPHA=1,ALPHA_WARMUP=$W,ALPHA_STABLE_PATIENCE=$P,ALPHA_REL=$REL,USE_ENSEMBLE=0"
        RUN_NAME="A_cal${CAL}_ema0_adapt1_w${W}_p${P}_rel${REL}"
        submit_job "$RUN_NAME" "$extra_env"
      done
    done

    # EMA on (best-performing decays), adaptive alpha off/on
    for DECAY in "${EMA_DECAYS_CORE[@]}"; do
      extra_env="CALIBRATE_TEACHER=$CAL,EMA_TEACHER=1,EMA_DECAY=$DECAY,ADAPTIVE_ALPHA=0,ALPHA_REL=$REL,USE_ENSEMBLE=0"
      RUN_NAME="A_cal${CAL}_ema${DECAY}_adapt0_rel${REL}"
      submit_job "$RUN_NAME" "$extra_env"

      for W in "${ADAPT_WARMUPS[@]}"; do
        for P in "${ADAPT_PATIENCE[@]}"; do
          extra_env="CALIBRATE_TEACHER=$CAL,EMA_TEACHER=1,EMA_DECAY=$DECAY,ADAPTIVE_ALPHA=1,ALPHA_WARMUP=$W,ALPHA_STABLE_PATIENCE=$P,ALPHA_REL=$REL,USE_ENSEMBLE=0"
          RUN_NAME="A_cal${CAL}_ema${DECAY}_adapt1_w${W}_p${P}_rel${REL}"
          submit_job "$RUN_NAME" "$extra_env"
        done
      done
    done
  done
done

# Block B: Self-train focus (keep base close to best EMA+adaptive).
ST_REL_VALUES=(0.0 0.04)
ST_CAL_FLAGS=(0 1)
ST_SOURCES=("teacher" "student")
ST_EPOCHS=(3 5)
ST_CONF_MIN=(0.0 0.2)
ST_CONF_POWER=(1.0 2.0)

for CAL in "${ST_CAL_FLAGS[@]}"; do
  for REL in "${ST_REL_VALUES[@]}"; do
    base_env="CALIBRATE_TEACHER=$CAL,EMA_TEACHER=1,EMA_DECAY=0.99,ADAPTIVE_ALPHA=1,ALPHA_WARMUP=0.0,ALPHA_STABLE_PATIENCE=2,ALPHA_REL=$REL,SELF_TRAIN=1,USE_ENSEMBLE=0"
    for SRC in "${ST_SOURCES[@]}"; do
      for E in "${ST_EPOCHS[@]}"; do
        for CM in "${ST_CONF_MIN[@]}"; do
          for CP in "${ST_CONF_POWER[@]}"; do
            RUN_NAME="B_cal${CAL}_st${SRC}_e${E}_cm${CM}_cp${CP}_rel${REL}"
            extra_env="$base_env,SELF_TRAIN_SOURCE=$SRC,SELF_TRAIN_EPOCHS=$E,SELF_TRAIN_CONF_MIN=$CM,SELF_TRAIN_CONF_POWER=$CP"
            submit_job "$RUN_NAME" "$extra_env"
          done
        done
      done
    done
  done
done

# Block C: Teacher calibration vs EMA decay sweep around the best config.
CAL_SWEEP_REL=(0.0)
EMA_SWEEP_DECAYS=(0.985 0.99 0.995)
for CAL in "${CAL_FLAGS[@]}"; do
  for REL in "${CAL_SWEEP_REL[@]}"; do
    for DECAY in "${EMA_SWEEP_DECAYS[@]}"; do
      extra_env="CALIBRATE_TEACHER=$CAL,EMA_TEACHER=1,EMA_DECAY=$DECAY,ADAPTIVE_ALPHA=1,ALPHA_WARMUP=0.0,ALPHA_STABLE_PATIENCE=2,ALPHA_REL=$REL,USE_ENSEMBLE=0"
      RUN_NAME="C_cal${CAL}_ema${DECAY}_rel${REL}"
      submit_job "$RUN_NAME" "$extra_env"
    done
  done
done

# Optional ensemble block (only if TEACHER_ENSEMBLE_CKPTS is provided).
if [ -n "$TEACHER_ENSEMBLE_CKPTS" ]; then
  ENSEMBLE_REL_VALUES=(0.0 0.06)
  for CAL in "${CAL_FLAGS[@]}"; do
    for REL in "${ENSEMBLE_REL_VALUES[@]}"; do
      extra_env="CALIBRATE_TEACHER=$CAL,EMA_TEACHER=0,ADAPTIVE_ALPHA=0,ALPHA_REL=$REL,USE_ENSEMBLE=1"
      RUN_NAME="E_cal${CAL}_adapt0_rel${REL}"
      submit_job "$RUN_NAME" "$extra_env"

      extra_env="CALIBRATE_TEACHER=$CAL,EMA_TEACHER=0,ADAPTIVE_ALPHA=1,ALPHA_WARMUP=0.0,ALPHA_STABLE_PATIENCE=2,ALPHA_REL=$REL,USE_ENSEMBLE=1"
      RUN_NAME="E_cal${CAL}_adapt1_rel${REL}"
      submit_job "$RUN_NAME" "$extra_env"
    done
  done
fi

echo ""
echo "Submitted $run_count fresh_KD runs."
