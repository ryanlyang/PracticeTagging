#!/bin/bash
set -euo pipefail

# fresh_KD sweep with ~500 runs across new KD options.
# Trains shared teacher/baseline once, then queues dependent runs.

MAX_RUNS=${MAX_RUNS:-500}
BATCH_SIZE=${BATCH_SIZE:-20}
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
BATCH_DIR=${BATCH_DIR:-"fresh_kd_batches"}
mkdir -p "$BATCH_DIR"
RUN_LIST="${BATCH_DIR}/fresh_kd_runlist_$(date +%s).txt"
> "$RUN_LIST"

submit_job() {
    local run_name="$1"
    shift
    local extra_env="$1"

    if [ "$run_count" -ge "$MAX_RUNS" ]; then
        return
    fi

    local base_env="RUN_NAME=${run_name}"
    base_env+=",SAVE_DIR=${SAVE_DIR}"
    base_env+=",SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS}"
    base_env+=",TEACHER_CKPT=${TEACHER_CKPT}"
    base_env+=",BASELINE_CKPT=${BASELINE_CKPT}"
    base_env+=",N_TRAIN_JETS=${N_TRAIN_JETS}"
    base_env+=",MAX_CONSTITS=${MAX_CONSTITS}"
    if [ -n "$TRAIN_PATH" ]; then
        base_env+=",TRAIN_PATH=${TRAIN_PATH}"
    fi

    # Explicit defaults to avoid env carryover between runs.
    base_env+=",TEMP_INIT=7.0,TEMP_FINAL=,ALPHA_INIT=0.5,ALPHA_FINAL="
    base_env+=",ALPHA_ATTN=0.05,ALPHA_REP=0.10,ALPHA_NCE=0.10,ALPHA_REL=0.0,TAU_NCE=0.10,NO_CONF_KD=0"
    base_env+=",CALIBRATE_TEACHER=0,TEACHER_CALIB_MAX_ITER=50,EMA_TEACHER=0,EMA_DECAY=0.999"
    base_env+=",ADAPTIVE_ALPHA=0,ALPHA_WARMUP=0.0,ALPHA_STABLE_PATIENCE=3,ALPHA_STABLE_DELTA=1e-4,ALPHA_WARMUP_MIN_EPOCHS=3"
    base_env+=",SELF_TRAIN=0,SELF_TRAIN_SOURCE=teacher,SELF_TRAIN_EPOCHS=5,SELF_TRAIN_LR=1e-4"
    base_env+=",SELF_TRAIN_CONF_MIN=0.0,SELF_TRAIN_CONF_POWER=1.0,SELF_TRAIN_HARD=0,SELF_TRAIN_PATIENCE=5"
    base_env+=",USE_ENSEMBLE=0"

    if [ -n "$extra_env" ]; then
        base_env+=",$extra_env"
    fi

    echo "$base_env" >> "$RUN_LIST"
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
echo "Queued $run_count fresh_KD runs into: $RUN_LIST"

split -l "$BATCH_SIZE" -d --additional-suffix=.list "$RUN_LIST" "$BATCH_DIR/batch_"

batch_count=0
for batch_file in "$BATCH_DIR"/batch_*.list; do
    if [ -z "${batch_file:-}" ]; then
        continue
    fi
    if [ -n "$DEP_JOB" ]; then
        TEACHER_ENSEMBLE_CKPTS="$TEACHER_ENSEMBLE_CKPTS" \
          sbatch --dependency=afterok:$DEP_JOB --gres="$GPU_GRES" \
          --export=ALL,BATCH_FILE="$batch_file" run_fresh_kd_batch.sh
    else
        TEACHER_ENSEMBLE_CKPTS="$TEACHER_ENSEMBLE_CKPTS" \
          sbatch --gres="$GPU_GRES" --export=ALL,BATCH_FILE="$batch_file" run_fresh_kd_batch.sh
    fi
    batch_count=$((batch_count + 1))
done

echo "Submitted $batch_count batch jobs (up to $BATCH_SIZE runs each)."
