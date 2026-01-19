#!/bin/bash

#SBATCH --job-name=fresh_kd_batch
#SBATCH --output=fresh_kd_logs/fresh_kd_batch_%j.out
#SBATCH --error=fresh_kd_logs/fresh_kd_batch_%j.err
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

mkdir -p fresh_kd_logs

if [ -z "${BATCH_FILE:-}" ]; then
    echo "BATCH_FILE not set."
    exit 1
fi

echo "=========================================="
echo "Fresh KD Batch Runner"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Batch File: $BATCH_FILE"
echo "=========================================="
echo ""

source ~/.bashrc
conda activate atlas_kd

cd $SLURM_SUBMIT_DIR

SUMMARY_FILE="fresh_kd_logs/batch_summary_${SLURM_JOB_ID}.txt"
echo "Batch summary ($BATCH_FILE)" > "$SUMMARY_FILE"
echo "Start: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

run_idx=0
while IFS= read -r line; do
    if [ -z "$line" ]; then
        continue
    fi
    run_idx=$((run_idx + 1))
    echo ""
    echo "=== Run ${run_idx} ==="
    echo "$line"

    IFS=',' read -r -a kv <<< "$line"
    for pair in "${kv[@]}"; do
        export "$pair"
    done

    bash run_fresh_kd_single.sh
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "FAIL: $RUN_NAME (exit $exit_code)" | tee -a "$SUMMARY_FILE"
        continue
    fi

    result_path="$SAVE_DIR/$RUN_NAME/results.npz"
    if [ -f "$result_path" ]; then
        export RESULT_PATH="$result_path"
        python - <<'PY' >> "$SUMMARY_FILE"
import os
import numpy as np

result_path = os.environ["RESULT_PATH"]
run_name = os.environ.get("RUN_NAME", "unknown")
data = np.load(result_path, allow_pickle=True)
auc_s = float(data.get("auc_student", np.nan))
auc_t = float(data.get("auc_teacher", np.nan))
auc_b = float(data.get("auc_baseline", np.nan))
br_s = float(data.get("br_student", np.nan))
print(f"{run_name} | auc_student={auc_s:.4f} | auc_teacher={auc_t:.4f} | auc_baseline={auc_b:.4f} | br_student={br_s:.2f}")
PY
    else
        echo "MISSING RESULTS: $RUN_NAME" | tee -a "$SUMMARY_FILE"
    fi
done < "$BATCH_FILE"

echo "" >> "$SUMMARY_FILE"
echo "End: $(date)" >> "$SUMMARY_FILE"

echo ""
echo "=========================================="
echo "Batch Complete"
echo "Summary: $SUMMARY_FILE"
echo "=========================================="

cat "$SUMMARY_FILE"
