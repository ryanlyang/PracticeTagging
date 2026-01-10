#!/bin/bash

# Focused Hyperparameter Search - Round 2
# Based on previous findings: High temp (10.0) and high alpha (0.9) perform best
# This search explores the high-performing region more thoroughly

echo "=========================================="
echo "Focused Hyperparameter Search - Round 2"
echo "=========================================="
echo ""
echo "Previous findings:"
echo "  - Temperature 10.0 consistently best (vs 3.0, 5.0, 7.0)"
echo "  - Alpha 0.9 consistently best (correlation r=0.619)"
echo "  - Combined scheduling achieved peak AUC (0.8242)"
echo ""
echo "New search strategy:"
echo "  - Focus on temperature range: 8.0 - 12.0"
echo "  - Focus on alpha range: 0.7 - 1.0"
echo "  - More granular annealing/scheduling targets"
echo "  - Target: ~120 configurations"
echo ""

# Check if shared models exist
TEACHER_MODEL="checkpoints/transformer_search/shared_models/teacher.pt"
BASELINE_MODEL="checkpoints/transformer_search/shared_models/baseline.pt"

if [ ! -f "$TEACHER_MODEL" ] || [ ! -f "$BASELINE_MODEL" ]; then
    echo "ERROR: Shared models not found!"
    echo ""
    echo "You must first train the shared teacher and baseline models:"
    echo "  sbatch train_shared_models.sh"
    echo ""
    echo "Wait for that job to complete, then run this script."
    echo ""
    echo "Expected files:"
    echo "  $TEACHER_MODEL"
    echo "  $BASELINE_MODEL"
    echo ""
    exit 1
fi

echo "Found shared models:"
echo "  Teacher:  $TEACHER_MODEL"
echo "  Baseline: $BASELINE_MODEL"
echo ""

# Create directories
mkdir -p transformer_logs
mkdir -p checkpoints/transformer_search

# Clear/append to summary file
SUMMARY_FILE="checkpoints/transformer_search/hyperparameter_search_results.txt"
echo "" >> $SUMMARY_FILE
echo "=======================================================" >> $SUMMARY_FILE
echo "FOCUSED HYPERPARAMETER SEARCH - ROUND 2" >> $SUMMARY_FILE
echo "Date: $(date)" >> $SUMMARY_FILE
echo "=======================================================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Counter for jobs
JOB_COUNT=0

# REFINED HYPERPARAMETER GRID based on previous results
# Temperature: Focus on high range (8.0-12.0) since 10.0 was best
TEMPERATURES=(8.0 9.0 10.0 11.0 12.0)

# Alpha: Focus on high range (0.7-1.0) since 0.9 was best
ALPHAS=(0.7 0.75 0.8 0.85 0.9 0.95)

# Temperature annealing targets: More granular
TEMP_ANNEAL_FINALS=(1.0 1.5 2.0 2.5 3.0)

# Alpha scheduling targets: More granular
ALPHA_FINALS=(0.1 0.2 0.3 0.4 0.5)

echo "===== BASELINE RUNS: Constant Temperature and Alpha ====="
echo "Grid: 5 temps × 6 alphas = 30 runs"
echo ""

# Baseline runs: Constant temperature and alpha
for TEMP in "${TEMPERATURES[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        RUN_NAME="r2_baseline_T${TEMP}_A${ALPHA}"

        echo "Submitting: $RUN_NAME (T=$TEMP const, A=$ALPHA const)"

        sbatch --export=ALL,TEMP_INIT=$TEMP,ALPHA_INIT=$ALPHA,RUN_NAME=$RUN_NAME \
               --job-name="kd_${RUN_NAME}" \
               run_transformer_single.sh

        JOB_COUNT=$((JOB_COUNT + 1))
        sleep 0.3
    done
done

echo ""
echo "===== TEMPERATURE ANNEALING RUNS ====="
echo "Expected: ~30 runs (only high-performing alpha values)"
echo ""

# Temperature annealing: Only use high alpha values (0.85, 0.9, 0.95)
HIGH_ALPHAS=(0.85 0.9 0.95)

for TEMP in "${TEMPERATURES[@]}"; do
    for TEMP_FINAL in "${TEMP_ANNEAL_FINALS[@]}"; do
        for ALPHA in "${HIGH_ALPHAS[@]}"; do
            # Only run if temp_init > temp_final
            if (( $(echo "$TEMP > $TEMP_FINAL" | bc -l) )); then
                RUN_NAME="r2_tempanneal_T${TEMP}to${TEMP_FINAL}_A${ALPHA}"

                echo "Submitting: $RUN_NAME (T: $TEMP->$TEMP_FINAL, A=$ALPHA const)"

                sbatch --export=ALL,TEMP_INIT=$TEMP,TEMP_FINAL=$TEMP_FINAL,ALPHA_INIT=$ALPHA,RUN_NAME=$RUN_NAME \
                       --job-name="kd_${RUN_NAME}" \
                       run_transformer_single.sh

                JOB_COUNT=$((JOB_COUNT + 1))
                sleep 0.3
            fi
        done
    done
done

echo ""
echo "===== ALPHA SCHEDULING RUNS ====="
echo "Expected: ~30 runs (focus on high temp, high alpha start)"
echo ""

# Alpha scheduling: Focus on high temperature and high starting alpha
HIGH_TEMPS=(10.0 11.0 12.0)
for TEMP in "${HIGH_TEMPS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        for ALPHA_FINAL in "${ALPHA_FINALS[@]}"; do
            # Only run if alpha_init > alpha_final
            if (( $(echo "$ALPHA > $ALPHA_FINAL" | bc -l) )); then
                RUN_NAME="r2_alphasched_T${TEMP}_A${ALPHA}to${ALPHA_FINAL}"

                echo "Submitting: $RUN_NAME (T=$TEMP const, A: $ALPHA->$ALPHA_FINAL)"

                sbatch --export=ALL,TEMP_INIT=$TEMP,ALPHA_INIT=$ALPHA,ALPHA_FINAL=$ALPHA_FINAL,RUN_NAME=$RUN_NAME \
                       --job-name="kd_${RUN_NAME}" \
                       run_transformer_single.sh

                JOB_COUNT=$((JOB_COUNT + 1))
                sleep 0.3
            fi
        done
    done
done

echo ""
echo "===== COMBINED ANNEALING + SCHEDULING RUNS ====="
echo "Expected: ~30 runs (best performing combinations)"
echo ""

# Combined: Only best temperature ranges and alpha ranges
BEST_TEMPS=(10.0 11.0 12.0)
BEST_ALPHAS=(0.85 0.9 0.95)
BEST_TEMP_FINALS=(1.0 2.0)
BEST_ALPHA_FINALS=(0.2 0.3)

for TEMP in "${BEST_TEMPS[@]}"; do
    for TEMP_FINAL in "${BEST_TEMP_FINALS[@]}"; do
        for ALPHA in "${BEST_ALPHAS[@]}"; do
            for ALPHA_FINAL in "${BEST_ALPHA_FINALS[@]}"; do
                # Only run if both temp_init > temp_final AND alpha_init > alpha_final
                if (( $(echo "$TEMP > $TEMP_FINAL" | bc -l) )) && (( $(echo "$ALPHA > $ALPHA_FINAL" | bc -l) )); then
                    RUN_NAME="r2_combined_T${TEMP}to${TEMP_FINAL}_A${ALPHA}to${ALPHA_FINAL}"

                    echo "Submitting: $RUN_NAME (T: $TEMP->$TEMP_FINAL, A: $ALPHA->$ALPHA_FINAL)"

                    sbatch --export=ALL,TEMP_INIT=$TEMP,TEMP_FINAL=$TEMP_FINAL,ALPHA_INIT=$ALPHA,ALPHA_FINAL=$ALPHA_FINAL,RUN_NAME=$RUN_NAME \
                           --job-name="kd_${RUN_NAME}" \
                           run_transformer_single.sh

                    JOB_COUNT=$((JOB_COUNT + 1))
                    sleep 0.3
                fi
            done
        done
    done
done

echo ""
echo "=========================================="
echo "Submission Complete!"
echo "=========================================="
echo "Total jobs submitted: $JOB_COUNT"
echo ""
echo "Search Strategy Summary:"
echo "  - Baseline (constant): ~30 runs"
echo "  - Temperature annealing: ~30 runs"
echo "  - Alpha scheduling: ~30 runs"
echo "  - Combined scheduling: ~30 runs"
echo "  - TOTAL: ~120 runs"
echo ""
echo "Key Differences from Round 1:"
echo "  1. Temperature range: 8.0-12.0 (was 3.0-10.0)"
echo "  2. Alpha range: 0.7-1.0 (was 0.3-0.9)"
echo "  3. More granular steps (0.5 temp, 0.05 alpha)"
echo "  4. More annealing/scheduling targets"
echo "  5. Focused on high-performing region"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs in:"
echo "  transformer_logs/"
echo ""
echo "Results will be saved to:"
echo "  checkpoints/transformer_search/r2_<strategy>_<config>/"
echo ""
echo "View summary after jobs complete:"
echo "  python analyze_hyperparameter_results.py"
echo ""
