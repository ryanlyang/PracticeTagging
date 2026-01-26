#!/bin/bash

# Master script to run complete hyperparameter search workflow
# 1. Train shared teacher and baseline models
# 2. Automatically queue all hyperparameter search jobs when shared models complete

echo "=========================================="
echo "FULL HYPERPARAMETER SEARCH WORKFLOW"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Submit shared models training job"
echo "  2. Wait for it to assign a job ID"
echo "  3. Submit all ~120 hyperparameter jobs with dependency"
echo "  4. Hyperparameter jobs will start automatically when shared models complete"
echo ""
echo "Starting workflow..."
echo ""

# Step 1: Submit shared models training
echo "Step 1: Submitting shared models training..."
SHARED_JOB=$(sbatch --parsable train_shared_models.sh)

if [ -z "$SHARED_JOB" ]; then
    echo "ERROR: Failed to submit shared models job"
    exit 1
fi

echo "✓ Shared models job submitted: $SHARED_JOB"
echo ""

# Step 2: Submit hyperparameter search with dependency
echo "Step 2: Submitting hyperparameter search (Round 2)..."
echo "These jobs will wait for job $SHARED_JOB to complete successfully"
echo ""

# Create separate directories for Round 2
mkdir -p transformer_logs_round2
mkdir -p checkpoints/transformer_search_round2

# Create new summary file for Round 2
SUMMARY_FILE="checkpoints/transformer_search_round2/hyperparameter_search_results.txt"
echo "FOCUSED HYPERPARAMETER SEARCH - ROUND 2" > $SUMMARY_FILE
echo "=======================================================" >> $SUMMARY_FILE
echo "Date: $(date)" >> $SUMMARY_FILE
echo "Dependency: Waiting for shared models job $SHARED_JOB" >> $SUMMARY_FILE
echo "=======================================================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Counter for jobs
JOB_COUNT=0

# REFINED HYPERPARAMETER GRID based on previous results
TEMPERATURES=(8.0 9.0 10.0 11.0 12.0)
ALPHAS=(0.7 0.75 0.8 0.85 0.9 0.95)
TEMP_ANNEAL_FINALS=(1.0 1.5 2.0 2.5 3.0)
ALPHA_FINALS=(0.1 0.2 0.3 0.4 0.5)

echo "===== BASELINE RUNS: Constant Temperature and Alpha ====="
echo "Grid: 5 temps × 6 alphas = 30 runs"
echo ""

# Baseline runs with dependency
for TEMP in "${TEMPERATURES[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        RUN_NAME="r2_baseline_T${TEMP}_A${ALPHA}"

        echo "Queuing: $RUN_NAME (will run after job $SHARED_JOB completes)"

        sbatch --dependency=afterok:$SHARED_JOB \
               --export=ALL,TEMP_INIT=$TEMP,ALPHA_INIT=$ALPHA,RUN_NAME=$RUN_NAME \
               --job-name="kd_${RUN_NAME}" \
               run_transformer_single_round2.sh > /dev/null

        JOB_COUNT=$((JOB_COUNT + 1))
    done
done

echo "✓ Queued $JOB_COUNT baseline jobs"
echo ""

echo "===== TEMPERATURE ANNEALING RUNS ====="
INITIAL_COUNT=$JOB_COUNT

HIGH_ALPHAS=(0.85 0.9 0.95)

for TEMP in "${TEMPERATURES[@]}"; do
    for TEMP_FINAL in "${TEMP_ANNEAL_FINALS[@]}"; do
        for ALPHA in "${HIGH_ALPHAS[@]}"; do
            if (( $(echo "$TEMP > $TEMP_FINAL" | bc -l) )); then
                RUN_NAME="r2_tempanneal_T${TEMP}to${TEMP_FINAL}_A${ALPHA}"

                sbatch --dependency=afterok:$SHARED_JOB \
                       --export=ALL,TEMP_INIT=$TEMP,TEMP_FINAL=$TEMP_FINAL,ALPHA_INIT=$ALPHA,RUN_NAME=$RUN_NAME \
                       --job-name="kd_${RUN_NAME}" \
                       run_transformer_single_round2.sh > /dev/null

                JOB_COUNT=$((JOB_COUNT + 1))
            fi
        done
    done
done

TEMP_JOBS=$((JOB_COUNT - INITIAL_COUNT))
echo "✓ Queued $TEMP_JOBS temperature annealing jobs"
echo ""

echo "===== ALPHA SCHEDULING RUNS ====="
INITIAL_COUNT=$JOB_COUNT

HIGH_TEMPS=(10.0 11.0 12.0)
for TEMP in "${HIGH_TEMPS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        for ALPHA_FINAL in "${ALPHA_FINALS[@]}"; do
            if (( $(echo "$ALPHA > $ALPHA_FINAL" | bc -l) )); then
                RUN_NAME="r2_alphasched_T${TEMP}_A${ALPHA}to${ALPHA_FINAL}"

                sbatch --dependency=afterok:$SHARED_JOB \
                       --export=ALL,TEMP_INIT=$TEMP,ALPHA_INIT=$ALPHA,ALPHA_FINAL=$ALPHA_FINAL,RUN_NAME=$RUN_NAME \
                       --job-name="kd_${RUN_NAME}" \
                       run_transformer_single_round2.sh > /dev/null

                JOB_COUNT=$((JOB_COUNT + 1))
            fi
        done
    done
done

ALPHA_JOBS=$((JOB_COUNT - INITIAL_COUNT))
echo "✓ Queued $ALPHA_JOBS alpha scheduling jobs"
echo ""

echo "===== COMBINED ANNEALING + SCHEDULING RUNS ====="
INITIAL_COUNT=$JOB_COUNT

BEST_TEMPS=(10.0 11.0 12.0)
BEST_ALPHAS=(0.85 0.9 0.95)
BEST_TEMP_FINALS=(1.0 2.0)
BEST_ALPHA_FINALS=(0.2 0.3)

for TEMP in "${BEST_TEMPS[@]}"; do
    for TEMP_FINAL in "${BEST_TEMP_FINALS[@]}"; do
        for ALPHA in "${BEST_ALPHAS[@]}"; do
            for ALPHA_FINAL in "${BEST_ALPHA_FINALS[@]}"; do
                if (( $(echo "$TEMP > $TEMP_FINAL" | bc -l) )) && (( $(echo "$ALPHA > $ALPHA_FINAL" | bc -l) )); then
                    RUN_NAME="r2_combined_T${TEMP}to${TEMP_FINAL}_A${ALPHA}to${ALPHA_FINAL}"

                    sbatch --dependency=afterok:$SHARED_JOB \
                           --export=ALL,TEMP_INIT=$TEMP,TEMP_FINAL=$TEMP_FINAL,ALPHA_INIT=$ALPHA,ALPHA_FINAL=$ALPHA_FINAL,RUN_NAME=$RUN_NAME \
                           --job-name="kd_${RUN_NAME}" \
                           run_transformer_single_round2.sh > /dev/null

                    JOB_COUNT=$((JOB_COUNT + 1))
                fi
            done
        done
    done
done

COMBINED_JOBS=$((JOB_COUNT - INITIAL_COUNT))
echo "✓ Queued $COMBINED_JOBS combined scheduling jobs"
echo ""

echo "=========================================="
echo "WORKFLOW SUBMISSION COMPLETE!"
echo "=========================================="
echo ""
echo "Job Summary:"
echo "  Shared models job: $SHARED_JOB (running now)"
echo "  Hyperparameter jobs: $JOB_COUNT (waiting for job $SHARED_JOB)"
echo ""
echo "Breakdown:"
echo "  - Baseline (constant): 30 jobs"
echo "  - Temperature annealing: $TEMP_JOBS jobs"
echo "  - Alpha scheduling: $ALPHA_JOBS jobs"
echo "  - Combined scheduling: $COMBINED_JOBS jobs"
echo ""
echo "How it works:"
echo "  1. Job $SHARED_JOB is training teacher and baseline models NOW"
echo "  2. All $JOB_COUNT hyperparameter jobs are queued with dependency"
echo "  3. They will automatically start when job $SHARED_JOB completes successfully"
echo "  4. If job $SHARED_JOB fails, dependent jobs will be cancelled"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER                     # See all your jobs"
echo "  squeue -u \$USER -t PD               # See pending (waiting) jobs"
echo "  squeue -u \$USER -t R                # See running jobs"
echo "  tail -f transformer_logs/train_shared_${SHARED_JOB}.out"
echo ""
echo "After completion:"
echo "  Logs: transformer_logs_round2/"
echo "  Results: checkpoints/transformer_search_round2/"
echo "  Analysis: python analyze_hyperparameter_results.py"
echo ""
