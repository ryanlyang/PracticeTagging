#!/bin/bash

# Master grid search script for extra_first_teach.py hyperparameter search
#
# Grid dimensions:
#   rampup_frac:     [0.1, 0.3]                    = 2 values
#   lambda_prob:     [0.25, 0.5, 1.0, 2.0]         = 4 values
#   lambda_emb:      [0.0, 0.1, 0.25, 0.5, 1.0]    = 5 values
#   attention_epoch: [0, 5, 20]                    = 3 values
#
# Total combinations: 2 × 4 × 5 × 3 = 120 jobs
#
# All jobs will use the same HLT seeds (123, 456) to ensure valid comparison
# and load pre-trained Teacher/Baseline/Union from shared_models/

set -e

echo "========================================================"
echo "EXTRA FIRST TEACH - HYPERPARAMETER GRID SEARCH"
echo "========================================================"
echo "Date: $(date)"
echo ""
echo "This script will:"
echo "  1. Submit a job to train shared models (Teacher, Baseline, Union)"
echo "  2. Submit 120 hyperparameter search jobs (all dependent on step 1)"
echo ""
echo "Grid dimensions:"
echo "  rampup_frac:     [0.1, 0.3]"
echo "  lambda_prob:     [0.25, 0.5, 1.0, 2.0]"
echo "  lambda_emb:      [0.0, 0.1, 0.25, 0.5, 1.0]"
echo "  attention_epoch: [0, 5, 20]"
echo ""
echo "Total: 2 × 4 × 5 × 3 = 120 jobs"
echo "========================================================"
echo ""

# Create directories
mkdir -p extra_logs
mkdir -p checkpoints/extra_search

# Step 1: Submit shared models training job
echo "Step 1: Submitting shared models training job..."
SHARED_JOB_OUTPUT=$(sbatch run_extra_shared.sh)
SHARED_JOB_ID=$(echo $SHARED_JOB_OUTPUT | awk '{print $NF}')

if [ -z "$SHARED_JOB_ID" ]; then
    echo "ERROR: Failed to submit shared models job"
    exit 1
fi

echo "  ✓ Shared models job submitted: Job ID $SHARED_JOB_ID"
echo ""

# Step 2: Submit all hyperparameter search jobs (dependent on shared job)
echo "Step 2: Submitting 120 hyperparameter search jobs..."
echo "  (All jobs will wait for shared models job $SHARED_JOB_ID to complete)"
echo ""

# Hyperparameter arrays
RAMPUP_FRACS=(0.1 0.3)
LAMBDA_PROBS=(0.25 0.5 1.0 2.0)
LAMBDA_EMBS=(0.0 0.1 0.25 0.5 1.0)
ATTENTION_EPOCHS=(0 5 20)

# Job counter
JOB_COUNT=0
SUBMITTED_JOBS=()

# Grid search loop
for RAMPUP in "${RAMPUP_FRACS[@]}"; do
    for LAMBDA_P in "${LAMBDA_PROBS[@]}"; do
        for LAMBDA_E in "${LAMBDA_EMBS[@]}"; do
            for ATT_EPOCH in "${ATTENTION_EPOCHS[@]}"; do
                # Create run name
                # Format: ramp{rampup}_lp{lambda_prob}_le{lambda_emb}_att{attention_epoch}
                RUN_NAME="ramp${RAMPUP}_lp${LAMBDA_P}_le${LAMBDA_E}_att${ATT_EPOCH}"

                # Submit job with dependency on shared models job
                JOB_OUTPUT=$(sbatch \
                    --dependency=afterok:$SHARED_JOB_ID \
                    --export=ALL,RUN_NAME=$RUN_NAME,RAMPUP_FRAC=$RAMPUP,LAMBDA_PROB=$LAMBDA_P,LAMBDA_EMB=$LAMBDA_E,ATTENTION_EPOCH=$ATT_EPOCH \
                    run_extra_single.sh)

                JOB_ID=$(echo $JOB_OUTPUT | awk '{print $NF}')
                SUBMITTED_JOBS+=($JOB_ID)
                JOB_COUNT=$((JOB_COUNT + 1))

                # Print progress every 10 jobs
                if [ $((JOB_COUNT % 10)) -eq 0 ]; then
                    echo "  Submitted $JOB_COUNT/120 jobs..."
                fi

                # Small delay to avoid overwhelming scheduler
                sleep 0.2
            done
        done
    done
done

echo ""
echo "========================================================"
echo "SUBMISSION COMPLETE"
echo "========================================================"
echo "Shared models job:       $SHARED_JOB_ID"
echo "Hyperparameter jobs:     $JOB_COUNT submitted"
echo "First job ID:            ${SUBMITTED_JOBS[0]}"
echo "Last job ID:             ${SUBMITTED_JOBS[-1]}"
echo ""
echo "All hyperparameter jobs will start after shared models job completes."
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check shared models progress:"
echo "  tail -f extra_logs/shared_${SHARED_JOB_ID}.out"
echo ""
echo "Check individual job logs:"
echo "  ls -lth extra_logs/job_*.out | head"
echo ""
echo "Results will be saved to:"
echo "  checkpoints/extra_search/<run_name>/"
echo ""
echo "Grid search parameters:"
echo "  rampup_frac ∈ {0.1, 0.3}"
echo "  lambda_prob ∈ {0.25, 0.5, 1.0, 2.0}"
echo "  lambda_emb ∈ {0.0, 0.1, 0.25, 0.5, 1.0}"
echo "  attention_epoch ∈ {0, 5, 20}"
echo ""
echo "HLT seeds (fixed for all runs):"
echo "  hlt_seed1 = 123"
echo "  hlt_seed2 = 456"
echo "========================================================"
