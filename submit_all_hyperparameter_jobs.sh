#!/bin/bash

# Submit all hyperparameter search jobs to SLURM
# Each configuration runs as a separate SLURM job

echo "=========================================="
echo "Submitting Hyperparameter Search Jobs"
echo "=========================================="
echo ""

# Create directories
mkdir -p transformer_logs
mkdir -p checkpoints/transformer_search

# Clear previous summary file
SUMMARY_FILE="checkpoints/transformer_search/hyperparameter_search_results.txt"
rm -f $SUMMARY_FILE
echo "Knowledge Distillation Hyperparameter Search Results" > $SUMMARY_FILE
echo "=====================================================" >> $SUMMARY_FILE
echo "Primary Metric: Background Rejection @ 50% signal efficiency" >> $SUMMARY_FILE
echo "Secondary Metric: AUC" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Define hyperparameter grid
TEMPERATURES=(3.0 5.0 7.0 10.0)
TEMP_ANNEAL_FINALS=(1.0 2.0)
ALPHAS=(0.3 0.5 0.7 0.9)
ALPHA_FINALS=(0.1 0.3)

# Counter for jobs
JOB_COUNT=0

echo "===== BASELINE RUNS: Constant Temperature and Alpha ====="
echo ""

# Baseline runs: Constant temperature and alpha
for TEMP in "${TEMPERATURES[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        RUN_NAME="baseline_T${TEMP}_A${ALPHA}"

        echo "Submitting: $RUN_NAME (T=$TEMP const, A=$ALPHA const)"

        sbatch --export=ALL,TEMP_INIT=$TEMP,ALPHA_INIT=$ALPHA,RUN_NAME=$RUN_NAME \
               --job-name="kd_${RUN_NAME}" \
               run_transformer_single.sh

        JOB_COUNT=$((JOB_COUNT + 1))
        sleep 0.5  # Small delay to avoid overwhelming the scheduler
    done
done

echo ""
echo "===== TEMPERATURE ANNEALING RUNS ====="
echo ""

# Temperature annealing runs: Fixed alpha, varying temperature schedule
for TEMP in "${TEMPERATURES[@]}"; do
    for TEMP_FINAL in "${TEMP_ANNEAL_FINALS[@]}"; do
        for ALPHA in "${ALPHAS[@]}"; do
            # Only run if temp_init > temp_final
            if (( $(echo "$TEMP > $TEMP_FINAL" | bc -l) )); then
                RUN_NAME="tempanneal_T${TEMP}to${TEMP_FINAL}_A${ALPHA}"

                echo "Submitting: $RUN_NAME (T: $TEMP->$TEMP_FINAL, A=$ALPHA const)"

                sbatch --export=ALL,TEMP_INIT=$TEMP,TEMP_FINAL=$TEMP_FINAL,ALPHA_INIT=$ALPHA,RUN_NAME=$RUN_NAME \
                       --job-name="kd_${RUN_NAME}" \
                       run_transformer_single.sh

                JOB_COUNT=$((JOB_COUNT + 1))
                sleep 0.5
            fi
        done
    done
done

echo ""
echo "===== ALPHA SCHEDULING RUNS ====="
echo ""

# Alpha scheduling runs: Fixed temperature, varying alpha schedule
for TEMP in "${TEMPERATURES[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        for ALPHA_FINAL in "${ALPHA_FINALS[@]}"; do
            # Only run if alpha_init > alpha_final
            if (( $(echo "$ALPHA > $ALPHA_FINAL" | bc -l) )); then
                RUN_NAME="alphasched_T${TEMP}_A${ALPHA}to${ALPHA_FINAL}"

                echo "Submitting: $RUN_NAME (T=$TEMP const, A: $ALPHA->$ALPHA_FINAL)"

                sbatch --export=ALL,TEMP_INIT=$TEMP,ALPHA_INIT=$ALPHA,ALPHA_FINAL=$ALPHA_FINAL,RUN_NAME=$RUN_NAME \
                       --job-name="kd_${RUN_NAME}" \
                       run_transformer_single.sh

                JOB_COUNT=$((JOB_COUNT + 1))
                sleep 0.5
            fi
        done
    done
done

echo ""
echo "===== COMBINED ANNEALING + SCHEDULING RUNS ====="
echo ""

# Combined annealing/scheduling runs: Both temperature and alpha vary
for TEMP in "${TEMPERATURES[@]}"; do
    for TEMP_FINAL in "${TEMP_ANNEAL_FINALS[@]}"; do
        for ALPHA in "${ALPHAS[@]}"; do
            for ALPHA_FINAL in "${ALPHA_FINALS[@]}"; do
                # Only run if both temp_init > temp_final AND alpha_init > alpha_final
                if (( $(echo "$TEMP > $TEMP_FINAL" | bc -l) )) && (( $(echo "$ALPHA > $ALPHA_FINAL" | bc -l) )); then
                    RUN_NAME="combined_T${TEMP}to${TEMP_FINAL}_A${ALPHA}to${ALPHA_FINAL}"

                    echo "Submitting: $RUN_NAME (T: $TEMP->$TEMP_FINAL, A: $ALPHA->$ALPHA_FINAL)"

                    sbatch --export=ALL,TEMP_INIT=$TEMP,TEMP_FINAL=$TEMP_FINAL,ALPHA_INIT=$ALPHA,ALPHA_FINAL=$ALPHA_FINAL,RUN_NAME=$RUN_NAME \
                           --job-name="kd_${RUN_NAME}" \
                           run_transformer_single.sh

                    JOB_COUNT=$((JOB_COUNT + 1))
                    sleep 0.5
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
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs in:"
echo "  transformer_logs/"
echo ""
echo "Results will be saved to:"
echo "  checkpoints/transformer_search/<run_name>/"
echo ""
echo "View summary after jobs complete:"
echo "  python analyze_hyperparameter_results.py"
echo ""
