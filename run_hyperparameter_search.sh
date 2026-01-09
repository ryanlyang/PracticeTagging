#!/bin/bash

# Hyperparameter Search for Knowledge Distillation
# This script runs multiple configurations to find the best Background Rejection @ 50% signal efficiency

echo "Starting Hyperparameter Search for Knowledge Distillation"
echo "=========================================================="

# Create base directory for all search results
SEARCH_DIR="checkpoints/transformer_search"
mkdir -p $SEARCH_DIR

# Clear previous summary file
SUMMARY_FILE="$SEARCH_DIR/hyperparameter_search_results.txt"
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

# Baseline runs: Constant temperature and alpha
echo ""
echo "===== BASELINE RUNS: Constant Temperature and Alpha ====="
echo ""

for TEMP in "${TEMPERATURES[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        RUN_NAME="baseline_T${TEMP}_A${ALPHA}"
        echo "Running: $RUN_NAME (Constant T=$TEMP, Alpha=$ALPHA)"

        python transformer_runner.py \
            --save_dir "$SEARCH_DIR" \
            --run_name "$RUN_NAME" \
            --temp_init $TEMP \
            --alpha_init $ALPHA \
            --device cpu

        if [ $? -ne 0 ]; then
            echo "ERROR: $RUN_NAME failed!"
        else
            echo "COMPLETED: $RUN_NAME"
        fi
        echo ""
    done
done

# Temperature annealing runs: Fixed alpha, varying temperature schedule
echo ""
echo "===== TEMPERATURE ANNEALING RUNS ====="
echo ""

for TEMP in "${TEMPERATURES[@]}"; do
    for TEMP_FINAL in "${TEMP_ANNEAL_FINALS[@]}"; do
        for ALPHA in "${ALPHAS[@]}"; do
            # Only run if temp_init > temp_final
            if (( $(echo "$TEMP > $TEMP_FINAL" | bc -l) )); then
                RUN_NAME="tempanneal_T${TEMP}to${TEMP_FINAL}_A${ALPHA}"
                echo "Running: $RUN_NAME (T: $TEMP -> $TEMP_FINAL, Alpha: $ALPHA constant)"

                python transformer_runner.py \
                    --save_dir "$SEARCH_DIR" \
                    --run_name "$RUN_NAME" \
                    --temp_init $TEMP \
                    --temp_final $TEMP_FINAL \
                    --alpha_init $ALPHA \
                    --device cpu

                if [ $? -ne 0 ]; then
                    echo "ERROR: $RUN_NAME failed!"
                else
                    echo "COMPLETED: $RUN_NAME"
                fi
                echo ""
            fi
        done
    done
done

# Alpha scheduling runs: Fixed temperature, varying alpha schedule
echo ""
echo "===== ALPHA SCHEDULING RUNS ====="
echo ""

for TEMP in "${TEMPERATURES[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        for ALPHA_FINAL in "${ALPHA_FINALS[@]}"; do
            # Only run if alpha_init > alpha_final (start with more KD, shift to hard labels)
            if (( $(echo "$ALPHA > $ALPHA_FINAL" | bc -l) )); then
                RUN_NAME="alphasched_T${TEMP}_A${ALPHA}to${ALPHA_FINAL}"
                echo "Running: $RUN_NAME (T: $TEMP constant, Alpha: $ALPHA -> $ALPHA_FINAL)"

                python transformer_runner.py \
                    --save_dir "$SEARCH_DIR" \
                    --run_name "$RUN_NAME" \
                    --temp_init $TEMP \
                    --alpha_init $ALPHA \
                    --alpha_final $ALPHA_FINAL \
                    --device cpu

                if [ $? -ne 0 ]; then
                    echo "ERROR: $RUN_NAME failed!"
                else
                    echo "COMPLETED: $RUN_NAME"
                fi
                echo ""
            fi
        done
    done
done

# Combined annealing/scheduling runs: Both temperature and alpha vary
echo ""
echo "===== COMBINED ANNEALING + SCHEDULING RUNS ====="
echo ""

for TEMP in "${TEMPERATURES[@]}"; do
    for TEMP_FINAL in "${TEMP_ANNEAL_FINALS[@]}"; do
        for ALPHA in "${ALPHAS[@]}"; do
            for ALPHA_FINAL in "${ALPHA_FINALS[@]}"; do
                # Only run if both temp_init > temp_final AND alpha_init > alpha_final
                if (( $(echo "$TEMP > $TEMP_FINAL" | bc -l) )) && (( $(echo "$ALPHA > $ALPHA_FINAL" | bc -l) )); then
                    RUN_NAME="combined_T${TEMP}to${TEMP_FINAL}_A${ALPHA}to${ALPHA_FINAL}"
                    echo "Running: $RUN_NAME (T: $TEMP -> $TEMP_FINAL, Alpha: $ALPHA -> $ALPHA_FINAL)"

                    python transformer_runner.py \
                        --save_dir "$SEARCH_DIR" \
                        --run_name "$RUN_NAME" \
                        --temp_init $TEMP \
                        --temp_final $TEMP_FINAL \
                        --alpha_init $ALPHA \
                        --alpha_final $ALPHA_FINAL \
                        --device cpu

                    if [ $? -ne 0 ]; then
                        echo "ERROR: $RUN_NAME failed!"
                    else
                        echo "COMPLETED: $RUN_NAME"
                    fi
                    echo ""
                fi
            done
        done
    done
done

echo ""
echo "=========================================================="
echo "Hyperparameter Search Complete!"
echo "=========================================================="
echo ""
echo "All results saved to: $SEARCH_DIR"
echo "Summary file: $SUMMARY_FILE"
echo ""
echo "To view sorted results by Background Rejection:"
echo "  cat $SUMMARY_FILE"
echo ""
echo "Individual run results are in:"
echo "  $SEARCH_DIR/<run_name>/results.png"
echo "  $SEARCH_DIR/<run_name>/results.npz"
