#!/bin/bash

# Master Grid Search for new_teacher_student.py
# 70 total configurations exploring KD mechanics, alignment losses, and teacher trust
#
# Structure:
#   Block A: Core KD mechanics (36 jobs)
#   Block B: Representation vs contrastive dominance (24 jobs)
#   Block C: Teacher-trust gating sanity (10 jobs)
#
# All jobs wait for shared models to complete before starting

echo "=========================================="
echo "NEW TEACHER-STUDENT KD GRID SEARCH"
echo "=========================================="
echo ""
echo "Total configurations: 70"
echo "  Block A: Core KD mechanics (36)"
echo "  Block B: Rep vs contrastive (24)"
echo "  Block C: Teacher trust gating (10)"
echo ""
echo "Step 1: Submitting shared models training..."

# Submit shared models training
SHARED_JOB=$(sbatch --parsable train_new_ts_shared.sh)

if [ -z "$SHARED_JOB" ]; then
    echo "ERROR: Failed to submit shared models job"
    exit 1
fi

echo "✓ Shared models job submitted: $SHARED_JOB"
echo ""
echo "Step 2: Queuing all 70 hyperparameter jobs (will wait for job $SHARED_JOB)..."
echo ""

# Create directories
mkdir -p new_ts_logs
mkdir -p checkpoints/new_ts_search

# Create summary file
SUMMARY_FILE="checkpoints/new_ts_search/grid_search_results.txt"
echo "NEW TEACHER-STUDENT KD GRID SEARCH" > $SUMMARY_FILE
echo "=======================================================" >> $SUMMARY_FILE
echo "Date: $(date)" >> $SUMMARY_FILE
echo "Total configurations: 70" >> $SUMMARY_FILE
echo "Dependency: Waiting for shared models job $SHARED_JOB" >> $SUMMARY_FILE
echo "=======================================================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

JOB_COUNT=0

# ============================================================================
# BLOCK A: CORE KD MECHANICS (36 jobs)
# ============================================================================
echo "===== BLOCK A: CORE KD MECHANICS (36 jobs) ====="
echo ""

# Fixed alignment settings for Block A
ALPHA_REP_A=0.10
ALPHA_NCE_A=0.10
TAU_NCE_A=0.10
ALPHA_ATTN_A=0.05

# A1-A3: WITH alignment losses (18 runs)
echo "Block A1-A3: WITH alignment losses (18 runs)"

TEMP_SCHEDULES=("7.0:" "10.0:4.0" "6.0:2.0")
TEMP_NAMES=("T7const" "T10to4" "T6to2")

ALPHA_SCHEDULES=("0.20:0.05" "0.50:0.20" "0.70:0.30")
ALPHA_NAMES=("A0.20to0.05" "A0.50to0.20" "A0.70to0.30")

CONF_OPTIONS=("" "OFF")
CONF_NAMES=("confON" "confOFF")

for t_idx in 0 1 2; do
    IFS=':' read -r TEMP_INIT TEMP_FINAL <<< "${TEMP_SCHEDULES[$t_idx]}"
    T_NAME="${TEMP_NAMES[$t_idx]}"

    for a_idx in 0 1 2; do
        IFS=':' read -r ALPHA_INIT ALPHA_FINAL <<< "${ALPHA_SCHEDULES[$a_idx]}"
        A_NAME="${ALPHA_NAMES[$a_idx]}"

        for c_idx in 0 1; do
            NO_CONF_KD="${CONF_OPTIONS[$c_idx]}"
            C_NAME="${CONF_NAMES[$c_idx]}"

            RUN_NAME="A_${T_NAME}_${A_NAME}_${C_NAME}_repnce"

            echo "Queuing: $RUN_NAME"

            sbatch --dependency=afterok:$SHARED_JOB \
                   --export=ALL,RUN_NAME=$RUN_NAME,TEMP_INIT=$TEMP_INIT,TEMP_FINAL=$TEMP_FINAL,ALPHA_INIT=$ALPHA_INIT,ALPHA_FINAL=$ALPHA_FINAL,ALPHA_ATTN=$ALPHA_ATTN_A,ALPHA_REP=$ALPHA_REP_A,ALPHA_NCE=$ALPHA_NCE_A,TAU_NCE=$TAU_NCE_A,NO_CONF_KD=$NO_CONF_KD \
                   --job-name="kd_${RUN_NAME}" \
                   run_new_ts_single.sh > /dev/null

            JOB_COUNT=$((JOB_COUNT + 1))
        done
    done
done

echo "✓ Block A1-A3 complete: 18 jobs queued"
echo ""

# A4: WITHOUT alignment losses (18 runs - same grid but alpha_rep=0, alpha_nce=0)
echo "Block A4: WITHOUT alignment losses (18 runs)"

ALPHA_REP_A4=0.00
ALPHA_NCE_A4=0.00

for t_idx in 0 1 2; do
    IFS=':' read -r TEMP_INIT TEMP_FINAL <<< "${TEMP_SCHEDULES[$t_idx]}"
    T_NAME="${TEMP_NAMES[$t_idx]}"

    for a_idx in 0 1 2; do
        IFS=':' read -r ALPHA_INIT ALPHA_FINAL <<< "${ALPHA_SCHEDULES[$a_idx]}"
        A_NAME="${ALPHA_NAMES[$a_idx]}"

        for c_idx in 0 1; do
            NO_CONF_KD="${CONF_OPTIONS[$c_idx]}"
            C_NAME="${CONF_NAMES[$c_idx]}"

            RUN_NAME="A_${T_NAME}_${A_NAME}_${C_NAME}_noalign"

            echo "Queuing: $RUN_NAME"

            sbatch --dependency=afterok:$SHARED_JOB \
                   --export=ALL,RUN_NAME=$RUN_NAME,TEMP_INIT=$TEMP_INIT,TEMP_FINAL=$TEMP_FINAL,ALPHA_INIT=$ALPHA_INIT,ALPHA_FINAL=$ALPHA_FINAL,ALPHA_ATTN=$ALPHA_ATTN_A,ALPHA_REP=$ALPHA_REP_A4,ALPHA_NCE=$ALPHA_NCE_A4,TAU_NCE=$TAU_NCE_A,NO_CONF_KD=$NO_CONF_KD \
                   --job-name="kd_${RUN_NAME}" \
                   run_new_ts_single.sh > /dev/null

            JOB_COUNT=$((JOB_COUNT + 1))
        done
    done
done

echo "✓ Block A4 complete: 18 jobs queued"
echo "✓ BLOCK A TOTAL: 36 jobs"
echo ""

# ============================================================================
# BLOCK B: REPRESENTATION VS CONTRASTIVE DOMINANCE (24 jobs)
# ============================================================================
echo "===== BLOCK B: REPRESENTATION VS CONTRASTIVE (24 jobs) ====="
echo ""

# Fixed settings for Block B
TEMP_INIT_B=10.0
TEMP_FINAL_B=4.0
ALPHA_INIT_B=0.50
ALPHA_FINAL_B=0.20
ALPHA_ATTN_B=0.05
NO_CONF_KD_B=""  # Conf-weighted ON

# B1: Rep-only strength (4 runs)
echo "Block B1: Rep-only strength (4 runs)"

REP_STRENGTHS=(0.00 0.05 0.10 0.20)

for ALPHA_REP_B1 in "${REP_STRENGTHS[@]}"; do
    RUN_NAME="B1_reponly_${ALPHA_REP_B1}"

    echo "Queuing: $RUN_NAME"

    sbatch --dependency=afterok:$SHARED_JOB \
           --export=ALL,RUN_NAME=$RUN_NAME,TEMP_INIT=$TEMP_INIT_B,TEMP_FINAL=$TEMP_FINAL_B,ALPHA_INIT=$ALPHA_INIT_B,ALPHA_FINAL=$ALPHA_FINAL_B,ALPHA_ATTN=$ALPHA_ATTN_B,ALPHA_REP=$ALPHA_REP_B1,ALPHA_NCE=0.00,TAU_NCE=0.10,NO_CONF_KD=$NO_CONF_KD_B \
           --job-name="kd_${RUN_NAME}" \
           run_new_ts_single.sh > /dev/null

    JOB_COUNT=$((JOB_COUNT + 1))
done

echo "✓ Block B1 complete: 4 jobs queued"
echo ""

# B2: NCE-only strength (4 runs)
echo "Block B2: NCE-only strength (4 runs)"

NCE_STRENGTHS=(0.00 0.05 0.10 0.20)

for ALPHA_NCE_B2 in "${NCE_STRENGTHS[@]}"; do
    RUN_NAME="B2_nceonly_${ALPHA_NCE_B2}"

    echo "Queuing: $RUN_NAME"

    sbatch --dependency=afterok:$SHARED_JOB \
           --export=ALL,RUN_NAME=$RUN_NAME,TEMP_INIT=$TEMP_INIT_B,TEMP_FINAL=$TEMP_FINAL_B,ALPHA_INIT=$ALPHA_INIT_B,ALPHA_FINAL=$ALPHA_FINAL_B,ALPHA_ATTN=$ALPHA_ATTN_B,ALPHA_REP=0.00,ALPHA_NCE=$ALPHA_NCE_B2,TAU_NCE=0.10,NO_CONF_KD=$NO_CONF_KD_B \
           --job-name="kd_${RUN_NAME}" \
           run_new_ts_single.sh > /dev/null

    JOB_COUNT=$((JOB_COUNT + 1))
done

echo "✓ Block B2 complete: 4 jobs queued"
echo ""

# B3: Joint rep+NCE grid (9 + 3 = 12 runs)
echo "Block B3: Joint rep+NCE grid (12 runs)"

JOINT_REPS=(0.05 0.10 0.20)
JOINT_NCES=(0.05 0.10 0.20)

# 3x3 grid with tau=0.10
for ALPHA_REP_B3 in "${JOINT_REPS[@]}"; do
    for ALPHA_NCE_B3 in "${JOINT_NCES[@]}"; do
        RUN_NAME="B3_joint_rep${ALPHA_REP_B3}_nce${ALPHA_NCE_B3}_tau0.10"

        echo "Queuing: $RUN_NAME"

        sbatch --dependency=afterok:$SHARED_JOB \
               --export=ALL,RUN_NAME=$RUN_NAME,TEMP_INIT=$TEMP_INIT_B,TEMP_FINAL=$TEMP_FINAL_B,ALPHA_INIT=$ALPHA_INIT_B,ALPHA_FINAL=$ALPHA_FINAL_B,ALPHA_ATTN=$ALPHA_ATTN_B,ALPHA_REP=$ALPHA_REP_B3,ALPHA_NCE=$ALPHA_NCE_B3,TAU_NCE=0.10,NO_CONF_KD=$NO_CONF_KD_B \
               --job-name="kd_${RUN_NAME}" \
               run_new_ts_single.sh > /dev/null

        JOB_COUNT=$((JOB_COUNT + 1))
    done
done

# 3 tau sensitivity runs at rep=0.10, nce=0.10
TAU_VALS=(0.07 0.10 0.20)

for TAU_NCE_B3 in "${TAU_VALS[@]}"; do
    RUN_NAME="B3_tausens_rep0.10_nce0.10_tau${TAU_NCE_B3}"

    echo "Queuing: $RUN_NAME"

    sbatch --dependency=afterok:$SHARED_JOB \
           --export=ALL,RUN_NAME=$RUN_NAME,TEMP_INIT=$TEMP_INIT_B,TEMP_FINAL=$TEMP_FINAL_B,ALPHA_INIT=$ALPHA_INIT_B,ALPHA_FINAL=$ALPHA_FINAL_B,ALPHA_ATTN=$ALPHA_ATTN_B,ALPHA_REP=0.10,ALPHA_NCE=0.10,TAU_NCE=$TAU_NCE_B3,NO_CONF_KD=$NO_CONF_KD_B \
           --job-name="kd_${RUN_NAME}" \
           run_new_ts_single.sh > /dev/null

    JOB_COUNT=$((JOB_COUNT + 1))
done

echo "✓ Block B3 complete: 12 jobs queued"
echo ""

# B4: Attention KD weight sanity (4 runs)
echo "Block B4: Attention KD weight sanity (4 runs)"

ATTN_WEIGHTS=(0.00 0.02 0.05 0.10)

for ALPHA_ATTN_B4 in "${ATTN_WEIGHTS[@]}"; do
    RUN_NAME="B4_attn${ALPHA_ATTN_B4}_rep0.10_nce0.10"

    echo "Queuing: $RUN_NAME"

    sbatch --dependency=afterok:$SHARED_JOB \
           --export=ALL,RUN_NAME=$RUN_NAME,TEMP_INIT=$TEMP_INIT_B,TEMP_FINAL=$TEMP_FINAL_B,ALPHA_INIT=$ALPHA_INIT_B,ALPHA_FINAL=$ALPHA_FINAL_B,ALPHA_ATTN=$ALPHA_ATTN_B4,ALPHA_REP=0.10,ALPHA_NCE=0.10,TAU_NCE=0.10,NO_CONF_KD=$NO_CONF_KD_B \
           --job-name="kd_${RUN_NAME}" \
           run_new_ts_single.sh > /dev/null

    JOB_COUNT=$((JOB_COUNT + 1))
done

echo "✓ Block B4 complete: 4 jobs queued"
echo "✓ BLOCK B TOTAL: 24 jobs"
echo ""

# ============================================================================
# BLOCK C: TEACHER-TRUST GATING SANITY (10 jobs)
# ============================================================================
echo "===== BLOCK C: TEACHER-TRUST GATING (10 jobs) ====="
echo ""

# Fixed settings for Block C
TEMP_INIT_C=10.0
TEMP_FINAL_C=4.0
ALPHA_INIT_C=0.50
ALPHA_FINAL_C=0.20
ALPHA_ATTN_C=0.05

# 5 regimes × 2 (conf ON/OFF) = 10 runs
REGIMES=(
    "plainKD:0.00:0.00"
    "reponly:0.10:0.00"
    "nceonly:0.00:0.10"
    "balanced:0.10:0.10"
    "repheavy:0.20:0.10"
)

REGIME_NAMES=("plainKD" "reponly" "nceonly" "balanced" "repheavy")

for r_idx in 0 1 2 3 4; do
    IFS=':' read -r REGIME_NAME ALPHA_REP_C ALPHA_NCE_C <<< "${REGIMES[$r_idx]}"

    # Conf ON
    RUN_NAME="C_${REGIME_NAME}_confON"
    echo "Queuing: $RUN_NAME"

    sbatch --dependency=afterok:$SHARED_JOB \
           --export=ALL,RUN_NAME=$RUN_NAME,TEMP_INIT=$TEMP_INIT_C,TEMP_FINAL=$TEMP_FINAL_C,ALPHA_INIT=$ALPHA_INIT_C,ALPHA_FINAL=$ALPHA_FINAL_C,ALPHA_ATTN=$ALPHA_ATTN_C,ALPHA_REP=$ALPHA_REP_C,ALPHA_NCE=$ALPHA_NCE_C,TAU_NCE=0.10,NO_CONF_KD="" \
           --job-name="kd_${RUN_NAME}" \
           run_new_ts_single.sh > /dev/null

    JOB_COUNT=$((JOB_COUNT + 1))

    # Conf OFF
    RUN_NAME="C_${REGIME_NAME}_confOFF"
    echo "Queuing: $RUN_NAME"

    sbatch --dependency=afterok:$SHARED_JOB \
           --export=ALL,RUN_NAME=$RUN_NAME,TEMP_INIT=$TEMP_INIT_C,TEMP_FINAL=$TEMP_FINAL_C,ALPHA_INIT=$ALPHA_INIT_C,ALPHA_FINAL=$ALPHA_FINAL_C,ALPHA_ATTN=$ALPHA_ATTN_C,ALPHA_REP=$ALPHA_REP_C,ALPHA_NCE=$ALPHA_NCE_C,TAU_NCE=0.10,NO_CONF_KD="OFF" \
           --job-name="kd_${RUN_NAME}" \
           run_new_ts_single.sh > /dev/null

    JOB_COUNT=$((JOB_COUNT + 1))
done

echo "✓ BLOCK C TOTAL: 10 jobs"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "=========================================="
echo "SUBMISSION COMPLETE!"
echo "=========================================="
echo ""
echo "Job Summary:"
echo "  Shared models job: $SHARED_JOB (running now)"
echo "  Grid search jobs: $JOB_COUNT (waiting for job $SHARED_JOB)"
echo ""
echo "Breakdown:"
echo "  Block A (Core KD): 36 jobs"
echo "  Block B (Rep vs NCE): 24 jobs"
echo "  Block C (Conf gating): 10 jobs"
echo "  TOTAL: 70 jobs"
echo ""
echo "How it works:"
echo "  1. Job $SHARED_JOB is training teacher and baseline models NOW"
echo "  2. All 70 grid jobs are queued with dependency"
echo "  3. They will automatically start when job $SHARED_JOB completes"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER                     # All jobs"
echo "  squeue -u \$USER -t PD               # Pending jobs"
echo "  squeue -u \$USER -t R                # Running jobs"
echo "  tail -f new_ts_logs/train_shared_${SHARED_JOB}.out"
echo ""
echo "After completion:"
echo "  Logs: new_ts_logs/"
echo "  Results: checkpoints/new_ts_search/"
echo "  Summary: checkpoints/new_ts_search/grid_search_results.txt"
echo ""
