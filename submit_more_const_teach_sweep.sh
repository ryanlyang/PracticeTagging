#!/bin/bash
set -euo pipefail

# Multi-view KD sweep for more_const_teach.py
# Default (new_setup + views 2/3/4) submits ~240 runs:
#   80 runs per view (Blocks A-D) × 3 views = 240
# Modes:
#   SWEEP_MODE=new_setup  -> block-based grid (similar to run_new_ts_grid_search.sh)
#   SWEEP_MODE=round2     -> focused KD grid (similar to transformer Round 2)
#   SWEEP_MODE=both       -> run both sets

SWEEP_MODE=${SWEEP_MODE:-"new_setup"}
N_HLT_VIEWS_LIST_STR=${N_HLT_VIEWS_LIST_STR:-"2 3 4"}
read -r -a N_HLT_VIEWS_LIST <<< "$N_HLT_VIEWS_LIST_STR"

# Fixed consistency knobs (override via environment)
LAMBDA_PROB=${LAMBDA_PROB:-1.0}
LAMBDA_EMB=${LAMBDA_EMB:-0.25}
RAMPUP_FRAC=${RAMPUP_FRAC:-0.2}
CONF_POWER=${CONF_POWER:-1.0}
CONF_MIN=${CONF_MIN:-0.0}

# Shared defaults and paths
SAVE_DIR=${SAVE_DIR:-"checkpoints/transformer_twohlt_sweep"}
HLT_SEED_BASE=${HLT_SEED_BASE:-123}
HLT_SEED_STEP=${HLT_SEED_STEP:-333}
TRAIN_PATH=${TRAIN_PATH:-""}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-""}
TEACHER_CKPT=${TEACHER_CKPT:-""}
SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS:-0}
DEPENDENCY_JOB=${DEPENDENCY_JOB:-""}

mkdir -p transformer_logs

case "$SWEEP_MODE" in
    new_setup|round2|both) ;;
    *)
        echo "ERROR: SWEEP_MODE must be new_setup, round2, or both"
        exit 1
        ;;
esac

run_count=0

tag_float() {
    echo "$1" | sed "s/\\./p/g"
}

submit_job() {
    local run_name="$1"
    local n_views="$2"
    local temp_init="$3"
    local temp_final="$4"
    local alpha_init="$5"
    local alpha_final="$6"
    local alpha_attn="$7"
    local alpha_rep="$8"
    local alpha_nce="$9"
    local tau_nce="${10}"
    local no_conf_kd="${11}"
    local lambda_prob_override="${12:-}"
    local lambda_emb_override="${13:-}"
    local conf_power_override="${14:-}"
    local conf_min_override="${15:-}"

    local lp="${lambda_prob_override:-$LAMBDA_PROB}"
    local le="${lambda_emb_override:-$LAMBDA_EMB}"
    local cp="${conf_power_override:-$CONF_POWER}"
    local cm="${conf_min_override:-$CONF_MIN}"

    local exports="ALL"
    exports+=",RUN_NAME=${run_name}"
    exports+=",N_HLT_VIEWS=${n_views}"
    exports+=",LAMBDA_PROB=${lp}"
    exports+=",LAMBDA_EMB=${le}"
    exports+=",RAMPUP_FRAC=${RAMPUP_FRAC}"
    exports+=",CONF_POWER=${cp}"
    exports+=",CONF_MIN=${cm}"
    exports+=",TEMP_INIT=${temp_init}"
    exports+=",TEMP_FINAL=${temp_final}"
    exports+=",ALPHA_INIT=${alpha_init}"
    exports+=",ALPHA_FINAL=${alpha_final}"
    exports+=",ALPHA_ATTN=${alpha_attn}"
    exports+=",ALPHA_REP=${alpha_rep}"
    exports+=",ALPHA_NCE=${alpha_nce}"
    exports+=",TAU_NCE=${tau_nce}"
    exports+=",NO_CONF_KD=${no_conf_kd}"
    exports+=",SAVE_DIR=${SAVE_DIR}"
    exports+=",HLT_SEED_BASE=${HLT_SEED_BASE}"
    exports+=",HLT_SEED_STEP=${HLT_SEED_STEP}"
    exports+=",TRAIN_PATH=${TRAIN_PATH}"
    exports+=",N_TRAIN_JETS=${N_TRAIN_JETS}"
    exports+=",MAX_CONSTITS=${MAX_CONSTITS}"
    exports+=",TEACHER_CKPT=${TEACHER_CKPT}"
    exports+=",SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS}"

    local sbatch_opts=()
    if [ -n "$DEPENDENCY_JOB" ]; then
        sbatch_opts+=(--dependency="afterok:${DEPENDENCY_JOB}")
    fi

    sbatch "${sbatch_opts[@]}" --export="$exports" run_more_const_teach_single.sh > /dev/null
    run_count=$((run_count + 1))
}

if [ "$SWEEP_MODE" = "new_setup" ] || [ "$SWEEP_MODE" = "both" ]; then
    echo "===== NEW SETUP GRID (block-based) ====="
    echo "Views: ${N_HLT_VIEWS_LIST[*]}"
    echo ""

    # ----------------------------------------------------------------------
    # BLOCK A: Core KD mechanics (36 runs per view)
    # ----------------------------------------------------------------------
    TEMP_SCHEDULES=("7.0:" "10.0:4.0" "6.0:2.0")
    TEMP_NAMES=("T7const" "T10to4" "T6to2")

    ALPHA_SCHEDULES=("0.20:0.05" "0.50:0.20" "0.70:0.30")
    ALPHA_NAMES=("A0p20to0p05" "A0p50to0p20" "A0p70to0p30")

    CONF_OPTIONS=(0 1)  # 0=conf ON, 1=conf OFF
    CONF_NAMES=("confON" "confOFF")

    # Fixed alignment for A1-A3
    ALPHA_ATTN_A=0.05
    ALPHA_REP_A=0.10
    ALPHA_NCE_A=0.10
    TAU_NCE_A=0.10

    for n_views in "${N_HLT_VIEWS_LIST[@]}"; do
        echo "BLOCK A (views=${n_views})"

        # A1-A3: with alignment
        for t_idx in 0 1 2; do
            IFS=':' read -r TEMP_INIT TEMP_FINAL <<< "${TEMP_SCHEDULES[$t_idx]}"
            T_NAME="${TEMP_NAMES[$t_idx]}"

            for a_idx in 0 1 2; do
                IFS=':' read -r ALPHA_INIT ALPHA_FINAL <<< "${ALPHA_SCHEDULES[$a_idx]}"
                A_NAME="${ALPHA_NAMES[$a_idx]}"

                for c_idx in 0 1; do
                    NO_CONF_KD="${CONF_OPTIONS[$c_idx]}"
                    C_NAME="${CONF_NAMES[$c_idx]}"

                    RUN_NAME="A_v${n_views}_${T_NAME}_${A_NAME}_${C_NAME}_repnce"
                    submit_job "$RUN_NAME" "$n_views" "$TEMP_INIT" "$TEMP_FINAL" "$ALPHA_INIT" "$ALPHA_FINAL" \
                        "$ALPHA_ATTN_A" "$ALPHA_REP_A" "$ALPHA_NCE_A" "$TAU_NCE_A" "$NO_CONF_KD"
                done
            done
        done

        # A4: without alignment
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

                    RUN_NAME="A_v${n_views}_${T_NAME}_${A_NAME}_${C_NAME}_noalign"
                    submit_job "$RUN_NAME" "$n_views" "$TEMP_INIT" "$TEMP_FINAL" "$ALPHA_INIT" "$ALPHA_FINAL" \
                        "$ALPHA_ATTN_A" "$ALPHA_REP_A4" "$ALPHA_NCE_A4" "$TAU_NCE_A" "$NO_CONF_KD"
                done
            done
        done
    done

    # ----------------------------------------------------------------------
    # BLOCK B: Rep vs NCE dominance (24 runs per view)
    # ----------------------------------------------------------------------
    TEMP_INIT_B=10.0
    TEMP_FINAL_B=4.0
    ALPHA_INIT_B=0.50
    ALPHA_FINAL_B=0.20
    ALPHA_ATTN_B=0.05
    TAU_NCE_B=0.10
    NO_CONF_KD_B=0

    REP_STRENGTHS=(0.00 0.05 0.10 0.20)
    NCE_STRENGTHS=(0.00 0.05 0.10 0.20)
    JOINT_REPS=(0.05 0.10 0.20)
    JOINT_NCES=(0.05 0.10 0.20)
    TAU_VALS=(0.07 0.10 0.20)
    ATTN_WEIGHTS=(0.00 0.02 0.05 0.10)

    for n_views in "${N_HLT_VIEWS_LIST[@]}"; do
        echo "BLOCK B (views=${n_views})"

        for ALPHA_REP_B1 in "${REP_STRENGTHS[@]}"; do
            rep_tag=$(tag_float "$ALPHA_REP_B1")
            RUN_NAME="B1_v${n_views}_reponly_${rep_tag}"
            submit_job "$RUN_NAME" "$n_views" "$TEMP_INIT_B" "$TEMP_FINAL_B" "$ALPHA_INIT_B" "$ALPHA_FINAL_B" \
                "$ALPHA_ATTN_B" "$ALPHA_REP_B1" "0.00" "$TAU_NCE_B" "$NO_CONF_KD_B"
        done

        for ALPHA_NCE_B2 in "${NCE_STRENGTHS[@]}"; do
            nce_tag=$(tag_float "$ALPHA_NCE_B2")
            RUN_NAME="B2_v${n_views}_nceonly_${nce_tag}"
            submit_job "$RUN_NAME" "$n_views" "$TEMP_INIT_B" "$TEMP_FINAL_B" "$ALPHA_INIT_B" "$ALPHA_FINAL_B" \
                "$ALPHA_ATTN_B" "0.00" "$ALPHA_NCE_B2" "$TAU_NCE_B" "$NO_CONF_KD_B"
        done

        for ALPHA_REP_B3 in "${JOINT_REPS[@]}"; do
            for ALPHA_NCE_B3 in "${JOINT_NCES[@]}"; do
                rep_tag=$(tag_float "$ALPHA_REP_B3")
                nce_tag=$(tag_float "$ALPHA_NCE_B3")
                RUN_NAME="B3_v${n_views}_joint_rep${rep_tag}_nce${nce_tag}_tau0p10"
                submit_job "$RUN_NAME" "$n_views" "$TEMP_INIT_B" "$TEMP_FINAL_B" "$ALPHA_INIT_B" "$ALPHA_FINAL_B" \
                    "$ALPHA_ATTN_B" "$ALPHA_REP_B3" "$ALPHA_NCE_B3" "0.10" "$NO_CONF_KD_B"
            done
        done

        for TAU_NCE_B3 in "${TAU_VALS[@]}"; do
            tau_tag=$(tag_float "$TAU_NCE_B3")
            RUN_NAME="B3_v${n_views}_tausens_rep0p10_nce0p10_tau${tau_tag}"
            submit_job "$RUN_NAME" "$n_views" "$TEMP_INIT_B" "$TEMP_FINAL_B" "$ALPHA_INIT_B" "$ALPHA_FINAL_B" \
                "$ALPHA_ATTN_B" "0.10" "0.10" "$TAU_NCE_B3" "$NO_CONF_KD_B"
        done

        for ALPHA_ATTN_B4 in "${ATTN_WEIGHTS[@]}"; do
            attn_tag=$(tag_float "$ALPHA_ATTN_B4")
            RUN_NAME="B4_v${n_views}_attn${attn_tag}_rep0p10_nce0p10"
            submit_job "$RUN_NAME" "$n_views" "$TEMP_INIT_B" "$TEMP_FINAL_B" "$ALPHA_INIT_B" "$ALPHA_FINAL_B" \
                "$ALPHA_ATTN_B4" "0.10" "0.10" "0.10" "$NO_CONF_KD_B"
        done
    done

    # ----------------------------------------------------------------------
    # BLOCK C: Teacher-trust gating sanity (10 runs per view)
    # ----------------------------------------------------------------------
    TEMP_INIT_C=10.0
    TEMP_FINAL_C=4.0
    ALPHA_INIT_C=0.50
    ALPHA_FINAL_C=0.20
    ALPHA_ATTN_C=0.05

    REGIMES=(
        "plainKD:0.00:0.00"
        "reponly:0.10:0.00"
        "nceonly:0.00:0.10"
        "balanced:0.10:0.10"
        "repheavy:0.20:0.10"
    )

    for n_views in "${N_HLT_VIEWS_LIST[@]}"; do
        echo "BLOCK C (views=${n_views})"

        for r in "${REGIMES[@]}"; do
            IFS=':' read -r REGIME_NAME ALPHA_REP_C ALPHA_NCE_C <<< "$r"

            RUN_NAME="C_v${n_views}_${REGIME_NAME}_confON"
            submit_job "$RUN_NAME" "$n_views" "$TEMP_INIT_C" "$TEMP_FINAL_C" "$ALPHA_INIT_C" "$ALPHA_FINAL_C" \
                "$ALPHA_ATTN_C" "$ALPHA_REP_C" "$ALPHA_NCE_C" "0.10" "0"

            RUN_NAME="C_v${n_views}_${REGIME_NAME}_confOFF"
            submit_job "$RUN_NAME" "$n_views" "$TEMP_INIT_C" "$TEMP_FINAL_C" "$ALPHA_INIT_C" "$ALPHA_FINAL_C" \
                "$ALPHA_ATTN_C" "$ALPHA_REP_C" "$ALPHA_NCE_C" "0.10" "1"
        done
    done

    # ----------------------------------------------------------------------
    # BLOCK D: Consistency weight sanity (10 runs per view)
    # ----------------------------------------------------------------------
    TEMP_INIT_D=10.0
    TEMP_FINAL_D=4.0
    ALPHA_INIT_D=0.50
    ALPHA_FINAL_D=0.20
    ALPHA_ATTN_D=0.05
    ALPHA_REP_D=0.10
    ALPHA_NCE_D=0.10
    TAU_NCE_D=0.10
    NO_CONF_KD_D=0

    CONSIST_COMBOS=(
        "lp0p5_le0p1:0.5:0.1"
        "lp1p0_le0p25:1.0:0.25"
        "lp2p0_le0p5:2.0:0.5"
        "lp1p5_le0p25:1.5:0.25"
        "lp1p0_le0p5:1.0:0.5"
    )
    CONF_POWERS_D=(1.0 2.0)

    for n_views in "${N_HLT_VIEWS_LIST[@]}"; do
        echo "BLOCK D (views=${n_views})"

        for combo in "${CONSIST_COMBOS[@]}"; do
            IFS=':' read -r COMBO_NAME LP_D LE_D <<< "$combo"
            for CP_D in "${CONF_POWERS_D[@]}"; do
                cp_tag=$(tag_float "$CP_D")
                RUN_NAME="D_v${n_views}_${COMBO_NAME}_cp${cp_tag}"
                submit_job "$RUN_NAME" "$n_views" "$TEMP_INIT_D" "$TEMP_FINAL_D" "$ALPHA_INIT_D" "$ALPHA_FINAL_D" \
                    "$ALPHA_ATTN_D" "$ALPHA_REP_D" "$ALPHA_NCE_D" "$TAU_NCE_D" "$NO_CONF_KD_D" \
                    "$LP_D" "$LE_D" "$CP_D" "$CONF_MIN"
            done
        done
    done
fi

if [ "$SWEEP_MODE" = "round2" ] || [ "$SWEEP_MODE" = "both" ]; then
    echo "===== ROUND 2 GRID (focused KD) ====="
    echo "Views: ${N_HLT_VIEWS_LIST[*]}"
    echo ""

    TEMPERATURES=(8.0 9.0 10.0 11.0 12.0)
    ALPHAS=(0.7 0.75 0.8 0.85 0.9 0.95)
    TEMP_FINALS=(1.0 1.5 2.0 2.5 3.0)
    ALPHA_FINALS=(0.1 0.2 0.3 0.4 0.5)

    HIGH_ALPHAS=(0.85 0.9 0.95)
    HIGH_TEMPS=(10.0 11.0 12.0)
    BEST_TEMPS=(10.0 11.0 12.0)
    BEST_ALPHAS=(0.85 0.9 0.95)
    BEST_TEMP_FINALS=(1.0 2.0)
    BEST_ALPHA_FINALS=(0.2 0.3)

    # Round 2 uses plain KD unless overridden
    R2_ALPHA_ATTN=${R2_ALPHA_ATTN:-0.00}
    R2_ALPHA_REP=${R2_ALPHA_REP:-0.00}
    R2_ALPHA_NCE=${R2_ALPHA_NCE:-0.00}
    R2_TAU_NCE=${R2_TAU_NCE:-0.10}
    R2_NO_CONF_KD=${R2_NO_CONF_KD:-0}

    for n_views in "${N_HLT_VIEWS_LIST[@]}"; do
        # Baseline: constant temp/alpha
        for TEMP in "${TEMPERATURES[@]}"; do
            for ALPHA in "${ALPHAS[@]}"; do
                t_tag=$(tag_float "$TEMP")
                a_tag=$(tag_float "$ALPHA")
                RUN_NAME="r2_v${n_views}_baseline_T${t_tag}_A${a_tag}"
                submit_job "$RUN_NAME" "$n_views" "$TEMP" "" "$ALPHA" "" \
                    "$R2_ALPHA_ATTN" "$R2_ALPHA_REP" "$R2_ALPHA_NCE" "$R2_TAU_NCE" "$R2_NO_CONF_KD"
            done
        done

        # Temp annealing (only high alphas)
        for TEMP in "${TEMPERATURES[@]}"; do
            for TEMP_FINAL in "${TEMP_FINALS[@]}"; do
                for ALPHA in "${HIGH_ALPHAS[@]}"; do
                    if (( $(echo "$TEMP > $TEMP_FINAL" | bc -l) )); then
                        t_tag=$(tag_float "$TEMP")
                        tf_tag=$(tag_float "$TEMP_FINAL")
                        a_tag=$(tag_float "$ALPHA")
                        RUN_NAME="r2_v${n_views}_tempanneal_T${t_tag}to${tf_tag}_A${a_tag}"
                        submit_job "$RUN_NAME" "$n_views" "$TEMP" "$TEMP_FINAL" "$ALPHA" "" \
                            "$R2_ALPHA_ATTN" "$R2_ALPHA_REP" "$R2_ALPHA_NCE" "$R2_TAU_NCE" "$R2_NO_CONF_KD"
                    fi
                done
            done
        done

        # Alpha scheduling (only high temps)
        for TEMP in "${HIGH_TEMPS[@]}"; do
            for ALPHA in "${ALPHAS[@]}"; do
                for ALPHA_FINAL in "${ALPHA_FINALS[@]}"; do
                    if (( $(echo "$ALPHA > $ALPHA_FINAL" | bc -l) )); then
                        t_tag=$(tag_float "$TEMP")
                        a_tag=$(tag_float "$ALPHA")
                        af_tag=$(tag_float "$ALPHA_FINAL")
                        RUN_NAME="r2_v${n_views}_alphasched_T${t_tag}_A${a_tag}to${af_tag}"
                        submit_job "$RUN_NAME" "$n_views" "$TEMP" "" "$ALPHA" "$ALPHA_FINAL" \
                            "$R2_ALPHA_ATTN" "$R2_ALPHA_REP" "$R2_ALPHA_NCE" "$R2_TAU_NCE" "$R2_NO_CONF_KD"
                    fi
                done
            done
        done

        # Combined annealing + scheduling
        for TEMP in "${BEST_TEMPS[@]}"; do
            for TEMP_FINAL in "${BEST_TEMP_FINALS[@]}"; do
                for ALPHA in "${BEST_ALPHAS[@]}"; do
                    for ALPHA_FINAL in "${BEST_ALPHA_FINALS[@]}"; do
                        if (( $(echo "$TEMP > $TEMP_FINAL" | bc -l) )) && (( $(echo "$ALPHA > $ALPHA_FINAL" | bc -l) )); then
                            t_tag=$(tag_float "$TEMP")
                            tf_tag=$(tag_float "$TEMP_FINAL")
                            a_tag=$(tag_float "$ALPHA")
                            af_tag=$(tag_float "$ALPHA_FINAL")
                            RUN_NAME="r2_v${n_views}_combined_T${t_tag}to${tf_tag}_A${a_tag}to${af_tag}"
                            submit_job "$RUN_NAME" "$n_views" "$TEMP" "$TEMP_FINAL" "$ALPHA" "$ALPHA_FINAL" \
                                "$R2_ALPHA_ATTN" "$R2_ALPHA_REP" "$R2_ALPHA_NCE" "$R2_TAU_NCE" "$R2_NO_CONF_KD"
                        fi
                    done
                done
            done
        done
    done
fi

echo "Submitted ${run_count} jobs to Slurm."
