#!/bin/bash

# Sweep toggles for unmerge_distr_model.py
# First run is "all-on".

RUN_SCRIPT="run_unmerge_distr_single.sh"
SAVE_DIR=${SAVE_DIR:-"checkpoints/unmerge_distr_sweep"}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
MAX_MERGE_COUNT=${MAX_MERGE_COUNT:-10}

mkdir -p unmerge_distr_logs

loss_types=("hungarian" "chamfer")
use_true_count=(1 0)
curriculum=(1 0)
distributional=(1 0)
physics_weight=("0.2" "0.0")
nll_weight=("1.0" "0.5")

count=0

# First run: all-on
RUN_NAME="all_on"
EXPORTS="ALL,"
EXPORTS+="RUN_NAME=$RUN_NAME,"
EXPORTS+="SAVE_DIR=$SAVE_DIR,"
EXPORTS+="N_TRAIN_JETS=$N_TRAIN_JETS,"
EXPORTS+="MAX_CONSTITS=$MAX_CONSTITS,"
EXPORTS+="MAX_MERGE_COUNT=$MAX_MERGE_COUNT,"
EXPORTS+="UNMERGE_LOSS=hungarian,"
EXPORTS+="USE_TRUE_COUNT=1,"
EXPORTS+="NO_CURRICULUM=0,"
EXPORTS+="PHYSICS_WEIGHT=0.2,"
EXPORTS+="NLL_WEIGHT=1.0,"
EXPORTS+="NO_DISTRIBUTIONAL=0"
echo "Submitting: $RUN_NAME"
sbatch --export="$EXPORTS" "$RUN_SCRIPT"
count=$((count + 1))

for LOSS in "${loss_types[@]}"; do
  for TRUEC in "${use_true_count[@]}"; do
    for CURR in "${curriculum[@]}"; do
      for DIST in "${distributional[@]}"; do
        for PHYS in "${physics_weight[@]}"; do
          for NLL in "${nll_weight[@]}"; do
            RUN_NAME="D_${LOSS}_tc${TRUEC}_cur${CURR}_dist${DIST}_phys${PHYS}_nll${NLL}"

            if [[ "$CURR" == "1" ]]; then
              NO_CURR=0
            else
              NO_CURR=1
            fi

            if [[ "$DIST" == "1" ]]; then
              NO_DIST=0
            else
              NO_DIST=1
            fi

            EXPORTS="ALL,"
            EXPORTS+="RUN_NAME=$RUN_NAME,"
            EXPORTS+="SAVE_DIR=$SAVE_DIR,"
            EXPORTS+="N_TRAIN_JETS=$N_TRAIN_JETS,"
            EXPORTS+="MAX_CONSTITS=$MAX_CONSTITS,"
            EXPORTS+="MAX_MERGE_COUNT=$MAX_MERGE_COUNT,"
            EXPORTS+="UNMERGE_LOSS=$LOSS,"
            EXPORTS+="USE_TRUE_COUNT=$TRUEC,"
            EXPORTS+="NO_CURRICULUM=$NO_CURR,"
            EXPORTS+="PHYSICS_WEIGHT=$PHYS,"
            EXPORTS+="NLL_WEIGHT=$NLL,"
            EXPORTS+="NO_DISTRIBUTIONAL=$NO_DIST"

            echo "Submitting: $RUN_NAME"
            sbatch --export="$EXPORTS" "$RUN_SCRIPT"
            count=$((count + 1))
          done
        done
      done
    done
  done
done

echo "Total jobs submitted: $count"
