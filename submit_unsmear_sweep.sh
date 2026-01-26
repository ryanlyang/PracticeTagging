#!/bin/bash

# Sweep over conditional diffusion unsmearing knobs.
# Uses run_unsmear_single.sh as the sbatch runner.

RUN_SCRIPT="run_unsmear_single.sh"
SAVE_DIR=${SAVE_DIR:-"checkpoints/unsmear_sweep"}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}

mkdir -p unsmear_logs

pred_types=("eps" "x0" "v")
cross_attn=(1 0)
self_cond=(1 0)
snr_weight=(1 0)
jet_loss=("0.1" "0.0")
sampling=("ddim" "ddpm")
guidance=("1.0" "1.5")

count=0

for PRED in "${pred_types[@]}"; do
  for XATTN in "${cross_attn[@]}"; do
    for SCOND in "${self_cond[@]}"; do
      for SNR in "${snr_weight[@]}"; do
        for JET in "${jet_loss[@]}"; do
          for SAMP in "${sampling[@]}"; do
            for GUID in "${guidance[@]}"; do
              # Only enable CFG dropout if guidance > 1
              if [[ "$GUID" == "1.0" ]]; then
                COND_DROP=0.0
              else
                COND_DROP=0.1
              fi

              if [[ "$SCOND" == "1" ]]; then
                SELF_COND_PROB=0.5
                NO_SELF_COND=0
              else
                SELF_COND_PROB=0.0
                NO_SELF_COND=1
              fi

              RUN_NAME="U_pred${PRED}_xattn${XATTN}_sc${SCOND}_snr${SNR}_jet${JET}_samp${SAMP}_g${GUID}"

              echo "Submitting: $RUN_NAME"
              EXPORTS="ALL,"
              EXPORTS+="RUN_NAME=$RUN_NAME,"
              EXPORTS+="SAVE_DIR=$SAVE_DIR,"
              EXPORTS+="N_TRAIN_JETS=$N_TRAIN_JETS,"
              EXPORTS+="MAX_CONSTITS=$MAX_CONSTITS,"
              EXPORTS+="PRED_TYPE=$PRED,"
              EXPORTS+="USE_CROSS_ATTN=$XATTN,"
              EXPORTS+="NO_SELF_COND=$NO_SELF_COND,"
              EXPORTS+="SELF_COND_PROB=$SELF_COND_PROB,"
              EXPORTS+="SNR_WEIGHT=$SNR,"
              EXPORTS+="SNR_GAMMA=5.0,"
              EXPORTS+="COND_DROP_PROB=$COND_DROP,"
              EXPORTS+="JET_LOSS_WEIGHT=$JET,"
              EXPORTS+="SAMPLING_METHOD=$SAMP,"
              EXPORTS+="GUIDANCE_SCALE=$GUID,"
              EXPORTS+="SAMPLE_STEPS=200,"
              EXPORTS+="N_SAMPLES_EVAL=1"
              sbatch --export="$EXPORTS" "$RUN_SCRIPT"

              count=$((count + 1))
            done
          done
        done
      done
    done
  done
done

echo "Total jobs submitted: $count"
