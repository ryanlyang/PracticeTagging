#!/bin/bash

# Sweep over conditional diffusion unsmearing knobs.
# Uses run_unsmear_single.sh as the sbatch runner.

RUN_SCRIPT="run_unsmear_single.sh"
RUNS_PER_JOB=${RUNS_PER_JOB:-10}
SAVE_DIR=${SAVE_DIR:-"checkpoints/unsmear_sweep"}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}

mkdir -p unsmear_logs
mkdir -p unsmear_batches

pred_types=("eps" "x0" "v")
cross_attn=(1 0)
self_cond=(1 0)
snr_weight=(1 0)
jet_loss=("0.1" "0.0")
sampling=("ddim" "ddpm")
guidance=("1.0" "1.5")

configs=()

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

              line="RUN_NAME=$RUN_NAME"
              line+=";SAVE_DIR=$SAVE_DIR"
              line+=";N_TRAIN_JETS=$N_TRAIN_JETS"
              line+=";MAX_CONSTITS=$MAX_CONSTITS"
              line+=";PRED_TYPE=$PRED"
              line+=";USE_CROSS_ATTN=$XATTN"
              line+=";NO_SELF_COND=$NO_SELF_COND"
              line+=";SELF_COND_PROB=$SELF_COND_PROB"
              line+=";SNR_WEIGHT=$SNR"
              line+=";SNR_GAMMA=5.0"
              line+=";COND_DROP_PROB=$COND_DROP"
              line+=";JET_LOSS_WEIGHT=$JET"
              line+=";SAMPLING_METHOD=$SAMP"
              line+=";GUIDANCE_SCALE=$GUID"
              line+=";SAMPLE_STEPS=200"
              line+=";N_SAMPLES_EVAL=1"
              configs+=("$line")
            done
          done
        done
      done
    done
  done
done

total_configs=${#configs[@]}
if [ "$total_configs" -eq 0 ]; then
  echo "No configs generated."
  exit 0
fi

job_count=0
batch_idx=0
for ((i=0; i<total_configs; i+=RUNS_PER_JOB)); do
  batch_idx=$((batch_idx + 1))
  batch_file="unsmear_batches/unsmear_batch_${batch_idx}.txt"
  : > "$batch_file"
  for ((j=i; j<i+RUNS_PER_JOB && j<total_configs; j++)); do
    echo "${configs[j]}" >> "$batch_file"
  done
  echo "Submitting batch $batch_idx: $batch_file"
  sbatch --export="ALL,BATCH_FILE=$batch_file" "$RUN_SCRIPT"
  job_count=$((job_count + 1))
done

echo "Total configs: $total_configs"
echo "Runs per job: $RUNS_PER_JOB"
echo "Total jobs submitted: $job_count"
