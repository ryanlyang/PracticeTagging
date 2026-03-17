#!/usr/bin/env bash
# Submit all 8 Stage-C sweep chunks to tier3.

set -euo pipefail

scripts=(
  run_finetune_stagec_from_stage2_sweep_tier3_chunk01.sh
  run_finetune_stagec_from_stage2_sweep_tier3_chunk02.sh
  run_finetune_stagec_from_stage2_sweep_tier3_chunk03.sh
  run_finetune_stagec_from_stage2_sweep_tier3_chunk04.sh
  run_finetune_stagec_from_stage2_sweep_tier3_chunk05.sh
  run_finetune_stagec_from_stage2_sweep_tier3_chunk06.sh
  run_finetune_stagec_from_stage2_sweep_tier3_chunk07.sh
  run_finetune_stagec_from_stage2_sweep_tier3_chunk08.sh
)

echo "Submitting ${#scripts[@]} sweep chunks..."
for s in "${scripts[@]}"; do
  if [[ ! -f "${s}" ]]; then
    echo "Missing script: ${s}"
    exit 2
  fi
  out="$(sbatch "${s}")"
  echo "${s}: ${out}"
done

echo "Done. Check queue with: squeue --me"
