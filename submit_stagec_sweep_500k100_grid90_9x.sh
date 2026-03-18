#!/usr/bin/env bash
# Submit all 9 chunked Stage-C sweep jobs (10 configs each, total 90).
#
# Usage:
#   bash submit_stagec_sweep_500k100_grid90_9x.sh

set -euo pipefail

scripts=(
  run_finetune_stagec_from_stage2_sweep_500k100_grid90_chunk01.sh
  run_finetune_stagec_from_stage2_sweep_500k100_grid90_chunk02.sh
  run_finetune_stagec_from_stage2_sweep_500k100_grid90_chunk03.sh
  run_finetune_stagec_from_stage2_sweep_500k100_grid90_chunk04.sh
  run_finetune_stagec_from_stage2_sweep_500k100_grid90_chunk05.sh
  run_finetune_stagec_from_stage2_sweep_500k100_grid90_chunk06.sh
  run_finetune_stagec_from_stage2_sweep_500k100_grid90_chunk07.sh
  run_finetune_stagec_from_stage2_sweep_500k100_grid90_chunk08.sh
  run_finetune_stagec_from_stage2_sweep_500k100_grid90_chunk09.sh
)

for s in "${scripts[@]}"; do
  if [[ ! -f "${s}" ]]; then
    echo "Missing script: ${s}"
    exit 2
  fi
done

echo "Submitting ${#scripts[@]} jobs..."
for s in "${scripts[@]}"; do
  out=$(sbatch "${s}")
  echo "${out}  <- ${s}"
done

echo "Done."

