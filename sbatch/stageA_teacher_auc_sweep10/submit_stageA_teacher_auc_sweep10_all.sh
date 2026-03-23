#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scripts=(
  run_stageA_sweep10_s01_balanced.sh
  run_stageA_sweep10_s02_kd_dominant.sh
  run_stageA_sweep10_s03_semantic_heavy.sh
  run_stageA_sweep10_s04_phys_budget_heavy.sh
  run_stageA_sweep10_s05_high_temp.sh
  run_stageA_sweep10_s06_low_temp.sh
  run_stageA_sweep10_s07_norm_off.sh
  run_stageA_sweep10_s08_zero_budget.sh
  run_stageA_sweep10_s09_oldstyle_proxy.sh
  run_stageA_sweep10_s10_longer_stageA.sh
)

for s in "${scripts[@]}"; do
  echo "Submitting ${s}"
  sbatch "${SCRIPT_DIR}/${s}"
done
