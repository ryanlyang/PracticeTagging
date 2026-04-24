#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_analyze_hlt_joint55_bin_gated_valsel_precomputed_1m.sh"

if [[ ! -f "${RUNNER}" ]]; then
  echo "ERROR: runner not found: ${RUNNER}" >&2
  exit 1
fi

echo "Submitting ${RUNNER}"
sbatch "${RUNNER}"

