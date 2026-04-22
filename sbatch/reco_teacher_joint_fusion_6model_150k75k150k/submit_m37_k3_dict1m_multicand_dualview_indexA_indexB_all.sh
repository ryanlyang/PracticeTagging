#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "Submitting Index A: physics descriptor (no learned index)..."
sbatch --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}" "${SCRIPT_DIR}/run_m37_k3_dict1m_multicand_dualview_150k75k300k.sh"

echo "Submitting Index B: landmark11 vector (no learned index)..."
sbatch --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}" "${SCRIPT_DIR}/run_m37_k3_dict1m_multicand_dualview_landmark11_150k75k300k.sh"

echo "Submitting Index B + learned HLT embedding index..."
sbatch --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}" "${SCRIPT_DIR}/run_m37_k3_dict1m_multicand_dualview_landmark11_learnedidx_150k75k300k.sh"
