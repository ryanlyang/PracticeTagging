#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "Submitting m36 descriptor-index run..."
sbatch --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}" "${SCRIPT_DIR}/run_m36_k3_dict1m_retrieval_dualview_150k75k300k.sh"

echo "Submitting m36 learned-index run..."
sbatch --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}" "${SCRIPT_DIR}/run_m36_k3_dict1m_retrieval_dualview_learnedidx_150k75k300k.sh"
