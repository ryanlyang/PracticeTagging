#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "Submitting m36 triple-run: residual_only + one-big-pull (5000 x 1)"
SELECTOR_MODE=residual_only \
RETRIEVAL_PER_ROUND=5000 \
RETRIEVAL_MAX_ROUNDS=1 \
bash "${SCRIPT_DIR}/submit_m36_k3_dict1m_retrieval_dualview_indexA_indexB_all.sh"

