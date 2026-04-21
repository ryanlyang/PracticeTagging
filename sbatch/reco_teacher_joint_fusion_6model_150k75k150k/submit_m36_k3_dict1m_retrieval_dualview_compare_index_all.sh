#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "Submitting m36 descriptor-index run..."
sbatch run_m36_k3_dict1m_retrieval_dualview_150k75k300k.sh

echo "Submitting m36 learned-index run..."
sbatch run_m36_k3_dict1m_retrieval_dualview_learnedidx_150k75k300k.sh

