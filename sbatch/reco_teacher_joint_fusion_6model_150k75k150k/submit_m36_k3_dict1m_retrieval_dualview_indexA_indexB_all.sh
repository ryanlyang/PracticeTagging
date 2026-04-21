#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "Submitting Index A: physics descriptor (no learned index)..."
sbatch run_m36_k3_dict1m_retrieval_dualview_150k75k300k.sh

echo "Submitting Index B: landmark11 vector (no learned index)..."
sbatch run_m36_k3_dict1m_retrieval_dualview_landmark11_150k75k300k.sh

echo "Submitting Index B + learned HLT embedding index..."
sbatch run_m36_k3_dict1m_retrieval_dualview_landmark11_learnedidx_150k75k300k.sh

