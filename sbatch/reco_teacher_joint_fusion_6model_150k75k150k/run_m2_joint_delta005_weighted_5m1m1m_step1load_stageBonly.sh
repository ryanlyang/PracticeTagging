#!/usr/bin/env bash
#SBATCH --job-name=m2j5b
#SBATCH --partition=tier3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=360G
#SBATCH --time=5-20:00:00
#SBATCH --requeue
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_joint_delta005_step1load_stageBonly_weighted_5m1m1m_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_joint_delta005_step1load_stageBonly_weighted_5m1m1m_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SKIP_STAGEC_JOINT=1
export STAGEC_EPOCHS="${STAGEC_EPOCHS:-0}"
export STAGEC_PATIENCE="${STAGEC_PATIENCE:-0}"
export STAGEC_MIN_EPOCHS="${STAGEC_MIN_EPOCHS:-0}"

exec bash "${SCRIPT_DIR}/run_m2_joint_delta005_weighted_5m1m1m_step1load_joint12.sh"
