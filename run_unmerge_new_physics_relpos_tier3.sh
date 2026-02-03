#!/usr/bin/env bash
#SBATCH --job-name=unmerge_ph
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6-12:00:00
#SBATCH --output=unmerge_new_physics_logs/unmerge_physics_%j.out
#SBATCH --error=unmerge_new_physics_logs/unmerge_physics_%j.err

set -euo pipefail

mkdir -p unmerge_new_physics_logs

RUN_NAME="${RUN_NAME:-physics02_relpos_new200k}"
PHYSICS_WEIGHT="${PHYSICS_WEIGHT:-0.2}"
OFFSET_JETS="${OFFSET_JETS:-200000}"
RELPOS_MODE="${RELPOS_MODE:-attn}"

source ~/.bashrc
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

python unmerge_new_ideas.py \
  --save_dir checkpoints/unmerge_new_physics_relpos \
  --run_name "${RUN_NAME}" \
  --n_train_jets 200000 \
  --offset_jets "${OFFSET_JETS}" \
  --max_constits 80 \
  --max_merge_count 10 \
  --physics_weight "${PHYSICS_WEIGHT}" \
  --unmerge_relpos_mode "${RELPOS_MODE}" \
  --device cuda
