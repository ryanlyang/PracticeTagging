#!/usr/bin/env bash
#SBATCH --job-name=unmerge_ph
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=19-23:00:00
#SBATCH --output=unmerge_new_physics_logs/unmerge_physics_%j.out
#SBATCH --error=unmerge_new_physics_logs/unmerge_physics_%j.err

set -euo pipefail

mkdir -p unmerge_new_physics_logs

RUN_NAME="${RUN_NAME:-physics02_relpos_new200k}"
PHYSICS_WEIGHT="${PHYSICS_WEIGHT:-0.2}"
OFFSET_JETS="${OFFSET_JETS:-200000}"
RELPOS_MODE="${RELPOS_MODE:-attn}"
N_TRAIN_JETS="${N_TRAIN_JETS:-200000}"
MAX_CONSTITS="${MAX_CONSTITS:-80}"
MAX_MERGE_COUNT="${MAX_MERGE_COUNT:-10}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

python unmerge_new_ideas.py \
  --save_dir checkpoints/unmerge_new_physics_relpos \
  --run_name "${RUN_NAME}" \
  --n_train_jets "${N_TRAIN_JETS}" \
  --offset_jets "${OFFSET_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --max_merge_count "${MAX_MERGE_COUNT}" \
  --physics_weight "${PHYSICS_WEIGHT}" \
  --unmerge_relpos_mode "${RELPOS_MODE}" \
  --device cuda
