#!/bin/bash

#SBATCH --job-name=unmerge_eval_new
#SBATCH --output=unmerge_eval_newdata_logs/unmerge_eval_new_%j.out
#SBATCH --error=unmerge_eval_newdata_logs/unmerge_eval_new_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

mkdir -p unmerge_eval_newdata_logs

source ~/.bashrc
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

python unmerge_model_eval_newdata.py \
  --ckpt_dir checkpoints/unmerge_model/default \
  --save_dir checkpoints/unmerge_model_eval \
  --run_name eval_newdata_offset200k \
  --n_eval_jets 200000 \
  --offset_jets 200000 \
  --stats_jets 200000 \
  --max_constits 80 \
  --max_merge_count 10 \
  --device cuda
