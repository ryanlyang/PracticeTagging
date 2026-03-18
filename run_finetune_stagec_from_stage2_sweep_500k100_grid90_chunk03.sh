#!/usr/bin/env bash
#SBATCH --job-name=stc5k03
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=offline_reconstructor_logs/stagec_sweep_500k100_g90_chunk03_%j.out
#SBATCH --error=offline_reconstructor_logs/stagec_sweep_500k100_g90_chunk03_%j.err

set -euo pipefail
export CHUNK_ID=2
bash run_finetune_stagec_from_stage2_sweep_500k100_grid90_chunk_common.sh

