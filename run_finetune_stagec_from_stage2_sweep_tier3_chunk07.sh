#!/usr/bin/env bash
#SBATCH --job-name=stcSw07
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=offline_reconstructor_logs/stagec_sweep_chunk07_%j.out
#SBATCH --error=offline_reconstructor_logs/stagec_sweep_chunk07_%j.err

set -euo pipefail
export CHUNK_ID=6
bash run_finetune_stagec_from_stage2_sweep_chunk_common.sh
