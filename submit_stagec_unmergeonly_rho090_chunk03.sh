#!/usr/bin/env bash
#SBATCH --job-name=nrivUO9C03
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=3-12:00:00
#SBATCH --output=offline_reconstructor_logs/stagec_uo_rho090_chunk03_%j.out
#SBATCH --error=offline_reconstructor_logs/stagec_uo_rho090_chunk03_%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

RUNNER="${ROOT_DIR}/run_offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_rho090_100k80_flags.sh"
if [[ ! -f "$RUNNER" ]]; then
  if [[ -f "run_offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_rho090_100k80_flags.sh" ]]; then
    RUNNER="run_offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_rho090_100k80_flags.sh"
  else
    echo "Runner not found in ROOT_DIR=$ROOT_DIR: $RUNNER" >&2
    exit 1
  fi
fi

set +u
source ~/.bashrc
set -u
conda activate atlas_kd

mkdir -p offline_reconstructor_logs
manifest="offline_reconstructor_logs/stagec_unmergeonly_rho090_chunk03_runs_${SLURM_JOB_ID:-manual}.tsv"
printf "idx\trun_name\tadded_target_scale\tlambda_cons\tlambda_reco\tstageC_lr_dual\tstageC_lr_reco\tprofile\texit_code\n" > "$manifest"

tagify() {
  local v="$1"
  v="${v//./p}"
  v="${v//-/m}"
  v="${v//+/p}"
  echo "$v"
}

run_one() {
  local idx="$1"
  local s="$2"
  local c="$3"
  local profile="$4"
  local lr_dual="$5"
  local lr_reco="$6"
  local l_reco="$7"

  local s_tag c_tag lr_d_tag lr_r_tag lr_reco_tag run_name exit_code
  s_tag=$(tagify "$s")
  c_tag=$(tagify "$c")
  lr_d_tag=$(tagify "$lr_dual")
  lr_r_tag=$(tagify "$lr_reco")
  lr_reco_tag=$(tagify "$l_reco")
  run_name="joint_uo_rho090_1MJ100C_s${s_tag}_c${c_tag}_lr${lr_reco_tag}_ld${lr_d_tag}_lr${lr_r_tag}_${profile}"

  echo "[chunk03] (${idx}/4) Starting $run_name"
  set +e
  RUN_NAME="$run_name" \
  N_TRAIN_JETS=1000000 \
  MAX_CONSTITS=100 \
  NUM_WORKERS=6 \
  ADDED_TARGET_SCALE="$s" \
  SELECTION_METRIC=auc \
  STAGEB_LAMBDA_RANK=0.0 \
  STAGEB_LAMBDA_CONS=0.0 \
  STAGEC_LR_DUAL="$lr_dual" \
  STAGEC_LR_RECO="$lr_reco" \
  LAMBDA_RECO="$l_reco" \
  LAMBDA_CONS="$c" \
  bash "$RUNNER"
  exit_code=$?
  set -e

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$idx" "$run_name" "$s" "$c" "$l_reco" "$lr_dual" "$lr_reco" "$profile" "$exit_code" >> "$manifest"

  if [[ "$exit_code" -ne 0 ]]; then
    echo "[chunk03] FAILED on $run_name (exit=$exit_code). Stopping chunk." >&2
    exit "$exit_code"
  fi
  echo "[chunk03] Completed $run_name"
}

echo "Running 4 sequential Stage-C runs for chunk03"
run_one 1 0.95 0.00 A 2e-5 1e-5 0.50
run_one 2 0.95 0.00 B 3e-5 1.5e-5 0.60
run_one 3 0.95 0.02 A 2e-5 1e-5 0.50
run_one 4 0.95 0.02 B 3e-5 1.5e-5 0.60


echo "Chunk 03 complete. Manifest: $manifest"
