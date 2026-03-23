#!/usr/bin/env bash
# Submit Stage-C sweep chunk05 (4 runs).
set -euo pipefail

RUNNER="run_offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_rho090_100k80_flags.sh"
if [[ ! -f "$RUNNER" ]]; then
  echo "Runner not found: $RUNNER" >&2
  exit 1
fi

mkdir -p offline_reconstructor_logs
stamp="$(date +%Y%m%d_%H%M%S)"
manifest="offline_reconstructor_logs/stagec_unmergeonly_rho090_chunk05_${stamp}.tsv"
echo -e "job_id\trun_name\tadded_target_scale\tlambda_cons\tlambda_reco\tstageC_lr_dual\tstageC_lr_reco\tprofile" > "$manifest"

tagify() {
  local v="$1"
  v="${v//./p}"
  v="${v//-/m}"
  v="${v//+/p}"
  echo "$v"
}

submit_one() {
  local s="$1"
  local c="$2"
  local prof="$3"
  local lr_dual="$4"
  local lr_reco="$5"
  local l_reco="$6"

  local s_tag c_tag lr_d_tag lr_r_tag lr_reco_tag run_name out job_id
  s_tag=$(tagify "$s")
  c_tag=$(tagify "$c")
  lr_d_tag=$(tagify "$lr_dual")
  lr_r_tag=$(tagify "$lr_reco")
  lr_reco_tag=$(tagify "$l_reco")
  run_name="joint_uo_rho090_1MJ100C_s${s_tag}_c${c_tag}_lr${lr_reco_tag}_ld${lr_d_tag}_lr${lr_r_tag}_${prof}"

  out=$(sbatch \
    --export=ALL,\
RUN_NAME="$run_name",\
N_TRAIN_JETS=1000000,\
MAX_CONSTITS=100,\
NUM_WORKERS=6,\
ADDED_TARGET_SCALE="$s",\
SELECTION_METRIC=auc,\
STAGEB_LAMBDA_RANK=0.0,\
STAGEB_LAMBDA_CONS=0.0,\
STAGEC_LR_DUAL="$lr_dual",\
STAGEC_LR_RECO="$lr_reco",\
LAMBDA_RECO="$l_reco",\
LAMBDA_CONS="$c" \
    "$RUNNER")
  job_id="${out##* }"
  echo -e "${job_id}\t${run_name}\t${s}\t${c}\t${l_reco}\t${lr_dual}\t${lr_reco}\t${prof}" >> "$manifest"
  echo "${out} | ${run_name}"
}

echo "Submitting 4 Stage-C jobs for chunk05..."
submit_one 1.00 0.00 A 2e-5 1e-5 0.50
submit_one 1.00 0.00 B 3e-5 1.5e-5 0.60
submit_one 1.00 0.02 A 2e-5 1e-5 0.50
submit_one 1.00 0.02 B 3e-5 1.5e-5 0.60

echo "Done. Manifest: $manifest"
