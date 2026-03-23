#!/usr/bin/env bash
# Submit a 32-run Stage-C-focused sweep for:
#   run_offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_rho090_100k80_flags.sh
#
# Grid:
#   ADDED_TARGET_SCALE in {0.90, 0.95, 1.00, 1.05}
#   LAMBDA_CONS in {0.00, 0.02, 0.04, 0.06}
#   PROFILE in {A, B}
#     A: LAMBDA_RECO=0.50, STAGEC_LR_DUAL=2e-5, STAGEC_LR_RECO=1e-5
#     B: LAMBDA_RECO=0.60, STAGEC_LR_DUAL=3e-5, STAGEC_LR_RECO=1.5e-5
#
# Total jobs: 4 * 4 * 2 = 32
#
# Usage:
#   bash submit_stagec_unmergeonly_rho090_32grid.sh

set -euo pipefail

RUNNER="run_offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_rho090_100k80_flags.sh"
if [[ ! -f "$RUNNER" ]]; then
  echo "Runner not found: $RUNNER" >&2
  exit 1
fi

mkdir -p offline_reconstructor_logs

stamp="$(date +%Y%m%d_%H%M%S)"
manifest="offline_reconstructor_logs/stagec_unmergeonly_rho090_32grid_${stamp}.tsv"

echo -e "job_id\trun_name\tadded_target_scale\tlambda_cons\tlambda_reco\tstageC_lr_dual\tstageC_lr_reco\tprofile" > "$manifest"

tagify() {
  local v="$1"
  v="${v//./p}"
  v="${v//-/m}"
  v="${v//+/p}"
  echo "$v"
}

scales=(0.90 0.95 1.00 1.05)
cons_vals=(0.00 0.02 0.04 0.06)
profiles=(A B)

echo "Submitting 32 Stage-C sweep jobs..."

for s in "${scales[@]}"; do
  for c in "${cons_vals[@]}"; do
    for prof in "${profiles[@]}"; do
      if [[ "$prof" == "A" ]]; then
        lr_dual="2e-5"
        lr_reco="1e-5"
        l_reco="0.50"
      else
        lr_dual="3e-5"
        lr_reco="1.5e-5"
        l_reco="0.60"
      fi

      s_tag="$(tagify "$s")"
      c_tag="$(tagify "$c")"
      lr_d_tag="$(tagify "$lr_dual")"
      lr_r_tag="$(tagify "$lr_reco")"
      lr_reco_tag="$(tagify "$l_reco")"

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

      # sbatch output format: "Submitted batch job <id>"
      job_id="${out##* }"
      echo -e "${job_id}\t${run_name}\t${s}\t${c}\t${l_reco}\t${lr_dual}\t${lr_reco}\t${prof}" >> "$manifest"
      echo "${out} | ${run_name}"
    done
  done
done

echo ""
echo "Submitted all jobs."
echo "Manifest: $manifest"
