#!/usr/bin/env bash
#SBATCH -J stgCRank
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o offline_reconstructor_logs/stagec_sweep_lambda_rank_%j.out
#SBATCH -e offline_reconstructor_logs/stagec_sweep_lambda_rank_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_DIR="${RUN_DIR:-checkpoints/offline_reconstructor_joint/joint_100k_80c_stage2save_auc_norankc_nopriv_unmergeonly_rho090_noflags_noconf}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint_stagec_refine_sweep_rank}"
SWEEP_TAG="${SWEEP_TAG:-stagec_sweep_lambda_rank}"

N_TRAIN_JETS="${N_TRAIN_JETS:-100000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-80}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SEED="${SEED:-0}"

STAGEC_EPOCHS="${STAGEC_EPOCHS:-65}"
STAGEC_PATIENCE="${STAGEC_PATIENCE:-14}"
STAGEC_MIN_EPOCHS="${STAGEC_MIN_EPOCHS:-25}"
SELECTION_METRIC="${SELECTION_METRIC:-auc}"
STAGEC_LR_DUAL="${STAGEC_LR_DUAL:-2e-5}"
STAGEC_LR_RECO="${STAGEC_LR_RECO:-1e-5}"
LAMBDA_RECO="${LAMBDA_RECO:-0.1}"
LAMBDA_CONS="${LAMBDA_CONS:-0.1}"
CORRECTED_WEIGHT_FLOOR="${CORRECTED_WEIGHT_FLOOR:-1e-4}"
USE_CORRECTED_FLAGS="${USE_CORRECTED_FLAGS:-0}"

# 8-value sweep for Stage-C rank term.
LAMBDA_RANK_VALUES=(0.005 0.01 0.02 0.03 0.05 0.07 0.1 0.15)

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

sanitize() {
  echo "$1" | sed 's/-/m/g; s/\./p/g; s/+//g'
}

summary_file="offline_reconstructor_logs/stagec_sweep_lambda_rank_summary_${SLURM_JOB_ID}.tsv"
cat > "${summary_file}" <<'TSV'
cfg_idx	run_name	status	stageC_lambda_rank	lambda_reco	lambda_cons	stageC_lr_dual	stageC_lr_reco	stage2_auc	stage2_fpr30	stage2_fpr50	stagec_auc	stagec_fpr30	stagec_fpr50	stagec_bestfpr50_auc	stagec_bestfpr50_fpr30	stagec_bestfpr50_fpr50	baseline_auc	baseline_fpr30	baseline_fpr50	teacher_auc	teacher_fpr30	teacher_fpr50
TSV

total_cfgs=${#LAMBDA_RANK_VALUES[@]}

echo "============================================================"
echo "Stage-C lambda_rank sweep: ${total_cfgs} configs"
echo "Run dir: ${RUN_DIR}"
echo "Save dir: ${SAVE_DIR}"
echo "Sweep tag: ${SWEEP_TAG}"
echo "Summary file: ${summary_file}"
echo "Locked knobs: lambda_reco=${LAMBDA_RECO}, lambda_cons=${LAMBDA_CONS}, lr_dual=${STAGEC_LR_DUAL}, lr_reco=${STAGEC_LR_RECO}"
echo "============================================================"

for i in "${!LAMBDA_RANK_VALUES[@]}"; do
  lambda_rank="${LAMBDA_RANK_VALUES[$i]}"
  run_name="${SWEEP_TAG}_cfg$(printf '%02d' "${i}")_lrank$(sanitize "${lambda_rank}")_lreco$(sanitize "${LAMBDA_RECO}")_lcons$(sanitize "${LAMBDA_CONS}")"
  metrics_path="${SAVE_DIR}/${run_name}/stagec_refine_metrics.json"

  echo
  echo ">>> config ${i}/$((total_cfgs-1))"
  echo "    run_name=${run_name}"
  echo "    stageC_lambda_rank=${lambda_rank}"

  cmd=(
    python finetune_stagec_from_stage2.py
    --run_dir "${RUN_DIR}"
    --save_dir "${SAVE_DIR}"
    --run_name "${run_name}"
    --n_train_jets "${N_TRAIN_JETS}"
    --offset_jets "${OFFSET_JETS}"
    --max_constits "${MAX_CONSTITS}"
    --num_workers "${NUM_WORKERS}"
    --seed "${SEED}"
    --stageC_epochs "${STAGEC_EPOCHS}"
    --stageC_patience "${STAGEC_PATIENCE}"
    --stageC_min_epochs "${STAGEC_MIN_EPOCHS}"
    --stageC_lr_dual "${STAGEC_LR_DUAL}"
    --stageC_lr_reco "${STAGEC_LR_RECO}"
    --stageC_lambda_rank "${lambda_rank}"
    --lambda_reco "${LAMBDA_RECO}"
    --lambda_cons "${LAMBDA_CONS}"
    --selection_metric "${SELECTION_METRIC}"
    --corrected_weight_floor "${CORRECTED_WEIGHT_FLOOR}"
    --device cuda
  )
  if [[ "${USE_CORRECTED_FLAGS}" == "1" ]]; then
    cmd+=(--use_corrected_flags)
  fi

  set +e
  "${cmd[@]}"
  rc=$?
  set -e

  if [[ "${rc}" -eq 0 && -f "${metrics_path}" ]]; then
    metrics_row="$(
      python3 - "${metrics_path}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    m = json.load(f)

def f(d, k):
    if not isinstance(d, dict):
        return "nan"
    v = d.get(k, None)
    if v is None:
        return "nan"
    try:
        return f"{float(v):.6f}"
    except Exception:
        return "nan"

s2 = m.get("test_stage2_loaded", {})
sj = m.get("test_stageC_selected", {})
sf = m.get("test_stageC_bestfpr50", {})
tb = m.get("test_baseline_loaded", {})
tt = m.get("test_teacher_loaded", {})

print("\t".join([
    "ok",
    f(s2, "auc"), f(s2, "fpr30"), f(s2, "fpr50"),
    f(sj, "auc"), f(sj, "fpr30"), f(sj, "fpr50"),
    f(sf, "auc"), f(sf, "fpr30"), f(sf, "fpr50"),
    f(tb, "auc"), f(tb, "fpr30"), f(tb, "fpr50"),
    f(tt, "auc"), f(tt, "fpr30"), f(tt, "fpr50"),
]))
PY
    )"
  else
    metrics_row=$'fail\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan'
  fi

  IFS=$'\t' read -r status \
    stage2_auc stage2_fpr30 stage2_fpr50 \
    stagec_auc stagec_fpr30 stagec_fpr50 \
    stagec_bestfpr50_auc stagec_bestfpr50_fpr30 stagec_bestfpr50_fpr50 \
    baseline_auc baseline_fpr30 baseline_fpr50 \
    teacher_auc teacher_fpr30 teacher_fpr50 \
    <<< "${metrics_row}"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${i}" "${run_name}" "${status}" \
    "${lambda_rank}" "${LAMBDA_RECO}" "${LAMBDA_CONS}" "${STAGEC_LR_DUAL}" "${STAGEC_LR_RECO}" \
    "${stage2_auc}" "${stage2_fpr30}" "${stage2_fpr50}" \
    "${stagec_auc}" "${stagec_fpr30}" "${stagec_fpr50}" \
    "${stagec_bestfpr50_auc}" "${stagec_bestfpr50_fpr30}" "${stagec_bestfpr50_fpr50}" \
    "${baseline_auc}" "${baseline_fpr30}" "${baseline_fpr50}" \
    "${teacher_auc}" "${teacher_fpr30}" "${teacher_fpr50}" \
    >> "${summary_file}"
done

echo
echo "============================================================"
echo "Completed lambda_rank sweep. Summary:"
echo "============================================================"
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "${summary_file}"
else
  cat "${summary_file}"
fi

echo
echo "Top configs by StageC selected FPR@30 (lower is better):"
python3 - "${summary_file}" <<'PY'
import csv
import sys

path = sys.argv[1]
rows = []
with open(path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        if r.get("status") != "ok":
            continue
        try:
            rows.append((float(r["stagec_fpr30"]), r))
        except Exception:
            continue
rows.sort(key=lambda x: x[0])
for rank, (_, r) in enumerate(rows[:5], start=1):
    print(
        f"{rank:>2}. cfg={r['cfg_idx']} run={r['run_name']} "
        f"lrank={r['stageC_lambda_rank']} | stagec_auc={r['stagec_auc']} "
        f"stagec_fpr30={r['stagec_fpr30']} stagec_fpr50={r['stagec_fpr50']}"
    )
PY

echo "Summary written to: ${summary_file}"
