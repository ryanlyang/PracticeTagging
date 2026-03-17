#!/usr/bin/env bash
#SBATCH -J stgCswpS
#SBATCH -p tier3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH -o offline_reconstructor_logs/stagec_sweep_small_%j.out
#SBATCH -e offline_reconstructor_logs/stagec_sweep_small_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_DIR="${RUN_DIR:-checkpoints/offline_reconstructor_joint/joint_100k_80c_stage2save_auc_norankc_nopriv_unmergeonly_rho090_noflags_noconf}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint_stagec_refine_sweep_small}"
SWEEP_TAG="${SWEEP_TAG:-stagec_sweep_small_reco_low}"

N_TRAIN_JETS="${N_TRAIN_JETS:-100000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-80}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SEED="${SEED:-0}"

STAGEC_EPOCHS="${STAGEC_EPOCHS:-65}"
STAGEC_PATIENCE="${STAGEC_PATIENCE:-14}"
STAGEC_MIN_EPOCHS="${STAGEC_MIN_EPOCHS:-25}"
SELECTION_METRIC="${SELECTION_METRIC:-auc}"
CORRECTED_WEIGHT_FLOOR="${CORRECTED_WEIGHT_FLOOR:-1e-4}"
USE_CORRECTED_FLAGS="${USE_CORRECTED_FLAGS:-0}"

# Requested sweep grid
LAMBDA_RECO_VALUES=(0.0 0.02 0.05 0.08)
LAMBDA_CONS_VALUES=(0.06)
LR_DUAL_VALUES=(1e-5 2e-5 3e-5)
LR_RECO_VALUES=(5e-6 1e-5 1.5e-5)

if [[ "${#LR_DUAL_VALUES[@]}" -ne "${#LR_RECO_VALUES[@]}" ]]; then
  echo "Error: LR_DUAL_VALUES and LR_RECO_VALUES must have same length"
  exit 2
fi

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

summary_file="offline_reconstructor_logs/stagec_sweep_small_summary_${SLURM_JOB_ID}.tsv"
cat > "${summary_file}" <<'TSV'
cfg_idx	run_name	status	lambda_reco	lambda_cons	stageC_lr_dual	stageC_lr_reco	stage2_auc	stage2_fpr30	stage2_fpr50	stagec_auc	stagec_fpr30	stagec_fpr50	stagec_bestfpr50_auc	stagec_bestfpr50_fpr30	stagec_bestfpr50_fpr50	baseline_auc	baseline_fpr30	baseline_fpr50	teacher_auc	teacher_fpr30	teacher_fpr50
TSV

cfg_idx=0
total_cfgs=$(( ${#LAMBDA_RECO_VALUES[@]} * ${#LAMBDA_CONS_VALUES[@]} * ${#LR_DUAL_VALUES[@]} ))

echo "============================================================"
echo "Stage-C small sweep: ${total_cfgs} configs"
echo "Run dir: ${RUN_DIR}"
echo "Save dir: ${SAVE_DIR}"
echo "Sweep tag: ${SWEEP_TAG}"
echo "Summary file: ${summary_file}"
echo "============================================================"

for lambda_reco in "${LAMBDA_RECO_VALUES[@]}"; do
  for lambda_cons in "${LAMBDA_CONS_VALUES[@]}"; do
    for k in "${!LR_DUAL_VALUES[@]}"; do
      stagec_lr_dual="${LR_DUAL_VALUES[$k]}"
      stagec_lr_reco="${LR_RECO_VALUES[$k]}"

      run_name="${SWEEP_TAG}_cfg$(printf '%02d' "${cfg_idx}")_lreco$(sanitize "${lambda_reco}")_lcons$(sanitize "${lambda_cons}")_lrd$(sanitize "${stagec_lr_dual}")_lrr$(sanitize "${stagec_lr_reco}")"
      metrics_path="${SAVE_DIR}/${run_name}/joint_stage_metrics.json"

      echo
      echo ">>> config ${cfg_idx}/$((total_cfgs-1))"
      echo "    run_name=${run_name}"
      echo "    lambda_reco=${lambda_reco} lambda_cons=${lambda_cons} lr_dual=${stagec_lr_dual} lr_reco=${stagec_lr_reco}"

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
        --stageC_lr_dual "${stagec_lr_dual}"
        --stageC_lr_reco "${stagec_lr_reco}"
        --lambda_reco "${lambda_reco}"
        --lambda_cons "${lambda_cons}"
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

      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${cfg_idx}" "${run_name}" "${status}" \
        "${lambda_reco}" "${lambda_cons}" "${stagec_lr_dual}" "${stagec_lr_reco}" \
        "${stage2_auc}" "${stage2_fpr30}" "${stage2_fpr50}" \
        "${stagec_auc}" "${stagec_fpr30}" "${stagec_fpr50}" \
        "${stagec_bestfpr50_auc}" "${stagec_bestfpr50_fpr30}" "${stagec_bestfpr50_fpr50}" \
        "${baseline_auc}" "${baseline_fpr30}" "${baseline_fpr50}" \
        "${teacher_auc}" "${teacher_fpr30}" "${teacher_fpr50}" \
        >> "${summary_file}"

      cfg_idx=$((cfg_idx + 1))
    done
  done
done

echo
echo "============================================================"
echo "Completed small sweep. Summary:"
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
        f"lreco={r['lambda_reco']} lcons={r['lambda_cons']} "
        f"lrd={r['stageC_lr_dual']} lrr={r['stageC_lr_reco']} "
        f"| stagec_auc={r['stagec_auc']} stagec_fpr30={r['stagec_fpr30']} stagec_fpr50={r['stagec_fpr50']}"
    )
PY

echo "Summary written to: ${summary_file}"
