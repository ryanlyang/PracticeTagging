#!/usr/bin/env bash
# Reproduce the old 30k/40c setup with jet-level regressor enabled.
# Same as run_offline_reconstructor_joint_dualview_30k40_repro.sh, plus:
# - --enable_jet_regressor
# - writes a compact jet-reg accuracy summary into the run checkpoint folder.
#SBATCH --job-name=offrecoJ30JR
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=offline_reconstructor_logs/offline_reco_joint_30k40_jetreg_%j.out
#SBATCH --error=offline_reconstructor_logs/offline_reco_joint_30k40_jetreg_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_30k_40c_repro_jetreg}"
N_TRAIN_JETS="${N_TRAIN_JETS:-30000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-40}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

echo "Running repro config (jet regressor ON):"
echo "python offline_reconstructor_joint_dualview.py --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --n_train_jets ${N_TRAIN_JETS} --offset_jets ${OFFSET_JETS} --max_constits ${MAX_CONSTITS} --num_workers ${NUM_WORKERS} --enable_jet_regressor --disable_final_kd --device cuda"

python offline_reconstructor_joint_dualview.py \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --n_train_jets "${N_TRAIN_JETS}" \
  --offset_jets "${OFFSET_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --num_workers "${NUM_WORKERS}" \
  --enable_jet_regressor \
  --disable_final_kd \
  --device cuda

RUN_DIR="${SAVE_DIR}/${RUN_NAME}"
JET_JSON="${RUN_DIR}/jet_regression_metrics.json"
SUMMARY_JSON="${RUN_DIR}/jet_regressor_accuracy_summary.json"
SUMMARY_TXT="${RUN_DIR}/jet_regressor_accuracy_summary.txt"

if [ -f "${JET_JSON}" ]; then
  python3 - "${JET_JSON}" "${SUMMARY_JSON}" "${SUMMARY_TXT}" <<'PY'
import json, sys
src, out_json, out_txt = sys.argv[1], sys.argv[2], sys.argv[3]
j = json.load(open(src, "r", encoding="utf-8"))
val = j.get("val", {})
test = j.get("test", {})
summary = {
    "enabled": bool(j.get("enabled", False)),
    "val_mae_pt": val.get("mae_pt"),
    "val_mae_e": val.get("mae_e"),
    "val_mae_m": val.get("mae_m"),
    "val_mae_tau21": val.get("mae_tau21"),
    "val_mae_tau32": val.get("mae_tau32"),
    "val_mae_n_added": val.get("mae_n_added"),
    "test_mae_pt": test.get("mae_pt"),
    "test_mae_e": test.get("mae_e"),
    "test_mae_m": test.get("mae_m"),
    "test_mae_tau21": test.get("mae_tau21"),
    "test_mae_tau32": test.get("mae_tau32"),
    "test_mae_n_added": test.get("mae_n_added"),
}
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
with open(out_txt, "w", encoding="utf-8") as f:
    f.write("Jet Regressor Accuracy Summary\n")
    f.write(f"enabled: {summary['enabled']}\n")
    for k, v in summary.items():
        if k == "enabled":
            continue
        f.write(f"{k}: {v}\n")
print(f"Saved: {out_json}")
print(f"Saved: {out_txt}")
PY
else
  echo "Warning: ${JET_JSON} not found; no accuracy summary generated."
fi

