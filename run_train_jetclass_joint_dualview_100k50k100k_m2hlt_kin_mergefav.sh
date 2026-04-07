#!/usr/bin/env bash
#SBATCH --job-name=jcJv1M2M
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=offline_reconstructor_logs/jetclass_joint_dualview_100k50k100k_m2hlt_kin_mergefav_%j.out
#SBATCH --error=offline_reconstructor_logs/jetclass_joint_dualview_100k50k100k_m2hlt_kin_mergefav_%j.err

set -euo pipefail

DATA_DIR="${DATA_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/jetclass_joint_dualview}"
RUN_NAME="${RUN_NAME:-jetclass_joint_v1_100k50k100k_m2hlt_kin_mergefav}"
SEED="${SEED:-52}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-2}"

N_TRAIN_JETS="${N_TRAIN_JETS:-100000}"
N_VAL_JETS="${N_VAL_JETS:-50000}"
N_TEST_JETS="${N_TEST_JETS:-100000}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
FEATURE_MODE="${FEATURE_MODE:-kin}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p offline_reconstructor_logs
mkdir -p "${SAVE_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONHASHSEED="${SEED}"

python - <<'PY'
import importlib.util
missing = [m for m in ("awkward", "uproot") if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(
        "[preflight] Missing modules: "
        + ", ".join(missing)
        + ". Install in env (e.g. python -m pip install --user weaver-core)."
    )
PY

CMD=(
  python -u train_jetclass_joint_dualview_stage2_unmergeonly_m2hlt.py
  --data_dir "${DATA_DIR}"
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"
  --feature_mode "${FEATURE_MODE}"
  --max_constits "${MAX_CONSTITS}"
  --train_files_per_class 8
  --val_files_per_class 1
  --test_files_per_class 1
  --n_train_jets "${N_TRAIN_JETS}"
  --n_val_jets "${N_VAL_JETS}"
  --n_test_jets "${N_TEST_JETS}"
  --batch_size 512
  --epochs 30
  --patience 8
  --lr 7e-4
  --weight_decay 1e-5
  --warmup_epochs 3
  --embed_dim 128
  --num_heads 8
  --num_layers 6
  --ff_dim 512
  --dropout 0.1
  --target_class HToBB
  --background_class ZJetsToNuNu
  --hlt_pt_threshold 1.3
  --merge_prob_scale 1.35
  --reassign_scale 1.00
  --smear_scale 1.00
  --eff_plateau_barrel 0.99
  --eff_plateau_endcap 0.97
  --eff_turnon_pt 1.4
  --eff_width_pt 0.20
  --reco_batch_size 96
  --stageA_epochs 90
  --stageA_patience 18
  --stageA_lr 2e-4
  --stageA_weight_decay 1e-5
  --stageA_warmup_epochs 5
  --stageA_stage1_epochs 20
  --stageA_stage2_epochs 55
  --stageA_min_full_scale_epochs 5
  --loss_set_mode hungarian
  --loss_w_set 1.0
  --loss_w_phys 0.0
  --loss_w_pt_ratio 0.0
  --loss_w_m_ratio 0.0
  --loss_w_e_ratio 0.0
  --loss_w_budget 0.65
  --loss_w_sparse 0.02
  --loss_w_local 0.05
  --stageB_epochs 35
  --stageB_patience 10
  --stageB_min_epochs 10
  --stageB_lr_dual 4e-4
  --stageC_epochs 45
  --stageC_patience 12
  --stageC_min_epochs 15
  --stageC_lr_dual 2e-4
  --stageC_lr_reco 1e-4
  --lambda_reco 0.4
  --lambda_cons 0.06
  --added_target_scale 0.90
)

echo "============================================================"
echo "JetClass Joint Dual-View V1 (m2hlt, kin-only, merge-favoring HLT)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Split: train=${N_TRAIN_JETS}, val=${N_VAL_JETS}, test=${N_TEST_JETS}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"

