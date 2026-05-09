#!/usr/bin/env bash
#SBATCH --job-name=jcCfV1H25L
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=offline_reconstructor_logs/jetclass_joint_dualview_confgen_v2attr_250k50k250k_stronger_canonical_v1hlt_hltplus25_%j.out
#SBATCH --error=offline_reconstructor_logs/jetclass_joint_dualview_confgen_v2attr_250k50k250k_stronger_canonical_v1hlt_hltplus25_%j.err

set -euo pipefail

DATA_DIR="${DATA_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/jetclass_joint_dualview}"
RUN_NAME="${RUN_NAME:-jetclass_joint_confgen_v2attr_250k50k250k_stronger_canonical_v1hlt_hltplus25_gentok56}"
SEED="${SEED:-52}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"

N_TRAIN_JETS="${N_TRAIN_JETS:-250000}"
N_VAL_JETS="${N_VAL_JETS:-50000}"
N_TEST_JETS="${N_TEST_JETS:-250000}"
MAX_CONSTITS="${MAX_CONSTITS:-128}"
FEATURE_MODE="${FEATURE_MODE:-full}"
FEATURE_PREPROCESSING="${FEATURE_PREPROCESSING:-canonical}"
CLASS_ASSIGNMENT="${CLASS_ASSIGNMENT:-canonical_labels}"
TARGET_CLASS="${TARGET_CLASS:-Hbb}"
BACKGROUND_CLASS="${BACKGROUND_CLASS:-QCD}"
STAGEC_EPOCHS="${STAGEC_EPOCHS:-0}"
STAGEC_PATIENCE="${STAGEC_PATIENCE:-2}"
STAGEC_MIN_EPOCHS="${STAGEC_MIN_EPOCHS:-0}"
STAGEC_LR_DUAL="${STAGEC_LR_DUAL:-2e-4}"
STAGEC_LR_RECO="${STAGEC_LR_RECO:-1e-4}"

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

export JETCLASS_STAGEA_W_SPARSE_SPLIT="${JETCLASS_STAGEA_W_SPARSE_SPLIT:-0.012}"
export JETCLASS_STAGEA_W_SPARSE_GEN="${JETCLASS_STAGEA_W_SPARSE_GEN:-0.003}"
export JETCLASS_STAGEA_W_GEN_FP="${JETCLASS_STAGEA_W_GEN_FP:-0.04}"
export JETCLASS_HLT_MODE="${JETCLASS_HLT_MODE:-v1}"

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
  python -u train_jetclass_joint_dualview_stage2_unmergeonly_m2hlt_confgen_ops_v2_attr.py
  --data_dir "${DATA_DIR}"
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"
  --feature_mode "${FEATURE_MODE}"
  --feature_preprocessing "${FEATURE_PREPROCESSING}"
  --class_assignment "${CLASS_ASSIGNMENT}"
  --max_constits "${MAX_CONSTITS}"
  --train_files_per_class 8
  --val_files_per_class 1
  --test_files_per_class 1
  --n_train_jets "${N_TRAIN_JETS}"
  --n_val_jets "${N_VAL_JETS}"
  --n_test_jets "${N_TEST_JETS}"
  --batch_size 512
  --epochs 60
  --patience 12
  --lr 7e-4
  --weight_decay 1e-5
  --warmup_epochs 3
  --embed_dim 128
  --num_heads 8
  --num_layers 6
  --ff_dim 512
  --dropout 0.1
  --target_class "${TARGET_CLASS}"
  --background_class "${BACKGROUND_CLASS}"
  --hlt_pt_threshold 1.875
  --merge_prob_scale 1.50
  --reassign_scale 1.56
  --smear_scale 1.56
  --eff_plateau_barrel 0.9375
  --eff_plateau_endcap 0.85
  --eff_turnon_pt 1.5
  --eff_width_pt 0.5625
  --reco_batch_size 96
  --stageA_epochs 90
  --stageA_patience 18
  --stageA_lr 2e-4
  --stageA_weight_decay 1e-5
  --stageA_warmup_epochs 5
  --stageA_stage1_epochs 20
  --stageA_stage2_epochs 55
  --stageA_min_full_scale_epochs 5
  --reco_max_generated_tokens 56
  --stageA_attr_epochs 12
  --stageA_attr_patience 4
  --stageA_attr_lr 2e-4
  --stageA_attr_weight_decay 1e-5
  --v2_attr_hidden_dim 128
  --v2_attr_slots 2
  --v2_mode_none_weight 0.20
  --lambda_attr_mode 0.10
  --lambda_attr_type 0.15
  --lambda_attr_charge 0.03
  --lambda_attr_track 0.03
  --loss_set_mode hungarian
  --loss_w_set 1.0
  --loss_w_phys 0.0
  --loss_w_pt_ratio 0.0
  --loss_w_m_ratio 0.0
  --loss_w_e_ratio 0.0
  --loss_w_budget 0.65
  --loss_w_sparse 0.012
  --loss_w_local 0.06
  --loss_gen_local_radius 0.08
  --stageB_epochs 60
  --stageB_patience 15
  --stageB_min_epochs 10
  --stageB_lr_dual 4e-4
  --stageC_epochs "${STAGEC_EPOCHS}"
  --stageC_patience "${STAGEC_PATIENCE}"
  --stageC_min_epochs "${STAGEC_MIN_EPOCHS}"
  --stageC_lr_dual "${STAGEC_LR_DUAL}"
  --stageC_lr_reco "${STAGEC_LR_RECO}"
  --lambda_reco 0.4
  --lambda_cons 0.06
  --added_target_scale 0.90
)

echo "============================================================"
echo "JetClass Joint Dual-View V2 (confgen reconstructor + v1/default HLT corruption)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Split: train=${N_TRAIN_JETS}, val=${N_VAL_JETS}, test=${N_TEST_JETS}"
echo "Class assignment: ${CLASS_ASSIGNMENT}"
echo "Feature preprocessing: ${FEATURE_PREPROCESSING}"
echo "Target/background: ${TARGET_CLASS} vs ${BACKGROUND_CLASS}"
echo "HLT profile: v1 +25% stronger corruption"
echo "StageC epochs: ${STAGEC_EPOCHS} (set to 0 for prejoint-only)"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"

