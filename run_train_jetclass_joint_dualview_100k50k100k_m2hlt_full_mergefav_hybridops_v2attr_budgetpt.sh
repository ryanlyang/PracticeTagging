#!/usr/bin/env bash
#SBATCH --job-name=jcHybV2F
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=offline_reconstructor_logs/jetclass_joint_dualview_100k50k100k_m2hlt_full_mergefav_hybridops_v2attr_budgetpt_%j.out
#SBATCH --error=offline_reconstructor_logs/jetclass_joint_dualview_100k50k100k_m2hlt_full_mergefav_hybridops_v2attr_budgetpt_%j.err

set -euo pipefail

DATA_DIR="${DATA_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/jetclass_joint_dualview}"
RUN_NAME="${RUN_NAME:-jetclass_joint_v2attr_100k50k100k_m2hlt_full_mergefav_hybridops_budgetpt}"
SEED="${SEED:-52}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-1}"

N_TRAIN_JETS="${N_TRAIN_JETS:-50000}"
N_VAL_JETS="${N_VAL_JETS:-20000}"
N_TEST_JETS="${N_TEST_JETS:-100000}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
FEATURE_MODE="${FEATURE_MODE:-full}"

# HLT profile knobs (m2hlt merge-favoring profile).
HLT_PT_THRESHOLD="${HLT_PT_THRESHOLD:-1.3}"
MERGE_PROB_SCALE="${MERGE_PROB_SCALE:-1.35}"
REASSIGN_SCALE="${REASSIGN_SCALE:-1.00}"
SMEAR_SCALE="${SMEAR_SCALE:-1.00}"
EFF_PLATEAU_BARREL="${EFF_PLATEAU_BARREL:-0.99}"
EFF_PLATEAU_ENDCAP="${EFF_PLATEAU_ENDCAP:-0.97}"
EFF_TURNON_PT="${EFF_TURNON_PT:-1.4}"
EFF_WIDTH_PT="${EFF_WIDTH_PT:-0.20}"

# Loss knobs requested: count/budget + jet-pt strong; mass/energy small.
RECO_MAX_GENERATED_TOKENS="${RECO_MAX_GENERATED_TOKENS:-32}"
LOSS_W_BUDGET="${LOSS_W_BUDGET:-0.80}"
LOSS_W_PT_RATIO="${LOSS_W_PT_RATIO:-0.25}"
LOSS_W_M_RATIO="${LOSS_W_M_RATIO:-0.03}"
LOSS_W_E_RATIO="${LOSS_W_E_RATIO:-0.03}"
LOSS_W_PHYS="${LOSS_W_PHYS:-0.00}"
LOSS_W_LOCAL="${LOSS_W_LOCAL:-0.10}"
LOSS_GEN_LOCAL_RADIUS="${LOSS_GEN_LOCAL_RADIUS:-0.08}"
LOSS_W_SPARSE="${LOSS_W_SPARSE:-0.02}"
ADDED_TARGET_SCALE="${ADDED_TARGET_SCALE:-1.0}"

# Joint training knobs.
LAMBDA_RECO="${LAMBDA_RECO:-0.4}"
LAMBDA_CONS="${LAMBDA_CONS:-0.03}"
STAGEC_LR_DUAL="${STAGEC_LR_DUAL:-1e-5}"
STAGEC_LR_RECO="${STAGEC_LR_RECO:-5e-6}"

# V2 attr-head knobs.
V2_ATTR_HIDDEN_DIM="${V2_ATTR_HIDDEN_DIM:-128}"
V2_ATTR_SLOTS="${V2_ATTR_SLOTS:-2}"
V2_MODE_NONE_WEIGHT="${V2_MODE_NONE_WEIGHT:-0.20}"
V2_MODE_LABEL_SMOOTHING="${V2_MODE_LABEL_SMOOTHING:-0.0}"
V2_TRACK_WEIGHT="${V2_TRACK_WEIGHT:-1.0}"
STAGEA_ATTR_EPOCHS="${STAGEA_ATTR_EPOCHS:-12}"
STAGEA_ATTR_PATIENCE="${STAGEA_ATTR_PATIENCE:-4}"
STAGEA_ATTR_LR="${STAGEA_ATTR_LR:-2e-4}"
STAGEA_ATTR_WEIGHT_DECAY="${STAGEA_ATTR_WEIGHT_DECAY:-1e-5}"
LAMBDA_ATTR_MODE="${LAMBDA_ATTR_MODE:-0.10}"
LAMBDA_ATTR_TYPE="${LAMBDA_ATTR_TYPE:-0.15}"
LAMBDA_ATTR_CHARGE="${LAMBDA_ATTR_CHARGE:-0.03}"
LAMBDA_ATTR_TRACK="${LAMBDA_ATTR_TRACK:-0.03}"

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
export PYTHONDONTWRITEBYTECODE=1

SCRIPT="${SCRIPT:-$(pwd)/train_jetclass_joint_dualview_stage2_unmergeonly_m2hlt_hybrid_ops_v2_attr.py}"

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
  python -u "${SCRIPT}"
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
  --hlt_pt_threshold "${HLT_PT_THRESHOLD}"
  --merge_prob_scale "${MERGE_PROB_SCALE}"
  --reassign_scale "${REASSIGN_SCALE}"
  --smear_scale "${SMEAR_SCALE}"
  --eff_plateau_barrel "${EFF_PLATEAU_BARREL}"
  --eff_plateau_endcap "${EFF_PLATEAU_ENDCAP}"
  --eff_turnon_pt "${EFF_TURNON_PT}"
  --eff_width_pt "${EFF_WIDTH_PT}"
  --reco_batch_size 96
  --stageA_epochs 90
  --stageA_patience 18
  --stageA_lr 2e-4
  --stageA_weight_decay 1e-5
  --stageA_warmup_epochs 5
  --stageA_stage1_epochs 20
  --stageA_stage2_epochs 55
  --stageA_min_full_scale_epochs 5
  --reco_max_generated_tokens "${RECO_MAX_GENERATED_TOKENS}"
  --loss_set_mode hungarian
  --loss_w_set 1.0
  --loss_w_phys "${LOSS_W_PHYS}"
  --loss_w_pt_ratio "${LOSS_W_PT_RATIO}"
  --loss_w_m_ratio "${LOSS_W_M_RATIO}"
  --loss_w_e_ratio "${LOSS_W_E_RATIO}"
  --loss_w_budget "${LOSS_W_BUDGET}"
  --loss_w_sparse "${LOSS_W_SPARSE}"
  --loss_w_local "${LOSS_W_LOCAL}"
  --loss_gen_local_radius "${LOSS_GEN_LOCAL_RADIUS}"
  --stageB_epochs 35
  --stageB_patience 10
  --stageB_min_epochs 10
  --stageB_lr_dual 4e-4
  --stageC_epochs 45
  --stageC_patience 12
  --stageC_min_epochs 15
  --stageC_lr_dual "${STAGEC_LR_DUAL}"
  --stageC_lr_reco "${STAGEC_LR_RECO}"
  --lambda_reco "${LAMBDA_RECO}"
  --lambda_cons "${LAMBDA_CONS}"
  --added_target_scale "${ADDED_TARGET_SCALE}"
  --v2_attr_hidden_dim "${V2_ATTR_HIDDEN_DIM}"
  --v2_attr_slots "${V2_ATTR_SLOTS}"
  --v2_mode_none_weight "${V2_MODE_NONE_WEIGHT}"
  --v2_mode_label_smoothing "${V2_MODE_LABEL_SMOOTHING}"
  --v2_track_weight "${V2_TRACK_WEIGHT}"
  --stageA_attr_epochs "${STAGEA_ATTR_EPOCHS}"
  --stageA_attr_patience "${STAGEA_ATTR_PATIENCE}"
  --stageA_attr_lr "${STAGEA_ATTR_LR}"
  --stageA_attr_weight_decay "${STAGEA_ATTR_WEIGHT_DECAY}"
  --lambda_attr_mode "${LAMBDA_ATTR_MODE}"
  --lambda_attr_type "${LAMBDA_ATTR_TYPE}"
  --lambda_attr_charge "${LAMBDA_ATTR_CHARGE}"
  --lambda_attr_track "${LAMBDA_ATTR_TRACK}"
)

echo "============================================================"
echo "JetClass Joint Dual-View V2 Attr (m2hlt mergefav + hybrid ops, full in/out)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Split: train=${N_TRAIN_JETS}, val=${N_VAL_JETS}, test=${N_TEST_JETS}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
