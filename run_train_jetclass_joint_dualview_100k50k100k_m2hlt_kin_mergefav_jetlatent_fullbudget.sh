#!/usr/bin/env bash
#SBATCH --job-name=jcJLatFB
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=offline_reconstructor_logs/jetclass_joint_dualview_100k50k100k_m2hlt_kin_mergefav_jetlatent_fullbudget_%j.out
#SBATCH --error=offline_reconstructor_logs/jetclass_joint_dualview_100k50k100k_m2hlt_kin_mergefav_jetlatent_fullbudget_%j.err

set -euo pipefail

DATA_DIR="${DATA_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/jetclass_joint_dualview}"
RUN_NAME="${RUN_NAME:-jetclass_joint_v1_100k50k100k_m2hlt_kin_mergefav_jetlatent_fullbudget}"
SEED="${SEED:-52}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-2}"

N_TRAIN_JETS="${N_TRAIN_JETS:-100000}"
N_VAL_JETS="${N_VAL_JETS:-50000}"
N_TEST_JETS="${N_TEST_JETS:-100000}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
FEATURE_MODE="${FEATURE_MODE:-kin}"

# HLT profile knobs
HLT_PT_THRESHOLD="${HLT_PT_THRESHOLD:-1.3}"
MERGE_PROB_SCALE="${MERGE_PROB_SCALE:-1.35}"
REASSIGN_SCALE="${REASSIGN_SCALE:-1.00}"
SMEAR_SCALE="${SMEAR_SCALE:-1.00}"
EFF_PLATEAU_BARREL="${EFF_PLATEAU_BARREL:-0.99}"
EFF_PLATEAU_ENDCAP="${EFF_PLATEAU_ENDCAP:-0.97}"
EFF_TURNON_PT="${EFF_TURNON_PT:-1.4}"
EFF_WIDTH_PT="${EFF_WIDTH_PT:-0.20}"

# Reconstructor/loss knobs
RECO_MAX_GENERATED_TOKENS="${RECO_MAX_GENERATED_TOKENS:-32}"
LOSS_W_BUDGET="${LOSS_W_BUDGET:-0.65}"
LOSS_W_LOCAL="${LOSS_W_LOCAL:-0.10}"
LOSS_GEN_LOCAL_RADIUS="${LOSS_GEN_LOCAL_RADIUS:-0.08}"
LOSS_W_SPARSE="${LOSS_W_SPARSE:-0.02}"
ADDED_TARGET_SCALE="${ADDED_TARGET_SCALE:-1.0}"

# Joint training knobs
LAMBDA_RECO="${LAMBDA_RECO:-0.4}"
LAMBDA_CONS="${LAMBDA_CONS:-0.03}"
STAGEC_LR_DUAL="${STAGEC_LR_DUAL:-1e-5}"
STAGEC_LR_RECO="${STAGEC_LR_RECO:-5e-6}"

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

# Stage-C controls (enabled per user request).
export JETCLASS_STAGEC_PROGRESSIVE_UNFREEZE="${JETCLASS_STAGEC_PROGRESSIVE_UNFREEZE:-1}"
export JETCLASS_STAGEC_UNFREEZE_PHASE1_EPOCHS="${JETCLASS_STAGEC_UNFREEZE_PHASE1_EPOCHS:-3}"
export JETCLASS_STAGEC_UNFREEZE_PHASE2_EPOCHS="${JETCLASS_STAGEC_UNFREEZE_PHASE2_EPOCHS:-7}"
export JETCLASS_STAGEC_UNFREEZE_LAST_N_ENCODER_LAYERS="${JETCLASS_STAGEC_UNFREEZE_LAST_N_ENCODER_LAYERS:-2}"
export JETCLASS_STAGEC_LAMBDA_PARAM_ANCHOR="${JETCLASS_STAGEC_LAMBDA_PARAM_ANCHOR:-0.02}"
export JETCLASS_STAGEC_LAMBDA_OUTPUT_ANCHOR="${JETCLASS_STAGEC_LAMBDA_OUTPUT_ANCHOR:-0.02}"
export JETCLASS_STAGEC_ANCHOR_DECAY="${JETCLASS_STAGEC_ANCHOR_DECAY:-0.97}"
export JETCLASS_STAGEC_RECO_RAMP_EPOCHS="${JETCLASS_STAGEC_RECO_RAMP_EPOCHS:-8}"

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
  python -u train_jetclass_joint_dualview_stage2_unmergeonly_m2hlt_jetlatent_set2set.py
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
  --loss_w_phys 0.0
  --loss_w_pt_ratio 0.0
  --loss_w_m_ratio 0.0
  --loss_w_e_ratio 0.0
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
)

echo "============================================================"
echo "JetClass Joint Dual-View V1 (m2hlt mergefav + jetlatent set2set, full budget)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Split: train=${N_TRAIN_JETS}, val=${N_VAL_JETS}, test=${N_TEST_JETS}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
