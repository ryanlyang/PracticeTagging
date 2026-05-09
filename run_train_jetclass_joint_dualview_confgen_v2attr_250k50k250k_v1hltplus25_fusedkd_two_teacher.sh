#!/usr/bin/env bash
#SBATCH --job-name=jcFk2Q
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=offline_reconstructor_logs/jetclass_joint_dualview_confgen_v2attr_250k50k250k_v1hltplus25_fusedkd_two_teacher_%j.out
#SBATCH --error=offline_reconstructor_logs/jetclass_joint_dualview_confgen_v2attr_250k50k250k_v1hltplus25_fusedkd_two_teacher_%j.err

set -euo pipefail

DATA_DIR="${DATA_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/jetclass_joint_dualview}"
RUN_NAME="${RUN_NAME:-jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_fusedkd_two_teacher}"
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

# Teacher run directories (two frozen experts to fuse).
TEACHER_A_RUN="${TEACHER_A_RUN:-checkpoints/jetclass_joint_dualview/jetclass_joint_confgen_v2attr_250k50k250k_stronger_canonical_path_gentok56_ablate_lcons003_recoonlydual}"
TEACHER_B_RUN="${TEACHER_B_RUN:-checkpoints/jetclass_joint_dualview/jetclass_joint_confgen_v2attr_250k50k250k_stronger_canonical_v1hlt_hltplus25_gentok56}"
# Optional: initialize student from an existing run.
DISTILL_INIT_RUN="${DISTILL_INIT_RUN:-}"

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

# Use v1/default HLT builder path with +25% stronger corruption numbers below.
export JETCLASS_HLT_MODE="${JETCLASS_HLT_MODE:-v1}"
# Keep generator lightly penalized in Stage-A.
export JETCLASS_STAGEA_W_SPARSE_SPLIT="${JETCLASS_STAGEA_W_SPARSE_SPLIT:-0.012}"
export JETCLASS_STAGEA_W_SPARSE_GEN="${JETCLASS_STAGEA_W_SPARSE_GEN:-0.0015}"
export JETCLASS_STAGEA_W_GEN_FP="${JETCLASS_STAGEA_W_GEN_FP:-0.04}"

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
  --enable_fused_kd_distill
  --distill_teacher_run_dir_a "${TEACHER_A_RUN}"
  --distill_teacher_run_dir_b "${TEACHER_B_RUN}"
  --distill_teacher_weight_a 0.50
  --distill_temp 2.5
  --distill_alpha_kl 1.0
  --distill_alpha_ce 0.25
  --distill_phase1_epochs 20
  --distill_phase1_patience 6
  --distill_phase1_min_epochs 5
  --distill_phase1_lr_dual 3e-4
  --distill_phase2_epochs 30
  --distill_phase2_patience 8
  --distill_phase2_min_epochs 10
  --distill_phase2_lr_dual 2e-4
  --distill_phase2_lr_reco 7e-5
  --distill_phase2_lambda_reco 0.20
  --distill_phase2_lambda_cons 0.03
  --distill_phase2_lambda_attr_mode 0.04
  --distill_phase2_lambda_attr_type 0.06
  --distill_phase2_lambda_attr_charge 0.01
  --distill_phase2_lambda_attr_track 0.01
  --train_reco_only_after_stageA
  --reco_only_epochs 60
  --reco_only_patience 15
  --reco_only_lr 4e-4
  --reco_only_warmup_epochs 3
  --reco_only_batch_size 512
  --stageC_epochs 0
  --stageC_patience 2
  --stageC_min_epochs 0
  --stageC_lr_dual 2e-4
  --stageC_lr_reco 1e-4
  --lambda_reco 0.4
  --lambda_cons 0.06
  --added_target_scale 0.90
)

if [[ -n "${DISTILL_INIT_RUN}" ]]; then
  CMD+=( --distill_init_run_dir "${DISTILL_INIT_RUN}" )
fi

echo "============================================================"
echo "JetClass fused-two-teacher KD distillation (v1 HLT +25%)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Teacher A: ${TEACHER_A_RUN}"
echo "Teacher B: ${TEACHER_B_RUN}"
echo "Split: train=${N_TRAIN_JETS}, val=${N_VAL_JETS}, test=${N_TEST_JETS}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
