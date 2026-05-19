#!/usr/bin/env bash
set -euo pipefail

ROOT="sbatch/reco_teacher_joint_fusion_6model_150k75k150k"
BASE="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k"

RUN_M12="${ROOT}/run_m12_dualreco_dualview_feat_noscale_weighted_5m1m1m.sh"
RUN_M15MID="${ROOT}/run_m15_dualreco_dualview_offdrop_mid_weighted_5m1m1m.sh"
RUN_M15HIGH="${ROOT}/run_m15_dualreco_dualview_offdrop_high_weighted_5m1m1m.sh"
RUN_M16="${ROOT}/run_m16_dualreco_dualview_topk60_weighted_5m1m1m.sh"
RUN_M17="${ROOT}/run_m17_dualreco_dualview_antioverlap_weighted_5m1m1m.sh"
RUN_ANALYZE="${ROOT}/run_analyze_hlt_joint12_bin_gated_fusion_valsel_weighted_5m1m1m.sh"

RECO_TIME="${RECO_TIME:-10-00:00:00}"
TAGGER_TIME="${TAGGER_TIME:-10-00:00:00}"
NUM_WORKERS_RECO="${NUM_WORKERS_RECO:-1}"
NUM_WORKERS_TAGGER="${NUM_WORKERS_TAGGER:-1}"
QUEUE_ANALYZE="${QUEUE_ANALYZE:-1}"
BASE_DEPS="${BASE_DEPS:-}"
INCLUDE_HLT_CANDIDATE="${INCLUDE_HLT_CANDIDATE:-1}"
STEP1_REF_NPZ="${STEP1_REF_NPZ:-}"

# STEP1 load dirs for each model pipeline.
M15MID_STEP1_DIR="${M15MID_STEP1_DIR:-${BASE}/model15_dualreco_dualview_offdrop_mid_weighted_5m1m1m/model15_dualreco_dualview_offdrop_mid_weighted_5m1m1m_seed0}"
M15HIGH_STEP1_DIR="${M15HIGH_STEP1_DIR:-${BASE}/model15_dualreco_dualview_offdrop_high_weighted_5m1m1m/model15_dualreco_dualview_offdrop_high_weighted_5m1m1m_seed0}"
M12_STEP1_DIR="${M12_STEP1_DIR:-${BASE}/model12_dualreco_dualview_feat_noscale_weighted_5m1m1m/model12_dualreco_dualview_feat_noscale_weighted_5m1m1m_seed0}"
M16_STEP1_DIR="${M16_STEP1_DIR:-${BASE}/model16_dualreco_dualview_topk60_weighted_5m1m1m/model16_dualreco_dualview_topk60_weighted_5m1m1m_seed0}"
M17_STEP1_DIR="${M17_STEP1_DIR:-${BASE}/model17_dualreco_dualview_antioverlap_weighted_5m1m1m/model17_dualreco_dualview_antioverlap_weighted_5m1m1m_seed0}"

# ---------- M15 MID (submit first) ----------
M15MID_A_SAVE="${BASE}/model15_dualreco_dualview_offdrop_mid_weighted_5m1m1m_recoAonly"
M15MID_A_RUN="model15_dualreco_dualview_offdrop_mid_weighted_5m1m1m_seed0_recoAonly"
M15MID_B_SAVE="${BASE}/model15_dualreco_dualview_offdrop_mid_weighted_5m1m1m_recoBonly"
M15MID_B_RUN="model15_dualreco_dualview_offdrop_mid_weighted_5m1m1m_seed0_recoBonly"
M15MID_T_SAVE="${BASE}/model15_dualreco_dualview_offdrop_mid_weighted_5m1m1m_from_splitAB"
M15MID_T_RUN="model15_dualreco_dualview_offdrop_mid_weighted_5m1m1m_seed0_from_splitAB"

j15m_a=$(sbatch --parsable --time="${RECO_TIME}" --export="ALL,RUN_NAME=${M15MID_A_RUN},SAVE_DIR=${M15MID_A_SAVE},STEP1_LOAD_DIR=${M15MID_STEP1_DIR},NUM_WORKERS=${NUM_WORKERS_RECO},TRAIN_ONLY_RECOA=1,TRAIN_ONLY_RECOB=0,STOP_AFTER_RECO_PRETRAIN=0" "${RUN_M15MID}")
j15m_b=$(sbatch --parsable --time="${RECO_TIME}" --export="ALL,RUN_NAME=${M15MID_B_RUN},SAVE_DIR=${M15MID_B_SAVE},STEP1_LOAD_DIR=${M15MID_STEP1_DIR},NUM_WORKERS=${NUM_WORKERS_RECO},TRAIN_ONLY_RECOA=0,TRAIN_ONLY_RECOB=1,STOP_AFTER_RECO_PRETRAIN=0" "${RUN_M15MID}")
j15m_t=$(sbatch --parsable --time="${TAGGER_TIME}" --dependency="afterok:${j15m_a}:${j15m_b}" --export="ALL,RUN_NAME=${M15MID_T_RUN},SAVE_DIR=${M15MID_T_SAVE},STEP1_LOAD_DIR=${M15MID_A_SAVE}/${M15MID_A_RUN},NUM_WORKERS=${NUM_WORKERS_TAGGER},LOAD_RECO_A_CKPT=${M15MID_A_SAVE}/${M15MID_A_RUN}/offline_reconstructor_A_stageA.pt,LOAD_RECO_B_CKPT=${M15MID_B_SAVE}/${M15MID_B_RUN}/offline_reconstructor_B_stageA.pt,TRAIN_ONLY_RECOA=0,TRAIN_ONLY_RECOB=0,STOP_AFTER_RECO_PRETRAIN=0" "${RUN_M15MID}")

# ---------- M15 HIGH (submit second) ----------
M15HIGH_A_SAVE="${BASE}/model15_dualreco_dualview_offdrop_high_weighted_5m1m1m_recoAonly"
M15HIGH_A_RUN="model15_dualreco_dualview_offdrop_high_weighted_5m1m1m_seed0_recoAonly"
M15HIGH_B_SAVE="${BASE}/model15_dualreco_dualview_offdrop_high_weighted_5m1m1m_recoBonly"
M15HIGH_B_RUN="model15_dualreco_dualview_offdrop_high_weighted_5m1m1m_seed0_recoBonly"
M15HIGH_T_SAVE="${BASE}/model15_dualreco_dualview_offdrop_high_weighted_5m1m1m_from_splitAB"
M15HIGH_T_RUN="model15_dualreco_dualview_offdrop_high_weighted_5m1m1m_seed0_from_splitAB"

j15h_a=$(sbatch --parsable --time="${RECO_TIME}" --export="ALL,RUN_NAME=${M15HIGH_A_RUN},SAVE_DIR=${M15HIGH_A_SAVE},STEP1_LOAD_DIR=${M15HIGH_STEP1_DIR},NUM_WORKERS=${NUM_WORKERS_RECO},TRAIN_ONLY_RECOA=1,TRAIN_ONLY_RECOB=0,STOP_AFTER_RECO_PRETRAIN=0" "${RUN_M15HIGH}")
j15h_b=$(sbatch --parsable --time="${RECO_TIME}" --export="ALL,RUN_NAME=${M15HIGH_B_RUN},SAVE_DIR=${M15HIGH_B_SAVE},STEP1_LOAD_DIR=${M15HIGH_STEP1_DIR},NUM_WORKERS=${NUM_WORKERS_RECO},TRAIN_ONLY_RECOA=0,TRAIN_ONLY_RECOB=1,STOP_AFTER_RECO_PRETRAIN=0" "${RUN_M15HIGH}")
j15h_t=$(sbatch --parsable --time="${TAGGER_TIME}" --dependency="afterok:${j15h_a}:${j15h_b}" --export="ALL,RUN_NAME=${M15HIGH_T_RUN},SAVE_DIR=${M15HIGH_T_SAVE},STEP1_LOAD_DIR=${M15HIGH_A_SAVE}/${M15HIGH_A_RUN},NUM_WORKERS=${NUM_WORKERS_TAGGER},LOAD_RECO_A_CKPT=${M15HIGH_A_SAVE}/${M15HIGH_A_RUN}/offline_reconstructor_A_stageA.pt,LOAD_RECO_B_CKPT=${M15HIGH_B_SAVE}/${M15HIGH_B_RUN}/offline_reconstructor_B_stageA.pt,TRAIN_ONLY_RECOA=0,TRAIN_ONLY_RECOB=0,STOP_AFTER_RECO_PRETRAIN=0" "${RUN_M15HIGH}")

# ---------- M12 ----------
M12_A_SAVE="${BASE}/model12_dualreco_dualview_feat_noscale_weighted_5m1m1m_recoAonly"
M12_A_RUN="model12_dualreco_dualview_feat_noscale_weighted_5m1m1m_seed0_recoAonly"
M12_B_SAVE="${BASE}/model12_dualreco_dualview_feat_noscale_weighted_5m1m1m_recoBonly"
M12_B_RUN="model12_dualreco_dualview_feat_noscale_weighted_5m1m1m_seed0_recoBonly"
M12_T_SAVE="${BASE}/model12_dualreco_dualview_feat_noscale_weighted_5m1m1m_from_splitAB"
M12_T_RUN="model12_dualreco_dualview_feat_noscale_weighted_5m1m1m_seed0_from_splitAB"

j12_a=$(sbatch --parsable --time="${RECO_TIME}" --export="ALL,RUN_NAME=${M12_A_RUN},SAVE_DIR=${M12_A_SAVE},STEP1_LOAD_DIR=${M12_STEP1_DIR},NUM_WORKERS=${NUM_WORKERS_RECO},TRAIN_ONLY_RECOA=1,TRAIN_ONLY_RECOB=0,STOP_AFTER_RECO_PRETRAIN=0" "${RUN_M12}")
j12_b=$(sbatch --parsable --time="${RECO_TIME}" --export="ALL,RUN_NAME=${M12_B_RUN},SAVE_DIR=${M12_B_SAVE},STEP1_LOAD_DIR=${M12_STEP1_DIR},NUM_WORKERS=${NUM_WORKERS_RECO},TRAIN_ONLY_RECOA=0,TRAIN_ONLY_RECOB=1,STOP_AFTER_RECO_PRETRAIN=0" "${RUN_M12}")
j12_t=$(sbatch --parsable --time="${TAGGER_TIME}" --dependency="afterok:${j12_a}:${j12_b}" --export="ALL,RUN_NAME=${M12_T_RUN},SAVE_DIR=${M12_T_SAVE},STEP1_LOAD_DIR=${M12_A_SAVE}/${M12_A_RUN},NUM_WORKERS=${NUM_WORKERS_TAGGER},LOAD_RECO_A_CKPT=${M12_A_SAVE}/${M12_A_RUN}/offline_reconstructor_A_stageA.pt,LOAD_RECO_B_CKPT=${M12_B_SAVE}/${M12_B_RUN}/offline_reconstructor_B_stageA.pt,TRAIN_ONLY_RECOA=0,TRAIN_ONLY_RECOB=0,STOP_AFTER_RECO_PRETRAIN=0" "${RUN_M12}")

# ---------- M16 ----------
M16_A_SAVE="${BASE}/model16_dualreco_dualview_topk60_weighted_5m1m1m_recoAonly"
M16_A_RUN="model16_dualreco_dualview_topk60_weighted_5m1m1m_seed0_recoAonly"
M16_B_SAVE="${BASE}/model16_dualreco_dualview_topk60_weighted_5m1m1m_recoBonly"
M16_B_RUN="model16_dualreco_dualview_topk60_weighted_5m1m1m_seed0_recoBonly"
M16_T_SAVE="${BASE}/model16_dualreco_dualview_topk60_weighted_5m1m1m_from_splitAB"
M16_T_RUN="model16_dualreco_dualview_topk60_weighted_5m1m1m_seed0_from_splitAB"

j16_a=$(sbatch --parsable --time="${RECO_TIME}" --export="ALL,RUN_NAME=${M16_A_RUN},SAVE_DIR=${M16_A_SAVE},STEP1_LOAD_DIR=${M16_STEP1_DIR},NUM_WORKERS=${NUM_WORKERS_RECO},TRAIN_ONLY_RECOA=1,TRAIN_ONLY_RECOB=0,STOP_AFTER_RECO_PRETRAIN=0" "${RUN_M16}")
j16_b=$(sbatch --parsable --time="${RECO_TIME}" --export="ALL,RUN_NAME=${M16_B_RUN},SAVE_DIR=${M16_B_SAVE},STEP1_LOAD_DIR=${M16_STEP1_DIR},NUM_WORKERS=${NUM_WORKERS_RECO},TRAIN_ONLY_RECOA=0,TRAIN_ONLY_RECOB=1,STOP_AFTER_RECO_PRETRAIN=0" "${RUN_M16}")
j16_t=$(sbatch --parsable --time="${TAGGER_TIME}" --dependency="afterok:${j16_a}:${j16_b}" --export="ALL,RUN_NAME=${M16_T_RUN},SAVE_DIR=${M16_T_SAVE},STEP1_LOAD_DIR=${M16_A_SAVE}/${M16_A_RUN},NUM_WORKERS=${NUM_WORKERS_TAGGER},LOAD_RECO_A_CKPT=${M16_A_SAVE}/${M16_A_RUN}/offline_reconstructor_A_stageA.pt,LOAD_RECO_B_CKPT=${M16_B_SAVE}/${M16_B_RUN}/offline_reconstructor_B_stageA.pt,TRAIN_ONLY_RECOA=0,TRAIN_ONLY_RECOB=0,STOP_AFTER_RECO_PRETRAIN=0" "${RUN_M16}")

# ---------- M17 ----------
M17_A_SAVE="${BASE}/model17_dualreco_dualview_antioverlap_weighted_5m1m1m_recoAonly"
M17_A_RUN="model17_dualreco_dualview_antioverlap_weighted_5m1m1m_seed0_recoAonly"
M17_B_SAVE="${BASE}/model17_dualreco_dualview_antioverlap_weighted_5m1m1m_recoBonly"
M17_B_RUN="model17_dualreco_dualview_antioverlap_weighted_5m1m1m_seed0_recoBonly"
M17_T_SAVE="${BASE}/model17_dualreco_dualview_antioverlap_weighted_5m1m1m_from_splitAB"
M17_T_RUN="model17_dualreco_dualview_antioverlap_weighted_5m1m1m_seed0_from_splitAB"

j17_a=$(sbatch --parsable --time="${RECO_TIME}" --export="ALL,RUN_NAME=${M17_A_RUN},SAVE_DIR=${M17_A_SAVE},STEP1_LOAD_DIR=${M17_STEP1_DIR},NUM_WORKERS=${NUM_WORKERS_RECO},TRAIN_ONLY_RECOA=1,TRAIN_ONLY_RECOB=0,STOP_AFTER_RECO_PRETRAIN=0" "${RUN_M17}")
j17_b=$(sbatch --parsable --time="${RECO_TIME}" --export="ALL,RUN_NAME=${M17_B_RUN},SAVE_DIR=${M17_B_SAVE},STEP1_LOAD_DIR=${M17_STEP1_DIR},NUM_WORKERS=${NUM_WORKERS_RECO},TRAIN_ONLY_RECOA=0,TRAIN_ONLY_RECOB=1,STOP_AFTER_RECO_PRETRAIN=0" "${RUN_M17}")
j17_t=$(sbatch --parsable --time="${TAGGER_TIME}" --dependency="afterok:${j17_a}:${j17_b}" --export="ALL,RUN_NAME=${M17_T_RUN},SAVE_DIR=${M17_T_SAVE},STEP1_LOAD_DIR=${M17_A_SAVE}/${M17_A_RUN},NUM_WORKERS=${NUM_WORKERS_TAGGER},LOAD_RECO_A_CKPT=${M17_A_SAVE}/${M17_A_RUN}/offline_reconstructor_A_stageA.pt,LOAD_RECO_B_CKPT=${M17_B_SAVE}/${M17_B_RUN}/offline_reconstructor_B_stageA.pt,TRAIN_ONLY_RECOA=0,TRAIN_ONLY_RECOB=0,STOP_AFTER_RECO_PRETRAIN=0" "${RUN_M17}")

echo "Submitted dual-5 parallel A/B + dependent tagger jobs:"
echo "  m15mid : A=${j15m_a} B=${j15m_b} T=${j15m_t}"
echo "  m15high: A=${j15h_a} B=${j15h_b} T=${j15h_t}"
echo "  m12    : A=${j12_a} B=${j12_b} T=${j12_t}"
echo "  m16    : A=${j16_a} B=${j16_b} T=${j16_t}"
echo "  m17    : A=${j17_a} B=${j17_b} T=${j17_t}"

if [[ "${QUEUE_ANALYZE}" == "1" ]]; then
  dep="${j15m_t}:${j15h_t}:${j12_t}:${j16_t}:${j17_t}"
  if [[ -n "${BASE_DEPS}" ]]; then
    dep="${BASE_DEPS}:${dep}"
  fi

  an_export="ALL,M12_RUN_DIR=${M12_T_SAVE}/${M12_T_RUN},M15MID_RUN_DIR=${M15MID_T_SAVE}/${M15MID_T_RUN},M15HIGH_RUN_DIR=${M15HIGH_T_SAVE}/${M15HIGH_T_RUN},M16_RUN_DIR=${M16_T_SAVE}/${M16_T_RUN},M17_RUN_DIR=${M17_T_SAVE}/${M17_T_RUN},INCLUDE_HLT_CANDIDATE=${INCLUDE_HLT_CANDIDATE}"
  if [[ -n "${STEP1_REF_NPZ}" ]]; then
    an_export="${an_export},STEP1_REF_NPZ=${STEP1_REF_NPZ}"
  fi

  j_an=$(sbatch --parsable --dependency="afterok:${dep}" --export="${an_export}" "${RUN_ANALYZE}")
  echo "  analyze12: ${j_an} (afterok:${dep})"
else
  echo "Analyze not queued (QUEUE_ANALYZE=${QUEUE_ANALYZE})."
fi
