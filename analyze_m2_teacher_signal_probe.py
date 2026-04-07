#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from unmerge_correct_hlt import (
    ParticleTransformer,
    DualViewCrossAttnClassifier,
    compute_features,
    get_stats,
    load_raw_constituents_from_h5,
    standardize,
)
from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as BASE_CONFIG,
    OfflineReconstructor,
    apply_hlt_effects_realistic_nomap,
)
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as joint_base

from analyze_m2_router_signal_sweep import (
    _clip_probs,
    _logit,
    _entropy_bernoulli,
    _js_bernoulli,
    _load_ckpt_state,
    _infer_dual_input_dims,
    _build_train_file_list,
    _offline_mask,
    _parse_corruptions,
    _apply_corruption_batch,
    _init_shift_acc,
    _update_shift_acc,
    _shift_features_from_acc,
    _corruption_full_features,
    _stack_features,
    _threshold_at_target_tpr,
    _lowfpr_cost,
    _score_metrics,
    _choose_threshold_by_cal,
    _save_csv,
    _infer_scores_and_diag,
)


def _deepcopy_cfg() -> Dict:
    return json.loads(json.dumps(BASE_CONFIG))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    x = x - np.mean(x)
    y = y - np.mean(y)
    sx = float(np.sqrt(np.mean(x * x)))
    sy = float(np.sqrt(np.mean(y * y)))
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.mean((x / sx) * (y / sy)))


def _rank_ordinal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(order.size, dtype=np.float64)
    return r


def _spearman_approx(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rank_ordinal(x), _rank_ordinal(y))


def _infer_pt_input_dim(sd: Dict[str, torch.Tensor]) -> int:
    if "input_proj.0.weight" in sd:
        return int(sd["input_proj.0.weight"].shape[1])
    if "input_proj.weight" in sd:
        return int(sd["input_proj.weight"].shape[1])
    raise RuntimeError("Could not infer ParticleTransformer input_dim from checkpoint keys")


@dataclass
class TeacherViews:
    p_teacher_hlt: np.ndarray
    p_teacher_reco: np.ndarray


def _infer_teacher_views(
    teacher: ParticleTransformer,
    reconstructor: OfflineReconstructor,
    feat_hlt_std: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt: np.ndarray,
    batch_size: int,
    device: torch.device,
    corrected_weight_floor: float,
    corrected_use_flags: bool,
    teacher_input_dim: int,
) -> TeacherViews:
    n = int(feat_hlt_std.shape[0])
    p_th = np.zeros((n,), dtype=np.float32)
    p_tr = np.zeros((n,), dtype=np.float32)

    teacher.eval()
    reconstructor.eval()

    with torch.no_grad():
        for s in range(0, n, int(batch_size)):
            e = min(n, s + int(batch_size))

            x = torch.from_numpy(feat_hlt_std[s:e]).to(device=device, dtype=torch.float32)
            m = torch.from_numpy(mask_hlt[s:e]).to(device=device, dtype=torch.bool)
            c = torch.from_numpy(const_hlt[s:e]).to(device=device, dtype=torch.float32)

            # Teacher on HLT view.
            logit_th = teacher(x, m).squeeze(1)

            # Teacher on reconstructed/soft-corrected view.
            out = reconstructor(x, m, c, stage_scale=1.0)
            feat_b, mask_b = joint_base.build_soft_corrected_view(
                out,
                weight_floor=float(corrected_weight_floor),
                scale_features_by_weight=True,
                include_flags=bool(corrected_use_flags),
            )
            if int(feat_b.shape[-1]) < int(teacher_input_dim):
                raise RuntimeError(
                    f"Teacher input_dim={teacher_input_dim} but corrected features have only {feat_b.shape[-1]} dims"
                )
            feat_t = feat_b[:, :, : int(teacher_input_dim)]
            logit_tr = teacher(feat_t, mask_b).squeeze(1)

            p_th[s:e] = torch.sigmoid(logit_th).detach().cpu().numpy().astype(np.float32)
            p_tr[s:e] = torch.sigmoid(logit_tr).detach().cpu().numpy().astype(np.float32)

    return TeacherViews(p_teacher_hlt=p_th, p_teacher_reco=p_tr)


def _rename_teacher_shift_feats(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in d.items():
        if k.startswith("joint_minus_hlt_"):
            k2 = "teacher_reco_minus_hlt_" + k[len("joint_minus_hlt_") :]
        elif k.startswith("joint_stability_"):
            k2 = "teacher_reco_stability_" + k[len("joint_stability_") :]
        elif k.startswith("hlt_"):
            k2 = "teacher_hlt_" + k[len("hlt_") :]
        elif k.startswith("joint_"):
            k2 = "teacher_reco_" + k[len("joint_") :]
        elif k == "mean_disagreement_growth":
            k2 = "teacher_mean_disagreement_growth"
        elif k == "std_disagreement_growth":
            k2 = "teacher_std_disagreement_growth"
        elif k == "max_abs_disagreement_growth":
            k2 = "teacher_max_abs_disagreement_growth"
        else:
            k2 = "teacher_" + k
        out[k2] = np.asarray(v, dtype=np.float32)
    return out


def _build_teacher_feature_dict(
    p_hlt: np.ndarray,
    p_joint: np.ndarray,
    p_teacher_hlt: np.ndarray,
    p_teacher_reco: np.ndarray,
    diag: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    ph = _clip_probs(p_hlt)
    pj = _clip_probs(p_joint)
    pth = _clip_probs(p_teacher_hlt)
    ptr = _clip_probs(p_teacher_reco)

    conf_h = np.abs(pth - 0.5) * 2.0
    conf_r = np.abs(ptr - 0.5) * 2.0
    ent_h = _entropy_bernoulli(pth)
    ent_r = _entropy_bernoulli(ptr)

    support_prod = (pj - ph) * (ptr - pth)

    out: Dict[str, np.ndarray] = {
        # core teacher outputs
        "teacher_prob_hlt": pth.astype(np.float32),
        "teacher_prob_reco": ptr.astype(np.float32),
        "teacher_logit_hlt": _logit(pth).astype(np.float32),
        "teacher_logit_reco": _logit(ptr).astype(np.float32),
        "teacher_prob_gap_reco_minus_hlt": (ptr - pth).astype(np.float32),
        "teacher_abs_prob_gap_views": np.abs(ptr - pth).astype(np.float32),
        "teacher_logit_gap_reco_minus_hlt": (_logit(ptr) - _logit(pth)).astype(np.float32),
        "teacher_conf_hlt": conf_h.astype(np.float32),
        "teacher_conf_reco": conf_r.astype(np.float32),
        "teacher_conf_gap_reco_minus_hlt": (conf_r - conf_h).astype(np.float32),
        "teacher_entropy_hlt": ent_h.astype(np.float32),
        "teacher_entropy_reco": ent_r.astype(np.float32),
        "teacher_entropy_gap_reco_minus_hlt": (ent_r - ent_h).astype(np.float32),
        "teacher_js_hlt_vs_reco": _js_bernoulli(pth, ptr).astype(np.float32),
        "teacher_pred_agree_views": ((pth >= 0.5) == (ptr >= 0.5)).astype(np.float32),
        # alignment / support wrt base models
        "teacher_alignment_abs_hlt": np.abs(ph - pth).astype(np.float32),
        "teacher_alignment_abs_joint": np.abs(pj - ptr).astype(np.float32),
        "teacher_alignment_gap_joint_minus_hlt": (np.abs(pj - ptr) - np.abs(ph - pth)).astype(np.float32),
        "teacher_joint_consensus_pred": ((pj >= 0.5) == (ptr >= 0.5)).astype(np.float32),
        "teacher_hlt_consensus_pred": ((ph >= 0.5) == (pth >= 0.5)).astype(np.float32),
        "teacher_support_product_joint_over_hlt": support_prod.astype(np.float32),
        "teacher_support_sign_joint_over_hlt": (support_prod >= 0.0).astype(np.float32),
        "teacher_support_margin_joint_over_hlt": np.abs(support_prod).astype(np.float32),
        # include base scores as context for interpretation
        "context_hlt_prob": ph.astype(np.float32),
        "context_joint_prob": pj.astype(np.float32),
        "context_signed_disagree_joint_minus_hlt": (pj - ph).astype(np.float32),
        "context_abs_disagree_joint_hlt": np.abs(pj - ph).astype(np.float32),
    }

    # Selected reco diagnostics as potential teacher-side context signals.
    for k in [
        "action_entropy_mean",
        "split_total_mean",
        "split_total_max",
        "budget_total",
        "budget_merge_pressure",
        "budget_eff_pressure",
        "expected_added_count",
        "added_count_uncertainty",
        "correction_magnitude_mean",
        "correction_magnitude_max",
        "split_angle_tanh_sat_frac",
        "split_frac_logit_sat_frac",
    ]:
        if k in diag:
            out[f"context_{k}"] = np.asarray(diag[k], dtype=np.float32)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Teacher-signal probe for routing joint vs HLT (correlation + hard-rule potential)."
    )
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--train_path", type=str, default="./data")
    ap.add_argument("--teacher_ckpt", type=str, default="")

    ap.add_argument("--router_offset_jets", type=int, default=375000)
    ap.add_argument("--router_n_analysis", type=int, default=200000)
    ap.add_argument("--router_n_test", type=int, default=200000)
    ap.add_argument("--ref_offset_jets", type=int, default=0)
    ap.add_argument("--ref_n_jets", type=int, default=375000)
    ap.add_argument("--max_constits", type=int, default=100)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hlt_seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--corrected_weight_floor", type=float, default=1e-4)

    ap.add_argument(
        "--corruptions",
        type=str,
        default="pt_noise:0.03,pt_noise:0.06,eta_phi_jitter:0.02,eta_phi_jitter:0.05,dropout:0.05,dropout:0.10,merge:0.10,merge:0.20,global_scale:0.03",
    )
    ap.add_argument("--feature_profile", type=str, default="core", choices=["core", "full"])
    ap.add_argument("--router_cal_frac", type=float, default=0.2)
    ap.add_argument("--cost_alpha_neg", type=float, default=4.0)
    ap.add_argument("--cost_tau", type=float, default=0.02)

    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--report_json", type=str, default="")
    ap.add_argument("--save_per_jet_npz", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir).strip()
        else (run_dir / "teacher_signal_probe")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    rng = np.random.default_rng(int(args.seed))

    data_setup_path = run_dir / "data_setup.json"
    if not data_setup_path.exists():
        raise FileNotFoundError(f"Missing data_setup.json in run_dir: {data_setup_path}")
    data_setup = json.loads(data_setup_path.read_text())

    cfg = _deepcopy_cfg()
    hlt_stats_path = run_dir / "hlt_stats.json"
    if hlt_stats_path.exists():
        hlt_obj = json.loads(hlt_stats_path.read_text())
        hcfg = hlt_obj.get("config", {})
        for k in list(cfg.get("hlt_effects", {}).keys()):
            if k in hcfg:
                cfg["hlt_effects"][k] = hcfg[k]

    train_files = _build_train_file_list(data_setup, args.train_path)

    # ---------------- Standardization stats from reference training setup ----------------
    ref_need = int(args.ref_offset_jets + args.ref_n_jets)
    const_ref_raw, labels_ref_all = load_raw_constituents_from_h5(
        train_files,
        max_jets=ref_need,
        max_constits=int(args.max_constits),
    )
    const_ref_raw = const_ref_raw[int(args.ref_offset_jets) : int(args.ref_offset_jets) + int(args.ref_n_jets)]
    labels_ref = labels_ref_all[int(args.ref_offset_jets) : int(args.ref_offset_jets) + int(args.ref_n_jets)].astype(np.int64)

    const_ref_off, mask_ref_off = _offline_mask(const_ref_raw, float(cfg["hlt_effects"]["pt_threshold_offline"]))
    feat_ref_off = compute_features(const_ref_off, mask_ref_off)
    idx_ref = np.arange(labels_ref.shape[0])
    tr_ref, _ = train_test_split(
        idx_ref,
        test_size=0.30,
        random_state=int(args.seed),
        stratify=labels_ref,
    )
    means, stds = get_stats(feat_ref_off, mask_ref_off, tr_ref)

    del const_ref_raw, labels_ref_all, labels_ref, const_ref_off, mask_ref_off, feat_ref_off, idx_ref, tr_ref
    gc.collect()

    # ---------------- Router dataset ----------------
    n_total_router = int(args.router_n_analysis + args.router_n_test)
    need = int(args.router_offset_jets + n_total_router)
    const_raw_all, labels_all = load_raw_constituents_from_h5(
        train_files,
        max_jets=need,
        max_constits=int(args.max_constits),
    )
    const_raw = const_raw_all[int(args.router_offset_jets) : int(args.router_offset_jets) + n_total_router]
    labels = labels_all[int(args.router_offset_jets) : int(args.router_offset_jets) + n_total_router].astype(np.int64)

    const_off, mask_off = _offline_mask(const_raw, float(cfg["hlt_effects"]["pt_threshold_offline"]))
    hlt_const, hlt_mask, _hlt_stats_gen, _ = apply_hlt_effects_realistic_nomap(
        const_off,
        mask_off,
        cfg,
        seed=int(args.hlt_seed),
    )
    feat_hlt = compute_features(hlt_const, hlt_mask)
    feat_hlt_std = standardize(feat_hlt, hlt_mask, means, stds).astype(np.float32)

    # ---------------- Load models ----------------
    baseline_ckpt = run_dir / "baseline.pt"
    reco_ckpt = run_dir / "offline_reconstructor.pt"
    dual_ckpt = run_dir / "dual_joint.pt"
    teacher_ckpt = Path(args.teacher_ckpt).expanduser().resolve() if str(args.teacher_ckpt).strip() else (run_dir / "teacher.pt")

    if not baseline_ckpt.exists() or not reco_ckpt.exists() or not dual_ckpt.exists() or not teacher_ckpt.exists():
        raise FileNotFoundError(
            "Missing checkpoints in run_dir. Needed: baseline.pt, offline_reconstructor.pt, dual_joint.pt, teacher.pt"
        )

    baseline_sd = _load_ckpt_state(baseline_ckpt, device)
    reco_sd = _load_ckpt_state(reco_ckpt, device)
    dual_sd = _load_ckpt_state(dual_ckpt, device)
    teacher_sd = _load_ckpt_state(teacher_ckpt, device)

    dual_in_a, dual_in_b = _infer_dual_input_dims(dual_sd)
    if int(dual_in_a) != 7:
        raise RuntimeError(f"This analysis currently supports dual input_dim_a=7 only. Found {dual_in_a}.")
    corrected_use_flags = bool(int(dual_in_b) == 12)

    teacher_in = _infer_pt_input_dim(teacher_sd)

    baseline = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline.load_state_dict(baseline_sd, strict=True)

    reconstructor = OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reconstructor.load_state_dict(reco_sd, strict=True)

    dual = DualViewCrossAttnClassifier(input_dim_a=int(dual_in_a), input_dim_b=int(dual_in_b), **cfg["model"]).to(device)
    dual.load_state_dict(dual_sd, strict=True)

    teacher = ParticleTransformer(input_dim=int(teacher_in), **cfg["model"]).to(device)
    teacher.load_state_dict(teacher_sd, strict=True)

    # ---------------- Clean inference ----------------
    clean = _infer_scores_and_diag(
        baseline=baseline,
        reconstructor=reconstructor,
        dual=dual,
        feat_hlt_std=feat_hlt_std,
        mask_hlt=hlt_mask,
        const_hlt=hlt_const,
        batch_size=int(args.batch_size),
        device=device,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=corrected_use_flags,
        collect_diag=True,
    )

    teacher_clean = _infer_teacher_views(
        teacher=teacher,
        reconstructor=reconstructor,
        feat_hlt_std=feat_hlt_std,
        mask_hlt=hlt_mask,
        const_hlt=hlt_const,
        batch_size=int(args.batch_size),
        device=device,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=corrected_use_flags,
        teacher_input_dim=int(teacher_in),
    )

    # ---------------- Corruption sweep (teacher views) ----------------
    full_profile = str(args.feature_profile).strip().lower() == "full"
    corruption_list = _parse_corruptions(args.corruptions)

    acc_teacher = _init_shift_acc(n_total_router)
    corr_p_th_list: List[np.ndarray] = []
    corr_p_tr_list: List[np.ndarray] = []
    corr_sev_list: List[float] = []
    corr_kind_list: List[str] = []

    for kind, severity in corruption_list:
        print(f"[corruption] kind={kind} severity={severity:.4f}")
        p_th_corr = np.zeros((n_total_router,), dtype=np.float32)
        p_tr_corr = np.zeros((n_total_router,), dtype=np.float32)

        for s in range(0, n_total_router, int(args.batch_size)):
            e = min(n_total_router, s + int(args.batch_size))
            c_b, m_b = _apply_corruption_batch(hlt_const[s:e], hlt_mask[s:e], kind, severity, rng)
            f_b = compute_features(c_b, m_b)
            f_b_std = standardize(f_b, m_b, means, stds).astype(np.float32)

            tv_b = _infer_teacher_views(
                teacher=teacher,
                reconstructor=reconstructor,
                feat_hlt_std=f_b_std,
                mask_hlt=m_b,
                const_hlt=c_b,
                batch_size=max(32, int(args.batch_size)),
                device=device,
                corrected_weight_floor=float(args.corrected_weight_floor),
                corrected_use_flags=corrected_use_flags,
                teacher_input_dim=int(teacher_in),
            )
            p_th_corr[s:e] = tv_b.p_teacher_hlt
            p_tr_corr[s:e] = tv_b.p_teacher_reco

        _update_shift_acc(
            acc_teacher,
            teacher_clean.p_teacher_hlt,
            teacher_clean.p_teacher_reco,
            p_th_corr,
            p_tr_corr,
        )
        if full_profile:
            corr_p_th_list.append(p_th_corr.copy())
            corr_p_tr_list.append(p_tr_corr.copy())
            corr_sev_list.append(float(severity))
            corr_kind_list.append(str(kind))

    shift_teacher = _rename_teacher_shift_feats(_shift_features_from_acc(acc_teacher))
    if full_profile and len(corr_p_th_list) > 0:
        full_teacher = _corruption_full_features(
            teacher_clean.p_teacher_hlt,
            teacher_clean.p_teacher_reco,
            corr_p_th_list,
            corr_p_tr_list,
            corr_sev_list,
            corr_kind_list,
        )
        shift_teacher.update(_rename_teacher_shift_feats(full_teacher))

    # ---------------- Assemble feature table ----------------
    feature_dict = _build_teacher_feature_dict(
        p_hlt=clean.p_hlt,
        p_joint=clean.p_joint,
        p_teacher_hlt=teacher_clean.p_teacher_hlt,
        p_teacher_reco=teacher_clean.p_teacher_reco,
        diag=clean.diag,
    )
    feature_dict.update(shift_teacher)

    # ---------------- Split analysis/test ----------------
    idx_all = np.arange(n_total_router)
    idx_analysis, idx_test = train_test_split(
        idx_all,
        test_size=float(args.router_n_test) / float(n_total_router),
        random_state=int(args.seed),
        stratify=labels,
    )
    if len(idx_analysis) != int(args.router_n_analysis) or len(idx_test) != int(args.router_n_test):
        raise RuntimeError(
            f"Split mismatch: got analysis={len(idx_analysis)} test={len(idx_test)}; "
            f"expected {args.router_n_analysis}/{args.router_n_test}"
        )

    y_a = labels[idx_analysis]
    y_t = labels[idx_test]
    ph_a = clean.p_hlt[idx_analysis]
    pj_a = clean.p_joint[idx_analysis]
    ph_t = clean.p_hlt[idx_test]
    pj_t = clean.p_joint[idx_test]

    idx_fit, idx_cal = train_test_split(
        np.arange(len(idx_analysis)),
        test_size=float(args.router_cal_frac),
        random_state=int(args.seed) + 17,
        stratify=y_a,
    )

    y_fit = y_a[idx_fit]
    y_cal = y_a[idx_cal]
    ph_fit = ph_a[idx_fit]
    pj_fit = pj_a[idx_fit]
    ph_cal = ph_a[idx_cal]
    pj_cal = pj_a[idx_cal]

    # Low-FPR-weighted routing labels.
    thr_h30 = _threshold_at_target_tpr(y_fit, ph_fit, 0.30)
    thr_h50 = _threshold_at_target_tpr(y_fit, ph_fit, 0.50)
    thr_j30 = _threshold_at_target_tpr(y_fit, pj_fit, 0.30)
    thr_j50 = _threshold_at_target_tpr(y_fit, pj_fit, 0.50)

    c_h_fit = _lowfpr_cost(y_fit, ph_fit, thr_h30, thr_h50, args.cost_alpha_neg, args.cost_tau)
    c_j_fit = _lowfpr_cost(y_fit, pj_fit, thr_j30, thr_j50, args.cost_alpha_neg, args.cost_tau)
    z_fit = (c_j_fit < c_h_fit).astype(np.int64)
    delta_cost_fit = c_h_fit - c_j_fit

    c_h_cal = _lowfpr_cost(y_cal, ph_cal, thr_h30, thr_h50, args.cost_alpha_neg, args.cost_tau)
    c_j_cal = _lowfpr_cost(y_cal, pj_cal, thr_j30, thr_j50, args.cost_alpha_neg, args.cost_tau)
    z_cal = (c_j_cal < c_h_cal).astype(np.int64)

    c_h_test = _lowfpr_cost(y_t, ph_t, thr_h30, thr_h50, args.cost_alpha_neg, args.cost_tau)
    c_j_test = _lowfpr_cost(y_t, pj_t, thr_j30, thr_j50, args.cost_alpha_neg, args.cost_tau)
    z_test = (c_j_test < c_h_test).astype(np.int64)

    # Threshold-aware teacher features.
    p_th = _clip_probs(teacher_clean.p_teacher_hlt)
    p_tr = _clip_probs(teacher_clean.p_teacher_reco)
    feature_dict["teacher_threshold_dist_hlt_to_hlt50"] = np.abs(p_th - thr_h50).astype(np.float32)
    feature_dict["teacher_threshold_dist_reco_to_joint50"] = np.abs(p_tr - thr_j50).astype(np.float32)
    feature_dict["teacher_threshold_dist_gap_recoMinusHlt"] = (
        np.abs(p_tr - thr_j50) - np.abs(p_th - thr_h50)
    ).astype(np.float32)

    # Teacher conformal-style p-value proxies.
    nc_th_all = np.minimum(p_th, 1.0 - p_th).astype(np.float64)
    nc_tr_all = np.minimum(p_tr, 1.0 - p_tr).astype(np.float64)
    nc_th_fit = np.sort(nc_th_all[idx_analysis][idx_fit])
    nc_tr_fit = np.sort(nc_tr_all[idx_analysis][idx_fit])
    pval_th = 1.0 - (np.searchsorted(nc_th_fit, nc_th_all, side="right") / max(len(nc_th_fit), 1))
    pval_tr = 1.0 - (np.searchsorted(nc_tr_fit, nc_tr_all, side="right") / max(len(nc_tr_fit), 1))
    feature_dict["teacher_conformal_pvalue_hlt"] = pval_th.astype(np.float32)
    feature_dict["teacher_conformal_pvalue_reco"] = pval_tr.astype(np.float32)
    feature_dict["teacher_conformal_pvalue_gap_reco_minus_hlt"] = (pval_tr - pval_th).astype(np.float32)

    if full_profile and len(corr_p_th_list) > 0:
        pth_corr = np.stack(corr_p_th_list, axis=0).astype(np.float64)
        ptr_corr = np.stack(corr_p_tr_list, axis=0).astype(np.float64)
        cross_th50 = np.mean(((p_th[None, :] - thr_h50) * (pth_corr - thr_h50) < 0.0), axis=0)
        cross_tr50 = np.mean(((p_tr[None, :] - thr_j50) * (ptr_corr - thr_j50) < 0.0), axis=0)
        cross_th30 = np.mean(((p_th[None, :] - thr_h30) * (pth_corr - thr_h30) < 0.0), axis=0)
        cross_tr30 = np.mean(((p_tr[None, :] - thr_j30) * (ptr_corr - thr_j30) < 0.0), axis=0)
        feature_dict["teacher_threshold_cross_flip_rate_hlt_fpr50"] = cross_th50.astype(np.float32)
        feature_dict["teacher_threshold_cross_flip_rate_reco_fpr50"] = cross_tr50.astype(np.float32)
        feature_dict["teacher_threshold_cross_flip_rate_gap_fpr50"] = (cross_tr50 - cross_th50).astype(np.float32)
        feature_dict["teacher_threshold_cross_flip_rate_hlt_fpr30"] = cross_th30.astype(np.float32)
        feature_dict["teacher_threshold_cross_flip_rate_reco_fpr30"] = cross_tr30.astype(np.float32)
        feature_dict["teacher_threshold_cross_flip_rate_gap_fpr30"] = (cross_tr30 - cross_th30).astype(np.float32)

    # ---------------- Per-signal ranking + hard-rule test metrics ----------------
    signal_rows: List[Dict[str, object]] = []

    for fn in sorted(feature_dict.keys()):
        s_all = np.asarray(feature_dict[fn], dtype=np.float64)
        s_fit = s_all[idx_analysis][idx_fit]
        s_cal = s_all[idx_analysis][idx_cal]
        s_test = s_all[idx_test]

        if s_fit.size == 0 or np.allclose(np.std(s_fit), 0.0):
            continue

        try:
            auc_raw = float(roc_auc_score(z_fit, s_fit))
        except Exception:
            auc_raw = float("nan")

        if not np.isfinite(auc_raw):
            direction = ">="
            winner_auc = 0.5
        else:
            direction = ">=" if auc_raw >= 0.5 else "<="
            winner_auc = max(auc_raw, 1.0 - auc_raw)

        pear_delta = _pearson(s_fit, delta_cost_fit)
        spear_delta = _spearman_approx(s_fit, delta_cost_fit)
        pear_win = _pearson(s_fit, z_fit.astype(np.float64))

        thr, _ = _choose_threshold_by_cal(
            y_cal=y_cal,
            p_h_cal=ph_cal,
            p_j_cal=pj_cal,
            gate_score_cal=s_cal,
            direction=direction,
            objective="fpr50",
        )
        m_test = s_test >= thr if direction == ">=" else s_test <= thr
        p_route_test = np.where(m_test, pj_t, ph_t)
        mtr_test = _score_metrics(y_t, p_route_test, ph_t)

        signal_rows.append(
            {
                "signal": fn,
                "winner_auc_fit": float(winner_auc),
                "auc_fit_raw": float(auc_raw),
                "pearson_with_delta_cost_fit": float(pear_delta),
                "spearman_with_delta_cost_fit": float(spear_delta),
                "pearson_with_route_winner_fit": float(pear_win),
                "direction": direction,
                "threshold": float(thr),
                "joint_route_frac_test": float(np.mean(m_test)),
                "auc_test": float(mtr_test["auc"]),
                "fpr30_test": float(mtr_test["fpr30"]),
                "fpr50_test": float(mtr_test["fpr50"]),
                "harm_bce_frac_neg_vs_hlt_test": float(mtr_test["harm_bce_frac_neg_vs_hlt"]),
            }
        )

    signal_rows.sort(
        key=lambda r: (
            -float(r["winner_auc_fit"]),
            -abs(float(r["pearson_with_delta_cost_fit"])) if np.isfinite(float(r["pearson_with_delta_cost_fit"])) else -0.0,
        )
    )

    # Best single-rule test performer among teacher signals.
    best_by_test = None
    for r in signal_rows:
        if best_by_test is None:
            best_by_test = r
            continue
        key = (-float(r["fpr50_test"]), float(r["auc_test"]), -float(r["fpr30_test"]))
        best_key = (-float(best_by_test["fpr50_test"]), float(best_by_test["auc_test"]), -float(best_by_test["fpr30_test"]))
        if key > best_key:
            best_by_test = r

    # Baseline comparators.
    base_hlt = _score_metrics(y_t, ph_t, ph_t)
    base_joint = _score_metrics(y_t, pj_t, ph_t)
    oracle_best = _score_metrics(y_t, np.where(z_test == 1, pj_t, ph_t), ph_t)

    rep = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "teacher_ckpt": str(teacher_ckpt),
        "feature_profile": str(args.feature_profile),
        "n_total_router": int(n_total_router),
        "router_n_analysis": int(args.router_n_analysis),
        "router_n_test": int(args.router_n_test),
        "n_teacher_signals": int(len(signal_rows)),
        "corruptions": [{"kind": k, "severity": float(v)} for k, v in corruption_list],
        "lowfpr_target": {
            "thr_h30": float(thr_h30),
            "thr_h50": float(thr_h50),
            "thr_j30": float(thr_j30),
            "thr_j50": float(thr_j50),
            "cost_alpha_neg": float(args.cost_alpha_neg),
            "cost_tau": float(args.cost_tau),
            "route_win_rate_fit": float(np.mean(z_fit)),
            "route_win_rate_test": float(np.mean(z_test)),
        },
        "comparators_test": {
            "hlt": base_hlt,
            "joint": base_joint,
            "oracle_best_of_two_lowfpr_cost": oracle_best,
        },
        "best_single_teacher_signal_test": best_by_test,
        "top_signal_rank": signal_rows[:120],
        "files": {
            "signal_rank_csv": str((out_dir / "teacher_signal_rank_fit.csv").resolve()),
        },
    }

    rep_path = (
        Path(args.report_json).expanduser().resolve()
        if str(args.report_json).strip()
        else (out_dir / "teacher_signal_probe_report.json")
    )
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with rep_path.open("w") as f:
        json.dump(rep, f, indent=2)

    _save_csv(out_dir / "teacher_signal_rank_fit.csv", signal_rows)

    if bool(args.save_per_jet_npz):
        np.savez_compressed(
            out_dir / "teacher_signal_perjet.npz",
            labels=labels.astype(np.int8),
            idx_analysis=idx_analysis.astype(np.int64),
            idx_test=idx_test.astype(np.int64),
            preds_hlt=clean.p_hlt.astype(np.float32),
            preds_joint=clean.p_joint.astype(np.float32),
            teacher_prob_hlt=teacher_clean.p_teacher_hlt.astype(np.float32),
            teacher_prob_reco=teacher_clean.p_teacher_reco.astype(np.float32),
            **{k: np.asarray(v, dtype=np.float32) for k, v in feature_dict.items()},
        )

    print("=" * 72)
    print("Teacher Signal Probe")
    print("=" * 72)
    print(f"Run dir: {run_dir}")
    print(f"Out dir: {out_dir}")
    print(f"Teacher ckpt: {teacher_ckpt}")
    print(f"Feature profile: {args.feature_profile}")
    print(f"N analysis/test: {len(idx_analysis)} / {len(idx_test)}")
    print(f"Teacher signals ranked: {len(signal_rows)}")
    print(f"Corruptions: {len(corruption_list)}")
    print()
    print("Comparators on held-out test:")
    print(
        f"  hlt      AUC={base_hlt['auc']:.6f} FPR30={base_hlt['fpr30']:.6f} FPR50={base_hlt['fpr50']:.6f}"
    )
    print(
        f"  joint    AUC={base_joint['auc']:.6f} FPR30={base_joint['fpr30']:.6f} FPR50={base_joint['fpr50']:.6f}"
    )
    print(
        f"  oracle   AUC={oracle_best['auc']:.6f} FPR30={oracle_best['fpr30']:.6f} FPR50={oracle_best['fpr50']:.6f}"
    )
    if best_by_test is not None:
        print()
        print("Best single teacher signal hard-route on held-out test:")
        print(
            f"  {best_by_test['signal']} ({best_by_test['direction']} {best_by_test['threshold']:.6g}) "
            f"AUC={best_by_test['auc_test']:.6f} "
            f"FPR30={best_by_test['fpr30_test']:.6f} "
            f"FPR50={best_by_test['fpr50_test']:.6f} "
            f"route_frac={best_by_test['joint_route_frac_test']:.4f}"
        )
    print()
    print(f"Saved report: {rep_path}")
    print(f"Saved signal-rank CSV: {out_dir / 'teacher_signal_rank_fit.csv'}")
    if bool(args.save_per_jet_npz):
        print(f"Saved per-jet NPZ: {out_dir / 'teacher_signal_perjet.npz'}")


if __name__ == "__main__":
    main()
