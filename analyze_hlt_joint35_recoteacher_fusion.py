#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fusion analysis for:
- Previous 18-model fusion setup
- Additional dualreco-dualview model family (using frozen/pre-joint scores)
- Additional m2 delta ablations (delta000, delta020) using joint scores

Outputs mirror the joint18 analyzer, with weighted keys named by model count:
- all{N}_weighted_raw/platt/iso_{valsel|oracle}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import analyze_hlt_joint18_recoteacher_fusion as base
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd as m


def _collect_candidates(results: Dict[str, object], n_models: int) -> List[Dict[str, float | str]]:
    out: List[Dict[str, float | str]] = []

    def _flt(v: object, default: float) -> float:
        try:
            f = float(v)
        except Exception:
            return float(default)
        if not np.isfinite(f):
            return float(default)
        return float(f)

    def _get(d: object, key: str, default: float) -> float:
        if isinstance(d, dict):
            return _flt(d.get(key, default), default)
        return float(default)

    for name, met in results["individual"].items():
        out.append(
            {
                "name": f"indiv::{name}",
                "fpr": _get(met, "fpr_test", float("inf")),
                "auc": _get(met, "auc_test", float("nan")),
                "oracle": False,
            }
        )

    for name, pack in results["pair_results_valsel"].items():
        te = pack.get("test_eval", {}) if isinstance(pack, dict) else {}
        out.append(
            {
                "name": f"pair_valsel::{name}",
                "fpr": _get(te, "fpr", float("inf")),
                "auc": _get(te, "auc", float("nan")),
                "oracle": False,
            }
        )

    for name, pack in results["pair_results_oracle"].items():
        out.append(
            {
                "name": f"pair_oracle::{name}",
                "fpr": _get(pack, "fpr", float("inf")),
                "auc": _get(pack, "auc", float("nan")),
                "oracle": True,
            }
        )

    for k in [
        f"all{int(n_models)}_weighted_raw_valsel",
        f"all{int(n_models)}_weighted_platt_valsel",
        f"all{int(n_models)}_weighted_iso_valsel",
    ]:
        pack = results.get(k, {}) if isinstance(results, dict) else {}
        te = pack.get("test_eval", {}) if isinstance(pack, dict) else {}
        out.append(
            {
                "name": k,
                "fpr": _get(te, "fpr", float("inf")),
                "auc": _get(te, "auc", float("nan")),
                "oracle": False,
            }
        )

    for k in [
        f"all{int(n_models)}_weighted_raw_oracle",
        f"all{int(n_models)}_weighted_platt_oracle",
        f"all{int(n_models)}_weighted_iso_oracle",
    ]:
        pack = results.get(k, {}) if isinstance(results, dict) else {}
        out.append(
            {
                "name": k,
                "fpr": _get(pack, "fpr", float("inf")),
                "auc": _get(pack, "auc", float("nan")),
                "oracle": True,
            }
        )

    for k in ["meta_raw", "meta_platt", "meta_iso"]:
        pack = results.get(k, {}) if isinstance(results, dict) else {}
        te = pack.get("test_eval", {}) if isinstance(pack, dict) else {}
        orc = pack.get("oracle_test", {}) if isinstance(pack, dict) else {}
        out.append(
            {
                "name": f"{k}::valsel",
                "fpr": _get(te, "fpr", float("inf")),
                "auc": _get(te, "auc", float("nan")),
                "oracle": False,
            }
        )
        out.append(
            {
                "name": f"{k}::oracle",
                "fpr": _get(orc, "fpr", float("inf")),
                "auc": _get(orc, "auc", float("nan")),
                "oracle": True,
            }
        )

    return out


def _load_model_scores(
    model_name: str,
    run_dir: Path,
    file_name: str,
    val_keys: List[str],
    test_keys: List[str],
    y_val_ref: np.ndarray,
    y_test_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    npz_path = run_dir / file_name
    z = base._load_npz(npz_path)

    yv = np.asarray(z["labels_val"], dtype=np.float32)
    yt = np.asarray(z["labels_test"], dtype=np.float32)
    if not np.array_equal(y_val_ref, yv):
        raise RuntimeError(f"Validation labels mismatch: {model_name} ({npz_path})")
    if not np.array_equal(y_test_ref, yt):
        raise RuntimeError(f"Test labels mismatch: {model_name} ({npz_path})")

    s_val = base._pick_score(z, val_keys, "val", ref_len=y_val_ref.size)
    s_test = base._pick_score(z, test_keys, "test", ref_len=y_test_ref.size)
    return s_val, s_test, str(npz_path.resolve())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fusion analysis for previous 18 + dualreco frozen family + m2 delta000/delta020"
    )

    # Previous 18-model set
    ap.add_argument("--joint_delta_run_dir", type=str, required=True)
    ap.add_argument("--reco_teacher_s09_run_dir", type=str, required=True)
    ap.add_argument("--corrected_s01_run_dir", type=str, required=True)
    ap.add_argument("--joint_s01_run_dir", type=str, required=True)
    ap.add_argument("--concat_run_dir", type=str, required=True)

    ap.add_argument("--m7_residual_run_dir", type=str, required=True)
    ap.add_argument("--m8_direct_residual_run_dir", type=str, required=True)
    ap.add_argument("--m9_low_run_dir", type=str, required=True)
    ap.add_argument("--m9_mid_run_dir", type=str, required=True)
    ap.add_argument("--m9_high_run_dir", type=str, required=True)

    ap.add_argument("--m4_k40_run_dir", type=str, required=True)
    ap.add_argument("--m4_k60_run_dir", type=str, required=True)
    ap.add_argument("--m4_k80_run_dir", type=str, required=True)

    ap.add_argument("--m10_run_dir", type=str, required=True)
    ap.add_argument("--m11_run_dir", type=str, required=True)
    ap.add_argument("--m12_run_dir", type=str, required=True)
    ap.add_argument("--m13_run_dir", type=str, required=True)

    # New m2 delta ablations
    ap.add_argument("--m2_delta000_run_dir", type=str, required=True)
    ap.add_argument("--m2_delta020_run_dir", type=str, required=True)

    # New dualreco-dualview family (use frozen scores)

    ap.add_argument("--m11_dual_run_dir", type=str, required=True)
    ap.add_argument("--m12_dual_run_dir", type=str, required=True)
    ap.add_argument("--m13_dual_run_dir", type=str, required=True)
    ap.add_argument("--m15_dual_low_run_dir", type=str, required=True)
    ap.add_argument("--m15_dual_mid_run_dir", type=str, required=True)
    ap.add_argument("--m15_dual_high_run_dir", type=str, required=True)
    ap.add_argument("--m16_dual_k40_run_dir", type=str, required=True)
    ap.add_argument("--m16_dual_k60_run_dir", type=str, required=True)
    ap.add_argument("--m16_dual_k80_run_dir", type=str, required=True)
    ap.add_argument("--m17_dual_run_dir", type=str, required=True)
    ap.add_argument("--m19_dual_run_dir", type=str, required=True)

    ap.add_argument("--target_tpr", type=float, default=0.50)
    ap.add_argument("--weight_step_2", type=float, default=0.01)
    ap.add_argument("--weight_samples_multi", type=int, default=12000)
    ap.add_argument("--pair_grid_step_multi", type=float, default=0.05)
    ap.add_argument("--meta_sel_frac", type=float, default=0.30)
    ap.add_argument("--meta_c_grid", type=str, default="0.05,0.1,0.3,1,3,10,30")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_name", type=str, default="fusion_hlt_joint31_analysis.json")
    args = ap.parse_args()

    target_tpr = float(args.target_tpr)

    dir_joint_delta = Path(args.joint_delta_run_dir)
    z2 = base._load_npz(dir_joint_delta / "fusion_scores_val_test.npz")
    y_val = np.asarray(z2["labels_val"], dtype=np.float32)
    y_test = np.asarray(z2["labels_test"], dtype=np.float32)

    scores_val: Dict[str, np.ndarray] = {
        "hlt": np.asarray(z2["preds_hlt_val"], dtype=np.float64),
        "joint_delta": np.asarray(z2["preds_joint_val"], dtype=np.float64),
    }
    scores_test: Dict[str, np.ndarray] = {
        "hlt": np.asarray(z2["preds_hlt_test"], dtype=np.float64),
        "joint_delta": np.asarray(z2["preds_joint_test"], dtype=np.float64),
    }

    score_files = {"joint_delta": str((dir_joint_delta / "fusion_scores_val_test.npz").resolve())}

    model_specs = [
        # Previous 18 additions beyond hlt/joint_delta
        ("reco_teacher_s09", Path(args.reco_teacher_s09_run_dir), "stageA_only_scores.npz", ["preds_reco_teacher_val"], ["preds_reco_teacher_test"]),
        ("corrected_s01", Path(args.corrected_s01_run_dir), "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),
        ("joint_s01", Path(args.joint_s01_run_dir), "fusion_scores_val_test.npz", ["preds_joint_val"], ["preds_joint_test"]),
        ("concat_corrected", Path(args.concat_run_dir), "concat_teacher_stageA_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),

        ("residual_m7", Path(args.m7_residual_run_dir), "stageA_residual_scores.npz", ["preds_residual_joint_val", "preds_residual_frozen_val"], ["preds_residual_joint_test", "preds_residual_frozen_test"]),
        ("direct_residual_m8", Path(args.m8_direct_residual_run_dir), "stageA_residual_scores.npz", ["preds_residual_joint_val", "preds_residual_frozen_val"], ["preds_residual_joint_test", "preds_residual_frozen_test"]),
        ("offdrop_low", Path(args.m9_low_run_dir), "stageA_residual_scores.npz", ["preds_residual_joint_val", "preds_residual_frozen_val"], ["preds_residual_joint_test", "preds_residual_frozen_test"]),
        ("offdrop_mid", Path(args.m9_mid_run_dir), "stageA_residual_scores.npz", ["preds_residual_joint_val", "preds_residual_frozen_val"], ["preds_residual_joint_test", "preds_residual_frozen_test"]),
        ("offdrop_high", Path(args.m9_high_run_dir), "stageA_residual_scores.npz", ["preds_residual_joint_val", "preds_residual_frozen_val"], ["preds_residual_joint_test", "preds_residual_frozen_test"]),

        ("corrected_k40", Path(args.m4_k40_run_dir), "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),
        ("corrected_k60", Path(args.m4_k60_run_dir), "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),
        ("corrected_k80", Path(args.m4_k80_run_dir), "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),

        ("antioverlap_m10", Path(args.m10_run_dir), "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),
        ("feat_noangle_m11", Path(args.m11_run_dir), "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),
        ("feat_noscale_m12", Path(args.m12_run_dir), "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),
        ("feat_coreshape_m13", Path(args.m13_run_dir), "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),

        # New m2 delta ablations
        ("joint_delta000", Path(args.m2_delta000_run_dir), "fusion_scores_val_test.npz", ["preds_joint_val"], ["preds_joint_test"]),
        ("joint_delta020", Path(args.m2_delta020_run_dir), "fusion_scores_val_test.npz", ["preds_joint_val"], ["preds_joint_test"]),

        # New dualreco frozen additions

        ("dual_m11_noangle", Path(args.m11_dual_run_dir), "dualreco_dualview_scores.npz", ["preds_dual_frozen_val", "preds_dualview_frozen_val"], ["preds_dual_frozen_test", "preds_dualview_frozen_test"]),
        ("dual_m12_noscale", Path(args.m12_dual_run_dir), "dualreco_dualview_scores.npz", ["preds_dual_frozen_val", "preds_dualview_frozen_val"], ["preds_dual_frozen_test", "preds_dualview_frozen_test"]),
        ("dual_m13_coreshape", Path(args.m13_dual_run_dir), "dualreco_dualview_scores.npz", ["preds_dual_frozen_val", "preds_dualview_frozen_val"], ["preds_dual_frozen_test", "preds_dualview_frozen_test"]),
        ("dual_m15_offdrop_low", Path(args.m15_dual_low_run_dir), "dualreco_dualview_scores.npz", ["preds_dual_frozen_val", "preds_dualview_frozen_val"], ["preds_dual_frozen_test", "preds_dualview_frozen_test"]),
        ("dual_m15_offdrop_mid", Path(args.m15_dual_mid_run_dir), "dualreco_dualview_scores.npz", ["preds_dual_frozen_val", "preds_dualview_frozen_val"], ["preds_dual_frozen_test", "preds_dualview_frozen_test"]),
        ("dual_m15_offdrop_high", Path(args.m15_dual_high_run_dir), "dualreco_dualview_scores.npz", ["preds_dual_frozen_val", "preds_dualview_frozen_val"], ["preds_dual_frozen_test", "preds_dualview_frozen_test"]),
        ("dual_m16_topk40", Path(args.m16_dual_k40_run_dir), "dualreco_dualview_scores.npz", ["preds_dual_frozen_val", "preds_dualview_frozen_val"], ["preds_dual_frozen_test", "preds_dualview_frozen_test"]),
        ("dual_m16_topk60", Path(args.m16_dual_k60_run_dir), "dualreco_dualview_scores.npz", ["preds_dual_frozen_val", "preds_dualview_frozen_val"], ["preds_dual_frozen_test", "preds_dualview_frozen_test"]),
        ("dual_m16_topk80", Path(args.m16_dual_k80_run_dir), "dualreco_dualview_scores.npz", ["preds_dual_frozen_val", "preds_dualview_frozen_val"], ["preds_dual_frozen_test", "preds_dualview_frozen_test"]),
        ("dual_m17_antioverlap", Path(args.m17_dual_run_dir), "dualreco_dualview_scores.npz", ["preds_dual_frozen_val", "preds_dualview_frozen_val"], ["preds_dual_frozen_test", "preds_dualview_frozen_test"]),
        ("dual_m19_basic", Path(args.m19_dual_run_dir), "dualreco_dualview_scores.npz", ["preds_dual_frozen_val", "preds_dualview_frozen_val"], ["preds_dual_frozen_test", "preds_dualview_frozen_test"]),
    ]

    for name, run_dir, file_name, val_keys, test_keys in model_specs:
        s_val, s_test, src = _load_model_scores(
            model_name=name,
            run_dir=run_dir,
            file_name=file_name,
            val_keys=val_keys,
            test_keys=test_keys,
            y_val_ref=y_val,
            y_test_ref=y_test,
        )
        scores_val[name] = s_val
        scores_test[name] = s_test
        score_files[name] = src

    model_order = [
        "hlt",
        "joint_delta",
        "reco_teacher_s09",
        "corrected_s01",
        "joint_s01",
        "concat_corrected",
        "residual_m7",
        "direct_residual_m8",
        "offdrop_low",
        "offdrop_mid",
        "offdrop_high",
        "corrected_k40",
        "corrected_k60",
        "corrected_k80",
        "antioverlap_m10",
        "feat_noangle_m11",
        "feat_noscale_m12",
        "feat_coreshape_m13",
        "joint_delta000",
        "joint_delta020",
        "dual_m11_noangle",
        "dual_m12_noscale",
        "dual_m13_coreshape",
        "dual_m15_offdrop_low",
        "dual_m15_offdrop_mid",
        "dual_m15_offdrop_high",
        "dual_m16_topk40",
        "dual_m16_topk60",
        "dual_m16_topk80",
        "dual_m17_antioverlap",
        "dual_m19_basic",
    ]

    for n in model_order:
        if n not in scores_val:
            raise KeyError(f"Missing model score: {n}")

    indiv: Dict[str, Dict[str, float]] = {}
    for name in model_order:
        v = base.auc_and_fpr_at_target(y_val, scores_val[name], target_tpr)
        t = base.auc_and_fpr_at_target(y_test, scores_test[name], target_tpr)
        indiv[name] = {
            "auc_val": float(v["auc"]),
            "fpr_val": float(v["fpr_at_target_tpr"]),
            "auc_test": float(t["auc"]),
            "fpr_test": float(t["fpr_at_target_tpr"]),
        }

    overlap_test = m.build_overlap_report_at_tpr(
        labels=y_test,
        model_preds={k: scores_test[k] for k in model_order},
        target_tpr=target_tpr,
    )

    pair_results_valsel: Dict[str, Dict[str, object]] = {}
    pair_results_oracle: Dict[str, Dict[str, object]] = {}
    for other in model_order[1:]:
        key = f"hlt_plus_{other}"
        pair_results_valsel[key] = m.select_weighted_combo_on_val_and_eval_test(
            labels_val=y_val,
            preds_a_val=scores_val["hlt"],
            preds_b_val=scores_val[other],
            labels_test=y_test,
            preds_a_test=scores_test["hlt"],
            preds_b_test=scores_test[other],
            name_a="hlt",
            name_b=other,
            target_tpr=target_tpr,
            weight_step=float(args.weight_step_2),
        )
        po = m.search_best_weighted_combo_at_tpr(
            labels=y_test,
            preds_a=scores_test["hlt"],
            preds_b=scores_test[other],
            name_a="hlt",
            name_b=other,
            target_tpr=target_tpr,
            weight_step=float(args.weight_step_2),
        )
        ps = po["w_a"] * scores_test["hlt"] + po["w_b"] * scores_test[other]
        pa = base.auc_and_fpr_at_target(y_test, ps, target_tpr)
        po["auc"] = float(pa["auc"])
        po["fpr_at_target_tpr_exact"] = float(pa["fpr_at_target_tpr"])
        pair_results_oracle[key] = po

    weight_candidates = base.generate_weight_candidates(
        n_models=len(model_order),
        n_random=int(args.weight_samples_multi),
        seed=int(args.seed),
        include_pair_grid=True,
        pair_step=float(args.pair_grid_step_multi),
    )

    mat_val = np.vstack([scores_val[n] for n in model_order])
    mat_test = np.vstack([scores_test[n] for n in model_order])

    all_weighted_raw_valsel = base.select_weighted_combo_multi_on_val_eval_test(
        y_val=y_val,
        score_mat_val=mat_val,
        y_test=y_test,
        score_mat_test=mat_test,
        model_names=model_order,
        target_tpr=target_tpr,
        weight_candidates=weight_candidates,
    )
    all_weighted_raw_oracle = base.search_best_weighted_combo_multi_at_tpr(
        labels=y_test,
        score_mat=mat_test,
        model_names=model_order,
        target_tpr=target_tpr,
        weight_candidates=weight_candidates,
    )

    cal_platt_meta: Dict[str, Dict[str, float]] = {}
    cal_iso_meta: Dict[str, Dict[str, float]] = {}
    scores_platt_val: Dict[str, np.ndarray] = {}
    scores_platt_test: Dict[str, np.ndarray] = {}
    scores_iso_val: Dict[str, np.ndarray] = {}
    scores_iso_test: Dict[str, np.ndarray] = {}

    for name in model_order:
        pv, pt, pm = base.calibrate_platt(y_val, scores_val[name], scores_test[name])
        iv, it, im = base.calibrate_isotonic(y_val, scores_val[name], scores_test[name])
        scores_platt_val[name] = pv
        scores_platt_test[name] = pt
        scores_iso_val[name] = iv
        scores_iso_test[name] = it
        cal_platt_meta[name] = pm
        cal_iso_meta[name] = im

    mat_platt_val = np.vstack([scores_platt_val[n] for n in model_order])
    mat_platt_test = np.vstack([scores_platt_test[n] for n in model_order])
    mat_iso_val = np.vstack([scores_iso_val[n] for n in model_order])
    mat_iso_test = np.vstack([scores_iso_test[n] for n in model_order])

    all_weighted_platt_valsel = base.select_weighted_combo_multi_on_val_eval_test(
        y_val=y_val,
        score_mat_val=mat_platt_val,
        y_test=y_test,
        score_mat_test=mat_platt_test,
        model_names=model_order,
        target_tpr=target_tpr,
        weight_candidates=weight_candidates,
    )
    all_weighted_platt_oracle = base.search_best_weighted_combo_multi_at_tpr(
        labels=y_test,
        score_mat=mat_platt_test,
        model_names=model_order,
        target_tpr=target_tpr,
        weight_candidates=weight_candidates,
    )

    all_weighted_iso_valsel = base.select_weighted_combo_multi_on_val_eval_test(
        y_val=y_val,
        score_mat_val=mat_iso_val,
        y_test=y_test,
        score_mat_test=mat_iso_test,
        model_names=model_order,
        target_tpr=target_tpr,
        weight_candidates=weight_candidates,
    )
    all_weighted_iso_oracle = base.search_best_weighted_combo_multi_at_tpr(
        labels=y_test,
        score_mat=mat_iso_test,
        model_names=model_order,
        target_tpr=target_tpr,
        weight_candidates=weight_candidates,
    )

    c_grid = [float(x.strip()) for x in str(args.meta_c_grid).split(",") if x.strip()]
    meta_raw = base.train_select_meta_fuser(
        X_val=base.build_meta_features(model_order, scores_val),
        y_val=y_val,
        X_test=base.build_meta_features(model_order, scores_test),
        y_test=y_test,
        target_tpr=target_tpr,
        sel_frac=float(args.meta_sel_frac),
        c_grid=c_grid,
        seed=int(args.seed),
    )
    meta_platt = base.train_select_meta_fuser(
        X_val=base.build_meta_features(model_order, scores_platt_val),
        y_val=y_val,
        X_test=base.build_meta_features(model_order, scores_platt_test),
        y_test=y_test,
        target_tpr=target_tpr,
        sel_frac=float(args.meta_sel_frac),
        c_grid=c_grid,
        seed=int(args.seed),
    )
    meta_iso = base.train_select_meta_fuser(
        X_val=base.build_meta_features(model_order, scores_iso_val),
        y_val=y_val,
        X_test=base.build_meta_features(model_order, scores_iso_test),
        y_test=y_test,
        target_tpr=target_tpr,
        sel_frac=float(args.meta_sel_frac),
        c_grid=c_grid,
        seed=int(args.seed),
    )

    n_models = len(model_order)

    run_dirs = {
        k: str(v)
        for k, v in vars(args).items()
        if k.endswith("_run_dir")
    }
    run_dirs["score_files"] = score_files

    results = {
        "config": {
            "target_tpr": target_tpr,
            "weight_step_2": float(args.weight_step_2),
            "weight_samples_multi": int(args.weight_samples_multi),
            "pair_grid_step_multi": float(args.pair_grid_step_multi),
            "meta_sel_frac": float(args.meta_sel_frac),
            "meta_c_grid": c_grid,
            "seed": int(args.seed),
            "n_models": int(n_models),
        },
        "run_dirs": run_dirs,
        "models_order": model_order,
        "individual": indiv,
        "overlap_test": overlap_test,
        "pair_results_valsel": pair_results_valsel,
        "pair_results_oracle": pair_results_oracle,
        f"all{int(n_models)}_weighted_raw_valsel": all_weighted_raw_valsel,
        f"all{int(n_models)}_weighted_raw_oracle": all_weighted_raw_oracle,
        f"all{int(n_models)}_weighted_platt_valsel": all_weighted_platt_valsel,
        f"all{int(n_models)}_weighted_platt_oracle": all_weighted_platt_oracle,
        f"all{int(n_models)}_weighted_iso_valsel": all_weighted_iso_valsel,
        f"all{int(n_models)}_weighted_iso_oracle": all_weighted_iso_oracle,
        "calibration": {
            "platt": cal_platt_meta,
            "isotonic": cal_iso_meta,
        },
        "meta_raw": {
            "selection": meta_raw["selection"],
            "test_eval": meta_raw["test_eval"],
            "oracle_test": meta_raw["oracle_test"],
        },
        "meta_platt": {
            "selection": meta_platt["selection"],
            "test_eval": meta_platt["test_eval"],
            "oracle_test": meta_platt["oracle_test"],
        },
        "meta_iso": {
            "selection": meta_iso["selection"],
            "test_eval": meta_iso["test_eval"],
            "oracle_test": meta_iso["oracle_test"],
        },
    }

    all_candidates = _collect_candidates(results, n_models=n_models)
    non_oracle = [x for x in all_candidates if not bool(x["oracle"])]
    oracle = [x for x in all_candidates if bool(x["oracle"])]
    non_oracle_sorted = sorted(non_oracle, key=lambda d: float(d["fpr"]))
    oracle_sorted = sorted(oracle, key=lambda d: float(d["fpr"]))
    results["best_summary"] = {
        "best_non_oracle": non_oracle_sorted[0] if non_oracle_sorted else None,
        "best_oracle": oracle_sorted[0] if oracle_sorted else None,
        "top10_non_oracle": non_oracle_sorted[:10],
        "top10_oracle": oracle_sorted[:10],
    }

    out_path = dir_joint_delta / str(args.output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    b = results["best_summary"]["best_non_oracle"]
    if b is not None:
        print(
            f"Best non-oracle @TPR={target_tpr:.2f}: {b['name']} | "
            f"FPR={float(b['fpr']):.6f} | AUC={float(b['auc']):.6f}"
        )
    bo = results["best_summary"]["best_oracle"]
    if bo is not None:
        print(
            f"Best oracle @TPR={target_tpr:.2f}: {bo['name']} | "
            f"FPR={float(bo['fpr']):.6f}"
        )

    print(f"Models fused: {n_models}")
    print(f"Saved fusion analysis to: {out_path}")


if __name__ == "__main__":
    main()
