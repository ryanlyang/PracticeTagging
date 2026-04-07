#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from unmerge_correct_hlt import (
    ParticleTransformer,
    compute_features,
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
    _build_train_file_list,
    _load_ckpt_state,
    _offline_mask,
)


def _deepcopy_cfg() -> Dict:
    return json.loads(json.dumps(BASE_CONFIG))


def _clip_probs(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


def _infer_teacher_input_dim(sd: Dict[str, torch.Tensor]) -> int:
    if "input_proj.0.weight" in sd:
        return int(sd["input_proj.0.weight"].shape[1])
    if "input_proj.weight" in sd:
        return int(sd["input_proj.weight"].shape[1])
    raise RuntimeError("Could not infer teacher input_dim from checkpoint keys")


def _threshold_at_target_tpr(y: np.ndarray, p: np.ndarray, target_tpr: float) -> float:
    y = np.asarray(y, dtype=np.int64)
    p = _clip_probs(p)
    pos = p[y == 1]
    if pos.size == 0:
        return 0.5
    q = float(np.clip(1.0 - float(target_tpr), 0.0, 1.0))
    return float(np.quantile(pos, q))


def _fpr_at_threshold(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    neg = (y == 0)
    denom = int(np.sum(neg))
    if denom == 0:
        return float("nan")
    return float(np.mean(p[neg] >= float(threshold)))


def _eval_metrics(
    y_val: np.ndarray,
    p_val: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
) -> Dict[str, float]:
    thr30 = _threshold_at_target_tpr(y_val, p_val, 0.30)
    thr50 = _threshold_at_target_tpr(y_val, p_val, 0.50)
    return {
        "auc_val": float(roc_auc_score(y_val, p_val)),
        "auc_test": float(roc_auc_score(y_test, p_test)),
        "fpr30_val": _fpr_at_threshold(y_val, p_val, thr30),
        "fpr30_test": _fpr_at_threshold(y_test, p_test, thr30),
        "fpr50_val": _fpr_at_threshold(y_val, p_val, thr50),
        "fpr50_test": _fpr_at_threshold(y_test, p_test, thr50),
        "thr30_from_val": float(thr30),
        "thr50_from_val": float(thr50),
    }


def _infer_teacher_probs_hlt_and_reco(
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
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(feat_hlt_std.shape[0])
    p_hlt = np.zeros((n,), dtype=np.float32)
    p_reco = np.zeros((n,), dtype=np.float32)

    teacher.eval()
    reconstructor.eval()

    with torch.no_grad():
        for s in range(0, n, int(batch_size)):
            e = min(n, s + int(batch_size))
            x = torch.from_numpy(feat_hlt_std[s:e]).to(device=device, dtype=torch.float32)
            m = torch.from_numpy(mask_hlt[s:e]).to(device=device, dtype=torch.bool)
            c = torch.from_numpy(const_hlt[s:e]).to(device=device, dtype=torch.float32)

            logit_hlt = teacher(x, m).squeeze(1)
            out = reconstructor(x, m, c, stage_scale=1.0)
            feat_corr, mask_corr = joint_base.build_soft_corrected_view(
                out,
                weight_floor=float(corrected_weight_floor),
                scale_features_by_weight=True,
                include_flags=bool(corrected_use_flags),
            )
            if int(feat_corr.shape[-1]) < int(teacher_input_dim):
                raise RuntimeError(
                    f"Teacher input_dim={teacher_input_dim} but corrected feature dim is {feat_corr.shape[-1]}"
                )
            feat_corr = feat_corr[:, :, : int(teacher_input_dim)]
            logit_reco = teacher(feat_corr, mask_corr).squeeze(1)

            p_hlt[s:e] = torch.sigmoid(logit_hlt).detach().cpu().numpy().astype(np.float32)
            p_reco[s:e] = torch.sigmoid(logit_reco).detach().cpu().numpy().astype(np.float32)

    return p_hlt, p_reco


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate teacher on HLT input vs reconstructed input from stage2 reconstructor.")
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--train_path", type=str, default="./data")
    ap.add_argument("--teacher_ckpt", type=str, default="")
    ap.add_argument("--reco_ckpt", type=str, default="")
    ap.add_argument("--max_constits", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    ap.add_argument("--corrected_use_flags", action="store_true")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--report_json", type=str, default="")
    ap.add_argument("--save_scores_npz", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    setup_path = run_dir / "data_setup.json"
    splits_path = run_dir / "data_splits.npz"
    if not setup_path.exists():
        raise FileNotFoundError(f"Missing: {setup_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing: {splits_path}")

    teacher_ckpt = Path(args.teacher_ckpt).expanduser().resolve() if str(args.teacher_ckpt).strip() else (run_dir / "teacher.pt")
    reco_ckpt = Path(args.reco_ckpt).expanduser().resolve() if str(args.reco_ckpt).strip() else (run_dir / "offline_reconstructor_stage2.pt")
    if not teacher_ckpt.exists():
        raise FileNotFoundError(f"Missing teacher checkpoint: {teacher_ckpt}")
    if not reco_ckpt.exists():
        raise FileNotFoundError(f"Missing reconstructor checkpoint: {reco_ckpt}")

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir).strip()
        else (run_dir / "teacher_hlt_vs_reco_stage2_eval")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    with setup_path.open("r", encoding="utf-8") as f:
        setup = json.load(f)
    splits = np.load(splits_path, allow_pickle=False)
    train_idx = splits["train_idx"].astype(np.int64)
    val_idx = splits["val_idx"].astype(np.int64)
    test_idx = splits["test_idx"].astype(np.int64)
    means = splits["means"].astype(np.float32)
    stds = splits["stds"].astype(np.float32)

    n_train_jets = int(setup["n_train_jets"])
    offset_jets = int(setup["offset_jets"])
    max_constits = int(args.max_constits)
    hlt_seed = int(setup.get("seed", 0))

    cfg = _deepcopy_cfg()
    if "hlt_effects" in setup:
        cfg["hlt_effects"].update(setup["hlt_effects"])

    train_files: List[Path] = _build_train_file_list(setup, args.train_path)
    max_jets_needed = int(offset_jets + n_train_jets)

    print("Loading offline constituents...")
    all_const_full, all_labels_full = load_raw_constituents_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=max_constits,
    )
    const_raw = all_const_full[offset_jets : offset_jets + n_train_jets]
    labels = all_labels_full[offset_jets : offset_jets + n_train_jets].astype(np.int64)

    print("Generating pseudo-HLT...")
    const_off, mask_off = _offline_mask(const_raw, float(cfg["hlt_effects"]["pt_threshold_offline"]))
    hlt_const, hlt_mask, _, _ = apply_hlt_effects_realistic_nomap(
        const_off,
        mask_off,
        cfg,
        seed=hlt_seed,
    )

    print("Computing/standardizing HLT features...")
    feat_hlt = compute_features(hlt_const, hlt_mask)
    feat_hlt_std = standardize(feat_hlt, hlt_mask, means, stds)

    device = torch.device(args.device)
    teacher_sd = _load_ckpt_state(teacher_ckpt, device)
    reco_sd = _load_ckpt_state(reco_ckpt, device)

    teacher_input_dim = _infer_teacher_input_dim(teacher_sd)
    teacher = ParticleTransformer(input_dim=teacher_input_dim, **cfg["model"]).to(device)
    teacher.load_state_dict(teacher_sd, strict=True)

    reconstructor = OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reconstructor.load_state_dict(reco_sd, strict=True)

    print("Running teacher on HLT and reconstructed views...")
    p_teacher_hlt, p_teacher_reco = _infer_teacher_probs_hlt_and_reco(
        teacher=teacher,
        reconstructor=reconstructor,
        feat_hlt_std=feat_hlt_std,
        mask_hlt=hlt_mask,
        const_hlt=hlt_const,
        batch_size=int(args.batch_size),
        device=device,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.corrected_use_flags),
        teacher_input_dim=teacher_input_dim,
    )

    y_val = labels[val_idx]
    y_test = labels[test_idx]

    hlt_metrics = _eval_metrics(y_val, p_teacher_hlt[val_idx], y_test, p_teacher_hlt[test_idx])
    reco_metrics = _eval_metrics(y_val, p_teacher_reco[val_idx], y_test, p_teacher_reco[test_idx])

    rep = {
        "run_dir": str(run_dir),
        "teacher_ckpt": str(teacher_ckpt),
        "reco_ckpt": str(reco_ckpt),
        "splits": {
            "n_train": int(train_idx.size),
            "n_val": int(val_idx.size),
            "n_test": int(test_idx.size),
        },
        "metrics": {
            "teacher_on_hlt": hlt_metrics,
            "teacher_on_reco_stage2": reco_metrics,
        },
    }

    rep_path = (
        Path(args.report_json).expanduser().resolve()
        if str(args.report_json).strip()
        else (out_dir / "teacher_hlt_vs_reco_report.json")
    )
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with rep_path.open("w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    if bool(args.save_scores_npz):
        np.savez_compressed(
            out_dir / "teacher_hlt_vs_reco_scores.npz",
            labels=labels.astype(np.int8),
            val_idx=val_idx.astype(np.int64),
            test_idx=test_idx.astype(np.int64),
            p_teacher_hlt=p_teacher_hlt.astype(np.float32),
            p_teacher_reco=p_teacher_reco.astype(np.float32),
        )

    print("=" * 72)
    print("Teacher-On-HLT vs Teacher-On-Reco(Stage2)")
    print("=" * 72)
    print(f"Run dir: {run_dir}")
    print(f"Teacher ckpt: {teacher_ckpt}")
    print(f"Reco ckpt: {reco_ckpt}")
    print()
    print("Val/Test metrics (thresholds from val positives):")
    print(
        "  Teacher on HLT        "
        f"AUC(val/test)={hlt_metrics['auc_val']:.6f}/{hlt_metrics['auc_test']:.6f} "
        f"FPR30(val/test)={hlt_metrics['fpr30_val']:.6f}/{hlt_metrics['fpr30_test']:.6f} "
        f"FPR50(val/test)={hlt_metrics['fpr50_val']:.6f}/{hlt_metrics['fpr50_test']:.6f}"
    )
    print(
        "  Teacher on RecoStage2 "
        f"AUC(val/test)={reco_metrics['auc_val']:.6f}/{reco_metrics['auc_test']:.6f} "
        f"FPR30(val/test)={reco_metrics['fpr30_val']:.6f}/{reco_metrics['fpr30_test']:.6f} "
        f"FPR50(val/test)={reco_metrics['fpr50_val']:.6f}/{reco_metrics['fpr50_test']:.6f}"
    )
    print()
    print(f"Saved report: {rep_path}")
    if bool(args.save_scores_npz):
        print(f"Saved scores: {out_dir / 'teacher_hlt_vs_reco_scores.npz'}")


if __name__ == "__main__":
    main()

