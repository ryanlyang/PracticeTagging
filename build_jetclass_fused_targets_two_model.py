#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build per-jet fused soft targets from two trained JetClass dual-view runs.

This is for training an offline distilled student:
  offline jet features -> fused teacher probabilities (bin-gated val-selected fusion).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import analyze_jetclass_two_model_bin_gated_fusion as fusion_utils
import train_jetclass_joint_dualview_stage2_unmergeonly_v2_attr as v2
from evaluate_jetclass_hlt_teacher_baseline import (
    CANONICAL_CLASS_ORDER,
    HLTParams,
    collect_files_by_class,
    compute_features,
    get_mean_std,
    load_split,
    split_files_by_class,
    standardize,
)


class JointEvalDataset(Dataset):
    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt4: np.ndarray,
        labels: np.ndarray,
    ):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt4 = torch.tensor(const_hlt4, dtype=torch.float32)
        self.labels = torch.tensor(labels.astype(np.int64), dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt4": self.const_hlt4[i],
            "label": self.labels[i],
        }


def _ns_from_json(path: Path) -> SimpleNamespace:
    return SimpleNamespace(**json.loads(path.read_text()))


def _verify_run_compat(args_a: SimpleNamespace, args_b: SimpleNamespace) -> None:
    keys = (
        "feature_mode",
        "feature_preprocessing",
        "class_assignment",
        "target_class",
        "background_class",
        "n_train_jets",
        "n_val_jets",
        "n_test_jets",
        "max_constits",
        "train_files_per_class",
        "val_files_per_class",
        "test_files_per_class",
        "shuffle_files",
        "hlt_pt_threshold",
        "merge_prob_scale",
        "reassign_scale",
        "smear_scale",
        "eff_plateau_barrel",
        "eff_plateau_endcap",
        "eff_turnon_pt",
        "eff_width_pt",
    )
    for k in keys:
        if str(getattr(args_a, k)) != str(getattr(args_b, k)):
            raise ValueError(f"Run mismatch for `{k}`: A={getattr(args_a, k)} vs B={getattr(args_b, k)}")


def _build_hlt_data_all(
    args_ref: SimpleNamespace,
    data_dir: Path,
) -> Tuple[
    Sequence[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    files_by_class = collect_files_by_class(data_dir.resolve())
    if str(args_ref.class_assignment) == "canonical_labels":
        class_names = list(CANONICAL_CLASS_ORDER)
    else:
        class_names = sorted(files_by_class.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    tr_files, va_files, te_files = split_files_by_class(
        files_by_class,
        n_train=int(args_ref.train_files_per_class),
        n_val=int(args_ref.val_files_per_class),
        n_test=int(args_ref.test_files_per_class),
        shuffle=bool(args_ref.shuffle_files),
        seed=int(args_ref.seed),
    )

    tr_tok_raw, tr_mask_raw, tr_y = load_split(
        tr_files,
        n_total=int(args_ref.n_train_jets),
        max_constits=int(args_ref.max_constits),
        class_to_idx=class_to_idx,
        seed=int(args_ref.seed) + 101,
        class_assignment=str(args_ref.class_assignment),
    )
    va_tok_raw, va_mask_raw, va_y = load_split(
        va_files,
        n_total=int(args_ref.n_val_jets),
        max_constits=int(args_ref.max_constits),
        class_to_idx=class_to_idx,
        seed=int(args_ref.seed) + 202,
        class_assignment=str(args_ref.class_assignment),
    )
    te_tok_raw, te_mask_raw, te_y = load_split(
        te_files,
        n_total=int(args_ref.n_test_jets),
        max_constits=int(args_ref.max_constits),
        class_to_idx=class_to_idx,
        seed=int(args_ref.seed) + 303,
        class_assignment=str(args_ref.class_assignment),
    )

    hlt_params = HLTParams(
        hlt_pt_threshold=float(args_ref.hlt_pt_threshold),
        merge_prob_scale=float(args_ref.merge_prob_scale),
        reassign_scale=float(args_ref.reassign_scale),
        smear_scale=float(args_ref.smear_scale),
        eff_plateau_barrel=float(args_ref.eff_plateau_barrel),
        eff_plateau_endcap=float(args_ref.eff_plateau_endcap),
        eff_turnon_pt=float(args_ref.eff_turnon_pt),
        eff_width_pt=float(args_ref.eff_width_pt),
    )

    tr_hlt_tok_raw, tr_hlt_mask_raw, _, _ = v2.build_hlt_view(
        tr_tok_raw,
        tr_mask_raw,
        params=hlt_params,
        seed=int(args_ref.seed) + 1001,
        return_provenance=True,
    )
    va_hlt_tok_raw, va_hlt_mask_raw, _, _ = v2.build_hlt_view(
        va_tok_raw,
        va_mask_raw,
        params=hlt_params,
        seed=int(args_ref.seed) + 1002,
        return_provenance=True,
    )
    te_hlt_tok_raw, te_hlt_mask_raw, _, _ = v2.build_hlt_view(
        te_tok_raw,
        te_mask_raw,
        params=hlt_params,
        seed=int(args_ref.seed) + 1003,
        return_provenance=True,
    )

    tr_feat_hlt = compute_features(
        tr_hlt_tok_raw,
        tr_hlt_mask_raw,
        feature_mode=str(args_ref.feature_mode),
        feature_preprocessing=str(args_ref.feature_preprocessing),
    )
    va_feat_hlt = compute_features(
        va_hlt_tok_raw,
        va_hlt_mask_raw,
        feature_mode=str(args_ref.feature_mode),
        feature_preprocessing=str(args_ref.feature_preprocessing),
    )
    te_feat_hlt = compute_features(
        te_hlt_tok_raw,
        te_hlt_mask_raw,
        feature_mode=str(args_ref.feature_mode),
        feature_preprocessing=str(args_ref.feature_preprocessing),
    )

    if str(args_ref.feature_preprocessing) != "canonical":
        tr_feat_off = compute_features(
            tr_tok_raw,
            tr_mask_raw,
            feature_mode=str(args_ref.feature_mode),
            feature_preprocessing=str(args_ref.feature_preprocessing),
        )
        idx_all = np.arange(len(tr_y))
        mean, std = get_mean_std(tr_feat_off, tr_mask_raw, idx_all)
        tr_feat_hlt = standardize(tr_feat_hlt, tr_hlt_mask_raw, mean, std)
        va_feat_hlt = standardize(va_feat_hlt, va_hlt_mask_raw, mean, std)
        te_feat_hlt = standardize(te_feat_hlt, te_hlt_mask_raw, mean, std)

    tr_hlt_const4 = tr_hlt_tok_raw[:, :, :4].astype(np.float32)
    va_hlt_const4 = va_hlt_tok_raw[:, :, :4].astype(np.float32)
    te_hlt_const4 = te_hlt_tok_raw[:, :, :4].astype(np.float32)

    return (
        class_names,
        tr_feat_hlt.astype(np.float32),
        tr_hlt_mask_raw,
        tr_hlt_const4,
        tr_y.astype(np.int64),
        va_feat_hlt.astype(np.float32),
        va_hlt_mask_raw,
        va_hlt_const4,
        va_y.astype(np.int64),
        te_feat_hlt.astype(np.float32),
        te_hlt_mask_raw,
        te_hlt_const4,
        te_y.astype(np.int64),
    )


def _build_loader(
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt4: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    ds = JointEvalDataset(feat_hlt, mask_hlt, const_hlt4, y)
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )


def _fuse(pa: np.ndarray, pb: np.ndarray, wa: float) -> np.ndarray:
    wa = float(np.clip(wa, 0.0, 1.0))
    return wa * pa + (1.0 - wa) * pb


def _build_quantile_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    q = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(x, q)
    edges = np.asarray(edges, dtype=np.float64)
    for i in range(1, len(edges)):
        if edges[i] < edges[i - 1]:
            edges[i] = edges[i - 1]
    return edges


def _bin_ids(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    b = np.searchsorted(edges[1:-1], x, side="right")
    return np.clip(b, 0, len(edges) - 2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build fused soft targets from two JetClass runs")
    ap.add_argument("--run_a_dir", type=Path, required=True)
    ap.add_argument("--run_b_dir", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, default=Path("/home/ryreu/atlas/PracticeTagging/data/jetclass_part0"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--weight_step", type=float, default=0.01)
    ap.add_argument("--n_bins", type=int, default=12)
    ap.add_argument("--min_bin_count", type=int, default=1200)
    ap.add_argument("--optimize_for", type=str, default="sigbg_fpr50", choices=["sigbg_fpr50", "targetbg_fpr50"])
    ap.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    args = ap.parse_args()

    run_a_dir = args.run_a_dir.resolve()
    run_b_dir = args.run_b_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    args_a = _ns_from_json(run_a_dir / "args.json")
    args_b = _ns_from_json(run_b_dir / "args.json")
    _verify_run_compat(args_a, args_b)

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device).startswith("cpu") else "cpu")

    (
        class_names,
        tr_feat_hlt,
        tr_hlt_mask,
        tr_hlt_const4,
        tr_y,
        va_feat_hlt,
        va_hlt_mask,
        va_hlt_const4,
        va_y,
        te_feat_hlt,
        te_hlt_mask,
        te_hlt_const4,
        te_y,
    ) = _build_hlt_data_all(args_a, args.data_dir)

    input_dim = int(tr_feat_hlt.shape[-1])
    n_classes = int(len(class_names))
    reco_a, dual_a, src_a = fusion_utils._build_models_for_run(run_a_dir, args_a, input_dim, n_classes, device)
    reco_b, dual_b, src_b = fusion_utils._build_models_for_run(run_b_dir, args_b, input_dim, n_classes, device)

    dl_tr = _build_loader(tr_feat_hlt, tr_hlt_mask, tr_hlt_const4, tr_y, args.batch_size, args.num_workers)
    dl_va = _build_loader(va_feat_hlt, va_hlt_mask, va_hlt_const4, va_y, args.batch_size, args.num_workers)
    dl_te = _build_loader(te_feat_hlt, te_hlt_mask, te_hlt_const4, te_y, args.batch_size, args.num_workers)

    pa_tr, ytr_a = fusion_utils._predict_probs(reco_a, dual_a, dl_tr, device, corrected_weight_floor=float(args.corrected_weight_floor))
    pb_tr, ytr_b = fusion_utils._predict_probs(reco_b, dual_b, dl_tr, device, corrected_weight_floor=float(args.corrected_weight_floor))
    pa_va, yva_a = fusion_utils._predict_probs(reco_a, dual_a, dl_va, device, corrected_weight_floor=float(args.corrected_weight_floor))
    pb_va, yva_b = fusion_utils._predict_probs(reco_b, dual_b, dl_va, device, corrected_weight_floor=float(args.corrected_weight_floor))
    pa_te, yte_a = fusion_utils._predict_probs(reco_a, dual_a, dl_te, device, corrected_weight_floor=float(args.corrected_weight_floor))
    pb_te, yte_b = fusion_utils._predict_probs(reco_b, dual_b, dl_te, device, corrected_weight_floor=float(args.corrected_weight_floor))

    if not (np.array_equal(ytr_a, ytr_b) and np.array_equal(yva_a, yva_b) and np.array_equal(yte_a, yte_b)):
        raise RuntimeError("Label mismatch between run A and run B predictions.")

    bg_name = str(args_a.background_class)
    tgt_name = str(args_a.target_class)
    w_global, _ = fusion_utils._search_best_weight(
        pa=pa_va,
        pb=pb_va,
        y=yva_a,
        class_names=class_names,
        background_class=bg_name,
        target_class=tgt_name,
        weight_step=float(args.weight_step),
        optimize_for=str(args.optimize_for),
    )

    bg_idx = class_names.index(bg_name)
    s_va = 1.0 - 0.5 * (pa_va[:, bg_idx] + pb_va[:, bg_idx])
    s_tr = 1.0 - 0.5 * (pa_tr[:, bg_idx] + pb_tr[:, bg_idx])
    s_te = 1.0 - 0.5 * (pa_te[:, bg_idx] + pb_te[:, bg_idx])
    edges = _build_quantile_edges(s_va, int(args.n_bins))
    bid_va = _bin_ids(s_va, edges)
    bid_tr = _bin_ids(s_tr, edges)
    bid_te = _bin_ids(s_te, edges)

    w_bins = np.full((int(args.n_bins),), float(w_global), dtype=np.float64)
    counts_bins = np.zeros((int(args.n_bins),), dtype=np.int64)
    for b in range(int(args.n_bins)):
        idx = np.where(bid_va == b)[0]
        counts_bins[b] = int(idx.size)
        if idx.size < int(args.min_bin_count):
            w_bins[b] = float(w_global)
            continue
        wb, _ = fusion_utils._search_best_weight(
            pa=pa_va[idx],
            pb=pb_va[idx],
            y=yva_a[idx],
            class_names=class_names,
            background_class=bg_name,
            target_class=tgt_name,
            weight_step=float(args.weight_step),
            optimize_for=str(args.optimize_for),
        )
        w_bins[b] = float(wb)

    p_tr_global = _fuse(pa_tr, pb_tr, w_global)
    p_va_global = _fuse(pa_va, pb_va, w_global)
    p_te_global = _fuse(pa_te, pb_te, w_global)

    p_tr_bin = _fuse(pa_tr, pb_tr, 0.0)
    p_va_bin = _fuse(pa_va, pb_va, 0.0)
    p_te_bin = _fuse(pa_te, pb_te, 0.0)
    for b in range(int(args.n_bins)):
        w = float(w_bins[b])
        m_tr = bid_tr == b
        m_va = bid_va == b
        m_te = bid_te == b
        if np.any(m_tr):
            p_tr_bin[m_tr] = _fuse(pa_tr[m_tr], pb_tr[m_tr], w)
        if np.any(m_va):
            p_va_bin[m_va] = _fuse(pa_va[m_va], pb_va[m_va], w)
        if np.any(m_te):
            p_te_bin[m_te] = _fuse(pa_te[m_te], pb_te[m_te], w)

    np.savez_compressed(
        out_dir / "fused_targets_train_val_test.npz",
        y_train=ytr_a.astype(np.int64),
        y_val=yva_a.astype(np.int64),
        y_test=yte_a.astype(np.int64),
        probs_a_train=pa_tr.astype(np.float32),
        probs_b_train=pb_tr.astype(np.float32),
        probs_a_val=pa_va.astype(np.float32),
        probs_b_val=pb_va.astype(np.float32),
        probs_a_test=pa_te.astype(np.float32),
        probs_b_test=pb_te.astype(np.float32),
        probs_fused_global_train=p_tr_global.astype(np.float32),
        probs_fused_global_val=p_va_global.astype(np.float32),
        probs_fused_global_test=p_te_global.astype(np.float32),
        probs_fused_bin_train=p_tr_bin.astype(np.float32),
        probs_fused_bin_val=p_va_bin.astype(np.float32),
        probs_fused_bin_test=p_te_bin.astype(np.float32),
        score_train=s_tr.astype(np.float32),
        score_val=s_va.astype(np.float32),
        score_test=s_te.astype(np.float32),
        bin_id_train=bid_tr.astype(np.int64),
        bin_id_val=bid_va.astype(np.int64),
        bin_id_test=bid_te.astype(np.int64),
        bin_edges=edges.astype(np.float64),
        bin_weights=w_bins.astype(np.float64),
    )

    meta = {
        "run_a_dir": str(run_a_dir),
        "run_b_dir": str(run_b_dir),
        "run_a_sources": src_a,
        "run_b_sources": src_b,
        "data_dir": str(args.data_dir.resolve()),
        "out_dir": str(out_dir),
        "class_names": list(class_names),
        "target_class": tgt_name,
        "background_class": bg_name,
        "n_train": int(len(ytr_a)),
        "n_val": int(len(yva_a)),
        "n_test": int(len(yte_a)),
        "weight_step": float(args.weight_step),
        "n_bins": int(args.n_bins),
        "min_bin_count": int(args.min_bin_count),
        "optimize_for": str(args.optimize_for),
        "global_weight_a_valsel": float(w_global),
        "bin_weights_a_valsel": w_bins.tolist(),
        "bin_counts_val": counts_bins.tolist(),
        "run_ref_dir": str(run_a_dir),
    }
    (out_dir / "fused_targets_metadata.json").write_text(json.dumps(meta, indent=2))

    print("============================================================")
    print("Built fused soft targets")
    print("============================================================")
    print(f"Run A: {run_a_dir}")
    print(f"Run B: {run_b_dir}")
    print(f"Out:   {out_dir}")
    print(f"Global weight A: {w_global:.4f}")
    print(f"Saved: {out_dir / 'fused_targets_train_val_test.npz'}")
    print(f"Saved: {out_dir / 'fused_targets_metadata.json'}")


if __name__ == "__main__":
    main()

