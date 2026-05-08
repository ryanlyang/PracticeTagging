#!/usr/bin/env python3
"""
Two-model bin-gated val-selected fusion for JetClass dual-view runs.

Workflow:
1) Rebuild val/test splits using run A's data config.
2) Load run A and run B Stage2 checkpoints (reconstructor + dual model).
3) Evaluate both on identical val/test HLT views.
4) Fit:
   - global val-selected weight w (A vs B),
   - per-bin val-selected weights (bin-gated fusion).
5) Report val/test metrics and save artifacts.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_confgen_ops as confgen_ops
import train_jetclass_joint_dualview_stage2_unmergeonly_v2_attr as v2
from evaluate_jetclass_hlt_teacher_baseline import (
    CANONICAL_CLASS_ORDER,
    HLTParams,
    collect_files_by_class,
    compute_features,
    eval_metrics,
    get_mean_std,
    load_split,
    split_files_by_class,
    standardize,
)
from offline_reconstructor_no_gt_local30kv2 import CONFIG as BASE_RECO_CONFIG


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


def _as_int(ns: SimpleNamespace, key: str) -> int:
    return int(getattr(ns, key))


def _as_float(ns: SimpleNamespace, key: str) -> float:
    return float(getattr(ns, key))


def _load_state_dict_from_run_dir(run_dir: Path, names: Sequence[str]) -> Tuple[Dict[str, torch.Tensor], Path]:
    tried: List[str] = []
    for n in names:
        p = (run_dir / n).resolve()
        tried.append(str(p))
        if not p.exists():
            continue
        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, dict) and "model" in obj:
            return obj["model"], p
        if isinstance(obj, dict):
            return obj, p
    raise FileNotFoundError(f"No checkpoint found in {run_dir}. Tried: {tried}")


def _build_eval_data(args_ref: SimpleNamespace, data_dir: Path) -> Tuple[
    Sequence[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str
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
    standardization_mode = "canonical_manual_fixed"
    if str(args_ref.feature_preprocessing) != "canonical":
        tr_feat_off = compute_features(
            tr_tok_raw,
            tr_mask_raw,
            feature_mode=str(args_ref.feature_mode),
            feature_preprocessing=str(args_ref.feature_preprocessing),
        )
        idx_all = np.arange(len(tr_y))
        mean, std = get_mean_std(tr_feat_off, tr_mask_raw, idx_all)
        va_feat_hlt = standardize(va_feat_hlt, va_hlt_mask_raw, mean, std)
        te_feat_hlt = standardize(te_feat_hlt, te_hlt_mask_raw, mean, std)
        standardization_mode = "learned_train_split"

    va_hlt_const4 = va_hlt_tok_raw[:, :, :4].astype(np.float32)
    te_hlt_const4 = te_hlt_tok_raw[:, :, :4].astype(np.float32)
    return (
        class_names,
        va_feat_hlt.astype(np.float32),
        va_hlt_mask_raw,
        va_hlt_const4,
        va_y.astype(np.int64),
        te_feat_hlt.astype(np.float32),
        te_hlt_mask_raw,
        te_hlt_const4,
        te_y.astype(np.int64),
        standardization_mode,
    )


def _build_models_for_run(
    run_dir: Path,
    run_args: SimpleNamespace,
    input_dim: int,
    n_classes: int,
    device: torch.device,
) -> Tuple[torch.nn.Module, torch.nn.Module, Dict[str, str]]:
    reco_cfg = copy.deepcopy(BASE_RECO_CONFIG)
    reco_cfg["reconstructor_model"]["embed_dim"] = int(run_args.reco_embed_dim)
    reco_cfg["reconstructor_model"]["num_heads"] = int(run_args.reco_num_heads)
    reco_cfg["reconstructor_model"]["num_layers"] = int(run_args.reco_num_layers)
    reco_cfg["reconstructor_model"]["ff_dim"] = int(run_args.reco_ff_dim)
    reco_cfg["reconstructor_model"]["dropout"] = float(run_args.reco_dropout)
    reco_cfg["reconstructor_model"]["max_split_children"] = int(run_args.reco_max_split_children)
    reco_cfg["reconstructor_model"]["max_generated_tokens"] = int(run_args.reco_max_generated_tokens)

    base_reco = confgen_ops.OfflineReconstructorConfidenceHybridOps(
        input_dim=int(input_dim),
        **reco_cfg["reconstructor_model"],
    ).to(device)
    reco = v2.ReconstructorWithAttrHeads(
        base=base_reco,
        input_dim=int(input_dim),
        hidden_dim=int(run_args.v2_attr_hidden_dim),
        attr_slots=int(run_args.v2_attr_slots),
    ).to(device)
    reco_sd, reco_src = _load_state_dict_from_run_dir(
        run_dir,
        ("offline_reconstructor_stage2.pt", "offline_reconstructor.pt"),
    )
    reco.load_state_dict(reco_sd, strict=True)
    reco.eval()

    dual = v2.JetClassDualViewTransformer(
        input_dim_a=int(input_dim),
        input_dim_b=10,
        n_classes=int(n_classes),
        embed_dim=int(run_args.embed_dim),
        num_heads=int(run_args.num_heads),
        num_layers=int(run_args.num_layers),
        ff_dim=int(run_args.ff_dim),
        dropout=float(run_args.dropout),
    ).to(device)
    dual_sd, dual_src = _load_state_dict_from_run_dir(
        run_dir,
        ("dual_joint_stage2.pt", "dual_joint.pt"),
    )
    dual.load_state_dict(dual_sd, strict=True)
    dual.eval()
    return reco, dual, {"reco_ckpt": str(reco_src), "dual_ckpt": str(dual_src)}


@torch.no_grad()
def _predict_probs(
    reco: torch.nn.Module,
    dual: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    corrected_weight_floor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    all_probs: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    for batch in loader:
        feat_hlt = batch["feat_hlt"].to(device)
        mask_hlt = batch["mask_hlt"].to(device)
        const_hlt4 = batch["const_hlt4"].to(device)
        y = batch["label"].to(device)
        reco_out = reco(feat_hlt, mask_hlt, const_hlt4, stage_scale=1.0)
        feat_b, mask_b = confgen_ops.build_soft_corrected_view_confgen_ops(
            reco_out,
            weight_floor=float(corrected_weight_floor),
            scale_features_by_weight=True,
            include_flags=False,
        )
        logits = dual(feat_hlt, mask_hlt, feat_b, mask_b)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_y.append(y.detach().cpu().numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_y, axis=0)


def _fuse(pa: np.ndarray, pb: np.ndarray, wa: float) -> np.ndarray:
    wa = float(np.clip(wa, 0.0, 1.0))
    return wa * pa + (1.0 - wa) * pb


def _metric_objective(m: Dict[str, float], optimize_for: str) -> Tuple[float, float]:
    # primary objective: smaller is better
    key = "signal_vs_bg_fpr50" if optimize_for == "sigbg_fpr50" else "target_vs_bg_ratio_fpr50"
    x = float(m.get(key, float("nan")))
    if np.isfinite(x):
        return x, 1.0
    # fallback to negative auc proxy when fpr50 is undefined
    auc_key = "signal_vs_bg_auc" if optimize_for == "sigbg_fpr50" else "target_vs_bg_ratio_auc"
    auc = float(m.get(auc_key, float("nan")))
    if np.isfinite(auc):
        return -auc, 0.0
    return float("inf"), -1.0


def _search_best_weight(
    pa: np.ndarray,
    pb: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str],
    background_class: str,
    target_class: str,
    weight_step: float,
    optimize_for: str,
) -> Tuple[float, Dict[str, float]]:
    ws = np.arange(0.0, 1.000001, float(weight_step))
    best_w = 0.5
    best_obj = float("inf")
    best_tie = -1e9
    best_m: Dict[str, float] = {}
    for w in ws:
        p = _fuse(pa, pb, float(w))
        m = eval_metrics(
            y_true=y,
            probs=p,
            class_names=class_names,
            background_class=background_class,
            target_class=target_class,
        )
        obj, tie = _metric_objective(m, optimize_for=optimize_for)
        if (obj < best_obj) or (np.isclose(obj, best_obj) and tie > best_tie):
            best_obj = obj
            best_tie = tie
            best_w = float(w)
            best_m = {k: float(v) if np.isscalar(v) else v for k, v in m.items()}
    return best_w, best_m


def _build_quantile_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    q = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(x, q)
    edges = np.asarray(edges, dtype=np.float64)
    # ensure strictly non-decreasing with tiny epsilon
    for i in range(1, len(edges)):
        if edges[i] < edges[i - 1]:
            edges[i] = edges[i - 1]
    return edges


def _bin_ids(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # right-inclusive final bin
    b = np.searchsorted(edges[1:-1], x, side="right")
    return np.clip(b, 0, len(edges) - 2)


def _to_float_dict(d: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in d.items():
        if isinstance(v, (float, int, np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Two-model JetClass bin-gated val-selected fusion")
    ap.add_argument("--run_a_dir", type=Path, required=True)
    ap.add_argument("--run_b_dir", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, default=Path("/home/ryreu/atlas/PracticeTagging/data/jetclass_part0"))
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--weight_step", type=float, default=0.01)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--min_bin_count", type=int, default=1200)
    ap.add_argument("--optimize_for", type=str, default="sigbg_fpr50", choices=["sigbg_fpr50", "targetbg_fpr50"])
    ap.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    args = ap.parse_args()

    run_a_dir = args.run_a_dir.resolve()
    run_b_dir = args.run_b_dir.resolve()
    args_a = _ns_from_json(run_a_dir / "args.json")
    args_b = _ns_from_json(run_b_dir / "args.json")

    # safety checks for comparable runs
    for k in (
        "feature_mode",
        "feature_preprocessing",
        "class_assignment",
        "target_class",
        "background_class",
        "n_val_jets",
        "n_test_jets",
        "max_constits",
        "hlt_pt_threshold",
        "merge_prob_scale",
        "reassign_scale",
        "smear_scale",
        "eff_plateau_barrel",
        "eff_plateau_endcap",
        "eff_turnon_pt",
        "eff_width_pt",
    ):
        if str(getattr(args_a, k)) != str(getattr(args_b, k)):
            raise ValueError(f"Run mismatch for `{k}`: A={getattr(args_a, k)} vs B={getattr(args_b, k)}")

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device).startswith("cpu") else "cpu")

    if args.out_dir is None:
        out_dir = (
            run_a_dir.parent
            / "fusion_reports"
            / f"{run_a_dir.name}__AND__{run_b_dir.name}__bin_gated_valsel"
        )
    else:
        out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    (
        class_names,
        va_feat_hlt,
        va_hlt_mask,
        va_hlt_const4,
        va_y,
        te_feat_hlt,
        te_hlt_mask,
        te_hlt_const4,
        te_y,
        standardization_mode,
    ) = _build_eval_data(args_a, args.data_dir)

    n_classes = int(len(class_names))
    input_dim = int(va_feat_hlt.shape[-1])
    reco_a, dual_a, src_a = _build_models_for_run(run_a_dir, args_a, input_dim, n_classes, device)
    reco_b, dual_b, src_b = _build_models_for_run(run_b_dir, args_b, input_dim, n_classes, device)

    dl_va = DataLoader(
        JointEvalDataset(va_feat_hlt, va_hlt_mask, va_hlt_const4, va_y),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    dl_te = DataLoader(
        JointEvalDataset(te_feat_hlt, te_hlt_mask, te_hlt_const4, te_y),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    pa_val, yv_a = _predict_probs(reco_a, dual_a, dl_va, device, corrected_weight_floor=float(args.corrected_weight_floor))
    pb_val, yv_b = _predict_probs(reco_b, dual_b, dl_va, device, corrected_weight_floor=float(args.corrected_weight_floor))
    if not np.array_equal(yv_a, yv_b):
        raise RuntimeError("Val labels mismatch between run A and B predictions.")
    pa_test, yt_a = _predict_probs(reco_a, dual_a, dl_te, device, corrected_weight_floor=float(args.corrected_weight_floor))
    pb_test, yt_b = _predict_probs(reco_b, dual_b, dl_te, device, corrected_weight_floor=float(args.corrected_weight_floor))
    if not np.array_equal(yt_a, yt_b):
        raise RuntimeError("Test labels mismatch between run A and B predictions.")

    bg_name = str(args_a.background_class)
    tgt_name = str(args_a.target_class)
    m_a_val = eval_metrics(yv_a, pa_val, class_names, bg_name, tgt_name)
    m_b_val = eval_metrics(yv_a, pb_val, class_names, bg_name, tgt_name)
    m_a_test = eval_metrics(yt_a, pa_test, class_names, bg_name, tgt_name)
    m_b_test = eval_metrics(yt_a, pb_test, class_names, bg_name, tgt_name)

    w_global, m_global_val = _search_best_weight(
        pa=pa_val,
        pb=pb_val,
        y=yv_a,
        class_names=class_names,
        background_class=bg_name,
        target_class=tgt_name,
        weight_step=float(args.weight_step),
        optimize_for=str(args.optimize_for),
    )
    p_global_test = _fuse(pa_test, pb_test, w_global)
    m_global_test = eval_metrics(yt_a, p_global_test, class_names, bg_name, tgt_name)

    bg_idx = class_names.index(bg_name)
    s_val = 1.0 - 0.5 * (pa_val[:, bg_idx] + pb_val[:, bg_idx])
    s_test = 1.0 - 0.5 * (pa_test[:, bg_idx] + pb_test[:, bg_idx])
    edges = _build_quantile_edges(s_val, int(args.n_bins))
    bid_val = _bin_ids(s_val, edges)
    bid_test = _bin_ids(s_test, edges)

    w_bins = np.full((int(args.n_bins),), float(w_global), dtype=np.float64)
    counts_bins = np.zeros((int(args.n_bins),), dtype=np.int64)
    for b in range(int(args.n_bins)):
        idx = np.where(bid_val == b)[0]
        counts_bins[b] = int(idx.size)
        if idx.size < int(args.min_bin_count):
            w_bins[b] = float(w_global)
            continue
        wb, _ = _search_best_weight(
            pa=pa_val[idx],
            pb=pb_val[idx],
            y=yv_a[idx],
            class_names=class_names,
            background_class=bg_name,
            target_class=tgt_name,
            weight_step=float(args.weight_step),
            optimize_for=str(args.optimize_for),
        )
        w_bins[b] = float(wb)

    p_bin_val = _fuse(pa_val, pb_val, 0.0)
    p_bin_test = _fuse(pa_test, pb_test, 0.0)
    for b in range(int(args.n_bins)):
        mval = bid_val == b
        mtest = bid_test == b
        wb = float(w_bins[b])
        if np.any(mval):
            p_bin_val[mval] = _fuse(pa_val[mval], pb_val[mval], wb)
        if np.any(mtest):
            p_bin_test[mtest] = _fuse(pa_test[mtest], pb_test[mtest], wb)
    m_bin_val = eval_metrics(yv_a, p_bin_val, class_names, bg_name, tgt_name)
    m_bin_test = eval_metrics(yt_a, p_bin_test, class_names, bg_name, tgt_name)

    report = {
        "run_a_dir": str(run_a_dir),
        "run_b_dir": str(run_b_dir),
        "run_a_sources": src_a,
        "run_b_sources": src_b,
        "data_dir": str(args.data_dir.resolve()),
        "out_dir": str(out_dir),
        "class_names": list(class_names),
        "target_class": str(tgt_name),
        "background_class": str(bg_name),
        "standardization_mode": standardization_mode,
        "val_size": int(len(yv_a)),
        "test_size": int(len(yt_a)),
        "optimize_for": str(args.optimize_for),
        "weight_step": float(args.weight_step),
        "n_bins": int(args.n_bins),
        "min_bin_count": int(args.min_bin_count),
        "global_weight_a_valsel": float(w_global),
        "bin_edges_signal_score": edges.tolist(),
        "bin_weights_a_valsel": w_bins.tolist(),
        "bin_counts_val": counts_bins.tolist(),
        "metrics": {
            "run_a_val": _to_float_dict(m_a_val),
            "run_b_val": _to_float_dict(m_b_val),
            "run_a_test": _to_float_dict(m_a_test),
            "run_b_test": _to_float_dict(m_b_test),
            "global_fusion_val": _to_float_dict(m_global_val),
            "global_fusion_test": _to_float_dict(m_global_test),
            "bin_gated_fusion_val": _to_float_dict(m_bin_val),
            "bin_gated_fusion_test": _to_float_dict(m_bin_test),
        },
    }

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        out_dir / "fusion_scores.npz",
        y_val=yv_a.astype(np.int64),
        y_test=yt_a.astype(np.int64),
        probs_a_val=pa_val.astype(np.float32),
        probs_b_val=pb_val.astype(np.float32),
        probs_a_test=pa_test.astype(np.float32),
        probs_b_test=pb_test.astype(np.float32),
        probs_global_val=_fuse(pa_val, pb_val, float(w_global)).astype(np.float32),
        probs_global_test=p_global_test.astype(np.float32),
        probs_bin_val=p_bin_val.astype(np.float32),
        probs_bin_test=p_bin_test.astype(np.float32),
        bin_edges=edges.astype(np.float64),
        bin_weights=w_bins.astype(np.float64),
    )

    print("============================================================")
    print("Two-Model JetClass Bin-Gated Fusion (Val-Selected)")
    print("============================================================")
    print(f"Run A: {run_a_dir}")
    print(f"Run B: {run_b_dir}")
    print(f"Out:   {out_dir}")
    print(
        f"Global fusion weight (A): {w_global:.3f} | "
        f"Val fpr50(sig-vs-bg)={float(m_global_val.get('signal_vs_bg_fpr50', float('nan'))):.6f} | "
        f"Test fpr50(sig-vs-bg)={float(m_global_test.get('signal_vs_bg_fpr50', float('nan'))):.6f}"
    )
    print(
        f"Bin-gated fusion            "
        f"Val fpr50(sig-vs-bg)={float(m_bin_val.get('signal_vs_bg_fpr50', float('nan'))):.6f} | "
        f"Test fpr50(sig-vs-bg)={float(m_bin_test.get('signal_vs_bg_fpr50', float('nan'))):.6f}"
    )
    print(f"Saved report: {out_dir / 'report.json'}")
    print(f"Saved scores: {out_dir / 'fusion_scores.npz'}")


if __name__ == "__main__":
    main()
