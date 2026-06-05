#!/usr/bin/env python3
"""
JetClass Level-1 jet calibrator.

This is a deliberately simple physical-recovery test:
  HLT constituents -> jet-level residuals to offline jet

The model predicts:
  - log(pT_offline / pT_HLT)
  - eta_offline - eta_HLT
  - wrapped(phi_offline - phi_HLT)

It then compares raw HLT vs calibrated-HLT for pT response/resolution and
jet-axis recovery. No classifier, teacher scores, or offline labels are used in
the prediction target beyond the offline jet four-vector.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'user')}")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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


IDX_PT = 0
IDX_ETA = 1
IDX_PHI = 2
IDX_E = 3
IDX_CHARGE = 4
IDX_PID0 = 5
IDX_PID4 = 9


def _wrap_delta_phi(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, 1e-8, np.inf))


def _ns_from_json(path: Path) -> SimpleNamespace:
    return SimpleNamespace(**json.loads(path.read_text()))


def _get_ref(args: argparse.Namespace) -> SimpleNamespace:
    if args.reference_run_dir is None:
        return SimpleNamespace(**vars(args))
    p = Path(args.reference_run_dir).resolve() / "args.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing reference args.json: {p}")
    ref = _ns_from_json(p)
    for key, val in vars(args).items():
        if key in {"reference_run_dir"}:
            continue
        if val is not None:
            setattr(ref, key, val)
    return ref


def _resolve_classes(args_ref: SimpleNamespace, files_by_class: Dict[str, Sequence[Path]]) -> List[str]:
    if str(getattr(args_ref, "class_assignment", "filename")) == "canonical_labels":
        class_names = list(CANONICAL_CLASS_ORDER)
    else:
        class_names = sorted(files_by_class.keys())

    include = str(getattr(args_ref, "include_classes", "") or "").strip()
    if include:
        wanted = [x.strip() for x in include.split(",") if x.strip()]
        missing = [x for x in wanted if x not in class_names]
        if missing:
            raise ValueError(f"Unknown include_classes entries: {missing}; available={class_names}")
        class_names = wanted
    return list(class_names)


def _build_hlt_view(tok: np.ndarray, mask: np.ndarray, params: HLTParams, seed: int, builder: str):
    if builder == "m2":
        from train_jetclass_joint_dualview_stage2_unmergeonly_m2hlt import _build_hlt_view_m2style

        return _build_hlt_view_m2style(tok, mask, params=params, seed=seed)

    from evaluate_jetclass_hlt_teacher_baseline import build_hlt_view

    return build_hlt_view(tok, mask, params=params, seed=seed)


def _load_one_split(
    split_files: Dict[str, List[Path]],
    n_total: int,
    max_constits: int,
    class_to_idx: Dict[str, int],
    seed: int,
    class_assignment: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tok, mask, y = load_split(
        split_files,
        n_total=int(n_total),
        max_constits=int(max_constits),
        class_to_idx=class_to_idx,
        seed=int(seed),
        class_assignment=str(class_assignment),
    )
    return tok.astype(np.float32), mask.astype(bool), y.astype(np.int64)


def _load_data(args_ref: SimpleNamespace):
    data_dir = Path(args_ref.data_dir).resolve()
    files_by_class = collect_files_by_class(data_dir)
    class_names = _resolve_classes(args_ref, files_by_class)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    if str(getattr(args_ref, "class_assignment", "filename")) == "filename":
        source_files = {c: files_by_class[c] for c in class_names}
    else:
        source_files = files_by_class

    tr_files, va_files, te_files = split_files_by_class(
        source_files,
        n_train=int(args_ref.train_files_per_class),
        n_val=int(args_ref.val_files_per_class),
        n_test=int(args_ref.test_files_per_class),
        shuffle=bool(args_ref.shuffle_files),
        seed=int(args_ref.seed),
    )

    print("Loading train split...")
    tr_tok, tr_mask, tr_y = _load_one_split(
        tr_files,
        int(args_ref.n_train_jets),
        int(args_ref.max_constits),
        class_to_idx,
        int(args_ref.seed) + 101,
        str(args_ref.class_assignment),
    )
    print("Loading val split...")
    va_tok, va_mask, va_y = _load_one_split(
        va_files,
        int(args_ref.n_val_jets),
        int(args_ref.max_constits),
        class_to_idx,
        int(args_ref.seed) + 202,
        str(args_ref.class_assignment),
    )
    print("Loading test split...")
    te_tok, te_mask, te_y = _load_one_split(
        te_files,
        int(args_ref.n_test_jets),
        int(args_ref.max_constits),
        class_to_idx,
        int(args_ref.seed) + 303,
        str(args_ref.class_assignment),
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
    builder = str(getattr(args_ref, "hlt_builder", "m2"))
    print(f"Building HLT-like corrupted splits with hlt_builder={builder}...")
    tr_hlt, tr_hlt_mask, tr_diag = _build_hlt_view(tr_tok, tr_mask, hlt_params, int(args_ref.seed) + 1001, builder)
    va_hlt, va_hlt_mask, va_diag = _build_hlt_view(va_tok, va_mask, hlt_params, int(args_ref.seed) + 1002, builder)
    te_hlt, te_hlt_mask, te_diag = _build_hlt_view(te_tok, te_mask, hlt_params, int(args_ref.seed) + 1003, builder)

    tr_feat = compute_features(
        tr_hlt,
        tr_hlt_mask,
        feature_mode=str(args_ref.feature_mode),
        feature_preprocessing=str(args_ref.feature_preprocessing),
    )
    va_feat = compute_features(
        va_hlt,
        va_hlt_mask,
        feature_mode=str(args_ref.feature_mode),
        feature_preprocessing=str(args_ref.feature_preprocessing),
    )
    te_feat = compute_features(
        te_hlt,
        te_hlt_mask,
        feature_mode=str(args_ref.feature_mode),
        feature_preprocessing=str(args_ref.feature_preprocessing),
    )
    if str(args_ref.feature_preprocessing) != "canonical":
        off_feat = compute_features(
            tr_tok,
            tr_mask,
            feature_mode=str(args_ref.feature_mode),
            feature_preprocessing=str(args_ref.feature_preprocessing),
        )
        mean, std = get_mean_std(off_feat, tr_mask, np.arange(len(tr_y)))
        tr_feat = standardize(tr_feat, tr_hlt_mask, mean, std)
        va_feat = standardize(va_feat, va_hlt_mask, mean, std)
        te_feat = standardize(te_feat, te_hlt_mask, mean, std)

    return {
        "class_names": class_names,
        "train": (tr_tok[:, :, :4], tr_mask, tr_hlt[:, :, :4], tr_hlt_mask, tr_feat, tr_y),
        "val": (va_tok[:, :, :4], va_mask, va_hlt[:, :, :4], va_hlt_mask, va_feat, va_y),
        "test": (te_tok[:, :, :4], te_mask, te_hlt[:, :, :4], te_hlt_mask, te_feat, te_y),
        "hlt_diagnostics": {"train": tr_diag, "val": va_diag, "test": te_diag},
    }


def _jet_p4(tokens: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
    pt = np.maximum(tokens[:, :, IDX_PT].astype(np.float64), 0.0)
    eta = tokens[:, :, IDX_ETA].astype(np.float64)
    phi = tokens[:, :, IDX_PHI].astype(np.float64)
    ene = np.maximum(tokens[:, :, IDX_E].astype(np.float64), 0.0)
    w = mask.astype(np.float64)
    px = (pt * np.cos(phi) * w).sum(axis=1)
    py = (pt * np.sin(phi) * w).sum(axis=1)
    pz = (pt * np.sinh(eta) * w).sum(axis=1)
    e = (ene * w).sum(axis=1)
    jet_pt = np.sqrt(px * px + py * py)
    jet_eta = np.arcsinh(pz / np.clip(jet_pt, 1e-8, np.inf))
    jet_phi = np.arctan2(py, px)
    p2 = px * px + py * py + pz * pz
    mass = np.sqrt(np.clip(e * e - p2, 0.0, np.inf))
    scalar_pt = (pt * w).sum(axis=1)
    return {
        "pt": jet_pt.astype(np.float32),
        "eta": jet_eta.astype(np.float32),
        "phi": jet_phi.astype(np.float32),
        "px": px.astype(np.float32),
        "py": py.astype(np.float32),
        "pz": pz.astype(np.float32),
        "e": e.astype(np.float32),
        "mass": mass.astype(np.float32),
        "scalar_pt": scalar_pt.astype(np.float32),
    }


def _global_features(tokens: np.ndarray, mask: np.ndarray, p4: Dict[str, np.ndarray]) -> np.ndarray:
    pt = np.maximum(tokens[:, :, IDX_PT].astype(np.float64), 0.0)
    eta = tokens[:, :, IDX_ETA].astype(np.float64)
    phi = tokens[:, :, IDX_PHI].astype(np.float64)
    charge = tokens[:, :, IDX_CHARGE].astype(np.float64) if tokens.shape[-1] > IDX_CHARGE else np.zeros_like(pt)
    pid = tokens[:, :, IDX_PID0 : IDX_PID4 + 1].astype(np.float64) if tokens.shape[-1] > IDX_PID4 else None
    w = mask.astype(np.float64)

    n = np.maximum(w.sum(axis=1), 1.0)
    jet_eta = p4["eta"].astype(np.float64)[:, None]
    jet_phi = p4["phi"].astype(np.float64)[:, None]
    deta = eta - jet_eta
    dphi = _wrap_delta_phi(phi - jet_phi)
    pt_sum = np.maximum((pt * w).sum(axis=1), 1e-8)
    pt_frac = pt / pt_sum[:, None]

    leading_pt = np.max(pt * w, axis=1)
    top3 = np.sort(pt * w, axis=1)[:, -3:].sum(axis=1)
    weighted_deta2 = (pt_frac * w * deta * deta).sum(axis=1)
    weighted_dphi2 = (pt_frac * w * dphi * dphi).sum(axis=1)
    charged_frac = ((np.abs(charge) > 0) * pt * w).sum(axis=1) / pt_sum

    cols = [
        _safe_log(p4["pt"]),
        p4["eta"].astype(np.float64),
        np.abs(p4["eta"].astype(np.float64)),
        _safe_log(p4["e"]),
        _safe_log(p4["mass"] + 1.0),
        np.log(n),
        p4["scalar_pt"].astype(np.float64) / np.clip(p4["pt"].astype(np.float64), 1e-8, np.inf),
        leading_pt / pt_sum,
        top3 / pt_sum,
        np.sqrt(np.maximum(weighted_deta2, 0.0)),
        np.sqrt(np.maximum(weighted_dphi2, 0.0)),
        charged_frac,
    ]
    if pid is not None:
        for k in range(pid.shape[-1]):
            cols.append((pid[:, :, k] * pt * w).sum(axis=1) / pt_sum)
    return np.stack(cols, axis=1).astype(np.float32)


def _targets(off_p4: Dict[str, np.ndarray], hlt_p4: Dict[str, np.ndarray]) -> np.ndarray:
    dlogpt = _safe_log(off_p4["pt"].astype(np.float64)) - _safe_log(hlt_p4["pt"].astype(np.float64))
    deta = off_p4["eta"].astype(np.float64) - hlt_p4["eta"].astype(np.float64)
    dphi = _wrap_delta_phi(off_p4["phi"].astype(np.float64) - hlt_p4["phi"].astype(np.float64))
    return np.stack([dlogpt, deta, dphi], axis=1).astype(np.float32)


def _apply_residual(hlt_p4: Dict[str, np.ndarray], pred: np.ndarray) -> Dict[str, np.ndarray]:
    dlogpt = np.clip(pred[:, 0].astype(np.float64), -2.0, 2.0)
    deta = np.clip(pred[:, 1].astype(np.float64), -1.5, 1.5)
    dphi = np.clip(pred[:, 2].astype(np.float64), -1.5, 1.5)
    pt = hlt_p4["pt"].astype(np.float64) * np.exp(dlogpt)
    eta = hlt_p4["eta"].astype(np.float64) + deta
    phi = hlt_p4["phi"].astype(np.float64) + dphi
    phi = np.arctan2(np.sin(phi), np.cos(phi))
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return {
        "pt": pt.astype(np.float32),
        "eta": eta.astype(np.float32),
        "phi": phi.astype(np.float32),
        "px": px.astype(np.float32),
        "py": py.astype(np.float32),
        "pz": pz.astype(np.float32),
    }


class CalibDataset(Dataset):
    def __init__(
        self,
        token_feat: np.ndarray,
        mask: np.ndarray,
        global_feat: np.ndarray,
        target_std: np.ndarray,
    ):
        self.token_feat = torch.tensor(token_feat, dtype=torch.float32)
        self.mask = torch.tensor(mask.astype(bool), dtype=torch.bool)
        self.global_feat = torch.tensor(global_feat, dtype=torch.float32)
        self.target = torch.tensor(target_std, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, idx: int):
        return self.token_feat[idx], self.mask[idx], self.global_feat[idx], self.target[idx]


class DeepSetCalibrator(nn.Module):
    def __init__(self, token_dim: int, global_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.token_net = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.global_net = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, token_feat: torch.Tensor, mask: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        h = self.token_net(token_feat)
        mask_f = mask.unsqueeze(-1).to(h.dtype)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        mean_pool = (h * mask_f).sum(dim=1) / denom
        h_max = h.masked_fill(~mask.unsqueeze(-1), -1e4)
        max_pool = h_max.max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        g = self.global_net(global_feat)
        return self.head(torch.cat([mean_pool, max_pool, g], dim=1))


@torch.no_grad()
def _predict(model: nn.Module, dataset: CalibDataset, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    out: List[np.ndarray] = []
    model.eval()
    for token_feat, mask, global_feat, _ in loader:
        token_feat = token_feat.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        global_feat = global_feat.to(device, non_blocking=True)
        pred = model(token_feat, mask, global_feat)
        out.append(pred.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def _axis_errors(pred_p4: Dict[str, np.ndarray], truth_p4: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    deta = pred_p4["eta"].astype(np.float64) - truth_p4["eta"].astype(np.float64)
    dphi = _wrap_delta_phi(pred_p4["phi"].astype(np.float64) - truth_p4["phi"].astype(np.float64))
    dr = np.sqrt(deta * deta + dphi * dphi)
    dot = (
        pred_p4["px"].astype(np.float64) * truth_p4["px"].astype(np.float64)
        + pred_p4["py"].astype(np.float64) * truth_p4["py"].astype(np.float64)
        + pred_p4["pz"].astype(np.float64) * truth_p4["pz"].astype(np.float64)
    )
    pred_norm = np.sqrt(pred_p4["px"].astype(np.float64) ** 2 + pred_p4["py"].astype(np.float64) ** 2 + pred_p4["pz"].astype(np.float64) ** 2)
    truth_norm = np.sqrt(truth_p4["px"].astype(np.float64) ** 2 + truth_p4["py"].astype(np.float64) ** 2 + truth_p4["pz"].astype(np.float64) ** 2)
    angle = np.arccos(np.clip(dot / np.clip(pred_norm * truth_norm, 1e-12, np.inf), -1.0, 1.0))
    return {"delta_eta": deta, "delta_phi": dphi, "delta_R": dr, "angle3d": angle}


def _axis_metrics(errors: Dict[str, np.ndarray]) -> Dict[str, float]:
    dr = errors["delta_R"][np.isfinite(errors["delta_R"])]
    deta = errors["delta_eta"][np.isfinite(errors["delta_eta"])]
    dphi = errors["delta_phi"][np.isfinite(errors["delta_phi"])]
    angle = errors["angle3d"][np.isfinite(errors["angle3d"])]
    return {
        "mean_deltaR": float(np.mean(dr)),
        "std_deltaR": float(np.std(dr)),
        "median_deltaR": float(np.median(dr)),
        "q68_deltaR": float(np.quantile(dr, 0.68)),
        "q90_deltaR": float(np.quantile(dr, 0.90)),
        "mean_abs_delta_eta": float(np.mean(np.abs(deta))),
        "mean_abs_delta_phi": float(np.mean(np.abs(dphi))),
        "mean_angle3d": float(np.mean(angle)),
        "std_angle3d": float(np.std(angle)),
    }


def _pt_edges(pt: np.ndarray, n_bins: int) -> np.ndarray:
    valid = np.isfinite(pt) & (pt > 1e-8)
    return np.unique(np.quantile(pt[valid], np.linspace(0.0, 1.0, int(n_bins) + 1)))


def _response_records(pt_truth: np.ndarray, pt_pred: np.ndarray, edges: np.ndarray, min_count: int):
    records = []
    valid = np.isfinite(pt_truth) & np.isfinite(pt_pred) & (pt_truth > 1e-8)
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        sel = valid & (pt_truth >= lo)
        sel = sel & (pt_truth < hi if i < len(edges) - 2 else pt_truth <= hi)
        if int(sel.sum()) < int(min_count):
            continue
        ratio = pt_pred[sel] / np.clip(pt_truth[sel], 1e-8, np.inf)
        records.append(
            {
                "pt_low": lo,
                "pt_high": hi,
                "pt_center": 0.5 * (lo + hi),
                "count": int(ratio.size),
                "response": float(np.mean(ratio)),
                "resolution": float(np.std(ratio)),
            }
        )
    return records


def _response_score(records: Sequence[Dict[str, float]]) -> float:
    if not records:
        return float("inf")
    counts = np.asarray([r["count"] for r in records], dtype=np.float64)
    terms = np.asarray([abs(r["response"] - 1.0) + r["resolution"] for r in records], dtype=np.float64)
    return float(np.average(terms, weights=counts))


def _metrics_bundle(truth_p4: Dict[str, np.ndarray], hlt_p4: Dict[str, np.ndarray], calib_p4: Dict[str, np.ndarray], n_bins: int, min_count: int):
    edges = _pt_edges(truth_p4["pt"], n_bins)
    hlt_records = _response_records(truth_p4["pt"], hlt_p4["pt"], edges, min_count)
    calib_records = _response_records(truth_p4["pt"], calib_p4["pt"], edges, min_count)
    hlt_axis = _axis_errors(hlt_p4, truth_p4)
    calib_axis = _axis_errors(calib_p4, truth_p4)
    hlt_axis_metrics = _axis_metrics(hlt_axis)
    calib_axis_metrics = _axis_metrics(calib_axis)
    return {
        "edges": edges,
        "hlt_response_records": hlt_records,
        "calib_response_records": calib_records,
        "hlt_response_score": _response_score(hlt_records),
        "calib_response_score": _response_score(calib_records),
        "hlt_axis_errors": hlt_axis,
        "calib_axis_errors": calib_axis,
        "hlt_axis_metrics": hlt_axis_metrics,
        "calib_axis_metrics": calib_axis_metrics,
        "hlt_axis_score": hlt_axis_metrics["mean_deltaR"] + hlt_axis_metrics["std_deltaR"],
        "calib_axis_score": calib_axis_metrics["mean_deltaR"] + calib_axis_metrics["std_deltaR"],
        "fraction_axis_improved": float(np.mean(calib_axis["delta_R"] < hlt_axis["delta_R"])),
    }


def _combined_score(bundle: Dict[str, object], axis_weight: float) -> float:
    return float(bundle["calib_response_score"]) + float(axis_weight) * float(bundle["calib_axis_score"])


def _plot_response(hlt_records, calib_records, out_path: Path):
    def arr(records, key):
        return np.asarray([r[key] for r in records], dtype=np.float64)

    plt.figure(figsize=(10, 4.2))
    plt.subplot(1, 2, 1)
    plt.plot(arr(hlt_records, "pt_center"), arr(hlt_records, "response"), "o-", label="HLT", color="steelblue")
    plt.plot(arr(calib_records, "pt_center"), arr(calib_records, "response"), "s--", label="Level-1 calibrated", color="forestgreen")
    plt.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    plt.xlabel("Offline jet pT")
    plt.ylabel("Response: pT / pT_offline")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.subplot(1, 2, 2)
    plt.plot(arr(hlt_records, "pt_center"), arr(hlt_records, "resolution"), "o-", label="HLT", color="steelblue")
    plt.plot(arr(calib_records, "pt_center"), arr(calib_records, "resolution"), "s--", label="Level-1 calibrated", color="forestgreen")
    plt.xlabel("Offline jet pT")
    plt.ylabel("Resolution: std(response)")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_axis(hlt_dr: np.ndarray, calib_dr: np.ndarray, out_path: Path):
    h = hlt_dr[np.isfinite(hlt_dr)]
    c = calib_dr[np.isfinite(calib_dr)]
    xmax = float(np.quantile(np.concatenate([h, c]), 0.995))
    bins = np.linspace(0.0, max(xmax, 1e-3), 90)
    plt.figure(figsize=(10, 4.2))
    plt.subplot(1, 2, 1)
    plt.hist(h, bins=bins, density=True, histtype="step", linewidth=1.8, color="steelblue", label="HLT")
    plt.hist(c, bins=bins, density=True, histtype="step", linewidth=1.8, color="forestgreen", label="Level-1 calibrated")
    plt.xlabel("DeltaR to offline jet axis")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.subplot(1, 2, 2)
    for vals, label, color in [(np.sort(h), "HLT", "steelblue"), (np.sort(c), "Level-1 calibrated", "forestgreen")]:
        y = np.arange(1, len(vals) + 1) / max(len(vals), 1)
        plt.plot(vals, y, color=color, label=label)
    plt.xlabel("DeltaR to offline jet axis")
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_axis_scatter(hlt_dr: np.ndarray, calib_dr: np.ndarray, out_path: Path, max_points: int, seed: int):
    valid = np.isfinite(hlt_dr) & np.isfinite(calib_dr)
    idx = np.flatnonzero(valid)
    if idx.size > int(max_points):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(idx, size=int(max_points), replace=False)
    x = hlt_dr[idx]
    y = calib_dr[idx]
    lim = float(np.quantile(np.concatenate([x, y]), 0.995)) if x.size else 1.0
    lim = max(lim, 1e-3)
    plt.figure(figsize=(5.2, 5.0))
    plt.scatter(x, y, s=3, alpha=0.15, color="black", rasterized=True)
    plt.plot([0.0, lim], [0.0, lim], color="crimson", linestyle="--", linewidth=1.2)
    plt.xlim(0.0, lim)
    plt.ylim(0.0, lim)
    plt.xlabel("HLT DeltaR to offline")
    plt.ylabel("Level-1 calibrated DeltaR to offline")
    plt.title("Points below diagonal improved")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _jsonable_records(records):
    return [{k: (int(v) if k == "count" else float(v)) for k, v in r.items()} for r in records]


def _prepare_split(split):
    off_tok, off_mask, hlt_tok, hlt_mask, hlt_feat, y = split
    off_p4 = _jet_p4(off_tok, off_mask)
    hlt_p4 = _jet_p4(hlt_tok, hlt_mask)
    glob = _global_features(hlt_tok, hlt_mask, hlt_p4)
    target = _targets(off_p4, hlt_p4)
    return off_p4, hlt_p4, hlt_feat.astype(np.float32), hlt_mask.astype(bool), glob.astype(np.float32), target.astype(np.float32), y


def _standardize_arrays(train: np.ndarray, *others: np.ndarray):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    outs = [(x - mean) / std for x in (train, *others)]
    return outs, mean.squeeze(0), std.squeeze(0)


def _train(args, model, train_ds, val_ds, val_context, device):
    loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best = {"score": float("inf"), "state": None, "epoch": -1}
    no_improve = 0

    target_w = torch.tensor([float(args.loss_w_logpt), float(args.loss_w_eta), float(args.loss_w_phi)], device=device)
    for ep in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        n_seen = 0
        for token_feat, mask, global_feat, target in loader:
            token_feat = token_feat.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            global_feat = global_feat.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            pred = model(token_feat, mask, global_feat)
            loss_raw = F.smooth_l1_loss(pred, target, reduction="none")
            loss = (loss_raw * target_w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            bs = int(target.shape[0])
            total += float(loss.item()) * bs
            n_seen += bs

        pred_val_std = _predict(model, val_ds, device, int(args.eval_batch_size))
        pred_val = pred_val_std * val_context["target_std"] + val_context["target_mean"]
        calib_val = _apply_residual(val_context["hlt_p4"], pred_val)
        bundle = _metrics_bundle(
            val_context["off_p4"],
            val_context["hlt_p4"],
            calib_val,
            int(args.response_n_bins),
            int(args.response_min_count),
        )
        score = _combined_score(bundle, float(args.axis_score_weight))
        print(
            f"Epoch {ep:03d}: train_loss={total / max(n_seen, 1):.6f} "
            f"val_resp HLT/cal={bundle['hlt_response_score']:.6f}/{bundle['calib_response_score']:.6f} "
            f"val_axis HLT/cal={bundle['hlt_axis_score']:.6f}/{bundle['calib_axis_score']:.6f} "
            f"combined={score:.6f}"
        )
        if score < float(best["score"]):
            best["score"] = float(score)
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["epoch"] = ep
            no_improve = 0
        else:
            no_improve += 1
        if ep >= int(args.min_epochs) and no_improve >= int(args.patience):
            print(f"Early stopping at epoch {ep}")
            break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return best


def _save_outputs(args, out_dir: Path, model, best, test_bundle, arrays, metadata):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "best": {k: v for k, v in best.items() if k != "state"},
            "args": vars(args),
            "metadata": metadata,
        },
        out_dir / "level1_jet_calibrator_best.pt",
    )
    _plot_response(
        test_bundle["hlt_response_records"],
        test_bundle["calib_response_records"],
        out_dir / "level1_pt_response_resolution.png",
    )
    _plot_axis(
        test_bundle["hlt_axis_errors"]["delta_R"],
        test_bundle["calib_axis_errors"]["delta_R"],
        out_dir / "level1_axis_deltaR_hist_cdf.png",
    )
    _plot_axis_scatter(
        test_bundle["hlt_axis_errors"]["delta_R"],
        test_bundle["calib_axis_errors"]["delta_R"],
        out_dir / "level1_axis_deltaR_scatter.png",
        int(args.scatter_max_points),
        int(args.seed),
    )
    summary = {
        "target": "offline_jetclass_jet_four_vector",
        "method": "Level-1 DeepSets jet residual calibrator",
        "prediction": ["log_pt_ratio", "delta_eta", "delta_phi"],
        "best_epoch": int(best["epoch"]),
        "selection": "validation response score + axis_score_weight * validation axis score",
        "test": {
            "hlt": {
                "response_score": float(test_bundle["hlt_response_score"]),
                "axis_score": float(test_bundle["hlt_axis_score"]),
                "response_records": _jsonable_records(test_bundle["hlt_response_records"]),
                "axis_metrics": test_bundle["hlt_axis_metrics"],
            },
            "level1_calibrated": {
                "response_score": float(test_bundle["calib_response_score"]),
                "axis_score": float(test_bundle["calib_axis_score"]),
                "response_records": _jsonable_records(test_bundle["calib_response_records"]),
                "axis_metrics": test_bundle["calib_axis_metrics"],
                "fraction_axis_improved_vs_hlt": float(test_bundle["fraction_axis_improved"]),
            },
            "improvement": {
                "response_score_hlt_minus_calibrated": float(test_bundle["hlt_response_score"] - test_bundle["calib_response_score"]),
                "axis_score_hlt_minus_calibrated": float(test_bundle["hlt_axis_score"] - test_bundle["calib_axis_score"]),
            },
        },
        "metadata": metadata,
        "outputs": {
            "checkpoint": str(out_dir / "level1_jet_calibrator_best.pt"),
            "pt_response_plot": str(out_dir / "level1_pt_response_resolution.png"),
            "axis_deltaR_plot": str(out_dir / "level1_axis_deltaR_hist_cdf.png"),
            "axis_scatter_plot": str(out_dir / "level1_axis_deltaR_scatter.png"),
            "arrays": str(out_dir / "level1_calibrator_test_arrays.npz"),
        },
    }
    (out_dir / "level1_jet_calibrator_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    np.savez_compressed(out_dir / "level1_calibrator_test_arrays.npz", **arrays)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--reference_run_dir", type=Path, default=None, help="Optional run dir whose args.json supplies split/HLT defaults.")
    p.add_argument("--data_dir", type=Path, default=Path("/home/ryreu/atlas/PracticeTagging/data/jetclass_part0"))
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=52)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--feature_mode", type=str, default="full", choices=["kin", "kinpid", "full"])
    p.add_argument("--feature_preprocessing", type=str, default="canonical", choices=["canonical", "legacy"])
    p.add_argument("--class_assignment", type=str, default="filename", choices=["filename", "canonical_labels"])
    p.add_argument("--include_classes", type=str, default="")
    p.add_argument("--max_constits", type=int, default=128)
    p.add_argument("--train_files_per_class", type=int, default=8)
    p.add_argument("--val_files_per_class", type=int, default=1)
    p.add_argument("--test_files_per_class", type=int, default=1)
    p.add_argument("--shuffle_files", action="store_true", default=False)
    p.add_argument("--n_train_jets", type=int, default=250000)
    p.add_argument("--n_val_jets", type=int, default=50000)
    p.add_argument("--n_test_jets", type=int, default=250000)

    p.add_argument("--hlt_builder", type=str, default="m2", choices=["m2", "default"])
    p.add_argument("--hlt_pt_threshold", type=float, default=1.30)
    p.add_argument("--merge_prob_scale", type=float, default=1.35)
    p.add_argument("--reassign_scale", type=float, default=1.00)
    p.add_argument("--smear_scale", type=float, default=1.00)
    p.add_argument("--eff_plateau_barrel", type=float, default=0.99)
    p.add_argument("--eff_plateau_endcap", type=float, default=0.97)
    p.add_argument("--eff_turnon_pt", type=float, default=1.40)
    p.add_argument("--eff_width_pt", type=float, default=0.20)

    p.add_argument("--hidden_dim", type=int, default=192)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--eval_batch_size", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--min_epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--loss_w_logpt", type=float, default=1.0)
    p.add_argument("--loss_w_eta", type=float, default=1.0)
    p.add_argument("--loss_w_phi", type=float, default=1.0)
    p.add_argument("--axis_score_weight", type=float, default=1.0)
    p.add_argument("--response_n_bins", type=int, default=10)
    p.add_argument("--response_min_count", type=int, default=500)
    p.add_argument("--scatter_max_points", type=int, default=50000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args_ref = _get_ref(args)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device).startswith("cpu") else "cpu")
    print(f"Using device: {device}")
    print(f"Output: {out_dir}")

    data = _load_data(args_ref)
    tr = _prepare_split(data["train"])
    va = _prepare_split(data["val"])
    te = _prepare_split(data["test"])

    tr_off, tr_hlt, tr_tokfeat, tr_mask, tr_glob, tr_tgt, _ = tr
    va_off, va_hlt, va_tokfeat, va_mask, va_glob, va_tgt, _ = va
    te_off, te_hlt, te_tokfeat, te_mask, te_glob, te_tgt, te_y = te

    (glob_std,), glob_mean, glob_scale = _standardize_arrays(tr_glob)
    va_glob_std = (va_glob - glob_mean[None, :]) / glob_scale[None, :]
    te_glob_std = (te_glob - glob_mean[None, :]) / glob_scale[None, :]
    (tgt_tr_std,), tgt_mean, tgt_scale = _standardize_arrays(tr_tgt)
    tgt_va_std = (va_tgt - tgt_mean[None, :]) / tgt_scale[None, :]
    tgt_te_std = (te_tgt - tgt_mean[None, :]) / tgt_scale[None, :]

    train_ds = CalibDataset(tr_tokfeat, tr_mask, glob_std, tgt_tr_std)
    val_ds = CalibDataset(va_tokfeat, va_mask, va_glob_std, tgt_va_std)
    test_ds = CalibDataset(te_tokfeat, te_mask, te_glob_std, tgt_te_std)

    model = DeepSetCalibrator(
        token_dim=int(tr_tokfeat.shape[-1]),
        global_dim=int(glob_std.shape[-1]),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    ).to(device)
    print(
        f"Model: token_dim={tr_tokfeat.shape[-1]} global_dim={glob_std.shape[-1]} "
        f"hidden={args.hidden_dim} params={sum(p.numel() for p in model.parameters())}"
    )

    val_context = {
        "off_p4": va_off,
        "hlt_p4": va_hlt,
        "target_mean": tgt_mean[None, :],
        "target_std": tgt_scale[None, :],
    }
    best = _train(args, model, train_ds, val_ds, val_context, device)

    pred_test_std = _predict(model, test_ds, device, int(args.eval_batch_size))
    pred_test = pred_test_std * tgt_scale[None, :] + tgt_mean[None, :]
    calib_test = _apply_residual(te_hlt, pred_test)
    test_bundle = _metrics_bundle(te_off, te_hlt, calib_test, int(args.response_n_bins), int(args.response_min_count))

    metadata = {
        "classes": list(data["class_names"]),
        "class_counts_test": {data["class_names"][i]: int((te_y == i).sum()) for i in range(len(data["class_names"]))},
        "n_train_jets": int(len(tr_tgt)),
        "n_val_jets": int(len(va_tgt)),
        "n_test_jets": int(len(te_tgt)),
        "target_mean": tgt_mean.astype(float).tolist(),
        "target_std": tgt_scale.astype(float).tolist(),
        "global_mean": glob_mean.astype(float).tolist(),
        "global_std": glob_scale.astype(float).tolist(),
        "hlt_builder": str(getattr(args_ref, "hlt_builder", "m2")),
        "hlt_params": {
            "hlt_pt_threshold": float(args_ref.hlt_pt_threshold),
            "merge_prob_scale": float(args_ref.merge_prob_scale),
            "reassign_scale": float(args_ref.reassign_scale),
            "smear_scale": float(args_ref.smear_scale),
            "eff_plateau_barrel": float(args_ref.eff_plateau_barrel),
            "eff_plateau_endcap": float(args_ref.eff_plateau_endcap),
            "eff_turnon_pt": float(args_ref.eff_turnon_pt),
            "eff_width_pt": float(args_ref.eff_width_pt),
        },
    }
    arrays = {
        "pt_offline": te_off["pt"].astype(np.float32),
        "pt_hlt": te_hlt["pt"].astype(np.float32),
        "pt_level1": calib_test["pt"].astype(np.float32),
        "eta_offline": te_off["eta"].astype(np.float32),
        "eta_hlt": te_hlt["eta"].astype(np.float32),
        "eta_level1": calib_test["eta"].astype(np.float32),
        "phi_offline": te_off["phi"].astype(np.float32),
        "phi_hlt": te_hlt["phi"].astype(np.float32),
        "phi_level1": calib_test["phi"].astype(np.float32),
        "deltaR_hlt": test_bundle["hlt_axis_errors"]["delta_R"].astype(np.float32),
        "deltaR_level1": test_bundle["calib_axis_errors"]["delta_R"].astype(np.float32),
        "pred_residual": pred_test.astype(np.float32),
        "target_residual": te_tgt.astype(np.float32),
        "y_test": te_y.astype(np.int64),
    }
    _save_outputs(args, out_dir, model, best, test_bundle, arrays, metadata)

    print("\nLevel-1 Jet Calibrator Test")
    print(f"  HLT response score:       {test_bundle['hlt_response_score']:.6f}")
    print(f"  Level-1 response score:   {test_bundle['calib_response_score']:.6f}")
    print(f"  Response improvement:     {test_bundle['hlt_response_score'] - test_bundle['calib_response_score']:.6f}")
    print(f"  HLT axis score:           {test_bundle['hlt_axis_score']:.6f}")
    print(f"  Level-1 axis score:       {test_bundle['calib_axis_score']:.6f}")
    print(f"  Axis improvement:         {test_bundle['hlt_axis_score'] - test_bundle['calib_axis_score']:.6f}")
    print(f"  Axis fraction improved:   {test_bundle['fraction_axis_improved']:.4f}")
    print(f"Saved summary: {out_dir / 'level1_jet_calibrator_summary.json'}")


if __name__ == "__main__":
    main()
