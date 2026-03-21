#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root is importable when running from plots/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as BASE_CONFIG,
    OfflineReconstructor,
    apply_hlt_effects_realistic_nomap,
    reconstruct_dataset,
    wrap_phi_np,
)
from unmerge_correct_hlt import (
    compute_features,
    standardize,
)


def _deepcopy_cfg() -> dict:
    return json.loads(json.dumps(BASE_CONFIG))


def _pairwise_dr(a_eta: np.ndarray, a_phi: np.ndarray, b_eta: np.ndarray, b_phi: np.ndarray) -> np.ndarray:
    deta = a_eta[:, None] - b_eta[None, :]
    dphi = wrap_phi_np(a_phi[:, None] - b_phi[None, :])
    return np.sqrt(deta * deta + dphi * dphi)


def _min_drs(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    if src.shape[0] == 0 or tgt.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    dr = _pairwise_dr(src[:, 1], src[:, 2], tgt[:, 1], tgt[:, 2])
    return dr.min(axis=1)


def _chamfer_eta_phi(pred: np.ndarray, truth: np.ndarray) -> float:
    if pred.shape[0] == 0 and truth.shape[0] == 0:
        return 0.0
    if pred.shape[0] == 0 or truth.shape[0] == 0:
        return float("nan")
    dr = _pairwise_dr(pred[:, 1], pred[:, 2], truth[:, 1], truth[:, 2])
    return float(dr.min(axis=1).mean() + dr.min(axis=0).mean())


def _greedy_unique_match(pred: np.ndarray, truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy one-to-one matching by minimum dR.
    Returns:
      matched_dR: [K]
      pt_ratio_pred_over_truth: [K]
    """
    m = pred.shape[0]
    n = truth.shape[0]
    if m == 0 or n == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    dr = _pairwise_dr(pred[:, 1], pred[:, 2], truth[:, 1], truth[:, 2])
    used_p = np.zeros((m,), dtype=bool)
    used_t = np.zeros((n,), dtype=bool)
    k = min(m, n)
    out_dr = []
    out_ratio = []

    for _ in range(k):
        d = dr.copy()
        d[used_p, :] = np.inf
        d[:, used_t] = np.inf
        flat = np.argmin(d)
        i, j = np.unravel_index(flat, d.shape)
        if not np.isfinite(d[i, j]):
            break
        used_p[i] = True
        used_t[j] = True
        out_dr.append(float(d[i, j]))
        out_ratio.append(float(pred[i, 0] / max(truth[j, 0], 1e-8)))

    return np.array(out_dr, dtype=np.float64), np.array(out_ratio, dtype=np.float64)


def _pct(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _binned_stats(
    x: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    min_count: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.full_like(centers, np.nan, dtype=np.float64)
    p68 = np.full_like(centers, np.nan, dtype=np.float64)
    cnt = np.zeros_like(centers, dtype=np.int64)
    for i in range(edges.size - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i + 1 < edges.size - 1:
            m = (x >= lo) & (x < hi)
        else:
            m = (x >= lo) & (x <= hi)
        c = int(m.sum())
        cnt[i] = c
        if c < int(min_count):
            continue
        yy = y[m]
        med[i] = float(np.median(yy))
        p68[i] = float(np.percentile(yy, 68))
    return centers, med, p68, cnt


def _load_constituent_rows(files: list[Path], global_rows: np.ndarray, max_constits: int) -> np.ndarray:
    """
    Load selected global row indices from concatenated H5 files into [N, max_constits, 4].
    global_rows are indices in the concatenation order across files.
    """
    if global_rows.size == 0:
        return np.zeros((0, max_constits, 4), dtype=np.float32)

    order = np.argsort(global_rows)
    rows_sorted = global_rows[order]
    out_sorted = np.zeros((rows_sorted.size, max_constits, 4), dtype=np.float32)

    cursor = 0
    start_global = 0
    for fp in files:
        with h5py.File(fp, "r") as f:
            n = int(f["labels"].shape[0])
            end_global = start_global + n
            # Rows in this file's range.
            lo = np.searchsorted(rows_sorted, start_global, side="left")
            hi = np.searchsorted(rows_sorted, end_global, side="left")
            if hi > lo:
                local = (rows_sorted[lo:hi] - start_global).astype(np.int64)
                pt = f["fjet_clus_pt"][local, :max_constits].astype(np.float32)
                eta = f["fjet_clus_eta"][local, :max_constits].astype(np.float32)
                phi = f["fjet_clus_phi"][local, :max_constits].astype(np.float32)
                ene = f["fjet_clus_E"][local, :max_constits].astype(np.float32)
                out_sorted[lo:hi] = np.stack([pt, eta, phi, ene], axis=-1)
            start_global = end_global
            cursor = hi
        if cursor >= rows_sorted.size:
            break

    # Restore original order of requested rows.
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return out_sorted[inv]


def main() -> None:
    ap = argparse.ArgumentParser(description="Reconstructor placement-quality analysis.")
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n_eval_jets", type=int, default=12000, help="Max # test jets to analyze.")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--weight_threshold", type=float, default=0.03)
    ap.add_argument("--disable_budget_topk", action="store_true")
    ap.add_argument("--seed", type=int, default=52)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "reconstructor_placement_pack"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_setup = json.loads((run_dir / "data_setup.json").read_text(encoding="utf-8"))
    splits = np.load(run_dir / "data_splits.npz")
    train_files = []
    for p in data_setup["train_files"]:
        cand = Path(p)
        if cand.exists():
            train_files.append(cand)
            continue
        # Fallback for runs trained on another machine path.
        local_cand = PROJECT_ROOT / "data" / cand.name
        if local_cand.exists():
            train_files.append(local_cand)
            continue
        raise FileNotFoundError(f"Could not resolve training file path: {p}")
    max_constits = int(data_setup["max_constits"])
    n_train_jets = int(data_setup["n_train_jets"])
    offset = int(data_setup["offset_jets"])
    seed = int(data_setup.get("seed", args.seed))

    cfg = _deepcopy_cfg()
    cfg["hlt_effects"].update(data_setup["hlt_effects"])

    test_idx = splits["test_idx"].astype(np.int64)
    if int(args.n_eval_jets) > 0 and test_idx.size > int(args.n_eval_jets):
        rs = np.random.RandomState(int(args.seed))
        test_idx = rs.choice(test_idx, size=int(args.n_eval_jets), replace=False)
    test_idx = np.sort(test_idx)

    # Load only selected rows from raw H5 and recreate pseudo-HLT for this subset.
    global_rows = offset + test_idx
    const_off = _load_constituent_rows(train_files, global_rows, max_constits=max_constits)
    mask_raw = const_off[:, :, 0] > 0.0
    mask_off = mask_raw & (const_off[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off[~mask_off] = 0.0
    hlt_const, hlt_mask, _, _ = apply_hlt_effects_realistic_nomap(
        const_off,
        mask_off,
        cfg,
        seed=seed,
    )
    feat_hlt = compute_features(hlt_const, hlt_mask)
    means = splits["means"].astype(np.float32)
    stds = splits["stds"].astype(np.float32)
    feat_hlt_std = standardize(feat_hlt, hlt_mask, means, stds)

    model_ckpt = torch.load(run_dir / "offline_reconstructor.pt", map_location="cpu")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    model.load_state_dict(model_ckpt["model"])
    model.eval()

    reco_const, reco_mask, reco_merge_flag, _, _, _, _, _, _ = reconstruct_dataset(
        model=model,
        feat_hlt=feat_hlt_std,
        mask_hlt=hlt_mask,
        const_hlt=hlt_const,
        max_constits=max_constits,
        device=device,
        batch_size=int(args.batch_size),
        weight_threshold=float(args.weight_threshold),
        use_budget_topk=not bool(args.disable_budget_topk),
    )

    # Placement analysis.
    dr_nearest_created = []
    deta_created = []
    dphi_created = []
    dr_match_created = []
    pt_ratio_match_created = []
    chamfer_hlt_off = []
    chamfer_reco_off = []
    hlt_dr_nearest = []
    hlt_deta_nearest = []
    hlt_dphi_nearest = []
    hlt_pt = []
    hlt_abseta = []
    created_pt = []
    created_abseta = []

    for k in range(test_idx.size):
        off = const_off[k][mask_off[k]]
        hlt = hlt_const[k][hlt_mask[k]]
        rec = reco_const[k][reco_mask[k]]
        rec_merge_mask = reco_merge_flag[k][reco_mask[k]] > 0.5
        rec_merge = rec[rec_merge_mask]

        c_hlt = _chamfer_eta_phi(hlt, off)
        c_rec = _chamfer_eta_phi(rec, off)
        if np.isfinite(c_hlt):
            chamfer_hlt_off.append(c_hlt)
        if np.isfinite(c_rec):
            chamfer_reco_off.append(c_rec)

        if hlt.shape[0] and off.shape[0]:
            dr_hlt = _pairwise_dr(hlt[:, 1], hlt[:, 2], off[:, 1], off[:, 2])
            nn_hlt = dr_hlt.argmin(axis=1)
            hlt_dr_nearest.extend(dr_hlt[np.arange(dr_hlt.shape[0]), nn_hlt].tolist())
            hlt_deta_nearest.extend((hlt[:, 1] - off[nn_hlt, 1]).tolist())
            hlt_dphi_nearest.extend(wrap_phi_np(hlt[:, 2] - off[nn_hlt, 2]).tolist())
            hlt_pt.extend(hlt[:, 0].tolist())
            hlt_abseta.extend(np.abs(hlt[:, 1]).tolist())

        if rec_merge.shape[0] == 0 or off.shape[0] == 0:
            continue

        dr = _pairwise_dr(rec_merge[:, 1], rec_merge[:, 2], off[:, 1], off[:, 2])
        nn = dr.argmin(axis=1)
        dr_nearest_created.extend(dr[np.arange(dr.shape[0]), nn].tolist())
        deta_created.extend((rec_merge[:, 1] - off[nn, 1]).tolist())
        dphi_created.extend(wrap_phi_np(rec_merge[:, 2] - off[nn, 2]).tolist())
        created_pt.extend(rec_merge[:, 0].tolist())
        created_abseta.extend(np.abs(rec_merge[:, 1]).tolist())

        m_dr, m_pt_ratio = _greedy_unique_match(rec_merge, off)
        dr_match_created.extend(m_dr.tolist())
        pt_ratio_match_created.extend(m_pt_ratio.tolist())

    dr_nearest_created = np.array(dr_nearest_created, dtype=np.float64)
    deta_created = np.array(deta_created, dtype=np.float64)
    dphi_created = np.array(dphi_created, dtype=np.float64)
    dr_match_created = np.array(dr_match_created, dtype=np.float64)
    pt_ratio_match_created = np.array(pt_ratio_match_created, dtype=np.float64)
    chamfer_hlt_off = np.array(chamfer_hlt_off, dtype=np.float64)
    chamfer_reco_off = np.array(chamfer_reco_off, dtype=np.float64)
    hlt_dr_nearest = np.array(hlt_dr_nearest, dtype=np.float64)
    hlt_deta_nearest = np.array(hlt_deta_nearest, dtype=np.float64)
    hlt_dphi_nearest = np.array(hlt_dphi_nearest, dtype=np.float64)
    hlt_pt = np.array(hlt_pt, dtype=np.float64)
    hlt_abseta = np.array(hlt_abseta, dtype=np.float64)
    created_pt = np.array(created_pt, dtype=np.float64)
    created_abseta = np.array(created_abseta, dtype=np.float64)

    summary = {
        "run_dir": str(run_dir),
        "n_test_jets_analyzed": int(test_idx.size),
        "n_created_merge_tokens_analyzed": int(dr_nearest_created.size),
        "created_merge_nearest_dR_mean": float(np.mean(dr_nearest_created)) if dr_nearest_created.size else None,
        "created_merge_nearest_dR_median": _pct(dr_nearest_created, 50),
        "created_merge_nearest_dR_p68": _pct(dr_nearest_created, 68),
        "created_merge_nearest_dR_p90": _pct(dr_nearest_created, 90),
        "created_merge_nearest_dR_frac_lt_001": float(np.mean(dr_nearest_created < 0.01)) if dr_nearest_created.size else None,
        "created_merge_nearest_dR_frac_lt_002": float(np.mean(dr_nearest_created < 0.02)) if dr_nearest_created.size else None,
        "created_merge_nearest_dR_frac_lt_005": float(np.mean(dr_nearest_created < 0.05)) if dr_nearest_created.size else None,
        "created_merge_matched_dR_median": _pct(dr_match_created, 50),
        "created_merge_matched_pt_ratio_median": _pct(pt_ratio_match_created, 50),
        "created_merge_matched_pt_ratio_p16": _pct(pt_ratio_match_created, 16),
        "created_merge_matched_pt_ratio_p84": _pct(pt_ratio_match_created, 84),
        "chamfer_eta_phi_hlt_to_offline_mean": float(np.mean(chamfer_hlt_off)) if chamfer_hlt_off.size else None,
        "chamfer_eta_phi_reco_to_offline_mean": float(np.mean(chamfer_reco_off)) if chamfer_reco_off.size else None,
        "chamfer_improvement_pct": (
            float(100.0 * (1.0 - np.mean(chamfer_reco_off) / max(np.mean(chamfer_hlt_off), 1e-12)))
            if chamfer_hlt_off.size and chamfer_reco_off.size
            else None
        ),
        "hlt_nearest_dR_mean": float(np.mean(hlt_dr_nearest)) if hlt_dr_nearest.size else None,
        "hlt_nearest_dR_median": _pct(hlt_dr_nearest, 50),
        "hlt_nearest_dR_p68": _pct(hlt_dr_nearest, 68),
        "hlt_nearest_dR_p90": _pct(hlt_dr_nearest, 90),
        "hlt_nearest_dR_frac_lt_001": float(np.mean(hlt_dr_nearest < 0.01)) if hlt_dr_nearest.size else None,
        "hlt_nearest_dR_frac_lt_002": float(np.mean(hlt_dr_nearest < 0.02)) if hlt_dr_nearest.size else None,
        "hlt_nearest_dR_frac_lt_005": float(np.mean(hlt_dr_nearest < 0.05)) if hlt_dr_nearest.size else None,
    }
    (out_dir / "placement_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot 1: dR distribution for created-merge tokens.
    if dr_nearest_created.size:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
        axs[0].hist(dr_nearest_created, bins=80, color="#1f77b4", alpha=0.9, density=True)
        axs[0].set_xlabel("Nearest offline dR for created-merge token")
        axs[0].set_ylabel("Density")
        axs[0].set_title("Placement Accuracy (Created Merge Tokens)")
        axs[0].grid(alpha=0.25)

        xs = np.sort(dr_nearest_created)
        ys = np.arange(1, xs.size + 1) / xs.size
        axs[1].plot(xs, ys, lw=2, color="#2ca02c")
        for thr in [0.01, 0.02, 0.05]:
            axs[1].axvline(thr, ls="--", lw=1, color="gray")
        axs[1].set_xlabel("Nearest offline dR")
        axs[1].set_ylabel("CDF")
        axs[1].set_title("CDF of Placement Error")
        axs[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "placement_created_merge_dR.png", dpi=220)
        plt.close(fig)

    # Plot 1b: dR distribution for HLT tokens (baseline vs offline).
    if hlt_dr_nearest.size:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
        axs[0].hist(hlt_dr_nearest, bins=80, color="#9467bd", alpha=0.9, density=True)
        axs[0].set_xlabel("Nearest offline dR for HLT token")
        axs[0].set_ylabel("Density")
        axs[0].set_title("Placement Accuracy (HLT Tokens)")
        axs[0].grid(alpha=0.25)

        xs = np.sort(hlt_dr_nearest)
        ys = np.arange(1, xs.size + 1) / xs.size
        axs[1].plot(xs, ys, lw=2, color="#8c564b")
        for thr in [0.01, 0.02, 0.05]:
            axs[1].axvline(thr, ls="--", lw=1, color="gray")
        axs[1].set_xlabel("Nearest offline dR")
        axs[1].set_ylabel("CDF")
        axs[1].set_title("CDF of HLT Placement Error")
        axs[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "placement_hlt_dR.png", dpi=220)
        plt.close(fig)

    # Plot 2: delta eta/phi cloud.
    if deta_created.size and dphi_created.size:
        fig = plt.figure(figsize=(6, 5.5))
        plt.hexbin(deta_created, dphi_created, gridsize=70, bins="log", cmap="viridis")
        plt.colorbar(label="log10(count)")
        plt.xlabel("Delta eta (pred - nearest offline)")
        plt.ylabel("Delta phi (pred - nearest offline)")
        plt.title("Created-Merge Placement Residuals")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(out_dir / "placement_created_merge_deta_dphi.png", dpi=220)
        plt.close(fig)

    # Plot 2b: HLT delta eta/phi cloud vs offline.
    if hlt_deta_nearest.size and hlt_dphi_nearest.size:
        fig = plt.figure(figsize=(6, 5.5))
        plt.hexbin(hlt_deta_nearest, hlt_dphi_nearest, gridsize=70, bins="log", cmap="magma")
        plt.colorbar(label="log10(count)")
        plt.xlabel("Delta eta (HLT - nearest offline)")
        plt.ylabel("Delta phi (HLT - nearest offline)")
        plt.title("HLT Placement Residuals")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(out_dir / "placement_hlt_deta_dphi.png", dpi=220)
        plt.close(fig)

    # Plot 3: PT ratio from unique greedy matching.
    if pt_ratio_match_created.size:
        fig = plt.figure(figsize=(6.5, 4.8))
        plt.hist(pt_ratio_match_created, bins=80, color="#ff7f0e", alpha=0.9, density=True)
        plt.axvline(1.0, color="black", ls="--", lw=1)
        plt.xlabel("Matched pT ratio (pred / offline)")
        plt.ylabel("Density")
        plt.title("Created-Merge pT Matching Quality")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "placement_created_merge_pt_ratio.png", dpi=220)
        plt.close(fig)

    # Plot 4: set-level chamfer comparison.
    if chamfer_hlt_off.size and chamfer_reco_off.size:
        fig = plt.figure(figsize=(6.8, 5.0))
        bins = np.linspace(
            min(float(chamfer_hlt_off.min()), float(chamfer_reco_off.min())),
            max(float(chamfer_hlt_off.max()), float(chamfer_reco_off.max())),
            80,
        )
        plt.hist(chamfer_hlt_off, bins=bins, histtype="step", lw=2, density=True, label="HLT vs Offline")
        plt.hist(chamfer_reco_off, bins=bins, histtype="step", lw=2, density=True, label="Reco vs Offline")
        plt.xlabel("Set-level eta-phi Chamfer")
        plt.ylabel("Density")
        plt.title("Set Matching Quality (Lower is Better)")
        plt.grid(alpha=0.25)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / "placement_chamfer_compare.png", dpi=220)
        plt.close(fig)

    # Plot 5: nearest-dR vs candidate pT (HLT vs created-merge).
    if hlt_dr_nearest.size and hlt_pt.size and dr_nearest_created.size and created_pt.size:
        # MeV bins, shown in GeV on x-axis.
        pt_edges = np.array([500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000], dtype=np.float64)
        c_h, m_h, p_h, n_h = _binned_stats(hlt_pt, hlt_dr_nearest, pt_edges, min_count=200)
        c_c, m_c, p_c, n_c = _binned_stats(created_pt, dr_nearest_created, pt_edges, min_count=200)

        fig = plt.figure(figsize=(7.2, 5.0))
        plt.plot(c_h / 1000.0, m_h, marker="o", lw=2, color="#9467bd", label="HLT median dR")
        plt.plot(c_h / 1000.0, p_h, marker="o", lw=1.5, ls="--", color="#9467bd", alpha=0.75, label="HLT p68 dR")
        plt.plot(c_c / 1000.0, m_c, marker="s", lw=2, color="#1f77b4", label="Created-unmerge median dR")
        plt.plot(c_c / 1000.0, p_c, marker="s", lw=1.5, ls="--", color="#1f77b4", alpha=0.75, label="Created-unmerge p68 dR")
        plt.xscale("log")
        plt.xlabel("Candidate pT [GeV]")
        plt.ylabel("Nearest offline dR")
        plt.title("Candidate Placement Error vs pT")
        plt.grid(alpha=0.25, which="both")
        plt.legend(frameon=False, fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / "placement_candidate_dR_vs_pt.png", dpi=220)
        plt.close(fig)

    # Plot 6: nearest-dR vs |eta| (HLT vs created-merge).
    if hlt_dr_nearest.size and hlt_abseta.size and dr_nearest_created.size and created_abseta.size:
        eta_edges = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.5, 3.0, 4.0, 5.0], dtype=np.float64)
        c_h, m_h, p_h, n_h = _binned_stats(hlt_abseta, hlt_dr_nearest, eta_edges, min_count=200)
        c_c, m_c, p_c, n_c = _binned_stats(created_abseta, dr_nearest_created, eta_edges, min_count=200)

        fig = plt.figure(figsize=(7.2, 5.0))
        plt.plot(c_h, m_h, marker="o", lw=2, color="#9467bd", label="HLT median dR")
        plt.plot(c_h, p_h, marker="o", lw=1.5, ls="--", color="#9467bd", alpha=0.75, label="HLT p68 dR")
        plt.plot(c_c, m_c, marker="s", lw=2, color="#1f77b4", label="Created-unmerge median dR")
        plt.plot(c_c, p_c, marker="s", lw=1.5, ls="--", color="#1f77b4", alpha=0.75, label="Created-unmerge p68 dR")
        plt.xlabel("Candidate |eta|")
        plt.ylabel("Nearest offline dR")
        plt.title("Candidate Placement Error vs |eta|")
        plt.grid(alpha=0.25)
        plt.legend(frameon=False, fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / "placement_candidate_dR_vs_abseta.png", dpi=220)
        plt.close(fig)

    print(f"Saved placement-focused pack to: {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
