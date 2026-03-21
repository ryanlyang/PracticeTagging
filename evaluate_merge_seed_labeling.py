#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate weak seed-label strategies for "merged-product" vs "not-merged" tokens.

This script:
1) Loads offline constituents from HDF5.
2) Builds pseudo-HLT with the same realistic corruption model used in the
   reconstructor pipeline (merge + efficiency loss + smearing + reassignment).
3) Keeps ancestry tracking to define evaluation truth:
     - merged-product truth: origin_counts >= 2
     - not-merged truth:     origin_counts == 1
4) Sweeps many seed-label strategies and reports quality metrics.
5) Saves diagnostics to understand failure modes if strategy quality is poor.

Notes:
- Seed labels are weak labels, so high precision + moderate coverage is often
  preferred over high recall.
- Optional per-jet cap on merged seeds can be tied to:
      cap_j = floor(cap_factor * (N_offline_j - N_hlt_j))
  (as requested for tests like cap_factor=0.90).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from generate_realistic_hlt import (
    apply_hlt_effects_realistic_with_tracking,
    load_raw_constituents_from_h5,
    wrap_phi,
)
from offline_reconstructor_no_gt_local30kv2 import CONFIG as BASE_CONFIG


def _deepcopy_cfg() -> Dict:
    return json.loads(json.dumps(BASE_CONFIG))


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _f1(p: float, r: float) -> float:
    return _safe_div(2.0 * p * r, p + r)


@dataclass
class Strategy:
    merged_nn_dr_max: float
    merged_density_min: int
    merged_pt_max: float
    not_nn_dr_min: float
    not_density_max: int
    not_pt_min: float
    cap_factor: float  # <0 => no cap
    overlap_policy: str  # "drop" or "prefer_merge" or "prefer_not"

    def as_dict(self) -> Dict:
        return {
            "merged_nn_dr_max": float(self.merged_nn_dr_max),
            "merged_density_min": int(self.merged_density_min),
            "merged_pt_max": float(self.merged_pt_max),
            "not_nn_dr_min": float(self.not_nn_dr_min),
            "not_density_max": int(self.not_density_max),
            "not_pt_min": float(self.not_pt_min),
            "cap_factor": float(self.cap_factor),
            "overlap_policy": str(self.overlap_policy),
        }


def compute_local_geometry_features(
    hlt_const: np.ndarray,
    hlt_mask: np.ndarray,
    density_radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      nn_dr: nearest-neighbor dR (inf for invalid tokens)
      density: count of neighbors within density_radius
    """
    n_jets, max_part, _ = hlt_const.shape
    nn_dr = np.full((n_jets, max_part), np.inf, dtype=np.float32)
    density = np.zeros((n_jets, max_part), dtype=np.float32)

    for j in range(n_jets):
        valid = np.where(hlt_mask[j])[0]
        if len(valid) == 0:
            continue
        if len(valid) == 1:
            nn_dr[j, valid[0]] = np.inf
            density[j, valid[0]] = 0.0
            continue

        eta = hlt_const[j, valid, 1]
        phi = hlt_const[j, valid, 2]
        deta = eta[:, None] - eta[None, :]
        dphi = wrap_phi(phi[:, None] - phi[None, :])
        dr = np.sqrt(deta * deta + dphi * dphi).astype(np.float32)

        np.fill_diagonal(dr, np.inf)
        nn = dr.min(axis=1)

        near = (dr < float(density_radius)).astype(np.int32)
        np.fill_diagonal(near, 0)
        dens = near.sum(axis=1).astype(np.float32)

        nn_dr[j, valid] = nn
        density[j, valid] = dens

    return nn_dr, density


def _apply_overlap_policy(
    merged_seed: np.ndarray,
    not_seed: np.ndarray,
    policy: str,
    nn_dr: np.ndarray,
    merged_nn_dr_max: float,
    not_nn_dr_min: float,
) -> Tuple[np.ndarray, np.ndarray]:
    overlap = merged_seed & not_seed
    if not np.any(overlap):
        return merged_seed, not_seed

    if policy == "drop":
        merged_seed = merged_seed & (~overlap)
        not_seed = not_seed & (~overlap)
        return merged_seed, not_seed

    if policy == "prefer_merge":
        not_seed = not_seed & (~overlap)
        return merged_seed, not_seed

    if policy == "prefer_not":
        merged_seed = merged_seed & (~overlap)
        return merged_seed, not_seed

    # margin-based tie-break as fallback
    merge_margin = (float(merged_nn_dr_max) - nn_dr)
    not_margin = (nn_dr - float(not_nn_dr_min))
    choose_merge = overlap & (merge_margin >= not_margin)
    choose_not = overlap & (~choose_merge)
    merged_seed = (merged_seed & (~overlap)) | choose_merge
    not_seed = (not_seed & (~overlap)) | choose_not
    return merged_seed, not_seed


def _per_jet_cap_merged(
    merged_seed: np.ndarray,
    score: np.ndarray,
    cap_per_jet: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Enforce per-jet max merged seed count by keeping highest-score candidates.
    """
    out = np.zeros_like(merged_seed, dtype=bool)
    n_jets = merged_seed.shape[0]
    for j in range(n_jets):
        k = int(max(cap_per_jet[j], 0))
        if k <= 0:
            continue
        cand = np.where(merged_seed[j] & valid_mask[j])[0]
        if cand.size == 0:
            continue
        if cand.size <= k:
            out[j, cand] = True
            continue
        cand_score = score[j, cand]
        top_rel = np.argpartition(cand_score, -k)[-k:]
        keep = cand[top_rel]
        out[j, keep] = True
    return out


def evaluate_strategy(
    strat: Strategy,
    hlt_pt: np.ndarray,
    nn_dr: np.ndarray,
    density: np.ndarray,
    valid_mask: np.ndarray,
    true_merged: np.ndarray,
    true_not: np.ndarray,
    count_gap: np.ndarray,
) -> Dict:
    # Candidate seeds from local rules.
    merged_seed = (
        valid_mask
        & (nn_dr <= float(strat.merged_nn_dr_max))
        & (density >= float(strat.merged_density_min))
        & (hlt_pt <= float(strat.merged_pt_max))
    )
    not_seed = (
        valid_mask
        & (nn_dr >= float(strat.not_nn_dr_min))
        & (density <= float(strat.not_density_max))
        & (hlt_pt >= float(strat.not_pt_min))
    )

    merged_seed, not_seed = _apply_overlap_policy(
        merged_seed=merged_seed,
        not_seed=not_seed,
        policy=strat.overlap_policy,
        nn_dr=nn_dr,
        merged_nn_dr_max=float(strat.merged_nn_dr_max),
        not_nn_dr_min=float(strat.not_nn_dr_min),
    )

    # Optional per-jet cap tied to offline-hlt count gap.
    if float(strat.cap_factor) >= 0.0:
        cap = np.floor(float(strat.cap_factor) * np.maximum(count_gap, 0)).astype(np.int32)
        merge_score = (-nn_dr + 0.03 * density).astype(np.float32)
        merged_seed = _per_jet_cap_merged(
            merged_seed=merged_seed,
            score=merge_score,
            cap_per_jet=cap,
            valid_mask=valid_mask,
        )
        # Ensure no overlap after cap.
        not_seed = not_seed & (~merged_seed)

    labeled = merged_seed | not_seed

    n_valid = int(valid_mask.sum())
    n_labeled = int(labeled.sum())
    n_unlabeled = int(n_valid - n_labeled)

    tp_m = int((merged_seed & true_merged).sum())
    fp_m = int((merged_seed & true_not).sum())
    fn_m = int((true_merged & (~merged_seed)).sum())
    pred_m = int(merged_seed.sum())
    true_m = int(true_merged.sum())

    tp_n = int((not_seed & true_not).sum())
    fp_n = int((not_seed & true_merged).sum())
    fn_n = int((true_not & (~not_seed)).sum())
    pred_n = int(not_seed.sum())
    true_n = int(true_not.sum())

    p_m = _safe_div(tp_m, pred_m)
    r_m = _safe_div(tp_m, true_m)
    f1_m = _f1(p_m, r_m)

    p_n = _safe_div(tp_n, pred_n)
    r_n = _safe_div(tp_n, true_n)
    f1_n = _f1(p_n, r_n)

    correct_labeled = int((merged_seed & true_merged).sum() + (not_seed & true_not).sum())
    acc_labeled = _safe_div(correct_labeled, n_labeled)
    coverage = _safe_div(n_labeled, n_valid)

    # A pragmatic ranking score that favors merged-class quality and reasonable coverage.
    score = 0.50 * f1_m + 0.20 * f1_n + 0.20 * coverage + 0.10 * acc_labeled

    return {
        **strat.as_dict(),
        "n_valid_tokens": n_valid,
        "n_labeled_tokens": n_labeled,
        "n_unlabeled_tokens": n_unlabeled,
        "coverage": coverage,
        "merged_precision": p_m,
        "merged_recall": r_m,
        "merged_f1": f1_m,
        "not_precision": p_n,
        "not_recall": r_n,
        "not_f1": f1_n,
        "labeled_accuracy": acc_labeled,
        "tp_merged": tp_m,
        "fp_merged": fp_m,
        "fn_merged": fn_m,
        "tp_not": tp_n,
        "fp_not": fp_n,
        "fn_not": fn_n,
        "strategy_score": score,
    }


def build_strategy_grid(
    hlt_pt_valid: np.ndarray,
    args: argparse.Namespace,
) -> List[Strategy]:
    # Robust to unit scale (MeV/GeV) by deriving pT cuts from percentiles.
    q50 = float(np.percentile(hlt_pt_valid, 50.0))
    q70 = float(np.percentile(hlt_pt_valid, 70.0))
    q80 = float(np.percentile(hlt_pt_valid, 80.0))
    q90 = float(np.percentile(hlt_pt_valid, 90.0))

    merged_nn_dr_max_list = [0.005, 0.007, 0.009, 0.011, 0.013]
    merged_density_min_list = [1, 2, 3]
    merged_pt_max_list = [q80, q90, float(np.inf)]

    not_nn_dr_min_list = [0.030, 0.040, 0.050, 0.060, 0.075]
    not_density_max_list = [0, 1, 2]
    not_pt_min_list = [q50, q70, q80]

    cap_factors = [-1.0, 0.90, 1.00]
    overlap_policies = ["drop", "prefer_not"]

    if args.fast_sweep:
        merged_nn_dr_max_list = [0.007, 0.009, 0.011]
        merged_density_min_list = [1, 2]
        merged_pt_max_list = [q90, float(np.inf)]
        not_nn_dr_min_list = [0.040, 0.060]
        not_density_max_list = [0, 1]
        not_pt_min_list = [q70, q80]
        cap_factors = [-1.0, 0.90]
        overlap_policies = ["drop"]

    grid = []
    for (
        mdr,
        mdens,
        mptmax,
        ndr,
        ndens,
        npt,
        capf,
        overlap,
    ) in itertools.product(
        merged_nn_dr_max_list,
        merged_density_min_list,
        merged_pt_max_list,
        not_nn_dr_min_list,
        not_density_max_list,
        not_pt_min_list,
        cap_factors,
        overlap_policies,
    ):
        # Ensure logical gap between merged and not-merged dr thresholds.
        if ndr <= mdr:
            continue
        grid.append(
            Strategy(
                merged_nn_dr_max=float(mdr),
                merged_density_min=int(mdens),
                merged_pt_max=float(mptmax),
                not_nn_dr_min=float(ndr),
                not_density_max=int(ndens),
                not_pt_min=float(npt),
                cap_factor=float(capf),
                overlap_policy=str(overlap),
            )
        )

    return grid


def _compute_bin_metrics(
    value: np.ndarray,
    bins: np.ndarray,
    pred_merged: np.ndarray,
    pred_not: np.ndarray,
    true_merged: np.ndarray,
    true_not: np.ndarray,
    valid: np.ndarray,
) -> List[Dict]:
    out = []
    for i in range(len(bins) - 1):
        lo = float(bins[i])
        hi = float(bins[i + 1])
        m = valid & (value >= lo) & (value < hi)
        n = int(m.sum())
        if n == 0:
            continue
        pm = int(pred_merged[m].sum())
        pn = int(pred_not[m].sum())
        tm = int(true_merged[m].sum())
        tn = int(true_not[m].sum())
        tp_m = int((pred_merged[m] & true_merged[m]).sum())
        tp_n = int((pred_not[m] & true_not[m]).sum())

        out.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "n_tokens": n,
                "coverage": _safe_div(pm + pn, n),
                "merged_precision": _safe_div(tp_m, pm),
                "merged_recall": _safe_div(tp_m, tm),
                "not_precision": _safe_div(tp_n, pn),
                "not_recall": _safe_div(tp_n, tn),
            }
        )
    return out


def _plot_nn_dr_hist(save_dir: Path, nn_dr_valid: np.ndarray, true_merged_valid: np.ndarray, true_not_valid: np.ndarray) -> None:
    plt.figure(figsize=(7.5, 5.0))
    bins = np.linspace(0.0, 0.12, 80)
    plt.hist(
        nn_dr_valid[true_not_valid],
        bins=bins,
        density=True,
        alpha=0.55,
        label="True not-merged",
    )
    plt.hist(
        nn_dr_valid[true_merged_valid],
        bins=bins,
        density=True,
        alpha=0.55,
        label="True merged-product",
    )
    plt.xlabel("HLT nearest-neighbor dR")
    plt.ylabel("Density")
    plt.title("Separability Hint: Nearest-Neighbor dR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "truth_nn_dr_distribution.png", dpi=180)
    plt.close()


def _plot_density_hist(save_dir: Path, density_valid: np.ndarray, true_merged_valid: np.ndarray, true_not_valid: np.ndarray) -> None:
    plt.figure(figsize=(7.5, 5.0))
    bins = np.arange(-0.5, 16.5, 1.0)
    plt.hist(
        density_valid[true_not_valid],
        bins=bins,
        density=True,
        alpha=0.55,
        label="True not-merged",
    )
    plt.hist(
        density_valid[true_merged_valid],
        bins=bins,
        density=True,
        alpha=0.55,
        label="True merged-product",
    )
    plt.xlabel("Local density (neighbors within radius)")
    plt.ylabel("Density")
    plt.title("Separability Hint: Local Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "truth_density_distribution.png", dpi=180)
    plt.close()


def _plot_top_scores(save_dir: Path, rows_sorted: List[Dict], top_k: int = 20) -> None:
    top = rows_sorted[:top_k]
    if not top:
        return
    labels = [f"#{i+1}" for i in range(len(top))]
    vals = [float(r["strategy_score"]) for r in top]

    plt.figure(figsize=(9, 4.8))
    plt.bar(np.arange(len(top)), vals)
    plt.xticks(np.arange(len(top)), labels, rotation=0)
    plt.ylabel("Strategy score")
    plt.xlabel("Rank")
    plt.title("Top Seed Strategies")
    plt.tight_layout()
    plt.savefig(save_dir / "seed_strategy_top_scores.png", dpi=180)
    plt.close()


def _plot_best_confusion(save_dir: Path, best: Dict) -> None:
    # Labeled-only confusion matrix, rows=true [merged, not], cols=pred [merged, not]
    cm = np.array(
        [
            [best["tp_merged"], best["fp_not"]],
            [best["fp_merged"], best["tp_not"]],
        ],
        dtype=np.float32,
    )
    row_sum = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm, np.maximum(row_sum, 1e-9))

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks([0, 1], labels=["Pred merged", "Pred not"])
    ax.set_yticks([0, 1], labels=["True merged", "True not"])
    ax.set_title("Best Strategy: Labeled-Only Confusion")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cmn[i, j]:.3f}\\n({int(cm[i, j])})", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_dir / "best_strategy_confusion_labeled.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep and evaluate merged/not-merged seed-label strategies.")
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_jets", type=int, default=200000)
    parser.add_argument("--offset_jets", type=int, default=0)
    parser.add_argument("--max_constits", type=int, default=100)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--save_dir", type=str, default="checkpoints/seed_labeling_eval")
    parser.add_argument("--run_name", type=str, default="rho090_style")
    parser.add_argument("--fast_sweep", action="store_true", help="Smaller strategy grid for quick iteration.")

    # Optional HLT knobs to probe sensitivity.
    parser.add_argument("--merge_radius", type=float, default=None)
    parser.add_argument("--smear_scale", type=float, default=1.0)
    parser.add_argument("--eff_scale", type=float, default=1.0)
    parser.add_argument("--density_radius", type=float, default=None)
    args = parser.parse_args()

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train_path)
    if train_path.is_dir():
        train_files = sorted(list(train_path.glob("*.h5")))
    else:
        train_files = [Path(p) for p in str(args.train_path).split(",") if p.strip()]
    if not train_files:
        raise FileNotFoundError(f"No .h5 files found in: {args.train_path}")

    need = int(args.offset_jets) + int(args.n_jets)
    print("Loading offline constituents...")
    const_full, labels_full = load_raw_constituents_from_h5(
        files=train_files,
        max_jets=need,
        max_constits=int(args.max_constits),
    )
    if const_full.shape[0] < need:
        raise RuntimeError(
            f"Need {need} jets (offset+n_jets) but loaded {const_full.shape[0]}."
        )
    const_off = const_full[int(args.offset_jets):need]
    labels = labels_full[int(args.offset_jets):need].astype(np.int64)

    cfg = _deepcopy_cfg()
    hcfg = cfg["hlt_effects"]
    if args.merge_radius is not None:
        hcfg["merge_radius"] = float(args.merge_radius)
    if args.density_radius is not None:
        hcfg["density_radius"] = float(args.density_radius)

    # Global scales (useful for controlled probes).
    smear_scale = float(args.smear_scale)
    for k in (
        "smear_a",
        "smear_b",
        "smear_c",
        "smear_sigma_min",
        "smear_sigma_max",
        "eta_smear_const",
        "eta_smear_inv_sqrt",
        "phi_smear_const",
        "phi_smear_inv_sqrt",
        "tail_sigma_scale",
        "tail_sigma_add",
    ):
        hcfg[k] = float(hcfg[k]) * smear_scale

    eff_scale = float(args.eff_scale)
    for k in (
        "eff_plateau_barrel",
        "eff_plateau_endcap",
        "eff_floor",
        "eff_ceil",
    ):
        hcfg[k] = float(np.clip(float(hcfg[k]) * eff_scale, 0.0, 0.999))

    raw_mask = const_off[:, :, 0] > 0.0
    off_mask = raw_mask & (const_off[:, :, 0] >= float(hcfg["pt_threshold_offline"]))

    print("Generating realistic pseudo-HLT with ancestry tracking...")
    hlt_const, hlt_mask, origin_counts, _, hlt_stats = apply_hlt_effects_realistic_with_tracking(
        const=const_off,
        mask=off_mask,
        cfg=cfg,
        seed=int(args.seed),
    )

    print("Computing token-local geometry features...")
    nn_dr, density = compute_local_geometry_features(
        hlt_const=hlt_const,
        hlt_mask=hlt_mask,
        density_radius=float(hcfg["density_radius"]),
    )

    valid = hlt_mask.astype(bool)
    hlt_pt = hlt_const[:, :, 0].astype(np.float32)
    hlt_abs_eta = np.abs(hlt_const[:, :, 1]).astype(np.float32)

    true_merged = valid & (origin_counts >= 2)
    true_not = valid & (origin_counts == 1)

    n_valid = int(valid.sum())
    n_true_merged = int(true_merged.sum())
    n_true_not = int(true_not.sum())
    frac_true_merged = _safe_div(n_true_merged, n_valid)
    frac_true_not = _safe_div(n_true_not, n_valid)

    print(f"Valid HLT tokens: {n_valid:,}")
    print(f"True merged-product tokens: {n_true_merged:,} ({100.0*frac_true_merged:.2f}%)")
    print(f"True not-merged tokens: {n_true_not:,} ({100.0*frac_true_not:.2f}%)")

    off_count = off_mask.sum(axis=1).astype(np.int32)
    hlt_count = hlt_mask.sum(axis=1).astype(np.int32)
    count_gap = (off_count - hlt_count).astype(np.int32)

    hlt_pt_valid = hlt_pt[valid]
    strategies = build_strategy_grid(hlt_pt_valid=hlt_pt_valid, args=args)
    print(f"Sweeping {len(strategies):,} seed strategies...")

    rows: List[Dict] = []
    for i, strat in enumerate(strategies):
        row = evaluate_strategy(
            strat=strat,
            hlt_pt=hlt_pt,
            nn_dr=nn_dr,
            density=density,
            valid_mask=valid,
            true_merged=true_merged,
            true_not=true_not,
            count_gap=count_gap,
        )
        row["strategy_id"] = i
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: float(r["strategy_score"]), reverse=True)
    best = rows_sorted[0]

    # Save full table.
    csv_path = save_root / "seed_strategy_sweep.csv"
    if rows_sorted:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
            writer.writeheader()
            writer.writerows(rows_sorted)

    top_k = min(10, len(rows_sorted))
    print("\nTop strategies (by strategy_score):")
    for rank in range(top_k):
        r = rows_sorted[rank]
        print(
            f"  #{rank+1:02d} score={r['strategy_score']:.4f} "
            f"cov={100.0*r['coverage']:.2f}% "
            f"M(P/R/F1)=({r['merged_precision']:.3f}/{r['merged_recall']:.3f}/{r['merged_f1']:.3f}) "
            f"N(P/R/F1)=({r['not_precision']:.3f}/{r['not_recall']:.3f}/{r['not_f1']:.3f}) "
            f"cap={r['cap_factor']}"
        )

    # Recompute best predictions for bin diagnostics.
    best_s = Strategy(
        merged_nn_dr_max=float(best["merged_nn_dr_max"]),
        merged_density_min=int(best["merged_density_min"]),
        merged_pt_max=float(best["merged_pt_max"]),
        not_nn_dr_min=float(best["not_nn_dr_min"]),
        not_density_max=int(best["not_density_max"]),
        not_pt_min=float(best["not_pt_min"]),
        cap_factor=float(best["cap_factor"]),
        overlap_policy=str(best["overlap_policy"]),
    )

    merged_seed_best = (
        valid
        & (nn_dr <= best_s.merged_nn_dr_max)
        & (density >= best_s.merged_density_min)
        & (hlt_pt <= best_s.merged_pt_max)
    )
    not_seed_best = (
        valid
        & (nn_dr >= best_s.not_nn_dr_min)
        & (density <= best_s.not_density_max)
        & (hlt_pt >= best_s.not_pt_min)
    )
    merged_seed_best, not_seed_best = _apply_overlap_policy(
        merged_seed=merged_seed_best,
        not_seed=not_seed_best,
        policy=best_s.overlap_policy,
        nn_dr=nn_dr,
        merged_nn_dr_max=best_s.merged_nn_dr_max,
        not_nn_dr_min=best_s.not_nn_dr_min,
    )
    if best_s.cap_factor >= 0.0:
        cap = np.floor(best_s.cap_factor * np.maximum(count_gap, 0)).astype(np.int32)
        merge_score = (-nn_dr + 0.03 * density).astype(np.float32)
        merged_seed_best = _per_jet_cap_merged(
            merged_seed=merged_seed_best,
            score=merge_score,
            cap_per_jet=cap,
            valid_mask=valid,
        )
        not_seed_best = not_seed_best & (~merged_seed_best)

    # Bin diagnostics for best strategy.
    pt_bins = np.percentile(hlt_pt_valid, [0, 10, 25, 40, 55, 70, 85, 95, 100])
    eta_bins = np.array([0.0, 0.4, 0.8, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0], dtype=np.float32)
    pt_bin_metrics = _compute_bin_metrics(
        value=hlt_pt,
        bins=np.unique(pt_bins),
        pred_merged=merged_seed_best,
        pred_not=not_seed_best,
        true_merged=true_merged,
        true_not=true_not,
        valid=valid,
    )
    eta_bin_metrics = _compute_bin_metrics(
        value=hlt_abs_eta,
        bins=np.unique(eta_bins),
        pred_merged=merged_seed_best,
        pred_not=not_seed_best,
        true_merged=true_merged,
        true_not=true_not,
        valid=valid,
    )

    # Summary JSON.
    summary = {
        "seed": int(args.seed),
        "n_jets": int(args.n_jets),
        "offset_jets": int(args.offset_jets),
        "max_constits": int(args.max_constits),
        "hlt_config": hcfg,
        "hlt_stats": hlt_stats,
        "global_truth": {
            "valid_tokens": int(n_valid),
            "true_merged_tokens": int(n_true_merged),
            "true_not_merged_tokens": int(n_true_not),
            "true_merged_fraction": float(frac_true_merged),
            "true_not_merged_fraction": float(frac_true_not),
            "mean_offline_count": float(off_count.mean()),
            "mean_hlt_count": float(hlt_count.mean()),
            "mean_count_gap": float(count_gap.mean()),
        },
        "best_strategy": best,
        "pt_bin_metrics_best": pt_bin_metrics,
        "abs_eta_bin_metrics_best": eta_bin_metrics,
    }
    with open(save_root / "seed_label_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plots.
    nn_valid = nn_dr[valid]
    dens_valid = density[valid]
    true_m_valid = true_merged[valid]
    true_n_valid = true_not[valid]
    _plot_nn_dr_hist(save_dir=save_root, nn_dr_valid=nn_valid, true_merged_valid=true_m_valid, true_not_valid=true_n_valid)
    _plot_density_hist(save_dir=save_root, density_valid=dens_valid, true_merged_valid=true_m_valid, true_not_valid=true_n_valid)
    _plot_top_scores(save_dir=save_root, rows_sorted=rows_sorted, top_k=20)
    _plot_best_confusion(save_dir=save_root, best=best)

    # Also save sampled arrays for optional quick notebook debugging.
    np.savez_compressed(
        save_root / "seed_eval_arrays_sample.npz",
        labels=labels.astype(np.int8),
        off_count=off_count.astype(np.int16),
        hlt_count=hlt_count.astype(np.int16),
        count_gap=count_gap.astype(np.int16),
        hlt_pt=hlt_pt.astype(np.float32),
        hlt_abs_eta=hlt_abs_eta.astype(np.float32),
        nn_dr=np.nan_to_num(nn_dr, posinf=9.9, neginf=9.9).astype(np.float32),
        density=density.astype(np.float32),
        valid=valid.astype(np.uint8),
        true_merged=true_merged.astype(np.uint8),
        true_not=true_not.astype(np.uint8),
        pred_merged_best=merged_seed_best.astype(np.uint8),
        pred_not_best=not_seed_best.astype(np.uint8),
    )

    print("\nSaved outputs:")
    print(f"  {save_root / 'seed_strategy_sweep.csv'}")
    print(f"  {save_root / 'seed_label_summary.json'}")
    print(f"  {save_root / 'truth_nn_dr_distribution.png'}")
    print(f"  {save_root / 'truth_density_distribution.png'}")
    print(f"  {save_root / 'seed_strategy_top_scores.png'}")
    print(f"  {save_root / 'best_strategy_confusion_labeled.png'}")
    print(f"  {save_root / 'seed_eval_arrays_sample.npz'}")


if __name__ == "__main__":
    main()
