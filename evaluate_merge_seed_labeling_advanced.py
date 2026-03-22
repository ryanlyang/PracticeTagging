#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced merged-seed strategy sweep.

Purpose:
- Improve merged-product seed quality beyond simple nearest-neighbor thresholding.
- Compare several strategy families on the same realistic HLT corruption setup:
  1) HLT-threshold strategies
  2) HLT top-k score strategies
  3) Offline-guided pair-centroid matching strategies

Evaluation truth (for analysis only):
- merged-product truth: origin_counts >= 2
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
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


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def _f1(p: float, r: float) -> float:
    return _safe_div(2.0 * p * r, p + r)


def compute_hlt_local_features(
    hlt_const: np.ndarray,
    hlt_mask: np.ndarray,
    density_radius: float,
) -> Dict[str, np.ndarray]:
    """
    Returns per-token features on HLT valid tokens:
      - nn_dr
      - density (neighbors within density_radius)
      - nn_pt_sim = exp(-|log(pt/pt_nn)|)
      - pt_low_score in [0,1] from per-jet pt rank (1=lower pt)
    """
    n_jets, max_part, _ = hlt_const.shape

    nn_dr = np.full((n_jets, max_part), np.inf, dtype=np.float32)
    density = np.zeros((n_jets, max_part), dtype=np.float32)
    nn_pt_sim = np.zeros((n_jets, max_part), dtype=np.float32)
    pt_low_score = np.zeros((n_jets, max_part), dtype=np.float32)

    for j in range(n_jets):
        valid = np.where(hlt_mask[j])[0]
        m = len(valid)
        if m == 0:
            continue

        pt_v = np.maximum(hlt_const[j, valid, 0], 1e-8)
        if m == 1:
            nn_dr[j, valid[0]] = np.inf
            density[j, valid[0]] = 0.0
            nn_pt_sim[j, valid[0]] = 0.0
            pt_low_score[j, valid[0]] = 1.0
            continue

        eta_v = hlt_const[j, valid, 1]
        phi_v = hlt_const[j, valid, 2]
        deta = eta_v[:, None] - eta_v[None, :]
        dphi = wrap_phi(phi_v[:, None] - phi_v[None, :])
        dr = np.sqrt(deta * deta + dphi * dphi).astype(np.float32)

        np.fill_diagonal(dr, np.inf)
        nn_idx = np.argmin(dr, axis=1)
        nn = dr[np.arange(m), nn_idx]

        near = (dr < float(density_radius)).astype(np.int32)
        np.fill_diagonal(near, 0)
        dens = near.sum(axis=1).astype(np.float32)

        nn_pt = pt_v[nn_idx]
        sim = np.exp(-np.abs(np.log(pt_v / np.maximum(nn_pt, 1e-8)))).astype(np.float32)

        # Low-pt score via percentile rank within jet.
        order = np.argsort(pt_v)
        rank = np.empty_like(order)
        rank[order] = np.arange(m)
        low_score = 1.0 - (rank.astype(np.float32) / max(float(m - 1), 1.0))

        nn_dr[j, valid] = nn
        density[j, valid] = dens
        nn_pt_sim[j, valid] = sim
        pt_low_score[j, valid] = low_score

    return {
        "nn_dr": nn_dr,
        "density": density,
        "nn_pt_sim": nn_pt_sim,
        "pt_low_score": pt_low_score,
    }


def compute_offline_pair_guidance(
    off_const: np.ndarray,
    off_mask: np.ndarray,
    hlt_const: np.ndarray,
    hlt_mask: np.ndarray,
    pair_dr_thresholds: List[float],
) -> Dict[float, Dict[str, np.ndarray]]:
    """
    For each requested offline pair dR threshold, compute per-HLT-token guidance:
      - min dR to nearest offline close-pair centroid
      - min |log(pt_hlt / pt_pair_centroid)| for that nearest centroid
    """
    thresholds = sorted(float(x) for x in pair_dr_thresholds)
    max_thr = float(max(thresholds))

    n_jets, max_part, _ = hlt_const.shape

    out = {}
    for t in thresholds:
        out[t] = {
            "pair_dr_min": np.full((n_jets, max_part), np.inf, dtype=np.float32),
            "pair_logpt_min": np.full((n_jets, max_part), np.inf, dtype=np.float32),
            "n_pairs": np.zeros(n_jets, dtype=np.int32),
        }

    for j in range(n_jets):
        off_idx = np.where(off_mask[j])[0]
        hlt_idx = np.where(hlt_mask[j])[0]
        if len(off_idx) < 2 or len(hlt_idx) == 0:
            continue

        pt = np.maximum(off_const[j, off_idx, 0], 1e-8)
        eta = off_const[j, off_idx, 1]
        phi = off_const[j, off_idx, 2]

        iu, ju = np.triu_indices(len(off_idx), k=1)
        if iu.size == 0:
            continue

        deta = eta[iu] - eta[ju]
        dphi = wrap_phi(phi[iu] - phi[ju])
        pair_dr = np.sqrt(deta * deta + dphi * dphi).astype(np.float32)

        use = pair_dr <= max_thr
        if not np.any(use):
            continue

        iu = iu[use]
        ju = ju[use]
        pair_dr = pair_dr[use]

        pti = pt[iu]
        ptj = pt[ju]
        pt_sum = np.maximum(pti + ptj, 1e-8)
        wi = pti / pt_sum
        wj = ptj / pt_sum

        c_eta = wi * eta[iu] + wj * eta[ju]
        c_phi = np.arctan2(
            wi * np.sin(phi[iu]) + wj * np.sin(phi[ju]),
            wi * np.cos(phi[iu]) + wj * np.cos(phi[ju]),
        ).astype(np.float32)
        c_pt = pt_sum.astype(np.float32)

        h_eta = hlt_const[j, hlt_idx, 1]
        h_phi = hlt_const[j, hlt_idx, 2]
        h_pt = np.maximum(hlt_const[j, hlt_idx, 0], 1e-8)

        for t in thresholds:
            sel = pair_dr <= t
            if not np.any(sel):
                continue

            et = c_eta[sel]
            ph = c_phi[sel]
            ptc = c_pt[sel]
            out[t]["n_pairs"][j] = int(et.shape[0])

            deta_hp = h_eta[:, None] - et[None, :]
            dphi_hp = wrap_phi(h_phi[:, None] - ph[None, :])
            dr_hp = np.sqrt(deta_hp * deta_hp + dphi_hp * dphi_hp).astype(np.float32)
            nn = np.argmin(dr_hp, axis=1)
            dr_min = dr_hp[np.arange(len(hlt_idx)), nn]
            ptc_nn = np.maximum(ptc[nn], 1e-8)
            logpt_min = np.abs(np.log(h_pt / ptc_nn)).astype(np.float32)

            out[t]["pair_dr_min"][j, hlt_idx] = dr_min
            out[t]["pair_logpt_min"][j, hlt_idx] = logpt_min

    return out


def apply_cap_per_jet(
    pred: np.ndarray,
    score: np.ndarray,
    count_gap: np.ndarray,
    cap_factor: float,
    valid_mask: np.ndarray,
) -> np.ndarray:
    if float(cap_factor) < 0.0:
        return pred

    out = np.zeros_like(pred, dtype=bool)
    n_jets = pred.shape[0]

    for j in range(n_jets):
        k = int(np.floor(float(cap_factor) * max(int(count_gap[j]), 0)))
        if k <= 0:
            continue

        cand = np.where(pred[j] & valid_mask[j])[0]
        if cand.size == 0:
            continue
        if cand.size <= k:
            out[j, cand] = True
            continue

        cs = score[j, cand]
        top = np.argpartition(cs, -k)[-k:]
        keep = cand[top]
        out[j, keep] = True

    return out


def evaluate_pred(
    pred_merged: np.ndarray,
    true_merged: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, float]:
    n_valid = int(valid_mask.sum())
    n_pred = int(pred_merged.sum())

    tp = int((pred_merged & true_merged).sum())
    fp = int((pred_merged & valid_mask & (~true_merged)).sum())
    fn = int((true_merged & (~pred_merged)).sum())

    precision = _safe_div(tp, n_pred)
    recall = _safe_div(tp, int(true_merged.sum()))
    f1 = _f1(precision, recall)
    coverage = _safe_div(n_pred, n_valid)

    return {
        "n_valid": n_valid,
        "n_pred": n_pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "coverage": coverage,
    }


def score_strategy(metrics: Dict[str, float], rank_by: str, min_recall: float) -> float:
    p = float(metrics["precision"])
    r = float(metrics["recall"])
    c = float(metrics["coverage"])
    f1 = float(metrics["f1"])

    if rank_by == "precision":
        # Favor high precision but softly penalize extremely tiny recall.
        gate = 1.0 if r >= float(min_recall) else (r / max(float(min_recall), 1e-6))
        return gate * (0.90 * p + 0.10 * c)
    if rank_by == "f1":
        return f1
    # balanced default
    return 0.55 * p + 0.30 * r + 0.15 * c


def build_threshold_strategies(fast: bool) -> List[Dict]:
    if fast:
        dr_max = [0.008, 0.011, 0.014]
        dens_min = [1, 2]
        pt_sim_min = [0.30, 0.45, 0.60]
        cap = [-1.0, 0.9, 1.0]
    else:
        dr_max = [0.006, 0.008, 0.010, 0.012, 0.014]
        dens_min = [1, 2, 3]
        pt_sim_min = [0.20, 0.35, 0.50, 0.65]
        cap = [-1.0, 0.7, 0.9, 1.0]

    out = []
    for a, b, c, d in itertools.product(dr_max, dens_min, pt_sim_min, cap):
        out.append(
            {
                "family": "hlt_threshold",
                "dr_max": float(a),
                "density_min": int(b),
                "pt_sim_min": float(c),
                "cap_factor": float(d),
            }
        )
    return out


def build_topk_strategies(fast: bool) -> List[Dict]:
    if fast:
        sigma = [0.006, 0.010]
        wdens = [0.05, 0.10]
        wsim = [0.20, 0.40]
        wlow = [0.00, 0.10]
        cap = [0.7, 0.9, 1.0]
    else:
        sigma = [0.005, 0.007, 0.010, 0.014]
        wdens = [0.03, 0.07, 0.12]
        wsim = [0.10, 0.25, 0.45]
        wlow = [0.00, 0.08, 0.15]
        cap = [0.6, 0.8, 0.9, 1.0, 1.2]

    out = []
    for s, wd, ws, wl, c in itertools.product(sigma, wdens, wsim, wlow, cap):
        out.append(
            {
                "family": "hlt_topk_score",
                "sigma": float(s),
                "w_density": float(wd),
                "w_ptsim": float(ws),
                "w_ptlow": float(wl),
                "cap_factor": float(c),
            }
        )
    return out


def build_offline_guided_strategies(fast: bool) -> List[Dict]:
    if fast:
        pair_thr = [0.010, 0.015]
        match_dr = [0.004, 0.007, 0.010]
        match_logpt = [0.20, 0.35, 0.50]
        dens_min = [0, 1]
        cap = [0.9, 1.0]
    else:
        pair_thr = [0.008, 0.010, 0.013, 0.016, 0.020]
        match_dr = [0.003, 0.005, 0.008, 0.012]
        match_logpt = [0.15, 0.25, 0.40, 0.60]
        dens_min = [0, 1, 2]
        cap = [0.8, 0.9, 1.0, 1.2]

    out = []
    for a, b, c, d, e in itertools.product(pair_thr, match_dr, match_logpt, dens_min, cap):
        out.append(
            {
                "family": "offline_pair_guided",
                "pair_dr_thr": float(a),
                "match_dr_max": float(b),
                "match_logpt_max": float(c),
                "density_min": int(d),
                "cap_factor": float(e),
            }
        )
    return out


def run_strategy(
    s: Dict,
    valid: np.ndarray,
    count_gap: np.ndarray,
    hlt_pt: np.ndarray,
    hlt_feat: Dict[str, np.ndarray],
    off_guidance: Dict[float, Dict[str, np.ndarray]] | None,
) -> np.ndarray:
    nn_dr = hlt_feat["nn_dr"]
    density = hlt_feat["density"]
    nn_pt_sim = hlt_feat["nn_pt_sim"]
    pt_low = hlt_feat["pt_low_score"]

    fam = s["family"]

    if fam == "hlt_threshold":
        pred = (
            valid
            & (nn_dr <= float(s["dr_max"]))
            & (density >= int(s["density_min"]))
            & (nn_pt_sim >= float(s["pt_sim_min"]))
        )
        score = (
            np.exp(-nn_dr / max(float(s["dr_max"]), 1e-6))
            + 0.10 * density
            + 0.30 * nn_pt_sim
        )
        pred = apply_cap_per_jet(pred, score, count_gap, float(s["cap_factor"]), valid)
        return pred

    if fam == "hlt_topk_score":
        sigma = max(float(s["sigma"]), 1e-6)
        score = (
            np.exp(-nn_dr / sigma)
            + float(s["w_density"]) * density
            + float(s["w_ptsim"]) * nn_pt_sim
            + float(s["w_ptlow"]) * pt_low
        )

        # candidate pool avoids selecting very isolated tokens by force
        cand = valid & np.isfinite(nn_dr) & (nn_dr <= 0.08)
        out = np.zeros_like(valid, dtype=bool)
        for j in range(valid.shape[0]):
            k = int(np.floor(float(s["cap_factor"]) * max(int(count_gap[j]), 0)))
            if k <= 0:
                continue
            idx = np.where(cand[j])[0]
            if idx.size == 0:
                continue
            if idx.size <= k:
                out[j, idx] = True
                continue
            sc = score[j, idx]
            top = np.argpartition(sc, -k)[-k:]
            out[j, idx[top]] = True
        return out

    if fam == "offline_pair_guided":
        if off_guidance is None:
            raise RuntimeError("offline guidance is required for offline_pair_guided strategies")
        key = float(s["pair_dr_thr"])
        g = off_guidance[key]
        drmin = g["pair_dr_min"]
        logpt = g["pair_logpt_min"]

        pred = (
            valid
            & (drmin <= float(s["match_dr_max"]))
            & (logpt <= float(s["match_logpt_max"]))
            & (density >= int(s["density_min"]))
        )

        score = np.exp(-drmin / max(float(s["match_dr_max"]), 1e-6)) + np.exp(
            -logpt / max(float(s["match_logpt_max"]), 1e-6)
        ) + 0.05 * density
        pred = apply_cap_per_jet(pred, score, count_gap, float(s["cap_factor"]), valid)
        return pred

    raise ValueError(f"Unknown family: {fam}")


def compute_bin_metrics(
    value: np.ndarray,
    bins: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    valid: np.ndarray,
) -> List[Dict]:
    rows = []
    for i in range(len(bins) - 1):
        lo = float(bins[i])
        hi = float(bins[i + 1])
        m = valid & (value >= lo) & (value < hi)
        n = int(m.sum())
        if n == 0:
            continue
        p = int(pred[m].sum())
        t = int(true[m].sum())
        tp = int((pred[m] & true[m]).sum())
        rows.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "n_tokens": n,
                "coverage": _safe_div(p, n),
                "precision": _safe_div(tp, p),
                "recall": _safe_div(tp, t),
            }
        )
    return rows


def plot_pr_scatter(save_root: Path, rows: List[Dict]) -> None:
    fams = sorted(set(r["family"] for r in rows))
    colors = {
        "hlt_threshold": "tab:blue",
        "hlt_topk_score": "tab:orange",
        "offline_pair_guided": "tab:green",
    }
    plt.figure(figsize=(7.5, 6.0))
    for fam in fams:
        rr = [r for r in rows if r["family"] == fam]
        if not rr:
            continue
        x = [float(r["recall"]) for r in rr]
        y = [float(r["precision"]) for r in rr]
        s = [40.0 + 280.0 * float(r["coverage"]) for r in rr]
        plt.scatter(x, y, s=s, alpha=0.55, label=fam, c=colors.get(fam, None))
    plt.xlabel("Merged recall")
    plt.ylabel("Merged precision")
    plt.title("Merged Seed Strategies: Precision vs Recall (size=coverage)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_root / "advanced_precision_recall_scatter.png", dpi=180)
    plt.close()


def plot_top_scores(save_root: Path, rows_sorted: List[Dict], top_k: int = 20) -> None:
    top = rows_sorted[:top_k]
    if not top:
        return
    labels = [f"{i+1}" for i in range(len(top))]
    vals = [float(r["strategy_score"]) for r in top]
    colors = []
    for r in top:
        if r["family"] == "hlt_threshold":
            colors.append("tab:blue")
        elif r["family"] == "hlt_topk_score":
            colors.append("tab:orange")
        else:
            colors.append("tab:green")
    plt.figure(figsize=(9.5, 4.8))
    plt.bar(np.arange(len(top)), vals, color=colors)
    plt.xticks(np.arange(len(top)), labels)
    plt.xlabel("Rank")
    plt.ylabel("Strategy score")
    plt.title("Top Advanced Strategies")
    plt.tight_layout()
    plt.savefig(save_root / "advanced_top_strategy_scores.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced merged-seed strategy sweep.")
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_jets", type=int, default=100000)
    parser.add_argument("--offset_jets", type=int, default=0)
    parser.add_argument("--max_constits", type=int, default=100)
    parser.add_argument("--seed", type=int, default=52)

    parser.add_argument("--save_dir", type=str, default="checkpoints/seed_labeling_eval")
    parser.add_argument("--run_name", type=str, default="seed_eval_advanced")

    parser.add_argument(
        "--families",
        type=str,
        default="hlt_threshold,hlt_topk_score,offline_pair_guided",
        help="Comma-separated family list.",
    )
    parser.add_argument("--fast_sweep", action="store_true")
    parser.add_argument("--rank_by", type=str, default="balanced", choices=["balanced", "precision", "f1"])
    parser.add_argument("--min_recall_for_precision", type=float, default=0.03)

    # Optional HLT config probes
    parser.add_argument("--merge_radius", type=float, default=None)
    parser.add_argument("--smear_scale", type=float, default=1.0)
    parser.add_argument("--eff_scale", type=float, default=1.0)
    args = parser.parse_args()

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    # --------------------------- Data load --------------------------- #
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
        raise RuntimeError(f"Need {need} jets but loaded {const_full.shape[0]}.")

    const_off = const_full[int(args.offset_jets):need]
    labels = labels_full[int(args.offset_jets):need].astype(np.int64)

    # ----------------------- HLT generation -------------------------- #
    cfg = _deepcopy_cfg()
    hcfg = cfg["hlt_effects"]

    if args.merge_radius is not None:
        hcfg["merge_radius"] = float(args.merge_radius)

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
    for k in ("eff_plateau_barrel", "eff_plateau_endcap", "eff_floor", "eff_ceil"):
        hcfg[k] = float(np.clip(float(hcfg[k]) * eff_scale, 0.0, 0.999))

    raw_mask = const_off[:, :, 0] > 0.0
    off_mask = raw_mask & (const_off[:, :, 0] >= float(hcfg["pt_threshold_offline"]))

    print("Generating realistic pseudo-HLT with tracking...")
    hlt_const, hlt_mask, origin_counts, _, hlt_stats = apply_hlt_effects_realistic_with_tracking(
        const=const_off,
        mask=off_mask,
        cfg=cfg,
        seed=int(args.seed),
    )

    valid = hlt_mask.astype(bool)
    true_merged = valid & (origin_counts >= 2)

    off_count = off_mask.sum(axis=1).astype(np.int32)
    hlt_count = hlt_mask.sum(axis=1).astype(np.int32)
    count_gap = (off_count - hlt_count).astype(np.int32)

    n_valid = int(valid.sum())
    n_true_merged = int(true_merged.sum())
    print(f"Valid HLT tokens: {n_valid:,}")
    print(f"True merged-product tokens: {n_true_merged:,} ({100.0*_safe_div(n_true_merged, n_valid):.2f}%)")

    # -------------------------- Features ----------------------------- #
    print("Computing HLT local features...")
    hlt_feat = compute_hlt_local_features(
        hlt_const=hlt_const,
        hlt_mask=hlt_mask,
        density_radius=float(hcfg["density_radius"]),
    )
    hlt_pt = np.maximum(hlt_const[:, :, 0], 1e-8).astype(np.float32)
    hlt_abs_eta = np.abs(hlt_const[:, :, 1]).astype(np.float32)

    families = [x.strip() for x in str(args.families).split(",") if x.strip()]

    strategies: List[Dict] = []
    if "hlt_threshold" in families:
        strategies.extend(build_threshold_strategies(args.fast_sweep))
    if "hlt_topk_score" in families:
        strategies.extend(build_topk_strategies(args.fast_sweep))

    off_guidance = None
    if "offline_pair_guided" in families:
        off_strats = build_offline_guided_strategies(args.fast_sweep)
        strategies.extend(off_strats)
        pair_thresholds = sorted(set(float(s["pair_dr_thr"]) for s in off_strats))
        print(f"Computing offline-pair guidance for thresholds: {pair_thresholds}")
        off_guidance = compute_offline_pair_guidance(
            off_const=const_off,
            off_mask=off_mask,
            hlt_const=hlt_const,
            hlt_mask=hlt_mask,
            pair_dr_thresholds=pair_thresholds,
        )

    print(f"Sweeping {len(strategies):,} advanced strategies...")

    rows: List[Dict] = []
    for sid, s in enumerate(strategies):
        pred = run_strategy(
            s=s,
            valid=valid,
            count_gap=count_gap,
            hlt_pt=hlt_pt,
            hlt_feat=hlt_feat,
            off_guidance=off_guidance,
        )
        m = evaluate_pred(pred_merged=pred, true_merged=true_merged, valid_mask=valid)
        sc = score_strategy(m, rank_by=args.rank_by, min_recall=float(args.min_recall_for_precision))

        row = {
            "strategy_id": int(sid),
            "family": s["family"],
            "strategy_score": float(sc),
            "precision": float(m["precision"]),
            "recall": float(m["recall"]),
            "f1": float(m["f1"]),
            "coverage": float(m["coverage"]),
            "n_pred": int(m["n_pred"]),
            "tp": int(m["tp"]),
            "fp": int(m["fp"]),
            "fn": int(m["fn"]),
            # Generic param slots
            "dr_max": float(s.get("dr_max", np.nan)),
            "density_min": float(s.get("density_min", np.nan)),
            "pt_sim_min": float(s.get("pt_sim_min", np.nan)),
            "cap_factor": float(s.get("cap_factor", np.nan)),
            "sigma": float(s.get("sigma", np.nan)),
            "w_density": float(s.get("w_density", np.nan)),
            "w_ptsim": float(s.get("w_ptsim", np.nan)),
            "w_ptlow": float(s.get("w_ptlow", np.nan)),
            "pair_dr_thr": float(s.get("pair_dr_thr", np.nan)),
            "match_dr_max": float(s.get("match_dr_max", np.nan)),
            "match_logpt_max": float(s.get("match_logpt_max", np.nan)),
        }
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: float(r["strategy_score"]), reverse=True)
    best = rows_sorted[0]

    # Best per family
    best_by_family = {}
    for fam in sorted(set(r["family"] for r in rows)):
        fam_rows = [r for r in rows_sorted if r["family"] == fam]
        if fam_rows:
            best_by_family[fam] = fam_rows[0]

    print("\nTop strategies:")
    for k, r in enumerate(rows_sorted[:12], 1):
        print(
            f"  #{k:02d} fam={r['family']} score={r['strategy_score']:.4f} "
            f"P/R/F1=({r['precision']:.3f}/{r['recall']:.3f}/{r['f1']:.3f}) "
            f"cov={100.0*r['coverage']:.2f}%"
        )

    # Save CSV
    csv_path = save_root / "advanced_seed_strategy_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        w.writeheader()
        w.writerows(rows_sorted)

    # Re-run best strategy to make binned diagnostics
    sid_to_strategy = {int(sid): s for sid, s in enumerate(strategies)}
    best_s = sid_to_strategy[int(best["strategy_id"])]
    best_pred = run_strategy(
        s=best_s,
        valid=valid,
        count_gap=count_gap,
        hlt_pt=hlt_pt,
        hlt_feat=hlt_feat,
        off_guidance=off_guidance,
    )

    pt_bins = np.percentile(hlt_pt[valid], [0, 10, 25, 40, 55, 70, 85, 95, 100])
    eta_bins = np.array([0.0, 0.4, 0.8, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0], dtype=np.float32)

    pt_bin = compute_bin_metrics(
        value=hlt_pt,
        bins=np.unique(pt_bins),
        pred=best_pred,
        true=true_merged,
        valid=valid,
    )
    eta_bin = compute_bin_metrics(
        value=hlt_abs_eta,
        bins=np.unique(eta_bins),
        pred=best_pred,
        true=true_merged,
        valid=valid,
    )

    summary = {
        "seed": int(args.seed),
        "n_jets": int(args.n_jets),
        "offset_jets": int(args.offset_jets),
        "max_constits": int(args.max_constits),
        "families": families,
        "rank_by": args.rank_by,
        "min_recall_for_precision": float(args.min_recall_for_precision),
        "hlt_config": hcfg,
        "hlt_stats": hlt_stats,
        "global": {
            "valid_tokens": int(n_valid),
            "true_merged_tokens": int(n_true_merged),
            "true_merged_fraction": _safe_div(n_true_merged, n_valid),
            "mean_offline_count": float(off_count.mean()),
            "mean_hlt_count": float(hlt_count.mean()),
            "mean_count_gap": float(count_gap.mean()),
        },
        "best_overall": best,
        "best_by_family": best_by_family,
        "pt_bin_metrics_best": pt_bin,
        "abs_eta_bin_metrics_best": eta_bin,
    }

    with open(save_root / "advanced_seed_label_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_pr_scatter(save_root, rows)
    plot_top_scores(save_root, rows_sorted, top_k=20)

    np.savez_compressed(
        save_root / "advanced_seed_eval_arrays.npz",
        labels=labels.astype(np.int8),
        off_count=off_count.astype(np.int16),
        hlt_count=hlt_count.astype(np.int16),
        count_gap=count_gap.astype(np.int16),
        valid=valid.astype(np.uint8),
        true_merged=true_merged.astype(np.uint8),
        pred_best=best_pred.astype(np.uint8),
        hlt_pt=hlt_pt.astype(np.float32),
        hlt_abs_eta=hlt_abs_eta.astype(np.float32),
        nn_dr=np.nan_to_num(hlt_feat["nn_dr"], posinf=9.9, neginf=9.9).astype(np.float32),
        density=hlt_feat["density"].astype(np.float32),
        nn_pt_sim=hlt_feat["nn_pt_sim"].astype(np.float32),
    )

    print("\nSaved outputs:")
    print(f"  {csv_path}")
    print(f"  {save_root / 'advanced_seed_label_summary.json'}")
    print(f"  {save_root / 'advanced_precision_recall_scatter.png'}")
    print(f"  {save_root / 'advanced_top_strategy_scores.png'}")
    print(f"  {save_root / 'advanced_seed_eval_arrays.npz'}")


if __name__ == "__main__":
    main()
