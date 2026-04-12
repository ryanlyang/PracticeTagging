#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Atlas-style global val-selected greedy fusion using precomputed dev-pool scores.

Goal:
- Keep the same greedy objective used by specialization_atlas:
  choose additions by best validation FPR (then AUC tie-break),
  with val-derived threshold evaluated on test.
- Use a large precomputed dev pool (e.g. 1M) and expanded model set (e.g. 55).

Inputs:
- precomputed_scores_npz from precompute_hlt_joint55_devpool_scores.py
- precomputed_manifest_json for model_order and metadata

Outputs:
- atlas55_precomputed_global_metrics.csv
- atlas55_precomputed_greedy_curve.csv
- atlas55_precomputed_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import analyze_hlt_joint31_specialization_atlas as atlas


def _parse_csv_models(spec: str) -> List[str]:
    out: List[str] = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if tok:
            out.append(tok)
    return out


def _pick_best_rows(rows: List[Dict[str, object]], tpr: float) -> Dict[str, object] | None:
    cand = [r for r in rows if float(r.get("target_tpr", -1.0)) == float(tpr)]
    if not cand:
        return None
    cand = sorted(cand, key=lambda r: (float(r.get("fpr_test", np.inf)), -float(r.get("auc_test", -np.inf))))
    return cand[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Atlas-style val-selected greedy fusion on precomputed dev pool")
    ap.add_argument("--precomputed_scores_npz", type=str, required=True)
    ap.add_argument("--precomputed_manifest_json", type=str, required=True)
    ap.add_argument("--fusion_json", type=str, default="", help="Optional: only recorded in report metadata")
    ap.add_argument("--target_tprs", type=str, default="0.50,0.30")
    ap.add_argument("--anchor_model", type=str, default="joint_delta")
    ap.add_argument("--candidate_models", type=str, default="", help="Optional CSV model subset. Empty -> all models_order except anchor")
    ap.add_argument("--greedy_max_add", type=int, default=12)
    ap.add_argument("--greedy_w_step", type=float, default=0.01)
    ap.add_argument("--greedy_calibration", type=str, default="iso", choices=["raw", "iso", "platt"])
    ap.add_argument("--dev_pool_size", type=int, default=1000000, help="If <=0, use all available dev jets")
    ap.add_argument("--dev_pool_offset", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--report_json", type=str, default="")
    args = ap.parse_args()

    scores_npz = Path(args.precomputed_scores_npz).expanduser().resolve()
    manifest_json = Path(args.precomputed_manifest_json).expanduser().resolve()
    if not scores_npz.exists():
        raise FileNotFoundError(f"precomputed_scores_npz not found: {scores_npz}")
    if not manifest_json.exists():
        raise FileNotFoundError(f"precomputed_manifest_json not found: {manifest_json}")

    with manifest_json.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    model_order = list(manifest.get("model_order", []))
    if not model_order:
        raise KeyError(f"Manifest missing non-empty model_order: {manifest_json}")

    z = np.load(scores_npz, allow_pickle=False)
    y_dev = np.asarray(z["labels_dev"], dtype=np.float32)
    y_test = np.asarray(z["labels_test"], dtype=np.float32)
    scores_dev = np.asarray(z["scores_dev"], dtype=np.float64)
    scores_test = np.asarray(z["scores_test"], dtype=np.float64)

    if scores_dev.ndim != 2 or scores_test.ndim != 2:
        raise ValueError("scores_dev/scores_test must be rank-2 [n_models, n_samples]")
    if scores_dev.shape[0] != len(model_order) or scores_test.shape[0] != len(model_order):
        raise ValueError(
            f"Model axis mismatch: model_order={len(model_order)} "
            f"scores_dev={scores_dev.shape} scores_test={scores_test.shape}"
        )

    n_avail = int(y_dev.shape[0])
    pool_size = int(args.dev_pool_size)
    if pool_size <= 0 or pool_size > n_avail:
        pool_size = n_avail
    pool_offset = int(max(args.dev_pool_offset, 0))
    if pool_offset + pool_size > n_avail:
        raise ValueError(
            f"Requested dev_pool_offset+dev_pool_size={pool_offset}+{pool_size} exceeds available {n_avail}"
        )

    idx = np.arange(pool_offset, pool_offset + pool_size, dtype=np.int64)
    y_val = np.asarray(y_dev[idx], dtype=np.float32)

    full_scores_val: Dict[str, np.ndarray] = {
        m: np.asarray(scores_dev[i, idx], dtype=np.float64) for i, m in enumerate(model_order)
    }
    full_scores_test: Dict[str, np.ndarray] = {
        m: np.asarray(scores_test[i], dtype=np.float64) for i, m in enumerate(model_order)
    }

    if str(args.anchor_model) not in full_scores_val:
        raise KeyError(f"anchor_model={args.anchor_model} not in model_order ({len(model_order)} models)")

    tprs = atlas._parse_float_list(args.target_tprs, [0.50, 0.30])

    if str(args.candidate_models).strip():
        subset = _parse_csv_models(args.candidate_models)
        use_models: List[str] = []
        for m in model_order:
            if m == str(args.anchor_model) or m in subset:
                use_models.append(m)
        missing = [m for m in subset if m not in full_scores_val]
        if missing:
            raise KeyError(f"candidate_models contain unknown ids: {missing}")
    else:
        use_models = list(model_order)

    if str(args.anchor_model) not in use_models:
        use_models = [str(args.anchor_model)] + [m for m in use_models if m != str(args.anchor_model)]

    scores_val: Dict[str, np.ndarray] = {m: full_scores_val[m] for m in use_models}
    scores_test_map: Dict[str, np.ndarray] = {m: full_scores_test[m] for m in use_models}

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir).strip()
        else (scores_npz.parent / "atlas_iso_55_1m_precomputed")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = atlas._calc_thresholds_for_tprs(y_val, scores_val, use_models, tprs)
    global_rows = atlas._global_rows(
        y_val=y_val,
        y_test=y_test,
        scores_val=scores_val,
        scores_test=scores_test_map,
        model_order=use_models,
        thresholds=thresholds,
        anchor_model=str(args.anchor_model),
    )

    greedy_rows: List[Dict[str, object]] = []
    for tpr in tprs:
        greedy_rows.extend(
            atlas._greedy_subset_rows(
                y_val=y_val,
                y_test=y_test,
                scores_val=scores_val,
                scores_test=scores_test_map,
                model_order=use_models,
                anchor_model=str(args.anchor_model),
                tpr=float(tpr),
                max_add=int(args.greedy_max_add),
                w_step=float(args.greedy_w_step),
                calibration=str(args.greedy_calibration),
            )
        )

    atlas._save_csv_dynamic(out_dir / "atlas55_precomputed_global_metrics.csv", global_rows)
    atlas._save_csv_dynamic(out_dir / "atlas55_precomputed_greedy_curve.csv", greedy_rows)

    best_single: Dict[str, Dict[str, object]] = {}
    best_greedy: Dict[str, Dict[str, object]] = {}
    for t in tprs:
        t_key = f"{float(t):.3f}"
        rr_single = [r for r in global_rows if float(r.get("target_tpr", -1.0)) == float(t)]
        if rr_single:
            rr_single = sorted(rr_single, key=lambda r: (float(r["fpr_test"]), -float(r["auc_test"])))
            best_single[t_key] = rr_single[0]

        rr_g = [
            r
            for r in greedy_rows
            if float(r.get("target_tpr", -1.0)) == float(t)
            and str(r.get("action", "")) in {"init_anchor", "add_model"}
        ]
        if rr_g:
            rr_g = sorted(rr_g, key=lambda r: (float(r["fpr_test"]), -float(r["auc_test"])))
            best_greedy[t_key] = rr_g[0]

    report = {
        "scores_npz": str(scores_npz),
        "manifest_json": str(manifest_json),
        "fusion_json": str(args.fusion_json),
        "out_dir": str(out_dir),
        "n_models_total": int(len(model_order)),
        "n_models_used": int(len(use_models)),
        "models_used": use_models,
        "anchor_model": str(args.anchor_model),
        "target_tprs": [float(x) for x in tprs],
        "dev_pool": {
            "available": int(n_avail),
            "offset": int(pool_offset),
            "size": int(pool_size),
            "n_pos": int((y_val > 0.5).sum()),
            "n_neg": int((y_val <= 0.5).sum()),
        },
        "greedy": {
            "max_add": int(args.greedy_max_add),
            "w_step": float(args.greedy_w_step),
            "calibration": str(args.greedy_calibration),
        },
        "best_single_by_tpr": best_single,
        "best_greedy_by_tpr": best_greedy,
        "files": {
            "global_csv": str((out_dir / "atlas55_precomputed_global_metrics.csv").resolve()),
            "greedy_csv": str((out_dir / "atlas55_precomputed_greedy_curve.csv").resolve()),
        },
    }

    report_json = (
        Path(args.report_json).expanduser().resolve()
        if str(args.report_json).strip()
        else (out_dir / "atlas55_precomputed_report.json")
    )
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 72)
    print("Atlas-Style Global Greedy (Precomputed Dev Pool)")
    print("=" * 72)
    print(f"Scores npz: {scores_npz}")
    print(f"Manifest:   {manifest_json}")
    print(f"Out dir:    {out_dir}")
    print(f"Models:     used={len(use_models)} / total={len(model_order)}")
    print(f"Dev pool:   offset={pool_offset}, size={pool_size}")
    print(f"Anchor:     {args.anchor_model}")
    print(f"TPRs:       {','.join(f'{float(x):.3f}' for x in tprs)}")
    print(f"Greedy:     max_add={int(args.greedy_max_add)}, w_step={float(args.greedy_w_step):.4f}, cal={args.greedy_calibration}")

    for t in tprs:
        tk = f"{float(t):.3f}"
        b1 = best_single.get(tk)
        b2 = best_greedy.get(tk)
        if b1 is not None:
            print(
                f"Best single @TPR={float(t):.3f}: {b1.get('model')} "
                f"FPR_test={float(b1.get('fpr_test', np.nan)):.6f} AUC_test={float(b1.get('auc_test', np.nan)):.6f}"
            )
        if b2 is not None:
            print(
                f"Best greedy @TPR={float(t):.3f}: step={int(b2.get('step', -1))} "
                f"FPR_test={float(b2.get('fpr_test', np.nan)):.6f} AUC_test={float(b2.get('auc_test', np.nan)):.6f} "
                f"selected={b2.get('selected_models', '')}"
            )

    print(f"Saved report: {report_json}")
    print(f"Saved global metrics: {out_dir / 'atlas55_precomputed_global_metrics.csv'}")
    print(f"Saved greedy curve: {out_dir / 'atlas55_precomputed_greedy_curve.csv'}")


if __name__ == "__main__":
    main()
