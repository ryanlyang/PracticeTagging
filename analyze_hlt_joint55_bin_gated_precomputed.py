#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bin-gated val-selected fusion on precomputed dev-pool scores for large model sets
(e.g. 55 models, 1M dev/val pool).

This mirrors the analyze31 bin-gated flow, but uses precomputed [n_models, n_samples]
score tensors so we do not need per-model run-dir key mappings.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

import analyze_hlt_joint31_bin_gated_fusion as bg


def _parse_float_list(spec: str, default: List[float]) -> List[float]:
    out: List[float] = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except Exception:
            continue
    if not out:
        return [float(x) for x in default]
    return out


def _parse_csv_models(spec: str) -> List[str]:
    out: List[str] = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if tok:
            out.append(tok)
    return out


def _w_grid_count(w_step: float) -> int:
    arr = np.arange(float(w_step), 1.0, float(w_step), dtype=np.float64)
    return int(arr.size)


def _format_sec(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _estimate_eval_counts(n_candidates: int, n_bins: int, n_tprs: int, global_max_add: int, bin_max_add: int, w_step: float) -> Dict[str, int]:
    n_w = max(1, _w_grid_count(w_step))
    n_c = max(1, int(n_candidates))
    n_b = max(1, int(n_bins))
    n_t = max(1, int(n_tprs))
    global_per_tpr = int(max(0, int(global_max_add)) * max(0, n_c - 1) * n_w)
    bin_per_tpr = int(max(0, int(bin_max_add)) * n_b * n_c * n_w)
    return {
        "w_grid": int(n_w),
        "global_per_tpr": int(global_per_tpr),
        "bin_per_tpr": int(bin_per_tpr),
        "total_all_tprs": int((global_per_tpr + bin_per_tpr) * n_t),
    }


def _pick_best_rows(summary_rows: List[Dict[str, object]], tpr: float) -> List[Dict[str, object]]:
    cand = [r for r in summary_rows if float(r.get("target_tpr", -1.0)) == float(tpr)]
    return sorted(cand, key=lambda r: (float(r.get("fpr_test", np.inf)), -float(r.get("auc_test", -np.inf))))


def main() -> None:
    ap = argparse.ArgumentParser(description="55-model bin-gated fusion on precomputed dev-pool scores")
    ap.add_argument("--precomputed_scores_npz", type=str, required=True)
    ap.add_argument("--precomputed_manifest_json", type=str, required=True)
    ap.add_argument("--fusion_json", type=str, default="", help="Optional metadata only")
    ap.add_argument("--target_tprs", type=str, default="0.50,0.30")
    ap.add_argument("--anchor_model", type=str, default="joint_delta")
    ap.add_argument("--candidate_models_all", type=str, default="", help="Optional CSV subset; empty means all models in manifest")
    ap.add_argument("--selection_mode", type=str, default="valsel", choices=["split", "valsel"])
    ap.add_argument("--router_cal_frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibration", type=str, default="iso", choices=["raw", "iso", "platt"])
    ap.add_argument("--score_band_edges", type=str, default="0.0,0.8,0.9,1.0")
    ap.add_argument("--dist_near_cut", type=float, default=0.0384)
    ap.add_argument("--dist_mid_low", type=float, default=0.06285)
    ap.add_argument("--dist_mid_high", type=float, default=0.07386)
    ap.add_argument("--global_max_add", type=int, default=6)
    ap.add_argument("--bin_max_add", type=int, default=3)
    ap.add_argument("--w_step", type=float, default=0.01)
    ap.add_argument("--min_bin_fit", type=int, default=1200)
    ap.add_argument("--min_global_improve", type=float, default=1e-6)
    ap.add_argument("--min_bin_improve", type=float, default=5e-6)
    ap.add_argument("--dev_pool_size", type=int, default=1000000, help="If <=0 use all available dev jets")
    ap.add_argument("--dev_pool_offset", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--report_json", type=str, default="")
    args = ap.parse_args()

    t0 = time.time()

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
    if str(args.anchor_model) not in model_order:
        raise KeyError(f"anchor_model={args.anchor_model} not in model_order")
    if "hlt" not in model_order:
        raise KeyError("Manifest/model_order missing required baseline model id: hlt")

    z = np.load(scores_npz, allow_pickle=False)
    y_dev = np.asarray(z["labels_dev"], dtype=np.float32)
    y_test = np.asarray(z["labels_test"], dtype=np.float32)
    scores_dev = np.asarray(z["scores_dev"], dtype=np.float64)
    scores_test = np.asarray(z["scores_test"], dtype=np.float64)
    if scores_dev.ndim != 2 or scores_test.ndim != 2:
        raise ValueError("scores_dev/scores_test must be [n_models, n_samples]")
    if scores_dev.shape[0] != len(model_order) or scores_test.shape[0] != len(model_order):
        raise ValueError(
            f"Model axis mismatch: model_order={len(model_order)} "
            f"scores_dev={scores_dev.shape} scores_test={scores_test.shape}"
        )
    if scores_dev.shape[1] != y_dev.shape[0]:
        raise ValueError(f"Dev scores/labels mismatch: {scores_dev.shape[1]} vs {y_dev.shape[0]}")
    if scores_test.shape[1] != y_test.shape[0]:
        raise ValueError(f"Test scores/labels mismatch: {scores_test.shape[1]} vs {y_test.shape[0]}")

    n_avail = int(y_dev.shape[0])
    pool_size = int(args.dev_pool_size)
    if pool_size <= 0 or pool_size > n_avail:
        pool_size = n_avail
    pool_offset = int(max(0, int(args.dev_pool_offset)))
    if pool_offset + pool_size > n_avail:
        raise ValueError(
            f"Requested dev_pool_offset+dev_pool_size={pool_offset}+{pool_size} exceeds available {n_avail}"
        )
    idx = np.arange(pool_offset, pool_offset + pool_size, dtype=np.int64)

    y_val = np.asarray(y_dev[idx], dtype=np.float32)
    y_te = np.asarray(y_test, dtype=np.float32)

    scores_val_all: Dict[str, np.ndarray] = {m: np.asarray(scores_dev[i, idx], dtype=np.float64) for i, m in enumerate(model_order)}
    scores_test_all: Dict[str, np.ndarray] = {m: np.asarray(scores_test[i], dtype=np.float64) for i, m in enumerate(model_order)}

    subset = _parse_csv_models(args.candidate_models_all)
    if subset:
        subset_set = set(subset)
        use_models = [m for m in model_order if (m in subset_set) or (m == str(args.anchor_model)) or (m == "hlt")]
        missing = [m for m in subset if m not in scores_val_all]
        if missing:
            raise KeyError(f"candidate_models_all contains unknown ids: {missing}")
    else:
        use_models = list(model_order)

    if str(args.anchor_model) not in use_models:
        use_models.insert(0, str(args.anchor_model))
    if "hlt" not in use_models:
        use_models.insert(0, "hlt")
    # Stable unique
    seen_models = set()
    use_models = [m for m in use_models if not (m in seen_models or seen_models.add(m))]

    target_tprs = [float(x) for x in _parse_float_list(args.target_tprs, [0.50, 0.30])]
    score_edges = _parse_float_list(args.score_band_edges, [0.0, 0.8, 0.9, 1.0])
    if len(score_edges) < 2:
        raise ValueError("score_band_edges must have at least two values")
    if not np.all(np.diff(np.asarray(score_edges, dtype=np.float64)) > 0.0):
        raise ValueError("score_band_edges must be strictly increasing")

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir).strip()
        else (scores_npz.parent / "bin_gated_fusion_55_valsel_1m")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_all = np.arange(y_val.shape[0], dtype=np.int64)
    yv_int = (y_val > 0.5).astype(np.int64)
    if str(args.selection_mode).lower() == "valsel":
        idx_fit = idx_all.copy()
        idx_ref = idx_all.copy()
    else:
        idx_fit, idx_ref = train_test_split(
            idx_all,
            test_size=float(np.clip(args.router_cal_frac, 0.1, 0.8)),
            random_state=int(args.seed),
            stratify=yv_int,
        )
    y_fit = y_val[idx_fit].astype(np.float32)
    y_ref = y_val[idx_ref].astype(np.float32)

    n_bins = int((len(score_edges) - 1) * 3)
    est = _estimate_eval_counts(
        n_candidates=len(use_models),
        n_bins=n_bins,
        n_tprs=len(target_tprs),
        global_max_add=int(args.global_max_add),
        bin_max_add=int(args.bin_max_add),
        w_step=float(args.w_step),
    )

    print("=" * 88)
    print("55-Model Bin-Gated Fusion (Precomputed Dev Pool)")
    print("=" * 88)
    print(f"Scores npz:       {scores_npz}")
    print(f"Manifest json:    {manifest_json}")
    if str(args.fusion_json).strip():
        print(f"Fusion json(meta):{Path(args.fusion_json).expanduser().resolve()}")
    print(f"Out dir:          {out_dir}")
    print(f"Models used:      {len(use_models)} / {len(model_order)}")
    print(f"Anchor:           {args.anchor_model}")
    print(f"TPRs:             {','.join(f'{x:.3f}' for x in target_tprs)}")
    print(f"Selection:        {args.selection_mode} (fit={len(idx_fit)} ref={len(idx_ref)})")
    print(f"Calibration:      {args.calibration}")
    print(f"Dev pool:         offset={pool_offset} size={pool_size} avail={n_avail}")
    print(f"Greedy settings:  global_max_add={args.global_max_add} bin_max_add={args.bin_max_add} w_step={args.w_step}")
    print(
        "Rough eval budget:"
        f" w_grid={est['w_grid']} global/tpr≈{est['global_per_tpr']:,}"
        f" bin/tpr≈{est['bin_per_tpr']:,} total≈{est['total_all_tprs']:,}"
    )
    print("=" * 88)

    update_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    score_dump: Dict[str, np.ndarray] = {}
    report_tpr: Dict[str, object] = {}
    timing_by_tpr: Dict[str, Dict[str, float]] = {}

    for tpr in target_tprs:
        tpr_start = time.time()
        cands = list(use_models)
        if str(args.anchor_model) not in cands:
            cands.insert(0, str(args.anchor_model))
        # stable unique
        seen = set()
        cands = [m for m in cands if not (m in seen or seen.add(m))]

        print(f"\n>>> TPR={tpr:.3f} | candidates={len(cands)} | elapsed={_format_sec(time.time() - t0)}")
        print("    Building raw fit/ref/test maps...")
        fit_map_raw = {m: np.asarray(scores_val_all[m][idx_fit], dtype=np.float64) for m in cands}
        ref_map_raw = {m: np.asarray(scores_val_all[m][idx_ref], dtype=np.float64) for m in cands}
        test_map_raw = {m: np.asarray(scores_test_all[m], dtype=np.float64) for m in cands}

        print("    Calibrating candidate score streams...")
        cal_start = time.time()
        fit_map: Dict[str, np.ndarray] = {}
        cal_map: Dict[str, np.ndarray] = {}
        test_map: Dict[str, np.ndarray] = {}
        for i, m in enumerate(cands, start=1):
            sf, sr, st = bg.calibrate_scores(
                y_fit=y_fit,
                s_fit=fit_map_raw[m],
                s_cal=ref_map_raw[m],
                s_test=test_map_raw[m],
                mode=str(args.calibration),
            )
            fit_map[m] = sf
            cal_map[m] = sr
            test_map[m] = st
            if (i % 8 == 0) or (i == len(cands)):
                print(f"      calibrated {i:>2}/{len(cands)} in {_format_sec(time.time() - cal_start)}")

        print("    Stage 1/2: global greedy blend...")
        g0 = time.time()
        s_fit_g, s_cal_g, s_test_g, rows_g = bg._greedy_global_blend(
            y_fit=y_fit,
            s_fit_map=fit_map,
            s_cal_map=cal_map,
            s_test_map=test_map,
            anchor_model=str(args.anchor_model),
            candidates=cands,
            target_tpr=float(tpr),
            max_add=int(args.global_max_add),
            w_step=float(args.w_step),
            min_improve=float(args.min_global_improve),
        )
        for r in rows_g:
            rr = dict(r)
            rr["target_tpr"] = float(tpr)
            update_rows.append(rr)
        print(f"      done global in {_format_sec(time.time() - g0)}")

        print("    Stage 2/2: bin-local updates...")
        thr_anchor_fit = bg.threshold_for_target_tpr(y_fit, fit_map_raw[str(args.anchor_model)], float(tpr))
        dist_fit = np.abs(fit_map_raw[str(args.anchor_model)] - float(thr_anchor_fit))
        dist_cal = np.abs(ref_map_raw[str(args.anchor_model)] - float(thr_anchor_fit))
        dist_test = np.abs(test_map_raw[str(args.anchor_model)] - float(thr_anchor_fit))
        bin_fit = bg._make_bin_ids(
            joint_score=fit_map_raw[str(args.anchor_model)],
            dist_to_joint_thr=dist_fit,
            score_edges=score_edges,
            near_cut=float(args.dist_near_cut),
            mid_lo=float(args.dist_mid_low),
            mid_hi=float(args.dist_mid_high),
        )
        bin_cal = bg._make_bin_ids(
            joint_score=ref_map_raw[str(args.anchor_model)],
            dist_to_joint_thr=dist_cal,
            score_edges=score_edges,
            near_cut=float(args.dist_near_cut),
            mid_lo=float(args.dist_mid_low),
            mid_hi=float(args.dist_mid_high),
        )
        bin_test = bg._make_bin_ids(
            joint_score=test_map_raw[str(args.anchor_model)],
            dist_to_joint_thr=dist_test,
            score_edges=score_edges,
            near_cut=float(args.dist_near_cut),
            mid_lo=float(args.dist_mid_low),
            mid_hi=float(args.dist_mid_high),
        )
        b0 = time.time()
        s_fit_b, s_cal_b, s_test_b, rows_b = bg._binwise_updates(
            y_fit=y_fit,
            s_fit_init=s_fit_g,
            s_cal_init=s_cal_g,
            s_test_init=s_test_g,
            s_fit_map=fit_map,
            s_cal_map=cal_map,
            s_test_map=test_map,
            candidates=cands,
            target_tpr=float(tpr),
            bin_fit=bin_fit,
            bin_cal=bin_cal,
            bin_test=bin_test,
            score_edges=score_edges,
            min_bin_fit=int(args.min_bin_fit),
            bin_max_add=int(args.bin_max_add),
            w_step=float(args.w_step),
            min_bin_improve=float(args.min_bin_improve),
        )
        for r in rows_b:
            rr = dict(r)
            rr["target_tpr"] = float(tpr)
            update_rows.append(rr)
        print(f"      done bin-local in {_format_sec(time.time() - b0)}")

        k = f"tpr{tpr:.3f}".replace(".", "p")
        score_dump[f"fused_global_fit_{k}"] = s_fit_g.astype(np.float32)
        score_dump[f"fused_global_cal_{k}"] = s_cal_g.astype(np.float32)
        score_dump[f"fused_global_test_{k}"] = s_test_g.astype(np.float32)
        score_dump[f"fused_bin_fit_{k}"] = s_fit_b.astype(np.float32)
        score_dump[f"fused_bin_cal_{k}"] = s_cal_b.astype(np.float32)
        score_dump[f"fused_bin_test_{k}"] = s_test_b.astype(np.float32)

        methods = {
            "anchor": (ref_map_raw[str(args.anchor_model)], test_map_raw[str(args.anchor_model)]),
            "global_blend": (s_cal_g, s_test_g),
            "bin_gated_blend": (s_cal_b, s_test_b),
            "hlt": (scores_val_all["hlt"][idx_ref], scores_test_all["hlt"]),
        }
        if "teacher" in scores_val_all and "teacher" in scores_test_all:
            methods["teacher"] = (scores_val_all["teacher"][idx_ref], scores_test_all["teacher"])

        tpr_report = {"target_tpr": float(tpr), "candidates": cands, "metrics": {}}
        for mname, (sc, st) in methods.items():
            ev = bg.eval_from_ref(
                y_ref=y_ref,
                s_ref=np.asarray(sc, dtype=np.float64),
                y_eval=y_te,
                s_eval=np.asarray(st, dtype=np.float64),
                target_tpr=float(tpr),
            )
            row = {
                "target_tpr": float(tpr),
                "method": mname,
                "auc_cal": float(ev["auc_ref"]),
                "auc_test": float(ev["auc_eval"]),
                "fpr_cal": float(ev["fpr_ref"]),
                "fpr_test": float(ev["fpr_eval"]),
                "tpr_cal": float(ev["tpr_ref"]),
                "tpr_test": float(ev["tpr_eval"]),
                "threshold_from_ref": float(ev["threshold_from_ref"]),
            }
            summary_rows.append(row)
            tpr_report["metrics"][mname] = row

        bin_rows = []
        for b in sorted(int(x) for x in np.unique(bin_fit)):
            bin_rows.append(
                {
                    "bin_id": int(b),
                    "label": bg._bin_label(int(b), score_edges),
                    "n_fit": int((bin_fit == b).sum()),
                    "n_cal": int((bin_cal == b).sum()),
                    "n_test": int((bin_test == b).sum()),
                }
            )
        tpr_report["bins"] = bin_rows
        report_tpr[f"{tpr:.4f}"] = tpr_report
        timing_by_tpr[f"{tpr:.4f}"] = {
            "elapsed_sec": float(time.time() - tpr_start),
        }
        print(f"    Finished TPR={tpr:.3f} in {_format_sec(time.time() - tpr_start)}")

    bg._save_csv_dynamic(out_dir / "atlas55_bingated_summary.csv", summary_rows)
    bg._save_csv_dynamic(out_dir / "atlas55_bingated_update_log.csv", update_rows)
    np.savez_compressed(
        out_dir / "atlas55_bingated_scores.npz",
        labels_fit=y_fit.astype(np.float32),
        labels_ref=y_ref.astype(np.float32),
        labels_test=y_te.astype(np.float32),
        **score_dump,
    )

    report = {
        "precomputed_scores_npz": str(scores_npz),
        "precomputed_manifest_json": str(manifest_json),
        "fusion_json": str(args.fusion_json),
        "out_dir": str(out_dir),
        "selection_mode": str(args.selection_mode),
        "calibration": str(args.calibration),
        "target_tprs": target_tprs,
        "models_total": int(len(model_order)),
        "models_used": int(len(use_models)),
        "models_used_list": use_models,
        "anchor_model": str(args.anchor_model),
        "dev_pool": {
            "available": int(n_avail),
            "offset": int(pool_offset),
            "size": int(pool_size),
            "n_pos": int((y_val > 0.5).sum()),
            "n_neg": int((y_val <= 0.5).sum()),
            "fit_size": int(len(idx_fit)),
            "ref_size": int(len(idx_ref)),
        },
        "settings": {
            "score_band_edges": score_edges,
            "dist_near_cut": float(args.dist_near_cut),
            "dist_mid_low": float(args.dist_mid_low),
            "dist_mid_high": float(args.dist_mid_high),
            "global_max_add": int(args.global_max_add),
            "bin_max_add": int(args.bin_max_add),
            "w_step": float(args.w_step),
            "min_bin_fit": int(args.min_bin_fit),
            "min_global_improve": float(args.min_global_improve),
            "min_bin_improve": float(args.min_bin_improve),
            "seed": int(args.seed),
            "router_cal_frac": float(args.router_cal_frac),
        },
        "rough_eval_budget": est,
        "timing_by_tpr": timing_by_tpr,
        "by_tpr": report_tpr,
        "files": {
            "summary_csv": str((out_dir / "atlas55_bingated_summary.csv").resolve()),
            "updates_csv": str((out_dir / "atlas55_bingated_update_log.csv").resolve()),
            "scores_npz": str((out_dir / "atlas55_bingated_scores.npz").resolve()),
        },
    }

    report_json = (
        Path(args.report_json).expanduser().resolve()
        if str(args.report_json).strip()
        else (out_dir / "atlas55_bingated_report.json")
    )
    report_json.parent.mkdir(parents=True, exist_ok=True)
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 88)
    print("55-Model Bin-Gated Fusion (Precomputed) complete")
    print("=" * 88)
    print(f"Total elapsed: {_format_sec(time.time() - t0)}")
    for tpr in target_tprs:
        key = f"{tpr:.4f}"
        ranked = _pick_best_rows(summary_rows, tpr)
        print(f"\nTPR={tpr:.3f} best methods:")
        for r in ranked[:5]:
            print(
                f"  {str(r['method']).ljust(16)} AUC_test={float(r['auc_test']):.6f} "
                f"FPR_test={float(r['fpr_test']):.6f} (cal={float(r['fpr_cal']):.6f})"
            )
        tsec = timing_by_tpr.get(key, {}).get("elapsed_sec", float("nan"))
        if np.isfinite(tsec):
            print(f"  [timing] elapsed={_format_sec(float(tsec))}")
    print(f"\nSaved report: {report_json}")
    print(f"Saved summary: {out_dir / 'atlas55_bingated_summary.csv'}")
    print(f"Saved updates: {out_dir / 'atlas55_bingated_update_log.csv'}")
    print(f"Saved scores: {out_dir / 'atlas55_bingated_scores.npz'}")


if __name__ == "__main__":
    main()

