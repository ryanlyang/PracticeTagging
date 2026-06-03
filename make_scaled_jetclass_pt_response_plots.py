#!/usr/bin/env python3
"""Make diagnostic pT-response plots after applying a global scale to best reco.

This is a diagnostic/calibration visualization, not an uncalibrated model result.
It reads arrays from analyze_jetclass_twelve_model_jet_response.py output and
writes a separate scaled_response_diagnostic folder.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'user')}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _build_edges(pt_truth: np.ndarray, n_bins: int) -> np.ndarray:
    valid = np.isfinite(pt_truth) & (pt_truth > 1e-8)
    pt = pt_truth[valid]
    if pt.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)
    edges = np.unique(np.quantile(pt, np.linspace(0.0, 1.0, int(max(n_bins, 1)) + 1)))
    if edges.size < 2:
        c = float(np.median(pt))
        edges = np.array([max(c * 0.9, 0.0), c * 1.1 + 1e-6], dtype=np.float64)
    return edges.astype(np.float64)


def _records(pt_truth: np.ndarray, pt_reco: np.ndarray, edges: np.ndarray, min_count: int) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    valid = np.isfinite(pt_truth) & np.isfinite(pt_reco) & (pt_truth > 1e-8)
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        m = valid & (pt_truth >= lo)
        m = m & (pt_truth < hi if i < len(edges) - 2 else pt_truth <= hi)
        if int(m.sum()) < int(min_count):
            continue
        ratio = pt_reco[m] / pt_truth[m]
        ratio = ratio[np.isfinite(ratio)]
        if ratio.size == 0:
            continue
        out.append({
            "pt_low": lo,
            "pt_high": hi,
            "pt_center": 0.5 * (lo + hi),
            "count": int(ratio.size),
            "response": float(np.mean(ratio)),
            "resolution": float(np.std(ratio)),
        })
    return out


def _score(records: Sequence[Dict[str, float]]) -> float:
    if not records:
        return float("inf")
    counts = np.asarray([max(float(r["count"]), 1.0) for r in records])
    terms = np.asarray([abs(float(r["response"]) - 1.0) + float(r["resolution"]) for r in records])
    return float(np.average(terms, weights=counts))


def _plot(hlt_records, reco_records, label, title, path: Path) -> None:
    def arr(records, key):
        return np.asarray([float(r[key]) for r in records], dtype=np.float64)
    plt.figure(figsize=(10, 4.2))
    plt.suptitle(title)
    plt.subplot(1, 2, 1)
    if hlt_records:
        plt.plot(arr(hlt_records, "pt_center"), arr(hlt_records, "response"), "o-", label="HLT", color="steelblue")
    if reco_records:
        plt.plot(arr(reco_records, "pt_center"), arr(reco_records, "response"), "s--", label=label, color="forestgreen")
    plt.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    plt.xlabel("Jet pT truth (offline)")
    plt.ylabel("Response: pT_reco / pT_truth")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.subplot(1, 2, 2)
    if hlt_records:
        plt.plot(arr(hlt_records, "pt_center"), arr(hlt_records, "resolution"), "o-", label="HLT", color="steelblue")
    if reco_records:
        plt.plot(arr(reco_records, "pt_center"), arr(reco_records, "resolution"), "s--", label=label, color="forestgreen")
    plt.xlabel("Jet pT truth (offline)")
    plt.ylabel("Resolution: std(pT_reco / pT_truth)")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _json_records(records):
    return [{k: (int(v) if k == "count" else float(v)) for k, v in r.items()} for r in records]


def _scale_records(records, scale: float):
    out = []
    for r in records:
        rr = dict(r)
        rr["response"] = float(scale) * float(rr["response"])
        rr["resolution"] = abs(float(scale)) * float(rr["resolution"])
        out.append(rr)
    return out


def _fit_scale(pt_truth: np.ndarray, pt_reco: np.ndarray, mode: str) -> float:
    valid = np.isfinite(pt_truth) & np.isfinite(pt_reco) & (pt_truth > 1e-8) & (pt_reco > 1e-8)
    if not valid.any():
        return 1.0
    ratio = pt_reco[valid] / pt_truth[valid]
    if mode == "median_response":
        return float(1.0 / np.median(ratio))
    if mode == "mean_response":
        return float(1.0 / np.mean(ratio))
    if mode == "least_squares":
        # min_s sum((s*reco - truth)^2)
        r = pt_reco[valid].astype(np.float64)
        t = pt_truth[valid].astype(np.float64)
        return float(np.dot(r, t) / max(np.dot(r, r), 1e-12))
    if mode == "score_grid":
        lo, hi = 0.5, 1.5
        base = np.linspace(lo, hi, 2001)
        best_s, best_score = 1.0, float("inf")
        for s in base:
            rec = _records(pt_truth[valid], s * pt_reco[valid], _build_edges(pt_truth[valid], 8), 300)
            sc = _score(rec)
            if sc < best_score:
                best_s, best_score = float(s), float(sc)
        return best_s
    raise ValueError(f"Unknown scale mode: {mode}")


def _cond_bins(summary: dict, y: np.ndarray, pt_truth: np.ndarray) -> List[Tuple[str, str, np.ndarray]]:
    classes = list(summary.get("classes", []))
    out = []
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for cls in ["QCD", "Hbb", "Wqq", "Tbqq"]:
        if cls in class_to_idx:
            out.append((f"class_{cls}", f"class = {cls}", y == class_to_idx[cls]))
    # We cannot reconstruct exact constituent-count bins from the NPZ, so preserve pT bins and class bins here.
    valid = np.isfinite(pt_truth)
    edges = np.quantile(pt_truth[valid], [0.0, 1.0/3.0, 2.0/3.0, 1.0])
    names = ["pt_low", "pt_mid", "pt_high"]
    for i, name in enumerate(names):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == 2:
            m = valid & (pt_truth >= lo) & (pt_truth <= hi)
            label = f"{name}: [{lo:.1f}, {hi:.1f}]"
        else:
            m = valid & (pt_truth >= lo) & (pt_truth < hi)
            label = f"{name}: [{lo:.1f}, {hi:.1f})"
        out.append((name, label, m))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_dir", type=Path, required=True)
    ap.add_argument("--scale_mode", choices=["median_response", "mean_response", "least_squares", "score_grid"], default="mean_response")
    ap.add_argument("--response_n_bins", type=int, default=8)
    ap.add_argument("--response_min_count", type=int, default=300)
    args = ap.parse_args()

    report_dir = args.report_dir.resolve()
    arrays_path = report_dir / "jet_pt_response_best_arrays.npz"
    summary_path = report_dir / "jet_pt_response_summary.json"
    if not arrays_path.exists():
        raise FileNotFoundError(arrays_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    d = np.load(arrays_path)
    summary = json.load(open(summary_path))
    pt_truth = d["pt_truth"].astype(np.float64)
    pt_hlt = d["pt_hlt"].astype(np.float64)
    pt_reco = d["pt_best_reco"].astype(np.float64)
    y = d["y_test"].astype(np.int64)

    scale = _fit_scale(pt_truth, pt_reco, args.scale_mode)
    pt_scaled = scale * pt_reco
    out_dir = report_dir / f"scaled_response_diagnostic_{args.scale_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    edges = _build_edges(pt_truth, args.response_n_bins)
    hlt_records = _records(pt_truth, pt_hlt, edges, args.response_min_count)
    raw_records = _records(pt_truth, pt_reco, edges, args.response_min_count)
    scaled_records = _records(pt_truth, pt_scaled, edges, args.response_min_count)
    best_name = summary.get("best_model", {}).get("name", "best_reco")

    _plot(hlt_records, scaled_records, f"Scaled reco ({best_name})", f"Diagnostic scaled pT response, scale={scale:.4f}", out_dir / "jet_pt_response_resolution_scaled_best.png")

    cond_reports = []
    cond_dir = out_dir / "conditional_response_plots"
    cond_dir.mkdir(exist_ok=True)
    source_cond = summary.get("conditional_response_bins", [])
    if source_cond:
        cond_iter = []
        for b in source_cond:
            h = b.get("hlt_records", [])
            raw = b.get("best_reco_records", [])
            s = _scale_records(raw, scale)
            cond_iter.append((b.get("key", "bin"), b.get("label", b.get("key", "bin")), int(b.get("n_jets", 0)), h, s))
    else:
        cond_iter = []
        for key, label, mask in _cond_bins(summary, y, pt_truth):
            n = int(mask.sum())
            if n < max(args.response_min_count, 10):
                continue
            e = _build_edges(pt_truth[mask], args.response_n_bins)
            h = _records(pt_truth[mask], pt_hlt[mask], e, args.response_min_count)
            s = _records(pt_truth[mask], pt_scaled[mask], e, args.response_min_count)
            cond_iter.append((key, label, n, h, s))

    for key, label, n, h, s in cond_iter:
        if n < max(args.response_min_count, 10):
            continue
        p = cond_dir / f"jet_pt_response_scaled_{key.lower()}.png"
        _plot(h, s, f"Scaled reco ({best_name})", f"Diagnostic scaled pT response: {label} (N={n})", p)
        cond_reports.append({
            "key": key,
            "label": label,
            "n_jets": n,
            "plot": str(p),
            "hlt_score": _score(h),
            "scaled_reco_score": _score(s),
            "improvement_vs_hlt_score": _score(h) - _score(s),
            "hlt_records": _json_records(h),
            "scaled_reco_records": _json_records(s),
        })

    out = {
        "diagnostic_only": True,
        "note": "Global post-hoc scale applied to best reco pT. This is not the raw uncalibrated model result.",
        "source_report_dir": str(report_dir),
        "best_model": best_name,
        "scale_mode": args.scale_mode,
        "scale": float(scale),
        "raw_reco_score": _score(raw_records),
        "hlt_score": _score(hlt_records),
        "scaled_reco_score": _score(scaled_records),
        "scaled_improvement_vs_hlt_score": _score(hlt_records) - _score(scaled_records),
        "hlt_records": _json_records(hlt_records),
        "raw_reco_records": _json_records(raw_records),
        "scaled_reco_records": _json_records(scaled_records),
        "conditional_response_bins": cond_reports,
        "outputs": {
            "plot": str(out_dir / "jet_pt_response_resolution_scaled_best.png"),
            "conditional_plot_dir": str(cond_dir),
            "summary_json": str(out_dir / "scaled_response_summary.json"),
        },
    }
    json.dump(out, open(out_dir / "scaled_response_summary.json", "w"), indent=2, sort_keys=True)
    print(f"scale_mode={args.scale_mode} scale={scale:.6f}")
    print(f"HLT score={out['hlt_score']:.6f} raw reco score={out['raw_reco_score']:.6f} scaled reco score={out['scaled_reco_score']:.6f}")
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
