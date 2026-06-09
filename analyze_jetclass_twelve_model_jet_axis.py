#!/usr/bin/env python3
"""
JetClass jet-axis recovery audit for a set of reconstructed HLT models.

The script selects one globally best reconstructor by jet-axis recovery
against offline JetClass constituents, then makes HLT-vs-reco plots for that
single fixed model. It never picks the best reconstructor per jet.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'user')}")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import analyze_jetclass_four_model_stacked_fusion as fusion
import analyze_jetclass_twelve_model_jet_response as response


def _wrap_delta_phi(phi: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(phi), np.cos(phi))


def _axis_from_ptetaphi(
    tokens: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pt_tok = np.maximum(tokens[:, :, 0], 0.0).astype(np.float64)
    eta_tok = tokens[:, :, 1].astype(np.float64)
    phi_tok = tokens[:, :, 2].astype(np.float64)
    e_tok = np.maximum(tokens[:, :, 3], 0.0).astype(np.float64)
    w = mask.astype(np.float64)
    if weights is not None:
        w = w * np.asarray(weights, dtype=np.float64)

    px = (pt_tok * np.cos(phi_tok) * w).sum(axis=1)
    py = (pt_tok * np.sin(phi_tok) * w).sum(axis=1)
    pz = (pt_tok * np.sinh(eta_tok) * w).sum(axis=1)
    e = (e_tok * w).sum(axis=1)
    pt = np.sqrt(px * px + py * py)
    eta = np.arcsinh(pz / np.clip(pt, 1e-8, np.inf))
    phi = np.arctan2(py, px)
    return pt, eta, phi, px, py, pz


def _axis_errors(
    eta_pred: np.ndarray,
    phi_pred: np.ndarray,
    pvec_pred: Tuple[np.ndarray, np.ndarray, np.ndarray],
    eta_truth: np.ndarray,
    phi_truth: np.ndarray,
    pvec_truth: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Dict[str, np.ndarray]:
    deta = eta_pred - eta_truth
    dphi = _wrap_delta_phi(phi_pred - phi_truth)
    dr = np.sqrt(deta * deta + dphi * dphi)

    px, py, pz = pvec_pred
    tx, ty, tz = pvec_truth
    dot = px * tx + py * ty + pz * tz
    norm = np.sqrt(px * px + py * py + pz * pz) * np.sqrt(tx * tx + ty * ty + tz * tz)
    cosang = np.clip(dot / np.clip(norm, 1e-12, np.inf), -1.0, 1.0)
    angle3d = np.arccos(cosang)
    return {"delta_eta": deta, "delta_phi": dphi, "delta_R": dr, "angle3d": angle3d}


def _finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def _metrics(errors: Dict[str, np.ndarray]) -> Dict[str, float]:
    dr = _finite(errors["delta_R"])
    angle = _finite(errors["angle3d"])
    deta = _finite(errors["delta_eta"])
    dphi = _finite(errors["delta_phi"])
    if dr.size == 0:
        return {
            "mean_deltaR": float("nan"),
            "std_deltaR": float("nan"),
            "median_deltaR": float("nan"),
            "q68_deltaR": float("nan"),
            "q90_deltaR": float("nan"),
            "mean_angle3d": float("nan"),
            "std_angle3d": float("nan"),
            "mean_delta_eta": float("nan"),
            "std_delta_eta": float("nan"),
            "mean_delta_phi": float("nan"),
            "std_delta_phi": float("nan"),
            "mean_abs_delta_eta": float("nan"),
            "mean_abs_delta_phi": float("nan"),
        }
    return {
        "mean_deltaR": float(np.mean(dr)),
        "std_deltaR": float(np.std(dr)),
        "median_deltaR": float(np.median(dr)),
        "q68_deltaR": float(np.quantile(dr, 0.68)),
        "q90_deltaR": float(np.quantile(dr, 0.90)),
        "mean_angle3d": float(np.mean(angle)) if angle.size else float("nan"),
        "std_angle3d": float(np.std(angle)) if angle.size else float("nan"),
        "mean_delta_eta": float(np.mean(deta)) if deta.size else float("nan"),
        "std_delta_eta": float(np.std(deta)) if deta.size else float("nan"),
        "mean_delta_phi": float(np.mean(dphi)) if dphi.size else float("nan"),
        "std_delta_phi": float(np.std(dphi)) if dphi.size else float("nan"),
        "mean_abs_delta_eta": float(np.mean(np.abs(deta))) if deta.size else float("nan"),
        "mean_abs_delta_phi": float(np.mean(np.abs(dphi))) if dphi.size else float("nan"),
    }


def _score(metrics: Dict[str, float], mean_weight: float, std_weight: float) -> float:
    return float(mean_weight) * float(metrics["mean_deltaR"]) + float(std_weight) * float(metrics["std_deltaR"])


@torch.no_grad()
def _predict_reco_axis(
    reco: torch.nn.Module,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt4: np.ndarray,
    device: torch.device,
    batch_size: int,
    weight_floor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(feat_hlt.shape[0])
    pt_out = np.zeros((n,), dtype=np.float32)
    eta_out = np.zeros((n,), dtype=np.float32)
    phi_out = np.zeros((n,), dtype=np.float32)
    px_out = np.zeros((n,), dtype=np.float32)
    py_out = np.zeros((n,), dtype=np.float32)
    pz_out = np.zeros((n,), dtype=np.float32)
    bs = int(max(batch_size, 1))
    for i in range(0, n, bs):
        j = min(n, i + bs)
        feat = torch.from_numpy(feat_hlt[i:j]).to(device=device, dtype=torch.float32, non_blocking=True)
        mask = torch.from_numpy(mask_hlt[i:j]).to(device=device, dtype=torch.bool, non_blocking=True)
        c4 = torch.from_numpy(const_hlt4[i:j]).to(device=device, dtype=torch.float32, non_blocking=True)
        reco_out = reco(feat, mask, c4, stage_scale=1.0)
        cand = reco_out["cand_tokens"]
        w = reco_out["cand_weights"].clamp(0.0, 1.0)
        if float(weight_floor) > 0.0:
            w = torch.where(w >= float(weight_floor), w, torch.zeros_like(w))

        pt_tok = cand[:, :, 0].clamp(min=0.0)
        eta_tok = cand[:, :, 1]
        phi_tok = cand[:, :, 2]
        px = (pt_tok * torch.cos(phi_tok) * w).sum(dim=1)
        py = (pt_tok * torch.sin(phi_tok) * w).sum(dim=1)
        pz = (pt_tok * torch.sinh(eta_tok) * w).sum(dim=1)
        pt = torch.sqrt(px.pow(2) + py.pow(2))
        eta = torch.asinh(pz / pt.clamp(min=1e-8))
        phi = torch.atan2(py, px)

        pt_out[i:j] = pt.detach().cpu().numpy().astype(np.float32)
        eta_out[i:j] = eta.detach().cpu().numpy().astype(np.float32)
        phi_out[i:j] = phi.detach().cpu().numpy().astype(np.float32)
        px_out[i:j] = px.detach().cpu().numpy().astype(np.float32)
        py_out[i:j] = py.detach().cpu().numpy().astype(np.float32)
        pz_out[i:j] = pz.detach().cpu().numpy().astype(np.float32)
    return pt_out, eta_out, phi_out, px_out, py_out, pz_out


def _plot_deltaR_hist_cdf(hlt_dr: np.ndarray, reco_dr: np.ndarray, label: str, out_path: Path) -> None:
    h = _finite(hlt_dr)
    r = _finite(reco_dr)
    max_x = float(np.quantile(np.concatenate([h, r]), 0.995)) if h.size and r.size else 1.0
    max_x = max(max_x, 1e-3)
    bins = np.linspace(0.0, max_x, 80)

    plt.figure(figsize=(10, 4.2))
    plt.subplot(1, 2, 1)
    plt.hist(h, bins=bins, density=True, histtype="step", linewidth=1.8, label="HLT", color="steelblue")
    plt.hist(r, bins=bins, density=True, histtype="step", linewidth=1.8, label=label, color="forestgreen")
    plt.xlabel("DeltaR to offline axis")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)

    plt.subplot(1, 2, 2)
    for vals, name, color in [(h, "HLT", "steelblue"), (r, label, "forestgreen")]:
        vals = np.sort(vals)
        y = np.arange(1, vals.size + 1, dtype=np.float64) / max(vals.size, 1)
        plt.plot(vals, y, label=name, color=color)
    plt.xlabel("DeltaR to offline axis")
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _binned_stats(pt_truth: np.ndarray, values: np.ndarray, edges: np.ndarray, min_count: int) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    valid = np.isfinite(pt_truth) & np.isfinite(values)
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        sel = valid & (pt_truth >= lo)
        sel = sel & (pt_truth < hi if i < len(edges) - 2 else pt_truth <= hi)
        n = int(sel.sum())
        if n < int(min_count):
            continue
        v = values[sel]
        out.append(
            {
                "pt_low": lo,
                "pt_high": hi,
                "pt_center": 0.5 * (lo + hi),
                "count": n,
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "q68": float(np.quantile(v, 0.68)),
            }
        )
    return out


def _plot_deltaR_vs_pt(
    hlt_bins: Sequence[Dict[str, float]],
    reco_bins: Sequence[Dict[str, float]],
    label: str,
    out_path: Path,
) -> None:
    def arr(records: Sequence[Dict[str, float]], key: str) -> np.ndarray:
        return np.asarray([float(r[key]) for r in records], dtype=np.float64)

    plt.figure(figsize=(10, 4.2))
    plt.subplot(1, 2, 1)
    if hlt_bins:
        plt.plot(arr(hlt_bins, "pt_center"), arr(hlt_bins, "mean"), "o-", label="HLT mean", color="steelblue")
        plt.plot(arr(hlt_bins, "pt_center"), arr(hlt_bins, "q68"), "o:", label="HLT q68", color="lightskyblue")
    if reco_bins:
        plt.plot(arr(reco_bins, "pt_center"), arr(reco_bins, "mean"), "s--", label=f"{label} mean", color="forestgreen")
        plt.plot(arr(reco_bins, "pt_center"), arr(reco_bins, "q68"), "s:", label=f"{label} q68", color="limegreen")
    plt.xlabel("Jet pT truth (offline)")
    plt.ylabel("DeltaR to offline axis")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, fontsize=8)

    plt.subplot(1, 2, 2)
    if hlt_bins:
        plt.plot(arr(hlt_bins, "pt_center"), arr(hlt_bins, "std"), "o-", label="HLT", color="steelblue")
    if reco_bins:
        plt.plot(arr(reco_bins, "pt_center"), arr(reco_bins, "std"), "s--", label=label, color="forestgreen")
    plt.xlabel("Jet pT truth (offline)")
    plt.ylabel("std(DeltaR)")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_scatter(hlt_dr: np.ndarray, reco_dr: np.ndarray, label: str, max_points: int, seed: int, out_path: Path) -> None:
    valid = np.isfinite(hlt_dr) & np.isfinite(reco_dr)
    idx = np.flatnonzero(valid)
    if idx.size > int(max_points):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(idx, size=int(max_points), replace=False)
    x = hlt_dr[idx]
    y = reco_dr[idx]
    lim = float(np.quantile(np.concatenate([x, y]), 0.995)) if x.size else 1.0
    lim = max(lim, 1e-3)

    plt.figure(figsize=(5.2, 5.0))
    plt.scatter(x, y, s=3, alpha=0.15, color="black", rasterized=True)
    plt.plot([0.0, lim], [0.0, lim], color="crimson", linestyle="--", linewidth=1.2)
    plt.xlim(0.0, lim)
    plt.ylim(0.0, lim)
    plt.xlabel("HLT DeltaR to offline axis")
    plt.ylabel(f"{label} DeltaR to offline axis")
    plt.title("Points below diagonal improved")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_delta_eta_phi(
    hlt_errors: Dict[str, np.ndarray],
    reco_errors: Dict[str, np.ndarray],
    label: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(10, 4.2))
    for panel, key, title in [(1, "delta_eta", "Delta eta"), (2, "delta_phi", "Delta phi")]:
        h = _finite(hlt_errors[key])
        r = _finite(reco_errors[key])
        max_abs = float(np.quantile(np.abs(np.concatenate([h, r])), 0.995)) if h.size and r.size else 0.1
        max_abs = max(max_abs, 1e-3)
        bins = np.linspace(-max_abs, max_abs, 100)
        plt.subplot(1, 2, panel)
        plt.hist(h, bins=bins, density=True, histtype="step", linewidth=1.8, label="HLT", color="steelblue")
        plt.hist(r, bins=bins, density=True, histtype="step", linewidth=1.8, label=label, color="forestgreen")
        plt.xlabel(f"{title} to offline axis")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.25)
        plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _jsonable_bins(records: Sequence[Dict[str, float]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for r in records:
        out.append({k: (int(v) if k == "count" else float(v)) for k, v in r.items()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec: name:kind:run_dir (kind in stage2, joint, reco_only_stagea)",
    )
    ap.add_argument("--data_dir", type=Path, default=Path("/home/ryreu/atlas/PracticeTagging/data/jetclass_part0"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--response_n_bins", type=int, default=8)
    ap.add_argument("--response_min_count", type=int, default=300)
    ap.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    ap.add_argument("--score_mean_weight", type=float, default=1.0)
    ap.add_argument("--score_std_weight", type=float, default=1.0)
    ap.add_argument("--max_test_jets", type=int, default=200000)
    ap.add_argument("--scatter_max_points", type=int, default=50000)
    ap.add_argument("--scatter_seed", type=int, default=52)
    ap.add_argument(
        "--plot_all_models",
        action="store_true",
        help="Also save HLT-vs-reco jet-axis PNGs for each supplied model.",
    )
    args = ap.parse_args()

    if fusion._IMPORT_ERROR is not None:
        raise RuntimeError("Failed to import JetClass dual-view dependencies.") from fusion._IMPORT_ERROR
    if fusion._EVAL_IMPORT_ERROR is not None:
        raise RuntimeError("Failed to import JetClass eval/data dependencies.") from fusion._EVAL_IMPORT_ERROR

    specs = [response._parse_model_spec(s) for s in args.model]
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate model names detected: {names}")
    for s in specs:
        if not s.run_dir.exists():
            raise FileNotFoundError(f"Model run_dir not found: {s.run_dir}")
        if not (s.run_dir / "args.json").exists():
            raise FileNotFoundError(f"Missing args.json in run_dir: {s.run_dir}")

    run_args_map = {s.name: response._ns_from_json(s.run_dir / "args.json") for s in specs}
    ref_args = copy.deepcopy(run_args_map[specs[0].name])
    for s in specs[1:]:
        fusion._check_run_compat(ref_args, run_args_map[s.name], s.name)
    if int(args.max_test_jets) > 0:
        ref_args.n_test_jets = int(args.max_test_jets)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or str(args.device).startswith("cpu") else "cpu")
    print(f"Using device: {device}")

    (
        class_names,
        te_off_const4,
        te_off_mask,
        te_hlt_const4,
        te_hlt_mask,
        te_feat_hlt,
        te_y,
    ) = response._build_test_data(ref_args, args.data_dir)

    input_dim = int(te_feat_hlt.shape[-1])
    pt_truth, eta_truth, phi_truth, px_truth, py_truth, pz_truth = _axis_from_ptetaphi(te_off_const4, te_off_mask)
    _, eta_hlt, phi_hlt, px_hlt, py_hlt, pz_hlt = _axis_from_ptetaphi(te_hlt_const4, te_hlt_mask)
    hlt_errors = _axis_errors(
        eta_hlt,
        phi_hlt,
        (px_hlt, py_hlt, pz_hlt),
        eta_truth,
        phi_truth,
        (px_truth, py_truth, pz_truth),
    )
    hlt_metrics = _metrics(hlt_errors)
    hlt_score = _score(hlt_metrics, float(args.score_mean_weight), float(args.score_std_weight))
    pt_edges = response._build_pt_edges(pt_truth, int(args.response_n_bins))
    hlt_bins = _binned_stats(pt_truth, hlt_errors["delta_R"], pt_edges, int(args.response_min_count))

    model_reports: Dict[str, Dict[str, object]] = {}
    all_model_plot_dir = out_dir / "all_model_axis_plots"
    if bool(args.plot_all_models):
        all_model_plot_dir.mkdir(parents=True, exist_ok=True)
    best_name = ""
    best_score = float("inf")
    best_errors: Dict[str, np.ndarray] | None = None
    best_axis: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    for spec in specs:
        print(f"Evaluating axis recovery for {spec.name}: {spec.run_dir}")
        reco, sources = response._load_reconstructor(spec, run_args_map[spec.name], input_dim, device)
        _, eta_reco, phi_reco, px_reco, py_reco, pz_reco = _predict_reco_axis(
            reco,
            te_feat_hlt,
            te_hlt_mask,
            te_hlt_const4,
            device=device,
            batch_size=int(args.batch_size),
            weight_floor=float(args.corrected_weight_floor),
        )
        reco_errors = _axis_errors(
            eta_reco,
            phi_reco,
            (px_reco, py_reco, pz_reco),
            eta_truth,
            phi_truth,
            (px_truth, py_truth, pz_truth),
        )
        metrics = _metrics(reco_errors)
        score = _score(metrics, float(args.score_mean_weight), float(args.score_std_weight))
        frac_improved = float(np.mean(reco_errors["delta_R"] < hlt_errors["delta_R"]))
        plot_paths: Dict[str, str] = {}
        if bool(args.plot_all_models):
            safe_name = response._slug(spec.name)
            label = f"Reco ({spec.name})"
            reco_bins_i = _binned_stats(pt_truth, reco_errors["delta_R"], pt_edges, int(args.response_min_count))
            hist_path = all_model_plot_dir / f"jet_axis_deltaR_hist_cdf_{safe_name}.png"
            vspt_path = all_model_plot_dir / f"jet_axis_deltaR_vs_pt_{safe_name}.png"
            scatter_path = all_model_plot_dir / f"jet_axis_deltaR_scatter_{safe_name}.png"
            etaphi_path = all_model_plot_dir / f"jet_axis_delta_eta_phi_{safe_name}.png"
            _plot_deltaR_hist_cdf(hlt_errors["delta_R"], reco_errors["delta_R"], label, hist_path)
            _plot_deltaR_vs_pt(hlt_bins, reco_bins_i, label, vspt_path)
            _plot_scatter(
                hlt_errors["delta_R"],
                reco_errors["delta_R"],
                label,
                int(args.scatter_max_points),
                int(args.scatter_seed),
                scatter_path,
            )
            _plot_delta_eta_phi(hlt_errors, reco_errors, label, etaphi_path)
            plot_paths = {
                "deltaR_hist_cdf": str(hist_path),
                "deltaR_vs_pt": str(vspt_path),
                "deltaR_scatter": str(scatter_path),
                "delta_eta_phi": str(etaphi_path),
            }
        model_reports[spec.name] = {
            "kind": spec.kind,
            "run_dir": str(spec.run_dir),
            "sources": sources,
            "score": float(score),
            "fraction_improved_vs_hlt": frac_improved,
            "metrics": metrics,
            "plots": plot_paths,
        }
        print(f"  score={score:.6f} meanDR={metrics['mean_deltaR']:.6f} stdDR={metrics['std_deltaR']:.6f} improved={frac_improved:.4f}")
        if score < best_score:
            best_name = spec.name
            best_score = float(score)
            best_errors = {k: v.copy() for k, v in reco_errors.items()}
            best_axis = (eta_reco.copy(), phi_reco.copy(), px_reco.copy())
        del reco
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if best_errors is None or best_axis is None:
        raise RuntimeError("No model produced axis metrics.")

    best_label = f"Best reco ({best_name})"
    reco_bins = _binned_stats(pt_truth, best_errors["delta_R"], pt_edges, int(args.response_min_count))
    _plot_deltaR_hist_cdf(hlt_errors["delta_R"], best_errors["delta_R"], best_label, out_dir / "jet_axis_deltaR_hist_cdf_best.png")
    _plot_deltaR_vs_pt(hlt_bins, reco_bins, best_label, out_dir / "jet_axis_deltaR_vs_pt_best.png")
    _plot_scatter(
        hlt_errors["delta_R"],
        best_errors["delta_R"],
        best_label,
        int(args.scatter_max_points),
        int(args.scatter_seed),
        out_dir / "jet_axis_deltaR_scatter_best.png",
    )
    _plot_delta_eta_phi(hlt_errors, best_errors, best_label, out_dir / "jet_axis_delta_eta_phi_best.png")

    best_metrics = model_reports[best_name]["metrics"]
    summary = {
        "target": "offline_jetclass_constituents",
        "quantity": "jet_axis_from_vector_sum",
        "selection": "one globally best reconstructor selected by mean_deltaR + std_deltaR; no per-jet oracle selection",
        "score_definition": "score_mean_weight*mean(deltaR) + score_std_weight*std(deltaR)",
        "score_mean_weight": float(args.score_mean_weight),
        "score_std_weight": float(args.score_std_weight),
        "n_test_jets": int(pt_truth.shape[0]),
        "classes": list(class_names),
        "class_counts": {class_names[i]: int((te_y == i).sum()) for i in range(len(class_names))},
        "hlt": {"score": float(hlt_score), "metrics": hlt_metrics},
        "best_model": {
            "name": best_name,
            "score": float(best_score),
            "metrics": best_metrics,
            "fraction_improved_vs_hlt": model_reports[best_name]["fraction_improved_vs_hlt"],
            "improvement_vs_hlt_score": float(hlt_score - best_score),
        },
        "models": model_reports,
        "binned_deltaR": {
            "hlt": _jsonable_bins(hlt_bins),
            "best_reco": _jsonable_bins(reco_bins),
        },
        "outputs": {
            "deltaR_hist_cdf": str(out_dir / "jet_axis_deltaR_hist_cdf_best.png"),
            "deltaR_vs_pt": str(out_dir / "jet_axis_deltaR_vs_pt_best.png"),
            "deltaR_scatter": str(out_dir / "jet_axis_deltaR_scatter_best.png"),
            "delta_eta_phi": str(out_dir / "jet_axis_delta_eta_phi_best.png"),
            "all_model_plot_dir": str(all_model_plot_dir) if bool(args.plot_all_models) else None,
            "summary_json": str(out_dir / "jet_axis_summary.json"),
            "arrays_npz": str(out_dir / "jet_axis_best_arrays.npz"),
        },
    }
    (out_dir / "jet_axis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    np.savez_compressed(
        out_dir / "jet_axis_best_arrays.npz",
        pt_truth=pt_truth.astype(np.float32),
        eta_truth=eta_truth.astype(np.float32),
        phi_truth=phi_truth.astype(np.float32),
        eta_hlt=eta_hlt.astype(np.float32),
        phi_hlt=phi_hlt.astype(np.float32),
        eta_best_reco=best_axis[0].astype(np.float32),
        phi_best_reco=best_axis[1].astype(np.float32),
        deltaR_hlt=hlt_errors["delta_R"].astype(np.float32),
        deltaR_best_reco=best_errors["delta_R"].astype(np.float32),
        delta_eta_hlt=hlt_errors["delta_eta"].astype(np.float32),
        delta_eta_best_reco=best_errors["delta_eta"].astype(np.float32),
        delta_phi_hlt=hlt_errors["delta_phi"].astype(np.float32),
        delta_phi_best_reco=best_errors["delta_phi"].astype(np.float32),
        y_test=te_y.astype(np.int64),
        pt_edges=pt_edges.astype(np.float64),
    )

    print("\nJet axis recovery summary")
    print(f"  HLT score:       {hlt_score:.6f}")
    print(f"  Best model:      {best_name}")
    print(f"  Best reco score: {best_score:.6f}")
    print(f"  Improvement:     {hlt_score - best_score:.6f}")
    print(f"  Fraction improved vs HLT: {float(model_reports[best_name]['fraction_improved_vs_hlt']):.4f}")
    print(f"Saved summary:     {out_dir / 'jet_axis_summary.json'}")
    if bool(args.plot_all_models):
        print(f"Saved per-model plots: {all_model_plot_dir}")


if __name__ == "__main__":
    main()
