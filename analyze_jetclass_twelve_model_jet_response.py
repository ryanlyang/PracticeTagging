#!/usr/bin/env python3
"""
JetClass pT response/resolution audit for a set of reconstructed HLT models.

This script evaluates the reconstructor checkpoints directly. It does not use
teacher scores, fused scores, classifier logits, or test labels for model
selection. The target is the offline JetClass constituent collection.
"""

from __future__ import annotations

import argparse
import copy
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

import analyze_jetclass_four_model_stacked_fusion as fusion


def _ns_from_json(path: Path) -> SimpleNamespace:
    return SimpleNamespace(**json.loads(path.read_text()))


def _parse_model_spec(s: str) -> fusion.ModelSpec:
    spec = fusion._parse_model_spec(s)
    if spec.kind == "baseline_hlt":
        raise ValueError("baseline_hlt has no reconstructor output; use stage2/joint/reco_only_stagea models.")
    return spec


def _resolve_class_names(args_ref: SimpleNamespace, files_by_class: Dict[str, Sequence[Path]]) -> List[str]:
    if str(args_ref.class_assignment) == "canonical_labels":
        class_names = list(fusion.CANONICAL_CLASS_ORDER)
    else:
        class_names = sorted(files_by_class.keys())

    include = getattr(args_ref, "include_classes", "")
    if include:
        resolved = fusion.v2._resolve_included_classes(str(include), class_names)
        if resolved:
            class_names = list(resolved)
    return class_names


def _build_test_data(
    args_ref: SimpleNamespace,
    data_dir: Path,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    files_by_class = fusion.collect_files_by_class(data_dir.resolve())
    class_names = _resolve_class_names(args_ref, files_by_class)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    source_files = files_by_class
    include = getattr(args_ref, "include_classes", "")
    if include and class_names and all(c in files_by_class for c in class_names):
        source_files = {c: files_by_class[c] for c in class_names}

    _, _, te_files = fusion.split_files_by_class(
        source_files,
        n_train=int(args_ref.train_files_per_class),
        n_val=int(args_ref.val_files_per_class),
        n_test=int(args_ref.test_files_per_class),
        shuffle=bool(args_ref.shuffle_files),
        seed=int(args_ref.seed),
    )

    te_tok_raw, te_mask_raw, te_y = fusion.load_split(
        te_files,
        n_total=int(args_ref.n_test_jets),
        max_constits=int(args_ref.max_constits),
        class_to_idx=class_to_idx,
        seed=int(args_ref.seed) + 303,
        class_assignment=str(args_ref.class_assignment),
    )

    hlt_params = fusion.HLTParams(
        hlt_pt_threshold=float(args_ref.hlt_pt_threshold),
        merge_prob_scale=float(args_ref.merge_prob_scale),
        reassign_scale=float(args_ref.reassign_scale),
        smear_scale=float(args_ref.smear_scale),
        eff_plateau_barrel=float(args_ref.eff_plateau_barrel),
        eff_plateau_endcap=float(args_ref.eff_plateau_endcap),
        eff_turnon_pt=float(args_ref.eff_turnon_pt),
        eff_width_pt=float(args_ref.eff_width_pt),
    )
    te_hlt_tok_raw, te_hlt_mask_raw, _, _ = fusion.v2.build_hlt_view(
        te_tok_raw,
        te_mask_raw,
        params=hlt_params,
        seed=int(args_ref.seed) + 1003,
        return_provenance=True,
    )

    te_feat_hlt = fusion.compute_features(
        te_hlt_tok_raw,
        te_hlt_mask_raw,
        feature_mode=str(args_ref.feature_mode),
        feature_preprocessing=str(args_ref.feature_preprocessing),
    )

    if str(args_ref.feature_preprocessing) != "canonical":
        tr_files, _, _ = fusion.split_files_by_class(
            source_files,
            n_train=int(args_ref.train_files_per_class),
            n_val=int(args_ref.val_files_per_class),
            n_test=int(args_ref.test_files_per_class),
            shuffle=bool(args_ref.shuffle_files),
            seed=int(args_ref.seed),
        )
        tr_tok_raw, tr_mask_raw, tr_y = fusion.load_split(
            tr_files,
            n_total=int(args_ref.n_train_jets),
            max_constits=int(args_ref.max_constits),
            class_to_idx=class_to_idx,
            seed=int(args_ref.seed) + 101,
            class_assignment=str(args_ref.class_assignment),
        )
        tr_feat_off = fusion.compute_features(
            tr_tok_raw,
            tr_mask_raw,
            feature_mode=str(args_ref.feature_mode),
            feature_preprocessing=str(args_ref.feature_preprocessing),
        )
        idx_all = np.arange(len(tr_y))
        mean, std = fusion.get_mean_std(tr_feat_off, tr_mask_raw, idx_all)
        te_feat_hlt = fusion.standardize(te_feat_hlt, te_hlt_mask_raw, mean, std)

    return (
        class_names,
        te_tok_raw[:, :, :4].astype(np.float32),
        te_mask_raw.astype(bool),
        te_hlt_tok_raw[:, :, :4].astype(np.float32),
        te_hlt_mask_raw.astype(bool),
        te_feat_hlt.astype(np.float32),
        te_y.astype(np.int64),
    )


def _jet_pt_from_ptetaphi(tokens: np.ndarray, mask: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    pt = np.maximum(tokens[:, :, 0], 0.0).astype(np.float64)
    phi = tokens[:, :, 2].astype(np.float64)
    w = mask.astype(np.float64)
    if weights is not None:
        w = w * np.asarray(weights, dtype=np.float64)
    px = (pt * np.cos(phi) * w).sum(axis=1)
    py = (pt * np.sin(phi) * w).sum(axis=1)
    return np.sqrt(px * px + py * py)


def _build_pt_edges(pt_truth: np.ndarray, n_bins: int) -> np.ndarray:
    valid = np.isfinite(pt_truth) & (pt_truth > 1e-8)
    pt = pt_truth[valid]
    if pt.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)
    q = np.linspace(0.0, 1.0, int(max(n_bins, 1)) + 1)
    edges = np.unique(np.quantile(pt, q))
    if edges.size < 2:
        center = float(np.median(pt))
        edges = np.array([max(center * 0.9, 0.0), center * 1.1 + 1e-6], dtype=np.float64)
    return edges.astype(np.float64)


def _response_records(
    pt_truth: np.ndarray,
    pt_reco: np.ndarray,
    edges: np.ndarray,
    min_count: int,
) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    valid = np.isfinite(pt_truth) & np.isfinite(pt_reco) & (pt_truth > 1e-8)
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        in_bin = valid & (pt_truth >= lo)
        in_bin = in_bin & (pt_truth < hi if i < len(edges) - 2 else pt_truth <= hi)
        n = int(in_bin.sum())
        if n < int(min_count):
            continue
        ratio = pt_reco[in_bin] / pt_truth[in_bin]
        ratio = ratio[np.isfinite(ratio)]
        if ratio.size == 0:
            continue
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


def _score_records(
    records: Sequence[Dict[str, float]],
    bias_weight: float,
    resolution_weight: float,
) -> float:
    if not records:
        return float("inf")
    counts = np.asarray([max(float(r["count"]), 1.0) for r in records], dtype=np.float64)
    terms = np.asarray(
        [
            float(bias_weight) * abs(float(r["response"]) - 1.0)
            + float(resolution_weight) * float(r["resolution"])
            for r in records
        ],
        dtype=np.float64,
    )
    return float(np.average(terms, weights=counts))


def _plot_response_resolution(
    hlt_records: Sequence[Dict[str, float]],
    reco_records: Sequence[Dict[str, float]],
    reco_label: str,
    out_path: Path,
) -> None:
    def arr(records: Sequence[Dict[str, float]], key: str) -> np.ndarray:
        return np.asarray([float(r[key]) for r in records], dtype=np.float64)

    plt.figure(figsize=(10, 4.2))
    plt.subplot(1, 2, 1)
    if hlt_records:
        plt.plot(arr(hlt_records, "pt_center"), arr(hlt_records, "response"), "o-", label="HLT", color="steelblue")
    if reco_records:
        plt.plot(arr(reco_records, "pt_center"), arr(reco_records, "response"), "s--", label=reco_label, color="forestgreen")
    plt.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    plt.xlabel("Jet pT truth (offline)")
    plt.ylabel("Response: pT_reco / pT_truth")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)

    plt.subplot(1, 2, 2)
    if hlt_records:
        plt.plot(arr(hlt_records, "pt_center"), arr(hlt_records, "resolution"), "o-", label="HLT", color="steelblue")
    if reco_records:
        plt.plot(arr(reco_records, "pt_center"), arr(reco_records, "resolution"), "s--", label=reco_label, color="forestgreen")
    plt.xlabel("Jet pT truth (offline)")
    plt.ylabel("Resolution: std(pT_reco / pT_truth)")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_model_score_bars(model_reports: Dict[str, Dict[str, object]], hlt_score: float, out_path: Path) -> None:
    names = list(model_reports.keys())
    scores = [float(model_reports[n]["score"]) for n in names]
    order = np.argsort(scores)
    names = [names[i] for i in order]
    scores = [scores[i] for i in order]
    colors = ["forestgreen" if i == 0 else "slategray" for i in range(len(names))]

    plt.figure(figsize=(10, max(4.5, 0.38 * len(names))))
    y = np.arange(len(names))
    plt.barh(y, scores, color=colors)
    plt.axvline(float(hlt_score), color="steelblue", linestyle="--", linewidth=1.5, label=f"HLT score={hlt_score:.4f}")
    plt.yticks(y, names)
    plt.gca().invert_yaxis()
    plt.xlabel("lower is better: |response - 1| + resolution")
    plt.title("JetClass pT Response Recovery Ranking")
    plt.grid(True, axis="x", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _load_reconstructor(
    spec: fusion.ModelSpec,
    run_args: SimpleNamespace,
    input_dim: int,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, str]]:
    reco_sd, reco_src = fusion._load_state_dict_from_run_dir(
        spec.run_dir,
        ("offline_reconstructor_stage2.pt", "offline_reconstructor.pt"),
    )
    reco_family = fusion._detect_reco_family_from_state_dict(reco_sd)
    reco = fusion._build_reconstructor_with_attrs(run_args, input_dim, device, reco_family=reco_family)
    try:
        reco.load_state_dict(reco_sd, strict=True)
    except RuntimeError as e:
        print(
            f"[load-warning] strict reco load failed for {spec.name} "
            f"(family={reco_family}); retrying strict=False. Error: {e}"
        )
        reco.load_state_dict(reco_sd, strict=False)
    reco.eval()
    return reco, {"reco_ckpt": str(reco_src), "reco_family": str(reco_family)}


@torch.no_grad()
def _predict_reco_pt(
    reco: torch.nn.Module,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt4: np.ndarray,
    device: torch.device,
    batch_size: int,
    weight_floor: float,
) -> np.ndarray:
    n = int(feat_hlt.shape[0])
    out = np.zeros((n,), dtype=np.float32)
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
        pt = cand[:, :, 0].clamp(min=0.0)
        phi = cand[:, :, 2]
        px = (pt * torch.cos(phi) * w).sum(dim=1)
        py = (pt * torch.sin(phi) * w).sum(dim=1)
        out[i:j] = torch.sqrt(px.pow(2) + py.pow(2)).detach().cpu().numpy().astype(np.float32)
    return out


def _jsonable_records(records: Sequence[Dict[str, float]]) -> List[Dict[str, object]]:
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
    ap.add_argument("--score_bias_weight", type=float, default=1.0)
    ap.add_argument("--score_resolution_weight", type=float, default=1.0)
    ap.add_argument(
        "--max_test_jets",
        type=int,
        default=0,
        help="Optional cap for quick debugging. Default 0 uses the full test split from args.json.",
    )
    args = ap.parse_args()

    if fusion._IMPORT_ERROR is not None:
        raise RuntimeError("Failed to import JetClass dual-view dependencies.") from fusion._IMPORT_ERROR
    if fusion._EVAL_IMPORT_ERROR is not None:
        raise RuntimeError("Failed to import JetClass eval/data dependencies.") from fusion._EVAL_IMPORT_ERROR

    specs = [_parse_model_spec(s) for s in args.model]
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate model names detected: {names}")

    for s in specs:
        if not s.run_dir.exists():
            raise FileNotFoundError(f"Model run_dir not found: {s.run_dir}")
        if not (s.run_dir / "args.json").exists():
            raise FileNotFoundError(f"Missing args.json in run_dir: {s.run_dir}")

    run_args_map = {s.name: _ns_from_json(s.run_dir / "args.json") for s in specs}
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
    ) = _build_test_data(ref_args, args.data_dir)

    input_dim = int(te_feat_hlt.shape[-1])
    pt_truth = _jet_pt_from_ptetaphi(te_off_const4, te_off_mask)
    pt_hlt = _jet_pt_from_ptetaphi(te_hlt_const4, te_hlt_mask)
    pt_edges = _build_pt_edges(pt_truth, int(args.response_n_bins))
    hlt_records = _response_records(pt_truth, pt_hlt, pt_edges, int(args.response_min_count))
    hlt_score = _score_records(hlt_records, float(args.score_bias_weight), float(args.score_resolution_weight))

    model_reports: Dict[str, Dict[str, object]] = {}
    best_name = ""
    best_score = float("inf")
    best_records: List[Dict[str, float]] = []
    best_pt_reco: np.ndarray | None = None

    for spec in specs:
        print(f"Evaluating response for {spec.name}: {spec.run_dir}")
        reco, sources = _load_reconstructor(spec, run_args_map[spec.name], input_dim, device)
        pt_reco = _predict_reco_pt(
            reco,
            te_feat_hlt,
            te_hlt_mask,
            te_hlt_const4,
            device=device,
            batch_size=int(args.batch_size),
            weight_floor=float(args.corrected_weight_floor),
        )
        records = _response_records(pt_truth, pt_reco, pt_edges, int(args.response_min_count))
        score = _score_records(records, float(args.score_bias_weight), float(args.score_resolution_weight))
        mean_ratio = float(np.nanmean(pt_reco / np.clip(pt_truth, 1e-8, np.inf)))
        model_reports[spec.name] = {
            "kind": spec.kind,
            "run_dir": str(spec.run_dir),
            "sources": sources,
            "score": float(score),
            "mean_response_unbinned": mean_ratio,
            "records": _jsonable_records(records),
        }
        print(f"  score={score:.6f} mean_response={mean_ratio:.6f}")
        if score < best_score:
            best_name = spec.name
            best_score = score
            best_records = records
            best_pt_reco = pt_reco.copy()
        del reco
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if best_pt_reco is None:
        raise RuntimeError("No model produced response records.")

    _plot_response_resolution(
        hlt_records,
        best_records,
        f"Best reco ({best_name})",
        out_dir / "jet_pt_response_resolution_best.png",
    )
    _plot_model_score_bars(model_reports, hlt_score, out_dir / "jet_pt_response_recovery_ranking.png")

    summary = {
        "target": "offline_jetclass_constituents",
        "quantity": "jet_pt",
        "score_definition": (
            "weighted mean over pT bins of "
            "score_bias_weight*abs(mean(pT_reco/pT_offline)-1) "
            "+ score_resolution_weight*std(pT_reco/pT_offline)"
        ),
        "score_bias_weight": float(args.score_bias_weight),
        "score_resolution_weight": float(args.score_resolution_weight),
        "response_n_bins": int(args.response_n_bins),
        "response_min_count": int(args.response_min_count),
        "corrected_weight_floor": float(args.corrected_weight_floor),
        "n_test_jets": int(pt_truth.shape[0]),
        "classes": list(class_names),
        "class_counts": {class_names[i]: int((te_y == i).sum()) for i in range(len(class_names))},
        "hlt": {
            "score": float(hlt_score),
            "mean_response_unbinned": float(np.nanmean(pt_hlt / np.clip(pt_truth, 1e-8, np.inf))),
            "records": _jsonable_records(hlt_records),
        },
        "best_model": {
            "name": best_name,
            "score": float(best_score),
            "improvement_vs_hlt_score": float(hlt_score - best_score),
        },
        "models": model_reports,
        "outputs": {
            "best_plot": str(out_dir / "jet_pt_response_resolution_best.png"),
            "ranking_plot": str(out_dir / "jet_pt_response_recovery_ranking.png"),
            "summary_json": str(out_dir / "jet_pt_response_summary.json"),
            "arrays_npz": str(out_dir / "jet_pt_response_best_arrays.npz"),
        },
    }
    (out_dir / "jet_pt_response_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    np.savez_compressed(
        out_dir / "jet_pt_response_best_arrays.npz",
        pt_truth=pt_truth.astype(np.float32),
        pt_hlt=pt_hlt.astype(np.float32),
        pt_best_reco=best_pt_reco.astype(np.float32),
        y_test=te_y.astype(np.int64),
        pt_edges=pt_edges.astype(np.float64),
    )

    print("\nJet pT response/recovery summary")
    print(f"  HLT score:       {hlt_score:.6f}")
    print(f"  Best model:      {best_name}")
    print(f"  Best reco score: {best_score:.6f}")
    print(f"  Improvement:     {hlt_score - best_score:.6f}")
    print(f"Saved summary:     {out_dir / 'jet_pt_response_summary.json'}")
    print(f"Saved best plot:   {out_dir / 'jet_pt_response_resolution_best.png'}")
    print(f"Saved ranking:     {out_dir / 'jet_pt_response_recovery_ranking.png'}")


if __name__ == "__main__":
    main()
