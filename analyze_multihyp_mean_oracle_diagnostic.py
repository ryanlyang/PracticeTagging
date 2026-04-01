#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnostic analyzer for multihyp-mean runs.

Reports, on val/test split:
- Aggregated multi-hyp performance (mean or lse)
- Each individual hypothesis performance
- Oracle-per-jet hypothesis upper bound
- Hypothesis diversity diagnostics
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_multihyp_mean as mh


class JointLiteDataset(Dataset):
    def __init__(
        self,
        feat_hlt_reco: np.ndarray,
        feat_hlt_dual: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt: np.ndarray,
        labels: np.ndarray,
    ):
        self.feat_hlt_reco = feat_hlt_reco.astype(np.float32, copy=False)
        self.feat_hlt_dual = feat_hlt_dual.astype(np.float32, copy=False)
        self.mask_hlt = mask_hlt.astype(np.bool_, copy=False)
        self.const_hlt = const_hlt.astype(np.float32, copy=False)
        self.labels = labels.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int):
        return {
            "feat_hlt_reco": torch.tensor(self.feat_hlt_reco[i], dtype=torch.float32),
            "feat_hlt_dual": torch.tensor(self.feat_hlt_dual[i], dtype=torch.float32),
            "mask_hlt": torch.tensor(self.mask_hlt[i], dtype=torch.bool),
            "const_hlt": torch.tensor(self.const_hlt[i], dtype=torch.float32),
            "label": torch.tensor(self.labels[i], dtype=torch.float32),
        }


def _parse_temps(s: str) -> Tuple[float, ...]:
    vals: List[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok:
            vals.append(max(float(tok), 1e-3))
    if not vals:
        vals = [0.85, 1.00]
    return tuple(vals)


def _load_state(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def _extract_base_model_state(state_dual: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefix = "base_model."
    out = {k[len(prefix) :]: v for k, v in state_dual.items() if k.startswith(prefix)}
    if out:
        return out
    return state_dual


def _infer_dual_input_dims_from_base_state(base_state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    ka = "input_proj_a.0.weight"
    kb = "input_proj_b.0.weight"
    if ka not in base_state or kb not in base_state:
        raise RuntimeError("Could not infer dual input dims from checkpoint.")
    return int(base_state[ka].shape[1]), int(base_state[kb].shape[1])


def _prepare_split_arrays(run_dir: Path):
    data_setup_p = run_dir / "data_setup.json"
    split_p = run_dir / "data_splits.npz"
    if not data_setup_p.exists():
        raise FileNotFoundError(f"Missing {data_setup_p}")
    if not split_p.exists():
        raise FileNotFoundError(f"Missing {split_p}")

    with open(data_setup_p, "r", encoding="utf-8") as f:
        ds = json.load(f)
    split_npz = np.load(split_p)

    train_files = [Path(p) for p in ds["train_files"]]
    n_train_jets = int(ds["n_train_jets"])
    offset_jets = int(ds["offset_jets"])
    max_constits = int(ds["max_constits"])
    seed = int(ds["seed"])

    max_jets_needed = int(offset_jets + n_train_jets)
    all_const_full, all_labels_full = base.load_raw_constituents_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=max_constits,
    )
    if int(all_const_full.shape[0]) < max_jets_needed:
        raise RuntimeError(f"Not enough jets ({all_const_full.shape[0]} < {max_jets_needed})")

    const_raw = all_const_full[offset_jets : offset_jets + n_train_jets]
    labels = all_labels_full[offset_jets : offset_jets + n_train_jets].astype(np.int64)

    cfg = base._deepcopy_config()
    for k, v in ds.get("hlt_effects", {}).items():
        cfg["hlt_effects"][k] = v

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0

    hlt_const, hlt_mask, _, _ = base.apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=seed,
    )
    feat_hlt = base.compute_features(hlt_const, hlt_mask)

    means = split_npz["means"].astype(np.float32)
    stds = split_npz["stds"].astype(np.float32)
    feat_hlt_std = base.standardize(feat_hlt, hlt_mask, means, stds).astype(np.float32)

    return {
        "feat_hlt_std": feat_hlt_std,
        "hlt_mask": hlt_mask.astype(np.bool_),
        "hlt_const": hlt_const.astype(np.float32),
        "labels": labels.astype(np.float32),
        "train_idx": split_npz["train_idx"].astype(np.int64),
        "val_idx": split_npz["val_idx"].astype(np.int64),
        "test_idx": split_npz["test_idx"].astype(np.int64),
    }


def _auc_fpr50(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    if p.size == 0:
        return {"auc": float("nan"), "fpr50": float("nan")}
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    fpr50 = base.fpr_at_target_tpr(fpr, tpr, 0.50)
    return {"auc": float(auc), "fpr50": float(fpr50)}


@torch.no_grad()
def _collect_predictions(
    reconstructor: torch.nn.Module,
    dual_base_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    hyp_temps: Tuple[float, ...],
    agg_mode: str,
    weight_floor: float,
    include_flags: bool,
    max_batches: int,
) -> Dict[str, np.ndarray]:
    reconstructor.eval()
    dual_base_model.eval()

    all_y: List[np.ndarray] = []
    all_p_agg: List[np.ndarray] = []
    all_p_h: List[np.ndarray] = []

    batch_count = 0
    h = len(hyp_temps)

    for batch in loader:
        batch_count += 1
        if max_batches > 0 and batch_count > max_batches:
            break

        feat_hlt_reco = batch["feat_hlt_reco"].to(device)
        feat_hlt_dual = batch["feat_hlt_dual"].to(device)
        mask_hlt = batch["mask_hlt"].to(device)
        const_hlt = batch["const_hlt"].to(device)
        y = batch["label"].to(device)

        reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)

        hyp_logits: List[torch.Tensor] = []
        for t in hyp_temps:
            if abs(float(t) - 1.0) < 1e-8:
                feat_b, mask_b = mh._build_soft_corrected_view_from_reco_override(
                    reco_out,
                    weight_floor=float(weight_floor),
                    scale_features_by_weight=True,
                    include_flags=bool(include_flags),
                    cand_weights_override=None,
                )
            else:
                w_h = mh._weights_with_temperature(reco_out["cand_weights"], float(t))
                feat_b, mask_b = mh._build_soft_corrected_view_from_reco_override(
                    reco_out,
                    weight_floor=float(weight_floor),
                    scale_features_by_weight=True,
                    include_flags=bool(include_flags),
                    cand_weights_override=w_h,
                )
            logits_h = dual_base_model(feat_hlt_dual, mask_hlt, feat_b, mask_b).squeeze(1)  # [B]
            hyp_logits.append(logits_h)

        logits_stack = torch.stack(hyp_logits, dim=1)  # [B,H]
        if agg_mode == "lse":
            logits_agg = torch.logsumexp(logits_stack, dim=1) - math.log(float(h))
        else:
            logits_agg = logits_stack.mean(dim=1)

        p_h = torch.sigmoid(logits_stack).detach().cpu().numpy().astype(np.float64)
        p_a = torch.sigmoid(logits_agg).detach().cpu().numpy().astype(np.float64)
        yy = y.detach().cpu().numpy().astype(np.float64)

        if p_h.shape[1] != h:
            raise RuntimeError(f"Unexpected hypothesis width: got {p_h.shape[1]}, expected {h}")

        all_p_h.append(p_h)
        all_p_agg.append(p_a)
        all_y.append(yy)

    y = np.concatenate(all_y) if all_y else np.zeros((0,), dtype=np.float64)
    p_agg = np.concatenate(all_p_agg) if all_p_agg else np.zeros((0,), dtype=np.float64)
    p_h = np.concatenate(all_p_h) if all_p_h else np.zeros((0, h), dtype=np.float64)

    if p_h.size > 0:
        # Choose oracle hypothesis index by correctness score:
        # y=1 -> maximize p_h ; y=0 -> maximize (1-p_h) == minimize p_h.
        corr = np.where(y[:, None] > 0.5, p_h, 1.0 - p_h)
        argmax_oracle = corr.argmax(axis=1).astype(np.int64)
        # Oracle output must remain a probability for class-1.
        p_oracle = p_h[np.arange(p_h.shape[0]), argmax_oracle]
    else:
        p_oracle = np.zeros((0,), dtype=np.float64)
        argmax_oracle = np.zeros((0,), dtype=np.int64)

    return {
        "y": y,
        "p_agg": p_agg,
        "p_h": p_h,
        "p_oracle": p_oracle,
        "oracle_h_idx": argmax_oracle,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--stage", type=str, default="joint", choices=["stage2", "joint"])
    ap.add_argument("--reco_ckpt", type=str, default="")
    ap.add_argument("--dual_ckpt", type=str, default="")
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--max_batches", type=int, default=-1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    ap.add_argument("--use_corrected_flags", action="store_true")
    ap.add_argument("--multihyp_temps", type=str, default="0.85,1.00")
    ap.add_argument("--multihyp_agg", type=str, default="mean", choices=["mean", "lse"])
    ap.add_argument("--report_json", type=str, default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    if args.reco_ckpt.strip():
        reco_ckpt = Path(args.reco_ckpt).resolve()
    else:
        reco_ckpt = run_dir / ("offline_reconstructor_stage2.pt" if args.stage == "stage2" else "offline_reconstructor.pt")
    if args.dual_ckpt.strip():
        dual_ckpt = Path(args.dual_ckpt).resolve()
    else:
        dual_ckpt = run_dir / ("dual_joint_stage2.pt" if args.stage == "stage2" else "dual_joint.pt")
    if not reco_ckpt.exists():
        raise FileNotFoundError(f"Missing reco checkpoint: {reco_ckpt}")
    if not dual_ckpt.exists():
        raise FileNotFoundError(f"Missing dual checkpoint: {dual_ckpt}")

    device = torch.device(args.device)
    state_dual = _load_state(dual_ckpt, device)
    state_reco = _load_state(reco_ckpt, device)
    base_state_dual = _extract_base_model_state(state_dual)
    in_a, in_b = _infer_dual_input_dims_from_base_state(base_state_dual)

    if in_a != 7:
        raise RuntimeError(
            f"Dual model expects input_dim_a={in_a}. This diagnostic currently supports 7 only."
        )

    hyp_temps = _parse_temps(args.multihyp_temps)

    data = _prepare_split_arrays(run_dir)
    idx = data[f"{args.split}_idx"]
    feat_hlt_std = data["feat_hlt_std"][idx]
    mask_hlt = data["hlt_mask"][idx]
    const_hlt = data["hlt_const"][idx]
    labels = data["labels"][idx]

    feat_hlt_dual = feat_hlt_std.copy()
    ds = JointLiteDataset(
        feat_hlt_reco=feat_hlt_std,
        feat_hlt_dual=feat_hlt_dual,
        mask_hlt=mask_hlt,
        const_hlt=const_hlt,
        labels=labels,
    )
    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    cfg = base._deepcopy_config()
    reconstructor = base.OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reconstructor = base.wrap_reconstructor_unmerge_only(reconstructor)
    reconstructor.load_state_dict(state_reco, strict=True)

    dual_base_model = mh._ORIG_DUAL_CLASS(
        input_dim_a=int(in_a),
        input_dim_b=int(in_b),
        **cfg["model"],
    ).to(device)
    dual_base_model.load_state_dict(base_state_dual, strict=True)

    out = _collect_predictions(
        reconstructor=reconstructor,
        dual_base_model=dual_base_model,
        loader=dl,
        device=device,
        hyp_temps=hyp_temps,
        agg_mode=str(args.multihyp_agg),
        weight_floor=float(args.corrected_weight_floor),
        include_flags=bool(args.use_corrected_flags),
        max_batches=int(args.max_batches),
    )

    y = out["y"]
    p_agg = out["p_agg"]
    p_h = out["p_h"]
    p_oracle = out["p_oracle"]
    oracle_h_idx = out["oracle_h_idx"]

    h = int(p_h.shape[1]) if p_h.ndim == 2 else len(hyp_temps)

    metrics = {
        "aggregated": _auc_fpr50(y, p_agg),
        "oracle_per_jet": _auc_fpr50(y, p_oracle),
        "per_hypothesis": [],
        "best_single_hyp_by_fpr50": {},
        "best_single_hyp_by_auc": {},
    }

    best_fpr = (float("inf"), -1, float("nan"))
    best_auc = (-float("inf"), -1, float("nan"))
    for i in range(h):
        m = _auc_fpr50(y, p_h[:, i])
        metrics["per_hypothesis"].append({"h_idx": int(i), "auc": m["auc"], "fpr50": m["fpr50"]})
        if np.isfinite(m["fpr50"]) and float(m["fpr50"]) < best_fpr[0]:
            best_fpr = (float(m["fpr50"]), int(i), float(m["auc"]))
        if np.isfinite(m["auc"]) and float(m["auc"]) > best_auc[0]:
            best_auc = (float(m["auc"]), int(i), float(m["fpr50"]))

    if best_fpr[1] >= 0:
        metrics["best_single_hyp_by_fpr50"] = {
            "h_idx": int(best_fpr[1]),
            "fpr50": float(best_fpr[0]),
            "auc": float(best_fpr[2]),
        }
    if best_auc[1] >= 0:
        metrics["best_single_hyp_by_auc"] = {
            "h_idx": int(best_auc[1]),
            "auc": float(best_auc[0]),
            "fpr50": float(best_auc[2]),
        }

    pairwise = []
    if p_h.size > 0:
        for i in range(h):
            for j in range(i + 1, h):
                pi = p_h[:, i]
                pj = p_h[:, j]
                mad = float(np.mean(np.abs(pi - pj)))
                if np.std(pi) > 0 and np.std(pj) > 0:
                    corr = float(np.corrcoef(pi, pj)[0, 1])
                else:
                    corr = float("nan")
                pairwise.append({"i": int(i), "j": int(j), "mean_abs_diff_prob": mad, "corr_prob": corr})

    oracle_counts = np.bincount(oracle_h_idx, minlength=h).astype(np.int64) if oracle_h_idx.size else np.zeros((h,), dtype=np.int64)
    oracle_frac = (oracle_counts / max(int(oracle_counts.sum()), 1)).astype(np.float64)

    report = {
        "run_dir": str(run_dir),
        "stage": str(args.stage),
        "split": str(args.split),
        "reco_ckpt": str(reco_ckpt),
        "dual_ckpt": str(dual_ckpt),
        "device": str(device),
        "n_jets": int(y.shape[0]),
        "input_dim_a": int(in_a),
        "input_dim_b": int(in_b),
        "hyp_temps": [float(t) for t in hyp_temps],
        "multihyp_agg": str(args.multihyp_agg),
        "metrics": metrics,
        "oracle_hypothesis_usage": {
            "counts": oracle_counts.astype(int).tolist(),
            "frac": oracle_frac.tolist(),
        },
        "hypothesis_diversity": {
            "pairwise": pairwise,
            "pairwise_mean_abs_diff_prob_mean": (
                float(np.mean([p["mean_abs_diff_prob"] for p in pairwise])) if pairwise else float("nan")
            ),
            "pairwise_corr_prob_mean": (
                float(np.nanmean([p["corr_prob"] for p in pairwise])) if pairwise else float("nan")
            ),
        },
    }

    print("=" * 60)
    print("MultiHyp Mean Oracle Diagnostic")
    print("=" * 60)
    print(f"run_dir: {run_dir}")
    print(f"stage/split: {args.stage}/{args.split}")
    print(f"n_jets: {report['n_jets']}")
    print(f"hyp_temps: {report['hyp_temps']}, agg={report['multihyp_agg']}")
    print(
        "aggregated: "
        f"AUC={report['metrics']['aggregated']['auc']:.4f}, "
        f"FPR50={report['metrics']['aggregated']['fpr50']:.6f}"
    )
    print(
        "oracle_per_jet: "
        f"AUC={report['metrics']['oracle_per_jet']['auc']:.4f}, "
        f"FPR50={report['metrics']['oracle_per_jet']['fpr50']:.6f}"
    )
    if report["metrics"]["best_single_hyp_by_fpr50"]:
        b = report["metrics"]["best_single_hyp_by_fpr50"]
        print(f"best_single_hyp_by_fpr50: h={b['h_idx']}, AUC={b['auc']:.4f}, FPR50={b['fpr50']:.6f}")
    print(
        "hyp_diversity: "
        f"mean_abs_diff_prob={report['hypothesis_diversity']['pairwise_mean_abs_diff_prob_mean']:.6f}, "
        f"mean_corr_prob={report['hypothesis_diversity']['pairwise_corr_prob_mean']:.6f}"
    )

    report_json = args.report_json.strip()
    if report_json:
        out_path = Path(report_json)
    else:
        out_dir = run_dir / "multihyp_oracle"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.stage}_{args.split}_multihyp_mean_oracle.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report json: {out_path}")


if __name__ == "__main__":
    main()
