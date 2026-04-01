#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze reconstructor constraint saturation on train/val/test split for a saved run.

What it reports:
- action distribution sharpness (entropy, near-uniform, highly-peaked rates)
- log-pt/log-E clamp pressure (pre-clamp exceedance rates)
- reassign tanh saturation pressure (|raw| > threshold)
- split-angle tanh saturation pressure
- split-existence sigmoid saturation pressure
- budget calibration clamp pressure (merge/eff scale hitting low/high clamp)
- split child mass utilization stats
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base


EPS = 1e-8


class NpJetDataset(Dataset):
    def __init__(self, feat_hlt: np.ndarray, mask_hlt: np.ndarray, const_hlt: np.ndarray):
        self.feat_hlt = feat_hlt
        self.mask_hlt = mask_hlt
        self.const_hlt = const_hlt

    def __len__(self) -> int:
        return int(self.feat_hlt.shape[0])

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.feat_hlt[idx], dtype=torch.float32),
            torch.tensor(self.mask_hlt[idx], dtype=torch.bool),
            torch.tensor(self.const_hlt[idx], dtype=torch.float32),
        )


def _load_checkpoint_model_state(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def _auto_reco_class_from_run_name(run_name: str) -> Optional[Tuple[str, str]]:
    rn = str(run_name).lower()
    mapping = [
        ("trueposterior", ("offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_trueposterior", "OfflineReconstructorTruePosterior")),
        ("anglecap035", ("offline_reconstructor_no_gt_local30kv2_anglecap035", "OfflineReconstructorAngleCap035")),
        ("budgetclamp6", ("offline_reconstructor_no_gt_local30kv2_budgetclamp6", "OfflineReconstructorBudgetClamp6")),
        ("splitk3_softgate", ("offline_reconstructor_no_gt_local30kv2_splitk3_softgate", "OfflineReconstructorSplitK3SoftGate")),
        ("splitcap120", ("offline_reconstructor_no_gt_local30kv2_splitcap120", "OfflineReconstructorSplitCap120")),
        ("ptclamp10", ("offline_reconstructor_no_gt_local30kv2_ptclamp10", "OfflineReconstructorPtClamp10")),
        ("splitmass_softmax", ("offline_reconstructor_no_gt_local30kv2_splitmass_softmax", "OfflineReconstructorSplitMassSoftmax")),
        ("splitctx_heads", ("offline_reconstructor_no_gt_local30kv2_splitctx_heads", "OfflineReconstructorSplitCtx")),
        ("splitheads_only", ("offline_reconstructor_no_gt_local30kv2_splitheads_only", "OfflineReconstructorSplitHeadsOnly")),
        ("ctxfilm_concat", ("offline_reconstructor_no_gt_local30kv2_ctxfilm_concat", "OfflineReconstructorCtxFilmConcat")),
        ("actiongates", ("offline_reconstructor_no_gt_local30kv2_actiongates", "OfflineReconstructorActionGates")),
        ("factorized_edit", ("offline_reconstructor_no_gt_local30kv2_factorized_edit", "OfflineReconstructorFactorizedEdit")),
        ("linear_unsmear", ("offline_reconstructor_no_gt_local30kv2_linear_unsmear", "OfflineReconstructorLinearUnsmear")),
    ]
    for key, val in mapping:
        if key in rn:
            return val
    return None


def _build_reconstructor(
    device: torch.device,
    reco_module: Optional[str],
    reco_class: Optional[str],
    run_name_hint: str,
    apply_unmerge_wrap: bool = True,
):
    cfg = base._deepcopy_config()

    module_name = reco_module
    class_name = reco_class
    if module_name is None or class_name is None:
        auto = _auto_reco_class_from_run_name(run_name_hint)
        if auto is not None:
            module_name, class_name = auto

    if module_name and class_name:
        mod = importlib.import_module(module_name)
        reco_cls = getattr(mod, class_name)
    else:
        reco_cls = base.OfflineReconstructor

    model = reco_cls(input_dim=7, **cfg["reconstructor_model"]).to(device)
    if apply_unmerge_wrap:
        model = base.wrap_reconstructor_unmerge_only(model)
    return model


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
        raise RuntimeError(f"Not enough jets in files ({all_const_full.shape[0]} < {max_jets_needed})")

    const_raw = all_const_full[offset_jets : offset_jets + n_train_jets]
    _ = all_labels_full[offset_jets : offset_jets + n_train_jets]

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
    feat_hlt_std = base.standardize(feat_hlt, hlt_mask, means, stds)

    out = {
        "feat_hlt_std": feat_hlt_std.astype(np.float32),
        "mask_hlt": hlt_mask.astype(np.bool_),
        "const_hlt": hlt_const.astype(np.float32),
        "train_idx": split_npz["train_idx"].astype(np.int64),
        "val_idx": split_npz["val_idx"].astype(np.int64),
        "test_idx": split_npz["test_idx"].astype(np.int64),
    }
    return out


class HookCache:
    def __init__(self, model):
        self.out: Dict[str, torch.Tensor] = {}
        self.handles = []
        self._try_register(model, "action_head")
        self._try_register(model, "unsmear_head")
        self._try_register(model, "reassign_head")
        self._try_register(model, "split_exist_head")
        self._try_register(model, "gen_exist_head")

    def _try_register(self, model, name: str):
        module = getattr(model, name, None)
        if module is None:
            return

        def _hook(_m, _inp, out):
            if torch.is_tensor(out):
                self.out[name] = out.detach()

        self.handles.append(module.register_forward_hook(_hook))

    def clear(self):
        self.out.clear()

    def close(self):
        for h in self.handles:
            h.remove()


def _mean_std_from_vals(vals: list[float]) -> Tuple[float, float]:
    if len(vals) == 0:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=-1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--stage_scale", type=float, default=1.0)
    p.add_argument("--diag_logpt_min", type=float, default=-9.0)
    p.add_argument("--diag_logpt_max", type=float, default=9.0)
    p.add_argument("--diag_loge_min", type=float, default=-9.0)
    p.add_argument("--diag_loge_max", type=float, default=11.0)
    p.add_argument("--reco_module", type=str, default="")
    p.add_argument("--reco_class", type=str, default="")
    p.add_argument("--no_unmerge_wrap", action="store_true")
    p.add_argument("--report_json", type=str, default="")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    device = torch.device(args.device)

    if args.checkpoint.strip():
        ckpt_path = Path(args.checkpoint).resolve()
    else:
        ckpt_path = run_dir / "offline_reconstructor.pt"
        if not ckpt_path.exists():
            ckpt_path = run_dir / "offline_reconstructor_stage2.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ds = _prepare_split_arrays(run_dir)
    idx_key = f"{args.split}_idx"
    idx = ds[idx_key]
    feat = ds["feat_hlt_std"][idx]
    mask = ds["mask_hlt"][idx]
    const = ds["const_hlt"][idx]

    with open(run_dir / "data_setup.json", "r", encoding="utf-8") as f:
        data_setup = json.load(f)
    run_name_hint = str(data_setup.get("run_name", run_dir.name))

    model = _build_reconstructor(
        device=device,
        reco_module=(args.reco_module.strip() or None),
        reco_class=(args.reco_class.strip() or None),
        run_name_hint=run_name_hint,
        apply_unmerge_wrap=(not bool(args.no_unmerge_wrap)),
    )
    state = _load_checkpoint_model_state(ckpt_path, device)
    model.load_state_dict(state, strict=False)
    model.eval()

    hooks = HookCache(model)
    loader = DataLoader(
        NpJetDataset(feat, mask, const),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(max(args.num_workers, 0)),
        pin_memory=(device.type == "cuda"),
    )

    n_tokens = 0.0
    n_jets = 0.0
    action_entropy_sum = 0.0
    action_uniform_sum = 0.0
    action_peak_sum = 0.0

    clamp_logpt_hi = 0.0
    clamp_logpt_lo = 0.0
    clamp_loge_hi = 0.0
    clamp_loge_lo = 0.0

    reassign_sat_sum = 0.0
    reassign_weighted_sat_sum = 0.0
    reassign_den = 0.0

    split_angle_sat_num = 0.0
    split_angle_sat_den = 0.0
    split_exist_sig_sat_num = 0.0
    split_exist_sig_sat_den = 0.0

    merge_scale_hi = 0.0
    merge_scale_lo = 0.0
    merge_scale_valid_jets = 0.0
    eff_scale_hi = 0.0
    eff_scale_lo = 0.0
    eff_scale_valid_jets = 0.0

    split_total_frac_vals: list[float] = []
    split_total_frac_p95_vals: list[float] = []

    max_batches = int(args.max_batches)
    stage_scale = float(args.stage_scale)
    logpt_min = float(args.diag_logpt_min)
    logpt_max = float(args.diag_logpt_max)
    loge_min = float(args.diag_loge_min)
    loge_max = float(args.diag_loge_max)
    with torch.no_grad():
        for bi, (feat_b, mask_b, const_b) in enumerate(loader):
            if max_batches > 0 and bi >= max_batches:
                break
            feat_b = feat_b.to(device, non_blocking=True)
            mask_b = mask_b.to(device, non_blocking=True)
            const_b = const_b.to(device, non_blocking=True)
            hooks.clear()
            out = model(feat_b, mask_b, const_b, stage_scale=stage_scale)

            action_prob = out["action_prob"].clamp(min=1e-8)
            mask_f = mask_b.float()
            tok = float(mask_f.sum().item())
            jets = float(mask_b.shape[0])
            if tok <= 0:
                continue
            n_tokens += tok
            n_jets += jets

            ent = -(action_prob * torch.log(action_prob)).sum(dim=-1) / np.log(action_prob.shape[-1])
            maxp = action_prob.max(dim=-1).values
            action_entropy_sum += float((ent * mask_f).sum().item())
            action_uniform_sum += float((((maxp < 0.45).float() * mask_f).sum().item()))
            action_peak_sum += float((((maxp > 0.90).float() * mask_f).sum().item()))

            p_unsmear = action_prob[..., 1]
            p_reassign = action_prob[..., 3]
            p_split = action_prob[..., 2]

            unsmear_raw = hooks.out.get("unsmear_head", None)
            if unsmear_raw is not None:
                d_logpt = stage_scale * (p_unsmear + 0.5 * p_reassign) * unsmear_raw[..., 0]
                pre_logpt = torch.log(const_b[..., 0].clamp(min=EPS)) + d_logpt
                clamp_logpt_hi += float((((pre_logpt > logpt_max).float() * mask_f).sum().item()))
                clamp_logpt_lo += float((((pre_logpt < logpt_min).float() * mask_f).sum().item()))

                d_loge = stage_scale * (p_unsmear + 0.5 * p_reassign) * unsmear_raw[..., 3]
                pre_loge = torch.log(const_b[..., 3].clamp(min=EPS)) + d_loge
                clamp_loge_hi += float((((pre_loge > loge_max).float() * mask_f).sum().item()))
                clamp_loge_lo += float((((pre_loge < loge_min).float() * mask_f).sum().item()))

            reassign_raw = hooks.out.get("reassign_head", None)
            if reassign_raw is not None:
                sat = (reassign_raw.abs() > 2.0).float()
                m = mask_f.unsqueeze(-1).expand_as(sat)
                w_reassign = p_reassign.unsqueeze(-1).expand_as(sat)
                reassign_sat_sum += float((sat * m).sum().item())
                reassign_den += float(m.sum().item())
                reassign_weighted_sat_sum += float((sat * m * w_reassign).sum().item())

            split_exist_raw = hooks.out.get("split_exist_head", None)
            gen_exist_raw = hooks.out.get("gen_exist_head", None)
            split_delta = out.get("split_delta", None)
            if split_exist_raw is not None:
                split_exist_prob = torch.sigmoid(split_exist_raw)
                sat_sig = ((split_exist_prob < 0.02) | (split_exist_prob > 0.98)).float()
                m = mask_f.unsqueeze(-1).expand_as(sat_sig)
                split_exist_sig_sat_num += float((sat_sig * m).sum().item())
                split_exist_sig_sat_den += float(m.sum().item())

                split_exist_pre = split_exist_prob * (p_split.unsqueeze(-1) * stage_scale) * mask_f.unsqueeze(-1)
                child_sum_pre = split_exist_pre.reshape(split_exist_pre.shape[0], -1).sum(dim=1)
                budget_merge = out["budget_merge"].reshape(-1)
                merge_valid = (child_sum_pre > EPS).float()
                raw_scale_merge = budget_merge / (child_sum_pre + EPS)
                merge_scale_hi += float((((raw_scale_merge > 4.0).float()) * merge_valid).sum().item())
                merge_scale_lo += float((((raw_scale_merge < 0.25).float()) * merge_valid).sum().item())
                merge_scale_valid_jets += float(merge_valid.sum().item())

                if split_delta is not None:
                    sat_ang = (split_delta[..., 1:].abs() > 2.0).float()
                    split_angle_sat_num += float((sat_ang * split_exist_pre.unsqueeze(-1)).sum().item())
                    split_angle_sat_den += float(split_exist_pre.sum().item() * 2.0)

            if gen_exist_raw is not None:
                # Hook output shape is [B, G, 1] for nn.Linear(..., 1); squeeze to [B, G]
                # to avoid accidental [B, B] broadcasting in downstream division.
                gen_pre = torch.sigmoid(gen_exist_raw.squeeze(-1)) * stage_scale
                gen_sum_pre = gen_pre.sum(dim=1)
                budget_eff = out["budget_eff"].reshape(-1)
                eff_valid = (gen_sum_pre > EPS).float()
                raw_scale_eff = budget_eff / (gen_sum_pre + EPS)
                eff_scale_hi += float((((raw_scale_eff > 4.0).float()) * eff_valid).sum().item())
                eff_scale_lo += float((((raw_scale_eff < 0.25).float()) * eff_valid).sum().item())
                eff_scale_valid_jets += float(eff_valid.sum().item())

            # Split child mass utilization diagnostics.
            L = int(action_prob.shape[1])
            n_child = int(out["child_weight"].shape[1])
            if n_child > 0 and L > 0 and n_child % L == 0:
                K = int(n_child // L)
                child_tok = out["cand_tokens"][:, L : L + n_child, :]
                child_pt = child_tok[:, :, 0].reshape(child_tok.shape[0], L, K)
                parent_pt = const_b[:, :, 0].clamp(min=EPS)
                total_frac = (child_pt.sum(dim=2) / parent_pt).clamp(min=0.0, max=5.0)
                total_frac = total_frac[mask_b]
                if total_frac.numel() > 0:
                    split_total_frac_vals.append(float(total_frac.mean().item()))
                    split_total_frac_p95_vals.append(float(torch.quantile(total_frac, 0.95).item()))

    hooks.close()

    if n_tokens <= 0:
        raise RuntimeError("No valid masked tokens processed.")

    report = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "split": str(args.split),
        "unmerge_wrap_applied": bool(not args.no_unmerge_wrap),
        "diag_logpt_min": logpt_min,
        "diag_logpt_max": logpt_max,
        "diag_loge_min": loge_min,
        "diag_loge_max": loge_max,
        "n_jets_processed": int(n_jets),
        "n_masked_tokens": int(n_tokens),
        "action_entropy_mean": action_entropy_sum / n_tokens,
        "action_near_uniform_frac": action_uniform_sum / n_tokens,
        "action_high_peak_frac": action_peak_sum / n_tokens,
        "clamp_logpt_hi_frac": clamp_logpt_hi / n_tokens,
        "clamp_logpt_lo_frac": clamp_logpt_lo / n_tokens,
        "clamp_loge_hi_frac": clamp_loge_hi / n_tokens,
        "clamp_loge_lo_frac": clamp_loge_lo / n_tokens,
        "reassign_raw_tanh_sat_frac": (reassign_sat_sum / max(reassign_den, EPS)),
        "reassign_raw_tanh_sat_weighted_by_p_reassign": (
            reassign_weighted_sat_sum / max(reassign_den, EPS)
        ),
        "split_exist_sigmoid_sat_frac": (split_exist_sig_sat_num / max(split_exist_sig_sat_den, EPS)),
        "split_angle_raw_tanh_sat_weighted_frac": (split_angle_sat_num / max(split_angle_sat_den, EPS)),
        "merge_budget_scale_clamp_hi_frac_jets": (
            merge_scale_hi / max(merge_scale_valid_jets, EPS) if merge_scale_valid_jets > 0 else float("nan")
        ),
        "merge_budget_scale_clamp_lo_frac_jets": (
            merge_scale_lo / max(merge_scale_valid_jets, EPS) if merge_scale_valid_jets > 0 else float("nan")
        ),
        "merge_budget_scale_valid_jets": int(merge_scale_valid_jets),
        "eff_budget_scale_clamp_hi_frac_jets": (
            eff_scale_hi / max(eff_scale_valid_jets, EPS) if eff_scale_valid_jets > 0 else float("nan")
        ),
        "eff_budget_scale_clamp_lo_frac_jets": (
            eff_scale_lo / max(eff_scale_valid_jets, EPS) if eff_scale_valid_jets > 0 else float("nan")
        ),
        "eff_budget_scale_valid_jets": int(eff_scale_valid_jets),
        "split_total_frac_mean_token": _mean_std_from_vals(split_total_frac_vals)[0],
        "split_total_frac_p95_token": _mean_std_from_vals(split_total_frac_p95_vals)[0],
    }

    print("============================================================")
    print("Reconstructor Constraint Saturation Report")
    print("============================================================")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    if args.report_json.strip():
        out_p = Path(args.report_json).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report json: {out_p}")


if __name__ == "__main__":
    main()
