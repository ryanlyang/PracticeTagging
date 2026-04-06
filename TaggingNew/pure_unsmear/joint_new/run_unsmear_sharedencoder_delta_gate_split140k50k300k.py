#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Repro-style runner for:
  unsmear_transformer_sharedencoder_origin_undetached.ipynb
with fixed split sizes:
  train=140k, val=50k, test=300k

Intent:
- Keep the notebook method and model choices intact.
- Change only data split counts (and make run non-notebook/sbatch friendly).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict

import h5py
import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import tool
from model import ParticleTransformerKD, SharedEncoderUnsmearClassifier


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _fpr_at_target_tpr(tpr: np.ndarray, fpr: np.ndarray, target_tpr: float) -> float:
    tpr = np.asarray(tpr, dtype=np.float64)
    fpr = np.asarray(fpr, dtype=np.float64)
    order = np.argsort(tpr)
    tpr_sorted = tpr[order]
    fpr_sorted = fpr[order]
    tpr_unique, unique_idx = np.unique(tpr_sorted, return_index=True)
    fpr_unique = fpr_sorted[unique_idx]
    return float(np.interp(float(target_tpr), tpr_unique, fpr_unique))


@torch.no_grad()
def _predict_joint_reco(model, loader, device: torch.device) -> np.ndarray:
    model.eval()
    outs = []
    for batch in loader:
        x = batch["x"].to(device)
        m = batch["mask"].to(device)
        reco, _logits = model(x, m)
        outs.append(reco.detach().cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 0, 0), dtype=np.float32)


def _metric_dict(res_1d: np.ndarray) -> Dict[str, float]:
    if res_1d.size == 0:
        return {
            "bias": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "abs_p50": float("nan"),
            "abs_p90": float("nan"),
            "abs_p99": float("nan"),
        }
    abs_r = np.abs(res_1d)
    return {
        "bias": float(np.mean(res_1d)),
        "mae": float(np.mean(abs_r)),
        "rmse": float(np.sqrt(np.mean(res_1d ** 2))),
        "abs_p50": float(np.quantile(abs_r, 0.50)),
        "abs_p90": float(np.quantile(abs_r, 0.90)),
        "abs_p99": float(np.quantile(abs_r, 0.99)),
    }


def _maybe_wrap_residual(name: str, feat_idx: int, residual: np.ndarray, feat_stds: np.ndarray) -> np.ndarray:
    if name == "dPhi":
        sc = float(feat_stds[feat_idx])
        return tool.wrap_dphi_np(residual * sc) / max(sc, 1e-8)
    return residual


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="test.h5")
    ap.add_argument("--run_name", type=str, default="unsmear_transformer_sharedencoder_delta_gate_joint_split140k50k300k")
    ap.add_argument("--output_root", type=str, default=str(THIS_DIR / "runs"))
    ap.add_argument("--shared_baseline_dir", type=str, default=str(THIS_DIR / "runs" / "shared_offline_hlt_baselines"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_particles", type=int, default=100)
    ap.add_argument("--feature_kind", type=str, default="7d", choices=["3d", "4d", "7d"])

    ap.add_argument("--train_count", type=int, default=140000)
    ap.add_argument("--val_count", type=int, default=50000)
    ap.add_argument("--test_count", type=int, default=300000)

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--early_stop_metric", type=str, default="val_auc_weighted", choices=["val_auc", "val_auc_weighted"])

    ap.add_argument("--load_shared_baselines", type=_str2bool, default=True)
    ap.add_argument("--load_joint_model", type=_str2bool, default=False)

    ap.add_argument("--resmear_each_epoch_baselines", type=_str2bool, default=True)
    ap.add_argument("--resmear_each_epoch_joint", type=_str2bool, default=True)
    ap.add_argument("--resmear_seed_stride", type=int, default=1)

    ap.add_argument("--kd_temperature", type=float, default=2.0)
    ap.add_argument("--kd_alpha", type=float, default=0.5)
    ap.add_argument("--kd_alpha_attn", type=float, default=0.0)

    ap.add_argument("--joint_unsmear_weight", type=float, default=1.6)
    ap.add_argument("--joint_cls_weight", type=float, default=0.8)
    ap.add_argument("--joint_phys_weight", type=float, default=0.0)
    ap.add_argument("--joint_unsmear_loss_mode", type=str, default="mask", choices=["mask", "hungarian"])

    ap.add_argument("--cls_use_delta_fusion", type=_str2bool, default=True)
    ap.add_argument("--cls_detach_delta_for_cls", type=_str2bool, default=False)
    ap.add_argument("--cls_gate_hidden_dim", type=int, default=128)
    ap.add_argument("--cls_gate_init_bias", type=float, default=-2.0)
    ap.add_argument("--cls_alpha_init", type=float, default=0.05)

    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()

    _set_seed(int(args.seed))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA but unavailable; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.output_root) / args.run_name
    fig_dir = out_dir / "figs"
    ckpt_dir = out_dir / "ckpts"
    metrics_dir = out_dir / "metrics"
    shared_baseline_ckpt_dir = Path(args.shared_baseline_dir) / "ckpts"
    tool.ensure_dir(fig_dir)
    tool.ensure_dir(ckpt_dir)
    tool.ensure_dir(metrics_dir)
    tool.ensure_dir(shared_baseline_ckpt_dir)

    feat_names = tool.get_feat_names(args.feature_kind)
    n_total = int(args.train_count + args.val_count + args.test_count)

    config = {
        "data_path": str(args.data_path),
        "n_jets": int(n_total),
        "split_counts": {
            "train": int(args.train_count),
            "val": int(args.val_count),
            "test": int(args.test_count),
        },
        "max_particles": int(args.max_particles),
        "feature_kind": str(args.feature_kind),
        "load_shared_baselines": bool(args.load_shared_baselines),
        "load_joint_model": bool(args.load_joint_model),
        "joint_model": {
            "input_dim": len(feat_names),
            "embed_dim": 128,
            "num_heads": 8,
            "num_layers": 6,
            "ff_dim": 512,
            "dropout": 0.1,
            "unsmear_decoder_layers": 2,
            "unsmear_decoder_heads": 8,
            "unsmear_decoder_ff_dim": 512,
            "unsmear_decoder_dropout": 0.1,
            "return_reco": True,
            "add_mask_channel": False,
            "mask_output": True,
            "use_positional_embedding": False,
            "max_seq_len": int(args.max_particles),
            "cls_use_delta_fusion": bool(args.cls_use_delta_fusion),
            "cls_detach_delta_for_cls": bool(args.cls_detach_delta_for_cls),
            "cls_gate_hidden_dim": int(args.cls_gate_hidden_dim),
            "cls_gate_init_bias": float(args.cls_gate_init_bias),
            "cls_alpha_init": float(args.cls_alpha_init),
        },
        "hlt_effects": {
            "pt_threshold_offline": 0.0,
            "pt_threshold_hlt": 0.0,
            "pt_resolution": 0.10,
            "eta_resolution": 0.03,
            "phi_resolution": 0.03,
        },
        "tagger": {
            "input_dim": len(feat_names),
            "embed_dim": 128,
            "num_heads": 8,
            "num_layers": 6,
            "ff_dim": 512,
            "dropout": 0.1,
        },
        "kd": {
            "enable": True,
            "temperature": float(args.kd_temperature),
            "alpha_kd": float(args.kd_alpha),
            "alpha_attn": float(args.kd_alpha_attn),
        },
        "training": {
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "warmup_epochs": int(args.warmup_epochs),
            "patience": int(args.patience),
            "early_stop_metric": str(args.early_stop_metric),
            "use_sample_weight_for_all_losses": False,
            "joint_unsmear_weight": float(args.joint_unsmear_weight),
            "joint_cls_weight": float(args.joint_cls_weight),
            "joint_phys_weight": float(args.joint_phys_weight),
            "joint_unsmear_loss_mode": str(args.joint_unsmear_loss_mode).strip().lower(),
            "feature_loss_weights": [1.0] * len(feat_names),
            "resmear_each_epoch_baselines": bool(args.resmear_each_epoch_baselines),
            "resmear_each_epoch_joint": bool(args.resmear_each_epoch_joint),
            "resmear_seed_stride": int(args.resmear_seed_stride),
        },
    }

    config_path = out_dir / "config.json"
    tool.save_config(config, config_path)

    print("Device:", device)
    print("Data path:", config["data_path"])
    print("Run dir:", out_dir)
    print("Feature kind:", config["feature_kind"], "feat_names:", feat_names)
    print("Feature loss weights:", dict(zip(feat_names, np.round(config["training"]["feature_loss_weights"], 4))))
    print("Joint physical consistency weight:", float(config["training"]["joint_phys_weight"]))
    print("Joint unsmear loss mode:", str(config["training"]["joint_unsmear_loss_mode"]))
    print("Use sample weight for all losses:", bool(config["training"]["use_sample_weight_for_all_losses"]))
    print("Delta fusion enabled:", bool(config["joint_model"]["cls_use_delta_fusion"]))
    print("Detach delta for classifier:", bool(config["joint_model"]["cls_detach_delta_for_cls"]))
    print("Gate hidden dim:", int(config["joint_model"]["cls_gate_hidden_dim"]))
    print("Gate init bias:", float(config["joint_model"]["cls_gate_init_bias"]))
    print("Alpha init:", float(config["joint_model"]["cls_alpha_init"]))

    h5_path = Path(config["data_path"])
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    S = int(config["max_particles"])
    with h5py.File(h5_path.as_posix(), "r") as f:
        n_avail = int(f["labels"].shape[0])
        if n_avail < n_total:
            raise ValueError(
                f"Requested n_jets={n_total} but file has only {n_avail}."
            )
        labels = f["labels"][:n_total].astype(np.int64)
        weights = f["weights"][:n_total].astype(np.float32)
        pt = f["fjet_clus_pt"][:n_total, :S].astype(np.float32)
        eta = f["fjet_clus_eta"][:n_total, :S].astype(np.float32)
        phi = f["fjet_clus_phi"][:n_total, :S].astype(np.float32)
        E = f["fjet_clus_E"][:n_total, :S].astype(np.float32)

    constituents_raw = np.stack([pt, eta, phi, E], axis=-1)
    mask_raw = pt > 0
    print("Raw:", constituents_raw.shape, "mask:", mask_raw.shape)
    print("Signal:", int(labels.sum()), "Bkg:", int((1 - labels).sum()))

    hcfg = tool.HLTEffectsCfg(**config["hlt_effects"])
    _, hlt_const, hlt_mask = tool.apply_hlt_effects_pair(
        constituents_raw,
        mask_raw,
        hcfg,
        seed=int(args.seed),
    )

    pt_thr_off = float(config["hlt_effects"]["pt_threshold_offline"])
    off_mask = mask_raw & (constituents_raw[:, :, 0] >= pt_thr_off)
    off_const = constituents_raw.copy()
    off_const[~off_mask] = 0.0
    hlt_const = hlt_const.copy()
    hlt_const[~hlt_mask] = 0.0

    axis_off = tool.compute_jet_axis(off_const, off_mask)
    axis_hlt = tool.compute_jet_axis(hlt_const, hlt_mask)
    feat_off = tool.compute_features_with_axis(off_const, off_mask, axis_off, kind=config["feature_kind"])
    feat_hlt = tool.compute_features_with_axis(hlt_const, hlt_mask, axis_hlt, kind=config["feature_kind"])

    idx = np.arange(len(labels))
    train_idx, rest_idx = train_test_split(
        idx,
        train_size=int(args.train_count),
        random_state=int(args.seed),
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        rest_idx,
        train_size=int(args.val_count),
        test_size=int(args.test_count),
        random_state=int(args.seed),
        stratify=labels[rest_idx],
    )
    print(f"Split: train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")

    feat_means, feat_stds = tool.get_stats(feat_off, off_mask, train_idx)
    feat_off_std = tool.standardize(feat_off, off_mask, feat_means, feat_stds, clip=10.0)
    feat_hlt_std = tool.standardize(feat_hlt, hlt_mask, feat_means, feat_stds, clip=10.0)
    common_mask = off_mask & hlt_mask

    x_joint = feat_hlt_std.copy()
    y_joint = feat_off_std.copy()
    x_joint[~common_mask] = 0.0
    y_joint[~common_mask] = 0.0

    train_const_raw = constituents_raw[train_idx]
    train_mask_raw = mask_raw[train_idx]

    print("Offline/HLT feature shape:", feat_off_std.shape, feat_hlt_std.shape)
    print("Mask identical:", bool(np.array_equal(off_mask, hlt_mask)))
    print("Common-mask fraction:", float(common_mask.mean()))
    print("Feat means:", np.round(feat_means, 4))
    print("Feat stds :", np.round(feat_stds, 4))
    print("Baseline epoch resmear enabled:", bool(config["training"].get("resmear_each_epoch_baselines", False)))
    print("Joint epoch resmear enabled:", bool(config["training"].get("resmear_each_epoch_joint", False)))

    BS = int(config["training"]["batch_size"])
    pin = bool(device.type == "cuda")

    train_ds_hlt = tool.JetDataset(
        feat_off_std[train_idx], feat_hlt_std[train_idx], labels[train_idx], off_mask[train_idx], hlt_mask[train_idx], weights[train_idx]
    )
    val_ds_hlt = tool.JetDataset(
        feat_off_std[val_idx], feat_hlt_std[val_idx], labels[val_idx], off_mask[val_idx], hlt_mask[val_idx], weights[val_idx]
    )
    test_ds_hlt = tool.JetDataset(
        feat_off_std[test_idx], feat_hlt_std[test_idx], labels[test_idx], off_mask[test_idx], hlt_mask[test_idx], weights[test_idx]
    )

    train_ds_joint = tool.JointJetDataset(
        x_joint[train_idx], y_joint[train_idx], common_mask[train_idx], labels[train_idx], weights[train_idx]
    )
    val_ds_joint = tool.JointJetDataset(
        x_joint[val_idx], y_joint[val_idx], common_mask[val_idx], labels[val_idx], weights[val_idx]
    )
    test_ds_joint = tool.JointJetDataset(
        x_joint[test_idx], y_joint[test_idx], common_mask[test_idx], labels[test_idx], weights[test_idx]
    )

    train_loader_hlt = DataLoader(train_ds_hlt, batch_size=BS, shuffle=True, drop_last=True, num_workers=int(args.num_workers), pin_memory=pin)
    val_loader_hlt = DataLoader(val_ds_hlt, batch_size=BS, shuffle=False, num_workers=int(args.num_workers), pin_memory=pin)
    test_loader_hlt = DataLoader(test_ds_hlt, batch_size=BS, shuffle=False, num_workers=int(args.num_workers), pin_memory=pin)

    train_loader_joint = DataLoader(train_ds_joint, batch_size=BS, shuffle=True, drop_last=True, num_workers=int(args.num_workers), pin_memory=pin)
    val_loader_joint = DataLoader(val_ds_joint, batch_size=BS, shuffle=False, num_workers=int(args.num_workers), pin_memory=pin)
    test_loader_joint = DataLoader(test_ds_joint, batch_size=BS, shuffle=False, num_workers=int(args.num_workers), pin_memory=pin)

    def make_epoch_hlt_train_loader(epoch: int):
        return tool.make_epoch_hlt_train_loader(
            epoch=int(epoch),
            batch_size=BS,
            feat_off_train=feat_off_std[train_idx],
            off_mask_train=off_mask[train_idx],
            labels_train=labels[train_idx],
            weights_train=weights[train_idx],
            train_const_raw=train_const_raw,
            train_mask_raw=train_mask_raw,
            cfg=hcfg,
            feature_kind=config["feature_kind"],
            means=feat_means,
            stds=feat_stds,
            seed=int(args.seed),
            fixed_feat_hlt_train=feat_hlt_std[train_idx],
            fixed_hlt_mask_train=hlt_mask[train_idx],
            seed_stride=int(config["training"].get("resmear_seed_stride", 1)),
            resmear_each_epoch=bool(config["training"].get("resmear_each_epoch_baselines", False)),
            clip=10.0,
        )

    def make_epoch_joint_train_loader(epoch: int):
        return tool.make_epoch_joint_train_loader(
            epoch=int(epoch),
            batch_size=BS,
            labels_train=labels[train_idx],
            weights_train=weights[train_idx],
            train_const_raw=train_const_raw,
            train_mask_raw=train_mask_raw,
            cfg=hcfg,
            feature_kind=config["feature_kind"],
            means=feat_means,
            stds=feat_stds,
            seed=int(args.seed),
            fixed_x_train=x_joint[train_idx],
            fixed_y_train=y_joint[train_idx],
            fixed_mask_train=common_mask[train_idx],
            seed_stride=int(config["training"].get("resmear_seed_stride", 1)),
            resmear_each_epoch=bool(config["training"].get("resmear_each_epoch_joint", True)),
            clip=10.0,
        )

    train_cfg = config["training"]
    kd_cfg = config["kd"]
    use_sample_weight_for_all_losses = bool(train_cfg.get("use_sample_weight_for_all_losses", True))
    joint_feature_loss_weights = np.asarray(train_cfg["feature_loss_weights"], dtype=np.float32)
    hlt_train_loader_factory = make_epoch_hlt_train_loader if bool(train_cfg.get("resmear_each_epoch_baselines", False)) else None
    joint_train_loader_factory = make_epoch_joint_train_loader if bool(train_cfg.get("resmear_each_epoch_joint", True)) else None

    epoch_metrics_paths = {
        "teacher_off": metrics_dir / "teacher_off_epoch_metrics.csv",
        "student_hlt": metrics_dir / "student_hlt_epoch_metrics.csv",
        "hlt_kd": metrics_dir / "hlt_kd_epoch_metrics.csv",
        "joint_no_kd": metrics_dir / "joint_no_kd_epoch_metrics.csv",
        "joint_with_kd": metrics_dir / "joint_with_kd_epoch_metrics.csv",
    }

    teacher = ParticleTransformerKD(**config["tagger"]).to(device)
    teacher_ckpt = shared_baseline_ckpt_dir / "teacher_offline.pt"
    teacher = tool.train_or_load_standard_model(
        "Teacher(OFF_FULL)", teacher, teacher_ckpt, train_loader_hlt, val_loader_hlt,
        device=device, feat_key="off", mask_key="mask_off",
        allow_load=bool(config.get("load_shared_baselines", True)),
        lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]),
        warmup_epochs=int(train_cfg["warmup_epochs"]), epochs=int(train_cfg["epochs"]),
        patience=int(train_cfg["patience"]), early_stop_metric=str(train_cfg["early_stop_metric"]),
        use_sample_weight_for_all_losses=use_sample_weight_for_all_losses,
        epoch_metrics_path=epoch_metrics_paths["teacher_off"],
    )

    student_hlt = ParticleTransformerKD(**config["tagger"]).to(device)
    student_hlt_ckpt = shared_baseline_ckpt_dir / "student_hlt.pt"
    student_hlt = tool.train_or_load_standard_model(
        "Student(HLT)", student_hlt, student_hlt_ckpt, train_loader_hlt, val_loader_hlt,
        device=device, feat_key="hlt", mask_key="mask_hlt",
        allow_load=bool(config.get("load_shared_baselines", True)),
        lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]),
        warmup_epochs=int(train_cfg["warmup_epochs"]), epochs=int(train_cfg["epochs"]),
        patience=int(train_cfg["patience"]), early_stop_metric=str(train_cfg["early_stop_metric"]),
        use_sample_weight_for_all_losses=use_sample_weight_for_all_losses,
        train_loader_factory=hlt_train_loader_factory,
        epoch_metrics_path=epoch_metrics_paths["student_hlt"],
    )

    student_hlt_kd = ParticleTransformerKD(**config["tagger"]).to(device)
    student_hlt_kd_ckpt = shared_baseline_ckpt_dir / "student_hlt_kd.pt"
    student_hlt_kd = tool.train_or_load_kd_standard_model(
        "Student(HLT)+KD", student_hlt_kd, teacher, student_hlt_kd_ckpt,
        train_loader_hlt, val_loader_hlt,
        device=device,
        allow_load=bool(config.get("load_shared_baselines", True)),
        lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]),
        warmup_epochs=int(train_cfg["warmup_epochs"]), epochs=int(train_cfg["epochs"]),
        patience=int(train_cfg["patience"]), early_stop_metric=str(train_cfg["early_stop_metric"]),
        use_sample_weight_for_all_losses=use_sample_weight_for_all_losses,
        kd_temperature=float(kd_cfg["temperature"]), kd_alpha=float(kd_cfg["alpha_kd"]), kd_alpha_attn=float(kd_cfg["alpha_attn"]),
        train_loader_factory=hlt_train_loader_factory,
        epoch_metrics_path=epoch_metrics_paths["hlt_kd"],
    )

    joint_model_no_kd = SharedEncoderUnsmearClassifier(**config["joint_model"]).to(device)
    joint_ckpt_no_kd = ckpt_dir / "joint_sharedencoder_no_kd.pt"
    joint_model_no_kd = tool.train_or_load_joint_model(
        "JointSharedEncoder(HLT,no_kd)", joint_model_no_kd, joint_ckpt_no_kd,
        train_loader_joint, val_loader_joint,
        device=device,
        feat_names=feat_names, feat_means=feat_means, feat_stds=feat_stds,
        feature_loss_weights=joint_feature_loss_weights,
        joint_phys_weight=float(train_cfg["joint_phys_weight"]),
        joint_unsmear_weight=float(train_cfg["joint_unsmear_weight"]),
        joint_cls_weight=float(train_cfg["joint_cls_weight"]),
        lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]),
        warmup_epochs=int(train_cfg["warmup_epochs"]), epochs=int(train_cfg["epochs"]),
        patience=int(train_cfg["patience"]), early_stop_metric=str(train_cfg["early_stop_metric"]),
        use_sample_weight_for_all_losses=use_sample_weight_for_all_losses,
        teacher=teacher, use_kd=False,
        kd_temperature=float(kd_cfg["temperature"]), kd_alpha=float(kd_cfg["alpha_kd"]), kd_alpha_attn=float(kd_cfg["alpha_attn"]),
        allow_load=bool(config.get("load_joint_model", False)),
        train_loader_factory=joint_train_loader_factory,
        epoch_metrics_path=epoch_metrics_paths["joint_no_kd"],
        unsmear_loss_mode=str(train_cfg.get("joint_unsmear_loss_mode", "mask")),
    )

    joint_model_with_kd = SharedEncoderUnsmearClassifier(**config["joint_model"]).to(device)
    joint_ckpt_with_kd = ckpt_dir / "joint_sharedencoder_with_kd.pt"
    joint_model_with_kd = tool.train_or_load_joint_model(
        "JointSharedEncoder(HLT,with_kd)", joint_model_with_kd, joint_ckpt_with_kd,
        train_loader_joint, val_loader_joint,
        device=device,
        feat_names=feat_names, feat_means=feat_means, feat_stds=feat_stds,
        feature_loss_weights=joint_feature_loss_weights,
        joint_phys_weight=float(train_cfg["joint_phys_weight"]),
        joint_unsmear_weight=float(train_cfg["joint_unsmear_weight"]),
        joint_cls_weight=float(train_cfg["joint_cls_weight"]),
        lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]),
        warmup_epochs=int(train_cfg["warmup_epochs"]), epochs=int(train_cfg["epochs"]),
        patience=int(train_cfg["patience"]), early_stop_metric=str(train_cfg["early_stop_metric"]),
        use_sample_weight_for_all_losses=use_sample_weight_for_all_losses,
        teacher=teacher, use_kd=True,
        kd_temperature=float(kd_cfg["temperature"]), kd_alpha=float(kd_cfg["alpha_kd"]), kd_alpha_attn=float(kd_cfg["alpha_attn"]),
        allow_load=bool(config.get("load_joint_model", False)),
        train_loader_factory=joint_train_loader_factory,
        epoch_metrics_path=epoch_metrics_paths["joint_with_kd"],
        unsmear_loss_mode=str(train_cfg.get("joint_unsmear_loss_mode", "mask")),
    )

    auc_teacher, auc_teacher_w, p_teacher, y_true, w_true = tool.evaluate(
        teacher, test_loader_hlt, device, "off", "mask_off",
        use_sample_weight_for_all_losses=use_sample_weight_for_all_losses,
    )
    auc_hlt, auc_hlt_w, p_hlt, _, _ = tool.evaluate(
        student_hlt, test_loader_hlt, device, "hlt", "mask_hlt",
        use_sample_weight_for_all_losses=use_sample_weight_for_all_losses,
    )
    hlt_kd_test = tool.eval_kd_student(
        student_hlt_kd, teacher, test_loader_hlt, device,
        {"kd": {"temperature": float(kd_cfg["temperature"]), "alpha_kd": float(kd_cfg["alpha_kd"]), "alpha_attn": float(kd_cfg["alpha_attn"]) }},
        use_sample_weight_for_all_losses=use_sample_weight_for_all_losses,
    )
    auc_hlt_kd = float(hlt_kd_test["auc"])
    auc_hlt_kd_w = float(hlt_kd_test["auc_weighted"])
    p_hlt_kd = np.asarray(hlt_kd_test["preds"])

    joint_test_no_kd = tool.eval_joint_model(
        joint_model_no_kd, test_loader_joint, device=device,
        feat_names=feat_names, feat_means=feat_means, feat_stds=feat_stds,
        feature_loss_weights=joint_feature_loss_weights,
        joint_phys_weight=float(train_cfg["joint_phys_weight"]),
        joint_unsmear_weight=float(train_cfg["joint_unsmear_weight"]),
        joint_cls_weight=float(train_cfg["joint_cls_weight"]),
        teacher=teacher, use_kd=False,
        kd_temperature=float(kd_cfg["temperature"]), kd_alpha=float(kd_cfg["alpha_kd"]), kd_alpha_attn=float(kd_cfg["alpha_attn"]),
        use_sample_weight_for_all_losses=use_sample_weight_for_all_losses,
        unsmear_loss_mode=str(train_cfg.get("joint_unsmear_loss_mode", "mask")),
    )
    auc_joint_no_kd = float(joint_test_no_kd["auc"])
    auc_joint_no_kd_w = float(joint_test_no_kd["auc_weighted"])
    p_joint_no_kd = np.asarray(joint_test_no_kd["preds"])
    y_joint_true = np.asarray(joint_test_no_kd["labels"])
    w_joint_true = np.asarray(joint_test_no_kd["weights"])

    joint_test_with_kd = tool.eval_joint_model(
        joint_model_with_kd, test_loader_joint, device=device,
        feat_names=feat_names, feat_means=feat_means, feat_stds=feat_stds,
        feature_loss_weights=joint_feature_loss_weights,
        joint_phys_weight=float(train_cfg["joint_phys_weight"]),
        joint_unsmear_weight=float(train_cfg["joint_unsmear_weight"]),
        joint_cls_weight=float(train_cfg["joint_cls_weight"]),
        teacher=teacher, use_kd=True,
        kd_temperature=float(kd_cfg["temperature"]), kd_alpha=float(kd_cfg["alpha_kd"]), kd_alpha_attn=float(kd_cfg["alpha_attn"]),
        use_sample_weight_for_all_losses=use_sample_weight_for_all_losses,
        unsmear_loss_mode=str(train_cfg.get("joint_unsmear_loss_mode", "mask")),
    )
    auc_joint_with_kd = float(joint_test_with_kd["auc"])
    auc_joint_with_kd_w = float(joint_test_with_kd["auc_weighted"])
    p_joint_with_kd = np.asarray(joint_test_with_kd["preds"])

    print(f"Teacher(OFF_FULL) AUC={auc_teacher:.5f}, weighted AUC={auc_teacher_w:.5f}")
    print(f"Student(HLT) AUC={auc_hlt:.5f}, weighted AUC={auc_hlt_w:.5f}")
    print(f"Student(HLT)+KD AUC={auc_hlt_kd:.5f}, weighted AUC={auc_hlt_kd_w:.5f}")
    print(f"Student(HLT)+KD test total loss={hlt_kd_test['total']:.5f}")
    print(f"Student(HLT)+KD test hard loss={hlt_kd_test['hard']:.5f}")
    print(f"Student(HLT)+KD test kd loss={hlt_kd_test['kd']:.5f}")
    print(f"Student(HLT)+KD test attn loss={hlt_kd_test['attn']:.5f}")
    print(f"JointSharedEncoder(HLT,no_kd) AUC={auc_joint_no_kd:.5f}, weighted AUC={auc_joint_no_kd_w:.5f}")
    print(f"JointSharedEncoder(HLT,with_kd) AUC={auc_joint_with_kd:.5f}, weighted AUC={auc_joint_with_kd_w:.5f}")
    print(f"Joint(no_kd) test unsmear loss={joint_test_no_kd['unsmear_total']:.5f}")
    print(f"Joint(with_kd) test unsmear loss={joint_test_with_kd['unsmear_total']:.5f}")
    print(f"Joint(no_kd) test cls hard loss={joint_test_no_kd['cls_hard_total']:.5f}")
    print(f"Joint(with_kd) test cls hard loss={joint_test_with_kd['cls_hard_total']:.5f}")
    print(f"Joint(with_kd) test cls kd loss={joint_test_with_kd['cls_kd_total']:.5f}")
    print(f"Joint(with_kd) test cls attn loss={joint_test_with_kd['cls_attn_total']:.5f}")
    print(f"Joint(no_kd) gate mean/std={joint_test_no_kd['gate_mean']:.4f}/{joint_test_no_kd['gate_std']:.4f}, alpha={joint_test_no_kd['alpha']:.4f}")
    print(f"Joint(with_kd) gate mean/std={joint_test_with_kd['gate_mean']:.4f}/{joint_test_with_kd['gate_std']:.4f}, alpha={joint_test_with_kd['alpha']:.4f}")

    roc_sample_weight_hlt = w_true if use_sample_weight_for_all_losses else None
    roc_sample_weight_hlt_kd = np.asarray(hlt_kd_test["weights"]) if use_sample_weight_for_all_losses else None
    roc_sample_weight_joint = w_joint_true if use_sample_weight_for_all_losses else None

    fpr_teacher, tpr_teacher, _, _ = tool.compute_roc(y_true, p_teacher, sample_weight=roc_sample_weight_hlt)
    fpr_hlt, tpr_hlt, _, _ = tool.compute_roc(y_true, p_hlt, sample_weight=roc_sample_weight_hlt)
    fpr_hlt_kd, tpr_hlt_kd, _, _ = tool.compute_roc(y_true, p_hlt_kd, sample_weight=roc_sample_weight_hlt_kd)
    fpr_joint_no_kd, tpr_joint_no_kd, _, _ = tool.compute_roc(y_joint_true, p_joint_no_kd, sample_weight=roc_sample_weight_joint)
    fpr_joint_with_kd, tpr_joint_with_kd, _, _ = tool.compute_roc(y_joint_true, p_joint_with_kd, sample_weight=roc_sample_weight_joint)

    ROC_TABLE_RECO_LABEL = "joint_with_kd"
    ROC_TABLE_RECO_NAME = "joint_with_kd"
    roc_table_curves = {
        "teacher": (tpr_teacher, fpr_teacher),
        "hlt_baseline": (tpr_hlt, fpr_hlt),
        "joint_no_kd": (tpr_joint_no_kd, fpr_joint_no_kd),
        "joint_with_kd": (tpr_joint_with_kd, fpr_joint_with_kd),
    }

    roc_table_rows = []
    for target_tpr in [0.30, 0.50]:
        teacher_fpr = _fpr_at_target_tpr(*roc_table_curves["teacher"], target_tpr)
        baseline_fpr = _fpr_at_target_tpr(*roc_table_curves["hlt_baseline"], target_tpr)
        reco_fpr = _fpr_at_target_tpr(*roc_table_curves[ROC_TABLE_RECO_NAME], target_tpr)
        denom = baseline_fpr - teacher_fpr
        recovery = np.nan if abs(denom) < 1e-12 else (baseline_fpr - reco_fpr) / denom
        roc_table_rows.append(
            {
                "Metric": f"FPR @ {int(round(target_tpr * 100))}%TPR",
                "Teacher": teacher_fpr,
                "HLT Baseline": baseline_fpr,
                ROC_TABLE_RECO_LABEL: reco_fpr,
                "Offline-HLT Gap Recovery": recovery,
            }
        )

    roc_operating_df = pd.DataFrame(roc_table_rows)
    roc_operating_df_display = roc_operating_df.copy()
    for col in ["Teacher", "HLT Baseline", ROC_TABLE_RECO_LABEL]:
        roc_operating_df_display[col] = roc_operating_df_display[col].map(lambda x: f"{100.0 * float(x):.4f}%")
    roc_operating_df_display["Offline-HLT Gap Recovery"] = roc_operating_df_display["Offline-HLT Gap Recovery"].map(
        lambda x: "nan" if pd.isna(x) else f"{100.0 * float(x):.2f}%"
    )

    plt.figure(figsize=(7, 6))
    plt.semilogy(tpr_teacher, np.clip(fpr_teacher, 1e-6, 1.0), lw=2.2, label=f"Teacher(OFF_FULL) AUC={auc_teacher:.4f}, wAUC={auc_teacher_w:.4f}")
    plt.semilogy(tpr_hlt, np.clip(fpr_hlt, 1e-6, 1.0), lw=2.0, label=f"Student(HLT) AUC={auc_hlt:.4f}, wAUC={auc_hlt_w:.4f}")
    plt.semilogy(tpr_hlt_kd, np.clip(fpr_hlt_kd, 1e-6, 1.0), lw=2.0, label=f"Student(HLT)+KD AUC={auc_hlt_kd:.4f}, wAUC={auc_hlt_kd_w:.4f}")
    plt.semilogy(tpr_joint_no_kd, np.clip(fpr_joint_no_kd, 1e-6, 1.0), lw=2.0, label=f"JointSharedEncoder(HLT,no_kd) AUC={auc_joint_no_kd:.4f}, wAUC={auc_joint_no_kd_w:.4f}")
    plt.semilogy(tpr_joint_with_kd, np.clip(fpr_joint_with_kd, 1e-6, 1.0), lw=2.0, label=f"JointSharedEncoder(HLT,with_kd) AUC={auc_joint_with_kd:.4f}, wAUC={auc_joint_with_kd_w:.4f}")
    plt.xlabel("True Positive Rate (Signal efficiency)")
    plt.ylabel("False Positive Rate")
    plt.title("Shared-encoder joint training ROC (test)")
    plt.xlim(0.0, 1.0)
    plt.ylim(1e-4, 1.0)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()

    roc_out = fig_dir / "sharedencoder_joint_downstream_roc_logfpr_5lines.png"
    plt.savefig(roc_out, dpi=180, bbox_inches="tight")
    plt.close()
    print("Saved figure:", roc_out)

    roc_table_out = out_dir / f"roc_operating_points_{ROC_TABLE_RECO_NAME}.csv"
    roc_operating_df.to_csv(roc_table_out, index=False)
    print("Saved table:", roc_table_out)
    print(roc_operating_df_display.to_string(index=False))

    # Unsmear reconstruction residual diagnostics (matches notebook behavior).
    pred_joint_no_kd_reco = _predict_joint_reco(joint_model_no_kd, test_loader_joint, device)
    pred_joint_with_kd_reco = _predict_joint_reco(joint_model_with_kd, test_loader_joint, device)

    x_test_std = x_joint[test_idx]
    y_test_std = y_joint[test_idx]
    mask_test = common_mask[test_idx]

    residual_sources = {
        "hlt": x_test_std - y_test_std,
        "joint_no_kd": pred_joint_no_kd_reco - y_test_std,
        "joint_with_kd": pred_joint_with_kd_reco - y_test_std,
    }
    plot_labels = {
        "hlt": "HLT baseline (post - pre)",
        "joint_no_kd": "Joint no KD (pred - pre)",
        "joint_with_kd": "Joint with KD (pred - pre)",
    }
    plot_colors = {
        "hlt": "#4C78A8",
        "joint_no_kd": "#F58518",
        "joint_with_kd": "#54A24B",
    }

    metrics_rows = []
    for feat_idx, feat_name in enumerate(feat_names):
        plt.figure(figsize=(6.6, 4.6))
        for method_name in ["hlt", "joint_no_kd", "joint_with_kd"]:
            residual = residual_sources[method_name][..., feat_idx][mask_test]
            residual = _maybe_wrap_residual(feat_name, feat_idx, residual, feat_stds)
            plt.hist(
                residual,
                bins=120,
                density=True,
                alpha=0.35,
                label=plot_labels[method_name],
                color=plot_colors[method_name],
            )
            mm = _metric_dict(residual)
            metrics_rows.append({"feature": feat_name, "method": method_name, **mm})

        plt.title(f"Residual compare: {feat_name}")
        plt.xlabel("Residual (std space)")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        out = fig_dir / f"joint_reco_residual_compare_{feat_name}.png"
        plt.savefig(out, dpi=160, bbox_inches="tight")
        plt.close()
        print("Saved figure:", out)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df[["feature", "method", "bias", "mae", "rmse", "abs_p50", "abs_p90", "abs_p99"]]

    print()
    print("=" * 100)
    print(f"Metrics summary (std space) | split=test | n_tokens={int(mask_test.sum())}")
    print("=" * 100)
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    metrics_out = out_dir / "joint_reco_metrics_summary_test.csv"
    metrics_df.to_csv(metrics_out, index=False)
    print("Saved table:", metrics_out)

    # Gate/alpha evolution plot from epoch tables.
    joint_metric_paths = {
        "joint_no_kd": epoch_metrics_paths["joint_no_kd"],
        "joint_with_kd": epoch_metrics_paths["joint_with_kd"],
    }
    joint_metric_frames = {}
    for model_name, metric_path in joint_metric_paths.items():
        if not Path(metric_path).is_file():
            print(f"Missing epoch metrics for {model_name}: {metric_path}")
            continue
        df = pd.read_csv(metric_path)
        if "epoch" not in df.columns:
            print(f"Skip {model_name}: missing epoch column in {metric_path}")
            continue
        joint_metric_frames[model_name] = df.sort_values("epoch").reset_index(drop=True)

    if joint_metric_frames:
        plot_colors = {
            "joint_no_kd": "#F58518",
            "joint_with_kd": "#54A24B",
        }

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
        for model_name, df in joint_metric_frames.items():
            x = df["epoch"].to_numpy()
            color = plot_colors.get(model_name, None)

            if {"train_gate_mean", "train_gate_std"}.issubset(df.columns):
                y = df["train_gate_mean"].to_numpy(dtype=float)
                y_std = df["train_gate_std"].to_numpy(dtype=float)
                axes[0].plot(x, y, marker="o", ms=4, lw=2.0, color=color, label=f"{model_name} train mean")
                axes[0].fill_between(x, np.clip(y - y_std, 0.0, 1.0), np.clip(y + y_std, 0.0, 1.0), color=color, alpha=0.16)

            if {"val_gate_mean", "val_gate_std"}.issubset(df.columns):
                y = df["val_gate_mean"].to_numpy(dtype=float)
                y_std = df["val_gate_std"].to_numpy(dtype=float)
                axes[0].plot(x, y, marker="s", ms=4, lw=1.8, ls="--", color=color, alpha=0.95, label=f"{model_name} val mean")
                axes[0].fill_between(x, np.clip(y - y_std, 0.0, 1.0), np.clip(y + y_std, 0.0, 1.0), color=color, alpha=0.08)

            if "alpha" in df.columns:
                axes[1].plot(x, df["alpha"], marker="o", ms=4, lw=2.0, color=color, label=model_name)

        axes[0].set_title("Gate mean with std band")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Gate value")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(fontsize=8)

        axes[1].set_title("Alpha across epochs")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Alpha")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(fontsize=8)

        fig.suptitle("Delta-fusion gate diagnostics")
        fig.tight_layout()
        gate_diag_out = fig_dir / "joint_gate_alpha_over_epochs.png"
        plt.savefig(gate_diag_out, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print("Saved figure:", gate_diag_out)

    summary = {
        "run_name": args.run_name,
        "data_path": str(h5_path),
        "seed": int(args.seed),
        "split_counts": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "auc": {
            "teacher_off": float(auc_teacher),
            "hlt": float(auc_hlt),
            "hlt_kd": float(auc_hlt_kd),
            "joint_no_kd": float(auc_joint_no_kd),
            "joint_with_kd": float(auc_joint_with_kd),
        },
        "auc_weighted": {
            "teacher_off": float(auc_teacher_w),
            "hlt": float(auc_hlt_w),
            "hlt_kd": float(auc_hlt_kd_w),
            "joint_no_kd": float(auc_joint_no_kd_w),
            "joint_with_kd": float(auc_joint_with_kd_w),
        },
    }
    summary_path = out_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
