#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot JetClass `feature_mode=full` constituent features (17 dims) by class.

Outputs:
1) Overlay distribution grid (17 subplots, one per feature, split by class)
2) Per-feature per-class summary stats CSV
3) Simple metadata JSON with run settings and class counts

This script reuses the same loader + feature builder used by training so that
plots correspond exactly to model inputs.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


FEATURE_NAMES_FULL17_LEGACY = [
    "d_eta",
    "d_phi",
    "log_pt",
    "log_e",
    "log_pt_rel",
    "log_e_rel",
    "d_r",
    "charge",
    "isChargedHadron",
    "isNeutralHadron",
    "isPhoton",
    "isElectron",
    "isMuon",
    "d0",
    "d0err",
    "dz",
    "dzerr",
]

FEATURE_NAMES_FULL17_CANONICAL = [
    "part_pt_log",
    "part_e_log",
    "part_logptrel",
    "part_logerel",
    "part_deltaR",
    "charge",
    "isChargedHadron",
    "isNeutralHadron",
    "isPhoton",
    "isElectron",
    "isMuon",
    "d0_tanh",
    "d0err_clip",
    "dz_tanh",
    "dzerr_clip",
    "d_eta",
    "d_phi",
]

DISCRETE_BINARY_FEATURES = {
    "isChargedHadron",
    "isNeutralHadron",
    "isPhoton",
    "isElectron",
    "isMuon",
}

SPIKE_HEAVY_FEATURES = {"d0", "dz", "d0_tanh", "dz_tanh"}
TOP_KINEMATIC_FEATURES = [
    "part_pt_log",
    "part_e_log",
    "part_logptrel",
    "part_logerel",
    "raw_dR",
    "d_eta",
    "d_phi",
]
TOP_PID_FEATURES = [
    "charge",
    "isChargedHadron",
    "isNeutralHadron",
    "isPhoton",
    "isElectron",
    "isMuon",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot JetClass full-17 feature distributions by class")
    p.add_argument("--data_dir", type=Path, default=Path("data/jetclass_part0"))
    p.add_argument("--output_dir", type=Path, default=Path("plots/jetclass_full17_features"))
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument(
        "--class_assignment",
        type=str,
        default="canonical_labels",
        choices=["filename", "canonical_labels"],
        help="Use filename classes or canonical label_* branches for class assignment.",
    )
    p.add_argument(
        "--feature_preprocessing",
        type=str,
        default="canonical",
        choices=["canonical", "legacy"],
        help="Feature preprocessing style before plotting.",
    )
    p.add_argument("--seed", type=int, default=52)
    p.add_argument("--max_constits", type=int, default=128)
    p.add_argument("--n_jets", type=int, default=50000)
    p.add_argument("--train_files_per_class", type=int, default=8)
    p.add_argument("--val_files_per_class", type=int, default=1)
    p.add_argument("--test_files_per_class", type=int, default=1)
    p.add_argument("--shuffle_files", action="store_true", default=False)
    p.add_argument("--bins", type=int, default=80)
    p.add_argument(
        "--max_constits_per_class",
        type=int,
        default=250000,
        help="Cap plotted constituents per class for speed/readability",
    )
    p.add_argument(
        "--clip_quantile_low",
        type=float,
        default=0.5,
        help="Lower quantile (percent) used for plotting range per feature",
    )
    p.add_argument(
        "--clip_quantile_high",
        type=float,
        default=99.5,
        help="Upper quantile (percent) used for plotting range per feature",
    )
    p.add_argument(
        "--extra_raw_dr_plot",
        action="store_true",
        default=False,
        help="Also save a physics-facing raw nonnegative dR plot by class.",
    )
    p.add_argument(
        "--top_constituent_only",
        action="store_true",
        default=False,
        help="Also save leading-pT constituent feature plots (one constituent per jet).",
    )
    return p.parse_args()


def choose_split(
    files_by_class: Dict[str, List[Path]],
    split_files_by_class_fn,
    n_train: int,
    n_val: int,
    n_test: int,
    shuffle: bool,
    seed: int,
    split_name: str,
) -> Dict[str, List[Path]]:
    tr, va, te = split_files_by_class_fn(
        files_by_class=files_by_class,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        shuffle=shuffle,
        seed=seed,
    )
    if split_name == "train":
        return tr
    if split_name == "val":
        return va
    return te


def downsample(arr: np.ndarray, max_n: int, rng: np.random.RandomState) -> np.ndarray:
    if arr.size <= max_n:
        return arr
    idx = rng.choice(arr.size, size=max_n, replace=False)
    return arr[idx]


def collect_feature_values_by_class(
    feat: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    max_constits_per_class: int,
    seed: int,
) -> Dict[str, List[np.ndarray]]:
    rng = np.random.RandomState(seed + 101)
    out: Dict[str, List[np.ndarray]] = {
        cls: [np.zeros((0,), dtype=np.float32) for _ in range(feat.shape[-1])]
        for cls in class_names
    }
    for i, cls in enumerate(class_names):
        class_sel = labels == i
        if not np.any(class_sel):
            continue
        f_cls = feat[class_sel]
        m_cls = mask[class_sel]
        for j in range(feat.shape[-1]):
            vals = f_cls[:, :, j][m_cls]
            vals = vals.astype(np.float32, copy=False)
            vals = downsample(vals, max_constits_per_class, rng)
            out[cls][j] = vals
    return out


def collect_raw_dr_by_class(
    raw_tok: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    max_constits_per_class: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Collect raw constituent dR >= 0 with respect to the jet axis."""
    baseline = importlib.import_module("evaluate_jetclass_hlt_teacher_baseline")
    pt = np.maximum(raw_tok[:, :, baseline.IDX_PT], 1e-8)
    eta = np.clip(raw_tok[:, :, baseline.IDX_ETA], -5.0, 5.0)
    phi = raw_tok[:, :, baseline.IDX_PHI]

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    w = mask.astype(np.float32)
    jet_px = (px * w).sum(axis=1, keepdims=True)
    jet_py = (py * w).sum(axis=1, keepdims=True)
    jet_pz = (pz * w).sum(axis=1, keepdims=True)
    jet_p = np.sqrt(jet_px * jet_px + jet_py * jet_py + jet_pz * jet_pz) + 1e-8
    jet_eta = 0.5 * np.log(np.clip((jet_p + jet_pz) / np.maximum(jet_p - jet_pz, 1e-8), 1e-8, 1e8))
    jet_phi = np.arctan2(jet_py, jet_px)

    d_eta = eta - jet_eta
    d_phi = np.arctan2(np.sin(phi - jet_phi), np.cos(phi - jet_phi))
    d_r = np.sqrt(d_eta * d_eta + d_phi * d_phi).astype(np.float32)
    d_r[~mask] = np.nan

    rng = np.random.RandomState(seed + 211)
    out: Dict[str, np.ndarray] = {}
    for i, cls in enumerate(class_names):
        vals = d_r[labels == i]
        vals = vals[np.isfinite(vals)]
        vals = vals.astype(np.float32, copy=False)
        vals = downsample(vals, max_constits_per_class, rng)
        out[cls] = vals
    return out


def compute_raw_dr(raw_tok: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Raw constituent dR >= 0 with respect to the jet axis."""
    baseline = importlib.import_module("evaluate_jetclass_hlt_teacher_baseline")
    pt = np.maximum(raw_tok[:, :, baseline.IDX_PT], 1e-8)
    eta = np.clip(raw_tok[:, :, baseline.IDX_ETA], -5.0, 5.0)
    phi = raw_tok[:, :, baseline.IDX_PHI]

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    w = mask.astype(np.float32)
    jet_px = (px * w).sum(axis=1, keepdims=True)
    jet_py = (py * w).sum(axis=1, keepdims=True)
    jet_pz = (pz * w).sum(axis=1, keepdims=True)
    jet_p = np.sqrt(jet_px * jet_px + jet_py * jet_py + jet_pz * jet_pz) + 1e-8
    jet_eta = 0.5 * np.log(np.clip((jet_p + jet_pz) / np.maximum(jet_p - jet_pz, 1e-8), 1e-8, 1e8))
    jet_phi = np.arctan2(jet_py, jet_px)

    d_eta = eta - jet_eta
    d_phi = np.arctan2(np.sin(phi - jet_phi), np.cos(phi - jet_phi))
    d_r = np.sqrt(d_eta * d_eta + d_phi * d_phi).astype(np.float32)
    d_r[~mask] = np.nan
    return d_r


def collect_top_constituent_values_by_class(
    raw_tok: np.ndarray,
    feat: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    feature_names: List[str],
) -> Tuple[Dict[str, List[np.ndarray]], List[str]]:
    baseline = importlib.import_module("evaluate_jetclass_hlt_teacher_baseline")
    valid_pt = np.where(mask, raw_tok[:, :, baseline.IDX_PT], -np.inf)
    top_idx = np.argmax(valid_pt, axis=1)
    rows = np.arange(raw_tok.shape[0])

    raw_dr = compute_raw_dr(raw_tok, mask)
    top_feat = feat[rows, top_idx, :].astype(np.float32, copy=False)
    top_raw_dr = raw_dr[rows, top_idx].astype(np.float32, copy=False)

    top_feature_names = list(feature_names)
    if "part_deltaR" in top_feature_names:
        dr_idx = top_feature_names.index("part_deltaR")
        top_feature_names[dr_idx] = "raw_dR"
        top_feat[:, dr_idx] = top_raw_dr

    out: Dict[str, List[np.ndarray]] = {
        cls: [np.zeros((0,), dtype=np.float32) for _ in range(len(top_feature_names))]
        for cls in class_names
    }
    for i, cls in enumerate(class_names):
        sel = labels == i
        if not np.any(sel):
            continue
        for j in range(len(top_feature_names)):
            out[cls][j] = top_feat[sel, j]
    return out, top_feature_names


def draw_raw_dr_plot(
    out_png: Path,
    out_pdf: Path,
    vals_by_class: Dict[str, np.ndarray],
    class_names: List[str],
    bins: int,
    q_high: float,
) -> None:
    import matplotlib.pyplot as plt

    pooled = np.concatenate([vals_by_class[c] for c in class_names if vals_by_class[c].size > 0], axis=0)
    xhi = float(np.percentile(pooled, q_high))
    xhi = max(xhi, 1e-3)
    bin_edges = np.linspace(0.0, xhi, int(bins) + 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    legend_handles = []
    legend_labels = []
    for c_idx, cls in enumerate(class_names):
        x = vals_by_class[cls]
        x = x[(x >= 0.0) & (x <= xhi)]
        if x.size == 0:
            continue
        h = ax.hist(
            x,
            bins=bin_edges,
            histtype="step",
            density=True,
            linewidth=1.3,
            color=colors[c_idx],
            alpha=0.95,
            label=cls,
        )
        legend_handles.append(h[2][0])
        legend_labels.append(cls)

    ax.set_xlim(0.0, xhi)
    ax.set_xlabel("raw dR")
    ax.set_ylabel("Density")
    ax.set_title("Raw Nonnegative dR by class")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    if legend_handles:
        ax.legend(legend_handles, legend_labels, frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def robust_range(values: np.ndarray, q_low: float, q_high: float) -> Tuple[float, float]:
    if values.size == 0:
        return -1.0, 1.0
    lo = float(np.percentile(values, q_low))
    hi = float(np.percentile(values, q_high))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0
    if hi <= lo:
        eps = 1e-3 if abs(lo) < 1.0 else 1e-3 * abs(lo)
        return lo - eps, hi + eps
    return lo, hi


def draw_feature_on_axis(
    ax,
    feat_name: str,
    feat_idx: int,
    vals_by_class: Dict[str, List[np.ndarray]],
    class_names: List[str],
    colors: np.ndarray,
    bins: int,
    q_low: float,
    q_high: float,
    show_class_legend: bool = False,
    show_charge_legend: bool = False,
):
    """Draw one feature on a given axis using continuous/discrete-appropriate style."""
    if feat_name == "charge":
        xloc = np.arange(len(class_names), dtype=np.float32)
        frac_neg, frac_zero, frac_pos = [], [], []
        for cls in class_names:
            x = vals_by_class[cls][feat_idx]
            if x.size == 0:
                frac_neg.append(0.0)
                frac_zero.append(0.0)
                frac_pos.append(0.0)
                continue
            q = np.rint(x).astype(np.int32, copy=False)
            q = np.clip(q, -1, 1)
            n = float(q.size)
            frac_neg.append(float(np.sum(q == -1)) / n)
            frac_zero.append(float(np.sum(q == 0)) / n)
            frac_pos.append(float(np.sum(q == 1)) / n)

        h1 = ax.bar(xloc, frac_neg, width=0.8, color="#d73027", alpha=0.95, label="q = -1")
        h2 = ax.bar(xloc, frac_zero, width=0.8, bottom=np.array(frac_neg), color="#bdbdbd", alpha=0.95, label="q = 0")
        h3 = ax.bar(
            xloc,
            frac_pos,
            width=0.8,
            bottom=np.array(frac_neg) + np.array(frac_zero),
            color="#4575b4",
            alpha=0.95,
            label="q = +1",
        )
        # Add headroom so text + legend do not sit on top of bars.
        ax.set_ylim(0.0, 1.32)
        ax.set_ylabel("Fraction of Constituents")
        ax.set_xticks(xloc)
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")
        # Add explicit percentages for charge composition per class.
        for i in range(len(class_names)):
            txt = (
                f"-1: {100.0*frac_neg[i]:.1f}%\n"
                f" 0: {100.0*frac_zero[i]:.1f}%\n"
                f"+1: {100.0*frac_pos[i]:.1f}%"
            )
            ax.text(
                xloc[i],
                1.03,
                txt,
                ha="center",
                va="bottom",
                fontsize=6,
                linespacing=0.95,
            )
        if show_charge_legend:
            ax.legend(
                handles=[h1, h2, h3],
                frameon=False,
                fontsize=9,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.9,
            )
        return [], []

    if feat_name in DISCRETE_BINARY_FEATURES:
        xloc = np.arange(len(class_names), dtype=np.float32)
        pos_rate = []
        for cls in class_names:
            x = vals_by_class[cls][feat_idx]
            if x.size == 0:
                pos_rate.append(0.0)
                continue
            pos_rate.append(float(np.mean(x > 0.5)))

        bars = ax.bar(xloc, pos_rate, width=0.8, color=colors, alpha=0.9)
        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("P(feature = 1)")
        ax.set_xticks(xloc)
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")
        # Add percentage labels on top of each class bar.
        for bar, p in zip(bars, pos_rate):
            ax.text(
                bar.get_x() + bar.get_width() * 0.5,
                min(1.04, p + 0.02),
                f"{100.0*p:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        return [], []

    pooled = np.concatenate([vals_by_class[c][feat_idx] for c in class_names if vals_by_class[c][feat_idx].size > 0], axis=0)
    xlo, xhi = robust_range(pooled, q_low=q_low, q_high=q_high)
    bin_edges = np.linspace(xlo, xhi, int(bins) + 1)

    handles, labels = [], []
    for c_idx, cls in enumerate(class_names):
        x = vals_by_class[cls][feat_idx]
        if x.size == 0:
            continue
        keep = (x >= xlo) & (x <= xhi)
        x_plot = x[keep]
        if x_plot.size == 0:
            continue
        h = ax.hist(
            x_plot,
            bins=bin_edges,
            histtype="step",
            density=True,
            linewidth=1.2,
            color=colors[c_idx],
            alpha=0.95,
            label=cls,
        )
        if show_class_legend:
            handles.append(h[2][0])
            labels.append(cls)

    ax.set_xlim(xlo, xhi)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    if feat_name in SPIKE_HEAVY_FEATURES:
        # d0/dz typically have a sharp near-zero spike; log-y reveals tails.
        ymin, ymax = ax.get_ylim()
        ax.set_yscale("log")
        ax.set_ylim(max(ymin, 1e-3), ymax)
        ax.axvline(0.0, color="k", linestyle=":", linewidth=0.8, alpha=0.7)
    return handles, labels


def write_stats_csv(
    path: Path,
    vals_by_class: Dict[str, List[np.ndarray]],
    feature_names: List[str],
) -> None:
    fields = [
        "feature",
        "class",
        "count",
        "mean",
        "std",
        "min",
        "p01",
        "p05",
        "p25",
        "p50",
        "p75",
        "p95",
        "p99",
        "max",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cls, lst in vals_by_class.items():
            for k, name in enumerate(feature_names):
                x = lst[k]
                if x.size == 0:
                    row = {c: "" for c in fields}
                    row["feature"] = name
                    row["class"] = cls
                    row["count"] = 0
                    writer.writerow(row)
                    continue
                row = {
                    "feature": name,
                    "class": cls,
                    "count": int(x.size),
                    "mean": float(np.mean(x)),
                    "std": float(np.std(x)),
                    "min": float(np.min(x)),
                    "p01": float(np.percentile(x, 1)),
                    "p05": float(np.percentile(x, 5)),
                    "p25": float(np.percentile(x, 25)),
                    "p50": float(np.percentile(x, 50)),
                    "p75": float(np.percentile(x, 75)),
                    "p95": float(np.percentile(x, 95)),
                    "p99": float(np.percentile(x, 99)),
                    "max": float(np.max(x)),
                }
                writer.writerow(row)


def draw_summary_means_by_class(
    out_png: Path,
    out_pdf: Path,
    vals_by_class: Dict[str, List[np.ndarray]],
    feature_names: List[str],
    class_names: List[str],
) -> None:
    import matplotlib.pyplot as plt

    n_feat = len(feature_names)
    ncols = 4
    nrows = int(np.ceil(n_feat / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(22, 4.2 * nrows))
    axes = np.atleast_1d(axes).flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    for j, feat_name in enumerate(feature_names):
        ax = axes[j]
        summary_vals = []
        ylabel = "Mean"
        for cls in class_names:
            x = vals_by_class[cls][j]
            if x.size == 0:
                summary_vals.append(0.0)
            elif feat_name in DISCRETE_BINARY_FEATURES:
                summary_vals.append(float(np.mean(x > 0.5)))
                ylabel = "Fraction"
            else:
                summary_vals.append(float(np.mean(x)))
        xpos = np.arange(len(class_names), dtype=np.float32)
        bars = ax.bar(xpos, summary_vals, color=colors, alpha=0.9)
        ax.set_title(feat_name, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_xticks(xpos)
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")
        for bar, v in zip(bars, summary_vals):
            ax.text(
                bar.get_x() + 0.5 * bar.get_width(),
                v,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    for k in range(n_feat, nrows * ncols):
        axes[k].axis("off")

    fig.suptitle("Top-constituent summary statistic by class", fontsize=14)
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.97])
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def assemble_feature_panel(
    feature_dir: Path,
    feature_names: List[str],
    selected_features: List[str],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    files = []
    for idx, name in enumerate(feature_names, start=1):
        if name in selected_features:
            files.append((idx, name, feature_dir / f"{idx:02d}_{name}.png"))
    if not files:
        return

    n = len(files)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11 * ncols, 6.2 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (_, _, img_path) in zip(axes, files):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis("off")
    for ax in axes[len(files):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=[0.01, 0.01, 1.0, 0.97])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt

    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Import lazily so --help works even if awkward/uproot are missing in shell env.
    baseline = importlib.import_module("evaluate_jetclass_hlt_teacher_baseline")
    collect_files_by_class = baseline.collect_files_by_class
    canonical_class_order = baseline.CANONICAL_CLASS_ORDER
    split_files_by_class = baseline.split_files_by_class
    load_split = baseline.load_split
    compute_features = baseline.compute_features

    files_by_class = collect_files_by_class(args.data_dir)
    split_files = choose_split(
        files_by_class=files_by_class,
        split_files_by_class_fn=split_files_by_class,
        n_train=int(args.train_files_per_class),
        n_val=int(args.val_files_per_class),
        n_test=int(args.test_files_per_class),
        shuffle=bool(args.shuffle_files),
        seed=int(args.seed),
        split_name=str(args.split),
    )
    if str(args.class_assignment) == "canonical_labels":
        class_names = list(canonical_class_order)
    else:
        class_names = sorted(split_files.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    print(f"Loading split={args.split} with n_jets={args.n_jets} from {args.data_dir}")
    raw_tok, mask, jet_labels = load_split(
        split_files=split_files,
        n_total=int(args.n_jets),
        max_constits=int(args.max_constits),
        class_to_idx=class_to_idx,
        seed=int(args.seed) + 7,
        class_assignment=str(args.class_assignment),
    )

    feat = compute_features(
        raw_tok,
        mask,
        feature_mode="full",
        feature_preprocessing=str(args.feature_preprocessing),
    )
    feature_names = (
        FEATURE_NAMES_FULL17_CANONICAL
        if str(args.feature_preprocessing) == "canonical"
        else FEATURE_NAMES_FULL17_LEGACY
    )
    if feat.shape[-1] != len(feature_names):
        raise RuntimeError(
            f"Expected 17 full features, got {feat.shape[-1]}. "
            "Feature builder may have changed."
        )

    vals_by_class = collect_feature_values_by_class(
        feat=feat,
        mask=mask,
        labels=jet_labels,
        class_names=class_names,
        max_constits_per_class=int(args.max_constits_per_class),
        seed=int(args.seed),
    )

    # Write stats table first so results are useful even if plotting fails.
    stats_csv = args.output_dir / "full17_feature_stats_by_class.csv"
    write_stats_csv(stats_csv, vals_by_class, feature_names)

    # Plot grid: 17 panels (5x4, last blank)
    n_feat = len(feature_names)
    ncols = 4
    nrows = 5
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(22, 22))
    axes = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    legend_handles = []
    legend_labels = []
    per_feature_dir = args.output_dir / "per_feature_png"
    per_feature_dir.mkdir(parents=True, exist_ok=True)

    print("Building distribution plots...")
    for j, feat_name in enumerate(feature_names):
        ax = axes[j]
        handles, legend_lbls = draw_feature_on_axis(
            ax=ax,
            feat_name=feat_name,
            feat_idx=j,
            vals_by_class=vals_by_class,
            class_names=class_names,
            colors=colors,
            bins=int(args.bins),
            q_low=float(args.clip_quantile_low),
            q_high=float(args.clip_quantile_high),
            show_class_legend=(len(legend_handles) == 0),
            show_charge_legend=False,
        )
        if handles and legend_lbls and len(legend_handles) == 0:
            legend_handles = handles
            legend_labels = legend_lbls

        ax.set_title(feat_name, fontsize=12)

        # Save one dedicated PNG per feature (easy for presentations/email).
        fig_single, ax_single = plt.subplots(figsize=(10.5, 6.0))
        draw_feature_on_axis(
            ax=ax_single,
            feat_name=feat_name,
            feat_idx=j,
            vals_by_class=vals_by_class,
            class_names=class_names,
            colors=colors,
            bins=int(args.bins),
            q_low=float(args.clip_quantile_low),
            q_high=float(args.clip_quantile_high),
            show_class_legend=(feat_name not in DISCRETE_BINARY_FEATURES and feat_name != "charge"),
            show_charge_legend=True,
        )
        ax_single.set_title(f"{feat_name} by class", fontsize=13)
        if feat_name not in DISCRETE_BINARY_FEATURES and feat_name != "charge":
            ax_single.legend(frameon=False, fontsize=8, ncol=2)
        fig_single.tight_layout()
        out_single = per_feature_dir / f"{j + 1:02d}_{feat_name}.png"
        fig_single.savefig(out_single, dpi=180, bbox_inches="tight")
        plt.close(fig_single)

    # Hide unused subplot slot.
    for k in range(n_feat, nrows * ncols):
        axes[k].axis("off")

    fig.suptitle(
        (
            f"JetClass Full-17 Constituent Feature Distributions by Class\n"
            f"split={args.split}, jets={args.n_jets}, max_constits={args.max_constits}, "
            f"constituent cap/class={args.max_constits_per_class}, "
            f"plot range={args.clip_quantile_low}-{args.clip_quantile_high} percentiles"
        ),
        fontsize=14,
    )
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(5, len(class_names)),
            frameon=False,
            fontsize=10,
            bbox_to_anchor=(0.5, 0.03),
        )
    fig.tight_layout(rect=[0.02, 0.07, 1.0, 0.95])

    out_png = args.output_dir / "full17_feature_distributions_by_class.png"
    out_pdf = args.output_dir / "full17_feature_distributions_by_class.pdf"
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_pdf)
    plt.close(fig)

    raw_dr_png = None
    raw_dr_pdf = None
    if bool(args.extra_raw_dr_plot):
        raw_dr_vals = collect_raw_dr_by_class(
            raw_tok=raw_tok,
            mask=mask,
            labels=jet_labels,
            class_names=class_names,
            max_constits_per_class=int(args.max_constits_per_class),
            seed=int(args.seed),
        )
        raw_dr_png = args.output_dir / "raw_dR_by_class.png"
        raw_dr_pdf = args.output_dir / "raw_dR_by_class.pdf"
        draw_raw_dr_plot(
            out_png=raw_dr_png,
            out_pdf=raw_dr_pdf,
            vals_by_class=raw_dr_vals,
            class_names=class_names,
            bins=int(args.bins),
            q_high=float(args.clip_quantile_high),
        )

    top_outputs = {}
    if bool(args.top_constituent_only):
        top_vals_by_class, top_feature_names = collect_top_constituent_values_by_class(
            raw_tok=raw_tok,
            feat=feat,
            mask=mask,
            labels=jet_labels,
            class_names=class_names,
            feature_names=feature_names,
        )

        top_stats_csv = args.output_dir / "topconst_feature_stats_by_class.csv"
        write_stats_csv(top_stats_csv, top_vals_by_class, top_feature_names)

        top_per_feature_dir = args.output_dir / "topconst_per_feature_png"
        top_per_feature_dir.mkdir(parents=True, exist_ok=True)
        top_plot_png = args.output_dir / "topconst_feature_distributions_by_class.png"
        top_plot_pdf = args.output_dir / "topconst_feature_distributions_by_class.pdf"
        top_summary_png = args.output_dir / "topconst_summary_mean_by_class.png"
        top_summary_pdf = args.output_dir / "topconst_summary_mean_by_class.pdf"
        top_raw_dr_png = args.output_dir / "topconst_raw_dR_by_class.png"
        top_raw_dr_pdf = args.output_dir / "topconst_raw_dR_by_class.pdf"

        top_fig, top_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(22, 22))
        top_axes = top_axes.flatten()
        top_legend_handles = []
        top_legend_labels = []
        for j, feat_name in enumerate(top_feature_names):
            ax = top_axes[j]
            handles, legend_lbls = draw_feature_on_axis(
                ax=ax,
                feat_name=feat_name,
                feat_idx=j,
                vals_by_class=top_vals_by_class,
                class_names=class_names,
                colors=colors,
                bins=int(args.bins),
                q_low=float(args.clip_quantile_low),
                q_high=float(args.clip_quantile_high),
                show_class_legend=(len(top_legend_handles) == 0),
                show_charge_legend=False,
            )
            if handles and legend_lbls and len(top_legend_handles) == 0:
                top_legend_handles = handles
                top_legend_labels = legend_lbls
            ax.set_title(feat_name, fontsize=12)

            fig_single, ax_single = plt.subplots(figsize=(10.5, 6.0))
            draw_feature_on_axis(
                ax=ax_single,
                feat_name=feat_name,
                feat_idx=j,
                vals_by_class=top_vals_by_class,
                class_names=class_names,
                colors=colors,
                bins=int(args.bins),
                q_low=float(args.clip_quantile_low),
                q_high=float(args.clip_quantile_high),
                show_class_legend=(feat_name not in DISCRETE_BINARY_FEATURES and feat_name != "charge"),
                show_charge_legend=True,
            )
            ax_single.set_title(f"Top-constituent {feat_name} by class", fontsize=13)
            if feat_name not in DISCRETE_BINARY_FEATURES and feat_name != "charge":
                ax_single.legend(frameon=False, fontsize=8, ncol=2)
            fig_single.tight_layout()
            out_single = top_per_feature_dir / f"{j + 1:02d}_{feat_name}.png"
            fig_single.savefig(out_single, dpi=180, bbox_inches="tight")
            plt.close(fig_single)

        for k in range(len(top_feature_names), nrows * ncols):
            top_axes[k].axis("off")

        top_fig.suptitle(
            (
                f"JetClass Top-Constituent Feature Distributions by Class\n"
                f"leading-pT constituent only, split={args.split}, jets={args.n_jets}"
            ),
            fontsize=14,
        )
        if top_legend_handles and top_legend_labels:
            top_fig.legend(
                top_legend_handles,
                top_legend_labels,
                loc="lower center",
                ncol=min(5, len(class_names)),
                frameon=False,
                fontsize=10,
                bbox_to_anchor=(0.5, 0.03),
            )
        top_fig.tight_layout(rect=[0.02, 0.07, 1.0, 0.95])
        top_fig.savefig(top_plot_png, dpi=180)
        top_fig.savefig(top_plot_pdf)
        plt.close(top_fig)

        draw_summary_means_by_class(
            out_png=top_summary_png,
            out_pdf=top_summary_pdf,
            vals_by_class=top_vals_by_class,
            feature_names=top_feature_names,
            class_names=class_names,
        )
        draw_raw_dr_plot(
            out_png=top_raw_dr_png,
            out_pdf=top_raw_dr_pdf,
            vals_by_class={cls: top_vals_by_class[cls][top_feature_names.index("raw_dR")] for cls in class_names},
            class_names=class_names,
            bins=int(args.bins),
            q_high=float(args.clip_quantile_high),
        )

        top_kin_panel = args.output_dir / "topconst_kinematic_panel.png"
        top_pid_panel = args.output_dir / "topconst_particleid_panel.png"
        assemble_feature_panel(
            feature_dir=top_per_feature_dir,
            feature_names=top_feature_names,
            selected_features=TOP_KINEMATIC_FEATURES,
            out_path=top_kin_panel,
            title="Top-constituent kinematic features by class",
        )
        assemble_feature_panel(
            feature_dir=top_per_feature_dir,
            feature_names=top_feature_names,
            selected_features=TOP_PID_FEATURES,
            out_path=top_pid_panel,
            title="Top-constituent particle-ID features by class",
        )

        top_outputs = {
            "top_plot_png": str(top_plot_png),
            "top_plot_pdf": str(top_plot_pdf),
            "top_stats_csv": str(top_stats_csv),
            "top_per_feature_png_dir": str(top_per_feature_dir),
            "top_summary_png": str(top_summary_png),
            "top_summary_pdf": str(top_summary_pdf),
            "top_raw_dr_png": str(top_raw_dr_png),
            "top_raw_dr_pdf": str(top_raw_dr_pdf),
            "top_kinematic_panel_png": str(top_kin_panel),
            "top_particleid_panel_png": str(top_pid_panel),
            "top_feature_names": top_feature_names,
        }

    # Save metadata for reproducibility.
    meta = {
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "split": str(args.split),
        "class_assignment": str(args.class_assignment),
        "feature_preprocessing": str(args.feature_preprocessing),
        "seed": int(args.seed),
        "n_jets": int(args.n_jets),
        "max_constits": int(args.max_constits),
        "max_constits_per_class": int(args.max_constits_per_class),
        "train_files_per_class": int(args.train_files_per_class),
        "val_files_per_class": int(args.val_files_per_class),
        "test_files_per_class": int(args.test_files_per_class),
        "shuffle_files": bool(args.shuffle_files),
        "bins": int(args.bins),
        "clip_quantile_low": float(args.clip_quantile_low),
        "clip_quantile_high": float(args.clip_quantile_high),
        "class_names": class_names,
        "class_jet_counts": {
            cls: int(np.sum(jet_labels == i)) for i, cls in enumerate(class_names)
        },
        "feature_names_full17": feature_names,
        "stats_csv": str(stats_csv),
        "plot_png": str(out_png),
        "plot_pdf": str(out_pdf),
        "per_feature_png_dir": str(per_feature_dir),
        "raw_dr_plot_png": str(raw_dr_png) if raw_dr_png is not None else None,
        "raw_dr_plot_pdf": str(raw_dr_pdf) if raw_dr_pdf is not None else None,
        "top_constituent_only": bool(args.top_constituent_only),
    }
    meta.update(top_outputs)
    with (args.output_dir / "run_metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")
    print(
        "Saved:\n"
        f"- {out_png}\n"
        f"- {out_pdf}\n"
        f"- {stats_csv}\n"
        f"- {per_feature_dir} (17 per-feature PNGs)\n"
        f"- {args.output_dir / 'run_metadata.json'}"
    )


if __name__ == "__main__":
    main()
