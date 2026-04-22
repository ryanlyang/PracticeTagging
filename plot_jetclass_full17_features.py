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
    }
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
