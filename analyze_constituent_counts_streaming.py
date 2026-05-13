#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streaming constituent-count analysis for:
  1) JetClass ROOT files stored inside a .tar archive
  2) ATLAS top-tagging train.h5

What it does:
  - Loads jets in manageable batches (not all at once).
  - Maintains running class-wise averages of constituent counts.
  - Continues until target jet count is reached (default: 200k) per dataset.
  - Produces bar plots:
      * JetClass: average constituents by class
      * train.h5: average constituents by class (Background, Top)
      * equivalent comparison: JetClass vs train.h5 (Top-like, Background-like)

Notes:
  - For JetClass top-equivalent, this script uses TTBar.
  - For JetClass background-equivalent, this script uses all NON-top classes
    (all classes except TTBar and TTBarLep), combined as a weighted average.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_JETCLASS_FILE_RE = re.compile(r"^(?P<cls>[A-Za-z0-9]+)_(?P<idx>\d{3})\.root$")


@dataclass
class RunningStat:
    n: int = 0
    s: float = 0.0

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        self.n += int(values.size)
        self.s += float(values.sum())

    @property
    def mean(self) -> float:
        if self.n <= 0:
            return float("nan")
        return self.s / float(self.n)


def _require_mod(name: str):
    try:
        return __import__(name)
    except Exception as exc:
        raise RuntimeError(
            f"Required dependency '{name}' is not available in this Python environment."
        ) from exc


def _distribute_quota(total: int, classes: Sequence[str]) -> Dict[str, int]:
    n = len(classes)
    base = total // n
    rem = total % n
    out: Dict[str, int] = {}
    for i, c in enumerate(classes):
        out[c] = base + (1 if i < rem else 0)
    return out


def _first_ttree_name(uproot_file) -> str:
    for k, v in uproot_file.classnames().items():
        if v == "TTree":
            # keys in uproot.classnames() include cycle numbers, e.g. "tree;1"
            return k
    raise RuntimeError("No TTree found in ROOT file.")


def _pick_branch(tree, candidates: Sequence[str]) -> str:
    keys = set(str(k) for k in tree.keys())
    for c in candidates:
        if c in keys:
            return c
    raise RuntimeError(
        f"None of the candidate branches found. Tried: {list(candidates)}"
    )


def _load_jetclass_counts_from_root_member(
    tf: tarfile.TarFile,
    member_name: str,
    max_jets: int,
    step_size: int,
    tmp_dir: Path,
) -> np.ndarray:
    """
    Extract one ROOT member to temp, stream constituent counts from one branch.
    Returns up to `max_jets` counts for that file.
    """
    uproot = _require_mod("uproot")
    ak = _require_mod("awkward")

    member = tf.getmember(member_name)
    extracted = tf.extractfile(member)
    if extracted is None:
        return np.zeros((0,), dtype=np.int32)

    tmp_path = tmp_dir / Path(member_name).name
    with tmp_path.open("wb") as f_out:
        shutil.copyfileobj(extracted, f_out)

    out_chunks: List[np.ndarray] = []
    taken = 0
    branch_candidates = (
        "part_pt",
        "part_px",  # fallback: count jagged lengths from px
    )

    with uproot.open(str(tmp_path)) as f:
        ttree = f[_first_ttree_name(f)]
        branch = _pick_branch(ttree, branch_candidates)
        for arrays in ttree.iterate([branch], step_size=step_size, library="ak"):
            counts = ak.to_numpy(ak.num(arrays[branch], axis=1)).astype(np.int32)
            if counts.size == 0:
                continue
            need = max_jets - taken
            if need <= 0:
                break
            counts = counts[:need]
            out_chunks.append(counts)
            taken += int(counts.size)
            if taken >= max_jets:
                break

    if not out_chunks:
        return np.zeros((0,), dtype=np.int32)
    return np.concatenate(out_chunks, axis=0)


def stream_jetclass_constituent_averages(
    tar_path: Path,
    target_jets: int,
    per_file_max_jets: int,
    step_size: int,
    verbose: bool = True,
) -> Dict[str, RunningStat]:
    """
    Stream JetClass counts from tar file in small chunks.
    Class-balanced target (roughly equal jets per class).
    """
    if target_jets <= 0:
        raise ValueError("target_jets must be > 0")
    if per_file_max_jets <= 0:
        raise ValueError("per_file_max_jets must be > 0")

    with tarfile.open(tar_path, "r") as tf:
        root_members = [
            m.name for m in tf.getmembers() if m.isfile() and m.name.endswith(".root")
        ]
        if not root_members:
            raise RuntimeError(f"No ROOT files found in tar: {tar_path}")

        by_class: Dict[str, List[str]] = {}
        for m in sorted(root_members):
            base = Path(m).name
            match = _JETCLASS_FILE_RE.match(base)
            if not match:
                continue
            cls = match.group("cls")
            by_class.setdefault(cls, []).append(m)

        classes = sorted(by_class.keys())
        if not classes:
            raise RuntimeError("Could not parse class names from ROOT member names.")

        quotas = _distribute_quota(target_jets, classes)
        remain = dict(quotas)
        file_idx = {c: 0 for c in classes}
        stats = {c: RunningStat() for c in classes}

        with tempfile.TemporaryDirectory(prefix="jetclass_tar_stream_") as tmp:
            tmp_dir = Path(tmp)
            while any(v > 0 for v in remain.values()):
                progressed = False
                for cls in classes:
                    if remain[cls] <= 0:
                        continue
                    members = by_class[cls]
                    m = members[file_idx[cls] % len(members)]
                    file_idx[cls] += 1

                    take_now = min(remain[cls], per_file_max_jets)
                    counts = _load_jetclass_counts_from_root_member(
                        tf=tf,
                        member_name=m,
                        max_jets=take_now,
                        step_size=step_size,
                        tmp_dir=tmp_dir,
                    )
                    if counts.size == 0:
                        continue

                    stats[cls].update(counts)
                    remain[cls] -= int(counts.size)
                    progressed = True

                    if verbose:
                        done = quotas[cls] - max(remain[cls], 0)
                        print(
                            f"[JetClass] cls={cls:12s} loaded={done}/{quotas[cls]} "
                            f"running_mean={stats[cls].mean:.3f}"
                        )
                if not progressed:
                    raise RuntimeError(
                        "No progress while streaming JetClass tar. "
                        "Check ROOT member integrity/dependencies."
                    )
    return stats


def _flatten_binary_labels(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y.astype(np.int64, copy=False)
    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0].astype(np.int64, copy=False)
    if y.ndim == 2:
        # one-hot or logits-like: take argmax
        return np.argmax(y, axis=1).astype(np.int64, copy=False)
    raise RuntimeError(f"Unsupported labels shape: {y.shape}")


def stream_h5_constituent_averages(
    h5_path: Path,
    target_jets: int,
    batch_size: int,
    verbose: bool = True,
) -> Dict[str, RunningStat]:
    """
    Stream train.h5 in batches and compute class-wise averages:
      label 0 -> Background
      label 1 -> Top

    Uses schema from existing project loaders:
      - labels
      - fjet_clus_pt
    """
    h5py = _require_mod("h5py")

    stats = {"Background": RunningStat(), "Top": RunningStat()}

    with h5py.File(str(h5_path), "r") as f:
        if "labels" not in f:
            raise RuntimeError("Expected dataset 'labels' not found in H5 file.")
        if "fjet_clus_pt" not in f:
            raise RuntimeError("Expected dataset 'fjet_clus_pt' not found in H5 file.")

        n_total = int(f["labels"].shape[0])
        n_target = min(int(target_jets), n_total)
        start = 0
        while start < n_target:
            end = min(start + batch_size, n_target)
            y = _flatten_binary_labels(f["labels"][start:end])
            pt = f["fjet_clus_pt"][start:end]
            counts = (pt > 0).sum(axis=1).astype(np.int32)

            mask_top = y == 1
            mask_bg = ~mask_top
            stats["Top"].update(counts[mask_top])
            stats["Background"].update(counts[mask_bg])

            if verbose:
                print(
                    f"[train.h5] loaded={end}/{n_target} "
                    f"Top_mean={stats['Top'].mean:.3f} "
                    f"Background_mean={stats['Background'].mean:.3f}"
                )
            start = end

    return stats


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _load_canonical_map(csv_path: Path) -> Dict[str, str]:
    """
    Load mapping: file_class -> canonical_class.
    If multiple canonical rows exist per file_class, keep the highest-fraction row.
    Expected CSV columns (from your file):
      file_class, canonical_class, count, fraction, n_events
    """
    if not csv_path.exists():
        return {}
    best: Dict[str, Tuple[float, str]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_cls = str(row.get("file_class", "")).strip()
            can_cls = str(row.get("canonical_class", "")).strip()
            if not file_cls or not can_cls:
                continue
            try:
                frac = float(row.get("fraction", 0.0))
            except Exception:
                frac = 0.0
            prev = best.get(file_cls)
            if prev is None or frac > prev[0]:
                best[file_cls] = (frac, can_cls)
    return {k: v[1] for k, v in best.items()}


def _remap_means_with_canonical(
    means: Dict[str, float],
    canonical_map: Dict[str, str],
) -> Dict[str, float]:
    """
    Remap bar labels from file_class names to canonical names.
    If canonical labels collide, append original file class for uniqueness.
    """
    if not canonical_map:
        return dict(means)
    out: Dict[str, float] = {}
    used: Dict[str, int] = {}
    for file_cls, val in means.items():
        can = canonical_map.get(file_cls, file_cls)
        label = can
        if label in used:
            label = f"{can} [{file_cls}]"
        used[label] = used.get(label, 0) + 1
        out[label] = val
    return out


def _bar_plot(
    means: Dict[str, float],
    title: str,
    out_path: Path,
    color: str = "#4C78A8",
) -> None:
    mpl = _require_mod("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    labels = list(means.keys())
    values = [means[k] for k in labels]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, values, color=color)
    ax.set_ylabel("Average # constituents")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _equivalent_compare_plot(
    jetclass_top_mean: float,
    jetclass_bg_mean: float,
    h5_top_mean: float,
    h5_bg_mean: float,
    out_path: Path,
) -> None:
    mpl = _require_mod("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    cats = ["Top-like", "Background-like"]
    x = np.arange(len(cats))
    w = 0.36

    y_jetclass = [jetclass_top_mean, jetclass_bg_mean]
    y_h5 = [h5_top_mean, h5_bg_mean]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2.0, y_jetclass, width=w, label="JetClass", color="#4C78A8")
    ax.bar(x + w / 2.0, y_h5, width=w, label="train.h5", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Average # constituents")
    ax.set_title("Equivalent-Class Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Streaming class-wise constituent-count averages for JetClass and train.h5"
    )
    p.add_argument(
        "--jetclass-tar",
        type=Path,
        default=Path(
            "/home/ryan/ComputerScience/ATLAS/HLT_Reco/ATLAS-top-tagging-open-data/data/JetClass_Pythia_train_100M_part0.tar"
        ),
    )
    p.add_argument(
        "--atlas-h5",
        type=Path,
        default=Path(
            "/home/ryan/ComputerScience/ATLAS/HLT_Reco/ATLAS-top-tagging-open-data/data/train.h5"
        ),
    )
    p.add_argument("--target-jets", type=int, default=200_000)
    p.add_argument(
        "--jetclass-per-file-max-jets",
        type=int,
        default=8_000,
        help="Max jets to read from one ROOT file before switching to next file.",
    )
    p.add_argument(
        "--jetclass-step-size",
        type=int,
        default=5_000,
        help="uproot iterate step size (entries per chunk).",
    )
    p.add_argument("--h5-batch-size", type=int, default=20_000)
    p.add_argument("--output-dir", type=Path, default=Path("plots/constituent_count_analysis"))
    p.add_argument(
        "--canonical-map-csv",
        type=Path,
        default=Path(
            "/home/ryan/ComputerScience/ATLAS/HLT_Reco/ATLAS-top-tagging-open-data/a_download_checkpoints/filename_vs_canonical_all_job21272179/filename_to_canonical_label_fraction_long.csv"
        ),
        help="CSV with columns file_class, canonical_class, fraction for display-label remapping.",
    )
    p.add_argument("--no-verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not bool(args.no_verbose)

    if verbose:
        print("=" * 72)
        print("Streaming constituent-count analysis")
        print(f"JetClass tar : {args.jetclass_tar}")
        print(f"train.h5     : {args.atlas_h5}")
        print(f"Target jets  : {args.target_jets}")
        print("=" * 72)

    # 1) JetClass (class-balanced streaming target)
    jetclass_stats = stream_jetclass_constituent_averages(
        tar_path=args.jetclass_tar,
        target_jets=int(args.target_jets),
        per_file_max_jets=int(args.jetclass_per_file_max_jets),
        step_size=int(args.jetclass_step_size),
        verbose=verbose,
    )
    jetclass_means = {k: v.mean for k, v in sorted(jetclass_stats.items())}
    canonical_map = _load_canonical_map(args.canonical_map_csv)
    jetclass_means_for_plot = _remap_means_with_canonical(jetclass_means, canonical_map)
    if verbose and canonical_map:
        print(f"[mapping] loaded {len(canonical_map)} file_class -> canonical_class entries")

    # 2) ATLAS train.h5 (binary classes)
    h5_stats = stream_h5_constituent_averages(
        h5_path=args.atlas_h5,
        target_jets=int(args.target_jets),
        batch_size=int(args.h5_batch_size),
        verbose=verbose,
    )
    h5_means = {k: h5_stats[k].mean for k in ["Background", "Top"]}

    # 3) Equivalent comparison
    # Top-equivalent: TTBar
    if "TTBar" not in jetclass_stats or jetclass_stats["TTBar"].n <= 0:
        raise RuntimeError("JetClass class 'TTBar' not found/empty; cannot build top-equivalent plot.")
    jc_top = jetclass_stats["TTBar"].mean

    # Background-equivalent proxy:
    # user-requested single class: ZJetsToNuNu
    if "ZJetsToNuNu" not in jetclass_stats or jetclass_stats["ZJetsToNuNu"].n <= 0:
        raise RuntimeError(
            "JetClass class 'ZJetsToNuNu' not found/empty; cannot build background-equivalent plot."
        )
    jc_bg = jetclass_stats["ZJetsToNuNu"].mean

    h5_top = h5_stats["Top"].mean
    h5_bg = h5_stats["Background"].mean

    # 4) Save plots
    out_dir = args.output_dir
    _bar_plot(
        means=jetclass_means_for_plot,
        title=f"JetClass: Avg Constituents by Class (streamed to {args.target_jets} jets)",
        out_path=out_dir / "jetclass_avg_constituents_by_class.png",
        color="#4C78A8",
    )
    _bar_plot(
        means=h5_means,
        title=f"train.h5: Avg Constituents by Class (streamed to {args.target_jets} jets)",
        out_path=out_dir / "trainh5_avg_constituents_by_class.png",
        color="#F58518",
    )
    _equivalent_compare_plot(
        jetclass_top_mean=jc_top,
        jetclass_bg_mean=jc_bg,
        h5_top_mean=h5_top,
        h5_bg_mean=h5_bg,
        out_path=out_dir / "equivalent_class_comparison.png",
    )

    # 5) Save numeric summary
    summary = {
        "config": {
            "jetclass_tar": str(args.jetclass_tar),
            "atlas_h5": str(args.atlas_h5),
            "canonical_map_csv": str(args.canonical_map_csv),
            "target_jets": int(args.target_jets),
            "jetclass_per_file_max_jets": int(args.jetclass_per_file_max_jets),
            "jetclass_step_size": int(args.jetclass_step_size),
            "h5_batch_size": int(args.h5_batch_size),
        },
        "jetclass": {
            c: {
                "canonical_class": canonical_map.get(c, c),
                "n_jets": jetclass_stats[c].n,
                "avg_constituents": jetclass_stats[c].mean,
            }
            for c in sorted(jetclass_stats.keys())
        },
        "train_h5": {
            c: {"n_jets": h5_stats[c].n, "avg_constituents": h5_stats[c].mean}
            for c in ["Background", "Top"]
        },
        "equivalent_mapping": {
            "top": {"jetclass": "TTBar", "train_h5": "Top"},
            "background": {
                "jetclass": "ZJetsToNuNu",
                "train_h5": "Background",
            },
        },
        "equivalent_means": {
            "jetclass_top_like": jc_top,
            "jetclass_background_like": jc_bg,
            "train_h5_top": h5_top,
            "train_h5_background": h5_bg,
        },
        "plots": {
            "jetclass_bar": str((out_dir / "jetclass_avg_constituents_by_class.png").resolve()),
            "trainh5_bar": str((out_dir / "trainh5_avg_constituents_by_class.png").resolve()),
            "equivalent_bar": str((out_dir / "equivalent_class_comparison.png").resolve()),
        },
    }
    _save_json(out_dir / "summary.json", summary)

    print("\nDone.")
    print(f"Saved plots + summary in: {out_dir.resolve()}")
    print("Key means:")
    print(f"  JetClass TTBar (Top-like): {jc_top:.3f}")
    print(f"  JetClass ZJetsToNuNu (Background-like): {jc_bg:.3f}")
    print(f"  train.h5 Top: {h5_top:.3f}")
    print(f"  train.h5 Background: {h5_bg:.3f}")


if __name__ == "__main__":
    main()
