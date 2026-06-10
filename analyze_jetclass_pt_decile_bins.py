#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compute equal-occupancy jet-pT bins for a streamed JetClass sample.

The default mode uses the stored JetClass ``jet_pt`` branch. Alternative modes
can compute scalar or vector pT from constituent branches.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


FILE_RE = re.compile(r"^(?P<cls>[A-Za-z0-9]+)_(?P<idx>\d{3})\.root$")


def collect_files_by_class(data_dir: Path) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for path in sorted(data_dir.glob("*.root")):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        out.setdefault(match.group("cls"), []).append(path)
    if not out:
        raise RuntimeError(f"No JetClass ROOT files found in {data_dir}")
    return out


def first_tree(file_path: Path):
    import uproot

    f = uproot.open(str(file_path))
    for key, cls in f.classnames().items():
        if cls == "TTree":
            return f[key]
    raise RuntimeError(f"No TTree found in {file_path}")


def distribute_quota(total: int, classes: Sequence[str]) -> Dict[str, int]:
    base = int(total) // len(classes)
    rem = int(total) % len(classes)
    return {cls: base + (1 if i < rem else 0) for i, cls in enumerate(classes)}


def pt_branches_for_mode(pt_mode: str) -> List[str]:
    if pt_mode == "jet_pt":
        return ["jet_pt"]
    if pt_mode == "scalar_sum_part_pt":
        return ["part_pt"]
    if pt_mode == "vector_sum_part_pxpy":
        return ["part_px", "part_py"]
    raise ValueError(f"Unknown pt_mode: {pt_mode}")


def compute_pt_from_arrays(arrays, pt_mode: str) -> np.ndarray:
    import awkward as ak

    if pt_mode == "jet_pt":
        return np.asarray(arrays["jet_pt"], dtype=np.float64)
    if pt_mode == "scalar_sum_part_pt":
        return ak.to_numpy(ak.sum(arrays["part_pt"], axis=1)).astype(np.float64)
    if pt_mode == "vector_sum_part_pxpy":
        px = ak.to_numpy(ak.sum(arrays["part_px"], axis=1)).astype(np.float64)
        py = ak.to_numpy(ak.sum(arrays["part_py"], axis=1)).astype(np.float64)
        return np.sqrt(px * px + py * py)
    raise ValueError(f"Unknown pt_mode: {pt_mode}")


def load_pt_values_from_file(file_path: Path, n_needed: int, pt_mode: str, step_size: int) -> np.ndarray:
    if n_needed <= 0:
        return np.zeros((0,), dtype=np.float64)
    tree = first_tree(file_path)
    branches = pt_branches_for_mode(pt_mode)
    missing = [b for b in branches if b not in set(str(k) for k in tree.keys())]
    if missing:
        raise RuntimeError(f"{file_path} is missing branches required for {pt_mode}: {missing}")

    chunks: List[np.ndarray] = []
    n_seen = 0
    for arrays in tree.iterate(branches, step_size=int(step_size), library="ak"):
        vals = compute_pt_from_arrays(arrays, pt_mode)
        vals = vals[np.isfinite(vals)]
        vals = vals[vals >= 0.0]
        if vals.size == 0:
            continue
        take = min(int(n_needed) - n_seen, int(vals.size))
        chunks.append(vals[:take])
        n_seen += take
        if n_seen >= int(n_needed):
            break
    if not chunks:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def load_pt_sample(
    data_dir: Path,
    n_jets: int,
    pt_mode: str,
    sampling: str,
    step_size: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    by_class = collect_files_by_class(data_dir)
    classes = sorted(by_class.keys())
    per_class_counts = {cls: 0 for cls in classes}

    all_vals: List[np.ndarray] = []
    if sampling == "class_balanced":
        quotas = distribute_quota(int(n_jets), classes)
        for cls in classes:
            remaining = quotas[cls]
            file_cursor = 0
            while remaining > 0:
                path = by_class[cls][file_cursor % len(by_class[cls])]
                file_cursor += 1
                vals = load_pt_values_from_file(path, remaining, pt_mode, step_size)
                if vals.size == 0:
                    if file_cursor > 2 * len(by_class[cls]):
                        raise RuntimeError(f"Could not load enough jets for class {cls}")
                    continue
                all_vals.append(vals)
                per_class_counts[cls] += int(vals.size)
                remaining -= int(vals.size)
    elif sampling == "sequential":
        remaining = int(n_jets)
        for cls in classes:
            for path in by_class[cls]:
                if remaining <= 0:
                    break
                vals = load_pt_values_from_file(path, remaining, pt_mode, step_size)
                if vals.size == 0:
                    continue
                all_vals.append(vals)
                per_class_counts[cls] += int(vals.size)
                remaining -= int(vals.size)
            if remaining <= 0:
                break
        if remaining > 0:
            raise RuntimeError(f"Only loaded {n_jets - remaining} jets, requested {n_jets}")
    else:
        raise ValueError(f"Unknown sampling mode: {sampling}")

    values = np.concatenate(all_vals, axis=0).astype(np.float64)
    if values.size != int(n_jets):
        raise RuntimeError(f"Loaded {values.size} jets, expected {n_jets}")
    return values, per_class_counts


def make_decile_table(values: np.ndarray, n_bins: int = 10) -> List[dict]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty 1D array")
    if values.size % n_bins != 0:
        raise ValueError("This script expects n_jets divisible by n_bins for exact 10% bins.")

    sorted_vals = np.sort(values)
    per_bin = values.size // n_bins
    rows: List[dict] = []
    for i in range(n_bins):
        lo_idx = i * per_bin
        hi_idx = (i + 1) * per_bin
        chunk = sorted_vals[lo_idx:hi_idx]
        rows.append(
            {
                "bin": i + 1,
                "rank_start": lo_idx,
                "rank_stop_exclusive": hi_idx,
                "low_geV": float(chunk[0]),
                "high_geV": float(chunk[-1]),
                "count": int(chunk.size),
                "fraction": float(chunk.size / values.size),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["bin", "rank_start", "rank_stop_exclusive", "low_geV", "high_geV", "count", "fraction"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_decile_bins(rows: Sequence[dict], out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r['low_geV']:.1f}-{r['high_geV']:.1f}" for r in rows]
    counts = [int(r["count"]) for r in rows]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(rows)), counts, color="#4C78A8")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Jets")
    ax.set_xlabel("pT bin edges [GeV]")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pt_hist(values: np.ndarray, rows: Sequence[dict], out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(values, bins=80, color="#72B7B2", alpha=0.82)
    for row in rows:
        ax.axvline(float(row["low_geV"]), color="black", linewidth=0.7, alpha=0.4)
    ax.axvline(float(rows[-1]["high_geV"]), color="black", linewidth=0.7, alpha=0.4)
    ax.set_xlabel("Jet pT [GeV]")
    ax.set_ylabel("Jets")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.30)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute exact decile jet-pT bins for JetClass.")
    p.add_argument("--data_dir", type=Path, default=Path("data/jetclass_part0"))
    p.add_argument("--out_dir", type=Path, default=Path("plots/jetclass_pt_decile_bins"))
    p.add_argument("--n_jets", type=int, default=100_000)
    p.add_argument("--step_size", type=int, default=20_000)
    p.add_argument("--sampling", choices=["class_balanced", "sequential"], default="class_balanced")
    p.add_argument(
        "--pt_mode",
        choices=["jet_pt", "scalar_sum_part_pt", "vector_sum_part_pxpy"],
        default="jet_pt",
        help="jet_pt uses the stored branch; scalar/vector modes compute from constituents.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.n_jets) <= 0:
        raise SystemExit("--n_jets must be positive")
    if int(args.n_jets) % 10 != 0:
        raise SystemExit("--n_jets must be divisible by 10 for exact decile bins")

    values, per_class_counts = load_pt_sample(
        data_dir=args.data_dir.resolve(),
        n_jets=int(args.n_jets),
        pt_mode=str(args.pt_mode),
        sampling=str(args.sampling),
        step_size=int(args.step_size),
    )
    rows = make_decile_table(values, n_bins=10)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "jetclass_pt_decile_bins.csv", rows)

    summary = {
        "data_dir": str(args.data_dir.resolve()),
        "n_jets": int(values.size),
        "pt_mode": str(args.pt_mode),
        "sampling": str(args.sampling),
        "per_class_counts": per_class_counts,
        "pt_min_geV": float(values.min()),
        "pt_max_geV": float(values.max()),
        "pt_mean_geV": float(values.mean()),
        "pt_median_geV": float(np.median(values)),
        "bins": rows,
    }
    with (out_dir / "jetclass_pt_decile_bins_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    title = f"JetClass pT decile bins ({values.size:,} jets, mode={args.pt_mode})"
    plot_decile_bins(rows, out_dir / "jetclass_pt_decile_bins_bar.png", title)
    plot_pt_hist(values, rows, out_dir / "jetclass_pt_distribution_with_deciles.png", title)

    print("JetClass pT decile bins")
    print(f"Data dir: {args.data_dir.resolve()}")
    print(f"Output:   {out_dir.resolve()}")
    print(f"Mode:     {args.pt_mode}")
    print(f"Sampling: {args.sampling}")
    print(f"Jets:     {values.size}")
    print("bin, low_geV, high_geV, count, fraction")
    for row in rows:
        print(
            f"{row['bin']:02d}, {row['low_geV']:.6g}, {row['high_geV']:.6g}, "
            f"{row['count']}, {row['fraction']:.3f}"
        )


if __name__ == "__main__":
    main()
