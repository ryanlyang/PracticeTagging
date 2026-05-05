#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a filename-class -> canonical label_* mapping table for JetClass ROOT files.

For each filename class (e.g. HToBB, TTBar, ZJetsToNuNu), this script computes
the fraction of events assigned to each canonical JetClass label branch
(label_QCD, label_Hbb, ..., label_Tbl).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import awkward as ak
import numpy as np
import uproot


FILE_RE = re.compile(r"^(?P<cls>[A-Za-z0-9]+)_(?P<idx>\d{3})\.root$")

CANONICAL_CLASS_TO_LABEL_BRANCH = {
    "QCD": "label_QCD",
    "Hbb": "label_Hbb",
    "Hcc": "label_Hcc",
    "Hgg": "label_Hgg",
    "H4q": "label_H4q",
    "Hqql": "label_Hqql",
    "Zqq": "label_Zqq",
    "Wqq": "label_Wqq",
    "Tbqq": "label_Tbqq",
    "Tbl": "label_Tbl",
}
CANONICAL_CLASS_ORDER = list(CANONICAL_CLASS_TO_LABEL_BRANCH.keys())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Map filename classes to canonical label_* fractions")
    p.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/home/ryreu/atlas/PracticeTagging/data/jetclass_part0"),
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("plots/jetclass_label_mapping"),
    )
    p.add_argument("--split", type=str, default="all", choices=["all", "train", "val", "test"])
    p.add_argument("--train_files_per_class", type=int, default=8)
    p.add_argument("--val_files_per_class", type=int, default=1)
    p.add_argument("--test_files_per_class", type=int, default=1)
    p.add_argument("--seed", type=int, default=52)
    return p.parse_args()


def get_first_tree(file_path: Path):
    f = uproot.open(str(file_path))
    for key in f.keys():
        obj = f[key]
        if hasattr(obj, "arrays") and hasattr(obj, "num_entries"):
            return obj
    raise RuntimeError(f"No TTree found in {file_path}")


def collect_files_by_class(data_dir: Path) -> Dict[str, List[Tuple[int, Path]]]:
    out: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    for p in data_dir.glob("*.root"):
        m = FILE_RE.match(p.name)
        if not m:
            continue
        out[m.group("cls")].append((int(m.group("idx")), p))
    for cls in out:
        out[cls].sort(key=lambda x: x[0])
    return out


def split_files_by_class(
    files_by_class: Dict[str, List[Tuple[int, Path]]],
    n_train: int,
    n_val: int,
    n_test: int,
) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]], Dict[str, List[Path]]]:
    tr: Dict[str, List[Path]] = {}
    va: Dict[str, List[Path]] = {}
    te: Dict[str, List[Path]] = {}
    for cls, pairs in files_by_class.items():
        files = [p for _, p in pairs]
        need = n_train + n_val + n_test
        if len(files) < need:
            raise RuntimeError(f"{cls}: need {need} files, found {len(files)}")
        tr[cls] = files[:n_train]
        va[cls] = files[n_train:n_train + n_val]
        te[cls] = files[n_train + n_val:n_train + n_val + n_test]
    return tr, va, te


def choose_split(
    files_by_class: Dict[str, List[Tuple[int, Path]]],
    split: str,
    n_train: int,
    n_val: int,
    n_test: int,
) -> Dict[str, List[Path]]:
    tr, va, te = split_files_by_class(files_by_class, n_train=n_train, n_val=n_val, n_test=n_test)
    if split == "train":
        return tr
    if split == "val":
        return va
    if split == "test":
        return te
    out: Dict[str, List[Path]] = {}
    for cls in tr:
        out[cls] = tr[cls] + va[cls] + te[cls]
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files_by_class = collect_files_by_class(args.data_dir.resolve())
    if not files_by_class:
        raise RuntimeError(f"No ROOT files found in {args.data_dir}")
    split_files = choose_split(
        files_by_class=files_by_class,
        split=str(args.split),
        n_train=int(args.train_files_per_class),
        n_val=int(args.val_files_per_class),
        n_test=int(args.test_files_per_class),
    )

    branches = [CANONICAL_CLASS_TO_LABEL_BRANCH[c] for c in CANONICAL_CLASS_ORDER]
    counts_by_filecls: Dict[str, np.ndarray] = {}
    events_by_filecls: Dict[str, int] = {}

    print(f"Reading split={args.split} from {args.data_dir}")
    for file_cls in sorted(split_files.keys()):
        total_counts = np.zeros((len(CANONICAL_CLASS_ORDER),), dtype=np.int64)
        total_events = 0
        for fp in split_files[file_cls]:
            tree = get_first_tree(fp)
            arr = tree.arrays(branches, library="ak")
            mats = [np.asarray(ak.to_numpy(arr[b]), dtype=np.float32) for b in branches]
            if not mats:
                continue
            y = np.stack(mats, axis=1)  # [N, 10]
            if y.shape[0] == 0:
                continue
            idx = np.argmax(y, axis=1)
            binc = np.bincount(idx, minlength=len(CANONICAL_CLASS_ORDER))
            total_counts += binc.astype(np.int64)
            total_events += int(y.shape[0])
        counts_by_filecls[file_cls] = total_counts
        events_by_filecls[file_cls] = total_events

    frac_by_filecls: Dict[str, np.ndarray] = {}
    for cls, cnt in counts_by_filecls.items():
        n = max(events_by_filecls.get(cls, 0), 1)
        frac_by_filecls[cls] = cnt.astype(np.float64) / float(n)

    # Wide CSV: one row per filename class, columns are canonical fractions and counts.
    wide_csv = args.output_dir / "filename_to_canonical_label_fraction_table.csv"
    fields = (
        ["file_class", "n_events"]
        + [f"frac_{c}" for c in CANONICAL_CLASS_ORDER]
        + [f"count_{c}" for c in CANONICAL_CLASS_ORDER]
        + ["top_canonical_label", "top_canonical_frac"]
    )
    with wide_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cls in sorted(counts_by_filecls.keys()):
            frac = frac_by_filecls[cls]
            cnt = counts_by_filecls[cls]
            top_i = int(np.argmax(frac))
            row = {
                "file_class": cls,
                "n_events": int(events_by_filecls[cls]),
                "top_canonical_label": CANONICAL_CLASS_ORDER[top_i],
                "top_canonical_frac": float(frac[top_i]),
            }
            for i, c in enumerate(CANONICAL_CLASS_ORDER):
                row[f"frac_{c}"] = float(frac[i])
                row[f"count_{c}"] = int(cnt[i])
            w.writerow(row)

    # Long CSV: one row per (filename class, canonical class).
    long_csv = args.output_dir / "filename_to_canonical_label_fraction_long.csv"
    with long_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["file_class", "canonical_class", "count", "fraction", "n_events"],
        )
        w.writeheader()
        for cls in sorted(counts_by_filecls.keys()):
            frac = frac_by_filecls[cls]
            cnt = counts_by_filecls[cls]
            n = int(events_by_filecls[cls])
            for i, c in enumerate(CANONICAL_CLASS_ORDER):
                w.writerow(
                    {
                        "file_class": cls,
                        "canonical_class": c,
                        "count": int(cnt[i]),
                        "fraction": float(frac[i]),
                        "n_events": n,
                    }
                )

    meta = {
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "split": str(args.split),
        "train_files_per_class": int(args.train_files_per_class),
        "val_files_per_class": int(args.val_files_per_class),
        "test_files_per_class": int(args.test_files_per_class),
        "canonical_classes": CANONICAL_CLASS_ORDER,
        "file_classes": sorted(counts_by_filecls.keys()),
        "n_events_by_file_class": events_by_filecls,
        "wide_csv": str(wide_csv),
        "long_csv": str(long_csv),
    }
    with (args.output_dir / "run_metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")
    print(f"Saved: {wide_csv}")
    print(f"Saved: {long_csv}")
    print(f"Saved: {args.output_dir / 'run_metadata.json'}")


if __name__ == "__main__":
    main()

