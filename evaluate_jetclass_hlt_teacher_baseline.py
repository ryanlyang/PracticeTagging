#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JetClass teacher-vs-baseline evaluation with HLT-like corruption.

Pipeline:
1) Load offline constituents from JetClass ROOT files.
2) Build HLT-like corrupted constituents (efficiency loss + smearing + reassignment + merging).
3) Train a teacher model on offline features.
4) Train a baseline model on corrupted (HLT-like) features.
5) Report/save metrics on a held-out test split.

This is a physics-informed proxy for HLT effects, not an exact trigger emulation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


try:
    import awkward as ak
    import uproot
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This script requires 'uproot' and 'awkward'. "
        "Install in your env, e.g. `python -m pip install --user weaver-core`."
    ) from exc


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
CANONICAL_CLASS_ORDER = tuple(CANONICAL_CLASS_TO_LABEL_BRANCH.keys())

# Backward-compatible aliases used by older run scripts in this repo.
CLASS_NAME_ALIASES = {
    "HToBB": "Hbb",
    "HToCC": "Hcc",
    "HToGG": "Hgg",
    "HToWW4Q": "H4q",
    "HToWW2Q1L": "Hqql",
    "TTBar": "Tbqq",
    "TTBarLep": "Tbl",
    "WToQQ": "Wqq",
    "ZToQQ": "Zqq",
    "ZJetsToNuNu": "QCD",
}


RAW_DIM = 14
IDX_PT = 0
IDX_ETA = 1
IDX_PHI = 2
IDX_E = 3
IDX_CHARGE = 4
IDX_PID0 = 5  # charged hadron
IDX_PID1 = 6  # neutral hadron
IDX_PID2 = 7  # photon
IDX_PID3 = 8  # electron
IDX_PID4 = 9  # muon
IDX_D0 = 10
IDX_D0ERR = 11
IDX_DZ = 12
IDX_DZERR = 13


TYPE_CH = 0
TYPE_NH = 1
TYPE_GAM = 2
TYPE_ELE = 3
TYPE_MU = 4
TYPE_UNK = 5
N_TYPES = 6

MERGE_MODE_NONE = 0
MERGE_MODE_SAME_TYPE = 1
MERGE_MODE_ELE_GAM = 2
MERGE_MODE_CH_NH = 3
N_MERGE_MODES = 4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fpr_at_target_tpr(fpr: np.ndarray, tpr: np.ndarray, target_tpr: float) -> float:
    if fpr.size == 0 or tpr.size == 0:
        return float("nan")
    idx = np.searchsorted(tpr, target_tpr, side="left")
    if idx <= 0:
        return float(fpr[0])
    if idx >= len(tpr):
        return float(fpr[-1])
    x0, x1 = float(tpr[idx - 1]), float(tpr[idx])
    y0, y1 = float(fpr[idx - 1]), float(fpr[idx])
    if abs(x1 - x0) < 1e-12:
        return y1
    a = (target_tpr - x0) / (x1 - x0)
    return y0 + a * (y1 - y0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate JetClass teacher vs HLT-like baseline")
    p.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/home/ryan/ComputerScience/ATLAS/HLT_Reco/ATLAS-top-tagging-open-data/data/jetclass_part0"),
    )
    p.add_argument("--save_dir", type=Path, default=Path("checkpoints/jetclass_hlt_teacher_baseline"))
    p.add_argument("--run_name", type=str, default="jetclass_hlt_eval")
    p.add_argument("--seed", type=int, default=52)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--feature_mode", type=str, default="full", choices=["kin", "kinpid", "full"])
    p.add_argument(
        "--feature_preprocessing",
        type=str,
        default="canonical",
        choices=["canonical", "legacy"],
        help=(
            "Feature preprocessing style. "
            "'canonical' mirrors JetClass YAML manual preprocessing as closely as possible; "
            "'legacy' keeps this repo's previous handcrafted feature layout."
        ),
    )
    p.add_argument(
        "--class_assignment",
        type=str,
        default="canonical_labels",
        choices=["filename", "canonical_labels"],
        help=(
            "How class labels are assigned to events: "
            "'filename' uses file-prefix classes, "
            "'canonical_labels' uses canonical label_* branches."
        ),
    )
    p.add_argument("--max_constits", type=int, default=128)

    p.add_argument("--train_files_per_class", type=int, default=8)
    p.add_argument("--val_files_per_class", type=int, default=1)
    p.add_argument("--test_files_per_class", type=int, default=1)
    p.add_argument("--shuffle_files", action="store_true", default=False)

    p.add_argument("--n_train_jets", type=int, default=30000)
    p.add_argument("--n_val_jets", type=int, default=5000)
    p.add_argument("--n_test_jets", type=int, default=10000)

    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_epochs", type=int, default=3)

    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--ff_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)

    # HLT-like corruption settings
    p.add_argument("--hlt_pt_threshold", type=float, default=1.2)
    p.add_argument("--merge_prob_scale", type=float, default=1.0)
    p.add_argument("--reassign_scale", type=float, default=1.0)
    p.add_argument("--smear_scale", type=float, default=1.0)
    p.add_argument("--eff_plateau_barrel", type=float, default=0.98)
    p.add_argument("--eff_plateau_endcap", type=float, default=0.94)
    p.add_argument("--eff_turnon_pt", type=float, default=0.9)
    p.add_argument("--eff_width_pt", type=float, default=0.4)
    p.add_argument("--target_class", type=str, default="HToBB")
    p.add_argument("--background_class", type=str, default="ZJetsToNuNu")

    return p.parse_args()


def get_first_tree(file_path: Path):
    f = uproot.open(str(file_path))
    for key in f.keys():
        obj = f[key]
        if hasattr(obj, "arrays") and hasattr(obj, "num_entries"):
            return obj
    raise RuntimeError(f"No TTree found in {file_path}")


def collect_files_by_class(data_dir: Path) -> Dict[str, List[Tuple[int, Path]]]:
    out: Dict[str, List[Tuple[int, Path]]] = {}
    for p in sorted(data_dir.glob("*.root")):
        m = FILE_RE.match(p.name)
        if not m:
            continue
        cls = m.group("cls")
        idx = int(m.group("idx"))
        out.setdefault(cls, []).append((idx, p.resolve()))
    if not out:
        raise RuntimeError(f"No JetClass ROOT files found in {data_dir}")
    for cls in out:
        out[cls].sort(key=lambda x: x[0])
    return out


def split_files_by_class(
    files_by_class: Dict[str, List[Tuple[int, Path]]],
    n_train: int,
    n_val: int,
    n_test: int,
    shuffle: bool,
    seed: int,
) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]], Dict[str, List[Path]]]:
    rng = random.Random(seed)
    tr: Dict[str, List[Path]] = {}
    va: Dict[str, List[Path]] = {}
    te: Dict[str, List[Path]] = {}
    need = n_train + n_val + n_test
    for cls, items in sorted(files_by_class.items()):
        if len(items) < need:
            raise ValueError(f"Class {cls} has {len(items)} files, needs >= {need}.")
        paths = [p for _, p in items]
        if shuffle:
            rng.shuffle(paths)
        tr[cls] = paths[:n_train]
        va[cls] = paths[n_train:n_train + n_val]
        te[cls] = paths[n_train + n_val:n_train + n_val + n_test]
    return tr, va, te


def class_quota(total: int, classes: Sequence[str]) -> Dict[str, int]:
    n = len(classes)
    base = total // n
    rem = total % n
    out: Dict[str, int] = {}
    for i, cls in enumerate(classes):
        out[cls] = base + (1 if i < rem else 0)
    return out


def resolve_branch_map(tree) -> Dict[str, str | None]:
    keys = set(str(k) for k in tree.keys())

    def pick(cands: Sequence[str]) -> str | None:
        for c in cands:
            if c in keys:
                return c
        return None

    return {
        "px": pick(["part_px"]),
        "py": pick(["part_py"]),
        "pz": pick(["part_pz"]),
        "energy": pick(["part_energy", "part_e", "part_E"]),
        "charge": pick(["part_charge"]),
        "is_ch": pick(["part_isChargedHadron"]),
        "is_nh": pick(["part_isNeutralHadron"]),
        "is_pho": pick(["part_isPhoton"]),
        "is_ele": pick(["part_isElectron"]),
        "is_mu": pick(["part_isMuon"]),
        "d0": pick(["part_d0val", "part_d0"]),
        "d0err": pick(["part_d0err"]),
        "dz": pick(["part_dzval", "part_dz"]),
        "dzerr": pick(["part_dzerr"]),
    }


def required_raw_branches(branch_map: Dict[str, str | None]) -> List[str]:
    req = ["px", "py", "pz", "energy"]
    for r in req:
        if branch_map[r] is None:
            raise RuntimeError(f"Missing required branch for {r}.")
    out = []
    for v in branch_map.values():
        if v is not None:
            out.append(v)
    return sorted(set(out))


def canonical_label_branches_for_classes(class_names: Sequence[str]) -> List[str]:
    out = []
    for cls in class_names:
        canonical = CLASS_NAME_ALIASES.get(cls, cls)
        if canonical not in CANONICAL_CLASS_TO_LABEL_BRANCH:
            raise ValueError(
                f"Class '{cls}' does not map to canonical JetClass labels. "
                f"Known canonical classes: {list(CANONICAL_CLASS_ORDER)}"
            )
        out.append(CANONICAL_CLASS_TO_LABEL_BRANCH[canonical])
    return out


def extract_chunk_class_indices(arrays, label_branches: Sequence[str]) -> np.ndarray:
    """
    Convert per-event canonical one-hot-ish label branches into class indices.
    Events with no positive label are marked as -1.
    """
    if len(label_branches) == 0:
        return np.zeros((0,), dtype=np.int64)
    mats = [np.asarray(ak.to_numpy(arrays[b]), dtype=np.float32) for b in label_branches]
    scores = np.stack(mats, axis=1)
    idx = np.argmax(scores, axis=1).astype(np.int64)
    valid = scores.sum(axis=1) > 0.5
    idx[~valid] = -1
    return idx


def _to_1d_float(arr_evt, n: int) -> np.ndarray:
    x = np.asarray(ak.to_numpy(arr_evt), dtype=np.float32)
    if x.ndim == 0:
        x = np.full((n,), float(x), dtype=np.float32)
    return x


def extract_tokens_from_chunk(
    arrays,
    branch_map: Dict[str, str | None],
    max_constits: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_evt = len(arrays[branch_map["px"]])  # required branch exists
    tokens = np.zeros((n_evt, max_constits, RAW_DIM), dtype=np.float32)
    mask = np.zeros((n_evt, max_constits), dtype=bool)

    for i in range(n_evt):
        px = _to_1d_float(arrays[branch_map["px"]][i], 0)
        py = _to_1d_float(arrays[branch_map["py"]][i], len(px))
        pz = _to_1d_float(arrays[branch_map["pz"]][i], len(px))
        e = _to_1d_float(arrays[branch_map["energy"]][i], len(px))
        if len(px) == 0:
            continue

        pt = np.sqrt(np.maximum(px * px + py * py, 1e-12))
        p = np.sqrt(np.maximum(px * px + py * py + pz * pz, 1e-12))
        eta = 0.5 * np.log(np.clip((p + pz) / np.maximum(p - pz, 1e-8), 1e-8, 1e8))
        phi = np.arctan2(py, px)
        ene = np.maximum(e, 1e-8)

        order = np.argsort(-pt)
        take = min(len(order), max_constits)
        idx = order[:take]

        tok = np.zeros((take, RAW_DIM), dtype=np.float32)
        tok[:, IDX_PT] = pt[idx]
        tok[:, IDX_ETA] = eta[idx]
        tok[:, IDX_PHI] = phi[idx]
        tok[:, IDX_E] = ene[idx]

        def fill_opt(src_key: str, dst_col: int) -> None:
            name = branch_map[src_key]
            if name is None:
                return
            arr = _to_1d_float(arrays[name][i], len(px))
            if len(arr) != len(px):
                return
            tok[:, dst_col] = arr[idx]

        fill_opt("charge", IDX_CHARGE)
        fill_opt("is_ch", IDX_PID0)
        fill_opt("is_nh", IDX_PID1)
        fill_opt("is_pho", IDX_PID2)
        fill_opt("is_ele", IDX_PID3)
        fill_opt("is_mu", IDX_PID4)
        fill_opt("d0", IDX_D0)
        fill_opt("d0err", IDX_D0ERR)
        fill_opt("dz", IDX_DZ)
        fill_opt("dzerr", IDX_DZERR)

        tokens[i, :take] = tok
        mask[i, :take] = True

    return tokens, mask


def load_split(
    split_files: Dict[str, List[Path]],
    n_total: int,
    max_constits: int,
    class_to_idx: Dict[str, int],
    seed: int,
    class_assignment: str = "filename",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    if class_assignment not in {"filename", "canonical_labels"}:
        raise ValueError(f"Unknown class_assignment={class_assignment}")
    classes = sorted(class_to_idx.keys(), key=lambda c: int(class_to_idx[c]))
    quotas = class_quota(n_total, classes)
    all_tok: List[np.ndarray] = []
    all_mask: List[np.ndarray] = []
    all_lab: List[np.ndarray] = []
    shared_files = sorted({p for paths in split_files.values() for p in paths})
    label_branches = canonical_label_branches_for_classes(classes) if class_assignment == "canonical_labels" else []

    if class_assignment == "canonical_labels":
        # Fast path: scan shared files once and fan out events to all classes at once.
        if len(shared_files) == 0:
            raise RuntimeError("No files available for canonical label assignment.")

        got_tok_by_cls: Dict[str, List[np.ndarray]] = {c: [] for c in classes}
        got_mask_by_cls: Dict[str, List[np.ndarray]] = {c: [] for c in classes}
        count_by_cls: Dict[str, int] = {c: 0 for c in classes}

        cursor = 0
        no_progress = 0
        max_iters = max(len(shared_files) * 120, 2000)

        while any(count_by_cls[c] < int(quotas[c]) for c in classes):
            file_path = shared_files[cursor % len(shared_files)]
            cursor += 1

            tree = get_first_tree(file_path)
            bmap = resolve_branch_map(tree)
            branches = required_raw_branches(bmap)
            keys = set(str(k) for k in tree.keys())
            missing = [b for b in label_branches if b not in keys]
            if missing:
                raise RuntimeError(
                    f"Missing canonical label branches in {file_path.name}: {missing}"
                )
            branches = sorted(set(branches + label_branches))

            n_entries = int(tree.num_entries)
            if n_entries <= 0:
                if cursor > max_iters:
                    break
                continue

            remain_total = int(sum(max(0, int(quotas[c]) - int(count_by_cls[c])) for c in classes))
            chunk = int(min(max(remain_total * 2, 4096), 20000, n_entries))
            start = int(rng.randint(0, max(1, n_entries - chunk + 1)))
            stop = min(n_entries, start + chunk)
            arr = tree.arrays(branches, entry_start=start, entry_stop=stop, library="ak")

            tok, msk = extract_tokens_from_chunk(arr, bmap, max_constits=max_constits)
            valid_evt = msk.any(axis=1)
            progressed = False
            if valid_evt.any():
                tok_v = tok[valid_evt]
                msk_v = msk[valid_evt]
                y_idx = extract_chunk_class_indices(arr, label_branches)[valid_evt]

                for cls in classes:
                    need = int(quotas[cls]) - int(count_by_cls[cls])
                    if need <= 0:
                        continue
                    cls_idx = int(class_to_idx[cls])
                    sel = np.flatnonzero(y_idx == cls_idx)
                    if sel.size == 0:
                        continue
                    take = sel[:need]
                    got_tok_by_cls[cls].append(tok_v[take])
                    got_mask_by_cls[cls].append(msk_v[take])
                    count_by_cls[cls] += int(take.size)
                    progressed = True

            if progressed:
                no_progress = 0
            else:
                no_progress += 1

            if cursor % 50 == 0:
                done = int(sum(min(int(count_by_cls[c]), int(quotas[c])) for c in classes))
                print(f"[load_split:canonical] progress {done}/{n_total} jets (iter={cursor})")

            if no_progress > len(shared_files) * 20 or cursor > max_iters:
                break

        missing_cls = [c for c in classes if int(count_by_cls[c]) < int(quotas[c])]
        if missing_cls:
            detail = ", ".join(
                f"{c}:{count_by_cls[c]}/{quotas[c]}" for c in missing_cls
            )
            raise RuntimeError(
                "Could not satisfy canonical class quotas during split loading. "
                f"Missing counts -> {detail}"
            )

        for cls in classes:
            need = int(quotas[cls])
            if need <= 0:
                continue
            tok_c = np.concatenate(got_tok_by_cls[cls], axis=0)[:need]
            msk_c = np.concatenate(got_mask_by_cls[cls], axis=0)[:need]
            lab_c = np.full((len(tok_c),), int(class_to_idx[cls]), dtype=np.int64)
            all_tok.append(tok_c)
            all_mask.append(msk_c)
            all_lab.append(lab_c)
    else:
        for cls in classes:
            need = int(quotas[cls])
            if need <= 0:
                continue
            if cls not in split_files:
                raise KeyError(
                    f"Class '{cls}' not found in split_files keys: {sorted(split_files.keys())}"
                )
            files = split_files[cls]
            if len(files) == 0:
                raise RuntimeError(f"No files available for class {cls}")

            got_tok: List[np.ndarray] = []
            got_mask: List[np.ndarray] = []
            cursor = 0
            while sum(len(x) for x in got_tok) < need:
                file_path = files[cursor % len(files)]
                cursor += 1
                tree = get_first_tree(file_path)
                bmap = resolve_branch_map(tree)
                branches = required_raw_branches(bmap)
                n_entries = int(tree.num_entries)
                if n_entries <= 0:
                    continue

                remain = need - sum(len(x) for x in got_tok)
                # read a random contiguous chunk; this keeps IO manageable.
                chunk = int(min(max(remain * 2, 1024), 20000, n_entries))
                start = int(rng.randint(0, max(1, n_entries - chunk + 1)))
                stop = min(n_entries, start + chunk)
                arr = tree.arrays(branches, entry_start=start, entry_stop=stop, library="ak")
                tok, msk = extract_tokens_from_chunk(arr, bmap, max_constits=max_constits)
                valid_evt = msk.any(axis=1)
                if valid_evt.any():
                    got_tok.append(tok[valid_evt])
                    got_mask.append(msk[valid_evt])
                if cursor > len(files) * 8 and sum(len(x) for x in got_tok) == 0:
                    break

            if not got_tok:
                raise RuntimeError(f"Could not load any jets for class {cls}")
            tok_c = np.concatenate(got_tok, axis=0)[:need]
            msk_c = np.concatenate(got_mask, axis=0)[:need]
            lab_c = np.full((len(tok_c),), int(class_to_idx[cls]), dtype=np.int64)
            all_tok.append(tok_c)
            all_mask.append(msk_c)
            all_lab.append(lab_c)

    tok = np.concatenate(all_tok, axis=0)
    msk = np.concatenate(all_mask, axis=0)
    lab = np.concatenate(all_lab, axis=0)

    perm = rng.permutation(len(lab))
    tok = tok[perm]
    msk = msk[perm]
    lab = lab[perm]
    return tok, msk, lab


@dataclass
class HLTParams:
    hlt_pt_threshold: float
    merge_prob_scale: float
    reassign_scale: float
    smear_scale: float
    eff_plateau_barrel: float
    eff_plateau_endcap: float
    eff_turnon_pt: float
    eff_width_pt: float


def infer_type_id(token: np.ndarray) -> int:
    pid = token[IDX_PID0:IDX_PID4 + 1]
    if np.max(pid) <= 0:
        return TYPE_UNK
    return int(np.argmax(pid))


def wrap_phi(phi: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(phi), np.cos(phi))


def token_to_p4(token: np.ndarray) -> Tuple[float, float, float, float]:
    pt = max(float(token[IDX_PT]), 1e-8)
    eta = float(token[IDX_ETA])
    phi = float(token[IDX_PHI])
    e = max(float(token[IDX_E]), 1e-8)
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * math.sinh(eta)
    return px, py, pz, e


def p4_to_ptetaphi(px: float, py: float, pz: float, e: float) -> Tuple[float, float, float, float]:
    pt = max(float(math.sqrt(px * px + py * py)), 1e-8)
    p = max(float(math.sqrt(px * px + py * py + pz * pz)), 1e-8)
    eta = 0.5 * math.log(max((p + pz) / max(p - pz, 1e-8), 1e-8))
    phi = math.atan2(py, px)
    return pt, eta, phi, max(float(e), 1e-8)


def allowed_merge_and_output_type(ti: int, tj: int) -> Tuple[bool, int | None]:
    # Same-type merges are allowed (except unknown).
    if ti == tj and ti != TYPE_UNK:
        return True, ti

    pair = {ti, tj}
    # Electron + photon -> electron
    if pair == {TYPE_ELE, TYPE_GAM}:
        return True, TYPE_ELE
    # Charged hadron + neutral hadron -> charged hadron
    if pair == {TYPE_CH, TYPE_NH}:
        return True, TYPE_CH

    return False, None


def infer_merge_mode(ti: int, tj: int) -> int:
    if ti == tj and ti != TYPE_UNK:
        return MERGE_MODE_SAME_TYPE
    pair = {ti, tj}
    if pair == {TYPE_ELE, TYPE_GAM}:
        return MERGE_MODE_ELE_GAM
    if pair == {TYPE_CH, TYPE_NH}:
        return MERGE_MODE_CH_NH
    return MERGE_MODE_NONE


def _empty_unmerge_provenance(max_constits: int) -> Dict[str, np.ndarray]:
    return {
        "split_target_mask": np.zeros((max_constits,), dtype=bool),
        "split_mode_target": np.zeros((max_constits,), dtype=np.int64),
        "parent_type_id": np.full((max_constits,), TYPE_UNK, dtype=np.int64),
        "child_type_a_target": np.full((max_constits,), TYPE_UNK, dtype=np.int64),
        "child_type_b_target": np.full((max_constits,), TYPE_UNK, dtype=np.int64),
        "child_attr_a_target": np.zeros((max_constits, 5), dtype=np.float32),
        "child_attr_b_target": np.zeros((max_constits, 5), dtype=np.float32),
    }


def merge_two_tokens(t1: np.ndarray, t2: np.ndarray, merged_type_override: int | None = None) -> np.ndarray:
    out = np.zeros((RAW_DIM,), dtype=np.float32)
    px1, py1, pz1, e1 = token_to_p4(t1)
    px2, py2, pz2, e2 = token_to_p4(t2)
    pt, eta, phi, e = p4_to_ptetaphi(px1 + px2, py1 + py2, pz1 + pz2, e1 + e2)
    out[IDX_PT] = pt
    out[IDX_ETA] = eta
    out[IDX_PHI] = phi
    out[IDX_E] = e

    # Either enforce merge-policy output type, or fallback to dominant-energy type.
    dom = t1 if t1[IDX_E] >= t2[IDX_E] else t2
    out[IDX_PID0:IDX_PID4 + 1] = 0.0
    merged_type = int(merged_type_override) if merged_type_override is not None else infer_type_id(dom)
    if merged_type != TYPE_UNK:
        out[IDX_PID0 + merged_type] = 1.0

    # Charge only if resulting merged token is track-like.
    charge_sum = float(t1[IDX_CHARGE] + t2[IDX_CHARGE])
    if merged_type in (TYPE_CH, TYPE_ELE, TYPE_MU):
        out[IDX_CHARGE] = float(np.clip(np.round(charge_sum), -1.0, 1.0))
    else:
        out[IDX_CHARGE] = 0.0

    # Track quantities: energy-weighted average over track-like inputs.
    ws = []
    d0s = []
    d0errs = []
    dzs = []
    dzerrs = []
    for t in (t1, t2):
        tt = infer_type_id(t)
        if tt in (TYPE_CH, TYPE_ELE, TYPE_MU):
            w = max(float(t[IDX_E]), 1e-8)
            ws.append(w)
            d0s.append(float(t[IDX_D0]))
            d0errs.append(float(t[IDX_D0ERR]))
            dzs.append(float(t[IDX_DZ]))
            dzerrs.append(float(t[IDX_DZERR]))
    if ws:
        w = np.asarray(ws, dtype=np.float64)
        wn = w / max(np.sum(w), 1e-8)
        out[IDX_D0] = float(np.sum(wn * np.asarray(d0s)))
        out[IDX_D0ERR] = float(np.sum(wn * np.asarray(d0errs)))
        out[IDX_DZ] = float(np.sum(wn * np.asarray(dzs)))
        out[IDX_DZERR] = float(np.sum(wn * np.asarray(dzerrs)))
    else:
        out[IDX_D0:IDX_DZERR + 1] = 0.0

    return out


def get_type_config() -> Dict[str, np.ndarray]:
    # Per type: CH, NH, PHO, ELE, MU, UNK
    plateau_barrel = np.array([0.99, 0.95, 0.97, 0.98, 0.985, 0.94], dtype=np.float64)
    plateau_endcap = np.array([0.97, 0.90, 0.93, 0.95, 0.965, 0.90], dtype=np.float64)
    turnon_pt = np.array([0.7, 1.2, 1.0, 0.8, 0.7, 1.2], dtype=np.float64)
    width_pt = np.array([0.35, 0.55, 0.45, 0.35, 0.30, 0.55], dtype=np.float64)

    smear_pt = np.array([0.035, 0.085, 0.060, 0.030, 0.025, 0.090], dtype=np.float64)
    smear_eta = np.array([0.0018, 0.0055, 0.0040, 0.0018, 0.0016, 0.0055], dtype=np.float64)
    smear_phi = np.array([0.0018, 0.0055, 0.0042, 0.0018, 0.0018, 0.0055], dtype=np.float64)

    reassign_prob = np.array([0.03, 0.12, 0.09, 0.05, 0.03, 0.10], dtype=np.float64)

    # Pairwise merge radius/probability.
    r = np.full((N_TYPES, N_TYPES), 0.008, dtype=np.float64)
    p = np.full((N_TYPES, N_TYPES), 0.12, dtype=np.float64)
    for i in range(N_TYPES):
        r[i, i] = 0.010
        p[i, i] = 0.18

    # Stronger neutral merging
    r[TYPE_NH, TYPE_NH] = r[TYPE_GAM, TYPE_GAM] = 0.028
    p[TYPE_NH, TYPE_NH] = p[TYPE_GAM, TYPE_GAM] = 0.65
    r[TYPE_NH, TYPE_GAM] = r[TYPE_GAM, TYPE_NH] = 0.026
    p[TYPE_NH, TYPE_GAM] = p[TYPE_GAM, TYPE_NH] = 0.60

    # Charged tracks merge less aggressively.
    r[TYPE_CH, :] = np.minimum(r[TYPE_CH, :], 0.012)
    r[:, TYPE_CH] = np.minimum(r[:, TYPE_CH], 0.012)
    p[TYPE_CH, :] = np.minimum(p[TYPE_CH, :], 0.22)
    p[:, TYPE_CH] = np.minimum(p[:, TYPE_CH], 0.22)

    # Muons rarely merge.
    r[TYPE_MU, :] = np.minimum(r[TYPE_MU, :], 0.006)
    r[:, TYPE_MU] = np.minimum(r[:, TYPE_MU], 0.006)
    p[TYPE_MU, :] = np.minimum(p[TYPE_MU, :], 0.06)
    p[:, TYPE_MU] = np.minimum(p[:, TYPE_MU], 0.06)

    # Electron/photon moderate sharing.
    r[TYPE_ELE, TYPE_GAM] = r[TYPE_GAM, TYPE_ELE] = 0.015
    p[TYPE_ELE, TYPE_GAM] = p[TYPE_GAM, TYPE_ELE] = 0.35

    return {
        "plateau_barrel": plateau_barrel,
        "plateau_endcap": plateau_endcap,
        "turnon_pt": turnon_pt,
        "width_pt": width_pt,
        "smear_pt": smear_pt,
        "smear_eta": smear_eta,
        "smear_phi": smear_phi,
        "reassign_prob": reassign_prob,
        "merge_radius": r,
        "merge_prob": p,
    }


def apply_hlt_corruption_single_jet(
    tok: np.ndarray,
    msk: np.ndarray,
    params: HLTParams,
    rng: np.random.RandomState,
    tcfg: Dict[str, np.ndarray],
    max_constits: int,
    return_provenance: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]] | Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, np.ndarray]]:
    diag = {
        "n_offline": 0.0,
        "n_after_eff": 0.0,
        "n_after_threshold": 0.0,
        "n_after_merge": 0.0,
        "drop_eff": 0.0,
        "drop_threshold": 0.0,
        "drop_merge": 0.0,
        "drop_total": 0.0,
        "merge_count": 0.0,
    }
    valid = tok[msk].copy()
    diag["n_offline"] = float(len(valid))
    empty_prov = _empty_unmerge_provenance(max_constits)
    if len(valid) == 0:
        out = (
            np.zeros((max_constits, RAW_DIM), dtype=np.float32),
            np.zeros((max_constits,), dtype=bool),
            diag,
        )
        if return_provenance:
            return out + (empty_prov,)
        return out

    # 1) Efficiency loss
    pt = np.maximum(valid[:, IDX_PT], 1e-8)
    abseta = np.abs(valid[:, IDX_ETA])
    plateau = np.where(abseta < 1.5, params.eff_plateau_barrel, params.eff_plateau_endcap)
    sig = 1.0 / (1.0 + np.exp(-(pt - params.eff_turnon_pt) / max(params.eff_width_pt, 1e-6)))
    keep_prob = np.clip(plateau * sig, 0.03, 0.999)
    keep = rng.rand(len(valid)) < keep_prob
    valid = valid[keep]
    diag["n_after_eff"] = float(len(valid))
    if len(valid) == 0:
        diag["drop_eff"] = diag["n_offline"]
        diag["drop_total"] = diag["n_offline"]
        out = (
            np.zeros((max_constits, RAW_DIM), dtype=np.float32),
            np.zeros((max_constits,), dtype=bool),
            diag,
        )
        if return_provenance:
            return out + (empty_prov,)
        return out

    # 2) Smearing + reassignment
    for i in range(len(valid)):
        tt = infer_type_id(valid[i])
        s_pt = float(tcfg["smear_pt"][tt] * params.smear_scale)
        s_eta = float(tcfg["smear_eta"][tt] * params.smear_scale)
        s_phi = float(tcfg["smear_phi"][tt] * params.smear_scale)
        old_pt = max(float(valid[i, IDX_PT]), 1e-8)

        pt_scale = 1.0 + rng.normal(0.0, s_pt)
        pt_new = max(old_pt * pt_scale, 1e-6)
        eta_new = float(valid[i, IDX_ETA] + rng.normal(0.0, s_eta))
        phi_new = float(valid[i, IDX_PHI] + rng.normal(0.0, s_phi))

        rp = float(tcfg["reassign_prob"][tt] * params.reassign_scale)
        if rng.rand() < rp:
            eta_new += rng.normal(0.0, 0.012)
            phi_new += rng.normal(0.0, 0.012)

        valid[i, IDX_PT] = pt_new
        valid[i, IDX_ETA] = np.clip(eta_new, -5.0, 5.0)
        valid[i, IDX_PHI] = wrap_phi(np.array([phi_new], dtype=np.float64))[0]
        valid[i, IDX_E] = max(pt_new * math.cosh(float(valid[i, IDX_ETA])), 1e-8)

    # 3) HLT threshold
    valid = valid[valid[:, IDX_PT] >= params.hlt_pt_threshold]
    diag["n_after_threshold"] = float(len(valid))
    if len(valid) == 0:
        diag["drop_eff"] = max(diag["n_offline"] - diag["n_after_eff"], 0.0)
        diag["drop_threshold"] = max(diag["n_after_eff"] - diag["n_after_threshold"], 0.0)
        diag["drop_total"] = diag["n_offline"]
        out = (
            np.zeros((max_constits, RAW_DIM), dtype=np.float32),
            np.zeros((max_constits,), dtype=bool),
            diag,
        )
        if return_provenance:
            return out + (empty_prov,)
        return out

    # Per-token provenance metadata for constrained unmerge supervision.
    def _default_meta() -> Dict[str, object]:
        return {
            "is_merged_token": False,
            "split_mode_target": MERGE_MODE_NONE,
            "child_type_a_target": TYPE_UNK,
            "child_type_b_target": TYPE_UNK,
            "child_attr_a_target": np.zeros((5,), dtype=np.float32),
            "child_attr_b_target": np.zeros((5,), dtype=np.float32),
        }

    meta: List[Dict[str, object]] = [_default_meta() for _ in range(len(valid))]

    # 4) Merging (greedy by best pair score)
    steps = 0
    merge_count = 0
    while len(valid) > 1 and steps < 512:
        steps += 1
        n = len(valid)
        best = None
        best_score = -1.0
        for i in range(n):
            ti = infer_type_id(valid[i])
            eta_i = float(valid[i, IDX_ETA])
            phi_i = float(valid[i, IDX_PHI])
            pt_i = max(float(valid[i, IDX_PT]), 1e-8)
            for j in range(i + 1, n):
                tj = infer_type_id(valid[j])
                deta = eta_i - float(valid[j, IDX_ETA])
                dphi = math.atan2(math.sin(phi_i - float(valid[j, IDX_PHI])), math.cos(phi_i - float(valid[j, IDX_PHI])))
                dr = math.sqrt(deta * deta + dphi * dphi)
                allow_merge, _out_type = allowed_merge_and_output_type(ti, tj)
                if not allow_merge:
                    continue

                rmax = float(tcfg["merge_radius"][ti, tj])
                if dr >= rmax:
                    continue
                pt_j = max(float(valid[j, IDX_PT]), 1e-8)
                pt_sim = math.exp(-abs(math.log(pt_i / pt_j)))
                dr_term = math.exp(-((dr / max(rmax, 1e-6)) ** 2))
                base_p = float(tcfg["merge_prob"][ti, tj])
                score = base_p * params.merge_prob_scale * pt_sim * dr_term
                if score > best_score:
                    best_score = score
                    best = (i, j, _out_type)
        if best is None:
            break
        if rng.rand() > np.clip(best_score, 0.0, 0.999):
            break
        i, j, out_type = best
        t_i = valid[i].copy()
        t_j = valid[j].copy()
        m_i = meta[i]
        m_j = meta[j]

        merged = merge_two_tokens(t_i, t_j, merged_type_override=out_type)
        merge_count += 1

        # Supervise only simple first-level merges: both parents are original (not already merged).
        split_mode_target = MERGE_MODE_NONE
        child_type_a_target = TYPE_UNK
        child_type_b_target = TYPE_UNK
        child_attr_a_target = np.zeros((5,), dtype=np.float32)
        child_attr_b_target = np.zeros((5,), dtype=np.float32)
        if (not bool(m_i["is_merged_token"])) and (not bool(m_j["is_merged_token"])):
            ti = infer_type_id(t_i)
            tj = infer_type_id(t_j)
            mode = infer_merge_mode(ti, tj)
            if mode != MERGE_MODE_NONE:
                split_mode_target = int(mode)
                # Deterministic child ordering: higher-energy source first.
                a, b = (t_i, t_j) if float(t_i[IDX_E]) >= float(t_j[IDX_E]) else (t_j, t_i)
                child_type_a_target = int(infer_type_id(a))
                child_type_b_target = int(infer_type_id(b))
                child_attr_a_target = a[[IDX_CHARGE, IDX_D0, IDX_D0ERR, IDX_DZ, IDX_DZERR]].astype(np.float32)
                child_attr_b_target = b[[IDX_CHARGE, IDX_D0, IDX_D0ERR, IDX_DZ, IDX_DZERR]].astype(np.float32)

        merged_meta = {
            "is_merged_token": True,
            "split_mode_target": int(split_mode_target),
            "child_type_a_target": int(child_type_a_target),
            "child_type_b_target": int(child_type_b_target),
            "child_attr_a_target": child_attr_a_target,
            "child_attr_b_target": child_attr_b_target,
        }

        keep = [k for k in range(len(valid)) if k not in (i, j)]
        if keep:
            valid = np.concatenate([valid[keep], merged[None, :]], axis=0)
            meta = [meta[k] for k in keep] + [merged_meta]
        else:
            valid = merged[None, :]
            meta = [merged_meta]

    order = np.argsort(-valid[:, IDX_PT])
    valid = valid[order]
    meta = [meta[int(k)] for k in order]
    take = min(len(valid), max_constits)
    out_tok = np.zeros((max_constits, RAW_DIM), dtype=np.float32)
    out_mask = np.zeros((max_constits,), dtype=bool)
    out_tok[:take] = valid[:take]
    out_mask[:take] = True
    diag["n_after_merge"] = float(len(valid))
    # In practice n_after_merge <= n_after_threshold, but keep numerically robust.
    diag["drop_eff"] = max(diag["n_offline"] - diag["n_after_eff"], 0.0)
    diag["drop_threshold"] = max(diag["n_after_eff"] - diag["n_after_threshold"], 0.0)
    diag["drop_merge"] = max(diag["n_after_threshold"] - diag["n_after_merge"], 0.0)
    diag["drop_total"] = max(diag["n_offline"] - diag["n_after_merge"], 0.0)
    diag["merge_count"] = float(merge_count)

    if not return_provenance:
        return out_tok, out_mask, diag

    prov = _empty_unmerge_provenance(max_constits)
    for i in range(take):
        prov["parent_type_id"][i] = int(infer_type_id(valid[i]))
        m = meta[i]
        mode = int(m["split_mode_target"])
        prov["split_mode_target"][i] = mode
        prov["split_target_mask"][i] = bool(mode != MERGE_MODE_NONE)
        prov["child_type_a_target"][i] = int(m["child_type_a_target"])
        prov["child_type_b_target"][i] = int(m["child_type_b_target"])
        prov["child_attr_a_target"][i] = np.asarray(m["child_attr_a_target"], dtype=np.float32)
        prov["child_attr_b_target"][i] = np.asarray(m["child_attr_b_target"], dtype=np.float32)
    return out_tok, out_mask, diag, prov


def build_hlt_view(
    tok: np.ndarray,
    msk: np.ndarray,
    params: HLTParams,
    seed: int,
    return_provenance: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]] | Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    n = len(tok)
    out_tok = np.zeros_like(tok, dtype=np.float32)
    out_msk = np.zeros_like(msk, dtype=bool)
    tcfg = get_type_config()
    diag_rows: List[Dict[str, float]] = []
    prov_rows: List[Dict[str, np.ndarray]] = []
    for i in tqdm(range(n), desc="Applying HLT-like corruption"):
        rng = np.random.RandomState(seed + i * 17 + 13)
        out = apply_hlt_corruption_single_jet(
            tok[i],
            msk[i],
            params,
            rng,
            tcfg,
            tok.shape[1],
            return_provenance=bool(return_provenance),
        )
        if return_provenance:
            ti, mi, di, pi = out
            prov_rows.append(pi)
        else:
            ti, mi, di = out
        out_tok[i] = ti
        out_msk[i] = mi
        diag_rows.append(di)

    keys = [
        "n_offline",
        "n_after_eff",
        "n_after_threshold",
        "n_after_merge",
        "drop_eff",
        "drop_threshold",
        "drop_merge",
        "drop_total",
        "merge_count",
    ]
    per_jet = {k: np.array([row[k] for row in diag_rows], dtype=np.float32) for k in keys}
    if not return_provenance:
        return out_tok, out_msk, per_jet

    if not prov_rows:
        prov = _empty_unmerge_provenance(tok.shape[1])
        prov = {k: np.repeat(v[None, ...], n, axis=0) for k, v in prov.items()}
    else:
        pkeys = list(prov_rows[0].keys())
        prov = {k: np.stack([row[k] for row in prov_rows], axis=0) for k in pkeys}
    return out_tok, out_msk, per_jet, prov


def summarize_hlt_diagnostics(per_jet: Dict[str, np.ndarray]) -> Dict[str, float]:
    def m(key: str) -> float:
        arr = per_jet.get(key)
        if arr is None or arr.size == 0:
            return float("nan")
        return float(np.mean(arr))

    n_off = per_jet.get("n_offline", np.array([], dtype=np.float32))
    n_merge = per_jet.get("n_after_merge", np.array([], dtype=np.float32))
    merge_count = per_jet.get("merge_count", np.array([], dtype=np.float32))
    if n_off.size == 0:
        return {}

    drop_total_sum = float(np.sum(np.maximum(n_off - n_merge, 0.0)))
    drop_eff_sum = float(np.sum(np.maximum(per_jet["drop_eff"], 0.0)))
    drop_thr_sum = float(np.sum(np.maximum(per_jet["drop_threshold"], 0.0)))
    drop_mer_sum = float(np.sum(np.maximum(per_jet["drop_merge"], 0.0)))
    denom = max(drop_total_sum, 1e-12)

    return {
        "n_offline_mean": m("n_offline"),
        "n_after_eff_mean": m("n_after_eff"),
        "n_after_threshold_mean": m("n_after_threshold"),
        "n_after_merge_mean": m("n_after_merge"),
        "drop_eff_mean": m("drop_eff"),
        "drop_threshold_mean": m("drop_threshold"),
        "drop_merge_mean": m("drop_merge"),
        "drop_total_mean": m("drop_total"),
        "drop_eff_share": float(drop_eff_sum / denom),
        "drop_threshold_share": float(drop_thr_sum / denom),
        "drop_merge_share": float(drop_mer_sum / denom),
        "mean_merges_per_jet": m("merge_count"),
        "frac_jets_with_any_merge": float(np.mean(merge_count > 0.0)),
    }


def compute_features(
    raw_tok: np.ndarray,
    mask: np.ndarray,
    feature_mode: str,
    feature_preprocessing: str = "canonical",
) -> np.ndarray:
    pt = np.maximum(raw_tok[:, :, IDX_PT], 1e-8)
    eta = np.clip(raw_tok[:, :, IDX_ETA], -5.0, 5.0)
    phi = raw_tok[:, :, IDX_PHI]
    ene = np.maximum(raw_tok[:, :, IDX_E], 1e-8)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    w = mask.astype(np.float32)
    jet_px = (px * w).sum(axis=1, keepdims=True)
    jet_py = (py * w).sum(axis=1, keepdims=True)
    jet_pz = (pz * w).sum(axis=1, keepdims=True)
    jet_e = (ene * w).sum(axis=1, keepdims=True)
    jet_pt = np.sqrt(jet_px * jet_px + jet_py * jet_py) + 1e-8
    jet_p = np.sqrt(jet_px * jet_px + jet_py * jet_py + jet_pz * jet_pz) + 1e-8
    jet_eta = 0.5 * np.log(np.clip((jet_p + jet_pz) / np.maximum(jet_p - jet_pz, 1e-8), 1e-8, 1e8))
    jet_phi = np.arctan2(jet_py, jet_px)

    d_eta = eta - jet_eta
    d_phi = np.arctan2(np.sin(phi - jet_phi), np.cos(phi - jet_phi))
    log_pt = np.log(pt + 1e-8)
    log_e = np.log(ene + 1e-8)
    log_pt_rel = np.log(pt / jet_pt + 1e-8)
    log_e_rel = np.log(ene / (jet_e + 1e-8) + 1e-8)
    d_r = np.sqrt(d_eta * d_eta + d_phi * d_phi)

    if feature_preprocessing == "legacy":
        # Historical repo layout:
        # [d_eta, d_phi, log_pt, log_e, log_pt_rel, log_e_rel, d_r] + auxiliaries.
        core = np.stack([d_eta, d_phi, log_pt, log_e, log_pt_rel, log_e_rel, d_r], axis=-1).astype(np.float32)
        if feature_mode == "kin":
            feat = core
        elif feature_mode == "kinpid":
            aux = raw_tok[:, :, IDX_CHARGE:IDX_PID4 + 1]
            feat = np.concatenate([core, aux], axis=-1).astype(np.float32)
        else:
            aux = raw_tok[:, :, IDX_CHARGE:IDX_DZERR + 1]
            feat = np.concatenate([core, aux], axis=-1).astype(np.float32)
    else:
        # Canonical-style preprocessing (JetClass YAML, manual mode):
        # part_pt_log:   (log_pt - 1.7) * 0.7
        # part_e_log:    (log_e - 2.0) * 0.7
        # part_logptrel: (log_pt_rel - (-4.7)) * 0.7
        # part_logerel:  (log_e_rel - (-4.7)) * 0.7
        # part_deltaR:   (d_r - 0.2) * 4.0
        # plus tanh(d0), tanh(dz), and error clips to [0, 1].
        f_pt_log = np.clip((log_pt - 1.7) * 0.7, -5.0, 5.0)
        f_e_log = np.clip((log_e - 2.0) * 0.7, -5.0, 5.0)
        f_logptrel = np.clip((log_pt_rel + 4.7) * 0.7, -5.0, 5.0)
        f_logerel = np.clip((log_e_rel + 4.7) * 0.7, -5.0, 5.0)
        f_delta_r = np.clip((d_r - 0.2) * 4.0, -5.0, 5.0)
        f_d_eta = np.clip(d_eta, -5.0, 5.0)
        f_d_phi = np.clip(d_phi, -5.0, 5.0)

        charge = np.clip(raw_tok[:, :, IDX_CHARGE], -5.0, 5.0)
        pid = np.clip(raw_tok[:, :, IDX_PID0:IDX_PID4 + 1], -5.0, 5.0)
        d0 = np.tanh(raw_tok[:, :, IDX_D0])
        d0err = np.clip(raw_tok[:, :, IDX_D0ERR], 0.0, 1.0)
        dz = np.tanh(raw_tok[:, :, IDX_DZ])
        dzerr = np.clip(raw_tok[:, :, IDX_DZERR], 0.0, 1.0)

        if feature_mode == "kin":
            feat = np.stack(
                [f_pt_log, f_e_log, f_logptrel, f_logerel, f_delta_r, f_d_eta, f_d_phi],
                axis=-1,
            ).astype(np.float32)
        elif feature_mode == "kinpid":
            feat = np.concatenate(
                [
                    np.stack([f_pt_log, f_e_log, f_logptrel, f_logerel, f_delta_r], axis=-1),
                    charge[:, :, None],
                    pid,
                    np.stack([f_d_eta, f_d_phi], axis=-1),
                ],
                axis=-1,
            ).astype(np.float32)
        else:
            feat = np.concatenate(
                [
                    np.stack([f_pt_log, f_e_log, f_logptrel, f_logerel, f_delta_r], axis=-1),
                    charge[:, :, None],
                    pid,
                    d0[:, :, None],
                    d0err[:, :, None],
                    dz[:, :, None],
                    dzerr[:, :, None],
                    np.stack([f_d_eta, f_d_phi], axis=-1),
                ],
                axis=-1,
            ).astype(np.float32)

    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    feat[~mask] = 0.0
    return feat.astype(np.float32)


def get_mean_std(feat: np.ndarray, mask: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    d = feat.shape[-1]
    means = np.zeros((d,), dtype=np.float64)
    stds = np.ones((d,), dtype=np.float64)
    for k in range(d):
        vals = feat[idx, :, k][mask[idx]]
        if vals.size == 0:
            means[k] = 0.0
            stds[k] = 1.0
        else:
            means[k] = float(np.mean(vals))
            std = float(np.std(vals))
            stds[k] = std if std > 1e-8 else 1.0
    return means.astype(np.float32), stds.astype(np.float32)


def standardize(feat: np.ndarray, mask: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = (feat - mean[None, None, :]) / std[None, None, :]
    out = np.clip(out, -10.0, 10.0)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out[~mask] = 0.0
    return out.astype(np.float32)


class JetDataset(Dataset):
    def __init__(self, feat: np.ndarray, mask: np.ndarray, label: np.ndarray):
        self.feat = torch.tensor(feat, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.label = torch.tensor(label, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.label.shape[0])

    def __getitem__(self, i: int):
        return {"feat": self.feat[i], "mask": self.mask[i], "label": self.label[i]}


class JetClassTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        mask_safe = mask.clone()
        empty = ~mask_safe.any(dim=1)
        if empty.any():
            mask_safe[empty, 0] = True
        h = self.input_proj(x.reshape(-1, self.input_dim)).view(bsz, seq_len, -1)
        h = self.encoder(h, src_key_padding_mask=~mask_safe)
        query = self.pool_query.expand(bsz, -1, -1)
        pooled, _ = self.pool_attn(query, h, h, key_padding_mask=~mask_safe, need_weights=False)
        z = self.norm(pooled.squeeze(1))
        return self.head(z)


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def get_scheduler(opt, warmup_epochs: int, total_epochs: int):
    def lr_lambda(ep: int) -> float:
        if ep < warmup_epochs:
            return (ep + 1) / max(warmup_epochs, 1)
        x = (ep - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * x))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def macro_auc_ovr(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    y_1h = np.eye(n_classes, dtype=np.int64)[y_true]
    try:
        return float(roc_auc_score(y_1h, probs, average="macro", multi_class="ovr"))
    except ValueError:
        return float("nan")


def eval_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: Sequence[str],
    background_class: str,
    target_class: str,
) -> Dict[str, float]:
    class_names = list(class_names)

    def resolve_name(name: str) -> str:
        if name in class_names:
            return name
        alias = CLASS_NAME_ALIASES.get(name, name)
        return alias if alias in class_names else name

    background_class = resolve_name(background_class)
    target_class = resolve_name(target_class)

    pred = np.argmax(probs, axis=1)
    acc = float((pred == y_true).mean())
    n_classes = probs.shape[1]
    auc_macro = macro_auc_ovr(y_true, probs, n_classes)
    out = {"acc": acc, "auc_macro_ovr": auc_macro}

    if background_class in class_names:
        bg_idx = class_names.index(background_class)
        y_sig = (y_true != bg_idx).astype(np.int64)
        score_sig = np.clip(1.0 - probs[:, bg_idx], 0.0, 1.0)
        if len(np.unique(y_sig)) > 1:
            fpr, tpr, _ = roc_curve(y_sig, score_sig)
            out["signal_vs_bg_fpr50"] = float(fpr_at_target_tpr(fpr, tpr, 0.50))
            out["signal_vs_bg_auc"] = float(roc_auc_score(y_sig, score_sig))
        else:
            out["signal_vs_bg_fpr50"] = float("nan")
            out["signal_vs_bg_auc"] = float("nan")
    else:
        out["signal_vs_bg_fpr50"] = float("nan")
        out["signal_vs_bg_auc"] = float("nan")

    # Professor-style binarized score: score_target / (score_target + score_bg)
    if target_class in class_names and background_class in class_names:
        tgt_idx = class_names.index(target_class)
        bg_idx = class_names.index(background_class)
        den = np.clip(probs[:, tgt_idx] + probs[:, bg_idx], 1e-8, None)
        score_tgt_over_pair = np.clip(probs[:, tgt_idx] / den, 0.0, 1.0)
        pair_mask = np.logical_or(y_true == tgt_idx, y_true == bg_idx)
        if pair_mask.any():
            y_pair = (y_true[pair_mask] == tgt_idx).astype(np.int64)
            s_pair = score_tgt_over_pair[pair_mask]
            if len(np.unique(y_pair)) > 1:
                fpr, tpr, _ = roc_curve(y_pair, s_pair)
                out["target_vs_bg_ratio_fpr50"] = float(fpr_at_target_tpr(fpr, tpr, 0.50))
                out["target_vs_bg_ratio_auc"] = float(roc_auc_score(y_pair, s_pair))
            else:
                out["target_vs_bg_ratio_fpr50"] = float("nan")
                out["target_vs_bg_ratio_auc"] = float("nan")
        else:
            out["target_vs_bg_ratio_fpr50"] = float("nan")
            out["target_vs_bg_ratio_auc"] = float("nan")
    else:
        out["target_vs_bg_ratio_fpr50"] = float("nan")
        out["target_vs_bg_ratio_auc"] = float("nan")
    return out


def train_epoch(model, loader, opt, device, n_classes: int) -> Dict[str, float]:
    model.train()
    total = 0.0
    n = 0
    all_probs = []
    all_y = []
    for b in loader:
        x = b["feat"].to(device, non_blocking=True)
        m = b["mask"].to(device, non_blocking=True)
        y = b["label"].to(device, non_blocking=True)
        opt.zero_grad()
        logits = model(x, m)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        bs = int(y.shape[0])
        total += float(loss.item()) * bs
        n += bs
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_y.append(y.detach().cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    ys = np.concatenate(all_y, axis=0)
    pred = np.argmax(probs, axis=1)
    acc = float((pred == ys).mean())
    auc_macro = macro_auc_ovr(ys, probs, n_classes)
    return {"loss": total / max(n, 1), "acc": acc, "auc_macro_ovr": auc_macro}


@torch.no_grad()
def eval_epoch(
    model,
    loader,
    device,
    class_names: Sequence[str],
    background_class: str,
    target_class: str,
) -> Dict[str, float]:
    model.eval()
    total = 0.0
    n = 0
    all_probs = []
    all_y = []
    for b in loader:
        x = b["feat"].to(device, non_blocking=True)
        m = b["mask"].to(device, non_blocking=True)
        y = b["label"].to(device, non_blocking=True)
        logits = model(x, m)
        loss = F.cross_entropy(logits, y)
        bs = int(y.shape[0])
        total += float(loss.item()) * bs
        n += bs
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_y.append(y.cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    ys = np.concatenate(all_y, axis=0)
    out = {"loss": total / max(n, 1)}
    out.update(
        eval_metrics(
            ys,
            probs,
            class_names=class_names,
            background_class=background_class,
            target_class=target_class,
        )
    )
    return out


def fit_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    n_classes: int,
    class_names: Sequence[str],
    background_class: str,
    target_class: str,
    args: argparse.Namespace,
    tag: str,
    save_dir: Path,
) -> Tuple[nn.Module, Dict[str, float], List[Dict[str, float]]]:
    device = torch.device(args.device if torch.cuda.is_available() or str(args.device).startswith("cpu") else "cpu")
    model = JetClassTransformer(
        input_dim=input_dim,
        n_classes=n_classes,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = get_scheduler(opt, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs)

    best_state = None
    best_metric = float("-inf")
    wait = 0
    hist: List[Dict[str, float]] = []

    for ep in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, opt, device=device, n_classes=n_classes)
        va = eval_epoch(
            model,
            val_loader,
            device=device,
            class_names=class_names,
            background_class=background_class,
            target_class=target_class,
        )
        sch.step()
        metric = va["auc_macro_ovr"] if np.isfinite(va["auc_macro_ovr"]) else va["acc"]
        improved = float(metric) > best_metric
        if improved:
            best_metric = float(metric)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        row = {
            "epoch": ep,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "train_auc_macro_ovr": tr["auc_macro_ovr"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "val_auc_macro_ovr": va["auc_macro_ovr"],
            "val_signal_vs_bg_fpr50": va["signal_vs_bg_fpr50"],
            "val_target_vs_bg_ratio_fpr50": va["target_vs_bg_ratio_fpr50"],
        }
        hist.append(row)
        print(
            f"{tag} ep {ep}: "
            f"train(loss/acc/auc)={tr['loss']:.4f}/{tr['acc']:.4f}/{tr['auc_macro_ovr']:.4f} "
            f"val(loss/acc/auc/fpr50_sigbg/fpr50_ratio)="
            f"{va['loss']:.4f}/{va['acc']:.4f}/{va['auc_macro_ovr']:.4f}/"
            f"{va['signal_vs_bg_fpr50']:.6f}/{va['target_vs_bg_ratio_fpr50']:.6f} "
            f"best={best_metric:.4f}"
        )
        if wait >= args.patience:
            print(f"Early stopping {tag} at epoch {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt = save_dir / f"{tag}_best.pt"
    torch.save(model.state_dict(), ckpt)
    best = eval_epoch(
        model,
        val_loader,
        device=device,
        class_names=class_names,
        background_class=background_class,
        target_class=target_class,
    )
    return model, best, hist


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    save_dir = (args.save_dir / args.run_name).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Save dir: {save_dir}")

    files_by_class = collect_files_by_class(args.data_dir.resolve())
    if str(args.class_assignment) == "canonical_labels":
        class_names = list(CANONICAL_CLASS_ORDER)
    else:
        class_names = sorted(files_by_class.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print("Classes:")
    for c in class_names:
        if c in files_by_class:
            print(f"  {c:12s} : {len(files_by_class[c])} files")
        else:
            # Canonical mode uses shared file pool; files are not grouped by canonical label.
            print(f"  {c:12s} : label_* branch")

    tr_files, va_files, te_files = split_files_by_class(
        files_by_class,
        n_train=args.train_files_per_class,
        n_val=args.val_files_per_class,
        n_test=args.test_files_per_class,
        shuffle=args.shuffle_files,
        seed=args.seed,
    )

    print("Loading train split...")
    tr_tok, tr_mask, tr_y = load_split(
        tr_files,
        n_total=args.n_train_jets,
        max_constits=args.max_constits,
        class_to_idx=class_to_idx,
        seed=args.seed + 101,
        class_assignment=args.class_assignment,
    )
    print("Loading val split...")
    va_tok, va_mask, va_y = load_split(
        va_files,
        n_total=args.n_val_jets,
        max_constits=args.max_constits,
        class_to_idx=class_to_idx,
        seed=args.seed + 202,
        class_assignment=args.class_assignment,
    )
    print("Loading test split...")
    te_tok, te_mask, te_y = load_split(
        te_files,
        n_total=args.n_test_jets,
        max_constits=args.max_constits,
        class_to_idx=class_to_idx,
        seed=args.seed + 303,
        class_assignment=args.class_assignment,
    )

    print(
        f"Loaded jets: train={len(tr_y)}, val={len(va_y)}, test={len(te_y)} | "
        f"mean constituents train={tr_mask.sum(axis=1).mean():.2f}"
    )

    hlt_params = HLTParams(
        hlt_pt_threshold=float(args.hlt_pt_threshold),
        merge_prob_scale=float(args.merge_prob_scale),
        reassign_scale=float(args.reassign_scale),
        smear_scale=float(args.smear_scale),
        eff_plateau_barrel=float(args.eff_plateau_barrel),
        eff_plateau_endcap=float(args.eff_plateau_endcap),
        eff_turnon_pt=float(args.eff_turnon_pt),
        eff_width_pt=float(args.eff_width_pt),
    )
    print("Building HLT-like corrupted splits...")
    tr_hlt_tok, tr_hlt_mask, tr_hlt_diag = build_hlt_view(tr_tok, tr_mask, params=hlt_params, seed=args.seed + 1001)
    va_hlt_tok, va_hlt_mask, va_hlt_diag = build_hlt_view(va_tok, va_mask, params=hlt_params, seed=args.seed + 1002)
    te_hlt_tok, te_hlt_mask, te_hlt_diag = build_hlt_view(te_tok, te_mask, params=hlt_params, seed=args.seed + 1003)

    tr_hlt_diag_summary = summarize_hlt_diagnostics(tr_hlt_diag)
    va_hlt_diag_summary = summarize_hlt_diagnostics(va_hlt_diag)
    te_hlt_diag_summary = summarize_hlt_diagnostics(te_hlt_diag)

    print(
        "Constituent means | offline/hlt: "
        f"train={tr_mask.sum(axis=1).mean():.2f}/{tr_hlt_mask.sum(axis=1).mean():.2f} "
        f"val={va_mask.sum(axis=1).mean():.2f}/{va_hlt_mask.sum(axis=1).mean():.2f} "
        f"test={te_mask.sum(axis=1).mean():.2f}/{te_hlt_mask.sum(axis=1).mean():.2f}"
    )
    print(
        "HLT drop decomposition (train): "
        f"eff={tr_hlt_diag_summary.get('drop_eff_share', float('nan')):.3f}, "
        f"thr={tr_hlt_diag_summary.get('drop_threshold_share', float('nan')):.3f}, "
        f"merge={tr_hlt_diag_summary.get('drop_merge_share', float('nan')):.3f}, "
        f"mean_merges/jet={tr_hlt_diag_summary.get('mean_merges_per_jet', float('nan')):.3f}"
    )

    # Features + standardization stats from train offline only (shared transform).
    tr_feat_off = compute_features(
        tr_tok,
        tr_mask,
        feature_mode=args.feature_mode,
        feature_preprocessing=args.feature_preprocessing,
    )
    va_feat_off = compute_features(
        va_tok,
        va_mask,
        feature_mode=args.feature_mode,
        feature_preprocessing=args.feature_preprocessing,
    )
    te_feat_off = compute_features(
        te_tok,
        te_mask,
        feature_mode=args.feature_mode,
        feature_preprocessing=args.feature_preprocessing,
    )

    tr_feat_hlt = compute_features(
        tr_hlt_tok,
        tr_hlt_mask,
        feature_mode=args.feature_mode,
        feature_preprocessing=args.feature_preprocessing,
    )
    va_feat_hlt = compute_features(
        va_hlt_tok,
        va_hlt_mask,
        feature_mode=args.feature_mode,
        feature_preprocessing=args.feature_preprocessing,
    )
    te_feat_hlt = compute_features(
        te_hlt_tok,
        te_hlt_mask,
        feature_mode=args.feature_mode,
        feature_preprocessing=args.feature_preprocessing,
    )

    idx_all = np.arange(len(tr_y))
    if str(args.feature_preprocessing) == "canonical":
        # Canonical JetClass preprocessing is manually fixed (YAML transforms), so
        # we do not apply extra dataset-dependent standardization here.
        mean = np.zeros((tr_feat_off.shape[-1],), dtype=np.float32)
        std = np.ones((tr_feat_off.shape[-1],), dtype=np.float32)
        standardization_mode = "canonical_manual_fixed"
    else:
        mean, std = get_mean_std(tr_feat_off, tr_mask, idx_all)
        tr_feat_off = standardize(tr_feat_off, tr_mask, mean, std)
        va_feat_off = standardize(va_feat_off, va_mask, mean, std)
        te_feat_off = standardize(te_feat_off, te_mask, mean, std)
        tr_feat_hlt = standardize(tr_feat_hlt, tr_hlt_mask, mean, std)
        va_feat_hlt = standardize(va_feat_hlt, va_hlt_mask, mean, std)
        te_feat_hlt = standardize(te_feat_hlt, te_hlt_mask, mean, std)
        standardization_mode = "learned_train_split"

    ds_tr_off = JetDataset(tr_feat_off, tr_mask, tr_y)
    ds_va_off = JetDataset(va_feat_off, va_mask, va_y)
    ds_te_off = JetDataset(te_feat_off, te_mask, te_y)
    ds_tr_hlt = JetDataset(tr_feat_hlt, tr_hlt_mask, tr_y)
    ds_va_hlt = JetDataset(va_feat_hlt, va_hlt_mask, va_y)
    ds_te_hlt = JetDataset(te_feat_hlt, te_hlt_mask, te_y)

    dl_tr_off = make_loader(ds_tr_off, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dl_va_off = make_loader(ds_va_off, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dl_te_off = make_loader(ds_te_off, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dl_tr_hlt = make_loader(ds_tr_hlt, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dl_va_hlt = make_loader(ds_va_hlt, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dl_te_hlt = make_loader(ds_te_hlt, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    input_dim = tr_feat_off.shape[-1]
    n_classes = len(class_names)

    print("\n=== Training Teacher (offline) ===")
    teacher, teacher_val, hist_teacher = fit_model(
        train_loader=dl_tr_off,
        val_loader=dl_va_off,
        input_dim=input_dim,
        n_classes=n_classes,
        class_names=class_names,
        background_class=args.background_class,
        target_class=args.target_class,
        args=args,
        tag="teacher_offline",
        save_dir=save_dir,
    )

    print("\n=== Training Baseline (HLT-like) ===")
    baseline, baseline_val, hist_baseline = fit_model(
        train_loader=dl_tr_hlt,
        val_loader=dl_va_hlt,
        input_dim=input_dim,
        n_classes=n_classes,
        class_names=class_names,
        background_class=args.background_class,
        target_class=args.target_class,
        args=args,
        tag="baseline_hlt",
        save_dir=save_dir,
    )

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device).startswith("cpu") else "cpu")
    teacher = teacher.to(device)
    baseline = baseline.to(device)
    teacher_test_on_off = eval_epoch(
        teacher,
        dl_te_off,
        device=device,
        class_names=class_names,
        background_class=args.background_class,
        target_class=args.target_class,
    )
    baseline_test_on_hlt = eval_epoch(
        baseline,
        dl_te_hlt,
        device=device,
        class_names=class_names,
        background_class=args.background_class,
        target_class=args.target_class,
    )
    teacher_test_on_hlt = eval_epoch(
        teacher,
        dl_te_hlt,
        device=device,
        class_names=class_names,
        background_class=args.background_class,
        target_class=args.target_class,
    )
    baseline_test_on_off = eval_epoch(
        baseline,
        dl_te_off,
        device=device,
        class_names=class_names,
        background_class=args.background_class,
        target_class=args.target_class,
    )

    summary = {
        "class_names": class_names,
        "n_classes": n_classes,
        "split_sizes": {"train": int(len(tr_y)), "val": int(len(va_y)), "test": int(len(te_y))},
        "feature_mode": args.feature_mode,
        "feature_preprocessing": args.feature_preprocessing,
        "feature_standardization_mode": standardization_mode,
        "class_assignment": args.class_assignment,
        "hlt_params": {
            "hlt_pt_threshold": args.hlt_pt_threshold,
            "merge_prob_scale": args.merge_prob_scale,
            "reassign_scale": args.reassign_scale,
            "smear_scale": args.smear_scale,
            "eff_plateau_barrel": args.eff_plateau_barrel,
            "eff_plateau_endcap": args.eff_plateau_endcap,
            "eff_turnon_pt": args.eff_turnon_pt,
            "eff_width_pt": args.eff_width_pt,
        },
        "binary_metric_config": {
            "target_class": args.target_class,
            "background_class": args.background_class,
        },
        "constituent_stats": {
            "train_offline_mean": float(tr_mask.sum(axis=1).mean()),
            "train_hlt_mean": float(tr_hlt_mask.sum(axis=1).mean()),
            "val_offline_mean": float(va_mask.sum(axis=1).mean()),
            "val_hlt_mean": float(va_hlt_mask.sum(axis=1).mean()),
            "test_offline_mean": float(te_mask.sum(axis=1).mean()),
            "test_hlt_mean": float(te_hlt_mask.sum(axis=1).mean()),
        },
        "hlt_diagnostics": {
            "train": tr_hlt_diag_summary,
            "val": va_hlt_diag_summary,
            "test": te_hlt_diag_summary,
        },
        "teacher_val_best": teacher_val,
        "baseline_val_best": baseline_val,
        "test_metrics": {
            "teacher_on_offline": teacher_test_on_off,
            "teacher_on_hlt": teacher_test_on_hlt,
            "baseline_on_hlt": baseline_test_on_hlt,
            "baseline_on_offline": baseline_test_on_off,
        },
        "history_files": {
            "teacher": "teacher_history.json",
            "baseline": "baseline_history.json",
        },
    }

    with (save_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (save_dir / "teacher_history.json").open("w") as f:
        json.dump(hist_teacher, f, indent=2)
    with (save_dir / "baseline_history.json").open("w") as f:
        json.dump(hist_baseline, f, indent=2)
    with (save_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    np.savez_compressed(save_dir / "hlt_diagnostics_train_perjet.npz", **tr_hlt_diag)
    np.savez_compressed(save_dir / "hlt_diagnostics_val_perjet.npz", **va_hlt_diag)
    np.savez_compressed(save_dir / "hlt_diagnostics_test_perjet.npz", **te_hlt_diag)

    print("\n=== Final Summary ===")
    print(
        f"Teacher test (offline): acc={teacher_test_on_off['acc']:.4f}, "
        f"auc_macro={teacher_test_on_off['auc_macro_ovr']:.4f}, "
        f"fpr50(sig-vs-bg)={teacher_test_on_off['signal_vs_bg_fpr50']:.6f}, "
        f"fpr50({args.target_class}/({args.target_class}+{args.background_class}))="
        f"{teacher_test_on_off['target_vs_bg_ratio_fpr50']:.6f}"
    )
    print(
        f"Baseline test (hlt):   acc={baseline_test_on_hlt['acc']:.4f}, "
        f"auc_macro={baseline_test_on_hlt['auc_macro_ovr']:.4f}, "
        f"fpr50(sig-vs-bg)={baseline_test_on_hlt['signal_vs_bg_fpr50']:.6f}, "
        f"fpr50({args.target_class}/({args.target_class}+{args.background_class}))="
        f"{baseline_test_on_hlt['target_vs_bg_ratio_fpr50']:.6f}"
    )
    print(f"Saved outputs to: {save_dir}")

    return summary


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
