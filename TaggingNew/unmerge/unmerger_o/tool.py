"""
Tools for Ordered unmerger (objectness-only version).

核心思想：
- 不再预测 k；decoder 输出 Kmax 个 slot 的 child_feat，同时输出每个 slot 的 objectness。
- 推理时用 prefix+threshold（或其它策略）决定每个 parent 生成多少 children。

注释约定：
- 代码注释使用中文
- print 的文字使用英文（便于日志/保存）
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, roc_curve


# -----------------------------
# Simple experiment I/O helpers
# -----------------------------


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_config(config: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return p


def save_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"state_dict": model.state_dict()}
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, p.as_posix())
    return p


def load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    p = Path(path)
    payload = torch.load(p.as_posix(), map_location=map_location)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state, strict=bool(strict))
    return payload if isinstance(payload, dict) else {"state_dict": state}


# -----------------------------
# Geometry helpers + feature conversions
# -----------------------------


def wrap_dphi(dphi: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(dphi), np.cos(dphi))


def compute_jet_axis(const: np.ndarray, mask: np.ndarray, eps: float = 1e-8):
    """Compute jet axis from (pt, eta, phi, E) using sum of 4-vectors."""
    pt = np.maximum(const[:, :, 0], eps)
    eta = np.clip(const[:, :, 1], -5, 5)
    phi = const[:, :, 2]
    E = np.maximum(const[:, :, 3], eps)

    m = mask.astype(np.float32)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    jet_px = (px * m).sum(axis=1)
    jet_py = (py * m).sum(axis=1)
    jet_pz = (pz * m).sum(axis=1)
    jet_E = (E * m).sum(axis=1)

    jet_p = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2) + eps
    jet_eta = 0.5 * np.log(np.clip((jet_p + jet_pz) / (jet_p - jet_pz + eps), eps, 1e8))
    jet_phi = np.arctan2(jet_py, jet_px)
    jet_pt = np.sqrt(jet_px**2 + jet_py**2) + eps

    return (
        jet_eta.astype(np.float32),
        jet_phi.astype(np.float32),
        jet_pt.astype(np.float32),
        jet_E.astype(np.float32),
    )


def raw_to_feats_with_axis(const: np.ndarray, mask: np.ndarray, jet_eta: np.ndarray, jet_phi: np.ndarray) -> np.ndarray:
    """Convert raw (pt,eta,phi,E) -> (log_pt, dEta, dPhi, log_E) using provided axis."""
    eps = 1e-8
    pt = np.maximum(const[:, :, 0], eps)
    eta = np.clip(const[:, :, 1], -5, 5)
    phi = const[:, :, 2]
    E = np.maximum(const[:, :, 3], eps)

    dEta = np.clip(eta - jet_eta[:, None], -5.0, 5.0)
    dPhi = wrap_dphi(phi - jet_phi[:, None])
    log_pt = np.log(pt)
    log_E = np.log(E)
    feats = np.stack([log_pt, dEta, dPhi, log_E], axis=-1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    feats[~mask] = 0.0
    return feats.astype(np.float32)


def raw_to_feats7_with_axis(
    const: np.ndarray,
    mask: np.ndarray,
    jet_eta: np.ndarray,
    jet_phi: np.ndarray,
    jet_pt: np.ndarray,
    jet_E: np.ndarray,
) -> np.ndarray:
    """Compute 7 engineered features in jet-axis frame (mirrors Baseline.compute_features)."""
    eps = 1e-8
    pt = np.maximum(const[:, :, 0], eps)
    eta = np.clip(const[:, :, 1], -5, 5)
    phi = const[:, :, 2]
    E = np.maximum(const[:, :, 3], eps)

    dEta = np.clip(eta - jet_eta[:, None], -5.0, 5.0)
    dPhi = wrap_dphi(phi - jet_phi[:, None])
    log_pt = np.log(pt + eps)
    log_E = np.log(E + eps)
    log_pt_over_jetpt = np.log(pt / np.maximum(jet_pt[:, None], eps) + eps)
    log_E_over_jetE = np.log(E / np.maximum(jet_E[:, None], eps) + eps)
    dR = np.sqrt(dEta**2 + dPhi**2)
    feats = np.stack(
        [dEta, dPhi, log_pt, log_E, log_pt_over_jetpt, log_E_over_jetE, dR],
        axis=-1,
    )
    feats = np.clip(np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)
    feats[~mask] = 0.0
    return feats.astype(np.float32)


def raw_to_feats(
    const: np.ndarray,
    mask: np.ndarray,
    hlt_axis: np.ndarray,
    *,
    kind: str = "4d",
) -> np.ndarray:
    """Feature front-end switch: kind in {"4d","7d"}."""
    jet_eta = hlt_axis[:, 0]
    jet_phi = hlt_axis[:, 1]
    jet_pt = hlt_axis[:, 2]
    jet_E = hlt_axis[:, 3]
    k = str(kind).lower()
    if k in ("4", "4d", "raw4", "logpt_detadphi_loge"):
        return raw_to_feats_with_axis(const, mask, jet_eta, jet_phi)
    if k in ("7", "7d", "eng7", "engineered7"):
        return raw_to_feats7_with_axis(const, mask, jet_eta, jet_phi, jet_pt, jet_E)
    raise ValueError(f"Unknown feature kind: {kind}")


def feats_to_raw(feats: np.ndarray, jet_eta: np.ndarray, jet_phi: np.ndarray) -> np.ndarray:
    """Map features back to raw (pt,eta,phi,E). Supports 4D and the Baseline-style 7D engineered features."""
    eps = 1e-8
    D = int(feats.shape[-1])
    if D == 4:
        # (log_pt, dEta, dPhi, log_E)
        log_pt = feats[..., 0]
        dEta = feats[..., 1]
        dPhi = feats[..., 2]
        log_E = feats[..., 3]
    elif D >= 7:
        # Baseline 7D order: (dEta, dPhi, log_pt, log_E, log(pt/jet_pt), log(E/jet_E), dR)
        dEta = feats[..., 0]
        dPhi = feats[..., 1]
        log_pt = feats[..., 2]
        log_E = feats[..., 3]
    else:
        raise ValueError(f"Unsupported feature dim for feats_to_raw: {D}")

    pt = np.maximum(np.exp(log_pt), eps)
    eta = np.clip(jet_eta[..., None] + dEta, -5, 5)
    phi = np.arctan2(np.sin(jet_phi[..., None] + dPhi), np.cos(jet_phi[..., None] + dPhi))
    E = np.maximum(np.exp(log_E), eps)
    raw = np.stack([pt, eta, phi, E], axis=-1)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return raw.astype(np.float32)


def get_stats_tokens(feat: np.ndarray, mask: np.ndarray, idx_sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    D = feat.shape[-1]
    means = np.zeros(D, dtype=np.float64)
    stds = np.zeros(D, dtype=np.float64)
    for d in range(D):
        vals = feat[idx_sel, :, d][mask[idx_sel]]
        means[d] = float(np.nanmean(vals))
        stds[d] = float(np.nanstd(vals) + 1e-8)
    return means.astype(np.float32), stds.astype(np.float32)


def standardize_tokens(feat: np.ndarray, mask: np.ndarray, means: np.ndarray, stds: np.ndarray, clip: float = 10.0) -> np.ndarray:
    std = (feat - means[None, None, :]) / stds[None, None, :]
    std = np.clip(std, -float(clip), float(clip))
    std = np.nan_to_num(std, 0.0)
    std[~mask] = 0.0
    return std.astype(np.float32)


# -----------------------------
# HLT simulation with compact merge samples
# -----------------------------


def apply_hlt_effects_collect_samples(
    const: np.ndarray,
    mask: np.ndarray,
    cfg: dict,
    *,
    seed: int = 42,
    k_max: int = 8,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Tuple[int, int]],
    List[np.ndarray],
]:
    """
    Apply HLT effects (threshold+merge+smear+eff).

    Returns:
      hlt_raw: [N,S,4]
      hlt_mask: [N,S]
      off_mask: [N,S] (offline threshold on raw)
      group_size: [N,S] int16 (0 for masked tokens)
      hlt_axis: [N,4] float32 (jet_eta, jet_phi, jet_pt, jet_E)
      samples: list of (jet_idx, parent_idx) for merged parents (group_size>1)
      children: list of child-index arrays (pt-sorted, truncated to k_max) matching `samples`
    """
    np.random.seed(int(seed))
    hcfg = cfg["hlt_effects"]
    n_jets, S, _ = const.shape
    hlt = const.copy()
    hlt_mask = mask.copy()

    # Offline threshold mask (reference)
    pt_thr_off = float(hcfg["pt_threshold_offline"])
    off_mask = hlt_mask & (hlt[:, :, 0] >= pt_thr_off)

    # HLT threshold
    pt_thr_hlt = float(hcfg["pt_threshold_hlt"])
    below = (hlt[:, :, 0] < pt_thr_hlt) & hlt_mask
    hlt_mask[below] = False
    hlt[~hlt_mask] = 0

    group_size = np.zeros((n_jets, S), dtype=np.int16)
    samples: List[Tuple[int, int]] = []  #记录第几个jet的第几个contituent是parent（after efficiency loss）
    children: List[np.ndarray] = []  #对应samples的child contituents的索引

    # Merge (record only merged parents)
    # group_size记录jet中每个contituent的merge数量
    # （0表示被merge，1表示未被merge但也没merge其它粒子，n表示merge了n-1个粒子加它自己就是n个粒子），
    #  g记录哪些contituents(索引)被merge到一起
    if bool(hcfg["merge_enabled"]) and float(hcfg["merge_radius"]) > 0:
        r = float(hcfg["merge_radius"])
        for jet_idx in range(n_jets):
            valid_idx = np.where(hlt_mask[jet_idx])[0]  #返回每个jet中有效的contituents的索引
            if len(valid_idx) == 0:
                continue
            g: dict[int, List[int]] = {int(ii): [int(ii)] for ii in valid_idx}
            # 初始化group_size，每个有效的contituent的group_size为1
            for ii in valid_idx:
                group_size[jet_idx, int(ii)] = 1

            if len(valid_idx) >= 2:
                to_remove = set()  #记录需要被移除的contituents的索引
                for a in range(len(valid_idx)):
                    i = int(valid_idx[a])
                    if i in to_remove:
                        #如果i已经在to_remove中，则跳过
                        continue
                    for b in range(a + 1, len(valid_idx)):
                        j = int(valid_idx[b])
                        if j in to_remove:
                            #如果j已经在to_remove中，则跳过
                            continue
                        #计算i和j的dEta和dPhi
                        deta = float(hlt[jet_idx, i, 1] - hlt[jet_idx, j, 1])
                        dphi = float(wrap_dphi(hlt[jet_idx, i, 2] - hlt[jet_idx, j, 2]))
                        dR = math.sqrt(deta * deta + dphi * dphi)
                        if dR < r:
                            pt_i = float(hlt[jet_idx, i, 0])
                            pt_j = float(hlt[jet_idx, j, 0])
                            pt_sum = pt_i + pt_j
                            if pt_sum < 1e-6:
                                continue
                            g[i].extend(g[j])  #把g[j]的contituents的索引添加到g[i]中
                            g.pop(j, None)  #删除key为j的这一列
                            group_size[jet_idx, i] = int(group_size[jet_idx, i] + group_size[jet_idx, j])  #更新group_size
                            group_size[jet_idx, j] = 0  #group_size[jet_idx, j]设置为0表示被merge

                            # pT-weighted merge in HLT raw space
                            w_i, w_j = pt_i / pt_sum, pt_j / pt_sum
                            hlt[jet_idx, i, 0] = pt_sum
                            hlt[jet_idx, i, 1] = w_i * hlt[jet_idx, i, 1] + w_j * hlt[jet_idx, j, 1]
                            phi_i, phi_j = float(hlt[jet_idx, i, 2]), float(hlt[jet_idx, j, 2])
                            hlt[jet_idx, i, 2] = math.atan2(
                                w_i * math.sin(phi_i) + w_j * math.sin(phi_j),
                                w_i * math.cos(phi_i) + w_j * math.cos(phi_j),
                            )
                            hlt[jet_idx, i, 3] = float(hlt[jet_idx, i, 3]) + float(hlt[jet_idx, j, 3])
                            to_remove.add(j)  #记录需要被移除的contituents的索引

                for idx in to_remove:
                    #将需要被移除的contituents的mask设置为False，即被merge
                    hlt_mask[jet_idx, idx] = False
                    #将需要被移除的contituents的raw设置为0，即被merge
                    hlt[jet_idx, idx] = 0

            # Record merged parents for this jet
            for p, ch in g.items():
                if not bool(hlt_mask[jet_idx, p]):
                    continue
                if len(ch) <= 1:
                    continue
                ch_arr = np.asarray(ch, dtype=np.int64)
                # Sort children by OFFLINE pt (raw const)
                pts = const[jet_idx, ch_arr, 0]
                order = np.argsort(-pts)
                ch_arr = ch_arr[order]
                if int(ch_arr.shape[0]) > int(k_max):
                    ch_arr = ch_arr[: int(k_max)]  #只保留前k_max个contituents
                samples.append((int(jet_idx), int(p)))  #记录第几个jet的第几个contituent是parent
                children.append(ch_arr)
    else:
        # No merge: group_size is 1 for valid tokens, but there are no merged-parent samples.
        for jet_idx in range(n_jets):
            valid_idx = np.where(hlt_mask[jet_idx])[0]
            group_size[jet_idx, valid_idx] = 1

    # Smearing
    valid = hlt_mask.copy()
    pt_noise = np.clip(
        np.random.normal(1.0, float(hcfg.get("pt_resolution", 0.0)), (n_jets, S)),
        0.5,
        1.5,
    )
    hlt[:, :, 0] = np.where(valid, hlt[:, :, 0] * pt_noise, 0)
    eta_noise = np.random.normal(0, float(hcfg.get("eta_resolution", 0.0)), (n_jets, S))
    hlt[:, :, 1] = np.where(valid, np.clip(hlt[:, :, 1] + eta_noise, -5, 5), 0)
    phi_noise = np.random.normal(0, float(hcfg.get("phi_resolution", 0.0)), (n_jets, S))
    new_phi = hlt[:, :, 2] + phi_noise
    hlt[:, :, 2] = np.where(valid, np.arctan2(np.sin(new_phi), np.cos(new_phi)), 0)
    # Recompute E from (pt, eta)
    hlt[:, :, 3] = np.where(valid, hlt[:, :, 0] * np.cosh(np.clip(hlt[:, :, 1], -5, 5)), 0)

    # Efficiency loss
    eff = float(hcfg.get("efficiency_loss", 0.0))
    if eff > 0:
        lost = (np.random.random((n_jets, S)) < eff) & hlt_mask
        hlt_mask[lost] = False
        hlt[lost] = 0
        group_size[lost] = 0

    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    jet_eta, jet_phi, jet_pt, jet_E = compute_jet_axis(hlt, hlt_mask, eps=1e-8)
    hlt_axis = np.stack([jet_eta, jet_phi, jet_pt, jet_E], axis=1).astype(np.float32)

    # Filter any samples whose parent token was later removed (e.g., by efficiency loss).
    if len(samples) > 0:
        s2: List[Tuple[int, int]] = []
        c2: List[np.ndarray] = []
        for (j, p), ch in zip(samples, children):
            if bool(hlt_mask[int(j), int(p)]):
                s2.append((int(j), int(p)))  #如果parent token没有被efficiency loss移除，则记录第几个jet的第几个contituent是parent
                c2.append(ch)  #同上，记录对应的child contituents的索引
        samples, children = s2, c2

    return hlt, hlt_mask, off_mask, group_size, hlt_axis, samples, children


# -----------------------------
# Targets + datasets
# -----------------------------


def build_parent_targets_from_group_size(hlt_mask: np.ndarray, group_size: np.ndarray) -> np.ndarray:
    """Build parent_gt of shape [N,S]."""
    gs = group_size.astype(np.int32, copy=False)
    parent_gt = (hlt_mask & (gs > 1)).astype(np.float32)  #parent_gt记录哪些contituents是parent(0，1)（after efficiency loss）
    return parent_gt


class JetParentDataset(Dataset):
    """One sample = one jet; supervise parentness on all tokens."""

    def __init__(
        self,
        jet_indices: Sequence[int],
        *,
        feat_hlt_std: np.ndarray,
        mask_hlt: np.ndarray,
        parent_gt: np.ndarray,
    ):
        self.jet_idx = np.asarray(jet_indices, dtype=np.int64)
        self.hlt = torch.tensor(feat_hlt_std, dtype=torch.float32)
        self.mask = torch.tensor(mask_hlt, dtype=torch.bool)
        self.parent_gt = torch.tensor(parent_gt, dtype=torch.float32)

    def __len__(self):
        return int(self.jet_idx.shape[0])

    def __getitem__(self, i):
        j = int(self.jet_idx[i])
        return {
            "hlt": self.hlt[j],
            "mask_hlt": self.mask[j],
            "parent_gt": self.parent_gt[j],
        }


class ParentRecoDataset(Dataset):
    """One sample = one merged parent token; supervise ordered child reconstruction + slot objectness."""

    def __init__(
        self,
        indices: Sequence[int],
        *,
        samples: Sequence[Tuple[int, int]],
        children: Sequence[np.ndarray],
        feat_hlt_std: np.ndarray,
        mask_hlt: np.ndarray,
        off_child_feat_std: np.ndarray,
        k_max: int,
        # Optional raw HLT tokens + axis for physical auxiliary losses (px/py)
        hlt_raw: Optional[np.ndarray] = None,
        hlt_axis: Optional[np.ndarray] = None,
        # Optionally add full token-level supervision for auxiliary losses in Stage2
        parent_gt: Optional[np.ndarray] = None,
    ):
        self.sel = np.asarray(indices, dtype=np.int64)
        self.samples = samples
        self.children = children
        self.hlt = torch.tensor(feat_hlt_std, dtype=torch.float32)
        self.mask = torch.tensor(mask_hlt, dtype=torch.bool)
        self.off_child_feat_std = torch.tensor(off_child_feat_std, dtype=torch.float32)
        self.k_max = int(k_max)
        self.parent_gt = None if parent_gt is None else torch.tensor(parent_gt, dtype=torch.float32)
        self.hlt_raw = None if hlt_raw is None else torch.tensor(hlt_raw, dtype=torch.float32)
        self.hlt_axis = None if hlt_axis is None else torch.tensor(hlt_axis, dtype=torch.float32)

    def __len__(self):
        return int(self.sel.shape[0])

    def __getitem__(self, i):
        sidx = int(self.sel[i])
        jet_idx, parent_idx = self.samples[sidx]
        ch = self.children[sidx]
        m = int(ch.shape[0])
        D = int(self.off_child_feat_std.shape[-1])
        tgt = torch.zeros((self.k_max, D), dtype=torch.float32)
        tgt_mask = torch.zeros((self.k_max,), dtype=torch.bool)
        tgt[:m] = self.off_child_feat_std[jet_idx, ch]
        tgt_mask[:m] = True

        out = {
            "hlt": self.hlt[jet_idx],
            "mask_hlt": self.mask[jet_idx],
            "parent_idx": torch.tensor(int(parent_idx), dtype=torch.long),
            "tgt": tgt,
            "tgt_mask": tgt_mask,  # objectness 的 GT（每个 slot 是否真实存在）
            "m_true": torch.tensor(float(m), dtype=torch.float32),
        }
        # 提供 px/py 守恒辅助 loss 所需量：parent(pt,phi) 和 jet_phi
        if self.hlt_raw is not None and self.hlt_axis is not None:
            out["parent_pt"] = self.hlt_raw[jet_idx, int(parent_idx), 0]
            out["parent_phi"] = self.hlt_raw[jet_idx, int(parent_idx), 2]
            out["jet_phi"] = self.hlt_axis[jet_idx, 1]
        if self.parent_gt is not None:
            out["parent_gt"] = self.parent_gt[jet_idx]
        return out


# -----------------------------
# Losses + training loops
# -----------------------------


def _masked_bce_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, *, pos_weight: float = 1.0
) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        return logits.new_tensor(0.0)
    logits_v = logits[mask]
    targets_v = targets[mask]
    pw = logits.new_tensor(float(pos_weight))
    return nn.BCEWithLogitsLoss(pos_weight=pw)(logits_v, targets_v)


def _slot_bce_with_logits(
    logits: torch.Tensor,
    targets_bool: torch.Tensor,
    *,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """
    Slot-level BCE.
    Args:
      logits: [B,K]
      targets_bool: [B,K] bool (True=真实 child)
    """
    y = targets_bool.to(dtype=logits.dtype)
    pw = logits.new_tensor(float(pos_weight))
    return nn.BCEWithLogitsLoss(pos_weight=pw)(logits, y)


def prefix_lengths_from_prob(prob: torch.Tensor, tau: float) -> torch.Tensor:
    """根据 prefix+threshold 规则，从 objectness 概率得到每个样本的前缀长度。"""
    if prob.ndim != 2:
        raise ValueError(f"prob must be [B,K], got shape={tuple(prob.shape)}")
    m = prob > float(tau)  # [B,K]
    B, K = m.shape
    first_false = (~m).to(dtype=torch.float32).argmax(dim=1)  # [B]
    all_true = m.all(dim=1)
    L = first_false.to(dtype=torch.long)
    L[all_true] = int(K)
    return L


@dataclass
class TrainCfgParent:
    # DataLoader 在 notebook 里创建，但这里保留 batch_size 字段方便 **CONFIG 解包
    batch_size: int = 256
    epochs: int = 30
    lr: float = 5e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 2
    patience: int = 8
    grad_clip: float = 1.0
    thr_parent: float = 0.7


@dataclass
class TrainCfgRecoObj:
    # DataLoader 在 notebook 里创建，但这里保留 batch_size 字段方便 **CONFIG 解包
    batch_size: int = 256
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 1
    patience: int = 6
    grad_clip: float = 1.0
    huber_delta: float = 1.0
    w_reco: float = 1.0
    w_obj: float = 0.5
    obj_pos_weight: float = 2.0
    thr_obj: float = 0.5
    # Stage2 防止表征漂移
    freeze_encoder: bool = True
    enc_lr_mult: float = 0.1
    # 可选：在 Stage2 加一个小权重 parentness aux，避免 parentness drift
    w_parent_aux: float = 0.0
    # 可选：px/py 守恒（方案B），守恒到 HLT parent raw token 的 (pt,phi)
    w_pxy: float = 0.0
    pxy_eps: float = 1e-6


def _pxy_loss_B(
    pred_child_std: torch.Tensor,
    tgt_mask: torch.Tensor,
    *,
    feat_means: torch.Tensor,
    feat_stds: torch.Tensor,
    jet_phi: torch.Tensor,
    parent_pt: torch.Tensor,
    parent_phi: torch.Tensor,
    pxy_eps: float,
) -> torch.Tensor:
    """
    px/py 守恒（方案B，相对向量误差），只对 GT slot(tgt_mask) 求和。
    Args:
      pred_child_std: [B,K,D] 标准化后的 child 预测
      tgt_mask: [B,K] bool
      feat_means/stds: [D]
      jet_phi: [B] HLT jet axis phi
      parent_pt: [B] HLT raw parent token pt
      parent_phi: [B] HLT raw parent token phi
    """
    eps = float(pxy_eps)
    B, K, D = pred_child_std.shape
    means = feat_means.view(1, 1, -1)
    stds = feat_stds.view(1, 1, -1)
    feats = pred_child_std * stds + means  # [B,K,D] unstandardized feats

    # indices depend on feature dim
    if int(D) == 4:
        log_pt = feats[..., 0]
        dphi = feats[..., 2]
    elif int(D) >= 7:
        dphi = feats[..., 1]
        log_pt = feats[..., 2]
    else:
        raise ValueError(f"Unsupported child dim for pxy loss: {D}")

    pt = torch.exp(log_pt).clamp_min(0.0)
    phi = torch.atan2(torch.sin(jet_phi.view(B, 1) + dphi), torch.cos(jet_phi.view(B, 1) + dphi))

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)

    m = tgt_mask.to(dtype=px.dtype)
    sum_px = (px * m).sum(dim=1)
    sum_py = (py * m).sum(dim=1)

    ppx = parent_pt * torch.cos(parent_phi)
    ppy = parent_pt * torch.sin(parent_phi)

    dx = sum_px - ppx
    dy = sum_py - ppy
    num = torch.sqrt(dx * dx + dy * dy + eps)
    den = torch.sqrt(ppx * ppx + ppy * ppy + eps)
    rel = num / den
    return rel.mean()


def _cosine_schedule(ep: int, warmup: int, total: int) -> float:
    if ep < warmup:
        return float(ep + 1) / float(max(1, warmup))
    t = float(ep - warmup) / float(max(1, total - warmup))
    return float(0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t)))))


@torch.no_grad()
def eval_parentness(model, loader, device: torch.device, *, pos_weight: float, cfg: TrainCfgParent) -> dict[str, float]:
    model.eval()
    tp = fp = fn = 0
    tot_parent = 0.0
    nb = 0
    all_prob: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    for batch in loader:
        x = batch["hlt"].to(device)
        m = batch["mask_hlt"].to(device)
        y_parent = batch["parent_gt"].to(device)
        out = model(x, m, parent_idx=None)
        parent_loss = _masked_bce_with_logits(out.parent_logit, y_parent, m, pos_weight=float(pos_weight))
        tot_parent += float(parent_loss.item())

        prob = torch.sigmoid(out.parent_logit)
        pred = (prob > float(cfg.thr_parent)) & m
        gt = (y_parent > 0.5) & m
        tp += int((pred & gt).sum().item())
        fp += int((pred & (~gt)).sum().item())
        fn += int(((~pred) & gt).sum().item())
        nb += 1

        all_prob.append(prob[m].detach().cpu().numpy())
        all_gt.append(y_parent[m].detach().cpu().numpy())

    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    # AUC（如果没有正/负样本会报错，这里做保护）
    try:
        pcat = np.concatenate(all_prob) if all_prob else np.zeros((0,), dtype=np.float32)
        ycat = np.concatenate(all_gt) if all_gt else np.zeros((0,), dtype=np.float32)
        auc = float(roc_auc_score(ycat, pcat)) if (pcat.size > 0 and (ycat.max() > ycat.min())) else float("nan")
    except Exception:
        auc = float("nan")
    return {
        "parent_loss": tot_parent / max(1, nb),
        "precision": float(prec),
        "recall": float(rec),
        "auc": float(auc),
    }


def train_parentness(
    model,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: TrainCfgParent,
    *,
    pos_weight: float,
    ckpt_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    best = float("inf")
    best_state = None
    no_imp = 0
    hist: dict[str, list[float]] = {"train_parent": [], "val_parent": [], "val_prec": [], "val_rec": [], "val_auc": []}
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        lr_scale = _cosine_schedule(ep - 1, int(cfg.warmup_epochs), int(cfg.epochs))
        for g in opt.param_groups:
            g["lr"] = float(cfg.lr) * float(lr_scale)

        running = 0.0
        n = 0
        for batch in train_loader:
            x = batch["hlt"].to(device)
            m = batch["mask_hlt"].to(device)
            y_parent = batch["parent_gt"].to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x, m, parent_idx=None)
            loss = _masked_bce_with_logits(out.parent_logit, y_parent, m, pos_weight=float(pos_weight))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()
            running += float(loss.item())
            n += 1

        val = eval_parentness(model, val_loader, device, pos_weight=float(pos_weight), cfg=cfg)
        if float(val["parent_loss"]) < best - 1e-6:
            best = float(val["parent_loss"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
            if ckpt_path is not None:
                save_checkpoint(model, ckpt_path, extra={"epoch": ep, "best": best, "val": val})
        else:
            no_imp += 1

        hist["train_parent"].append(float(running / max(1, n)))
        hist["val_parent"].append(float(val["parent_loss"]))
        hist["val_prec"].append(float(val["precision"]))
        hist["val_rec"].append(float(val["recall"]))
        hist["val_auc"].append(float(val["auc"]))
        print(
            f"[Parent] Ep {ep:03d}: train_parent={hist['train_parent'][-1]:.4f} "
            f"val_parent={val['parent_loss']:.4f} prec={val['precision']:.3f} rec={val['recall']:.3f} auc={val['auc']:.4f} "
            f"no_imp={no_imp}"
        )
        if no_imp >= int(cfg.patience):
            print("[Parent] Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best": best, "history": hist}


@torch.no_grad()
def eval_reco_obj(
    model,
    loader,
    device: torch.device,
    *,
    cfg: TrainCfgRecoObj,
    pos_weight_parent: float = 1.0,
    feat_means: Optional[np.ndarray] = None,
    feat_stds: Optional[np.ndarray] = None,
) -> dict[str, float]:
    model.eval()
    tot = 0.0
    tot_reco = 0.0
    tot_obj = 0.0
    tot_parent_aux = 0.0
    tot_pxy = 0.0
    nb = 0
    mae_L = 0.0
    slot_tp = slot_fp = slot_fn = 0
    feat_means_t = None
    feat_stds_t = None
    if feat_means is not None and feat_stds is not None:
        feat_means_t = torch.tensor(np.asarray(feat_means, dtype=np.float32), device=device)
        feat_stds_t = torch.tensor(np.asarray(feat_stds, dtype=np.float32), device=device)
    for batch in loader:
        x = batch["hlt"].to(device)
        m = batch["mask_hlt"].to(device)
        parent_idx = batch["parent_idx"].to(device)
        tgt = batch["tgt"].to(device)
        tgt_mask = batch["tgt_mask"].to(device)
        out = model(x, m, parent_idx=parent_idx)
        pred = out.child_feat
        obj_logit = out.obj_logit
        assert pred is not None and obj_logit is not None

        # reco loss：只在真 slot 上回归
        w_mask = tgt_mask.unsqueeze(-1).to(pred.dtype)
        loss_reco = F.smooth_l1_loss(pred * w_mask, tgt * w_mask, reduction="sum", beta=float(cfg.huber_delta))
        denom = w_mask.sum().clamp_min(1.0)
        loss_reco = loss_reco / denom

        # obj loss：所有 slot 二分类
        loss_obj = _slot_bce_with_logits(obj_logit, tgt_mask, pos_weight=float(cfg.obj_pos_weight))

        loss = float(cfg.w_reco) * loss_reco + float(cfg.w_obj) * loss_obj
        tot_reco += float(loss_reco.item())
        tot_obj += float(loss_obj.item())

        # optional px/py conservation (scheme B) to HLT parent raw token
        if (
            float(getattr(cfg, "w_pxy", 0.0)) > 0.0
            and feat_means_t is not None
            and feat_stds_t is not None
            and ("parent_pt" in batch)
            and ("parent_phi" in batch)
            and ("jet_phi" in batch)
        ):
            parent_pt = batch["parent_pt"].to(device)
            parent_phi = batch["parent_phi"].to(device)
            jet_phi = batch["jet_phi"].to(device)
            loss_pxy = _pxy_loss_B(
                pred,
                tgt_mask,
                feat_means=feat_means_t,
                feat_stds=feat_stds_t,
                jet_phi=jet_phi,
                parent_pt=parent_pt,
                parent_phi=parent_phi,
                pxy_eps=float(getattr(cfg, "pxy_eps", 1e-6)),
            )
            loss = loss + float(cfg.w_pxy) * loss_pxy
            tot_pxy += float(loss_pxy.item())

        # optional aux parentness loss（防 drift）
        if float(getattr(cfg, "w_parent_aux", 0.0)) > 0.0 and ("parent_gt" in batch):
            y_parent = batch["parent_gt"].to(device)
            loss_parent_aux = _masked_bce_with_logits(out.parent_logit, y_parent, m, pos_weight=float(pos_weight_parent))
            loss = loss + float(cfg.w_parent_aux) * loss_parent_aux
            tot_parent_aux += float(loss_parent_aux.item())

        tot += float(loss.item())
        nb += 1

        # prefix-length 诊断：L_pred vs m_true
        prob = torch.sigmoid(obj_logit)
        L_pred = prefix_lengths_from_prob(prob, float(cfg.thr_obj)).to(dtype=torch.float32)
        m_true = batch["m_true"].to(device)
        mae_L += float((L_pred - m_true).abs().mean().item())

        # slot-level PR（用 prefix 产生 pred slots）
        K = int(tgt_mask.shape[1])
        kk = torch.arange(K, device=device).view(1, -1).expand(int(tgt_mask.shape[0]), -1)
        pred_slot = kk < L_pred.to(dtype=torch.long).view(-1, 1)
        gt_slot = tgt_mask
        slot_tp += int((pred_slot & gt_slot).sum().item())
        slot_fp += int((pred_slot & (~gt_slot)).sum().item())
        slot_fn += int(((~pred_slot) & gt_slot).sum().item())

    prec = slot_tp / max(1, (slot_tp + slot_fp))
    rec = slot_tp / max(1, (slot_tp + slot_fn))
    return {
        "loss": tot / max(1, nb),
        "loss_reco": tot_reco / max(1, nb),
        "loss_obj": tot_obj / max(1, nb),
        "loss_parent_aux": tot_parent_aux / max(1, nb),
        "loss_pxy": tot_pxy / max(1, nb),
        "mae_L_vs_m": mae_L / max(1, nb) if nb > 0 else float("nan"),
        "slot_prec": float(prec),
        "slot_rec": float(rec),
    }


def train_reco_teacher_forced_obj(
    model,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: TrainCfgRecoObj,
    *,
    feat_means: np.ndarray,
    feat_stds: np.ndarray,
    pos_weight_parent: float = 1.0,
    ckpt_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    # 两个选择：冻结 encoder 或 encoder 低 LR、decoder 高 LR
    orig_flags = [(p, bool(p.requires_grad)) for p in model.parameters()]

    def _set_requires_grad(params: list[torch.nn.Parameter], flag: bool) -> None:
        for p in params:
            p.requires_grad = bool(flag)

    # Split params into encoder-side and decoder-side
    enc_params: list[torch.nn.Parameter] = []
    dec_params: list[torch.nn.Parameter] = []
    for name in ["input_proj", "encoder", "parent_head"]:
        if hasattr(model, name):
            mod = getattr(model, name)
            if isinstance(mod, nn.Module):
                enc_params += [p for p in mod.parameters()]
    # Decoder-side (includes query_embed parameter)
    if hasattr(model, "query_embed") and isinstance(getattr(model, "query_embed"), torch.nn.Parameter):
        dec_params.append(getattr(model, "query_embed"))
    for name in ["decoder", "cond_proj", "child_head", "obj_head"]:
        if hasattr(model, name):
            mod = getattr(model, name)
            if isinstance(mod, nn.Module):
                dec_params += [p for p in mod.parameters()]

    # De-duplicate while preserving order
    seen: set[int] = set()
    enc_params_u: list[torch.nn.Parameter] = []
    for p in enc_params:
        if id(p) not in seen:
            enc_params_u.append(p)
            seen.add(id(p))
    dec_params_u: list[torch.nn.Parameter] = []
    for p in dec_params:
        if id(p) not in seen:
            dec_params_u.append(p)
            seen.add(id(p))

    frozen_mods: list[nn.Module] = []
    if bool(getattr(cfg, "freeze_encoder", True)):
        _set_requires_grad(enc_params_u, False)
        for name in ["input_proj", "encoder", "parent_head"]:
            if hasattr(model, name) and isinstance(getattr(model, name), nn.Module):
                frozen_mods.append(getattr(model, name))
        _set_requires_grad(dec_params_u, True)
        param_groups = [
            {"params": [p for p in dec_params_u if p.requires_grad], "base_lr": float(cfg.lr)},
        ]
    else:
        _set_requires_grad(enc_params_u, True)
        _set_requires_grad(dec_params_u, True)
        enc_lr = float(cfg.lr) * float(getattr(cfg, "enc_lr_mult", 0.1))
        param_groups = [
            {"params": [p for p in enc_params_u if p.requires_grad], "base_lr": float(enc_lr)},
            {"params": [p for p in dec_params_u if p.requires_grad], "base_lr": float(cfg.lr)},
        ]

    if sum(len(g["params"]) for g in param_groups) == 0:
        raise RuntimeError("No trainable parameters found for reco training (check freeze/lr settings).")
    opt = torch.optim.AdamW(param_groups, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    feat_means_t = torch.tensor(np.asarray(feat_means, dtype=np.float32), device=device)
    feat_stds_t = torch.tensor(np.asarray(feat_stds, dtype=np.float32), device=device)

    best = float("inf")
    best_state = None
    no_imp = 0
    hist: dict[str, list[float]] = {
        "train_total": [],
        "train_reco": [],
        "train_obj": [],
        "train_parent_aux": [],
        "val_total": [],
        "val_reco": [],
        "val_obj": [],
        "val_parent_aux": [],
        "val_mae_L": [],
        "val_slot_prec": [],
        "val_slot_rec": [],
    }

    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        for fm in frozen_mods:
            fm.eval()
        lr_scale = _cosine_schedule(ep - 1, int(cfg.warmup_epochs), int(cfg.epochs))
        for g in opt.param_groups:
            g["lr"] = float(g.get("base_lr", float(cfg.lr))) * float(lr_scale)

        running_total = 0.0
        running_reco = 0.0
        running_obj = 0.0
        running_parent_aux = 0.0
        running_pxy = 0.0
        n = 0
        for batch in train_loader:
            x = batch["hlt"].to(device)
            m = batch["mask_hlt"].to(device)
            parent_idx = batch["parent_idx"].to(device)
            tgt = batch["tgt"].to(device)
            tgt_mask = batch["tgt_mask"].to(device)
            opt.zero_grad(set_to_none=True)

            out = model(x, m, parent_idx=parent_idx)
            pred = out.child_feat
            obj_logit = out.obj_logit
            assert pred is not None and obj_logit is not None

            w_mask = tgt_mask.unsqueeze(-1).to(pred.dtype)
            loss_reco = F.smooth_l1_loss(pred * w_mask, tgt * w_mask, reduction="sum", beta=float(cfg.huber_delta))
            denom = w_mask.sum().clamp_min(1.0)
            loss_reco = loss_reco / denom

            loss_obj = _slot_bce_with_logits(obj_logit, tgt_mask, pos_weight=float(cfg.obj_pos_weight))

            loss = float(cfg.w_reco) * loss_reco + float(cfg.w_obj) * loss_obj

            # optional px/py conservation (scheme B) to HLT parent raw token
            if float(getattr(cfg, "w_pxy", 0.0)) > 0.0 and ("parent_pt" in batch) and ("parent_phi" in batch) and ("jet_phi" in batch):
                parent_pt = batch["parent_pt"].to(device)
                parent_phi = batch["parent_phi"].to(device)
                jet_phi = batch["jet_phi"].to(device)
                loss_pxy = _pxy_loss_B(
                    pred,
                    tgt_mask,
                    feat_means=feat_means_t,
                    feat_stds=feat_stds_t,
                    jet_phi=jet_phi,
                    parent_pt=parent_pt,
                    parent_phi=parent_phi,
                    pxy_eps=float(getattr(cfg, "pxy_eps", 1e-6)),
                )
                loss = loss + float(cfg.w_pxy) * loss_pxy
                running_pxy += float(loss_pxy.item())

            # optional aux parentness
            if float(getattr(cfg, "w_parent_aux", 0.0)) > 0.0 and ("parent_gt" in batch):
                y_parent = batch["parent_gt"].to(device)
                loss_parent_aux = _masked_bce_with_logits(out.parent_logit, y_parent, m, pos_weight=float(pos_weight_parent))
                loss = loss + float(cfg.w_parent_aux) * loss_parent_aux
                running_parent_aux += float(loss_parent_aux.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()

            running_total += float(loss.item())
            running_reco += float(loss_reco.item())
            running_obj += float(loss_obj.item())
            n += 1

        val = eval_reco_obj(
            model,
            val_loader,
            device,
            cfg=cfg,
            pos_weight_parent=float(pos_weight_parent),
            feat_means=feat_means,
            feat_stds=feat_stds,
        )
        if float(val["loss"]) < best - 1e-6:
            best = float(val["loss"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
            if ckpt_path is not None:
                save_checkpoint(model, ckpt_path, extra={"epoch": ep, "best": best, "val": val})
        else:
            no_imp += 1

        hist["train_total"].append(float(running_total / max(1, n)))
        hist["train_reco"].append(float(running_reco / max(1, n)))
        hist["train_obj"].append(float(running_obj / max(1, n)))
        hist["train_parent_aux"].append(float(running_parent_aux / max(1, n)))
        hist.setdefault("train_pxy", []).append(float(running_pxy / max(1, n)))
        hist["val_total"].append(float(val["loss"]))
        hist["val_reco"].append(float(val["loss_reco"]))
        hist["val_obj"].append(float(val["loss_obj"]))
        hist["val_parent_aux"].append(float(val["loss_parent_aux"]))
        hist.setdefault("val_pxy", []).append(float(val.get("loss_pxy", 0.0)))
        hist["val_mae_L"].append(float(val["mae_L_vs_m"]))
        hist["val_slot_prec"].append(float(val["slot_prec"]))
        hist["val_slot_rec"].append(float(val["slot_rec"]))

        print(
            f"[Reco+Obj] Ep {ep:03d}: train_total={hist['train_total'][-1]:.4f} "
            f"(reco={hist['train_reco'][-1]:.4f}, obj={hist['train_obj'][-1]:.4f}, p_aux={hist['train_parent_aux'][-1]:.4f}, pxy={hist['train_pxy'][-1]:.4f}) "
            f"val_total={val['loss']:.4f} (reco={val['loss_reco']:.4f}, obj={val['loss_obj']:.4f}, p_aux={val['loss_parent_aux']:.4f}, pxy={val.get('loss_pxy', 0.0):.4f}) "
            f"maeL={val['mae_L_vs_m']:.3f} slot_prec={val['slot_prec']:.3f} slot_rec={val['slot_rec']:.3f} "
            f"no_imp={no_imp}"
        )
        if no_imp >= int(cfg.patience):
            print("[Reco+Obj] Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    for p, f in orig_flags:
        p.requires_grad = bool(f)
    return {"best": best, "history": hist}


# -----------------------------
# Inference: build unmerged view (Ordered + objectness)
# -----------------------------


def pack_topN_raw(tokens_raw: np.ndarray, max_particles: int) -> tuple[np.ndarray, np.ndarray]:
    if tokens_raw.shape[0] == 0:
        out = np.zeros((max_particles, 4), dtype=np.float32)
        mask = np.zeros((max_particles,), dtype=np.bool_)
        return out, mask
    pts = tokens_raw[:, 0]
    order = np.argsort(-pts)
    tokens_raw = tokens_raw[order]
    m = min(int(max_particles), int(tokens_raw.shape[0]))
    out = np.zeros((max_particles, 4), dtype=np.float32)
    mask = np.zeros((max_particles,), dtype=np.bool_)
    out[:m] = tokens_raw[:m]
    mask[:m] = True
    return out, mask


def _obj_len_from_prob(prob_1d: np.ndarray, tau: float, mode: str) -> int:
    """从单个 parent 的 objectness 概率序列得到要取的 child 数。"""
    p = np.asarray(prob_1d, dtype=np.float32)
    K = int(p.shape[0])
    md = str(mode).lower()
    if md in ("prefix", "prefix_threshold", "prefix+threshold"):
        L = 0
        for i in range(K):
            if float(p[i]) > float(tau):
                L += 1
            else:
                break
        return int(L)
    if md in ("threshold", "any"):
        # 取所有 > tau 的 slot（保持 slot 顺序）
        return int(np.sum(p > float(tau)))
    if md in ("sum", "expectation"):
        return int(max(0, min(K, int(np.round(float(p.sum()))))))
    raise ValueError(f"Unknown obj_mode: {mode}")


@torch.no_grad()
def build_unmerged_view_ordered(
    model,
    device: torch.device,
    *,
    hlt_raw: np.ndarray,
    hlt_mask: np.ndarray,
    hlt_feat_std: np.ndarray,
    hlt_axis: np.ndarray,  # [N,4] => (eta,phi,pt,E)
    feat_means: np.ndarray,
    feat_stds: np.ndarray,
    max_particles: int,
    thr_parent: float = 0.7,
    thr_obj: float = 0.5,
    obj_mode: str = "prefix",
    max_split_parents: Optional[int] = 20,
    max_children_per_parent: Optional[int] = 8,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build unmerged raw+feats view by:
      - selecting parent tokens by parentness
      - decoding ordered children for each selected parent
      - using objectness to decide how many slots are real
      - packing top-N by pt
    """
    model.eval()
    N, S, _ = hlt_raw.shape
    out_raw = np.zeros((N, max_particles, 4), dtype=np.float32)
    out_mask = np.zeros((N, max_particles), dtype=np.bool_)

    bs = int(max(1, batch_size))
    for s0 in range(0, N, bs):
        s1 = min(N, s0 + bs)
        xb = torch.tensor(hlt_feat_std[s0:s1], dtype=torch.float32, device=device)
        mb = torch.tensor(hlt_mask[s0:s1], dtype=torch.bool, device=device)
        outb = model(xb, mb, parent_idx=None)
        prob_b = torch.sigmoid(outb.parent_logit).detach().cpu().numpy()  # [B,S]

        for bi, j in enumerate(range(s0, s1)):
            prob = prob_b[bi]
            valid = np.where(hlt_mask[j])[0]
            parents = np.asarray([i for i in valid if prob[int(i)] > float(thr_parent)], dtype=np.int64)
            if max_split_parents is not None and parents.shape[0] > int(max_split_parents):
                order = np.argsort(-prob[parents])[: int(max_split_parents)]
                parents = parents[order]

            keep = hlt_mask[j].copy()
            tokens: list[np.ndarray] = []
            for p in parents:
                keep[int(p)] = False
            keep_idx = np.where(keep)[0]
            if keep_idx.size > 0:
                tokens.append(hlt_raw[j, keep_idx])

            if parents.size > 0:
                P = int(parents.shape[0])
                xj = xb[bi : bi + 1]  # [1,S,D]
                mj = mb[bi : bi + 1]  # [1,S]
                xP = xj.repeat(P, 1, 1)
                mP = mj.repeat(P, 1)
                pidx = torch.tensor(parents, dtype=torch.long, device=device)
                outP = model(xP, mP, parent_idx=pidx)
                child = outP.child_feat.detach().cpu().numpy()  # [P,K,D] in standardized feat space
                obj_prob = torch.sigmoid(outP.obj_logit).detach().cpu().numpy()  # [P,K]

                for pi in range(P):
                    L = _obj_len_from_prob(obj_prob[pi], float(thr_obj), mode=str(obj_mode))
                    if max_children_per_parent is not None:
                        L = min(int(L), int(max_children_per_parent))
                    L = max(0, min(int(L), int(child.shape[1])))
                    if L <= 0:
                        continue
                    feats = child[pi, :L]  # [L,D] (std)
                    feats = feats * feat_stds[None, :] + feat_means[None, :]  # unstandardize
                    je = np.asarray([hlt_axis[j, 0]], dtype=np.float32)
                    jp = np.asarray([hlt_axis[j, 1]], dtype=np.float32)
                    raw_child = feats_to_raw(feats[None, ...], je, jp)[0]
                    tokens.append(raw_child)

            tokens_raw = np.concatenate(tokens, axis=0) if tokens else np.zeros((0, 4), dtype=np.float32)
            packed, pmask = pack_topN_raw(tokens_raw, max_particles=int(max_particles))
            out_raw[j] = packed
            out_mask[j] = pmask

    # Build standardized features in the same feature space for downstream (optional)
    D = int(np.asarray(feat_means).shape[0])
    kind = "7d" if D >= 7 else "4d"
    feats = raw_to_feats(out_raw, out_mask, hlt_axis, kind=kind)
    feats_std = standardize_tokens(feats, out_mask, feat_means, feat_stds, clip=10.0)
    return out_raw, out_mask, feats_std


# -----------------------------
# Downstream tagger datasets + KD utilities (copied from unmerger_k)
# -----------------------------


class JetTaggerDataset(Dataset):
    """
    One sample = one jet for classification.

    Provides OFF/HLT/UNM views so the same DataLoader can be reused across models.
    """

    def __init__(
        self,
        jet_indices: Sequence[int],
        *,
        y: np.ndarray,
        w: Optional[np.ndarray],
        off_feat_std: np.ndarray,
        off_mask: np.ndarray,
        hlt_feat_std: np.ndarray,
        hlt_mask: np.ndarray,
        unm_feat_std: Optional[np.ndarray] = None,
        unm_mask: Optional[np.ndarray] = None,
    ):
        self.jet_idx = np.asarray(jet_indices, dtype=np.int64)
        self.y = torch.tensor(y, dtype=torch.float32)
        if w is None:
            w = np.ones(len(y), dtype=np.float32)
        self.w = torch.tensor(w, dtype=torch.float32)
        self.off = torch.tensor(off_feat_std, dtype=torch.float32)
        self.off_mask = torch.tensor(off_mask, dtype=torch.bool)
        self.hlt = torch.tensor(hlt_feat_std, dtype=torch.float32)
        self.hlt_mask = torch.tensor(hlt_mask, dtype=torch.bool)
        self.unm = None if unm_feat_std is None else torch.tensor(unm_feat_std, dtype=torch.float32)
        self.unm_mask = None if unm_mask is None else torch.tensor(unm_mask, dtype=torch.bool)

    def __len__(self):
        return int(self.jet_idx.shape[0])

    def __getitem__(self, i):
        j = int(self.jet_idx[i])
        out = {
            "y": self.y[j],
            "w": self.w[j],
            "off": self.off[j],
            "off_mask": self.off_mask[j],
            "hlt": self.hlt[j],
            "hlt_mask": self.hlt_mask[j],
        }
        if self.unm is not None and self.unm_mask is not None:
            out["unm"] = self.unm[j]
            out["unm_mask"] = self.unm_mask[j]
        return out


def kd_kl_loss_binary(student_logit: torch.Tensor, teacher_logit: torch.Tensor, T: float) -> torch.Tensor:
    """KL between 2-class softmax distributions built from a single logit."""
    s2 = torch.stack([torch.zeros_like(student_logit), student_logit], dim=-1) / float(T)
    t2 = torch.stack([torch.zeros_like(teacher_logit), teacher_logit], dim=-1) / float(T)
    logp_s = F.log_softmax(s2, dim=-1)
    p_t = F.softmax(t2, dim=-1)
    return F.kl_div(logp_s, p_t, reduction="batchmean") * (float(T) ** 2)


def attn_loss(s_attn: torch.Tensor, t_attn: torch.Tensor, s_mask: torch.Tensor, t_mask: torch.Tensor) -> torch.Tensor:
    """Attention distillation loss (entropy + max), expects [B,S] attention weights."""
    eps = 1e-8
    if int(s_attn.shape[-1]) != int(t_attn.shape[-1]):
        L = int(min(int(s_attn.shape[-1]), int(t_attn.shape[-1])))
        s_attn = s_attn[:, :L]
        t_attn = t_attn[:, :L]
        s_mask = s_mask[:, :L]
        t_mask = t_mask[:, :L]
    s_valid = s_attn * s_mask.to(dtype=s_attn.dtype)
    t_valid = t_attn * t_mask.to(dtype=t_attn.dtype)
    s_ent = -(s_valid * torch.log(s_valid + eps)).sum(dim=1)
    t_ent = -(t_valid * torch.log(t_valid + eps)).sum(dim=1)
    return F.mse_loss(s_ent, t_ent) + F.mse_loss(s_valid.max(dim=1)[0], t_valid.max(dim=1)[0])


@torch.no_grad()
def eval_auc_logits(model, loader, device: torch.device, *, kind: str) -> float:
    """
    kind:
      - 'teacher'  : model(off, off_mask)
      - 'hlt'      : model(hlt, hlt_mask)
      - 'dual'     : model(hlt, hlt_mask, unm, unm_mask)
    """
    model.eval()
    ys = []
    ps = []
    for batch in loader:
        y = batch["y"].to(device)
        if kind == "teacher":
            logit = model(batch["off"].to(device), batch["off_mask"].to(device))
        elif kind == "hlt":
            logit = model(batch["hlt"].to(device), batch["hlt_mask"].to(device))
        elif kind == "dual":
            logit = model(
                batch["hlt"].to(device),
                batch["hlt_mask"].to(device),
                batch["unm"].to(device),
                batch["unm_mask"].to(device),
            )
        else:
            raise ValueError(kind)
        ys.append(y.detach().cpu().numpy())
        ps.append(torch.sigmoid(logit).detach().cpu().numpy())
    y_np = np.concatenate(ys, axis=0)
    p_np = np.concatenate(ps, axis=0)
    return float(roc_auc_score(y_np, p_np))


@torch.no_grad()
def collect_probs_logits(model, loader, device: torch.device, *, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (p, y) numpy arrays for ROC plotting."""
    model.eval()
    ys = []
    ps = []
    for batch in loader:
        y = batch["y"].to(device)
        if kind == "teacher":
            logit = model(batch["off"].to(device), batch["off_mask"].to(device))
        elif kind == "hlt":
            logit = model(batch["hlt"].to(device), batch["hlt_mask"].to(device))
        elif kind == "dual":
            logit = model(
                batch["hlt"].to(device),
                batch["hlt_mask"].to(device),
                batch["unm"].to(device),
                batch["unm_mask"].to(device),
            )
        else:
            raise ValueError(kind)
        ys.append(y.detach().cpu().numpy())
        ps.append(torch.sigmoid(logit).detach().cpu().numpy())
    y_np = np.concatenate(ys, axis=0).astype(np.float64)
    p_np = np.concatenate(ps, axis=0).astype(np.float64)
    return p_np, y_np


def plot_roc_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    *,
    title: str = "ROC (test)",
    save_path: str | Path | None = None,
    dpi: int = 160,
    log_fpr: bool = False,
    fpr_clip: float = 1e-6,
):
    """curves[name] = (fpr, tpr, auc)."""
    import matplotlib.pyplot as plt  # type: ignore

    plt.figure(figsize=(6.8, 6.0))
    for name, (fpr, tpr, auc) in curves.items():
        if bool(log_fpr):
            # x=TPR, y=FPR(log)（与 unsmear notebook 的绘图风格一致）
            y = np.clip(fpr, float(fpr_clip), 1.0)
            plt.semilogy(tpr, y, label=f"{name} (AUC={auc:.4f})")
        else:
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

    if bool(log_fpr):
        plt.xlabel("TPR")
        plt.ylabel("FPR (log)")
        plt.xlim(0.0, 1.0)
        plt.ylim(max(float(fpr_clip), 1e-6), 1.0)
        plt.grid(True, which="both", alpha=0.25)
    else:
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid(True, alpha=0.25)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p.as_posix(), dpi=int(dpi), bbox_inches="tight")
        print(f"Saved figure: {p}")
    plt.show()


def train_teacher(
    model,
    train_loader,
    val_loader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    grad_clip: float = 1.0,
) -> tuple[float, dict]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best = -1.0
    best_state = None
    no_imp = 0
    for ep in range(1, int(epochs) + 1):
        model.train()
        for batch in train_loader:
            y = batch["y"].to(device)
            x = batch["off"].to(device)
            m = batch["off_mask"].to(device)
            opt.zero_grad(set_to_none=True)
            logit = model(x, m)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
        val_auc = eval_auc_logits(model, val_loader, device, kind="teacher")
        improved = val_auc > best + 1e-4
        if improved:
            best = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        print(f"[Teacher] Ep {ep:03d}: val_auc={val_auc:.5f} best={best:.5f} no_imp={no_imp}")
        if no_imp >= int(patience):
            print("[Teacher] Early stopping.")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best, {"best_auc": best}


def train_hlt_baseline(
    model,
    train_loader,
    val_loader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    grad_clip: float = 1.0,
) -> tuple[float, dict]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best = -1.0
    best_state = None
    no_imp = 0
    for ep in range(1, int(epochs) + 1):
        model.train()
        for batch in train_loader:
            y = batch["y"].to(device)
            x = batch["hlt"].to(device)
            m = batch["hlt_mask"].to(device)
            opt.zero_grad(set_to_none=True)
            logit = model(x, m)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
        val_auc = eval_auc_logits(model, val_loader, device, kind="hlt")
        improved = val_auc > best + 1e-4
        if improved:
            best = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        print(f"[HLT] Ep {ep:03d}: val_auc={val_auc:.5f} best={best:.5f} no_imp={no_imp}")
        if no_imp >= int(patience):
            print("[HLT] Early stopping.")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best, {"best_auc": best}


def train_hlt_kd(
    student,
    teacher,
    train_loader,
    val_loader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    temperature: float = 3.0,
    alpha_kd: float = 0.5,
    alpha_attn: float = 0.0,
    grad_clip: float = 1.0,
) -> tuple[float, dict]:
    opt = torch.optim.AdamW(student.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best = -1.0
    best_state = None
    no_imp = 0
    teacher.eval()
    for ep in range(1, int(epochs) + 1):
        student.train()
        for batch in train_loader:
            y = batch["y"].to(device)
            hlt = batch["hlt"].to(device)
            hlt_m = batch["hlt_mask"].to(device)
            off = batch["off"].to(device)
            off_m = batch["off_mask"].to(device)
            opt.zero_grad(set_to_none=True)
            if float(alpha_attn) > 0.0:
                s_logit, s_attn = student(hlt, hlt_m, return_attention=True)
                ce = F.binary_cross_entropy_with_logits(s_logit, y)
                with torch.no_grad():
                    t_logit, t_attn = teacher(off, off_m, return_attention=True)
                kd = kd_kl_loss_binary(s_logit, t_logit, T=float(temperature))
                la = attn_loss(s_attn, t_attn, hlt_m, off_m)
                loss = (1.0 - float(alpha_kd)) * ce + float(alpha_kd) * kd + float(alpha_attn) * la
            else:
                s_logit = student(hlt, hlt_m)
                ce = F.binary_cross_entropy_with_logits(s_logit, y)
                with torch.no_grad():
                    t_logit = teacher(off, off_m)
                kd = kd_kl_loss_binary(s_logit, t_logit, T=float(temperature))
                loss = (1.0 - float(alpha_kd)) * ce + float(alpha_kd) * kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), float(grad_clip))
            opt.step()
        val_auc = eval_auc_logits(student, val_loader, device, kind="hlt")
        improved = val_auc > best + 1e-4
        if improved:
            best = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        print(f"[HLT+KD] Ep {ep:03d}: val_auc={val_auc:.5f} best={best:.5f} no_imp={no_imp}")
        if no_imp >= int(patience):
            print("[HLT+KD] Early stopping.")
            break
    if best_state is not None:
        student.load_state_dict(best_state)
    return best, {"best_auc": best}


def train_dual_student_kd(
    student,
    teacher,
    train_loader,
    val_loader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    temperature: float = 3.0,
    alpha_kd: float = 0.5,
    alpha_attn: float = 0.0,
    grad_clip: float = 1.0,
) -> tuple[float, dict]:
    opt = torch.optim.AdamW(student.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best = -1.0
    best_state = None
    no_imp = 0
    teacher.eval()
    for ep in range(1, int(epochs) + 1):
        student.train()
        for batch in train_loader:
            y = batch["y"].to(device)
            hlt = batch["hlt"].to(device)
            hlt_m = batch["hlt_mask"].to(device)
            unm = batch["unm"].to(device)
            unm_m = batch["unm_mask"].to(device)
            off = batch["off"].to(device)
            off_m = batch["off_mask"].to(device)
            opt.zero_grad(set_to_none=True)
            if float(alpha_attn) > 0.0:
                s_logit, s_attn = student(hlt, hlt_m, unm, unm_m, return_attention=True)
            else:
                s_logit = student(hlt, hlt_m, unm, unm_m)
            ce = F.binary_cross_entropy_with_logits(s_logit, y)
            with torch.no_grad():
                if float(alpha_attn) > 0.0:
                    t_logit, t_attn = teacher(off, off_m, return_attention=True)
                else:
                    t_logit = teacher(off, off_m)
            kd = kd_kl_loss_binary(s_logit, t_logit, T=float(temperature))
            loss = (1.0 - float(alpha_kd)) * ce + float(alpha_kd) * kd
            if float(alpha_attn) > 0.0:
                loss = loss + float(alpha_attn) * attn_loss(s_attn, t_attn, hlt_m, off_m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), float(grad_clip))
            opt.step()
        val_auc = eval_auc_logits(student, val_loader, device, kind="dual")
        improved = val_auc > best + 1e-4
        if improved:
            best = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        print(f"[Dual+KD] Ep {ep:03d}: val_auc={val_auc:.5f} best={best:.5f} no_imp={no_imp}")
        if no_imp >= int(patience):
            print("[Dual+KD] Early stopping.")
            break
    if best_state is not None:
        student.load_state_dict(best_state)
    return best, {"best_auc": best}

