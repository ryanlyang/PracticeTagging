#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Load offline jet constituents from HDF5 and build a more realistic pseudo-HLT view.

Focus of this script:
  - Data loading (same style as unmerge_correct_hlt.py: raw 4-vectors from h5)
  - HLT generation with:
      1) pT threshold
      2) local merging
      3) pT/eta/density-dependent efficiency
      4) heteroscedastic smearing with heavy tails
      5) optional local reassignment

No model training is done here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from tqdm import tqdm


RANDOM_SEED = 52


def wrap_phi(phi: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(phi), np.cos(phi))


def load_raw_constituents_from_h5(
    files: List[Path],
    max_jets: int,
    max_constits: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw constituent 4-vectors [pT, eta, phi, E] and labels."""
    const_list = []
    label_list = []
    jets_read = 0

    for fname in files:
        if jets_read >= max_jets:
            break
        with h5py.File(fname, "r") as f:
            n_file = int(f["labels"].shape[0])
            take = min(n_file, max_jets - jets_read)

            pt = f["fjet_clus_pt"][:take, :max_constits].astype(np.float32)
            eta = f["fjet_clus_eta"][:take, :max_constits].astype(np.float32)
            phi = f["fjet_clus_phi"][:take, :max_constits].astype(np.float32)
            ene = f["fjet_clus_E"][:take, :max_constits].astype(np.float32)
            const = np.stack([pt, eta, phi, ene], axis=-1).astype(np.float32)

            labels = f["labels"][:take]
            if labels.ndim > 1:
                if labels.shape[1] == 1:
                    labels = labels[:, 0]
                else:
                    labels = np.argmax(labels, axis=1)
            labels = labels.astype(np.int64)

            const_list.append(const)
            label_list.append(labels)
            jets_read += take

    if not const_list:
        raise RuntimeError("No jets were loaded from input files.")

    all_const = np.concatenate(const_list, axis=0)
    all_labels = np.concatenate(label_list, axis=0)
    return all_const, all_labels


def _compute_local_density(
    eta: np.ndarray,
    phi: np.ndarray,
    valid_idx: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    Count nearby valid constituents within deltaR < radius for each valid token.
    Returns dense array aligned to valid_idx order.
    """
    if len(valid_idx) <= 1:
        return np.zeros(len(valid_idx), dtype=np.float32)
    eta_v = eta[valid_idx]
    phi_v = phi[valid_idx]
    deta = eta_v[:, None] - eta_v[None, :]
    dphi = wrap_phi(phi_v[:, None] - phi_v[None, :])
    dR = np.sqrt(deta * deta + dphi * dphi)
    neigh = (dR < radius).astype(np.int32)
    np.fill_diagonal(neigh, 0)
    return neigh.sum(axis=1).astype(np.float32)


def apply_hlt_effects_realistic_with_tracking(
    const: np.ndarray,
    mask: np.ndarray,
    cfg: Dict,
    seed: int = RANDOM_SEED,
):
    """
    Build pseudo-HLT constituents and track offline ancestry per HLT token.

    Returns:
      hlt_const, hlt_mask, origin_counts, origin_lists, stats
    """
    rs = np.random.RandomState(int(seed))
    hcfg = cfg["hlt_effects"]
    n_jets, max_part, _ = const.shape

    hlt = const.copy()
    hlt_mask = mask.copy()

    origin_counts = hlt_mask.astype(np.int32)
    origin_lists = [[([idx] if hlt_mask[j, idx] else []) for idx in range(max_part)] for j in range(n_jets)]

    n_initial = int(hlt_mask.sum())

    # ------------------------------------------------------------------ #
    # 1) HLT pT threshold (pre-merge)
    # ------------------------------------------------------------------ #
    pt_threshold = float(hcfg["pt_threshold_hlt"])
    below_threshold = (hlt[:, :, 0] < pt_threshold) & hlt_mask
    hlt_mask[below_threshold] = False
    hlt[~hlt_mask] = 0
    origin_counts[~hlt_mask] = 0
    for j in range(n_jets):
        for idx in np.where(below_threshold[j])[0]:
            origin_lists[j][idx] = []
    n_lost_threshold_pre = int(below_threshold.sum())

    # ------------------------------------------------------------------ #
    # 2) Local merging
    # ------------------------------------------------------------------ #
    n_merged = 0
    merge_radius = float(hcfg["merge_radius"])
    if hcfg["merge_enabled"] and merge_radius > 0:
        for jet_idx in range(n_jets):
            valid_idx = np.where(hlt_mask[jet_idx])[0]
            if len(valid_idx) < 2:
                continue

            to_remove = set()
            for i in range(len(valid_idx)):
                idx_i = valid_idx[i]
                if idx_i in to_remove:
                    continue
                for j in range(i + 1, len(valid_idx)):
                    idx_j = valid_idx[j]
                    if idx_j in to_remove:
                        continue
                    deta = hlt[jet_idx, idx_i, 1] - hlt[jet_idx, idx_j, 1]
                    dphi = wrap_phi(hlt[jet_idx, idx_i, 2] - hlt[jet_idx, idx_j, 2])
                    dR = float(np.sqrt(deta * deta + dphi * dphi))
                    if dR >= merge_radius:
                        continue

                    pt_i = float(hlt[jet_idx, idx_i, 0])
                    pt_j = float(hlt[jet_idx, idx_j, 0])
                    pt_sum = pt_i + pt_j
                    if pt_sum < 1e-8:
                        continue
                    w_i = pt_i / pt_sum
                    w_j = pt_j / pt_sum

                    hlt[jet_idx, idx_i, 0] = pt_sum
                    hlt[jet_idx, idx_i, 1] = w_i * hlt[jet_idx, idx_i, 1] + w_j * hlt[jet_idx, idx_j, 1]
                    phi_i = hlt[jet_idx, idx_i, 2]
                    phi_j = hlt[jet_idx, idx_j, 2]
                    hlt[jet_idx, idx_i, 2] = np.arctan2(
                        w_i * np.sin(phi_i) + w_j * np.sin(phi_j),
                        w_i * np.cos(phi_i) + w_j * np.cos(phi_j),
                    )
                    hlt[jet_idx, idx_i, 3] = hlt[jet_idx, idx_i, 3] + hlt[jet_idx, idx_j, 3]

                    origin_counts[jet_idx, idx_i] += origin_counts[jet_idx, idx_j]
                    origin_lists[jet_idx][idx_i].extend(origin_lists[jet_idx][idx_j])
                    to_remove.add(idx_j)
                    n_merged += 1

            for idx in to_remove:
                hlt_mask[jet_idx, idx] = False
                hlt[jet_idx, idx] = 0
                origin_counts[jet_idx, idx] = 0
                origin_lists[jet_idx][idx] = []

    # ------------------------------------------------------------------ #
    # 3) Realistic efficiency (pT/eta/density + per-jet quality)
    # ------------------------------------------------------------------ #
    # Per-jet latent quality (shared correlation across constituents in a jet).
    jet_q = rs.lognormal(mean=0.0, sigma=float(hcfg["jet_quality_sigma"]), size=n_jets).astype(np.float32)
    jet_q = np.clip(jet_q, float(hcfg["jet_quality_min"]), float(hcfg["jet_quality_max"]))

    # Local density for each token (neighbors within density_radius).
    density = np.zeros((n_jets, max_part), dtype=np.float32)
    for j in range(n_jets):
        valid = np.where(hlt_mask[j])[0]
        if len(valid) == 0:
            continue
        density_j = _compute_local_density(
            eta=hlt[j, :, 1],
            phi=hlt[j, :, 2],
            valid_idx=valid,
            radius=float(hcfg["density_radius"]),
        )
        density[j, valid] = density_j

    abs_eta = np.abs(hlt[:, :, 1])
    pt = np.maximum(hlt[:, :, 0], 1e-8)
    eta_plateau = np.where(
        abs_eta < hcfg["eta_break"],
        hcfg["eff_plateau_barrel"],
        hcfg["eff_plateau_endcap"],
    ).astype(np.float32)
    pt50 = np.where(
        abs_eta < hcfg["eta_break"],
        hcfg["eff_pt50_barrel"],
        hcfg["eff_pt50_endcap"],
    ).astype(np.float32)
    width = np.where(
        abs_eta < hcfg["eta_break"],
        hcfg["eff_width_barrel"],
        hcfg["eff_width_endcap"],
    ).astype(np.float32)
    turn_on = 1.0 / (1.0 + np.exp(-(pt - pt50) / np.maximum(width, 1e-6)))
    density_term = np.exp(-float(hcfg["eff_density_alpha"]) * density)
    q_eff = np.clip(jet_q[:, None], float(hcfg["eff_quality_min"]), float(hcfg["eff_quality_max"]))
    eps = eta_plateau * turn_on * density_term * q_eff
    eps = np.clip(eps, float(hcfg["eff_floor"]), float(hcfg["eff_ceil"]))

    u = rs.random_sample((n_jets, max_part))
    lost_eff = (u > eps) & hlt_mask
    hlt_mask[lost_eff] = False
    hlt[lost_eff] = 0
    origin_counts[lost_eff] = 0
    n_lost_eff = int(lost_eff.sum())
    for j in range(n_jets):
        for idx in np.where(lost_eff[j])[0]:
            origin_lists[j][idx] = []

    # ------------------------------------------------------------------ #
    # 4) Smearing + heavy tails + local reassignment (on surviving tokens)
    # ------------------------------------------------------------------ #
    n_reassigned = 0
    for j in range(n_jets):
        valid = np.where(hlt_mask[j])[0]
        if len(valid) == 0:
            continue

        pt_j = np.maximum(hlt[j, valid, 0], 1e-8)
        eta_j = hlt[j, valid, 1]
        phi_j = hlt[j, valid, 2]
        abs_eta_j = np.abs(eta_j)
        dens_j = density[j, valid]

        eta_scale = 1.0 + float(hcfg["smear_eta_scale"]) * abs_eta_j
        q = float(jet_q[j])

        # Relative pT resolution model: sqrt((a/sqrt(pt))^2 + b^2 + (c/pt)^2)
        sigma_rel = np.sqrt(
            (float(hcfg["smear_a"]) / np.sqrt(pt_j)) ** 2
            + float(hcfg["smear_b"]) ** 2
            + (float(hcfg["smear_c"]) / pt_j) ** 2
        )
        sigma_rel = sigma_rel * eta_scale * q
        sigma_rel = np.clip(sigma_rel, float(hcfg["smear_sigma_min"]), float(hcfg["smear_sigma_max"]))

        tail_prob = (
            float(hcfg["tail_base"])
            + float(hcfg["tail_eta_coeff"]) * abs_eta_j
            + float(hcfg["tail_density_coeff"]) * dens_j
        )
        tail_prob = np.clip(tail_prob, 0.0, float(hcfg["tail_prob_max"]))
        is_tail = rs.random_sample(len(valid)) < tail_prob

        ratio = rs.normal(loc=1.0, scale=sigma_rel)
        tail_sigma = float(hcfg["tail_sigma_scale"]) * sigma_rel + float(hcfg["tail_sigma_add"])
        ratio_tail = rs.normal(loc=float(hcfg["tail_mu"]), scale=tail_sigma)
        ratio[is_tail] = ratio_tail[is_tail]
        ratio = np.clip(ratio, float(hcfg["pt_resp_min"]), float(hcfg["pt_resp_max"]))
        pt_new = np.clip(pt_j * ratio, 1e-8, None)

        sigma_eta = (
            float(hcfg["eta_smear_const"])
            + float(hcfg["eta_smear_inv_sqrt"]) / np.sqrt(pt_j)
        ) * eta_scale * q
        sigma_phi = (
            float(hcfg["phi_smear_const"])
            + float(hcfg["phi_smear_inv_sqrt"]) / np.sqrt(pt_j)
        ) * eta_scale * q
        eta_new = eta_j + rs.normal(loc=0.0, scale=sigma_eta)
        phi_new = wrap_phi(phi_j + rs.normal(loc=0.0, scale=sigma_phi))

        # Optional local reassignment: move token slightly toward nearest neighbor.
        if float(hcfg["reassign_prob_base"]) > 0.0 and len(valid) > 1:
            p_reassign = float(hcfg["reassign_prob_base"]) + float(hcfg["reassign_density_coeff"]) * dens_j
            p_reassign = np.clip(p_reassign, 0.0, float(hcfg["reassign_prob_max"]))
            do_reassign = rs.random_sample(len(valid)) < p_reassign
            for ii in np.where(do_reassign)[0]:
                src = ii
                # Nearest neighbor in current smeared coordinates.
                deta = eta_new[src] - eta_new
                dphi = wrap_phi(phi_new[src] - phi_new)
                dR = np.sqrt(deta * deta + dphi * dphi)
                dR[src] = 1e9
                nn = int(np.argmin(dR))
                if dR[nn] > float(hcfg["reassign_radius"]):
                    continue
                lam = rs.uniform(float(hcfg["reassign_strength_min"]), float(hcfg["reassign_strength_max"]))
                eta_new[src] = (1.0 - lam) * eta_new[src] + lam * eta_new[nn]
                phi_new[src] = np.arctan2(
                    (1.0 - lam) * np.sin(phi_new[src]) + lam * np.sin(phi_new[nn]),
                    (1.0 - lam) * np.cos(phi_new[src]) + lam * np.cos(phi_new[nn]),
                )
                n_reassigned += 1

        eta_new = np.clip(eta_new, -5.0, 5.0)
        phi_new = wrap_phi(phi_new)
        e_new = pt_new * np.cosh(eta_new)  # keep massless consistency

        hlt[j, valid, 0] = pt_new
        hlt[j, valid, 1] = eta_new
        hlt[j, valid, 2] = phi_new
        hlt[j, valid, 3] = e_new

    # Optional post-smear threshold.
    post_thr = float(hcfg["post_smear_pt_threshold"])
    n_lost_threshold_post = 0
    if post_thr > 0:
        below_post = (hlt[:, :, 0] < post_thr) & hlt_mask
        hlt_mask[below_post] = False
        hlt[below_post] = 0
        origin_counts[below_post] = 0
        n_lost_threshold_post = int(below_post.sum())
        for j in range(n_jets):
            for idx in np.where(below_post[j])[0]:
                origin_lists[j][idx] = []

    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    hlt[~hlt_mask] = 0.0

    n_final = int(hlt_mask.sum())
    stats = {
        "n_jets": int(n_jets),
        "n_initial": int(n_initial),
        "n_lost_threshold_pre": int(n_lost_threshold_pre),
        "n_merged_pairs": int(n_merged),
        "n_lost_eff": int(n_lost_eff),
        "n_reassigned": int(n_reassigned),
        "n_lost_threshold_post": int(n_lost_threshold_post),
        "n_final": int(n_final),
        "avg_offline_per_jet": float(mask.sum(axis=1).mean()),
        "avg_hlt_per_jet": float(hlt_mask.sum(axis=1).mean()),
    }
    return hlt, hlt_mask, origin_counts, origin_lists, stats


def default_hlt_cfg() -> Dict:
    return {
        "hlt_effects": {
            # Thresholding/merging
            "pt_threshold_hlt": 1.5,
            "pt_threshold_offline": 0.5,
            "merge_enabled": True,
            "merge_radius": 0.01,
            # Efficiency model
            "eta_break": 1.5,
            "eff_plateau_barrel": 0.98,
            "eff_plateau_endcap": 0.94,
            "eff_pt50_barrel": 1.6,
            "eff_pt50_endcap": 1.9,
            "eff_width_barrel": 0.20,
            "eff_width_endcap": 0.25,
            "eff_density_alpha": 0.055,
            "eff_quality_min": 0.90,
            "eff_quality_max": 1.06,
            "eff_floor": 0.02,
            "eff_ceil": 0.995,
            # Smearing model
            "smear_a": 0.35,
            "smear_b": 0.012,
            "smear_c": 0.08,
            "smear_eta_scale": 0.08,
            "smear_sigma_min": 0.004,
            "smear_sigma_max": 0.40,
            "eta_smear_const": 0.0008,
            "eta_smear_inv_sqrt": 0.010,
            "phi_smear_const": 0.0008,
            "phi_smear_inv_sqrt": 0.010,
            # Heavy tails for pT response
            "tail_base": 0.015,
            "tail_eta_coeff": 0.010,
            "tail_density_coeff": 0.010,
            "tail_prob_max": 0.25,
            "tail_mu": 0.98,
            "tail_sigma_scale": 2.5,
            "tail_sigma_add": 0.015,
            "pt_resp_min": 0.40,
            "pt_resp_max": 1.60,
            # Density and reassignment
            "density_radius": 0.04,
            "reassign_prob_base": 0.01,
            "reassign_density_coeff": 0.006,
            "reassign_prob_max": 0.08,
            "reassign_radius": 0.08,
            "reassign_strength_min": 0.20,
            "reassign_strength_max": 0.65,
            # Per-jet correlated quality
            "jet_quality_sigma": 0.08,
            "jet_quality_min": 0.75,
            "jet_quality_max": 1.35,
            # Post-smear threshold (0 disables)
            "post_smear_pt_threshold": 0.0,
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_train_jets", type=int, default=200000)
    parser.add_argument("--offset_jets", type=int, default=0)
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument("--save_dir", type=str, default="checkpoints/realistic_hlt")
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--save_origin_lists", action="store_true")
    parser.add_argument("--merge_radius", type=float, default=None)
    parser.add_argument("--pt_threshold_hlt", type=float, default=None)
    parser.add_argument("--post_smear_pt_threshold", type=float, default=None)
    args = parser.parse_args()

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train_path)
    if train_path.is_dir():
        train_files = sorted(list(train_path.glob("*.h5")))
    else:
        train_files = [Path(p) for p in str(args.train_path).split(",") if p.strip()]
    if len(train_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in: {args.train_path}")

    max_jets_needed = args.offset_jets + args.n_train_jets
    print("Loading offline constituents from HDF5...")
    all_const_full, all_labels_full = load_raw_constituents_from_h5(
        train_files, max_jets=max_jets_needed, max_constits=args.max_constits
    )
    if all_const_full.shape[0] < max_jets_needed:
        raise RuntimeError(
            f"Not enough jets for offset {args.offset_jets} + n_train_jets {args.n_train_jets}. "
            f"Got {all_const_full.shape[0]}."
        )
    const_off = all_const_full[args.offset_jets : args.offset_jets + args.n_train_jets]
    labels = all_labels_full[args.offset_jets : args.offset_jets + args.n_train_jets]
    mask_off = const_off[:, :, 0] > 0.0

    cfg = default_hlt_cfg()
    if args.merge_radius is not None:
        cfg["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    if args.pt_threshold_hlt is not None:
        cfg["hlt_effects"]["pt_threshold_hlt"] = float(args.pt_threshold_hlt)
    if args.post_smear_pt_threshold is not None:
        cfg["hlt_effects"]["post_smear_pt_threshold"] = float(args.post_smear_pt_threshold)

    print("Generating realistic pseudo-HLT...")
    hlt_const, hlt_mask, origin_counts, origin_lists, stats = apply_hlt_effects_realistic_with_tracking(
        const_off, mask_off, cfg, seed=args.seed
    )

    print("\nHLT Simulation Statistics:")
    print(f"  Jets: {stats['n_jets']:,}")
    print(f"  Offline particles: {stats['n_initial']:,}")
    print(f"  Lost to pre-threshold: {stats['n_lost_threshold_pre']:,}")
    print(f"  Merging operations: {stats['n_merged_pairs']:,}")
    print(f"  Lost to efficiency: {stats['n_lost_eff']:,}")
    print(f"  Reassigned tokens: {stats['n_reassigned']:,}")
    print(f"  Lost to post-threshold: {stats['n_lost_threshold_post']:,}")
    print(f"  HLT particles: {stats['n_final']:,}")
    print(
        f"  Avg per jet: Offline={stats['avg_offline_per_jet']:.2f}, "
        f"HLT={stats['avg_hlt_per_jet']:.2f}"
    )

    # Save arrays for downstream pipelines.
    np.savez_compressed(
        save_root / "hlt_dataset.npz",
        const_off=const_off.astype(np.float32),
        mask_off=mask_off.astype(bool),
        hlt_const=hlt_const.astype(np.float32),
        hlt_mask=hlt_mask.astype(bool),
        labels=labels.astype(np.int64),
        origin_counts=origin_counts.astype(np.int16),
    )
    if args.save_origin_lists:
        np.save(save_root / "origin_lists.npy", np.array(origin_lists, dtype=object), allow_pickle=True)

    with open(save_root / "hlt_stats.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": cfg["hlt_effects"],
                "stats": stats,
                "seed": int(args.seed),
                "offset_jets": int(args.offset_jets),
                "n_train_jets": int(args.n_train_jets),
                "max_constits": int(args.max_constits),
                "saved_origin_lists": bool(args.save_origin_lists),
            },
            f,
            indent=2,
        )

    print(f"\nSaved HLT dataset to: {save_root}")
    print(f"  - {save_root / 'hlt_dataset.npz'}")
    print(f"  - {save_root / 'hlt_stats.json'}")
    if args.save_origin_lists:
        print(f"  - {save_root / 'origin_lists.npy'}")


if __name__ == "__main__":
    main()

