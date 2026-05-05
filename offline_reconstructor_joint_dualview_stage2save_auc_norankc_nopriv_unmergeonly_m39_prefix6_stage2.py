#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
m39 stage-2:
Train one final top tagger from six completed m39 prefix-specialist runs.

For each prefix run:
1) Load saved carry predictor + m28-style reconstructor.
2) Rebuild deterministic candidates for train/val/test with K=1 (that run's prefix).
3) Merge all six runs into one K=6 candidate pool per jet, deterministically rank by residual.
4) Train one final MultiCandidate dualview top tagger (NoGate + Gated).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m33_detfeas_dualview as m33
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m39_prefixspecialist_detresid_multicand as m39
import offline_reconstructor_joint_dualview_seq2seq_nexttoken_m28_sinkset_noar as m28
from unmerge_correct_hlt import compute_features, get_stats, standardize


def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _resolve_prefix_for_run(run_dir: Path, fallback: int) -> int:
    rpt = run_dir / "m39_report.json"
    if rpt.is_file():
        try:
            with open(rpt, "r", encoding="utf-8") as f:
                d = json.load(f)
            return int(d.get("candidate_generation", {}).get("max_prefix", fallback))
        except Exception:
            return int(fallback)
    return int(fallback)


def _resolve_quantization_for_run(run_dir: Path) -> Tuple[str, str]:
    rpt = run_dir / "m39_report.json"
    if rpt.is_file():
        try:
            with open(rpt, "r", encoding="utf-8") as f:
                d = json.load(f)
            q = d.get("quantization", {})
            cb_path = str(q.get("codebook_path", "")).strip()
            cb_label = str(q.get("codebook_label", "")).strip()
            return cb_path, cb_label
        except Exception:
            return "", ""
    return "", ""


def _gather_axis1(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    # idx: [N, M]
    if arr.ndim == 2:
        return np.take_along_axis(arr, idx, axis=1)
    if arr.ndim == 3:
        return np.take_along_axis(arr, idx[:, :, None], axis=1)
    if arr.ndim == 4:
        return np.take_along_axis(arr, idx[:, :, None, None], axis=1)
    raise ValueError(f"Unsupported ndim={arr.ndim} for gather")


def _combine_outputs(
    outputs: Sequence[m39.CandidateSplitOutput],
    keep_m: int,
    eps_total: float,
    eps_count: float,
) -> m39.CandidateSplitOutput:
    if len(outputs) < 1:
        raise RuntimeError("No candidate outputs to combine.")

    # Concat K=1 per run -> K=R combined.
    off_const = np.concatenate([o.off_const for o in outputs], axis=1)
    off_mask = np.concatenate([o.off_mask for o in outputs], axis=1)
    hlt_const = np.concatenate([o.hlt_const for o in outputs], axis=1)
    hlt_mask = np.concatenate([o.hlt_mask for o in outputs], axis=1)

    res_total = np.concatenate([o.res_total for o in outputs], axis=1)
    res_set = np.concatenate([o.res_set for o in outputs], axis=1)
    res_count = np.concatenate([o.res_count for o in outputs], axis=1)
    res_pt = np.concatenate([o.res_pt for o in outputs], axis=1)
    res_mass = np.concatenate([o.res_mass for o in outputs], axis=1)
    feasible = np.concatenate([o.feasible for o in outputs], axis=1)

    prefix_len = np.concatenate([o.prefix_len for o in outputs], axis=1)
    prefix_carry_mean = np.concatenate([o.prefix_carry_mean for o in outputs], axis=1)
    prefix_carry_min = np.concatenate([o.prefix_carry_min for o in outputs], axis=1)
    conf_mean = np.concatenate([o.conf_mean for o in outputs], axis=1)
    stop_len = np.concatenate([o.stop_len for o in outputs], axis=1)

    n, k = res_total.shape
    m = int(min(max(1, keep_m), k))
    order = np.argsort(res_total, axis=1)
    pick = order[:, :m]

    off_const_m = _gather_axis1(off_const, pick)
    off_mask_m = _gather_axis1(off_mask, pick)
    hlt_const_m = _gather_axis1(hlt_const, pick)
    hlt_mask_m = _gather_axis1(hlt_mask, pick)

    res_total_m = _gather_axis1(res_total, pick)
    res_set_m = _gather_axis1(res_set, pick)
    res_count_m = _gather_axis1(res_count, pick)
    res_pt_m = _gather_axis1(res_pt, pick)
    res_mass_m = _gather_axis1(res_mass, pick)
    feasible_m = _gather_axis1(feasible, pick)

    prefix_len_m = _gather_axis1(prefix_len, pick)
    prefix_carry_mean_m = _gather_axis1(prefix_carry_mean, pick)
    prefix_carry_min_m = _gather_axis1(prefix_carry_min, pick)
    conf_mean_m = _gather_axis1(conf_mean, pick)
    stop_len_m = _gather_axis1(stop_len, pick)

    rank_norm_m = np.zeros((n, m), dtype=np.float32)
    if m > 1:
        rank_norm_m[:] = np.arange(m, dtype=np.float32)[None, :] / float(m - 1)

    feas_all = (res_total <= float(eps_total)) & (res_count <= float(eps_count))
    feasible_count_all = feas_all.sum(axis=1).astype(np.float32)
    best_residual_all = res_total.min(axis=1).astype(np.float32)

    return m39.CandidateSplitOutput(
        off_const=off_const_m.astype(np.float32),
        off_mask=off_mask_m.astype(bool),
        hlt_const=hlt_const_m.astype(np.float32),
        hlt_mask=hlt_mask_m.astype(bool),
        res_total=res_total_m.astype(np.float32),
        res_set=res_set_m.astype(np.float32),
        res_count=res_count_m.astype(np.float32),
        res_pt=res_pt_m.astype(np.float32),
        res_mass=res_mass_m.astype(np.float32),
        feasible=feasible_m.astype(np.float32),
        prefix_len=prefix_len_m.astype(np.float32),
        prefix_carry_mean=prefix_carry_mean_m.astype(np.float32),
        prefix_carry_min=prefix_carry_min_m.astype(np.float32),
        conf_mean=conf_mean_m.astype(np.float32),
        stop_len=stop_len_m.astype(np.float32),
        rank_norm=rank_norm_m.astype(np.float32),
        feasible_count_all=feasible_count_all,
        best_residual_all=best_residual_all,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="m39 stage2: merge six prefix specialists and train one final dualview top tagger")

    p.add_argument("--train_path", type=str, default="./data")
    p.add_argument("--save_dir", type=str, default="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model39_prefix6_stage2")
    p.add_argument("--run_name", type=str, default="model39_prefix6_stage2_150k75k300k_seed0")

    p.add_argument("--stage1_save_dir", type=str, default="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model39_prefixspecialist_detresid_multicand")
    p.add_argument(
        "--stage1_run_names",
        type=str,
        default=(
            "model39_prefixspecialist_detresid_multicand_150k75k300k_seed0_pfx0,"
            "model39_prefixspecialist_detresid_multicand_150k75k300k_seed0_pfx3,"
            "model39_prefixspecialist_detresid_multicand_150k75k300k_seed0_pfx6,"
            "model39_prefixspecialist_detresid_multicand_150k75k300k_seed0_pfx9,"
            "model39_prefixspecialist_detresid_multicand_150k75k300k_seed0_pfx12,"
            "model39_prefixspecialist_detresid_multicand_150k75k300k_seed0_pfx15"
        ),
    )
    p.add_argument("--stage1_prefix_fallbacks", type=str, default="0,3,6,9,12,15")
    p.add_argument("--stage2_keep_m", type=int, default=6)

    p.add_argument("--n_train_jets", type=int, default=370000)
    p.add_argument("--n_train_split", type=int, default=50000)
    p.add_argument("--n_val_split", type=int, default=20000)
    p.add_argument("--n_test_split", type=int, default=300000)
    p.add_argument("--offset_jets", type=int, default=0)
    p.add_argument("--max_constits", type=int, default=100)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=80)
    p.add_argument("--candidate_gen_batch", type=int, default=64)
    p.add_argument("--use_train_weights", action="store_true")

    # deterministic D_hard
    p.add_argument("--merge_radius", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["merge_radius"]))
    p.add_argument("--eff_plateau_barrel", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["eff_plateau_barrel"]))
    p.add_argument("--eff_plateau_endcap", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["eff_plateau_endcap"]))
    p.add_argument("--smear_a", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_a"]))
    p.add_argument("--smear_b", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_b"]))
    p.add_argument("--smear_c", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_c"]))
    p.add_argument("--dhard_seed_offset", type=int, default=1337)

    # carry model arch (must match stage1)
    p.add_argument("--carry_embed_dim", type=int, default=128)
    p.add_argument("--carry_num_heads", type=int, default=4)
    p.add_argument("--carry_num_layers", type=int, default=3)
    p.add_argument("--carry_ff_dim", type=int, default=384)
    p.add_argument("--carry_dropout", type=float, default=0.10)

    # reco model arch (must match stage1)
    p.add_argument("--reco_embed_dim", type=int, default=384)
    p.add_argument("--reco_num_heads", type=int, default=8)
    p.add_argument("--reco_num_enc_layers", type=int, default=6)
    p.add_argument("--reco_num_dec_layers", type=int, default=6)
    p.add_argument("--reco_ff_dim", type=int, default=1024)
    p.add_argument("--reco_dropout", type=float, default=0.10)

    # seeded generation / residual ranking (should match stage1 runs)
    p.add_argument("--seed_temp", type=float, default=0.35)
    p.add_argument("--search_eps_total", type=float, default=0.60)
    p.add_argument("--search_eps_count", type=float, default=0.30)
    p.add_argument("--search_w_chamfer", type=float, default=1.0)
    p.add_argument("--search_w_count", type=float, default=0.25)
    p.add_argument("--search_w_pt", type=float, default=0.12)
    p.add_argument("--search_w_mass", type=float, default=0.08)

    # final dualview tagger
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--ff_dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--dual_epochs", type=int, default=90)
    p.add_argument("--dual_patience", type=int, default=16)
    p.add_argument("--dual_lr", type=float, default=1.2e-4)
    p.add_argument("--dual_weight_decay", type=float, default=1e-4)

    p.add_argument("--save_fusion_scores", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    m33.set_seed(int(args.seed))

    device = torch.device(args.device)
    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    run_names = _split_csv(args.stage1_run_names)
    fallback_prefixes = [int(x) for x in _split_csv(args.stage1_prefix_fallbacks)]
    if len(run_names) != len(fallback_prefixes):
        raise RuntimeError("stage1_run_names and stage1_prefix_fallbacks must have same length.")

    run_dirs = [Path(args.stage1_save_dir) / rn for rn in run_names]
    for rd in run_dirs:
        if not rd.is_dir():
            raise RuntimeError(f"Missing stage1 run directory: {rd}")

    prefixes = [
        _resolve_prefix_for_run(rd, pf)
        for rd, pf in zip(run_dirs, fallback_prefixes)
    ]

    print("=" * 72)
    print("Model-39 Prefix6 Stage2 (Single final top tagger from six reconstructor runs)")
    print(f"Run: {save_root}")
    print("Stage1 runs:")
    for rn, pf in zip(run_names, prefixes):
        print(f"  - {rn} (prefix={pf})")
    print("=" * 72)

    # ---------------------------------------------------------------------
    # Data + deterministic pseudo-HLT
    # ---------------------------------------------------------------------
    train_files = base._parse_h5_path_arg(str(args.train_path))
    max_needed = int(args.offset_jets + args.n_train_jets)
    all_const, all_labels, all_train_w = base.load_raw_constituents_labels_weights_from_h5(
        files=train_files,
        max_jets=max_needed,
        max_constits=int(args.max_constits),
        use_train_weights=bool(args.use_train_weights),
    )
    if all_const.shape[0] < max_needed:
        raise RuntimeError(f"Requested {max_needed} jets but found {all_const.shape[0]}.")

    const_raw = all_const[args.offset_jets : args.offset_jets + args.n_train_jets]
    labels = all_labels[args.offset_jets : args.offset_jets + args.n_train_jets].astype(np.int64)
    train_w = all_train_w[args.offset_jets : args.offset_jets + args.n_train_jets].astype(np.float32)

    total_need = int(args.n_train_split + args.n_val_split + args.n_test_split)
    if total_need > const_raw.shape[0]:
        raise RuntimeError(f"Requested splits sum to {total_need} but loaded jets are only {const_raw.shape[0]}")

    idx_all = np.arange(len(labels), dtype=np.int64)
    if total_need < len(idx_all):
        idx_use, _ = train_test_split(
            idx_all,
            train_size=total_need,
            random_state=int(args.seed),
            stratify=labels[idx_all],
        )
    else:
        idx_use = idx_all

    train_idx, rem_idx = train_test_split(
        idx_use,
        train_size=int(args.n_train_split),
        random_state=int(args.seed),
        stratify=labels[idx_use],
    )
    val_idx, test_idx = train_test_split(
        rem_idx,
        train_size=int(args.n_val_split),
        test_size=int(args.n_test_split),
        random_state=int(args.seed),
        stratify=labels[rem_idx],
    )

    cfg = base._deepcopy_config()
    cfg["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    cfg["hlt_effects"]["eff_plateau_barrel"] = float(args.eff_plateau_barrel)
    cfg["hlt_effects"]["eff_plateau_endcap"] = float(args.eff_plateau_endcap)
    cfg["hlt_effects"]["smear_a"] = float(args.smear_a)
    cfg["hlt_effects"]["smear_b"] = float(args.smear_b)
    cfg["hlt_effects"]["smear_c"] = float(args.smear_c)

    raw_mask = const_raw[:, :, 0] > 0.0
    mask_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~mask_off] = 0.0

    print("Generating pseudo-HLT (deterministic keyed D_hard)...")
    jet_keys = (np.arange(len(const_off), dtype=np.int64) + int(args.offset_jets)).astype(np.int64)
    const_hlt, mask_hlt, hlt_stats = m33._apply_hlt_effects_deterministic_keyed(
        const=const_off,
        mask=mask_off,
        cfg=cfg,
        jet_keys=jet_keys,
        base_seed=int(args.seed + args.dhard_seed_offset),
    )
    print(
        "HLT stats: "
        f"avg_offline={hlt_stats.get('avg_offline_per_jet', float('nan')):.2f}, "
        f"avg_hlt={hlt_stats.get('avg_hlt_per_jet', float('nan')):.2f}, "
        f"merged={hlt_stats.get('n_merged_pairs', 0)}, eff_lost={hlt_stats.get('n_lost_eff', 0)}"
    )
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    print("Computing standardized features for train/val/test...")
    feat_off_tr = compute_features(const_off[train_idx], mask_off[train_idx])
    feat_off_va = compute_features(const_off[val_idx], mask_off[val_idx])
    feat_off_te = compute_features(const_off[test_idx], mask_off[test_idx])
    feat_hlt_tr = compute_features(const_hlt[train_idx], mask_hlt[train_idx])
    feat_hlt_va = compute_features(const_hlt[val_idx], mask_hlt[val_idx])
    feat_hlt_te = compute_features(const_hlt[test_idx], mask_hlt[test_idx])

    means, stds = get_stats(feat_off_tr, mask_off[train_idx], np.arange(feat_off_tr.shape[0], dtype=np.int64))
    feat_off_tr = standardize(feat_off_tr, mask_off[train_idx], means, stds)
    feat_off_va = standardize(feat_off_va, mask_off[val_idx], means, stds)
    feat_off_te = standardize(feat_off_te, mask_off[test_idx], means, stds)
    feat_hlt_tr = standardize(feat_hlt_tr, mask_hlt[train_idx], means, stds)
    feat_hlt_va = standardize(feat_hlt_va, mask_hlt[val_idx], means, stds)
    feat_hlt_te = standardize(feat_hlt_te, mask_hlt[test_idx], means, stds)

    sw_train = train_w[train_idx] if bool(args.use_train_weights) else np.ones((len(train_idx),), dtype=np.float32)
    sw_val = train_w[val_idx] if bool(args.use_train_weights) else np.ones((len(val_idx),), dtype=np.float32)
    sw_test = train_w[test_idx] if bool(args.use_train_weights) else np.ones((len(test_idx),), dtype=np.float32)

    # ---------------------------------------------------------------------
    # Candidate generation from six stage1 runs
    # ---------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 1: Rebuild deterministic candidates from six m39 specialist runs")
    print("=" * 72)

    c_tr_runs: List[m39.CandidateSplitOutput] = []
    c_va_runs: List[m39.CandidateSplitOutput] = []
    c_te_runs: List[m39.CandidateSplitOutput] = []
    quant_info: List[Dict[str, str]] = []
    quantizer_cache: Dict[str, m39.TokenCodebookQuantizer] = {}

    for ridx, (rd, pf) in enumerate(zip(run_dirs, prefixes)):
        print(f"[{ridx+1}/{len(run_dirs)}] Loading stage1 models: {rd.name} (prefix={pf})")
        cb_path, cb_label = _resolve_quantization_for_run(rd)
        token_quantizer: Optional[m39.TokenCodebookQuantizer] = None
        if cb_path:
            if cb_path not in quantizer_cache:
                quantizer_cache[cb_path] = m39._load_token_codebook_quantizer(cb_path)
            token_quantizer = quantizer_cache[cb_path]
            print(f"  quantized decode: {token_quantizer.strategy} | {cb_path} ({cb_label})")
        quant_info.append({"run": rd.name, "codebook_path": cb_path, "codebook_label": cb_label})

        carry_ckpt = rd / "carry_predictor.pt"
        reco_ckpt = rd / "reco_completer_m28style.pt"
        if not carry_ckpt.is_file():
            raise RuntimeError(f"Missing checkpoint: {carry_ckpt}")
        if not reco_ckpt.is_file():
            raise RuntimeError(f"Missing checkpoint: {reco_ckpt}")

        carry_model = m39.CarryoverTokenPredictor(
            input_dim=7,
            embed_dim=int(args.carry_embed_dim),
            num_heads=int(args.carry_num_heads),
            num_layers=int(args.carry_num_layers),
            ff_dim=int(args.carry_ff_dim),
            dropout=float(args.carry_dropout),
            max_tokens=int(args.max_constits),
        ).to(device)
        carry_state = torch.load(carry_ckpt, map_location=device)
        carry_model.load_state_dict(carry_state["model"])

        reco_model = m28.HLT2OfflineSeq2Seq(
            input_dim_hlt=7,
            token_dim=5,
            embed_dim=int(args.reco_embed_dim),
            num_heads=int(args.reco_num_heads),
            num_enc_layers=int(args.reco_num_enc_layers),
            num_dec_layers=int(args.reco_num_dec_layers),
            ff_dim=int(args.reco_ff_dim),
            dropout=float(args.reco_dropout),
            max_hlt_tokens=int(args.max_constits),
            max_decode_tokens=int(args.max_constits),
            use_coord_residual_param=False,
            num_hypotheses=1,
        ).to(device)
        reco_state = torch.load(reco_ckpt, map_location=device)
        reco_model.load_state_dict(reco_state["model"])

        c_tr = m39._generate_candidates_split(
            reco_model=reco_model,
            carry_model=carry_model,
            feat_hlt=feat_hlt_tr,
            const_hlt=const_hlt[train_idx],
            mask_hlt=mask_hlt[train_idx],
            jet_keys=jet_keys[train_idx],
            cfg=cfg,
            seed_offset=int(args.seed + args.dhard_seed_offset),
            candidate_k=1,
            keep_m=1,
            max_prefix=int(pf),
            seed_temp=float(args.seed_temp),
            eps_total=float(args.search_eps_total),
            eps_count=float(args.search_eps_count),
            w_chamfer=float(args.search_w_chamfer),
            w_count=float(args.search_w_count),
            w_pt=float(args.search_w_pt),
            w_mass=float(args.search_w_mass),
            batch_size=int(args.candidate_gen_batch),
            device=device,
            seed=int(args.seed) + 1000 * ridx + 101,
            token_quantizer=token_quantizer,
        )
        c_va = m39._generate_candidates_split(
            reco_model=reco_model,
            carry_model=carry_model,
            feat_hlt=feat_hlt_va,
            const_hlt=const_hlt[val_idx],
            mask_hlt=mask_hlt[val_idx],
            jet_keys=jet_keys[val_idx],
            cfg=cfg,
            seed_offset=int(args.seed + args.dhard_seed_offset),
            candidate_k=1,
            keep_m=1,
            max_prefix=int(pf),
            seed_temp=float(args.seed_temp),
            eps_total=float(args.search_eps_total),
            eps_count=float(args.search_eps_count),
            w_chamfer=float(args.search_w_chamfer),
            w_count=float(args.search_w_count),
            w_pt=float(args.search_w_pt),
            w_mass=float(args.search_w_mass),
            batch_size=int(args.candidate_gen_batch),
            device=device,
            seed=int(args.seed) + 1000 * ridx + 202,
            token_quantizer=token_quantizer,
        )
        c_te = m39._generate_candidates_split(
            reco_model=reco_model,
            carry_model=carry_model,
            feat_hlt=feat_hlt_te,
            const_hlt=const_hlt[test_idx],
            mask_hlt=mask_hlt[test_idx],
            jet_keys=jet_keys[test_idx],
            cfg=cfg,
            seed_offset=int(args.seed + args.dhard_seed_offset),
            candidate_k=1,
            keep_m=1,
            max_prefix=int(pf),
            seed_temp=float(args.seed_temp),
            eps_total=float(args.search_eps_total),
            eps_count=float(args.search_eps_count),
            w_chamfer=float(args.search_w_chamfer),
            w_count=float(args.search_w_count),
            w_pt=float(args.search_w_pt),
            w_mass=float(args.search_w_mass),
            batch_size=int(args.candidate_gen_batch),
            device=device,
            seed=int(args.seed) + 1000 * ridx + 303,
            token_quantizer=token_quantizer,
        )

        c_tr_runs.append(c_tr)
        c_va_runs.append(c_va)
        c_te_runs.append(c_te)

        print(
            f"  bestR train/val/test = {np.mean(c_tr.best_residual_all):.4f} / "
            f"{np.mean(c_va.best_residual_all):.4f} / {np.mean(c_te.best_residual_all):.4f}"
        )
        del carry_model, reco_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    c_tr = _combine_outputs(
        outputs=c_tr_runs,
        keep_m=int(args.stage2_keep_m),
        eps_total=float(args.search_eps_total),
        eps_count=float(args.search_eps_count),
    )
    c_va = _combine_outputs(
        outputs=c_va_runs,
        keep_m=int(args.stage2_keep_m),
        eps_total=float(args.search_eps_total),
        eps_count=float(args.search_eps_count),
    )
    c_te = _combine_outputs(
        outputs=c_te_runs,
        keep_m=int(args.stage2_keep_m),
        eps_total=float(args.search_eps_total),
        eps_count=float(args.search_eps_count),
    )

    print(
        "Merged candidate stats: "
        f"train(feasible_all={np.mean(c_tr.feasible_count_all):.2f}/{len(run_dirs)}, bestR={np.mean(c_tr.best_residual_all):.4f}) "
        f"val(feasible_all={np.mean(c_va.feasible_count_all):.2f}/{len(run_dirs)}, bestR={np.mean(c_va.best_residual_all):.4f}) "
        f"test(feasible_all={np.mean(c_te.feasible_count_all):.2f}/{len(run_dirs)}, bestR={np.mean(c_te.best_residual_all):.4f})"
    )

    # ---------------------------------------------------------------------
    # Build arrays + train final top taggers
    # ---------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 2: Train final dualview top taggers from merged six-run candidates")
    print("=" * 72)

    mv_tr = m39._build_m38_multicandidate_arrays(c_tr, max_constits=int(args.max_constits), candidate_k=len(run_dirs))
    mv_va = m39._build_m38_multicandidate_arrays(c_va, max_constits=int(args.max_constits), candidate_k=len(run_dirs))
    mv_te = m39._build_m38_multicandidate_arrays(c_te, max_constits=int(args.max_constits), candidate_k=len(run_dirs))

    meta_tr = mv_tr["cand_meta"]
    meta_mean = meta_tr.reshape(-1, meta_tr.shape[-1]).mean(axis=0, keepdims=True).astype(np.float32)
    meta_std = (meta_tr.reshape(-1, meta_tr.shape[-1]).std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    summary_mean = mv_tr["summary_feat"].mean(axis=0, keepdims=True).astype(np.float32)
    summary_std = (mv_tr["summary_feat"].std(axis=0, keepdims=True) + 1e-6).astype(np.float32)

    mm = meta_mean.reshape(1, 1, -1)
    ms = meta_std.reshape(1, 1, -1)
    sm = summary_mean.reshape(1, -1)
    ss = summary_std.reshape(1, -1)
    mv_tr["cand_meta"] = ((mv_tr["cand_meta"] - mm) / ms).astype(np.float32)
    mv_va["cand_meta"] = ((mv_va["cand_meta"] - mm) / ms).astype(np.float32)
    mv_te["cand_meta"] = ((mv_te["cand_meta"] - mm) / ms).astype(np.float32)
    mv_tr["summary_feat"] = ((mv_tr["summary_feat"] - sm) / ss).astype(np.float32)
    mv_va["summary_feat"] = ((mv_va["summary_feat"] - sm) / ss).astype(np.float32)
    mv_te["summary_feat"] = ((mv_te["summary_feat"] - sm) / ss).astype(np.float32)

    ds_dv_tr = m39.DualViewM38Dataset(
        feat_hlt=feat_hlt_tr,
        mask_hlt=mask_hlt[train_idx],
        cand_tokens=mv_tr["cand_tokens"],
        cand_masks=mv_tr["cand_masks"],
        cand_meta=mv_tr["cand_meta"],
        summary_feat=mv_tr["summary_feat"],
        labels=labels[train_idx],
        sample_weight=sw_train,
    )
    ds_dv_va = m39.DualViewM38Dataset(
        feat_hlt=feat_hlt_va,
        mask_hlt=mask_hlt[val_idx],
        cand_tokens=mv_va["cand_tokens"],
        cand_masks=mv_va["cand_masks"],
        cand_meta=mv_va["cand_meta"],
        summary_feat=mv_va["summary_feat"],
        labels=labels[val_idx],
        sample_weight=sw_val,
    )
    ds_dv_te = m39.DualViewM38Dataset(
        feat_hlt=feat_hlt_te,
        mask_hlt=mask_hlt[test_idx],
        cand_tokens=mv_te["cand_tokens"],
        cand_masks=mv_te["cand_masks"],
        cand_meta=mv_te["cand_meta"],
        summary_feat=mv_te["summary_feat"],
        labels=labels[test_idx],
        sample_weight=sw_test,
    )

    dl_dv_tr = DataLoader(ds_dv_tr, batch_size=int(args.batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_dv_va = DataLoader(ds_dv_va, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    dl_dv_te = DataLoader(ds_dv_te, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    model_nog = m39.MultiCandidateM38NoGate(
        cand_meta_dim=int(mv_tr["cand_meta"].shape[-1]),
        summary_dim=int(mv_tr["summary_feat"].shape[-1]),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=max(2, int(args.num_layers)),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    model_nog, met_nog = m39._train_m38_model(
        model=model_nog,
        train_loader=dl_dv_tr,
        val_loader=dl_dv_va,
        device=device,
        epochs=int(args.dual_epochs),
        lr=float(args.dual_lr),
        weight_decay=float(args.dual_weight_decay),
        patience=int(args.dual_patience),
        name="M38Stage2NoGate",
    )

    model_gat = m39.MultiCandidateM38Gated(
        cand_meta_dim=int(mv_tr["cand_meta"].shape[-1]),
        summary_dim=int(mv_tr["summary_feat"].shape[-1]),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=max(2, int(args.num_layers)),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    model_gat, met_gat = m39._train_m38_model(
        model=model_gat,
        train_loader=dl_dv_tr,
        val_loader=dl_dv_va,
        device=device,
        epochs=int(args.dual_epochs),
        lr=float(args.dual_lr),
        weight_decay=float(args.dual_weight_decay),
        patience=int(args.dual_patience),
        name="M38Stage2Gated",
    )

    auc_nog, fpr50_nog, pred_nog, lab_final, w_final = m39._eval_m38_model(model_nog, dl_dv_te, device)
    auc_gat, fpr50_gat, pred_gat, _lab2, _w2 = m39._eval_m38_model(model_gat, dl_dv_te, device)

    print("\n" + "=" * 72)
    print("FINAL TEST")
    print("=" * 72)
    print(
        f"m39-stage2 NoGate AUC={auc_nog:.4f} FPR50={fpr50_nog:.6f} | "
        f"m39-stage2 Gated AUC={auc_gat:.4f} FPR50={fpr50_gat:.6f}"
    )

    torch.save({"model": model_nog.state_dict(), "metrics": met_nog}, save_root / "m39_stage2_nogate.pt")
    torch.save({"model": model_gat.state_dict(), "metrics": met_gat}, save_root / "m39_stage2_gated.pt")

    np.savez_compressed(
        save_root / "m39_stage2_test_scores.npz",
        labels_test=lab_final.astype(np.float32),
        preds_m39_stage2_nogate=pred_nog.astype(np.float32),
        preds_m39_stage2_gated=pred_gat.astype(np.float32),
        sample_weight=np.asarray(w_final, dtype=np.float32),
        auc_m39_stage2_nogate=float(auc_nog),
        auc_m39_stage2_gated=float(auc_gat),
        fpr50_m39_stage2_nogate=float(fpr50_nog),
        fpr50_m39_stage2_gated=float(fpr50_gat),
    )

    if bool(args.save_fusion_scores):
        np.savez_compressed(
            save_root / "fusion_scores_test.npz",
            labels_test=lab_final.astype(np.float32),
            preds_m39_stage2_nogate=np.asarray(pred_nog, dtype=np.float32),
            preds_m39_stage2_gated=np.asarray(pred_gat, dtype=np.float32),
            sample_weight=np.asarray(w_final, dtype=np.float32),
        )

    report = {
        "model": "m39_prefix6_stage2",
        "seed": int(args.seed),
        "stage1_run_names": run_names,
        "stage1_prefixes": [int(x) for x in prefixes],
        "stage1_quantization": quant_info,
        "split": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "candidate_generation": {
            "runs": int(len(run_dirs)),
            "keep_m": int(args.stage2_keep_m),
            "eps_total": float(args.search_eps_total),
            "eps_count": float(args.search_eps_count),
            "weights": {
                "chamfer": float(args.search_w_chamfer),
                "count": float(args.search_w_count),
                "pt": float(args.search_w_pt),
                "mass": float(args.search_w_mass),
            },
            "train": {
                "mean_feasible_all": float(np.mean(c_tr.feasible_count_all)),
                "mean_best_residual": float(np.mean(c_tr.best_residual_all)),
            },
            "val": {
                "mean_feasible_all": float(np.mean(c_va.feasible_count_all)),
                "mean_best_residual": float(np.mean(c_va.best_residual_all)),
            },
            "test": {
                "mean_feasible_all": float(np.mean(c_te.feasible_count_all)),
                "mean_best_residual": float(np.mean(c_te.best_residual_all)),
            },
        },
        "m39_stage2_nogate": {
            "auc_test": float(auc_nog),
            "fpr50_test": float(fpr50_nog),
            "metrics": met_nog,
        },
        "m39_stage2_gated": {
            "auc_test": float(auc_gat),
            "fpr50_test": float(fpr50_gat),
            "metrics": met_gat,
        },
    }
    with open(save_root / "m39_stage2_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    np.savez_compressed(
        save_root / "data_splits.npz",
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        means=means.astype(np.float32),
        stds=stds.astype(np.float32),
    )
    print(f"Saved: {save_root}")


if __name__ == "__main__":
    main()
