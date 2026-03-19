#!/usr/bin/env python3
"""Run the joint pure-unsmear notebook with CLI-configured settings.

This executes notebook code cells directly (no Jupyter runtime needed), while
allowing key CONFIG values to be overridden from command line.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    repo_root = here.parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebook",
        type=Path,
        default=here / "unsmear_transformer_sharedencoder.ipynb",
        help="Notebook to execute.",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=repo_root / "data" / "test.h5",
        help="Path to input HDF5 file.",
    )
    parser.add_argument("--run_name", type=str, default="unsmear_transformer_sharedencoder_50j_40c")
    parser.add_argument("--n_jets", type=int, default=50)
    parser.add_argument("--max_particles", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--joint_unsmear_weight", type=float, default=1.0)
    parser.add_argument("--joint_cls_weight", type=float, default=1.0)
    parser.add_argument("--kd_enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kd_temperature", type=float, default=3.0)
    parser.add_argument("--kd_alpha", type=float, default=0.5)
    parser.add_argument("--kd_alpha_attn", type=float, default=0.0)
    parser.add_argument("--load_shared_baselines", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load_joint_model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resmear_each_epoch_baselines", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resmear_each_epoch_joint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--execute_until_cell",
        type=int,
        default=4,
        help="Execute notebook code cells from 0 through this index (inclusive).",
    )
    return parser.parse_args()


def _exec_cell(src: str, cell_idx: int, ctx: dict) -> None:
    code = compile(src, f"notebook_cell_{cell_idx}", "exec")
    exec(code, ctx)  # noqa: S102


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args()

    nb_path = args.notebook.resolve()
    if not nb_path.is_file():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")
    if not args.data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = ["".join(c.get("source", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    if not code_cells:
        raise RuntimeError("No code cells found in notebook.")

    old_cwd = Path.cwd()
    os.chdir(nb_path.parent)
    try:
        ctx: dict = {"__name__": "__main__"}
        _exec_cell(code_cells[0], 0, ctx)

        cfg = ctx["CONFIG"]
        cfg["data_path"] = str(args.data_path)
        cfg["n_jets"] = int(args.n_jets)
        cfg["max_particles"] = int(args.max_particles)
        cfg["load_shared_baselines"] = bool(args.load_shared_baselines)
        cfg["load_joint_model"] = bool(args.load_joint_model)

        safe_batch = max(1, min(int(args.batch_size), max(1, int(0.7 * int(args.n_jets)))))
        cfg["training"]["batch_size"] = int(safe_batch)
        cfg["training"]["epochs"] = int(args.epochs)
        cfg["training"]["warmup_epochs"] = int(args.warmup_epochs)
        cfg["training"]["patience"] = int(args.patience)
        cfg["training"]["lr"] = float(args.lr)
        cfg["training"]["weight_decay"] = float(args.weight_decay)
        cfg["training"]["joint_unsmear_weight"] = float(args.joint_unsmear_weight)
        cfg["training"]["joint_cls_weight"] = float(args.joint_cls_weight)
        cfg["training"]["resmear_each_epoch_baselines"] = bool(args.resmear_each_epoch_baselines)
        cfg["training"]["resmear_each_epoch_joint"] = bool(args.resmear_each_epoch_joint)

        cfg["kd"]["enable"] = bool(args.kd_enable)
        cfg["kd"]["temperature"] = float(args.kd_temperature)
        cfg["kd"]["alpha_kd"] = float(args.kd_alpha)
        cfg["kd"]["alpha_attn"] = float(args.kd_alpha_attn)

        cfg["joint_model"]["max_seq_len"] = int(args.max_particles)

        ctx["seed"] = int(args.seed)
        np_mod = ctx["np"]
        torch_mod = ctx["torch"]
        np_mod.random.seed(int(args.seed))
        torch_mod.manual_seed(int(args.seed))

        module_dir = Path(ctx["MODULE_DIR"]).resolve()
        run_name = str(args.run_name)
        out_dir = module_dir / "runs" / run_name
        fig_dir = out_dir / "figs"
        ckpt_dir = out_dir / "ckpts"

        ctx["RUN_NAME"] = run_name
        ctx["OUT_DIR"] = out_dir.as_posix()
        ctx["FIG_DIR"] = fig_dir.as_posix()
        ctx["CKPT_DIR"] = ckpt_dir.as_posix()

        tool_mod = ctx["tool"]
        tool_mod.ensure_dir(fig_dir)
        tool_mod.ensure_dir(ckpt_dir)
        tool_mod.ensure_dir(Path(ctx["SHARED_BASELINE_CKPT_DIR"]))

        feat_names = tool_mod.get_feat_names(cfg["feature_kind"])
        ctx["feat_names"] = feat_names
        cfg["joint_model"]["input_dim"] = len(feat_names)
        cfg["joint_model"]["max_seq_len"] = int(cfg["max_particles"])
        cfg["tagger"]["input_dim"] = len(feat_names)

        config_path = out_dir / "config.json"
        tool_mod.save_config(cfg, config_path)
        ctx["config_path"] = config_path.as_posix()

        print("==================================================")
        print("Running joint pure-unsmear shared-encoder notebook")
        print("==================================================")
        print(f"Notebook: {nb_path}")
        print(f"Data path: {cfg['data_path']}")
        print(f"Run dir: {out_dir}")
        print(f"n_jets={cfg['n_jets']} max_particles={cfg['max_particles']} batch_size={cfg['training']['batch_size']}")
        print(f"epochs={cfg['training']['epochs']} patience={cfg['training']['patience']} lr={cfg['training']['lr']}")
        print(f"kd_enable={cfg['kd']['enable']} kd_T={cfg['kd']['temperature']} kd_alpha={cfg['kd']['alpha_kd']}")
        print("==================================================")

        max_idx = min(int(args.execute_until_cell), len(code_cells) - 1)
        for i in range(1, max_idx + 1):
            print(f"\n[runner] Executing notebook cell {i}/{max_idx}")
            _exec_cell(code_cells[i], i, ctx)

        print("\n[runner] Completed successfully")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
