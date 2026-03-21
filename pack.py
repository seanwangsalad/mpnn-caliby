#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
CALIBY_ROOT = REPO_ROOT / "caliby"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Caliby sidechain packing on a directory of PDB files.")
    parser.add_argument("--checkpoint", required=True, help="Exact Caliby checkpoint path or checkpoint name.")
    parser.add_argument("--pdb-dir", required=True, help="Directory containing input PDB files to pack.")
    parser.add_argument("--output-dir", required=True, help="Directory for packed outputs and metrics.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker count.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose backend logging.")
    return parser.parse_args()


def ensure_local_paths() -> None:
    path = str(CALIBY_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)

    os.environ.setdefault("PDB_MIRROR_PATH", "")
    os.environ.setdefault("CCD_MIRROR_PATH", "")
    os.environ.setdefault("MODEL_PARAMS_DIR", str(REPO_ROOT / "model_params"))


def main() -> None:
    args = parse_args()
    ensure_local_paths()

    import lightning as L
    import pandas as pd
    import torch
    from omegaconf import OmegaConf

    from caliby.eval.eval_utils import eval_metrics
    from caliby.eval.eval_utils.eval_setup_utils import get_pdb_files
    from caliby.eval.eval_utils.seq_des_utils import get_seq_des_model, run_sidechain_packing

    L.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = get_pdb_files(
        pdb_dir=str(Path(args.pdb_dir)),
        pdb_name_list=None,
        pdb_name_ext=None,
        n_subsample=None,
    )

    seq_des_cfg = OmegaConf.create(
        {
            "model_name": "atom_mpnn",
            "atom_mpnn": {
                "ckpt_name_or_path": args.checkpoint,
                "sampling_cfg": str(CALIBY_ROOT / "caliby" / "configs" / "seq_des" / "atom_mpnn_inference.yaml"),
                "overrides": {
                    "num_workers": args.num_workers,
                    "verbose": bool(args.verbose),
                },
            },
        }
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_des_model = get_seq_des_model(seq_des_cfg, device=device)

    outputs = run_sidechain_packing(
        model=seq_des_model["model"],
        data_cfg=seq_des_model["data_cfg"],
        sampling_cfg=seq_des_model["sampling_cfg"],
        pdb_paths=pdb_files,
        device=device,
        out_dir=str(output_dir),
    )
    del seq_des_model

    id_to_metrics = eval_metrics.run_packing_metrics_eval(
        input_pdbs=pdb_files,
        pred_pdbs=outputs["out_pdb"],
        out_dir=str(output_dir),
        num_workers=args.num_workers,
    )

    metrics_df = pd.DataFrame([{"example_id": eid, **metrics} for eid, metrics in id_to_metrics.items()])
    output_df = pd.DataFrame(outputs)
    output_df = pd.merge(
        output_df,
        metrics_df,
        left_on="example_id",
        right_on="example_id",
        how="inner",
        validate="1:1",
    )
    output_df.to_csv(output_dir / "packing_metrics.csv", index=False)


if __name__ == "__main__":
    main()
