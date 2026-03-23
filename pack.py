#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
CALIBY_ROOT = REPO_ROOT / "caliby"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thin wrapper around Caliby sidechain packing."
    )
    parser.add_argument("--checkpoint", required=True, help="Absolute Caliby checkpoint path.")
    parser.add_argument("--pdb-dir", required=True, help="Directory of PDB files to pack.")
    parser.add_argument("--output-dir", required=True, help="Destination output directory.")
    args = parser.parse_args()
    if not Path(args.checkpoint).expanduser().is_absolute():
        raise ValueError("--checkpoint must be an absolute path.")
    return args


def main() -> None:
    args = parse_args()

    env = os.environ.copy()
    env.setdefault("PDB_MIRROR_PATH", "")
    env.setdefault("CCD_MIRROR_PATH", "")
    env.setdefault("MODEL_PARAMS_DIR", str(REPO_ROOT / "model_params"))

    subprocess.run(
        [
            sys.executable,
            str(CALIBY_ROOT / "caliby" / "eval" / "sampling" / "sidechain_pack.py"),
            f"ckpt_name_or_path={args.checkpoint}",
            f"input_cfg.pdb_dir={args.pdb_dir}",
            f"out_dir={args.output_dir}",
        ],
        check=True,
        cwd=str(CALIBY_ROOT),
        env=env,
    )


if __name__ == "__main__":
    main()
