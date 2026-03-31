#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
CALIBY_ROOT = REPO_ROOT / "caliby"
AA1_TO_3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


def resolve_cli_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thin wrapper around Caliby sidechain packing."
    )
    parser.add_argument("--checkpoint", required=True, help="Caliby checkpoint path.")
    parser.add_argument("--pdb-dir", required=True, help="Directory of PDB files to pack.")
    parser.add_argument("--output-dir", required=True, help="Destination output directory.")
    parser.add_argument(
        "--designed-csv",
        default="",
        help="Optional inverse_fold output CSV. If provided, sequences are grafted onto backbone PDBs before packing.",
    )
    return parser.parse_args()


def parse_residue_order(backbone_path: Path) -> dict[str, list[tuple[int, str]]]:
    chain_residue_order: dict[str, list[tuple[int, str]]] = {}
    with backbone_path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            resnum = int(line[22:26])
            icode = line[26]
            key = (resnum, icode)
            chain_residue_order.setdefault(chain, [])
            if key not in chain_residue_order[chain]:
                chain_residue_order[chain].append(key)
    return chain_residue_order


def is_blank_cell(value: str) -> bool:
    value = str(value or "").strip()
    return not value or value.lower() in {"nan", "none", "null"}


def build_design_stem(name: str, seq_idx: str) -> str:
    clean_name = Path(str(name).strip()).stem
    clean_seq_idx = str(seq_idx).strip()
    if not clean_name:
        raise ValueError("Encountered an empty design name in designed CSV.")
    if not clean_seq_idx:
        raise ValueError(f"Encountered an empty seq_idx for {clean_name} in designed CSV.")
    return f"{clean_name}_{clean_seq_idx}"


def convert_packed_cifs_to_pdbs(output_dir: Path) -> None:
    packed_cif_dir = output_dir / "packed_samples"
    if not packed_cif_dir.is_dir():
        return

    cif_paths = sorted(path for path in packed_cif_dir.iterdir() if path.is_file() and path.suffix.lower() == ".cif")
    if not cif_paths:
        return

    import pymol

    packed_pdb_dir = output_dir / "packed_samples_pdbs"
    packed_pdb_dir.mkdir(parents=True, exist_ok=True)
    pymol.finish_launching(["pymol", "-qc"])
    try:
        for cif_path in cif_paths:
            object_name = "structure"
            out_path = packed_pdb_dir / f"{cif_path.stem}.pdb"
            pymol.cmd.load(str(cif_path), object_name)
            pymol.cmd.save(str(out_path), object_name)
            pymol.cmd.delete(object_name)
    finally:
        pymol.cmd.quit()


def graft_sequences_to_pdbs(pdb_dir: Path, csv_path: Path, output_dir: Path) -> tuple[Path, list[str]]:
    graft_dir = output_dir / "grafted_inputs"
    graft_dir.mkdir(parents=True, exist_ok=True)
    written_names: list[str] = []

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is empty.")

        fieldnames = [field.strip() for field in reader.fieldnames]
        reader.fieldnames = fieldnames
        chain_columns = [field for field in fieldnames if field not in {"name", "seq_idx", "score"} and len(field) == 1]
        if not chain_columns:
            raise ValueError(f"{csv_path} does not contain any single-letter chain columns.")

        written = 0
        for row in reader:
            name = Path(str(row["name"]).strip()).stem
            seq_idx = str(row["seq_idx"]).strip()
            design_stem = build_design_stem(name, seq_idx)
            backbone_path = pdb_dir / f"{name}.pdb"
            if not backbone_path.is_file():
                raise ValueError(f"Backbone PDB not found for {name}: {backbone_path}")

            chain_residue_order = parse_residue_order(backbone_path)
            resnum_to_aa: dict[str, dict[tuple[int, str], str]] = {}
            designed_chains = [chain for chain in chain_columns if not is_blank_cell(row.get(chain, ""))]
            if not designed_chains:
                raise ValueError(f"Row {name}/{seq_idx} has no designed chain sequences in {csv_path}")

            for chain in designed_chains:
                chain_seq = str(row.get(chain, "")).strip().upper()
                residues = chain_residue_order.get(chain, [])
                if not residues:
                    raise ValueError(f"Chain {chain} not found in backbone PDB {backbone_path}")
                if len(chain_seq) != len(residues):
                    raise ValueError(
                        f"Sequence length mismatch for {name} chain {chain}: got {len(chain_seq)} aa, "
                        f"expected {len(residues)} from backbone {backbone_path}"
                    )
                invalid = sorted({aa for aa in chain_seq if aa not in AA1_TO_3})
                if invalid:
                    raise ValueError(f"Unsupported amino acids in graft sequence for {name} chain {chain}: {invalid}")
                resnum_to_aa[chain] = {key: AA1_TO_3[aa1] for key, aa1 in zip(residues, chain_seq, strict=True)}

            out_lines: list[str] = []
            with backbone_path.open() as handle:
                for line in handle:
                    if line.startswith("ATOM"):
                        chain = line[21]
                        resnum = int(line[22:26])
                        icode = line[26]
                        key = (resnum, icode)
                        if chain in resnum_to_aa and key in resnum_to_aa[chain]:
                            line = line[:17] + resnum_to_aa[chain][key] + line[20:]
                    out_lines.append(line)

            out_path = graft_dir / f"{design_stem}.pdb"
            with out_path.open("w") as handle:
                handle.writelines(out_lines)
            written_names.append(out_path.name)
            written += 1

    if written == 0:
        raise ValueError(f"No grafted PDBs were written from {csv_path}")
    return graft_dir, written_names


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_cli_path(args.checkpoint)
    pdb_dir = resolve_cli_path(args.pdb_dir)
    output_dir = resolve_cli_path(args.output_dir)
    designed_csv = resolve_cli_path(args.designed_csv) if args.designed_csv else None

    output_dir.mkdir(parents=True, exist_ok=True)
    if designed_csv:
        input_pdb_dir, pdb_names = graft_sequences_to_pdbs(pdb_dir, designed_csv, output_dir)
    else:
        input_pdb_dir = pdb_dir
        pdb_names = sorted(
            path.name for path in input_pdb_dir.iterdir() if path.is_file() and path.suffix.lower() in {".pdb", ".cif"}
        )
    if not pdb_names:
        raise ValueError(f"No .pdb or .cif files found in {input_pdb_dir}")

    env = os.environ.copy()
    env.setdefault("PDB_MIRROR_PATH", "")
    env.setdefault("CCD_MIRROR_PATH", "")
    env.setdefault("MODEL_PARAMS_DIR", str(REPO_ROOT / "model_params"))
    env["PYTHONPATH"] = (
        f"{CALIBY_ROOT}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(CALIBY_ROOT)
    )

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("\n".join(pdb_names) + "\n")
        manifest_path = Path(handle.name)

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "caliby.eval.sampling.sidechain_pack",
                f"ckpt_name_or_path={checkpoint_path}",
                f"input_cfg.pdb_dir={input_pdb_dir}",
                f"input_cfg.pdb_name_list={manifest_path}",
                f"out_dir={output_dir}",
            ],
            check=True,
            cwd=str(CALIBY_ROOT),
            env=env,
        )
        convert_packed_cifs_to_pdbs(output_dir)
    finally:
        manifest_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
