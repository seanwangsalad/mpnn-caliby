#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path


def resolve_cli_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quickly generate a fixed_positions CSV from a directory of PDBs."
    )
    parser.add_argument("--pdb-dir", required=True, help="Directory containing input PDB files.")
    parser.add_argument("--output-csv", required=True, help="Destination fixed_positions CSV.")
    parser.add_argument(
        "--fixed",
        default="",
        help="Comma-separated chain IDs to fully fix, e.g. A,B,C.",
    )
    return parser.parse_args()


def parse_fixed_chains(raw_value: str) -> set[str]:
    chains = {token.strip().upper() for token in raw_value.split(",") if token.strip()}
    invalid = sorted(chain for chain in chains if len(chain) != 1)
    if invalid:
        raise ValueError(f"--fixed must be a comma-separated list of one-letter chain IDs: {invalid}")
    return chains


def collect_chain_ids(pdb_path: Path) -> list[str]:
    chains: OrderedDict[str, None] = OrderedDict()
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            chain = line[21].strip()
            if not chain:
                continue
            chains.setdefault(chain, None)
    if not chains:
        raise ValueError(f"No ATOM records with chain IDs found in {pdb_path}")
    return list(chains.keys())


def main() -> None:
    args = parse_args()
    pdb_dir = resolve_cli_path(args.pdb_dir)
    output_csv = resolve_cli_path(args.output_csv)
    fixed_chains = parse_fixed_chains(args.fixed)

    pdb_paths = sorted(path for path in pdb_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdb")
    if not pdb_paths:
        raise ValueError(f"No .pdb files found in {pdb_dir}")

    per_target_chains: list[tuple[str, list[str]]] = []
    all_chains: OrderedDict[str, None] = OrderedDict()
    for pdb_path in pdb_paths:
        target_name = pdb_path.stem
        chain_ids = collect_chain_ids(pdb_path)
        per_target_chains.append((target_name, chain_ids))
        for chain_id in chain_ids:
            all_chains.setdefault(chain_id, None)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", *all_chains.keys()]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for target_name, chain_ids in per_target_chains:
            row = {field: "" for field in fieldnames}
            row["name"] = target_name
            for chain_id in chain_ids:
                row[chain_id] = "1-" if chain_id in fixed_chains else ""
            writer.writerow(row)


if __name__ == "__main__":
    main()
