#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import shutil
import sys
import tempfile
from collections import OrderedDict, defaultdict
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parent
CALIBY_ROOT = REPO_ROOT / "caliby"
PROTEIN_MPNN_ROOT = REPO_ROOT / "ProteinMPNN"
PositionSpan = tuple[int | None, int | None]
AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")
PDB_AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inverse folding with either Caliby or ProteinMPNN and normalize outputs to one CSV."
    )
    parser.add_argument("--pdb-dir", required=True, help="Directory containing input PDB files.")
    parser.add_argument(
        "--fixed-positions-csv",
        required=True,
        help='CSV with columns name,A,B,C,... where each chain cell is a quoted range string like "1-10,20-30,40-50", "15-", or "-15".',
    )
    parser.add_argument("--model-type", required=True, choices=["caliby", "mpnn", "proteinmpnn"])
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path. Relative paths are resolved from the current working directory. For ProteinMPNN, this must be an exact .pt file path.",
    )
    parser.add_argument(
        "--restricted-aas",
        default="",
        help="Globally omitted amino acids as a string of one-letter codes, e.g. CFLY.",
    )
    parser.add_argument(
        "--bias-jsonl",
        default="",
        help='Optional ProteinMPNN amino-acid bias dictionary string, e.g. \'{"W": 1.5, "F": 1.5}\'. Ignored by Caliby.',
    )
    parser.add_argument(
        "--num_seq_per_target",
        type=int,
        required=True,
        help="Number of sampled sequences generated per target/PDB.",
    )
    parser.add_argument("--output-csv", default="inverse_fold_outputs.csv", help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed. 0 keeps backend default behavior.")
    parser.add_argument(
        "--temp",
        "-temp",
        type=float,
        default=None,
        help="Sampling temperature. Defaults to 0.01 for Caliby and 0.6 for ProteinMPNN.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Caliby num_workers, and ProteinMPNN batch_size. Defaults to 1.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose backend logging.")
    args = parser.parse_args()

    if args.num_seq_per_target < 1:
        raise ValueError("--num_seq_per_target must be >= 1.")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1.")
    if args.model_type in {"mpnn", "proteinmpnn"} and args.num_seq_per_target % args.num_workers != 0:
        raise ValueError("For ProteinMPNN, --num_seq_per_target must be divisible by --num-workers.")
    if args.model_type == "caliby" and args.bias_jsonl:
        raise ValueError("--bias-jsonl is only supported for ProteinMPNN.")

    return args


def ensure_local_paths() -> None:
    for path in (str(CALIBY_ROOT), str(PROTEIN_MPNN_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)

    os.environ.setdefault("PDB_MIRROR_PATH", "")
    os.environ.setdefault("CCD_MIRROR_PATH", "")
    os.environ.setdefault("MODEL_PARAMS_DIR", str(REPO_ROOT / "model_params"))


def resolve_cli_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def normalize_name(raw_name: str) -> str:
    name = str(raw_name).strip()
    if not name:
        raise ValueError("Encountered an empty PDB name in fixed_positions.csv.")
    return Path(name).stem


def parse_restricted_aas(raw_value: str) -> str:
    value = (raw_value or "").strip()
    if not value:
        return ""

    parsed = [char.upper() for char in value if not char.isspace() and char != ","]

    invalid = sorted({aa for aa in parsed if aa not in AA_ALPHABET})
    if invalid:
        raise ValueError(f"Unsupported amino acids in --restricted-aas: {invalid}")
    return "".join(dict.fromkeys(parsed))


def get_mpnn_omit_aas(restricted_aas: str) -> str:
    return "".join(dict.fromkeys((restricted_aas or "") + "X"))


def parse_bias_dict(raw_value: str) -> dict[str, float]:
    value = (raw_value or "").strip()
    if not value:
        return {}

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            "--bias-jsonl must be a Python-style dictionary string such as "
            '\'{"W": 1.5, "F": 1.5, "Y": 1.0, "L": 1.0, "I": 1.0}\'.'
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("--bias-jsonl must parse to a dictionary of amino-acid biases.")

    normalized: dict[str, float] = {}
    for key, val in parsed.items():
        aa = str(key).strip().upper()
        if aa not in AA_ALPHABET or len(aa) != 1:
            raise ValueError(f"Unsupported amino acid in --bias-jsonl: {key!r}")
        try:
            normalized[aa] = float(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Bias value for amino acid {aa!r} must be numeric, got {val!r}") from exc
    return normalized


def write_bias_jsonl(pdb_dir: Path, bias_dict: dict[str, float]) -> Path:
    bias_path = pdb_dir / "proteinmpnn_bias.jsonl"
    bias_path.write_text(json.dumps(bias_dict, sort_keys=True) + "\n")
    return bias_path


def parse_position_cell(raw_value: str) -> list[PositionSpan]:
    value = str(raw_value or "").strip()
    if not value or value.lower() in {"nan", "none", "null", "[]"}:
        return []

    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1].strip()
        if not value:
            return []

    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return []

    spans: list[PositionSpan] = []
    for item in items:
        token = str(item).strip()
        match = re.fullmatch(r"(\d+)-(\d+)", token)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            if end < start:
                raise ValueError(f"Invalid residue range: {token}")
            spans.append((start, end))
        elif match := re.fullmatch(r"(\d+)-", token):
            spans.append((int(match.group(1)), None))
        elif match := re.fullmatch(r"-(\d+)", token):
            spans.append((None, int(match.group(1))))
        elif re.fullmatch(r"\d+", token):
            position = int(token)
            spans.append((position, position))
        else:
            raise ValueError(
                f"Invalid fixed-position token {token!r}. Expected comma-separated residues or ranges like "
                '"1-10,20-30,40-50", "15-", or "-15".'
            )

    for start, end in spans:
        if start is not None and start < 1:
            raise ValueError(f"Residue indices must be >= 1: {start}")
        if end is not None and end < 1:
            raise ValueError(f"Residue indices must be >= 1: {end}")
    return spans


def resolve_position_spans(spans: list[PositionSpan], chain_length: int) -> list[int]:
    positions: list[int] = []
    for start, end in spans:
        resolved_start = 1 if start is None else start
        resolved_end = chain_length if end is None else end
        if resolved_start > chain_length:
            continue
        if resolved_end < 1:
            continue
        if resolved_end < resolved_start:
            raise ValueError(f"Invalid fixed-position span after resolving against chain length {chain_length}: {(start, end)}")
        positions.extend(range(max(1, resolved_start), min(chain_length, resolved_end) + 1))
    return sorted(dict.fromkeys(positions))


def load_fixed_positions_table(path: Path) -> tuple[list[str], OrderedDict[str, dict[str, list[PositionSpan]]]]:
    with path.open(newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        if not sample:
            raise ValueError(f"{path} is empty.")
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(handle, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty.")

        fieldnames = [field.lstrip("\ufeff").strip() for field in reader.fieldnames]
        reader.fieldnames = fieldnames
        if "name" not in fieldnames:
            raise ValueError("fixed_positions.csv must include a 'name' column.")

        chain_columns = [field for field in fieldnames if field != "name"]
        if not chain_columns:
            raise ValueError("fixed_positions.csv must include at least one chain column such as A or B.")
        invalid_columns = [field for field in chain_columns if not re.fullmatch(r"[A-Za-z]", field)]
        if invalid_columns:
            raise ValueError(f"Chain columns must be single-letter chain IDs. Invalid columns: {invalid_columns}")

        rows: OrderedDict[str, dict[str, list[PositionSpan]]] = OrderedDict()
        for raw_row in reader:
            name = normalize_name(raw_row["name"])
            if name in rows:
                raise ValueError(f"Duplicate PDB name in fixed_positions.csv: {name}")
            rows[name] = {chain: parse_position_cell(raw_row.get(chain, "")) for chain in chain_columns}

    return chain_columns, rows


def resolve_named_inputs(
    pdb_dir: Path,
    names: list[str],
    *,
    allow_directories: bool,
) -> tuple[OrderedDict[str, Path], dict[str, str]]:
    input_paths: OrderedDict[str, Path] = OrderedDict()
    input_kinds: dict[str, str] = {}

    for name in names:
        pdb_path = pdb_dir / f"{name}.pdb"
        dir_path = pdb_dir / name
        has_pdb = pdb_path.is_file()
        has_dir = dir_path.is_dir()

        if has_pdb and has_dir:
            raise ValueError(f"Ambiguous input for {name!r}: both {pdb_path} and {dir_path} exist.")
        if has_pdb:
            input_paths[name] = pdb_path
            input_kinds[name] = "pdb"
            continue
        if has_dir:
            if not allow_directories:
                raise ValueError(f"ProteinMPNN requires {name} to resolve to {pdb_path}, not the directory {dir_path}.")
            input_paths[name] = dir_path
            input_kinds[name] = "dir"
            continue
        raise ValueError(f"Could not resolve {name!r} under {pdb_dir}. Expected either {pdb_path.name} or directory {name}/.")

    return input_paths, input_kinds


def resolve_primary_conformer_pdb(ensemble_dir: Path) -> Path:
    primary_pdb = ensemble_dir / f"{ensemble_dir.name}.pdb"
    if not primary_pdb.is_file():
        raise ValueError(
            f"Ensemble directory {ensemble_dir} must contain a primary conformer named {primary_pdb.name} "
            "for fixed-position parsing."
        )
    return primary_pdb


def parse_pdb_chain_sequences(pdb_path: Path) -> OrderedDict[str, str]:
    chains: OrderedDict[str, list[str]] = OrderedDict()
    seen_residues: set[tuple[str, str]] = set()
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            resname = line[17:20].strip().upper()
            if line.startswith("HETATM") and resname != "MSE":
                continue
            chain_id = line[21].strip()
            if not chain_id:
                continue
            residue_key = (chain_id, f"{line[22:26].strip()}{line[26].strip()}")
            if residue_key in seen_residues:
                continue
            seen_residues.add(residue_key)
            chains.setdefault(chain_id, []).append(PDB_AA3_TO_AA1.get(resname, "X"))

    if not chains:
        raise ValueError(f"Failed to extract chain sequences from {pdb_path}")
    return OrderedDict((chain_id, "".join(seq)) for chain_id, seq in chains.items())


def collapse_positions(chain_id: str, positions: list[int]) -> list[str]:
    if not positions:
        return []

    collapsed: list[str] = []
    start = prev = positions[0]
    for position in positions[1:]:
        if position == prev + 1:
            prev = position
            continue
        collapsed.append(f"{chain_id}{start}" if start == prev else f"{chain_id}{start}-{prev}")
        start = prev = position
    collapsed.append(f"{chain_id}{start}" if start == prev else f"{chain_id}{start}-{prev}")
    return collapsed


def split_sequence_for_chains(sequence: str, chains: list[str]) -> dict[str, str]:
    parts = sequence.split("/") if sequence else []
    if len(parts) != len(chains):
        raise ValueError(f"Expected {len(chains)} chain sequences, got {len(parts)} from sequence: {sequence}")
    return dict(zip(chains, parts, strict=True))


def load_inputs(
    pdb_dir: Path, fixed_positions_csv: Path, model_type: str
) -> tuple[
    list[str],
    OrderedDict[str, Path],
    dict[str, str],
    OrderedDict[str, dict[str, list[int]]],
    dict[str, OrderedDict[str, str]],
]:
    chain_columns, fixed_position_specs = load_fixed_positions_table(fixed_positions_csv)
    input_paths, input_kinds = resolve_named_inputs(
        pdb_dir,
        list(fixed_position_specs.keys()),
        allow_directories=model_type == "caliby",
    )

    chain_sequences = {}
    for name, path in input_paths.items():
        if input_kinds[name] == "pdb":
            chain_sequences[name] = parse_pdb_chain_sequences(path)
        else:
            chain_sequences[name] = parse_pdb_chain_sequences(resolve_primary_conformer_pdb(path))

    fixed_positions: OrderedDict[str, dict[str, list[int]]] = OrderedDict()
    for name, chains in chain_sequences.items():
        available = set(chains)
        invalid = [chain for chain in chain_columns if fixed_position_specs[name].get(chain) and chain not in available]
        if invalid:
            target_label = f"{name}.pdb" if input_kinds[name] == "pdb" else f"{name}/"
            raise ValueError(f"{target_label} has no chains matching constrained columns: {invalid}")
        fixed_positions[name] = {}
        for chain in chain_columns:
            spans = fixed_position_specs[name].get(chain, [])
            if chain not in chains:
                fixed_positions[name][chain] = []
                continue
            fixed_positions[name][chain] = resolve_position_spans(spans, len(chains[chain]))

    return chain_columns, input_paths, input_kinds, fixed_positions, chain_sequences


def build_output_rows(
    chain_columns: list[str],
    per_name_records: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in sorted(per_name_records):
        for record in sorted(per_name_records[name], key=lambda item: int(item["seq_idx"])):
            row = {"name": name, "seq_idx": record["seq_idx"]}
            for chain in chain_columns:
                row[chain] = record["chains"].get(chain, "")
            row["score"] = record["score"]
            rows.append(row)
    return rows


def write_output_csv(path: Path, chain_columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "seq_idx", *chain_columns, "score"])
        writer.writeheader()
        writer.writerows(rows)


def get_sanity_output_path(output_csv: str, suffix: str) -> Path:
    output_path = Path(output_csv)
    return output_path.with_name(f"{output_path.stem}{suffix}")


def get_effective_temp(args: argparse.Namespace) -> float:
    if args.temp is not None:
        return args.temp
    if args.model_type == "caliby":
        return 0.01
    return 0.6


def run_caliby(
    args: argparse.Namespace,
    chain_columns: list[str],
    input_paths: OrderedDict[str, Path],
    input_kinds: dict[str, str],
    fixed_positions: OrderedDict[str, dict[str, list[int]]],
    chain_sequences: dict[str, OrderedDict[str, str]],
) -> list[dict[str, object]]:
    ensure_local_paths()

    import lightning as L
    import pandas as pd
    import torch
    from omegaconf import OmegaConf

    from caliby.eval.eval_utils.eval_setup_utils import get_ensemble_constraint_df, process_conformer_dirs
    from caliby.eval.eval_utils.seq_des_utils import get_seq_des_model, run_seq_des, run_seq_des_ensemble

    L.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)

    single_pdb_map = OrderedDict((name, path) for name, path in input_paths.items() if input_kinds[name] == "pdb")
    ensemble_dir_map = OrderedDict((name, path) for name, path in input_paths.items() if input_kinds[name] == "dir")
    constraint_rows = []
    for name in input_paths:
        fixed_tokens: list[str] = []
        for chain in chain_columns:
            if chain not in chain_sequences[name]:
                continue
            fixed_tokens.extend(collapse_positions(chain, fixed_positions[name].get(chain, [])))
        constraint_rows.append(
            {
                "pdb_key": name,
                "fixed_pos_seq": ",".join(fixed_tokens) if fixed_tokens else None,
            }
        )
    pos_constraint_df = pd.DataFrame(constraint_rows)
    sanity_csv_path = get_sanity_output_path(args.output_csv, ".caliby_constraints.csv")
    sanity_inputs_csv_path = get_sanity_output_path(args.output_csv, ".caliby_inputs.csv")
    sanity_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pos_constraint_df.to_csv(sanity_csv_path, index=False)
    caliby_inputs_df = pos_constraint_df.copy()
    caliby_inputs_df["restricted_aas"] = args.restricted_aas or ""
    caliby_inputs_df["temp"] = get_effective_temp(args)
    caliby_inputs_df.to_csv(sanity_inputs_csv_path, index=False)

    sampling_cfg_path = CALIBY_ROOT / "caliby" / "configs" / "seq_des" / "atom_mpnn_inference.yaml"
    seq_des_cfg = OmegaConf.create(
        {
            "model_name": "atom_mpnn",
            "atom_mpnn": {
                "ckpt_name_or_path": args.checkpoint,
                "sampling_cfg": str(sampling_cfg_path),
                "overrides": {
                    "num_seqs_per_pdb": args.num_seq_per_target,
                    "omit_aas": list(args.restricted_aas) if args.restricted_aas else None,
                    "potts_sampling_cfg": {
                        "potts_temperature": get_effective_temp(args),
                    },
                    "num_workers": args.num_workers,
                    "verbose": bool(args.verbose),
                },
            },
        }
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_des_model = get_seq_des_model(seq_des_cfg, device=device)

    per_name_records: dict[str, list[dict[str, object]]] = defaultdict(list)
    seq_indices: dict[str, int] = defaultdict(int)

    with tempfile.TemporaryDirectory(prefix="inverse_fold_caliby_") as temp_dir:
        if single_pdb_map:
            outputs = run_seq_des(
                model=seq_des_model["model"],
                data_cfg=seq_des_model["data_cfg"],
                sampling_cfg=seq_des_model["sampling_cfg"],
                pdb_paths=[str(path) for path in single_pdb_map.values()],
                device=device,
                pos_constraint_df=pos_constraint_df[pos_constraint_df["pdb_key"].isin(single_pdb_map)],
                out_dir=temp_dir,
            )

            for name, seq_string, score in zip(outputs["example_id"], outputs["seq"], outputs["U"], strict=True):
                all_chain_ids = list(chain_sequences[name].keys())
                chain_to_sequence = dict(zip(all_chain_ids, seq_string.split(":"), strict=True))
                seq_indices[name] += 1
                per_name_records[name].append(
                    {
                        "seq_idx": seq_indices[name],
                        "chains": {chain: chain_to_sequence.get(chain, "") for chain in chain_columns},
                        "score": score,
                    }
                )

        if ensemble_dir_map:
            pdb_to_conformers = process_conformer_dirs(
                [str(path) for path in ensemble_dir_map.values()],
                max_num_conformers=None,
                include_primary_conformer=True,
            )
            ensemble_constraint_df = get_ensemble_constraint_df(
                pos_constraint_df[pos_constraint_df["pdb_key"].isin(ensemble_dir_map)],
                pdb_to_conformers,
            )
            outputs = run_seq_des_ensemble(
                model=seq_des_model["model"],
                data_cfg=seq_des_model["data_cfg"],
                sampling_cfg=seq_des_model["sampling_cfg"],
                pdb_to_conformers=pdb_to_conformers,
                device=device,
                pos_constraint_df=ensemble_constraint_df,
                out_dir=temp_dir,
            )

            for name, seq_string, score in zip(outputs["example_id"], outputs["seq"], outputs["U"], strict=True):
                if name not in ensemble_dir_map:
                    continue
                all_chain_ids = list(chain_sequences[name].keys())
                chain_to_sequence = dict(zip(all_chain_ids, seq_string.split(":"), strict=True))
                seq_indices[name] += 1
                per_name_records[name].append(
                    {
                        "seq_idx": seq_indices[name],
                        "chains": {chain: chain_to_sequence.get(chain, "") for chain in chain_columns},
                        "score": score,
                    }
                )

    return build_output_rows(chain_columns, per_name_records)


def resolve_mpnn_checkpoint(checkpoint: str) -> tuple[str, str]:
    checkpoint_path = resolve_cli_path(checkpoint)
    if checkpoint_path.suffix != ".pt":
        raise ValueError(f"ProteinMPNN --checkpoint must point to an exact .pt file, got: {checkpoint}")
    if not checkpoint_path.is_file():
        raise ValueError(f"ProteinMPNN checkpoint file not found: {checkpoint_path}")
    return str(checkpoint_path.parent), checkpoint_path.stem


def parse_mpnn_fasta(path: Path, designed_chains: list[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    sample_re = re.compile(r"sample=(\d+), score=([-+0-9.eE]+)")
    current_header: str | None = None

    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_header = line[1:]
                continue
            if current_header is None or current_header.startswith(path.stem + ","):
                continue

            match = sample_re.search(current_header)
            if match is None:
                raise ValueError(f"Could not parse sample header from {path}: {current_header}")
            seq_idx = int(match.group(1))
            score = float(match.group(2))
            records.append(
                {
                    "seq_idx": seq_idx,
                    "chains": split_sequence_for_chains(line, designed_chains),
                    "score": score,
                }
            )
    return records


def run_proteinmpnn(
    args: argparse.Namespace,
    chain_columns: list[str],
    input_paths: OrderedDict[str, Path],
    fixed_positions: OrderedDict[str, dict[str, list[int]]],
    chain_sequences: dict[str, OrderedDict[str, str]],
) -> list[dict[str, object]]:
    ensure_local_paths()

    import protein_mpnn_run

    weights_dir, model_name = resolve_mpnn_checkpoint(args.checkpoint)

    with tempfile.TemporaryDirectory(prefix="inverse_fold_mpnn_inputs_") as input_dir, tempfile.TemporaryDirectory(
        prefix="inverse_fold_mpnn_outputs_"
    ) as output_dir:
        input_dir_path = Path(input_dir)
        output_dir_path = Path(output_dir)

        chain_id_dict: dict[str, tuple[list[str], list[str]]] = {}
        fixed_positions_dict: dict[str, dict[str, list[int]]] = {}
        designed_chain_orders: dict[str, list[str]] = {}

        for name, pdb_path in input_paths.items():
            staged_path = input_dir_path / pdb_path.name
            try:
                os.symlink(pdb_path, staged_path)
            except OSError:
                shutil.copy2(pdb_path, staged_path)

            available_chains = list(chain_sequences[name].keys())
            designed_chains = sorted(chain for chain in chain_columns if chain in available_chains)
            if not designed_chains:
                raise ValueError(f"{name}.pdb has no chains matching the design columns {chain_columns}")

            fixed_chains = [chain for chain in available_chains if chain not in designed_chains]
            designed_chain_orders[name] = designed_chains
            chain_id_dict[name] = (designed_chains, fixed_chains)
            fixed_positions_dict[name] = {
                chain: fixed_positions[name].get(chain, []) if chain in designed_chains else []
                for chain in available_chains
            }

        chain_id_jsonl = output_dir_path / "chain_id.jsonl"
        fixed_positions_jsonl = output_dir_path / "fixed_positions.jsonl"
        chain_id_jsonl.write_text(json.dumps(chain_id_dict) + "\n")
        fixed_positions_jsonl.write_text(json.dumps(fixed_positions_dict) + "\n")

        sanity_chain_id_jsonl = get_sanity_output_path(args.output_csv, ".proteinmpnn_chain_id.jsonl")
        sanity_fixed_positions_jsonl = get_sanity_output_path(args.output_csv, ".proteinmpnn_fixed_positions.jsonl")
        sanity_summary_csv = get_sanity_output_path(args.output_csv, ".proteinmpnn_inputs.csv")
        sanity_chain_id_jsonl.parent.mkdir(parents=True, exist_ok=True)
        sanity_chain_id_jsonl.write_text(json.dumps(chain_id_dict) + "\n")
        sanity_fixed_positions_jsonl.write_text(json.dumps(fixed_positions_dict) + "\n")
        with sanity_summary_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "name",
                    "designed_chains",
                    "fixed_chains",
                    "fixed_positions_json",
                    "bias_dict",
                    "bias_jsonl",
                    "temp",
                    "num_seq_per_target",
                    "num_workers",
                ],
            )
            writer.writeheader()
            for name in input_paths:
                designed_chains, fixed_chains = chain_id_dict[name]
                writer.writerow(
                    {
                        "name": name,
                        "designed_chains": ",".join(designed_chains),
                        "fixed_chains": ",".join(fixed_chains),
                        "fixed_positions_json": json.dumps(fixed_positions_dict[name], sort_keys=True),
                        "bias_dict": json.dumps(args.bias_dict, sort_keys=True) if args.bias_dict else "",
                        "bias_jsonl": args.bias_jsonl,
                        "temp": get_effective_temp(args),
                        "num_seq_per_target": args.num_seq_per_target,
                        "num_workers": args.num_workers,
                    }
                )

        mpnn_args = SimpleNamespace(
            suppress_print=0 if args.verbose else 1,
            folder_with_pdbs_path=str(input_dir_path),
            chain_list="",
            position_list=False,
            ca_only=False,
            path_to_model_weights=weights_dir,
            model_name=model_name,
            use_soluble_model=False,
            seed=args.seed,
            save_score=0,
            save_probs=0,
            score_only=0,
            path_to_fasta="",
            conditional_probs_only=0,
            conditional_probs_only_backbone=0,
            unconditional_probs_only=0,
            backbone_noise=0.0,
            num_seq_per_target=args.num_seq_per_target,
            batch_size=args.num_workers,
            max_length=200000,
            sampling_temp=str(get_effective_temp(args)),
            out_folder=str(output_dir_path),
            pdb_path="",
            pdb_path_chains="",
            jsonl_path=None,
            chain_id_jsonl=str(chain_id_jsonl),
            fixed_positions_jsonl=str(fixed_positions_jsonl),
            omit_AAs=get_mpnn_omit_aas(args.restricted_aas),
            bias_AA_jsonl=args.bias_jsonl or "",
            bias_by_res_jsonl="",
            omit_AA_jsonl="",
            pssm_jsonl="",
            pssm_multi=0.0,
            pssm_threshold=0.0,
            pssm_log_odds_flag=0,
            pssm_bias_flag=0,
            tied_positions_jsonl="",
        )
        protein_mpnn_run.main(mpnn_args)

        per_name_records: dict[str, list[dict[str, object]]] = {}
        for name in input_paths:
            fasta_path = output_dir_path / "seqs" / f"{name}.fa"
            if not fasta_path.is_file():
                raise ValueError(f"ProteinMPNN did not produce {fasta_path}")
            per_name_records[name] = parse_mpnn_fasta(fasta_path, designed_chain_orders[name])

    return build_output_rows(chain_columns, per_name_records)


def main() -> None:
    args = parse_args()
    args.restricted_aas = parse_restricted_aas(args.restricted_aas)
    args.pdb_dir = str(resolve_cli_path(args.pdb_dir))
    args.fixed_positions_csv = str(resolve_cli_path(args.fixed_positions_csv))
    args.output_csv = str(resolve_cli_path(args.output_csv))
    args.checkpoint = str(resolve_cli_path(args.checkpoint))
    args.bias_dict = parse_bias_dict(args.bias_jsonl)
    args.bias_jsonl = str(write_bias_jsonl(Path(args.pdb_dir), args.bias_dict)) if args.bias_dict else ""

    chain_columns, input_paths, input_kinds, fixed_positions, chain_sequences = load_inputs(
        Path(args.pdb_dir), Path(args.fixed_positions_csv), args.model_type
    )

    if args.model_type == "caliby":
        rows = run_caliby(args, chain_columns, input_paths, input_kinds, fixed_positions, chain_sequences)
    else:
        rows = run_proteinmpnn(args, chain_columns, input_paths, fixed_positions, chain_sequences)

    write_output_csv(Path(args.output_csv), chain_columns, rows)


if __name__ == "__main__":
    main()
