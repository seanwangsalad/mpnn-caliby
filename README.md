# mpnn-caliby

Unified wrapper for running inverse folding with either vendored `caliby` or vendored `ProteinMPNN`.

The main entry point is:

- `inverse_fold.py`
- `pack.py`
- `fast_fixed_pos_csv.py`

## Install

Create and activate a conda environment first. Do not use the base environment.

```bash
cd mpnn-caliby
conda create -n mpnn-caliby python=3.12
conda activate mpnn-caliby
bash setup_shared_env.sh
```

What `setup_shared_env.sh` does:

- updates the active conda env from `environment.yml`
- installs `caliby` in editable mode

Quick sanity check:

```bash
python -c "import numpy, torch, caliby; print(numpy.__version__, torch.__version__, caliby.__file__)"
```

## Run

Show CLI help:

```bash
python inverse_fold.py --help
```

```bash
python pack.py --help
```

```bash
python fast_fixed_pos_csv.py --help
```

## Fast Fixed CSV

Generate a `fixed_positions.csv` quickly from a directory of PDBs:

```bash
python fast_fixed_pos_csv.py \
  --pdb-dir /path/to/pdbs \
  --output-csv fixed_positions.csv \
  --fixed A,B
```

Behavior:

- scans all `.pdb` files in `--pdb-dir`
- writes one CSV row per PDB, using the PDB basename as `name`
- includes all chains observed across the directory as CSV columns
- for chains listed in `--fixed`, writes `1-` so that chain is fully fixed
- leaves other existing chains blank so they are fully redesignable

### ProteinMPNN

```bash
python inverse_fold.py \
  --pdb-dir /path/to/inputs \
  --fixed-positions-csv /path/to/fixed.csv \
  --model-type proteinmpnn \
  --checkpoint /path/to/model.pt \
  --num_seq_per_target 2 \
  --output-csv output.csv
```

Optional ProteinMPNN flags:

- `--restricted-aas CYF`
- `--bias-jsonl '{"W": 1.5, "F": 1.5, "Y": 1.0, "L": 1.0, "I": 1.0}'`
- `--temp 0.6`
- `--num-workers 1`
- `--seed 1`
- `--verbose`

### Caliby

```bash
python inverse_fold.py \
  --pdb-dir /path/to/inputs \
  --fixed-positions-csv /path/to/fixed.csv \
  --model-type caliby \
  --checkpoint /path/to/caliby.ckpt \
  --num_seq_per_target 2 \
  --output-csv output.csv
```

Optional Caliby flags:

- `--restricted-aas CYF`
- `--temp 0.01`
- `--num-workers 1`
- `--seed 1`
- `--verbose`

## Packing

### Pack Existing PDBs

```bash
python pack.py \
  --checkpoint /path/to/caliby_packer.ckpt \
  --pdb-dir /path/to/pdbs \
  --output-dir /path/to/packed_output
```

### Pack Designed Sequences From `output.csv`

If you first ran `inverse_fold.py` and want to graft the designed sequences back onto backbone PDBs before sidechain packing:

```bash
python pack.py \
  --checkpoint /path/to/caliby_packer.ckpt \
  --pdb-dir /path/to/backbone_pdbs \
  --designed-csv /path/to/output.csv \
  --output-dir /path/to/packed_output
```

Behavior in grafting mode:

- reads `name`, `seq_idx`, and chain columns from the design CSV
- loads backbone PDBs from `<pdb-dir>/<name>.pdb`
- rewrites residue names on designed chains using the designed sequences
- writes replicated grafted PDBs to `<output-dir>/grafted_inputs/` as `<name><seq_idx>.pdb`
- runs Caliby sidechain packing on those grafted PDBs

## Input Layout

`--pdb-dir` is the root input directory.

Each row in `fixed_positions.csv` is resolved by `name` as either:

- `<name>.pdb`
- `<name>/`

Rules:

- if `<name>.pdb` exists, the row is treated as a single structure
- if `<name>/` exists and `--model-type caliby` is used, the row is treated as a Caliby ensemble directory
- ProteinMPNN only supports the `.pdb` case

For Caliby ensemble rows, the primary conformer is expected at:

- `<name>/<name>.pdb`

## Fixed Positions CSV

The file must contain:

- a `name` column
- one-letter chain columns such as `A`, `B`, `C`

Example:

```csv
name,A,B
example1,"1-10,20-30,40-50","-15"
example2,"15-",""
```

Meaning:

- chain columns define which chains should be redesigned
- residues listed in each chain cell are fixed within that redesigned chain
- PDB chains not listed as CSV columns remain fixed for ProteinMPNN

Accepted residue syntax:

- `"1-10,20-30,40-50"`
- `"15-"` meaning residue 15 through the end of the chain
- `"-15"` meaning residue 1 through 15
- `"1,2,5-8,10"`
- blank, `[]`, `nan`, `none` meaning no fixed positions

Notes:

- if chain cells contain commas, quote them in the CSV
- both comma-delimited and tab-delimited tables are accepted

## Restricted Amino Acids

`--restricted-aas` is a simple string of one-letter amino acid codes.

Example:

```bash
--restricted-aas CYF
```

This omits:

- `C`
- `Y`
- `F`

Whitespace and commas are ignored, and letters are uppercased.

## Bias JSONL

`--bias-jsonl` is ProteinMPNN-only.

Pass an inline dictionary string, for example:

```bash
--bias-jsonl '{"W": 1.5, "F": 1.5, "Y": 1.0, "L": 1.0, "I": 1.0}'
```

The wrapper writes that dictionary to `--pdb-dir/proteinmpnn_bias.jsonl` and passes the generated file to ProteinMPNN as both:

- `bias_AA_jsonl`
- `bias_by_res_jsonl`

Caliby rejects this option.

## Temperature

`--temp` / `-temp` applies to both backends.

Defaults:

- Caliby: `0.01`
- ProteinMPNN: `0.6`

Behavior:

- for Caliby, this is passed as `potts_sampling_cfg.potts_temperature`
- for ProteinMPNN, this is passed as `sampling_temp`

## num_workers

`--num-workers` applies to both backends, but with different meanings.

- for Caliby, it is passed as `num_workers`
- for ProteinMPNN, it is passed as `batch_size`

Default:

- `1`

## Outputs

Main output:

- `output.csv`

Schema:

- `name`
- `seq_idx`
- requested chain columns such as `A`, `B`, `C`
- `score`

### Caliby sanity-check files

Written next to `--output-csv`:

- `<stem>.caliby_constraints.csv`
- `<stem>.caliby_inputs.csv`

`<stem>.caliby_inputs.csv` includes the parsed fixed-position constraints plus `restricted_aas`.

### ProteinMPNN sanity-check files

Written next to `--output-csv`:

- `<stem>.proteinmpnn_chain_id.jsonl`
- `<stem>.proteinmpnn_fixed_positions.jsonl`
- `<stem>.proteinmpnn_inputs.csv`

These let you inspect the converted backend inputs directly.

## Notes

- for ProteinMPNN, `--num_seq_per_target` must be divisible by `--num-workers`
- `--checkpoint` for ProteinMPNN must be an exact `.pt` file path
- relative CLI paths are resolved from the current working directory
- ProteinMPNN FASTA outputs are generated in a temporary directory and deleted after parsing
