# mpnn-caliby

Unified wrapper for running inverse folding with either vendored `caliby` or vendored `ProteinMPNN`.

The main entry point is:

- `inverse_fold.py`

## Install

Create and activate a conda environment first. Do not use the base environment.

```bash
cd /home/seanwang/mpnn-caliby
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

### ProteinMPNN

```bash
python inverse_fold.py \
  --pdb-dir /path/to/inputs \
  --fixed-positions-csv /path/to/fixed.csv \
  --model-type proteinmpnn \
  --checkpoint /path/to/model.pt \
  --num-repeats 2 \
  --output-csv output.csv
```

Optional ProteinMPNN flags:

- `--restricted-aas CYF`
- `--bias-jsonl /path/to/bias.jsonl`
- `--sampling-temp 0.1`
- `--seed 1`
- `--verbose`

### Caliby

```bash
python inverse_fold.py \
  --pdb-dir /path/to/inputs \
  --fixed-positions-csv /path/to/fixed.csv \
  --model-type caliby \
  --checkpoint /path/to/caliby.ckpt \
  --num-repeats 2 \
  --output-csv output.csv
```

Optional Caliby flags:

- `--restricted-aas CYF`
- `--num-workers 0`
- `--seed 1`
- `--verbose`

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

It is passed through to ProteinMPNN as both:

- `bias_AA_jsonl`
- `bias_by_res_jsonl`

Caliby rejects this option.

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

- ProteinMPNN is always run with internal `batch_size=1`
- `--checkpoint` for ProteinMPNN must be an exact `.pt` file path
- relative paths for CLI file arguments are resolved from the repo root
- ProteinMPNN FASTA outputs are generated in a temporary directory and deleted after parsing
