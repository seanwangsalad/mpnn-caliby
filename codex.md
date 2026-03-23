# Codex Log

## Scope Completed

Added a shared environment setup and a unified inverse-folding entry script for the vendored `caliby` and `ProteinMPNN` codebases.

Files added:

- `environment.yml`
- `setup_shared_env.sh`
- `inverse_fold.py`
- local fixes in `ProteinMPNN/protein_mpnn_run.py`

## What Was Changed

### 1. Shared Conda Environment

Created `environment.yml` with a minimal conda base:

- `python=3.12`
- `pip`
- `git`
- `rdkit`

Created `setup_shared_env.sh` to:

- require the user to create and activate a conda env first
- update the currently active conda env from `environment.yml`
- refuse to install into the `base` conda env
- install `caliby` in editable mode with `pip install -e`

Rationale:

- Caliby already constrains the stack more tightly than ProteinMPNN.
- The vendored `ProteinMPNN` code here is plain Python and can be run from the same env.
- This avoids maintaining separate incompatible environments while keeping environment creation under explicit user control.

### 2. Unified CLI: `inverse_fold.py`

Implemented a root-level script that normalizes inputs and outputs across both backends.

Supported inputs:

- `--pdb-dir`
- `--fixed-positions-csv`
- `--model-type {caliby, mpnn, proteinmpnn}`
- `--checkpoint`
- `--restricted-aas`
- `--bias-jsonl`
- `--num_seq_per_target`
- `--output-csv`
- optional controls:
  - `--seed`
  - `--temp`
  - `--num-workers`
  - `--verbose`

Supported output schema:

- `name`
- `seq_idx`
- chain columns from the input CSV, e.g. `A`, `B`, `C`
- `score`

### 3. CSV Parsing Rules

`fixed_positions.csv` is expected to contain:

- a `name` column
- single-letter chain columns such as `A`, `B`, `C`

Name resolution under `--pdb-dir`:

- each CSV `name` is resolved as either `<name>.pdb` or `<name>/`
- if `<name>.pdb` exists, the row is treated as a single-structure input
- if `<name>/` exists and `--model-type caliby` is used, the row is treated as an ensemble directory and routed through Caliby `seq_des_ensemble`
- ProteinMPNN does not accept directory rows; its CSV names must resolve to concrete `.pdb` files

Interpretation:

- the chain columns define which chains should be redesigned
- each populated chain cell contains residues to keep fixed within that redesigned chain
- chains not listed as CSV columns are ignored in final CSV output
- for ProteinMPNN, chains present in the PDB but not listed in CSV columns are treated as fixed/not redesigned

Accepted residue cell formats:

- quoted comma-separated residue ranges such as `"1-10,20-30,40-50"`
- open-ended ranges: `"15-"` means residue 15 through the end of the chain, and `"-15"` means residue 1 through 15
- quoted comma-separated singletons/ranges such as `"1,2,5-8,10"`
- blank / `[]` / `nan` / `none` => no fixed positions for that chain

Important CSV note:

- because chain cells contain commas, those values should be quoted in the CSV so they remain within a single chain column
- both comma-delimited and tab-delimited input tables are accepted
- a UTF-8 BOM on the header is tolerated

### 4. Caliby Integration

For Caliby, the implementation uses:

- `seq_des` for rows whose CSV `name` resolves to `<name>.pdb`
- `seq_des_ensemble` for rows whose CSV `name` resolves to `<name>/`

Implementation notes:

- imports Caliby internals directly instead of shelling out
- constructs a temporary `pos_constraint_df`
- maps CSV fixed positions into Caliby `fixed_pos_seq` strings like `A1-5,B10`
- uses:
  - `caliby.eval.eval_utils.seq_des_utils.get_seq_des_model`
  - `caliby.eval.eval_utils.seq_des_utils.run_seq_des`
  - `caliby.eval.eval_utils.seq_des_utils.run_seq_des_ensemble`
- passes `omit_aas` from `--restricted-aas`
- returns Caliby `U` as `score`
- parses sampled sequences from Caliby’s colon-separated chain output using the primary structure chain order
- writes sanity-check files alongside `--output-csv`:
  - `<stem>.caliby_constraints.csv`
  - `<stem>.caliby_inputs.csv`

Current limitation:

- only sequence design is wired in
- no packing

### 5. ProteinMPNN Integration

For ProteinMPNN, the implementation:

- stages input PDBs into a temporary directory
- generates temporary:
  - `chain_id.jsonl`
  - `fixed_positions.jsonl`
- calls `ProteinMPNN/protein_mpnn_run.py` programmatically via `main(SimpleNamespace(...))`
- parses generated FASTA files under the temporary output directory
- writes persistent sanity-check files alongside `--output-csv`:
  - `<stem>.proteinmpnn_chain_id.jsonl`
  - `<stem>.proteinmpnn_fixed_positions.jsonl`
  - `<stem>.proteinmpnn_inputs.csv`

Design chain behavior:

- redesigned chains are the CSV chain columns that actually exist in the PDB
- all other PDB chains are fixed
- fixed positions are applied only within redesigned chains

Score behavior:

- uses ProteinMPNN sampled `score=` from FASTA headers
- does not use `global_score`

Bias handling:

- `--bias-jsonl` is passed to ProteinMPNN as both `bias_AA_jsonl` and `bias_by_res_jsonl`
- ignored for Caliby

Restricted amino acid handling:

- `--restricted-aas` is interpreted as a simple string of one-letter codes
- for example, `CFLY` means omit `C`, `F`, `L`, and `Y`
- whitespace and commas are ignored and letters are uppercased

Temperature handling:

- `--temp` / `-temp` applies to both backends
- default is `0.01` for Caliby
- default is `0.6` for ProteinMPNN
- for Caliby, it is passed as `potts_sampling_cfg.potts_temperature`
- for ProteinMPNN, it is passed as `sampling_temp`

num_workers handling:

- `--num-workers` applies to both backends
- for Caliby, it is passed as `num_workers`
- for ProteinMPNN, it is passed directly as `batch_size`
- default is `1`

Checkpoint resolution:

- all `--checkpoint` inputs must be absolute paths
- for ProteinMPNN, `--checkpoint` must point to the exact `.pt` file

Important repo-specific note:

- this vendored checkout only includes `ProteinMPNN/model_weights/abmpnn.pt`
- it does not include the usual upstream `vanilla_model_weights/` tree
- the vendored `protein_mpnn_run.py` was locally patched so:
  - `fixed_positions_dict` is initialized safely
  - `chain_id_jsonl` is respected when `chain_list` is empty

## Validation Performed

Executed:

- `python -m py_compile inverse_fold.py`
- `python inverse_fold.py --help`
- `bash -n setup_shared_env.sh`

All passed.

Did not execute full end-to-end inference in-session because:

- dependencies are not installed in the live sandbox env
- model weights may need downloading or local provisioning

## Assumptions Captured

1. The user’s CSV chain columns indicate the chains to redesign.
2. Residues listed under each chain are fixed residues within those redesigned chains.
3. For ProteinMPNN, all chains present in the structure but absent from CSV chain columns should remain fixed.
4. For Caliby ensemble rows, the primary conformer is expected at `<name>/<name>.pdb`.
5. The final output should contain only the chain columns explicitly requested by the CSV, not every chain present in the structure.

## Known Limitations / Risks

1. `setup_shared_env.sh` installs Caliby editable into the currently active conda env, so full dependency resolution depends on external package/network availability when run.
2. Caliby’s dependency stack includes git-based packages and HuggingFace weight download paths that are not exercised by static validation.
3. ProteinMPNN in this repo may be a customized fork; the helper script `make_fixed_positions_dict.py` is not used directly because its CSV format does not match the desired interface.
4. PDB parsing in `inverse_fold.py` is lightweight and assumes standard PDB chain/residue formatting.
5. The script currently only discovers `*.pdb` files, not `.cif`.
6. For ProteinMPNN, `num_seq_per_target` must be divisible by `num_workers`, since `num_workers` is used as `batch_size` in this wrapper.

## Future Extension Points

1. Add Caliby packing mode and packed-structure outputs.
2. Add richer Caliby-side sanity outputs if full sampling config inspection is needed.
3. Add CIF input support.
4. Add explicit tests with tiny fixture PDBs and fixture CSVs.
5. Decide whether `score` for ProteinMPNN should optionally expose `global_score` instead.
6. Add richer chain-selection controls if future inputs need redesigning only a subset of CSV columns per run.
