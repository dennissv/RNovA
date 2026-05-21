# RNovA PathSearcher

**RNovA PathSearcher** is a standalone module in the RNovA framework for high-accuracy *de novo* peptide sequencing. In the open PTM discovery workflow, it serves as the first-stage component for identifying high-confidence backbone paths from tandem MS/MS spectra using a graph-based model with RoPE-enhanced attention.

## Features

- Graph-based spectrum representation
- RoPE-based edge modeling (no manual annotation required)
- Efficient path prediction from noisy or incomplete spectra
- Open PTM compatible

## Installation

This directory is vendored inside the personal RNovA monorepo. Use the root
workflow and root `pyproject.toml`; there is intentionally no module-local
requirements file.

From the repository root on the Linux/GPU host:

```bash
uv sync --extra inference --extra flash
uv run rnova download-checkpoints
uv run rnova doctor --mode inference /path/to/mgf-data
```

## Usage

```bash
python Inference_Node.py input1.mgf input2.mgf ... inputN.mgf
```

For normal use, prefer the root command:

```bash
uv run rnova run /path/to/mgf-data --use-unimod --top-k-ptms 10
```

### Arguments

- **MGF files**: One or more input `.mgf` files containing MS/MS spectra.

No candidate amino acid list is needed at this stage. The model identifies plausible fragment paths directly from spectral graph topology.

## Output

For each input `.mgf` file, the output will be saved in the same directory as the input file.
The output filename will be the original filename with `_rnova_denovo_path.csv` appended.

Each output CSV file contains the following columns:

1. **Scan number**: The scan number from the input MGF file
2. **Path**: The predicted node mass sequence path with possible mass gaps or ambiguous edges
3. **Node score**: A score representing the confidence of the predicted node
4. **Node class**: Fragment ion type used to build this node.

Example:

```csv
23375,0.0000;215.1278;362.1968;548.2750;711.3373;858.4080;971.4921;1127.5934,10.0000;4.7344;4.3711;4.4336;4.6719;4.3398;4.9570;10.0000,2;5;5;4;4;5;5;3
```

## Developer

Zeping Mao

## Citation

If you use this module in your work, please cite:

No publication yet.
