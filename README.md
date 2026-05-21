# RNovA Personal Monorepo

This repository is a personal, self-contained RNovA monorepo. It vendors the PathSearcher and SeqFiller inference code as normal directories and uses one root `uv` workflow for setup, checks, checkpoint installation, dry-runs, and real inference.

The recommended entry point is the root `rnova` CLI.

## Quick Start

On any machine, including a non-GPU laptop, the lightweight workflow should work:

```bash
git clone <your-rnova-repo-url>
cd RNovA
uv sync
uv run rnova --help
uv run rnova doctor --mode dry-run ./data
uv run rnova run ./data --use-unimod --top-k-ptms 10 --dry-run
```

For real inference, use Linux/WSL/Ubuntu with an NVIDIA GPU, CUDA-visible PyTorch, and FlashAttention:

```bash
uv sync --extra inference --extra flash
uv run rnova download-checkpoints
uv run rnova setup
uv run rnova doctor --mode inference ./data
uv run rnova run ./data --use-unimod --top-k-ptms 10
```

The legacy shell entry point still works as a wrapper around the CLI:

```bash
./RNovA.sh ./data true 10
```

## WSL/Ubuntu Workstation Setup

1. Install system basics:

```bash
sudo apt update
sudo apt install -y git build-essential python3-dev
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Confirm the GPU is visible before installing the heavy extras:

```bash
nvidia-smi
```

3. Clone this repo and install the inference environment:

```bash
git clone <your-rnova-repo-url>
cd RNovA
uv sync --extra inference --extra flash
```

If FlashAttention fails to build in your CUDA/PyTorch stack, try:

```bash
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation
```

The root `pyproject.toml` also declares uv build overrides for `flash-attn`
so its isolated build environment receives the same `torch` version that the
runtime environment will use, plus `packaging` and `ninja`. That avoids the
common `ModuleNotFoundError: No module named 'torch'` failure during
`uv sync --extra inference --extra flash`.

4. Install checkpoints and build the SeqFiller extension:

```bash
uv run rnova download-checkpoints
uv run rnova setup
uv run rnova doctor --mode inference /path/to/mgf-data
```

5. Run the workflow:

```bash
uv run rnova run /path/to/mgf-data --use-unimod --top-k-ptms 10
```

## Commands

### Check Installation

```bash
uv run rnova doctor --mode dry-run [INPUT_DIR]
uv run rnova doctor --mode workflow [INPUT_DIR]
uv run rnova doctor --mode inference [INPUT_DIR]
```

`dry-run` checks the lightweight CLI/input path. `workflow` adds clustering/post-processing imports. `inference` adds PyTorch/CUDA visibility, FlashAttention, checkpoints, and the SeqFiller compiled extension.

`rnova run` performs `inference` preflight before real work starts. `rnova run --dry-run` only performs the lightweight dry-run checks.

### Download Checkpoints

```bash
uv run rnova download-checkpoints
```

Downloads `RNovA_Checkpoint.zip` from Zenodo record 18352464 with a timeout, size cap, atomic temp file, and mandatory checksum verification. The current Zenodo metadata supplies md5 `e114813f043dbc93b800aa48d4797ada`; SHA-256 can be enabled in the CLI helper if upstream publishes one. The installed files are:

```text
RNovA_PathSearcher_Inference/save/rnova.pt
RNovA_SeqFiller_Inference/save/rnova.pt
```

To use an existing archive:

```bash
uv run rnova download-checkpoints --archive /path/to/RNovA_Checkpoint.zip
```

Only for a trusted local archive with no matching checksum metadata:

```bash
uv run rnova download-checkpoints --archive /path/to/RNovA_Checkpoint.zip --allow-unverified-archive
```

### Build SeqFiller Extension

```bash
uv run rnova setup
```

Runs:

```bash
cd RNovA_SeqFiller_Inference
python setup.py build_ext --inplace
```

If you only want to build this extension without the full inference stack:

```bash
uv sync --extra setup
uv run rnova setup
```

Generated native files such as `knapsack_build.c` and `knapsack_build*.so` are intentionally ignored. Keep `knapsack_build.pyx` in source control and rebuild locally with `rnova setup`.

### Dry Run

```bash
uv run rnova run ./data --use-unimod --top-k-ptms 10 --dry-run
```

Prints the exact six-stage workflow commands without running inference.

### Run Workflow

```bash
uv run rnova run INPUT_DIR --use-unimod --top-k-ptms 10
```

Use `--refresh-unimod` with `--use-unimod` to refresh the cached UniMod XML before annotation.

The workflow runs:

1. Decoy MGF generation
2. PathSearcher inference
3. FDR stage 1
4. Clustering and alignment to choose top-k PTMs
5. SeqFiller inference
6. FDR stage 2

## Input Format

RNovA expects `.mgf` files directly inside the input directory. Scan identifiers are read in this order:

- `SCANS=123`
- `TITLE=... scan=123`
- `TITLE=file.123.123.2`
- spectrum index fallback

MGF parsing is intentionally strict. Malformed `BEGIN/END IONS` blocks, missing precursor/charge data, empty spectra, malformed peak lines, and zero or negative peak intensities fail early with file/spectrum/line context.

## Outputs

Each input `.mgf` produces intermediate and final files next to the input:

```text
<sample>_rnova_denovo_path.csv
<sample>_rnova_denovo_path_001fdr.csv
filled_peptides.csv
filled_peptides_with_PTM.csv
<sample>_rnova_denovo_seq.csv
<sample>_rnova_denovo_seq001fdr.csv
decoy_mgf/<sample>.mgf.decoy_random0.40.mgf
```

These outputs are ignored by git.

## Release Hygiene

This repo should stay source-only:

- no submodules or `.gitmodules`
- no generated Cython C files or platform-specific native extensions
- no checkpoints, downloaded archives, caches, virtualenvs, pyc files, or inference outputs
- no module-local dependency files; root `pyproject.toml` and `uv.lock` are the source of truth

Before pushing a release branch:

```bash
uv --cache-dir .cache/uv lock --check
uv --cache-dir .cache/uv run pytest -q
uv --cache-dir .cache/uv run rnova doctor --mode dry-run ./data
uv --cache-dir .cache/uv run rnova run ./data --dry-run --use-unimod --top-k-ptms 10
```

## Notes

- This repository vendors the former PathSearcher and SeqFiller submodules as normal directories for simpler personal use.
- Default `uv sync` is intentionally lightweight so CLI help, doctor, and dry-runs work on non-GPU laptops. Install `--extra inference --extra flash` only on the Linux/GPU machine where you plan to run the models.
- The root CLI uses absolute paths when invoking module scripts, so `./data` remains valid even when a stage runs from inside a module directory.

## Contents

- `RNovA_PathSearcher_Inference/`: vendored PathSearcher inference code
- `RNovA_SeqFiller_Inference/`: vendored SeqFiller inference code
- `workflow.py`: clustering and alignment to derive top-k PTMs
- `FDR_stage1.py`: node/path-level FDR filtering on PathSearcher outputs
- `FDR-stage2.py`: sequence-level FDR filtering on SeqFiller outputs
- `decoy_spectrum_generator.py`: decoy MGF generator
- `src/rnova/`: root CLI, validation, checkpoint, pipeline, and MGF helpers
