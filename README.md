# RNovA Personal Monorepo

This repository is a personal, self-contained RNovA monorepo. It vendors the PathSearcher and SeqFiller inference code as normal directories and uses one root `uv` workflow for setup, checks, checkpoint installation, dry-runs, and real inference.

The recommended entry point is the root `rnova` CLI.

## Quick Start

On any machine, including a non-GPU laptop, the lightweight workflow should work:

```bash
git clone https://github.com/dennissv/RNovA.git
cd RNovA
uv sync
uv run rnova --help
uv run rnova doctor --mode dry-run ./data
uv run rnova run ./data --use-unimod --top-k-ptms 10 --dry-run
```

For real inference, use Linux/WSL/Ubuntu with an NVIDIA GPU, CUDA-visible PyTorch, and FlashAttention:

```bash
uv sync --extra inference --extra flash --locked
uv run rnova download-checkpoints
uv run rnova setup
uv run rnova doctor --mode inference ./data
uv run rnova run ./data --use-unimod --top-k-ptms 10
```

`rnova run` now defaults to the fast desktop-GPU profile, quiet logs, automatic batch sizes, and progress bars when running interactively.

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
git clone https://github.com/dennissv/RNovA.git
cd RNovA
uv sync --extra inference --extra flash --locked
```

Important: plain `uv sync` intentionally creates the lightweight laptop/dev
environment. On a GPU workstation, running plain `uv sync` after installing the
inference extras will prune optional packages such as `torch`, `triton`, and
`flash-attn`. Use `uv sync --extra inference --extra flash --locked` whenever
you want the real inference environment. If you need to sync without pruning
already-installed optional packages, uv also supports `uv sync --inexact`, but
the explicit extras command is the recommended workstation workflow.

If FlashAttention fails to build in your CUDA/PyTorch stack, try:

```bash
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation
```

The root `pyproject.toml` also declares uv build overrides for `flash-attn`
so its isolated build environment receives the same `torch` version that the
runtime environment will use, plus `packaging` and `ninja`. That avoids the
common `ModuleNotFoundError: No module named 'torch'` failure during
`uv sync --extra inference --extra flash --locked`.

4. Install checkpoints and build the SeqFiller extension:

```bash
uv run rnova download-checkpoints
uv run rnova setup
uv run rnova doctor --mode inference /path/to/mgf-data
```

5. Optionally tune the GPU settings on the workstation:

```bash
uv run rnova tune /path/to/mgf-data --sample-spectra 64 --max-memory-frac 0.85
```

The tuning result is cached in `.cache/rnova/tuning.json`. Future `rnova run`
commands use it unless you pass explicit batch/worker options. Use `--retune`
on `rnova run` to refresh the cache before starting a real workflow run.

6. Run the workflow:

```bash
uv run rnova run /path/to/mgf-data --use-unimod --top-k-ptms 10
```

For 12 GiB cards such as a 4070 Ti, the default automatic settings start at
PathSearcher batch 8, SeqFiller batch 8, and WSL worker count 0. You can still
pin the values manually:

```bash
uv run rnova run /path/to/mgf-data \
  --use-unimod \
  --top-k-ptms 10 \
  --path-batch-size 8 \
  --seq-batch-size 8 \
  --path-num-workers 0 \
  --seq-num-workers 0
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

Prints the resolved runtime settings and exact six-stage workflow commands without running inference.

### Run Workflow

```bash
uv run rnova run INPUT_DIR --use-unimod --top-k-ptms 10
```

Use `--refresh-unimod` with `--use-unimod` to refresh the cached UniMod XML before annotation.

Useful GPU/runtime controls:

- `--speed-profile fast|exact|max`: `fast` is the default; `exact` disables TF32; `max` also tries `torch.compile`.
- `--progress auto|on|off`: controls progress bars; `auto` shows them only on interactive terminals.
- `--log-level warning|info|debug`: quiet by default.
- `--debug-inference`: enables detailed encoder/decoder/cache diagnostics.
- `--path-batch-size auto|N` and `--seq-batch-size auto|N`: automatic or manual inference batches.
- `--path-num-workers auto|N` and `--seq-num-workers auto|N`: automatic or manual DataLoader workers.
- `--path-cache-policy auto|legacy`: `auto` uses a smaller bounded decoder cache; `legacy` restores the old oversized allocation.
- `--null-workers auto|N`: controls workflow null-distribution sampling workers.
- `--refresh-workflow-cache`: rebuilds cached workflow null distributions.
- `--retune`: tune GPU settings before running.

Automatic batch sizes are based on visible GPU memory:

```text
<10 GiB   PathSearcher 4,  SeqFiller 4
10-15 GiB PathSearcher 8,  SeqFiller 8
16-23 GiB PathSearcher 16, SeqFiller 16
>=24 GiB  PathSearcher 32, SeqFiller 32
```

WSL defaults to `0` DataLoader workers. Native Linux defaults to PathSearcher
`2` and SeqFiller `1`.

### Tune GPU Settings

```bash
uv run rnova tune INPUT_DIR --sample-spectra 64 --max-memory-frac 0.85
```

`rnova tune` samples spectra into `.cache/rnova/tune`, benchmarks candidate
batch sizes on PathSearcher and SeqFiller, rejects settings that exceed the
requested peak memory fraction, and stores the selected settings in
`.cache/rnova/tuning.json`. It does not change model weights, workflow stages,
FDR, clustering, PTM logic, or output filenames.

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

## Troubleshooting GPU Stalls

PathSearcher and SeqFiller run with `model.eval()` and `torch.inference_mode()` during inference. If stage 2 appears stuck with high GPU memory/utilization, it is usually one of:

- first-run CUDA/Triton/FlashAttention kernel compilation
- a batch that is too large for the GPU
- WSL `nvidia-smi` process accounting lagging or not showing Linux-side Python clearly

Try a smaller batch first:

```bash
uv run rnova run /path/to/mgf-data \
  --use-unimod \
  --top-k-ptms 10 \
  --path-batch-size 4 \
  --seq-batch-size 4 \
  --path-num-workers 0 \
  --seq-num-workers 0
```

For the old detailed breadcrumbs, add:

```bash
--debug-inference --log-level debug --progress-interval 1
```

If the smaller batch works, raise the batch sizes gradually or run `rnova tune`.

## Contents

- `RNovA_PathSearcher_Inference/`: vendored PathSearcher inference code
- `RNovA_SeqFiller_Inference/`: vendored SeqFiller inference code
- `workflow.py`: clustering and alignment to derive top-k PTMs
- `FDR_stage1.py`: node/path-level FDR filtering on PathSearcher outputs
- `FDR-stage2.py`: sequence-level FDR filtering on SeqFiller outputs
- `decoy_spectrum_generator.py`: decoy MGF generator
- `src/rnova/`: root CLI, validation, checkpoint, pipeline, and MGF helpers
