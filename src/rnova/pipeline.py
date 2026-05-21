from __future__ import annotations

import csv
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .errors import RNovAError
from .validation import (
    PATHSEARCHER_CHECKPOINT,
    PATHSEARCHER_DIR,
    REQUIRED_VENDOR_FILES,
    REPO_ROOT,
    SEQFILLER_CHECKPOINT,
    SEQFILLER_DIR,
    assert_checks_pass,
    check_directory_writable,
    collect_checks,
    find_mgf_files,
    seqfiller_extension_files,
)


BASE_CANDIDATE_AMINO_ACIDS = "A;C|UniMod:4;D;E;F;G;H;K;L;M;N;P;Q;R;S;T;V;W;Y"


@dataclass
class PlannedCommand:
    step: str
    args: list[str]
    cwd: Path

    def display(self) -> str:
        return f"(cd {self.cwd} && {shlex.join(self.args)})"


def expected_decoy_path(mgf_file: Path, decoy_dir: Path) -> Path:
    return decoy_dir / f"{mgf_file.name}.decoy_random0.40.mgf"


def build_commands(
    input_dir: str | Path,
    *,
    use_unimod: bool,
    refresh_unimod: bool = False,
    top_k_ptms: int,
    path_batch_size: int | None = None,
    path_num_workers: int | None = None,
    seq_batch_size: int | None = None,
    seq_num_workers: int | None = None,
    progress_interval: int | None = None,
    topk_annotation: str = "<topk_ptm_annotation from workflow>",
) -> list[PlannedCommand]:
    input_path = Path(input_dir).expanduser().resolve()
    decoy_dir = input_path / "decoy_mgf"
    mgf_files = find_mgf_files(input_path)
    decoy_mgfs = [expected_decoy_path(mgf, decoy_dir) for mgf in mgf_files]
    workflow_args = [str(REPO_ROOT / "workflow.py"), str(input_path / "*_rnova_denovo_path_001fdr.csv")]
    if use_unimod:
        workflow_args.append("--use-unimod")
    if refresh_unimod:
        workflow_args.append("--refresh-unimod")
    workflow_args.extend(["--topk-ptm", str(top_k_ptms)])

    candidate_amino_acids = BASE_CANDIDATE_AMINO_ACIDS
    if topk_annotation:
        candidate_amino_acids = f"{candidate_amino_acids};{topk_annotation}"

    pathsearcher_args = [sys.executable, "Inference_Node.py"]
    if path_batch_size is not None:
        pathsearcher_args.extend(["--batch-size", str(path_batch_size)])
    if path_num_workers is not None:
        pathsearcher_args.extend(["--num-workers", str(path_num_workers)])
    if progress_interval is not None:
        pathsearcher_args.extend(["--progress-interval", str(progress_interval)])
    pathsearcher_args.extend([*map(str, mgf_files), *map(str, decoy_mgfs)])

    seqfiller_args = [sys.executable, "Inference_Sequence.py"]
    if seq_batch_size is not None:
        seqfiller_args.extend(["--batch-size", str(seq_batch_size)])
    if seq_num_workers is not None:
        seqfiller_args.extend(["--num-workers", str(seq_num_workers)])
    seqfiller_args.extend([*map(str, mgf_files), *map(str, decoy_mgfs), candidate_amino_acids])

    return [
        PlannedCommand(
            "1/6 Generate decoy MGF files",
            [sys.executable, str(REPO_ROOT / "decoy_spectrum_generator.py"), str(input_path), str(decoy_dir)],
            REPO_ROOT,
        ),
        PlannedCommand(
            "2/6 Run PathSearcher inference",
            pathsearcher_args,
            PATHSEARCHER_DIR,
        ),
        PlannedCommand(
            "3/6 Run FDR stage 1",
            [sys.executable, str(REPO_ROOT / "FDR_stage1.py"), str(input_path), str(decoy_dir)],
            REPO_ROOT,
        ),
        PlannedCommand(
            "4/6 Run clustering/alignment workflow",
            [sys.executable, *workflow_args],
            REPO_ROOT,
        ),
        PlannedCommand(
            "5/6 Run SeqFiller inference",
            seqfiller_args,
            SEQFILLER_DIR,
        ),
        PlannedCommand(
            "6/6 Run FDR stage 2",
            [sys.executable, str(REPO_ROOT / "FDR-stage2.py"), str(input_path), str(decoy_dir)],
            REPO_ROOT,
        ),
    ]


def run_pipeline(
    input_dir: str | Path,
    *,
    use_unimod: bool,
    top_k_ptms: int,
    refresh_unimod: bool = False,
    path_batch_size: int | None = None,
    path_num_workers: int | None = None,
    seq_batch_size: int | None = None,
    seq_num_workers: int | None = None,
    progress_interval: int | None = None,
    dry_run: bool = False,
) -> None:
    input_path = Path(input_dir).expanduser().resolve()
    if top_k_ptms <= 0:
        raise RNovAError("--top-k-ptms must be greater than 0")
    _validate_positive_optional("--path-batch-size", path_batch_size)
    _validate_nonnegative_optional("--path-num-workers", path_num_workers)
    _validate_positive_optional("--seq-batch-size", seq_batch_size)
    _validate_nonnegative_optional("--seq-num-workers", seq_num_workers)
    _validate_positive_optional("--progress-interval", progress_interval)
    mgf_files = _validate_input(input_path)
    _validate_vendor_files()

    if dry_run:
        assert_checks_pass(collect_checks(input_path, mode="dry-run"))
        _print_dry_run(
            input_path,
            use_unimod=use_unimod,
            refresh_unimod=refresh_unimod,
            top_k_ptms=top_k_ptms,
            path_batch_size=path_batch_size,
            path_num_workers=path_num_workers,
            seq_batch_size=seq_batch_size,
            seq_num_workers=seq_num_workers,
            progress_interval=progress_interval,
        )
        return

    _validate_runtime(input_path)

    commands = build_commands(
        input_path,
        use_unimod=use_unimod,
        refresh_unimod=refresh_unimod,
        top_k_ptms=top_k_ptms,
        path_batch_size=path_batch_size,
        path_num_workers=path_num_workers,
        seq_batch_size=seq_batch_size,
        seq_num_workers=seq_num_workers,
        progress_interval=progress_interval,
    )

    _run(commands[0])
    decoy_mgfs = [expected_decoy_path(mgf, input_path / "decoy_mgf") for mgf in mgf_files]
    _require_files("decoy generation", decoy_mgfs)

    _run(commands[1])
    _validate_pathsearcher_outputs(mgf_files, decoy_mgfs)
    _run(commands[2])
    _validate_fdr_stage1_outputs(mgf_files)
    workflow_output = _run(commands[3], capture_stdout=True)
    _validate_workflow_outputs(input_path)
    topk_annotation = _parse_topk_annotation(workflow_output)
    seq_command = build_commands(
        input_path,
        use_unimod=use_unimod,
        refresh_unimod=refresh_unimod,
        top_k_ptms=top_k_ptms,
        path_batch_size=path_batch_size,
        path_num_workers=path_num_workers,
        seq_batch_size=seq_batch_size,
        seq_num_workers=seq_num_workers,
        progress_interval=progress_interval,
        topk_annotation=topk_annotation,
    )[4]
    _run(seq_command)
    _validate_seqfiller_outputs(mgf_files, decoy_mgfs)
    _run(commands[5])
    _validate_fdr_stage2_outputs(mgf_files)


def setup_seqfiller_extension() -> None:
    _validate_vendor_files()
    command = PlannedCommand(
        "Build SeqFiller extension",
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        SEQFILLER_DIR,
    )
    _run(command)


def _validate_input(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise RNovAError(f"Input directory does not exist: {input_path}")
    if not input_path.is_dir():
        raise RNovAError(f"Input path is not a directory: {input_path}")
    if not check_directory_writable(input_path):
        raise RNovAError(f"Input/output directory is not writable: {input_path}")
    mgf_files = find_mgf_files(input_path)
    if not mgf_files:
        raise RNovAError(f"No .mgf files found in {input_path}")
    return mgf_files


def _validate_positive_optional(name: str, value: int | None) -> None:
    if value is not None and value <= 0:
        raise RNovAError(f"{name} must be greater than 0")


def _validate_nonnegative_optional(name: str, value: int | None) -> None:
    if value is not None and value < 0:
        raise RNovAError(f"{name} must be greater than or equal to 0")


def _validate_vendor_files() -> None:
    missing = [str(path) for path in REQUIRED_VENDOR_FILES if not path.exists()]
    if missing:
        raise RNovAError("Vendored RNovA files are missing:\n" + "\n".join(missing))


def _validate_runtime(input_path: Path) -> None:
    assert_checks_pass(collect_checks(input_path, mode="inference", strict_gpu=True))

    missing = []
    for checkpoint in (PATHSEARCHER_CHECKPOINT, SEQFILLER_CHECKPOINT):
        if not checkpoint.exists():
            missing.append(str(checkpoint))
    if missing:
        raise RNovAError(
            "Checkpoint files are missing. Run `uv run rnova download-checkpoints`.\n"
            + "\n".join(missing)
        )
    if not seqfiller_extension_files():
        raise RNovAError("SeqFiller extension is missing or incompatible. Run `uv run rnova setup`.")


def _path_output(mgf_file: Path) -> Path:
    return mgf_file.parent / f"{mgf_file.stem}_rnova_denovo_path.csv"


def _path_fdr_output(mgf_file: Path) -> Path:
    return mgf_file.parent / f"{mgf_file.stem}_rnova_denovo_path_001fdr.csv"


def _seq_output(mgf_file: Path) -> Path:
    return mgf_file.parent / f"{mgf_file.stem}_rnova_denovo_seq.csv"


def _seq_fdr_output(mgf_file: Path) -> Path:
    return mgf_file.parent / f"{mgf_file.stem}_rnova_denovo_seq001fdr.csv"


def _validate_pathsearcher_outputs(mgf_files: list[Path], decoy_mgfs: list[Path]) -> None:
    outputs = [_path_output(path) for path in [*mgf_files, *decoy_mgfs]]
    _require_csv_outputs(
        "PathSearcher inference",
        outputs,
        required_columns=("scan", "node_mass", "score", "node_class"),
        require_rows=True,
    )


def _validate_fdr_stage1_outputs(mgf_files: list[Path]) -> None:
    _require_csv_outputs(
        "FDR stage 1",
        [_path_fdr_output(path) for path in mgf_files],
        required_columns=("scan", "node_mass", "node_score"),
        require_rows=False,
    )


def _validate_workflow_outputs(input_path: Path) -> None:
    _require_csv_outputs(
        "clustering/alignment workflow",
        [input_path / "filled_peptides.csv"],
        required_columns=("raw_sequence", "filled_sequence", "cluster_center"),
        require_rows=False,
    )


def _validate_seqfiller_outputs(mgf_files: list[Path], decoy_mgfs: list[Path]) -> None:
    outputs = [_seq_output(path) for path in [*mgf_files, *decoy_mgfs]]
    _require_csv_outputs(
        "SeqFiller inference",
        outputs,
        required_columns=("title", "sequence", "score"),
        require_rows=True,
    )


def _validate_fdr_stage2_outputs(mgf_files: list[Path]) -> None:
    _require_csv_outputs(
        "FDR stage 2",
        [_seq_fdr_output(path) for path in mgf_files],
        required_columns=("title", "sequence", "score"),
        require_rows=False,
    )


def _require_files(stage: str, paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RNovAError(f"{stage} did not create expected files:\n" + "\n".join(missing))


def _require_csv_outputs(
    stage: str,
    paths: list[Path],
    *,
    required_columns: tuple[str, ...],
    require_rows: bool,
) -> None:
    _require_files(stage, paths)
    failures: list[str] = []
    for path in paths:
        try:
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                header = reader.fieldnames or []
                missing_columns = [column for column in required_columns if column not in header]
                if missing_columns:
                    failures.append(f"{path}: missing columns {', '.join(missing_columns)}")
                    continue
                if require_rows and not any(True for _ in reader):
                    failures.append(f"{path}: no data rows")
        except OSError as exc:
            failures.append(f"{path}: could not read CSV ({exc})")
    if failures:
        raise RNovAError(f"{stage} produced invalid intermediate files:\n" + "\n".join(failures))


def _print_dry_run(
    input_path: Path,
    *,
    use_unimod: bool,
    refresh_unimod: bool,
    top_k_ptms: int,
    path_batch_size: int | None,
    path_num_workers: int | None,
    seq_batch_size: int | None,
    seq_num_workers: int | None,
    progress_interval: int | None,
) -> None:
    print(f"Input directory: {input_path}")
    print(f"Input MGF files found: {len(find_mgf_files(input_path))}")
    print(f"Decoy output directory: {input_path / 'decoy_mgf'}")
    print(f"PathSearcher checkpoint: {PATHSEARCHER_CHECKPOINT}")
    print(f"SeqFiller checkpoint: {SEQFILLER_CHECKPOINT}")
    print()
    for command in build_commands(
        input_path,
        use_unimod=use_unimod,
        refresh_unimod=refresh_unimod,
        top_k_ptms=top_k_ptms,
        path_batch_size=path_batch_size,
        path_num_workers=path_num_workers,
        seq_batch_size=seq_batch_size,
        seq_num_workers=seq_num_workers,
        progress_interval=progress_interval,
    ):
        print(f"{command.step}:")
        print(f"  {command.display()}")


def _run(command: PlannedCommand, *, capture_stdout: bool = False) -> str:
    print(f"{command.step}:")
    print(f"$ {command.display()}")
    if capture_stdout:
        completed = subprocess.run(
            command.args,
            cwd=command.cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
    else:
        completed = subprocess.run(command.args, cwd=command.cwd, check=False)

    if completed.returncode != 0:
        raise RNovAError(f"{command.step} failed with exit code {completed.returncode}")

    return completed.stdout if capture_stdout else ""


def _parse_topk_annotation(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("topk_ptm_annotation: "):
            return line.split(": ", 1)[1].strip()
    raise RNovAError("Workflow did not print `topk_ptm_annotation: ...`")
