from __future__ import annotations

import csv
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .errors import RNovAError
from .speed import RuntimeSettings, resolve_runtime_settings
from .tuning import run_tuning
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
STAGE_NAMES = (
    "1/6 Generate decoy MGF files",
    "2/6 Run PathSearcher inference",
    "3/6 Run FDR stage 1",
    "4/6 Run clustering/alignment workflow",
    "5/6 Run SeqFiller inference",
    "6/6 Run FDR stage 2",
)


@dataclass
class PlannedCommand:
    step: str
    args: list[str]
    cwd: Path

    def display(self) -> str:
        return f"(cd {self.cwd} && {shlex.join(self.args)})"


@dataclass
class ResumePlan:
    start_index: int | None
    completed_steps: list[str]
    reason: str
    cleanup_paths: list[Path]


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
    speed_profile: str = "fast",
    progress: str = "auto",
    log_level: str = "warning",
    debug_inference: bool = False,
    path_cache_policy: str = "auto",
    null_workers: int = 1,
    refresh_workflow_cache: bool = False,
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
    workflow_args.extend(["--progress", progress, "--log-level", log_level])
    workflow_args.extend(["--null-workers", str(null_workers)])
    if refresh_workflow_cache:
        workflow_args.append("--refresh-workflow-cache")

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
    pathsearcher_args.extend(["--speed-profile", speed_profile])
    pathsearcher_args.extend(["--progress", progress, "--log-level", log_level])
    pathsearcher_args.extend(["--path-cache-policy", path_cache_policy])
    if debug_inference:
        pathsearcher_args.append("--debug-inference")
    pathsearcher_args.extend([*map(str, mgf_files), *map(str, decoy_mgfs)])

    seqfiller_args = [sys.executable, "Inference_Sequence.py"]
    if seq_batch_size is not None:
        seqfiller_args.extend(["--batch-size", str(seq_batch_size)])
    if seq_num_workers is not None:
        seqfiller_args.extend(["--num-workers", str(seq_num_workers)])
    seqfiller_args.extend(["--speed-profile", speed_profile])
    seqfiller_args.extend(["--progress", progress, "--log-level", log_level])
    if debug_inference:
        seqfiller_args.append("--debug-inference")
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
    path_batch_size: str | int | None = "auto",
    path_num_workers: str | int | None = "auto",
    seq_batch_size: str | int | None = "auto",
    seq_num_workers: str | int | None = "auto",
    progress_interval: int | None = None,
    speed_profile: str = "fast",
    progress: str = "auto",
    log_level: str = "warning",
    debug_inference: bool = False,
    retune: bool = False,
    path_cache_policy: str = "auto",
    null_workers: str | int | None = "auto",
    refresh_workflow_cache: bool = False,
    resume: bool = False,
    assume_yes: bool = False,
    dry_run: bool = False,
) -> None:
    input_path = Path(input_dir).expanduser().resolve()
    if top_k_ptms <= 0:
        raise RNovAError("--top-k-ptms must be greater than 0")
    _validate_positive_optional("--progress-interval", progress_interval)
    mgf_files = _validate_input(input_path)
    _validate_vendor_files()
    settings = _resolve_settings(
        input_path,
        speed_profile=speed_profile,
        progress=progress,
        log_level=log_level,
        path_cache_policy=path_cache_policy,
        path_batch_size=path_batch_size,
        path_num_workers=path_num_workers,
        seq_batch_size=seq_batch_size,
        seq_num_workers=seq_num_workers,
        null_workers=null_workers,
        use_tuning=not retune,
    )

    if dry_run:
        assert_checks_pass(collect_checks(input_path, mode="dry-run"))
        resume_plan = None
        if resume:
            decoy_mgfs = [expected_decoy_path(mgf, input_path / "decoy_mgf") for mgf in mgf_files]
            resume_plan = _prepare_resume_plan(
                input_path,
                mgf_files,
                decoy_mgfs,
                use_unimod=use_unimod,
                top_k_ptms=top_k_ptms,
            )
        _print_dry_run(
            input_path,
            use_unimod=use_unimod,
            refresh_unimod=refresh_unimod,
            top_k_ptms=top_k_ptms,
            settings=settings,
            progress_interval=progress_interval,
            debug_inference=debug_inference,
            refresh_workflow_cache=refresh_workflow_cache,
            resume_plan=resume_plan,
        )
        return

    if retune:
        tuning_path = run_tuning(
            input_path,
            speed_profile=speed_profile,
            path_cache_policy=path_cache_policy,
            progress=progress,
            log_level=log_level,
        )
        print(f"Tuning saved: {tuning_path}")
        settings = _resolve_settings(
            input_path,
            speed_profile=speed_profile,
            progress=progress,
            log_level=log_level,
            path_cache_policy=path_cache_policy,
            path_batch_size=path_batch_size,
            path_num_workers=path_num_workers,
            seq_batch_size=seq_batch_size,
            seq_num_workers=seq_num_workers,
            null_workers=null_workers,
            use_tuning=True,
        )

    _validate_runtime(input_path)

    commands = build_commands(
        input_path,
        use_unimod=use_unimod,
        refresh_unimod=refresh_unimod,
        top_k_ptms=top_k_ptms,
        path_batch_size=settings.path_batch_size,
        path_num_workers=settings.path_num_workers,
        seq_batch_size=settings.seq_batch_size,
        seq_num_workers=settings.seq_num_workers,
        progress_interval=progress_interval,
        speed_profile=settings.speed_profile,
        progress=settings.progress,
        log_level=settings.log_level,
        debug_inference=debug_inference,
        path_cache_policy=settings.path_cache_policy,
        null_workers=settings.null_workers,
        refresh_workflow_cache=refresh_workflow_cache,
    )

    show_command = settings.log_level == "debug"
    decoy_mgfs = [expected_decoy_path(mgf, input_path / "decoy_mgf") for mgf in mgf_files]
    start_index = 0
    if resume:
        resume_plan = _prepare_resume_plan(
            input_path,
            mgf_files,
            decoy_mgfs,
            use_unimod=use_unimod,
            top_k_ptms=top_k_ptms,
        )
        if resume_plan.start_index is None:
            print("Resume detected that all workflow stages already have valid outputs.")
            return
        _confirm_resume(resume_plan, assume_yes=assume_yes)
        _cleanup_resume_outputs(resume_plan.cleanup_paths)
        start_index = resume_plan.start_index

    if start_index <= 0:
        _run(commands[0], show_command=show_command)
        _require_files("decoy generation", decoy_mgfs)

    if start_index <= 1:
        _run(commands[1], show_command=show_command)
        _validate_pathsearcher_outputs(mgf_files, decoy_mgfs)
    if start_index <= 2:
        _run(commands[2], show_command=show_command)
        _validate_fdr_stage1_outputs(mgf_files)
    if start_index <= 3:
        workflow_output = _run(commands[3], capture_stdout=True, show_command=show_command, echo_stdout=show_command)
        _validate_workflow_outputs(input_path)
        topk_annotation = _parse_topk_annotation(workflow_output)
        _write_pipeline_state(
            input_path,
            topk_annotation=topk_annotation,
            use_unimod=use_unimod,
            top_k_ptms=top_k_ptms,
        )
    else:
        topk_annotation = _read_pipeline_state(
            input_path,
            use_unimod=use_unimod,
            top_k_ptms=top_k_ptms,
        )
        if topk_annotation is None and start_index <= 4:
            raise RNovAError(
                "Cannot resume SeqFiller because the saved top-k PTM annotation is missing. "
                "Rerun with `--resume` so stage 4 can be regenerated."
            )
    seq_command = build_commands(
        input_path,
        use_unimod=use_unimod,
        refresh_unimod=refresh_unimod,
        top_k_ptms=top_k_ptms,
        path_batch_size=settings.path_batch_size,
        path_num_workers=settings.path_num_workers,
        seq_batch_size=settings.seq_batch_size,
        seq_num_workers=settings.seq_num_workers,
        progress_interval=progress_interval,
        speed_profile=settings.speed_profile,
        progress=settings.progress,
        log_level=settings.log_level,
        debug_inference=debug_inference,
        path_cache_policy=settings.path_cache_policy,
        null_workers=settings.null_workers,
        refresh_workflow_cache=refresh_workflow_cache,
        topk_annotation=topk_annotation,
    )[4]
    if start_index <= 4:
        _run(seq_command, show_command=show_command)
        _validate_seqfiller_outputs(mgf_files, decoy_mgfs)
    if start_index <= 5:
        _run(commands[5], show_command=show_command)
        _validate_fdr_stage2_outputs(mgf_files)
    print(f"RNovA workflow finished for {len(mgf_files)} input MGF file(s).")


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


def _resolve_settings(input_path: Path, **kwargs) -> RuntimeSettings:
    try:
        return resolve_runtime_settings(input_path, **kwargs)
    except ValueError as exc:
        raise RNovAError(str(exc)) from exc


def _validate_vendor_files() -> None:
    missing = [str(path) for path in REQUIRED_VENDOR_FILES if not path.exists()]
    if missing:
        raise RNovAError("Vendored RNovA files are missing:\n" + "\n".join(missing))


def _prepare_resume_plan(
    input_path: Path,
    mgf_files: list[Path],
    decoy_mgfs: list[Path],
    *,
    use_unimod: bool,
    top_k_ptms: int,
) -> ResumePlan:
    completed_steps: list[str] = []
    start_index: int | None = None
    reason = "all expected outputs are present"
    for index, step in enumerate(STAGE_NAMES):
        ok, detail = _stage_complete(index, input_path, mgf_files, decoy_mgfs)
        if ok:
            completed_steps.append(step)
            continue
        start_index = index
        reason = detail
        break

    if start_index is None:
        return ResumePlan(None, completed_steps, reason, [])

    if start_index == 4 and _read_pipeline_state(
        input_path,
        use_unimod=use_unimod,
        top_k_ptms=top_k_ptms,
    ) is None:
        start_index = 3
        reason = (
            "SeqFiller outputs are missing, and the saved top-k PTM annotation "
            "from stage 4 is not available; stage 4 must be regenerated first"
        )
        completed_steps = completed_steps[:3]
        cleanup_paths = [
            *_stage_output_paths(3, input_path, mgf_files, decoy_mgfs),
            *_stage_output_paths(4, input_path, mgf_files, decoy_mgfs),
        ]
        return ResumePlan(start_index, completed_steps, reason, cleanup_paths)

    cleanup_paths = _stage_output_paths(start_index, input_path, mgf_files, decoy_mgfs)
    return ResumePlan(start_index, completed_steps, reason, cleanup_paths)


def _stage_complete(
    index: int,
    input_path: Path,
    mgf_files: list[Path],
    decoy_mgfs: list[Path],
) -> tuple[bool, str]:
    try:
        if index == 0:
            _require_files("decoy generation", decoy_mgfs)
        elif index == 1:
            _validate_pathsearcher_outputs(mgf_files, decoy_mgfs)
        elif index == 2:
            _validate_fdr_stage1_outputs(mgf_files)
        elif index == 3:
            _validate_workflow_outputs(input_path)
        elif index == 4:
            _validate_seqfiller_outputs(mgf_files, decoy_mgfs)
        elif index == 5:
            _validate_fdr_stage2_outputs(mgf_files)
        else:
            raise ValueError(index)
    except RNovAError as exc:
        return False, str(exc)
    return True, "ok"


def _stage_output_paths(
    index: int,
    input_path: Path,
    mgf_files: list[Path],
    decoy_mgfs: list[Path],
) -> list[Path]:
    if index == 0:
        return decoy_mgfs
    if index == 1:
        return [_path_output(path) for path in [*mgf_files, *decoy_mgfs]]
    if index == 2:
        return [_path_fdr_output(path) for path in mgf_files]
    if index == 3:
        return [
            input_path / "filled_peptides.csv",
            input_path / "filled_peptides_with_PTM.csv",
            _pipeline_state_path(input_path),
        ]
    if index == 4:
        return [_seq_output(path) for path in [*mgf_files, *decoy_mgfs]]
    if index == 5:
        return [_seq_fdr_output(path) for path in mgf_files]
    raise ValueError(index)


def _confirm_resume(plan: ResumePlan, *, assume_yes: bool) -> None:
    assert plan.start_index is not None
    _print_resume_plan(plan)
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise RNovAError("Resume needs confirmation; rerun with --yes to skip the prompt")
    answer = input("Delete those failed-stage outputs and continue? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        raise RNovAError("Resume cancelled")


def _print_resume_plan(plan: ResumePlan) -> None:
    if plan.start_index is None:
        print("Resume detected a complete workflow; no stage needs to run.")
        return
    print("Resume detection:")
    if plan.completed_steps:
        print("  Completed stages:")
        for step in plan.completed_steps:
            print(f"    {step}")
    else:
        print("  Completed stages: none")
    print(f"  Restart stage: {STAGE_NAMES[plan.start_index]}")
    print(f"  Reason: {plan.reason}")
    existing_cleanup = [path for path in plan.cleanup_paths if path.exists()]
    if existing_cleanup:
        print("  Outputs to remove before restarting this stage:")
        for path in existing_cleanup:
            print(f"    {path}")
    else:
        print("  Outputs to remove before restarting this stage: none found")


def _cleanup_resume_outputs(paths: list[Path]) -> None:
    for path in paths:
        try:
            if path.is_file():
                path.unlink()
        except OSError as exc:
            raise RNovAError(f"Could not remove failed-stage output {path}: {exc}") from exc


def _pipeline_state_path(input_path: Path) -> Path:
    return input_path / ".cache" / "rnova" / "pipeline_state.json"


def _write_pipeline_state(
    input_path: Path,
    *,
    topk_annotation: str,
    use_unimod: bool,
    top_k_ptms: int,
) -> None:
    state_path = _pipeline_state_path(input_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "topk_annotation": topk_annotation,
        "use_unimod": use_unimod,
        "top_k_ptms": top_k_ptms,
    }
    tmp_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(state_path)


def _read_pipeline_state(
    input_path: Path,
    *,
    use_unimod: bool,
    top_k_ptms: int,
) -> str | None:
    try:
        payload = json.loads(_pipeline_state_path(input_path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("use_unimod") != use_unimod:
        return None
    if payload.get("top_k_ptms") != top_k_ptms:
        return None
    topk_annotation = payload.get("topk_annotation")
    if not isinstance(topk_annotation, str):
        return None
    return topk_annotation


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
    settings: RuntimeSettings,
    progress_interval: int | None,
    debug_inference: bool,
    refresh_workflow_cache: bool,
    resume_plan: ResumePlan | None = None,
) -> None:
    print(f"Input directory: {input_path}")
    print(f"Input MGF files found: {len(find_mgf_files(input_path))}")
    print(f"Decoy output directory: {input_path / 'decoy_mgf'}")
    print(f"PathSearcher checkpoint: {PATHSEARCHER_CHECKPOINT}")
    print(f"SeqFiller checkpoint: {SEQFILLER_CHECKPOINT}")
    print(f"Speed profile: {settings.speed_profile}")
    print(f"Progress: {settings.progress}")
    print(f"Log level: {settings.log_level}")
    print(f"Runtime settings source: {settings.source}")
    print(f"PathSearcher batch/workers: {settings.path_batch_size}/{settings.path_num_workers}")
    print(f"SeqFiller batch/workers: {settings.seq_batch_size}/{settings.seq_num_workers}")
    print(f"PathSearcher cache policy: {settings.path_cache_policy}")
    print(f"Workflow null workers: {settings.null_workers}")
    if resume_plan is not None:
        print()
        _print_resume_plan(resume_plan)
    print()
    for command in build_commands(
        input_path,
        use_unimod=use_unimod,
        refresh_unimod=refresh_unimod,
        top_k_ptms=top_k_ptms,
        path_batch_size=settings.path_batch_size,
        path_num_workers=settings.path_num_workers,
        seq_batch_size=settings.seq_batch_size,
        seq_num_workers=settings.seq_num_workers,
        progress_interval=progress_interval,
        speed_profile=settings.speed_profile,
        progress=settings.progress,
        log_level=settings.log_level,
        debug_inference=debug_inference,
        path_cache_policy=settings.path_cache_policy,
        null_workers=settings.null_workers,
        refresh_workflow_cache=refresh_workflow_cache,
    ):
        print(f"{command.step}:")
        print(f"  {command.display()}")


def _run(
    command: PlannedCommand,
    *,
    capture_stdout: bool = False,
    show_command: bool = False,
    echo_stdout: bool = True,
) -> str:
    print(f"{command.step}:")
    if show_command:
        print(f"$ {command.display()}")
    if capture_stdout:
        completed = subprocess.run(
            command.args,
            cwd=command.cwd,
            text=True,
            stdout=subprocess.PIPE,
            check=False,
        )
        if echo_stdout and completed.stdout:
            print(completed.stdout, end="")
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
