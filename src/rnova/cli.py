from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .checkpoints import download_checkpoints
from .errors import RNovAError
from .pipeline import run_pipeline, setup_seqfiller_extension
from .speed import LOG_LEVELS, PATH_CACHE_POLICIES, PROGRESS_MODES, SPEED_PROFILES
from .tuning import SAMPLE_STRATEGIES, run_tuning
from .validation import VALID_DOCTOR_MODES, collect_checks, print_checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rnova", description="RNovA personal monorepo workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the end-to-end RNovA workflow")
    run_parser.add_argument("input_dir", help="Directory containing input .mgf files")
    run_parser.add_argument("--use-unimod", action="store_true", help="Enable UniMod annotation")
    run_parser.add_argument(
        "--refresh-unimod",
        action="store_true",
        help="Refresh cached UniMod XML during workflow annotation",
    )
    run_parser.add_argument("--top-k-ptms", type=int, default=10, help="Number of PTMs to pass to SeqFiller")
    run_parser.add_argument(
        "--speed-profile",
        choices=SPEED_PROFILES,
        default="fast",
        help="Runtime speed profile (default: fast)",
    )
    run_parser.add_argument(
        "--progress",
        choices=PROGRESS_MODES,
        default="auto",
        help="Progress bar behavior (default: auto)",
    )
    run_parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="warning",
        help="Subprocess log verbosity (default: warning)",
    )
    run_parser.add_argument(
        "--path-batch-size",
        default="auto",
        help="PathSearcher inference batch size: auto or a positive integer (default: auto)",
    )
    run_parser.add_argument(
        "--path-num-workers",
        default="auto",
        help="PathSearcher DataLoader workers: auto or a nonnegative integer (default: auto)",
    )
    run_parser.add_argument(
        "--seq-batch-size",
        default="auto",
        help="SeqFiller inference batch size: auto or a positive integer (default: auto)",
    )
    run_parser.add_argument(
        "--seq-num-workers",
        default="auto",
        help="SeqFiller DataLoader workers: auto or a nonnegative integer (default: auto)",
    )
    run_parser.add_argument(
        "--progress-interval",
        type=int,
        help="With --debug-inference, log PathSearcher decoder internals every N steps",
    )
    run_parser.add_argument(
        "--debug-inference",
        action="store_true",
        help="Enable detailed encoder/decoder/cache diagnostics inside inference scripts",
    )
    run_parser.add_argument(
        "--retune",
        action="store_true",
        help="Tune GPU batch settings before running",
    )
    run_parser.add_argument(
        "--path-cache-policy",
        choices=PATH_CACHE_POLICIES,
        default="auto",
        help="PathSearcher decoder cache allocation policy (default: auto)",
    )
    run_parser.add_argument(
        "--null-workers",
        default="auto",
        help="Workflow null-distribution workers: auto or a positive integer (default: auto)",
    )
    run_parser.add_argument(
        "--refresh-workflow-cache",
        action="store_true",
        help="Rebuild cached workflow null distributions",
    )
    run_parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running them")
    run_parser.set_defaults(func=_run)

    tune_parser = subparsers.add_parser("tune", help="Benchmark small GPU samples and cache run settings")
    tune_parser.add_argument("input_dir", help="Directory containing input .mgf files")
    tune_parser.add_argument("--sample-spectra", type=int, default=64, help="Number of spectra to sample")
    tune_parser.add_argument(
        "--sample-strategy",
        choices=SAMPLE_STRATEGIES,
        default="representative",
        help="Tuning sample selection strategy (default: representative)",
    )
    tune_parser.add_argument(
        "--max-memory-frac",
        type=float,
        default=0.85,
        help="Maximum accepted peak GPU memory fraction (default: 0.85)",
    )
    tune_parser.add_argument(
        "--speed-profile",
        choices=SPEED_PROFILES,
        default="fast",
        help="Runtime speed profile to tune (default: fast)",
    )
    tune_parser.add_argument(
        "--progress",
        choices=PROGRESS_MODES,
        default="auto",
        help="Progress bar behavior during tuning (default: auto)",
    )
    tune_parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="warning",
        help="Subprocess log verbosity during tuning (default: warning)",
    )
    tune_parser.add_argument(
        "--path-cache-policy",
        choices=PATH_CACHE_POLICIES,
        default="auto",
        help="PathSearcher decoder cache allocation policy during tuning (default: auto)",
    )
    tune_parser.set_defaults(func=_tune)

    doctor_parser = subparsers.add_parser("doctor", help="Check installation and runtime readiness")
    doctor_parser.add_argument("input_dir", nargs="?", help="Optional input directory to validate")
    doctor_parser.add_argument(
        "--mode",
        choices=VALID_DOCTOR_MODES,
        default="inference",
        help="Depth of checks to run (default: inference)",
    )
    doctor_parser.add_argument(
        "--strict-gpu",
        action="store_true",
        help="Treat missing CUDA as a failure in inference mode",
    )
    doctor_parser.set_defaults(func=_doctor)

    setup_parser = subparsers.add_parser("setup", help="Build the SeqFiller Cython extension")
    setup_parser.set_defaults(func=_setup)

    checkpoint_parser = subparsers.add_parser("download-checkpoints", help="Download and install model checkpoints")
    checkpoint_parser.add_argument("--archive", help="Use an existing RNovA_Checkpoint.zip instead of downloading")
    checkpoint_parser.add_argument("--force", action="store_true", help="Overwrite existing archive/checkpoint files")
    checkpoint_parser.add_argument(
        "--allow-unverified-archive",
        action="store_true",
        help="Install an archive without checksum verification. Dangerous; only use for a trusted local archive.",
    )
    checkpoint_parser.set_defaults(func=_download_checkpoints)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except RNovAError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run(args: argparse.Namespace) -> None:
    run_pipeline(
        args.input_dir,
        use_unimod=args.use_unimod,
        refresh_unimod=args.refresh_unimod,
        top_k_ptms=args.top_k_ptms,
        path_batch_size=args.path_batch_size,
        path_num_workers=args.path_num_workers,
        seq_batch_size=args.seq_batch_size,
        seq_num_workers=args.seq_num_workers,
        progress_interval=args.progress_interval,
        speed_profile=args.speed_profile,
        progress=args.progress,
        log_level=args.log_level,
        debug_inference=args.debug_inference,
        retune=args.retune,
        path_cache_policy=args.path_cache_policy,
        null_workers=args.null_workers,
        refresh_workflow_cache=args.refresh_workflow_cache,
        dry_run=args.dry_run,
    )


def _tune(args: argparse.Namespace) -> None:
    tuning_path = run_tuning(
        args.input_dir,
        sample_spectra=args.sample_spectra,
        max_memory_frac=args.max_memory_frac,
        speed_profile=args.speed_profile,
        progress=args.progress,
        log_level=args.log_level,
        path_cache_policy=args.path_cache_policy,
        sample_strategy=args.sample_strategy,
    )
    print(f"Tuning saved: {tuning_path}")


def _doctor(args: argparse.Namespace) -> None:
    checks = collect_checks(args.input_dir, mode=args.mode, strict_gpu=args.strict_gpu)
    print_checks(checks)
    if any(check.failed for check in checks):
        raise RNovAError("doctor found one or more failed checks")


def _setup(args: argparse.Namespace) -> None:
    setup_seqfiller_extension()


def _download_checkpoints(args: argparse.Namespace) -> None:
    placed = download_checkpoints(
        archive=Path(args.archive) if args.archive else None,
        force=args.force,
        allow_unverified_archive=args.allow_unverified_archive,
    )
    for path in placed:
        print(f"Installed checkpoint: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
