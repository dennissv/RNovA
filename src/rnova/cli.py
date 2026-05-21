from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .checkpoints import download_checkpoints
from .errors import RNovAError
from .pipeline import run_pipeline, setup_seqfiller_extension
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
    run_parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running them")
    run_parser.set_defaults(func=_run)

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
        dry_run=args.dry_run,
    )


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
