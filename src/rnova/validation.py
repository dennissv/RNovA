from __future__ import annotations

import importlib
import importlib.machinery
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .errors import RNovAError
from .mgf import MGFParseError, read_mgf_blocks


REPO_ROOT = Path(__file__).resolve().parents[2]

PATHSEARCHER_DIR = REPO_ROOT / "RNovA_PathSearcher_Inference"
SEQFILLER_DIR = REPO_ROOT / "RNovA_SeqFiller_Inference"
PATHSEARCHER_CHECKPOINT = PATHSEARCHER_DIR / "save" / "rnova.pt"
SEQFILLER_CHECKPOINT = SEQFILLER_DIR / "save" / "rnova.pt"


def required_vendor_files(repo_root: Path = REPO_ROOT) -> tuple[Path, ...]:
    return (
        repo_root / "RNovA_PathSearcher_Inference" / "Inference_Node.py",
        repo_root / "RNovA_SeqFiller_Inference" / "Inference_Sequence.py",
        repo_root / "RNovA_SeqFiller_Inference" / "setup.py",
    )


REQUIRED_VENDOR_FILES = required_vendor_files()

DRY_RUN_IMPORTS: tuple[str, ...] = ()
WORKFLOW_IMPORTS = (
    "numpy",
    "pandas",
    "numba",
    "tqdm",
    "requests",
)
INFERENCE_IMPORTS = WORKFLOW_IMPORTS + (
    "einops",
    "polars",
    "hydra",
    "omegaconf",
    "torch",
    "triton",
    "Cython",
)

VALID_DOCTOR_MODES = ("dry-run", "workflow", "inference")


@dataclass
class Check:
    name: str
    status: str
    detail: str

    @property
    def failed(self) -> bool:
        return self.status == "fail"


def find_mgf_files(input_dir: str | Path) -> list[Path]:
    path = Path(input_dir).expanduser().resolve()
    if not path.exists():
        return []
    return sorted(path.glob("*.mgf"))


def check_directory_writable(path: str | Path) -> bool:
    directory = Path(path)
    if not directory.exists() or not directory.is_dir():
        return False
    try:
        with tempfile.NamedTemporaryFile(prefix=".rnova-write-test-", dir=directory, delete=True):
            pass
    except OSError:
        return False
    return True


def checkpoint_paths(repo_root: Path = REPO_ROOT) -> tuple[Path, Path]:
    return (
        repo_root / "RNovA_PathSearcher_Inference" / "save" / "rnova.pt",
        repo_root / "RNovA_SeqFiller_Inference" / "save" / "rnova.pt",
    )


def seqfiller_extension_files(repo_root: Path = REPO_ROOT) -> list[Path]:
    data_dir = repo_root / "RNovA_SeqFiller_Inference" / "data"
    candidates: list[Path] = []
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        if suffix == ".so":
            bare_extension = data_dir / "knapsack_build.so"
            if bare_extension.exists():
                candidates.append(bare_extension)
        else:
            candidates.extend(data_dir.glob(f"knapsack_build*{suffix}"))
    return sorted(candidates)


def any_seqfiller_extension_files(repo_root: Path = REPO_ROOT) -> list[Path]:
    data_dir = repo_root / "RNovA_SeqFiller_Inference" / "data"
    return sorted(data_dir.glob("knapsack_build*.so")) + sorted(
        data_dir.glob("knapsack_build*.pyd")
    )


def collect_checks(
    input_dir: str | Path | None = None,
    *,
    mode: str = "inference",
    strict_gpu: bool = False,
    repo_root: Path = REPO_ROOT,
) -> list[Check]:
    if mode not in VALID_DOCTOR_MODES:
        raise RNovAError(f"mode must be one of: {', '.join(VALID_DOCTOR_MODES)}")

    checks: list[Check] = []

    version = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info >= (3, 10):
        checks.append(Check("Python", "ok", version))
    else:
        checks.append(Check("Python", "fail", f"{version}; Python >= 3.10 required"))

    for file_path in required_vendor_files(repo_root):
        if file_path.exists():
            checks.append(Check(f"Vendored file {file_path.name}", "ok", str(file_path)))
        else:
            checks.append(Check(f"Vendored file {file_path.name}", "fail", f"Missing {file_path}"))

    imports = {
        "dry-run": DRY_RUN_IMPORTS,
        "workflow": WORKFLOW_IMPORTS,
        "inference": INFERENCE_IMPORTS,
    }[mode]

    for import_name in imports:
        if importlib.util.find_spec(import_name) is None:
            checks.append(Check(f"Import {import_name}", "fail", _missing_import_detail(import_name, mode)))
        else:
            checks.append(Check(f"Import {import_name}", "ok", "available"))

    if mode == "inference":
        flash_status = "ok" if importlib.util.find_spec("flash_attn") else "fail"
        flash_detail = "available" if flash_status == "ok" else (
            "missing; run `uv sync --locked` on the GPU workstation. "
            "If only flash-attn failed to build, try `MAX_JOBS=4 uv pip install flash-attn --no-build-isolation`."
        )
        checks.append(Check("Import flash_attn", flash_status, flash_detail))

        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is None:
            checks.append(
                Check(
                    "CUDA",
                    "fail",
                    "torch is not installed; run `uv sync --locked` on the GPU workstation. "
                    "If this is a test-only sync that intentionally skipped GPU packages, use dry-run or workflow mode.",
                )
            )
        else:
            try:
                torch = importlib.import_module("torch")
                if torch.cuda.is_available():
                    checks.append(Check("CUDA", "ok", f"{torch.cuda.device_count()} CUDA device(s) visible"))
                else:
                    checks.append(
                        Check(
                            "CUDA",
                            "fail" if strict_gpu else "warn",
                            "torch imports, but no CUDA device is visible",
                        )
                    )
            except Exception as exc:  # pragma: no cover - defensive doctor output
                checks.append(Check("CUDA", "fail", f"torch import failed: {exc}"))

        for checkpoint in checkpoint_paths(repo_root):
            if checkpoint.exists():
                checks.append(Check(f"Checkpoint {checkpoint.parent.parent.name}", "ok", str(checkpoint)))
            else:
                checks.append(Check(f"Checkpoint {checkpoint.parent.parent.name}", "fail", f"Missing {checkpoint}"))

        compatible_extensions = seqfiller_extension_files(repo_root)
        if compatible_extensions:
            checks.append(Check("SeqFiller extension", "ok", ", ".join(map(str, compatible_extensions))))
        else:
            existing = any_seqfiller_extension_files(repo_root)
            if existing:
                checks.append(
                    Check(
                        "SeqFiller extension",
                        "fail",
                        "found extension for a different Python/platform: "
                        + ", ".join(map(str, existing)),
                    )
                )
            else:
                checks.append(
                    Check(
                        "SeqFiller extension",
                        "fail",
                        "missing; run `uv run rnova setup`",
                    )
                )

    if input_dir is not None:
        input_path = Path(input_dir).expanduser().resolve()
        if not input_path.exists():
            checks.append(Check("Input directory", "fail", f"Missing {input_path}"))
        else:
            mgfs = find_mgf_files(input_path)
            if mgfs:
                checks.append(Check("Input MGF files", "ok", f"{len(mgfs)} file(s) in {input_path}"))
                invalid_mgfs = _invalid_mgf_details(mgfs)
                if invalid_mgfs:
                    checks.append(Check("Input MGF parse", "fail", "; ".join(invalid_mgfs)))
                else:
                    checks.append(Check("Input MGF parse", "ok", "all input MGF files parsed"))
            else:
                checks.append(Check("Input MGF files", "fail", f"No .mgf files found in {input_path}"))
            if check_directory_writable(input_path):
                checks.append(Check("Output directory writable", "ok", str(input_path)))
            else:
                checks.append(Check("Output directory writable", "fail", f"Cannot write to {input_path}"))

    return checks


def print_checks(checks: list[Check]) -> None:
    for check in checks:
        marker = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}[check.status]
        print(f"[{marker}] {check.name}: {check.detail}")


def _missing_import_detail(import_name: str, mode: str) -> str:
    if mode == "inference":
        return (
            "not installed; run `uv sync --locked` on the GPU workstation. "
            "If this is a test-only sync that intentionally skipped GPU packages, use dry-run or workflow mode."
        )
    if mode == "workflow":
        return "not installed; run `uv sync --locked`"
    return "not installed"


def _invalid_mgf_details(mgf_files: list[Path]) -> list[str]:
    invalid: list[str] = []
    for mgf_file in mgf_files:
        try:
            read_mgf_blocks(mgf_file)
        except (MGFParseError, OSError) as exc:
            invalid.append(f"{mgf_file}: {exc}")
    return invalid


def failed_checks(checks: list[Check]) -> list[Check]:
    return [check for check in checks if check.failed]


def assert_checks_pass(checks: list[Check]) -> None:
    failed = failed_checks(checks)
    if failed:
        details = "\n".join(f"- {check.name}: {check.detail}" for check in failed)
        raise RNovAError(f"preflight failed:\n{details}")
