from __future__ import annotations

import subprocess
from pathlib import Path


BANNED_SUFFIXES = (
    ".so",
    ".pyd",
    ".dylib",
    ".pt",
    ".ckpt",
    ".zip",
    ".pyc",
)
BANNED_PARTS = {".venv", ".cache", "__pycache__", "build", "dist", ".pytest_cache"}
BANNED_NAMES = {"knapsack_build.c"}


def test_generated_artifacts_are_not_trackable() -> None:
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    untracked = subprocess.run(
        ["git", "ls-files", "-o", "--exclude-standard"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()

    offenders = sorted(
        path for path in [*tracked, *untracked]
        if (root / path).exists() and _is_banned(Path(path))
    )

    assert offenders == []


def test_flash_attn_build_dependencies_are_declared() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text()

    assert "[tool.uv.extra-build-dependencies]" in pyproject
    assert "flash-attn" in pyproject
    assert 'requirement = "torch"' in pyproject
    assert "match-runtime = true" in pyproject
    assert '"packaging"' in pyproject
    assert '"ninja"' in pyproject


def _is_banned(path: Path) -> bool:
    if path.name in BANNED_NAMES:
        return True
    if any(part in BANNED_PARTS for part in path.parts):
        return True
    return path.suffix in BANNED_SUFFIXES
