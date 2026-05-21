from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from rnova.pipeline import build_commands, expected_decoy_path
from rnova.validation import collect_checks, find_mgf_files, seqfiller_extension_files


MGF = """BEGIN IONS
TITLE=sample.1.1.2
PEPMASS=500.0
CHARGE=2+
100.0 10.0
200.0 20.0
END IONS
"""


def test_build_commands_uses_absolute_paths(tmp_path) -> None:
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    mgf = input_dir / "a.mgf"
    mgf.write_text(MGF)

    commands = build_commands(input_dir, use_unimod=True, top_k_ptms=10)

    assert str(mgf.resolve()) in commands[1].args
    assert str(expected_decoy_path(mgf.resolve(), input_dir.resolve() / "decoy_mgf")) in commands[1].args
    assert "--use-unimod" in commands[3].args


def test_build_commands_can_refresh_unimod(tmp_path) -> None:
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    (input_dir / "a.mgf").write_text(MGF)

    commands = build_commands(input_dir, use_unimod=True, refresh_unimod=True, top_k_ptms=10)

    assert "--refresh-unimod" in commands[3].args


def test_build_commands_can_override_inference_batch_sizes(tmp_path) -> None:
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    (input_dir / "a.mgf").write_text(MGF)

    commands = build_commands(
        input_dir,
        use_unimod=False,
        top_k_ptms=10,
        path_batch_size=8,
        path_num_workers=0,
        seq_batch_size=4,
        seq_num_workers=0,
        progress_interval=2,
    )

    assert commands[1].args[2:8] == [
        "--batch-size",
        "8",
        "--num-workers",
        "0",
        "--progress-interval",
        "2",
    ]
    assert commands[4].args[2:6] == ["--batch-size", "4", "--num-workers", "0"]
    assert "--speed-profile" in commands[1].args
    assert "--path-cache-policy" in commands[1].args
    assert "--log-level" in commands[4].args


def test_cli_dry_run_prints_planned_commands(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    (input_dir / "a.mgf").write_text(MGF)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}{os.pathsep}{root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "rnova.cli",
            "run",
            str(input_dir),
            "--use-unimod",
            "--top-k-ptms",
            "10",
            "--dry-run",
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "2/6 Run PathSearcher inference" in completed.stdout
    assert "5/6 Run SeqFiller inference" in completed.stdout
    assert "a.mgf.decoy_random0.40.mgf" in completed.stdout


def test_cli_rejects_non_positive_top_k(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    (input_dir / "a.mgf").write_text(MGF)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}{os.pathsep}{root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "rnova.cli",
            "run",
            str(input_dir),
            "--top-k-ptms",
            "0",
            "--dry-run",
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "--top-k-ptms must be greater than 0" in completed.stderr


def test_cli_dry_run_prints_batch_overrides(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    (input_dir / "a.mgf").write_text(MGF)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}{os.pathsep}{root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "rnova.cli",
            "run",
            str(input_dir),
            "--dry-run",
            "--path-batch-size",
            "8",
            "--path-num-workers",
            "0",
            "--seq-batch-size",
            "4",
            "--seq-num-workers",
            "0",
            "--progress-interval",
            "2",
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Inference_Node.py --batch-size 8 --num-workers 0 --progress-interval 2" in completed.stdout
    assert "Inference_Sequence.py --batch-size 4 --num-workers 0" in completed.stdout


def test_cli_dry_run_prints_speed_defaults(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    (input_dir / "a.mgf").write_text(MGF)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}{os.pathsep}{root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "rnova.cli",
            "run",
            str(input_dir),
            "--dry-run",
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Speed profile: fast" in completed.stdout
    assert "Progress: auto" in completed.stdout
    assert "Log level: warning" in completed.stdout
    assert "--speed-profile fast --progress auto --log-level warning" in completed.stdout


def test_cli_dry_run_accepts_debug_and_cache_policy(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    (input_dir / "a.mgf").write_text(MGF)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}{os.pathsep}{root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "rnova.cli",
            "run",
            str(input_dir),
            "--dry-run",
            "--debug-inference",
            "--path-cache-policy",
            "legacy",
            "--progress",
            "off",
            "--log-level",
            "debug",
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "PathSearcher cache policy: legacy" in completed.stdout
    assert "--path-cache-policy legacy --debug-inference" in completed.stdout
    assert "--progress off --log-level debug" in completed.stdout


def test_find_mgf_files_and_missing_extension_validation(tmp_path) -> None:
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("nope")
    assert find_mgf_files(input_dir) == []

    empty_root = tmp_path / "empty-root"
    empty_root.mkdir()
    assert seqfiller_extension_files(empty_root) == []


def test_collect_checks_reports_missing_checkpoints_and_empty_input(tmp_path) -> None:
    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    for relative in [
        "RNovA_PathSearcher_Inference/Inference_Node.py",
        "RNovA_SeqFiller_Inference/Inference_Sequence.py",
        "RNovA_SeqFiller_Inference/setup.py",
    ]:
        path = fake_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    input_dir = tmp_path / "empty-data"
    input_dir.mkdir()

    checks = collect_checks(input_dir, repo_root=fake_root)
    failed = {check.name: check.detail for check in checks if check.failed}

    assert failed["Checkpoint RNovA_PathSearcher_Inference"].startswith("Missing")
    assert failed["Checkpoint RNovA_SeqFiller_Inference"].startswith("Missing")
    assert failed["SeqFiller extension"] == "missing; run `uv run rnova setup`"
    assert failed["Input MGF files"].startswith("No .mgf files found")


def test_collect_checks_dry_run_mode_skips_inference_requirements(tmp_path) -> None:
    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    for relative in [
        "RNovA_PathSearcher_Inference/Inference_Node.py",
        "RNovA_SeqFiller_Inference/Inference_Sequence.py",
        "RNovA_SeqFiller_Inference/setup.py",
    ]:
        path = fake_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    (input_dir / "a.mgf").write_text(MGF)

    checks = collect_checks(input_dir, mode="dry-run", repo_root=fake_root)

    assert not [check for check in checks if check.failed]
    assert all(not check.name.startswith("Checkpoint") for check in checks)


def test_collect_checks_reports_invalid_mgf(tmp_path) -> None:
    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    for relative in [
        "RNovA_PathSearcher_Inference/Inference_Node.py",
        "RNovA_SeqFiller_Inference/Inference_Sequence.py",
        "RNovA_SeqFiller_Inference/setup.py",
    ]:
        path = fake_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    (input_dir / "bad.mgf").write_text(
        """BEGIN IONS
TITLE=bad.1.1.2
PEPMASS=500.0
CHARGE=2+
100.0 0.0
END IONS
"""
    )

    checks = collect_checks(input_dir, mode="dry-run", repo_root=fake_root)
    failed = {check.name: check.detail for check in checks if check.failed}

    assert "Input MGF parse" in failed
    assert "intensity must be positive" in failed["Input MGF parse"]
