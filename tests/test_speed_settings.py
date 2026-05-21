from __future__ import annotations

from pathlib import Path

from rnova.speed import (
    RuntimeSettings,
    batch_size_for_gpu_memory,
    load_tuning,
    resolve_runtime_settings,
    save_tuning,
)


MGF = """BEGIN IONS
TITLE=sample.1.1.2
PEPMASS=500.0
CHARGE=2+
100.0 10.0
END IONS
"""


def test_batch_size_for_desktop_gpu_memory_tiers() -> None:
    assert batch_size_for_gpu_memory(8) == 4
    assert batch_size_for_gpu_memory(12) == 8
    assert batch_size_for_gpu_memory(20) == 16
    assert batch_size_for_gpu_memory(24) == 32
    assert batch_size_for_gpu_memory(None) == 8


def test_resolve_runtime_settings_uses_auto_defaults(monkeypatch, tmp_path: Path) -> None:
    input_dir = _input_dir(tmp_path)
    monkeypatch.setattr("rnova.speed.cuda_total_memory_gib", lambda: 12)
    monkeypatch.setattr("rnova.speed.default_worker_counts", lambda: (0, 0))

    settings = resolve_runtime_settings(input_dir, use_tuning=False)

    assert settings.speed_profile == "fast"
    assert settings.path_batch_size == 8
    assert settings.seq_batch_size == 8
    assert settings.path_num_workers == 0
    assert settings.seq_num_workers == 0
    assert settings.log_level == "warning"
    assert settings.source == "auto"


def test_tuning_cache_is_used_and_explicit_values_win(monkeypatch, tmp_path: Path) -> None:
    input_dir = _input_dir(tmp_path)
    tuning_path = tmp_path / "tuning.json"
    monkeypatch.setattr("rnova.speed.cuda_total_memory_gib", lambda: 24)
    monkeypatch.setattr("rnova.speed.default_worker_counts", lambda: (2, 1))

    save_tuning(
        input_dir,
        tuning_path=tuning_path,
        settings=RuntimeSettings(
            speed_profile="fast",
            progress="auto",
            log_level="warning",
            path_cache_policy="auto",
            path_batch_size=16,
            path_num_workers=0,
            seq_batch_size=12,
            seq_num_workers=0,
            null_workers=2,
            source="tuning-cache",
        ),
        benchmark_results=[],
    )

    cached = load_tuning(input_dir, tuning_path=tuning_path)
    assert cached is not None

    settings = resolve_runtime_settings(input_dir, tuning_path=tuning_path)
    assert settings.source == "tuning-cache"
    assert settings.path_batch_size == 16
    assert settings.seq_batch_size == 12
    assert settings.path_num_workers == 0

    overridden = resolve_runtime_settings(
        input_dir,
        path_batch_size="4",
        seq_num_workers="3",
        tuning_path=tuning_path,
    )
    assert overridden.path_batch_size == 4
    assert overridden.seq_batch_size == 12
    assert overridden.seq_num_workers == 3


def _input_dir(tmp_path: Path) -> Path:
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    (input_dir / "a.mgf").write_text(MGF)
    return input_dir
