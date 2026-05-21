from __future__ import annotations

from pathlib import Path

import pytest

from rnova.errors import RNovAError
from rnova.mgf import MGFSpectrumBlock, MGFPeak
from rnova.speed import (
    RuntimeSettings,
    batch_size_for_gpu_memory,
    candidate_batch_sizes,
    load_tuning,
    resolve_runtime_settings,
    save_tuning,
)
from rnova.tuning import (
    _BlockCandidate,
    _candidate_batch_sizes_for_sample,
    _choose_best_stage_result,
    _format_tuning_failure,
    _select_sample_blocks,
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


def test_tuning_candidates_use_desktop_scan_ladder() -> None:
    assert candidate_batch_sizes(12) == [8, 16, 32, 64, 128]


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


def test_tuning_selects_best_stage_independently() -> None:
    results = [
        {"status": "ok", "batch_size": 1, "spectra_per_second": 1.0, "peak_memory_gib": 2.0},
        {"status": "failed", "batch_size": 2, "stderr": "boom"},
        {"status": "ok", "batch_size": 4, "spectra_per_second": 3.0, "peak_memory_gib": 5.0},
    ]

    best = _choose_best_stage_result(results, total_memory_gib=12, max_memory_frac=0.85)

    assert best is not None
    assert best["batch_size"] == 4


def test_tuning_failure_message_includes_stage_details(tmp_path: Path) -> None:
    message = _format_tuning_failure(
        [{"status": "failed", "batch_size": 1, "stderr": "Traceback\nCUDA OOM"}],
        [{"status": "ok", "batch_size": 1, "spectra_per_second": 2.0, "peak_memory_gib": 11.0}],
        failure_path=tmp_path / "tuning_failure.json",
        total_memory_gib=12,
        max_memory_frac=0.85,
    )

    assert "PathSearcher attempts:" in message
    assert "batch 1: failed (CUDA OOM)" in message
    assert "SeqFiller attempts:" in message
    assert "batch 1: ok" in message
    assert "Full tuning details:" in message


def test_tuning_candidates_skip_batches_larger_than_sample() -> None:
    selected, skipped = _candidate_batch_sizes_for_sample(32, total_memory_gib=12)

    assert selected == [8, 16, 32]
    assert skipped == [64, 128]


def test_tuning_candidates_require_enough_sample_spectra() -> None:
    with pytest.raises(RNovAError, match="--sample-spectra must be at least 8"):
        _candidate_batch_sizes_for_sample(7, total_memory_gib=12)


def test_representative_tuning_sample_includes_heavy_spectra(tmp_path: Path) -> None:
    candidates = [
        _block_candidate(tmp_path, ordinal=0, peak_count=10),
        _block_candidate(tmp_path, ordinal=1, peak_count=300),
        _block_candidate(tmp_path, ordinal=2, peak_count=20),
        _block_candidate(tmp_path, ordinal=3, peak_count=250),
        _block_candidate(tmp_path, ordinal=4, peak_count=30),
        _block_candidate(tmp_path, ordinal=5, peak_count=40),
    ]

    selected = _select_sample_blocks(candidates, 4, strategy="representative")

    assert [candidate.peak_count for candidate in selected][:2] == [300, 250]
    assert len(selected) == 4


def test_first_tuning_sample_preserves_original_prefix(tmp_path: Path) -> None:
    candidates = [
        _block_candidate(tmp_path, ordinal=0, peak_count=10),
        _block_candidate(tmp_path, ordinal=1, peak_count=300),
        _block_candidate(tmp_path, ordinal=2, peak_count=20),
    ]

    selected = _select_sample_blocks(candidates, 2, strategy="first")

    assert [candidate.ordinal for candidate in selected] == [0, 1]


def _input_dir(tmp_path: Path) -> Path:
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    (input_dir / "a.mgf").write_text(MGF)
    return input_dir


def _block_candidate(tmp_path: Path, *, ordinal: int, peak_count: int) -> _BlockCandidate:
    peak = MGFPeak(moverz=100.0, intensity=10.0)
    block = MGFSpectrumBlock(
        index=ordinal + 1,
        title=str(ordinal + 1),
        scans=None,
        precursor_moverz=500.0,
        charge=2,
        peaks=tuple(peak for _ in range(peak_count)),
        header_lines=("BEGIN IONS\n", "PEPMASS=500.0\n", "CHARGE=2+\n"),
    )
    return _BlockCandidate(
        block=block,
        path=tmp_path / "sample.mgf",
        ordinal=ordinal,
        peak_count=peak_count,
    )
