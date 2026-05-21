from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import RNovAError
from .mgf import MGFSpectrumBlock, read_mgf_blocks
from .speed import (
    DEFAULT_TUNING_PATH,
    RuntimeSettings,
    candidate_batch_sizes,
    cuda_device_info,
    resolve_runtime_settings,
    save_tuning,
)
from .validation import (
    PATHSEARCHER_DIR,
    REPO_ROOT,
    SEQFILLER_DIR,
    assert_checks_pass,
    collect_checks,
    find_mgf_files,
    seqfiller_extension_files,
)


BASE_CANDIDATE_AMINO_ACIDS = "A;C|UniMod:4;D;E;F;G;H;K;L;M;N;P;Q;R;S;T;V;W;Y"
SAMPLE_STRATEGIES = ("representative", "first")


@dataclass(frozen=True)
class _BlockCandidate:
    block: MGFSpectrumBlock
    path: Path
    ordinal: int
    peak_count: int


def run_tuning(
    input_dir: str | Path,
    *,
    sample_spectra: int = 64,
    max_memory_frac: float = 0.85,
    speed_profile: str = "fast",
    path_cache_policy: str = "auto",
    progress: str = "auto",
    log_level: str = "warning",
    sample_strategy: str = "representative",
    tuning_path: Path = DEFAULT_TUNING_PATH,
) -> Path:
    if sample_spectra <= 0:
        raise RNovAError("--sample-spectra must be greater than 0")
    if not 0 < max_memory_frac <= 1:
        raise RNovAError("--max-memory-frac must be greater than 0 and at most 1")
    if sample_strategy not in SAMPLE_STRATEGIES:
        raise RNovAError(f"--sample-strategy must be one of: {', '.join(SAMPLE_STRATEGIES)}")

    input_path = Path(input_dir).expanduser().resolve()
    assert_checks_pass(collect_checks(input_path, mode="inference", strict_gpu=True))
    if not seqfiller_extension_files():
        raise RNovAError("SeqFiller extension is missing or incompatible. Run `uv run rnova setup`.")
    mgf_files = find_mgf_files(input_path)
    if not mgf_files:
        raise RNovAError(f"No .mgf files found in {input_path}")

    device = cuda_device_info()
    total_memory_gib = device.get("total_memory_gib") if device.get("available") else None
    base_settings = resolve_runtime_settings(
        input_path,
        speed_profile=speed_profile,
        progress=progress,
        log_level=log_level,
        path_cache_policy=path_cache_policy,
        use_tuning=False,
    )

    tune_dir = REPO_ROOT / ".cache" / "rnova" / "tune"
    tune_dir.mkdir(parents=True, exist_ok=True)
    sample_mgf = tune_dir / "sample.mgf"
    sample_summary = _write_sample_mgf(
        mgf_files,
        sample_mgf,
        sample_spectra,
        strategy=sample_strategy,
    )
    sampled = int(sample_summary["sampled_spectra"])
    if sampled == 0:
        raise RNovAError("Tuning sample did not contain any spectra")
    _print_sample_summary(sample_summary)
    candidates, skipped_candidates = _candidate_batch_sizes_for_sample(
        sampled,
        total_memory_gib=total_memory_gib,
    )
    print(f"Tuning candidates: {', '.join(map(str, candidates))}", flush=True)
    if skipped_candidates:
        skipped = ", ".join(map(str, skipped_candidates))
        print(
            f"Skipping candidate batch size(s) {skipped}; sample only has {sampled} spectra.",
            flush=True,
        )

    benchmark_results: list[dict[str, Any]] = []
    path_results = _benchmark_stage_candidates(
        "pathsearcher",
        candidates=candidates,
        num_workers=base_settings.path_num_workers,
        sample_mgf=sample_mgf,
        speed_profile=speed_profile,
        progress=progress,
        log_level=log_level,
        path_cache_policy=path_cache_policy,
        tune_dir=tune_dir,
    )
    seq_results = _benchmark_stage_candidates(
        "seqfiller",
        candidates=candidates,
        num_workers=base_settings.seq_num_workers,
        sample_mgf=sample_mgf,
        speed_profile=speed_profile,
        progress=progress,
        log_level=log_level,
        path_cache_policy=path_cache_policy,
        tune_dir=tune_dir,
    )
    benchmark_results.extend(
        [
            {"stage": "pathsearcher", "results": path_results},
            {"stage": "seqfiller", "results": seq_results},
        ]
    )

    path_best = _choose_best_stage_result(
        path_results,
        total_memory_gib=total_memory_gib,
        max_memory_frac=max_memory_frac,
    )
    seq_best = _choose_best_stage_result(
        seq_results,
        total_memory_gib=total_memory_gib,
        max_memory_frac=max_memory_frac,
    )
    if path_best is None or seq_best is None:
        failure_path = _write_tuning_failure(
            tune_dir,
            input_path=input_path,
            sampled=sampled,
            device=device,
            base_settings=base_settings,
            benchmark_results=benchmark_results,
            max_memory_frac=max_memory_frac,
            sample_summary=sample_summary,
        )
        raise RNovAError(
            _format_tuning_failure(
                path_results,
                seq_results,
                failure_path=failure_path,
                total_memory_gib=total_memory_gib,
                max_memory_frac=max_memory_frac,
            )
        )

    tuned_settings = RuntimeSettings(
        speed_profile=speed_profile,
        progress=progress,
        log_level=log_level,
        path_cache_policy=path_cache_policy,
        path_batch_size=int(path_best["batch_size"]),
        seq_batch_size=int(seq_best["batch_size"]),
        path_num_workers=base_settings.path_num_workers,
        seq_num_workers=base_settings.seq_num_workers,
        null_workers=base_settings.null_workers,
        source="tuning-cache",
    )
    return save_tuning(
        input_path,
        settings=tuned_settings,
        benchmark_results=benchmark_results,
        device=device,
        sample_spectra=sampled,
        sample_summary=sample_summary,
        max_memory_frac=max_memory_frac,
        tuning_path=tuning_path,
    )


def _candidate_batch_sizes_for_sample(
    sampled: int,
    *,
    total_memory_gib: float | None,
) -> tuple[list[int], list[int]]:
    candidates = candidate_batch_sizes(total_memory_gib)
    selected = [candidate for candidate in candidates if candidate <= sampled]
    skipped = [candidate for candidate in candidates if candidate > sampled]
    if not selected:
        first_candidate = candidates[0] if candidates else "the first candidate batch size"
        raise RNovAError(
            f"--sample-spectra must be at least {first_candidate} to tune the current batch ladder"
        )
    return selected, skipped


def _benchmark_stage_candidates(
    stage: str,
    *,
    candidates: list[int],
    num_workers: int,
    sample_mgf: Path,
    speed_profile: str,
    progress: str,
    log_level: str,
    path_cache_policy: str,
    tune_dir: Path,
) -> list[dict[str, Any]]:
    results = []
    for batch_size in candidates:
        print(f"Tuning {stage} batch size {batch_size}...", flush=True)
        results.append(
            _run_inference_benchmark(
                stage,
                batch_size=batch_size,
                num_workers=num_workers,
                sample_mgf=sample_mgf,
                speed_profile=speed_profile,
                progress=progress,
                log_level=log_level,
                path_cache_policy=path_cache_policy,
                tune_dir=tune_dir,
            )
        )
    return results


def _choose_best_stage_result(
    results: list[dict[str, Any]],
    *,
    total_memory_gib: float | None,
    max_memory_frac: float,
) -> dict[str, Any] | None:
    accepted = []
    for result in results:
        if result.get("status") != "ok":
            continue
        peak_gib = float(result.get("peak_memory_gib", 0))
        if total_memory_gib is not None and peak_gib > total_memory_gib * max_memory_frac:
            result["rejected_reason"] = (
                f"peak memory {peak_gib:.2f} GiB exceeded "
                f"{max_memory_frac:.0%} of {total_memory_gib:.2f} GiB"
            )
            continue
        accepted.append(result)
    if not accepted:
        return None
    return max(accepted, key=lambda result: result.get("spectra_per_second", 0))


def _write_tuning_failure(
    tune_dir: Path,
    *,
    input_path: Path,
    sampled: int,
    device: dict[str, Any],
    base_settings: RuntimeSettings,
    benchmark_results: list[dict[str, Any]],
    max_memory_frac: float,
    sample_summary: dict[str, Any] | None = None,
) -> Path:
    failure_path = tune_dir / "tuning_failure.json"
    payload = {
        "input_dir": str(input_path),
        "sample_spectra": sampled,
        "device": device,
        "base_settings": asdict(base_settings),
        "max_memory_frac": max_memory_frac,
        "sample_summary": sample_summary or {},
        "benchmark_results": benchmark_results,
    }
    failure_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return failure_path


def _format_tuning_failure(
    path_results: list[dict[str, Any]],
    seq_results: list[dict[str, Any]],
    *,
    failure_path: Path,
    total_memory_gib: float | None,
    max_memory_frac: float,
) -> str:
    header = "Tuning did not find successful settings."
    if total_memory_gib is not None:
        header += f" Memory limit was {max_memory_frac:.0%} of {total_memory_gib:.2f} GiB."
    return "\n".join(
        [
            header,
            _format_stage_results("PathSearcher", path_results),
            _format_stage_results("SeqFiller", seq_results),
            f"Full tuning details: {failure_path}",
        ]
    )


def _format_stage_results(stage_name: str, results: list[dict[str, Any]]) -> str:
    lines = [f"{stage_name} attempts:"]
    for result in results:
        batch = result.get("batch_size", "?")
        status = result.get("status", "unknown")
        if status == "ok":
            peak = result.get("peak_memory_gib", 0)
            reserved = result.get("peak_memory_reserved_gib")
            allocated = result.get("peak_memory_allocated_gib")
            speed = result.get("spectra_per_second", 0)
            rejected = result.get("rejected_reason")
            if rejected:
                lines.append(f"  batch {batch}: rejected ({rejected})")
            elif reserved is not None and allocated is not None:
                lines.append(
                    f"  batch {batch}: ok ({speed:.3f} spectra/s, "
                    f"reserved {reserved:.2f} GiB, allocated {allocated:.2f} GiB)"
                )
            else:
                lines.append(f"  batch {batch}: ok ({speed:.3f} spectra/s, peak {peak:.2f} GiB)")
            continue
        detail = _first_error_line(result)
        lines.append(f"  batch {batch}: failed ({detail})")
    return "\n".join(lines)


def _first_error_line(result: dict[str, Any]) -> str:
    for key in ("stderr", "stdout"):
        text = str(result.get(key) or "").strip()
        if not text:
            continue
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line:
                return line[:240]
    return f"exit code {result.get('returncode', 'unknown')}"


def _write_sample_mgf(
    mgf_files: list[Path],
    output_path: Path,
    sample_spectra: int,
    *,
    strategy: str,
) -> dict[str, Any]:
    candidates = _collect_block_candidates(mgf_files)
    blocks = _select_sample_blocks(candidates, sample_spectra, strategy=strategy)

    with output_path.open("w") as handle:
        for block in blocks:
            for header_line in block.block.header_lines:
                handle.write(header_line if header_line.endswith("\n") else f"{header_line}\n")
            for peak in block.block.peaks:
                handle.write(f"{peak.moverz:.8f} {peak.intensity:.8f}\n")
            end_line = block.block.end_line
            handle.write(end_line if end_line.endswith("\n") else f"{end_line}\n")

    return _sample_summary(candidates, blocks, strategy=strategy)


def _collect_block_candidates(mgf_files: list[Path]) -> list[_BlockCandidate]:
    candidates: list[_BlockCandidate] = []
    for mgf_file in mgf_files:
        for block in read_mgf_blocks(mgf_file):
            candidates.append(
                _BlockCandidate(
                    block=block,
                    path=mgf_file,
                    ordinal=len(candidates),
                    peak_count=len(block.peaks),
                )
            )
    return candidates


def _select_sample_blocks(
    candidates: list[_BlockCandidate],
    sample_spectra: int,
    *,
    strategy: str,
) -> list[_BlockCandidate]:
    if sample_spectra <= 0:
        return []
    if len(candidates) <= sample_spectra or strategy == "first":
        return candidates[:sample_spectra]

    heavy_target = min(len(candidates), max(1, math.ceil(sample_spectra * 0.5)))
    selected: dict[int, _BlockCandidate] = {}
    for candidate in sorted(candidates, key=lambda item: (-item.peak_count, item.ordinal))[:heavy_target]:
        selected[candidate.ordinal] = candidate

    remaining_slots = sample_spectra - len(selected)
    for candidate in _evenly_spaced_candidates(candidates, remaining_slots):
        selected.setdefault(candidate.ordinal, candidate)
        if len(selected) >= sample_spectra:
            break

    if len(selected) < sample_spectra:
        for candidate in candidates:
            selected.setdefault(candidate.ordinal, candidate)
            if len(selected) >= sample_spectra:
                break

    return sorted(selected.values(), key=lambda item: (-item.peak_count, item.ordinal))


def _evenly_spaced_candidates(
    candidates: list[_BlockCandidate],
    count: int,
) -> list[_BlockCandidate]:
    if count <= 0 or not candidates:
        return []
    if count == 1:
        return [candidates[len(candidates) // 2]]
    last_index = len(candidates) - 1
    return [candidates[round(index * last_index / (count - 1))] for index in range(count)]


def _sample_summary(
    candidates: list[_BlockCandidate],
    selected: list[_BlockCandidate],
    *,
    strategy: str,
) -> dict[str, Any]:
    all_peaks = [candidate.peak_count for candidate in candidates]
    selected_peaks = [candidate.peak_count for candidate in selected]
    return {
        "strategy": strategy,
        "total_spectra": len(candidates),
        "sampled_spectra": len(selected),
        "full_peak_count": _peak_count_stats(all_peaks),
        "sample_peak_count": _peak_count_stats(selected_peaks),
        "sample_files": sorted({str(candidate.path) for candidate in selected}),
    }


def _peak_count_stats(values: list[int]) -> dict[str, int]:
    if not values:
        return {"min": 0, "median": 0, "p95": 0, "max": 0}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _percentile(ordered: list[int], fraction: float) -> int:
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _print_sample_summary(summary: dict[str, Any]) -> None:
    full = summary["full_peak_count"]
    sample = summary["sample_peak_count"]
    print(
        (
            f"Tuning sample: {summary['sampled_spectra']} of {summary['total_spectra']} spectra "
            f"({summary['strategy']}; raw peaks full max/p95={full['max']}/{full['p95']}, "
            f"sample max/p95={sample['max']}/{sample['p95']})"
        ),
        flush=True,
    )


def _run_inference_benchmark(
    stage: str,
    *,
    batch_size: int,
    num_workers: int,
    sample_mgf: Path,
    speed_profile: str,
    progress: str,
    log_level: str,
    path_cache_policy: str,
    tune_dir: Path,
) -> dict[str, Any]:
    benchmark_json = tune_dir / f"{stage}_batch_{batch_size}.json"
    try:
        benchmark_json.unlink()
    except FileNotFoundError:
        pass
    if stage == "pathsearcher":
        command = [
            sys.executable,
            "Inference_Node.py",
            "--batch-size",
            str(batch_size),
            "--num-workers",
            str(num_workers),
            "--speed-profile",
            speed_profile,
            "--progress",
            progress,
            "--log-level",
            log_level,
            "--path-cache-policy",
            path_cache_policy,
            "--benchmark-json",
            str(benchmark_json),
            str(sample_mgf),
        ]
        cwd = PATHSEARCHER_DIR
    elif stage == "seqfiller":
        command = [
            sys.executable,
            "Inference_Sequence.py",
            "--batch-size",
            str(batch_size),
            "--num-workers",
            str(num_workers),
            "--speed-profile",
            speed_profile,
            "--progress",
            progress,
            "--log-level",
            log_level,
            "--benchmark-json",
            str(benchmark_json),
            str(sample_mgf),
            BASE_CANDIDATE_AMINO_ACIDS,
        ]
        cwd = SEQFILLER_DIR
    else:
        raise ValueError(stage)

    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        return {
            "status": "failed",
            "batch_size": batch_size,
            "command": command,
            "cwd": str(cwd),
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
    try:
        result = json.loads(benchmark_json.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "failed",
            "batch_size": batch_size,
            "command": command,
            "cwd": str(cwd),
            "returncode": completed.returncode,
            "stderr": f"benchmark JSON was not readable: {exc}",
        }
    result["status"] = "ok"
    result["batch_size"] = batch_size
    return result
