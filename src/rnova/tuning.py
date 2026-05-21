from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
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


def run_tuning(
    input_dir: str | Path,
    *,
    sample_spectra: int = 64,
    max_memory_frac: float = 0.85,
    speed_profile: str = "fast",
    path_cache_policy: str = "auto",
    progress: str = "auto",
    log_level: str = "warning",
    tuning_path: Path = DEFAULT_TUNING_PATH,
) -> Path:
    if sample_spectra <= 0:
        raise RNovAError("--sample-spectra must be greater than 0")
    if not 0 < max_memory_frac <= 1:
        raise RNovAError("--max-memory-frac must be greater than 0 and at most 1")

    input_path = Path(input_dir).expanduser().resolve()
    assert_checks_pass(collect_checks(input_path, mode="inference", strict_gpu=True))
    if not seqfiller_extension_files():
        raise RNovAError("SeqFiller extension is missing or incompatible. Run `uv run rnova setup`.")
    mgf_files = find_mgf_files(input_path)
    if not mgf_files:
        raise RNovAError(f"No .mgf files found in {input_path}")

    device = cuda_device_info()
    total_memory_gib = device.get("total_memory_gib") if device.get("available") else None
    candidates = candidate_batch_sizes(total_memory_gib)
    if not candidates:
        raise RNovAError("Could not choose candidate batch sizes for tuning")

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
    sampled = _write_sample_mgf(mgf_files, sample_mgf, sample_spectra)
    if sampled == 0:
        raise RNovAError("Tuning sample did not contain any spectra")

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
        max_memory_frac=max_memory_frac,
        tuning_path=tuning_path,
    )


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
) -> Path:
    failure_path = tune_dir / "tuning_failure.json"
    payload = {
        "input_dir": str(input_path),
        "sample_spectra": sampled,
        "device": device,
        "base_settings": asdict(base_settings),
        "max_memory_frac": max_memory_frac,
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
            speed = result.get("spectra_per_second", 0)
            rejected = result.get("rejected_reason")
            if rejected:
                lines.append(f"  batch {batch}: rejected ({rejected})")
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


def _write_sample_mgf(mgf_files: list[Path], output_path: Path, sample_spectra: int) -> int:
    blocks: list[MGFSpectrumBlock] = []
    for mgf_file in mgf_files:
        for block in read_mgf_blocks(mgf_file):
            blocks.append(block)
            if len(blocks) >= sample_spectra:
                break
        if len(blocks) >= sample_spectra:
            break

    with output_path.open("w") as handle:
        for block in blocks:
            for header_line in block.header_lines:
                handle.write(header_line if header_line.endswith("\n") else f"{header_line}\n")
            for peak in block.peaks:
                handle.write(f"{peak.moverz:.8f} {peak.intensity:.8f}\n")
            handle.write(block.end_line if block.end_line.endswith("\n") else f"{block.end_line}\n")
    return len(blocks)


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
