from __future__ import annotations

import json
import subprocess
import sys
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
    accepted: list[dict[str, Any]] = []
    for batch_size in candidates:
        path_result = _run_inference_benchmark(
            "pathsearcher",
            batch_size=batch_size,
            num_workers=base_settings.path_num_workers,
            sample_mgf=sample_mgf,
            speed_profile=speed_profile,
            progress=progress,
            log_level=log_level,
            path_cache_policy=path_cache_policy,
            tune_dir=tune_dir,
        )
        seq_result = _run_inference_benchmark(
            "seqfiller",
            batch_size=batch_size,
            num_workers=base_settings.seq_num_workers,
            sample_mgf=sample_mgf,
            speed_profile=speed_profile,
            progress=progress,
            log_level=log_level,
            path_cache_policy=path_cache_policy,
            tune_dir=tune_dir,
        )
        combined = {
            "batch_size": batch_size,
            "pathsearcher": path_result,
            "seqfiller": seq_result,
        }
        benchmark_results.append(combined)
        if path_result["status"] != "ok" or seq_result["status"] != "ok":
            continue
        peak_gib = max(path_result.get("peak_memory_gib", 0), seq_result.get("peak_memory_gib", 0))
        if total_memory_gib is not None and peak_gib > total_memory_gib * max_memory_frac:
            combined["rejected_reason"] = "peak memory exceeded max-memory-frac"
            continue
        accepted.append(combined)

    if not accepted:
        raise RNovAError("Tuning did not find a successful batch size; try a smaller sample or inspect .cache/rnova/tune")

    best = max(
        accepted,
        key=lambda result: min(
            result["pathsearcher"].get("spectra_per_second", 0),
            result["seqfiller"].get("spectra_per_second", 0),
        ),
    )
    tuned_settings = RuntimeSettings(
        speed_profile=speed_profile,
        progress=progress,
        log_level=log_level,
        path_cache_policy=path_cache_policy,
        path_batch_size=int(best["batch_size"]),
        seq_batch_size=int(best["batch_size"]),
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
            "returncode": completed.returncode,
            "stderr": f"benchmark JSON was not readable: {exc}",
        }
    result["status"] = "ok"
    result["batch_size"] = batch_size
    return result
