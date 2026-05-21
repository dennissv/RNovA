from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPEED_PROFILES = ("fast", "exact", "max")
PROGRESS_MODES = ("auto", "on", "off")
LOG_LEVELS = ("warning", "info", "debug")
PATH_CACHE_POLICIES = ("auto", "legacy")
TUNING_VERSION = 2
DEFAULT_TUNING_PATH = Path(__file__).resolve().parents[2] / ".cache" / "rnova" / "tuning.json"


@dataclass(frozen=True)
class RuntimeSettings:
    speed_profile: str
    progress: str
    log_level: str
    path_cache_policy: str
    path_batch_size: int
    path_num_workers: int
    seq_batch_size: int
    seq_num_workers: int
    null_workers: int
    source: str


def input_signature(input_dir: str | Path) -> str:
    input_path = Path(input_dir).expanduser().resolve()
    entries: list[dict[str, Any]] = []
    for path in sorted(input_path.glob("*.mgf")):
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return _stable_json_hash(entries)


def load_tuning(input_dir: str | Path, tuning_path: Path = DEFAULT_TUNING_PATH) -> dict[str, Any] | None:
    try:
        record = json.loads(tuning_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if record.get("version") != TUNING_VERSION:
        return None
    if record.get("input_signature") != input_signature(input_dir):
        return None
    settings = record.get("settings")
    if not isinstance(settings, dict):
        return None
    required = {"path_batch_size", "seq_batch_size", "path_num_workers", "seq_num_workers"}
    if not required.issubset(settings):
        return None
    return record


def save_tuning(
    input_dir: str | Path,
    *,
    settings: RuntimeSettings | dict[str, int],
    benchmark_results: list[dict[str, Any]],
    device: dict[str, Any] | None = None,
    sample_spectra: int | None = None,
    sample_summary: dict[str, Any] | None = None,
    max_memory_frac: float | None = None,
    tuning_path: Path = DEFAULT_TUNING_PATH,
) -> Path:
    if isinstance(settings, RuntimeSettings):
        settings_data = asdict(settings)
    else:
        settings_data = dict(settings)
    record = {
        "version": TUNING_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "input_signature": input_signature(input_dir),
        "settings": settings_data,
        "benchmark_results": benchmark_results,
        "device": device or {},
        "sample_spectra": sample_spectra,
        "sample_summary": sample_summary or {},
        "max_memory_frac": max_memory_frac,
    }
    tuning_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = tuning_path.with_suffix(f"{tuning_path.suffix}.tmp")
    tmp_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(tuning_path)
    return tuning_path


def resolve_runtime_settings(
    input_dir: str | Path,
    *,
    speed_profile: str = "fast",
    progress: str = "auto",
    log_level: str = "warning",
    path_cache_policy: str = "auto",
    path_batch_size: str | int | None = "auto",
    path_num_workers: str | int | None = "auto",
    seq_batch_size: str | int | None = "auto",
    seq_num_workers: str | int | None = "auto",
    null_workers: str | int | None = "auto",
    use_tuning: bool = True,
    tuning_path: Path = DEFAULT_TUNING_PATH,
) -> RuntimeSettings:
    _validate_choice("--speed-profile", speed_profile, SPEED_PROFILES)
    _validate_choice("--progress", progress, PROGRESS_MODES)
    _validate_choice("--log-level", log_level, LOG_LEVELS)
    _validate_choice("--path-cache-policy", path_cache_policy, PATH_CACHE_POLICIES)

    tuning = load_tuning(input_dir, tuning_path=tuning_path) if use_tuning else None
    tuning_settings = tuning.get("settings", {}) if tuning else {}
    total_memory_gib = cuda_total_memory_gib()
    default_batch = batch_size_for_gpu_memory(total_memory_gib)

    resolved_path_batch = _resolve_auto_int(
        "--path-batch-size",
        path_batch_size,
        tuned=tuning_settings.get("path_batch_size"),
        default=default_batch,
    )
    resolved_seq_batch = _resolve_auto_int(
        "--seq-batch-size",
        seq_batch_size,
        tuned=tuning_settings.get("seq_batch_size"),
        default=default_batch,
    )
    path_worker_default, seq_worker_default = default_worker_counts()
    resolved_path_workers = _resolve_auto_int(
        "--path-num-workers",
        path_num_workers,
        tuned=tuning_settings.get("path_num_workers"),
        default=path_worker_default,
        minimum=0,
    )
    resolved_seq_workers = _resolve_auto_int(
        "--seq-num-workers",
        seq_num_workers,
        tuned=tuning_settings.get("seq_num_workers"),
        default=seq_worker_default,
        minimum=0,
    )
    resolved_null_workers = _resolve_null_workers(null_workers, speed_profile=speed_profile)

    return RuntimeSettings(
        speed_profile=speed_profile,
        progress=progress,
        log_level=log_level,
        path_cache_policy=path_cache_policy,
        path_batch_size=resolved_path_batch,
        path_num_workers=resolved_path_workers,
        seq_batch_size=resolved_seq_batch,
        seq_num_workers=resolved_seq_workers,
        null_workers=resolved_null_workers,
        source="tuning-cache" if tuning else "auto",
    )


def batch_size_for_gpu_memory(total_memory_gib: float | None) -> int:
    if total_memory_gib is None:
        return 8
    if total_memory_gib < 10:
        return 4
    if total_memory_gib < 16:
        return 8
    if total_memory_gib < 24:
        return 16
    return 32


def default_worker_counts() -> tuple[int, int]:
    if is_wsl():
        return 0, 0
    if platform.system() == "Linux":
        return 2, 1
    return 0, 0


def is_wsl() -> bool:
    if platform.system() != "Linux":
        return False
    text = " ".join(
        part.lower()
        for part in (
            platform.release(),
            _read_text(Path("/proc/version")),
        )
    )
    return "microsoft" in text or "wsl" in text


def cuda_total_memory_gib() -> float | None:
    try:
        import torch
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
    except Exception:
        return None
    return float(props.total_memory) / 1024**3


def cuda_device_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "detail": str(exc)}
    try:
        if not torch.cuda.is_available():
            return {"available": False}
        props = torch.cuda.get_device_properties(0)
        return {
            "available": True,
            "name": props.name,
            "total_memory_gib": props.total_memory / 1024**3,
            "device_count": torch.cuda.device_count(),
        }
    except Exception as exc:
        return {"available": False, "detail": str(exc)}


def candidate_batch_sizes(total_memory_gib: float | None = None) -> list[int]:
    return [8, 16, 32, 64, 128]


def _resolve_auto_int(
    name: str,
    value: str | int | None,
    *,
    tuned: Any,
    default: int,
    minimum: int = 1,
) -> int:
    if value is None or value == "auto":
        if tuned is not None:
            try:
                tuned_value = int(tuned)
            except (TypeError, ValueError):
                tuned_value = default
            return _validate_int(name, tuned_value, minimum=minimum)
        return _validate_int(name, default, minimum=minimum)
    return _validate_int(name, value, minimum=minimum)


def _resolve_null_workers(value: str | int | None, *, speed_profile: str) -> int:
    if value is None or value == "auto":
        if speed_profile == "exact":
            return 1
        cpu_count = os.cpu_count() or 1
        return max(1, min(4, cpu_count // 2 or 1))
    return _validate_int("--null-workers", value, minimum=1)


def _validate_int(name: str, value: str | int, *, minimum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be 'auto' or an integer") from exc
    if parsed < minimum:
        if minimum == 1:
            raise ValueError(f"{name} must be greater than 0")
        raise ValueError(f"{name} must be greater than or equal to {minimum}")
    return parsed


def _validate_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        joined = ", ".join(choices)
        raise ValueError(f"{name} must be one of: {joined}")


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def _stable_json_hash(value: Any) -> str:
    import hashlib

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
