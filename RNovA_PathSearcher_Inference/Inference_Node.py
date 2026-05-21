import argparse
import csv
import json
import logging
import sys
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

try:
    from model import RNovA
    from data import RnovaDataset, RnovaCollator, DataPrefetcher, Environment
except ModuleNotFoundError as exc:
    if exc.name == "flash_attn":
        raise SystemExit(
            "flash-attn is required for RNovA inference. Install the inference/flash "
            "extras on a compatible Linux CUDA host before running real inference."
        ) from exc
    raise

from hydra import initialize, compose

try:
    from rnova.mgf import read_mgf as _read_mgf
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from rnova.mgf import read_mgf as _read_mgf

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run RNovA PathSearcher inference.")
    parser.add_argument("mgf_files", nargs="+", help="Input .mgf files")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override configs/train/train.yaml batch_size for this run",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker count",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Log decoder progress every N steps inside a batch",
    )
    parser.add_argument(
        "--speed-profile",
        choices=("fast", "exact", "max"),
        default="fast",
        help="Runtime speed profile",
    )
    parser.add_argument(
        "--progress",
        choices=("auto", "on", "off"),
        default="auto",
        help="Progress bar behavior",
    )
    parser.add_argument(
        "--log-level",
        choices=("warning", "info", "debug"),
        default="warning",
        help="Log verbosity",
    )
    parser.add_argument(
        "--debug-inference",
        action="store_true",
        help="Enable detailed encoder/decoder/cache diagnostics",
    )
    parser.add_argument(
        "--path-cache-policy",
        choices=("auto", "legacy"),
        default="auto",
        help="Decoder cache allocation policy",
    )
    parser.add_argument(
        "--benchmark-json",
        help="Internal tuning hook: write elapsed/throughput/peak-memory metrics",
    )
    return parser.parse_args(argv)


def read_mgf(mgf_file):
    return _read_mgf(mgf_file)


def main(argv=None):
    args = parse_args(argv)
    log_level = "debug" if args.debug_inference else args.log_level
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    if args.batch_size is not None and args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be greater than or equal to 0")
    if args.progress_interval <= 0:
        raise SystemExit("--progress-interval must be greater than 0")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for PathSearcher inference, but no CUDA device is visible")

    _apply_speed_profile(args.speed_profile)
    progress_enabled = _progress_enabled(args.progress)
    started_at = time.perf_counter()
    total_spectra = 0
    output_files = 0

    with initialize(config_path="configs", version_base=None): cfg = compose(config_name="config")
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    local_rank = 0
    try:
        torch.cuda.set_device(local_rank)
    except RuntimeError as exc:
        raise SystemExit(f"Could not select CUDA device {local_rank}: {exc}") from exc
    _reset_peak_memory_stats(local_rank)
    device_props = torch.cuda.get_device_properties(local_rank)
    logger.debug(
        "CUDA device %s: %s, %.1f GiB total memory",
        local_rank,
        device_props.name,
        device_props.total_memory / 1024**3,
    )
    logger.debug(
        "PathSearcher runtime: batch_size=%s, num_workers=%s, progress_interval=%s",
        cfg.train.batch_size,
        args.num_workers,
        args.progress_interval,
    )
    logger.debug("Initializing PathSearcher model")
    model = RNovA(cfg).to(local_rank)
    model.eval()
    logger.debug("Loading PathSearcher checkpoint")
    model.load_state_dict(torch.load('save/rnova.pt',map_location={'cuda:0': f'cuda:{local_rank}'},weights_only=True))
    logger.debug("PathSearcher checkpoint loaded")
    if args.speed_profile == "max":
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            logger.warning("torch.compile failed; continuing without it", exc_info=args.debug_inference)

    file_iter = tqdm(args.mgf_files, desc="PathSearcher files", unit="file", disable=not progress_enabled)
    for mgf_file in file_iter:
        logger.info("Start analysing %s", mgf_file)
        write_file_name = mgf_file[:-4]+'_rnova_denovo_path.csv'

        logger.debug("Parsing MGF %s", mgf_file)
        spectra = read_mgf(mgf_file)
        total_spectra += len(spectra)
        logger.info("Parsed %s spectra from %s", len(spectra), mgf_file)
        ds = RnovaDataset(cfg,spectra)
        collator = RnovaCollator(cfg)
        inference_dl = DataLoader(
            ds,
            batch_size=cfg.train.batch_size,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        batch_count = len(inference_dl)
        logger.info("Prepared %s PathSearcher batch(es)", batch_count)
        debug_logger = logger if args.debug_inference else None
        inference_dl = DataPrefetcher(inference_dl,local_rank,logger=debug_logger)
        inference_dl = Environment(
            cfg,
            model,
            inference_dl,
            local_rank,
            logger=debug_logger,
            progress_interval=args.progress_interval,
            cache_policy=args.path_cache_policy,
        )
        with open(write_file_name, 'w', newline='') as fw:
            writer = csv.writer(fw)
            writer.writerow(['scan', 'node_mass', 'score', 'node_class'])
            batch_iter = tqdm(
                inference_dl,
                total=batch_count,
                desc=f"PathSearcher {Path(mgf_file).name}",
                unit="batch",
                leave=False,
                disable=not progress_enabled,
            )
            with torch.inference_mode():
                for node_seq, score_seq, class_seq, title, _ in batch_iter:
                    for node, s, c, t in zip(node_seq, score_seq, class_seq, title):
                        path = np.array2string(node,separator=';',formatter={'float_kind': lambda x: f"{x:.4f}"},max_line_width=999999).strip('[]')
                        path_score = np.array2string(s,separator=';',formatter={'float_kind': lambda x: f"{x:.4f}"},max_line_width=999999).strip('[]')
                        class_list = np.array2string(c,separator=';',max_line_width=999999).strip('[]')
                        writer.writerow([t, path, path_score, class_list])
        output_files += 1
        logger.info("Results saved to %s_rnova_denovo_path.csv", mgf_file[:-4])

    elapsed = time.perf_counter() - started_at
    metrics = {
        "stage": "pathsearcher",
        "elapsed_seconds": elapsed,
        "spectra": total_spectra,
        "spectra_per_second": total_spectra / elapsed if elapsed > 0 else 0.0,
        "peak_memory_gib": _peak_memory_gib(local_rank),
        "batch_size": cfg.train.batch_size,
        "num_workers": args.num_workers,
        "speed_profile": args.speed_profile,
        "path_cache_policy": args.path_cache_policy,
    }
    if args.benchmark_json:
        Path(args.benchmark_json).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    else:
        print(f"PathSearcher wrote {output_files} file(s) for {total_spectra} spectra.")


def _progress_enabled(progress: str) -> bool:
    if progress == "on":
        return True
    if progress == "off":
        return False
    return sys.stderr.isatty()


def _apply_speed_profile(profile: str) -> None:
    if profile == "exact":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _reset_peak_memory_stats(device: int) -> None:
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except RuntimeError as exc:
        logger.debug("Could not reset CUDA peak memory stats for device %s: %s", device, exc)


def _peak_memory_gib(device: int) -> float:
    try:
        return torch.cuda.max_memory_allocated(device) / 1024**3
    except RuntimeError as exc:
        logger.debug("Could not read CUDA peak memory stats for device %s: %s", device, exc)
        return 0.0


if __name__ == "__main__":
    main()
