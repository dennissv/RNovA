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
    parser = argparse.ArgumentParser(description="Run RNovA SeqFiller inference.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override configs/train/train.yaml batch_size for this run",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="DataLoader worker count",
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
        help="Enable detailed inference diagnostics",
    )
    parser.add_argument(
        "--benchmark-json",
        help="Internal tuning hook: write elapsed/throughput/peak-memory metrics",
    )
    parser.add_argument("mgf_files", nargs="+", help="Input .mgf files")
    parser.add_argument("candidate_amino_acids", help="Semicolon-separated candidate amino-acid list")
    return parser.parse_args(argv)


def read_mgf(mgf_file):
    return _read_mgf(mgf_file, default_charge=2)


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
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for SeqFiller inference, but no CUDA device is visible")

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
    logger.debug("SeqFiller runtime: batch_size=%s, num_workers=%s", cfg.train.batch_size, args.num_workers)
    logger.debug("Initializing SeqFiller model")
    model = RNovA(cfg).to(local_rank)
    model.eval()
    logger.debug("Loading SeqFiller checkpoint")
    model_checkpoint = torch.load('save/rnova.pt',map_location={'cuda:0': f'cuda:{local_rank}'},weights_only=True)
    model.load_state_dict(model_checkpoint)
    logger.debug("SeqFiller checkpoint loaded")
    if args.speed_profile == "max":
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            logger.warning("torch.compile failed; continuing without it", exc_info=args.debug_inference)

    candidate_amino_acids = args.candidate_amino_acids.split(';')
    file_iter = tqdm(args.mgf_files, desc="SeqFiller files", unit="file", disable=not progress_enabled)
    for mgf_file in file_iter:
        logger.info("Start analysing %s", mgf_file)
        logger.debug("Parsing MGF %s", mgf_file)
        spectra = read_mgf(mgf_file)
        total_spectra += len(spectra)
        logger.info("Parsed %s spectra from %s", len(spectra), mgf_file)

        ds = RnovaDataset(cfg,spectra)
        collator = RnovaCollator(cfg)
        train_dl = DataLoader(
            ds,
            batch_size=cfg.train.batch_size,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        batch_count = len(train_dl)
        logger.info("Prepared %s SeqFiller batch(es)", batch_count)
        train_dl = DataPrefetcher(train_dl, local_rank)
        environment = Environment(cfg, model, train_dl, local_rank, candidate_amino_acids)

        with open(mgf_file[:-4]+'_rnova_denovo_seq.csv', 'w', newline='') as fw:
            writer = csv.writer(fw)
            writer.writerow(['title', 'sequence', 'score'])
            batch_iter = tqdm(
                environment,
                total=batch_count,
                desc=f"SeqFiller {Path(mgf_file).name}",
                unit="batch",
                leave=False,
                disable=not progress_enabled,
            )
            with torch.inference_mode():
                for results, results_score, titles in batch_iter:
                    for result, result_score, title in zip(results, results_score, titles):
                        result_str = []
                        for aa in result:
                            name = aa.amino_acid_name
                            # 只保留非空的修饰项
                            mods = [mod for mod in [aa.r_group_PTM, aa.n_terminal_PTM, aa.c_terminal_PTM] if mod]
                            if mods: name += f"[{'|'.join(mods)}]"
                            result_str.append(name)
                        result_str = ''.join(result_str)
                        result_score = ';'.join(f"{s:.4f}" for s in result_score)
                        writer.writerow([title, result_str, result_score])
        output_files += 1
        logger.info("Results saved to %s_rnova_denovo_seq.csv", mgf_file[:-4])

    elapsed = time.perf_counter() - started_at
    peak_allocated_gib = _peak_memory_allocated_gib(local_rank)
    peak_reserved_gib = _peak_memory_reserved_gib(local_rank)
    metrics = {
        "stage": "seqfiller",
        "elapsed_seconds": elapsed,
        "spectra": total_spectra,
        "spectra_per_second": total_spectra / elapsed if elapsed > 0 else 0.0,
        "peak_memory_gib": max(peak_allocated_gib, peak_reserved_gib),
        "peak_memory_allocated_gib": peak_allocated_gib,
        "peak_memory_reserved_gib": peak_reserved_gib,
        "batch_size": cfg.train.batch_size,
        "num_workers": args.num_workers,
        "speed_profile": args.speed_profile,
    }
    if args.benchmark_json:
        Path(args.benchmark_json).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    else:
        print(
            f"SeqFiller wrote {output_files} file(s) for {total_spectra} spectra. "
            f"Peak CUDA reserved {peak_reserved_gib:.2f} GiB, allocated {peak_allocated_gib:.2f} GiB."
        )


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


def _peak_memory_allocated_gib(device: int) -> float:
    try:
        return torch.cuda.max_memory_allocated(device) / 1024**3
    except RuntimeError as exc:
        logger.debug("Could not read CUDA allocated peak memory for device %s: %s", device, exc)
        return 0.0


def _peak_memory_reserved_gib(device: int) -> float:
    try:
        return torch.cuda.max_memory_reserved(device) / 1024**3
    except RuntimeError as exc:
        logger.debug("Could not read CUDA reserved peak memory for device %s: %s", device, exc)
        return 0.0


if __name__ == "__main__":
    main()
