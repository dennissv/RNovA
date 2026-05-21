import argparse
import csv
import logging
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

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
    return parser.parse_args(argv)


def read_mgf(mgf_file):
    return _read_mgf(mgf_file)


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    if args.batch_size is not None and args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be greater than or equal to 0")
    if args.progress_interval <= 0:
        raise SystemExit("--progress-interval must be greater than 0")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for PathSearcher inference, but no CUDA device is visible")

    with initialize(config_path="configs", version_base=None): cfg = compose(config_name="config")
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    local_rank = 0
    torch.cuda.set_device(local_rank)
    device_props = torch.cuda.get_device_properties(local_rank)
    logger.info(
        "CUDA device %s: %s, %.1f GiB total memory",
        local_rank,
        device_props.name,
        device_props.total_memory / 1024**3,
    )
    logger.info(
        "PathSearcher runtime: batch_size=%s, num_workers=%s, progress_interval=%s",
        cfg.train.batch_size,
        args.num_workers,
        args.progress_interval,
    )
    logger.info("Initializing PathSearcher model")
    model = RNovA(cfg).to(local_rank)
    model.eval()
    logger.info("Loading PathSearcher checkpoint")
    model.load_state_dict(torch.load('save/rnova.pt',map_location={'cuda:0': f'cuda:{local_rank}'},weights_only=True))
    logger.info("PathSearcher checkpoint loaded")

    for mgf_file in args.mgf_files:
        logger.info("Start analysing %s", mgf_file)
        write_file_name = mgf_file[:-4]+'_rnova_denovo_path.csv'

        logger.info("Parsing MGF %s", mgf_file)
        spectra = read_mgf(mgf_file)
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
        inference_dl = DataPrefetcher(inference_dl,local_rank,logger=logger)
        inference_dl = Environment(
            cfg,
            model,
            inference_dl,
            local_rank,
            logger=logger,
            progress_interval=args.progress_interval,
        )
        with open(write_file_name, 'w', newline='') as fw:
            writer = csv.writer(fw)
            writer.writerow(['scan', 'node_mass', 'score', 'node_class'])
            with torch.no_grad():
                for batch_index, (node_seq, score_seq, class_seq, title, _) in enumerate(inference_dl, start=1):
                    logger.info("Writing PathSearcher batch %s/%s", batch_index, batch_count)
                    for node, s, c, t in zip(node_seq, score_seq, class_seq, title):
                        path = np.array2string(node,separator=';',formatter={'float_kind': lambda x: f"{x:.4f}"},max_line_width=999999).strip('[]')
                        path_score = np.array2string(s,separator=';',formatter={'float_kind': lambda x: f"{x:.4f}"},max_line_width=999999).strip('[]')
                        class_list = np.array2string(c,separator=';',max_line_width=999999).strip('[]')
                        writer.writerow([t, path, path_score, class_list])
        logger.info("Results saved to %s_rnova_denovo_path.csv", mgf_file[:-4])

if __name__ == "__main__":
    main()
