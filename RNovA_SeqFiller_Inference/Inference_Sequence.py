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
    parser.add_argument("mgf_files", nargs="+", help="Input .mgf files")
    parser.add_argument("candidate_amino_acids", help="Semicolon-separated candidate amino-acid list")
    return parser.parse_args(argv)


def read_mgf(mgf_file):
    return _read_mgf(mgf_file, default_charge=2)


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
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for SeqFiller inference, but no CUDA device is visible")

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
    logger.info("SeqFiller runtime: batch_size=%s, num_workers=%s", cfg.train.batch_size, args.num_workers)
    logger.info("Initializing SeqFiller model")
    model = RNovA(cfg).to(local_rank)
    model.eval()
    logger.info("Loading SeqFiller checkpoint")
    model_checkpoint = torch.load('save/rnova.pt',map_location={'cuda:0': f'cuda:{local_rank}'},weights_only=True)
    model.load_state_dict(model_checkpoint)
    logger.info("SeqFiller checkpoint loaded")

    candidate_amino_acids = args.candidate_amino_acids.split(';')
    for mgf_file in args.mgf_files:
        logger.info("Start analysing %s", mgf_file)
        logger.info("Parsing MGF %s", mgf_file)
        spectra = read_mgf(mgf_file)
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
            with torch.no_grad():
                for batch_index, (results, results_score, titles) in enumerate(environment, start=1):
                    logger.info("Writing SeqFiller batch %s/%s", batch_index, batch_count)
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
        logger.info("Results saved to %s_rnova_denovo_seq.csv", mgf_file[:-4])

if __name__ == "__main__":
    main()
