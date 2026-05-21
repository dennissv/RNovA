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

def read_mgf(mgf_file):
    return _read_mgf(mgf_file, default_charge=2)

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    if len(sys.argv) < 3:
        raise SystemExit("SeqFiller inference requires at least one MGF file and a candidate amino-acid list")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for SeqFiller inference, but no CUDA device is visible")

    with initialize(config_path="configs", version_base=None): cfg = compose(config_name="config")
    local_rank = 0
    torch.cuda.set_device(local_rank)
    model = RNovA(cfg).to(local_rank)
    model_checkpoint = torch.load('save/rnova.pt',map_location={'cuda:0': f'cuda:{local_rank}'},weights_only=True)
    model.load_state_dict(model_checkpoint)

    mgf_files, candidate_amino_acids = sys.argv[1:-1], sys.argv[-1]
    candidate_amino_acids = candidate_amino_acids.split(';')
    for mgf_file in mgf_files:
        logger.info("Start analysing %s", mgf_file)
        spectra = read_mgf(mgf_file)

        ds = RnovaDataset(cfg,spectra)
        collator = RnovaCollator(cfg)
        train_dl = DataLoader(ds,batch_size=cfg.train.batch_size,collate_fn=collator,num_workers=1,pin_memory=True)
        train_dl = DataPrefetcher(train_dl, local_rank)
        environment = Environment(cfg, model, train_dl, local_rank, candidate_amino_acids)

        with open(mgf_file[:-4]+'_rnova_denovo_seq.csv', 'w', newline='') as fw:
            writer = csv.writer(fw)
            writer.writerow(['title', 'sequence', 'score'])
            for results, results_score, titles in environment:
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
