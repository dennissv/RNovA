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
    return _read_mgf(mgf_file)

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    mgf_files = sys.argv[1:]
    if not mgf_files:
        raise SystemExit("No MGF files supplied to PathSearcher inference")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for PathSearcher inference, but no CUDA device is visible")

    with initialize(config_path="configs", version_base=None): cfg = compose(config_name="config")
    local_rank = 0
    torch.cuda.set_device(local_rank)
    model = RNovA(cfg).to(local_rank)
    model.load_state_dict(torch.load('save/rnova.pt',map_location={'cuda:0': f'cuda:{local_rank}'},weights_only=True))

    for mgf_file in mgf_files:
        logger.info("Start analysing %s", mgf_file)
        write_file_name = mgf_file[:-4]+'_rnova_denovo_path.csv'

        spectra = read_mgf(mgf_file)
        ds = RnovaDataset(cfg,spectra)
        collator = RnovaCollator(cfg)
        inference_dl = DataLoader(ds,batch_size=cfg.train.batch_size,collate_fn=collator,num_workers=2,pin_memory=True)
        inference_dl = DataPrefetcher(inference_dl,local_rank)
        inference_dl = Environment(cfg, model, inference_dl, local_rank)
        with open(write_file_name, 'w', newline='') as fw:
            writer = csv.writer(fw)
            writer.writerow(['scan', 'node_mass', 'score', 'node_class'])
            for node_seq, score_seq, class_seq, title, _ in inference_dl:
                for node, s, c, t in zip(node_seq, score_seq, class_seq, title):
                    path = np.array2string(node,separator=';',formatter={'float_kind': lambda x: f"{x:.4f}"},max_line_width=999999).strip('[]')
                    path_score = np.array2string(s,separator=';',formatter={'float_kind': lambda x: f"{x:.4f}"},max_line_width=999999).strip('[]')
                    class_list = np.array2string(c,separator=';',max_line_width=999999).strip('[]')
                    writer.writerow([t, path, path_score, class_list])
        logger.info("Results saved to %s_rnova_denovo_path.csv", mgf_file[:-4])

if __name__ == "__main__":
    main()
