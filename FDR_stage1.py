# merge all csv files in RNC-seq
import argparse
import csv
import logging
import os
import sys
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

try:
    from rnova.fdr import (
        FDRComputationError,
        find_exact_fdr_threshold,
        parse_score_series,
        require_columns,
        require_equal_lengths,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from rnova.fdr import (
        FDRComputationError,
        find_exact_fdr_threshold,
        parse_score_series,
        require_columns,
        require_equal_lengths,
    )

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Compute FDR stage 1 from target/decoy CSVs.")
    parser.add_argument("target_directory", help="Folder with target *_rnova_denovo_path.csv files")
    parser.add_argument("decoy_directory", help="Folder with decoy *_rnova_denovo_path.csv files")

    return parser.parse_args()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    args = parse_args()
    target_directory = Path(args.target_directory)
    decoy_directory = Path(args.decoy_directory)
    output_directory = Path(args.target_directory)

    all_csv_files = sorted(glob(os.path.join(target_directory, '*rnova_denovo_path.csv')))
    if not all_csv_files:
        raise FDRComputationError(f"No target path CSV files found in {target_directory}")

    for ff in all_csv_files:
        logger.info(ff)
        sample = Path(ff).name.replace('_rnova_denovo_path.csv', '')
        logger.info(sample)
        target_df = pd.read_csv(ff)
        decoy_csv = decoy_directory / f"{sample}.mgf.decoy_random0.40_rnova_denovo_path.csv"
        if not decoy_csv.exists():
            raise FDRComputationError(f"Missing decoy path CSV for sample {sample}: {decoy_csv}")
        decoy_df = pd.read_csv(decoy_csv)
        require_columns(target_df.columns, ("scan", "node_mass", "score", "node_class"), source=str(ff))
        require_columns(decoy_df.columns, ("scan", "node_mass", "score", "node_class"), source=str(decoy_csv))
        require_equal_lengths(len(target_df), len(decoy_df), source=sample)
        tagged = []
        for idx in range(len(decoy_df)):
            decoy = decoy_df.iloc[idx]
            target = target_df.iloc[idx]
            t_score = parse_score_series(target['score'], source=f"{ff} row {idx}", trim_edges=True)
            d_score = parse_score_series(decoy['score'], source=f"{decoy_csv} row {idx}", trim_edges=True)
            if sum(t_score) > sum(d_score):
                tagged += [(i, 0) for i in t_score]
            else:
                tagged += [(i, 1) for i in d_score]
        threshold_score, fdr, threshold_item = find_exact_fdr_threshold(
            tagged,
            label=f"{sample} node-level FDR",
        )
        logger.info("node-level fdr %s q_value threshold %s", fdr, threshold_item)
        # whole path fdr
        tagged = []
        for idx in range(len(decoy_df)):
            decoy = decoy_df.iloc[idx]
            target = target_df.iloc[idx]
            t_score = parse_score_series(target['score'], source=f"{ff} row {idx}", trim_edges=True)
            t_sum = sum([i for i in t_score if i > threshold_score])
            d_score = parse_score_series(decoy['score'], source=f"{decoy_csv} row {idx}", trim_edges=True)
            d_sum = sum([i for i in d_score if i > threshold_score])
            if t_sum > d_sum:
                tagged.append((t_sum, 0))
            else:
                tagged.append((d_sum, 1))
        threshold_path_score, _, _ = find_exact_fdr_threshold(
            tagged,
            label=f"{sample} path-level FDR",
        )
        output_path = output_directory / f"{sample}_rnova_denovo_path_001fdr.csv"
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['scan', 'node_mass', 'node_score'])
            for idx, row in target_df.iterrows():
                node_mass = np.fromstring(str(row['node_mass']), sep=';')
                node_score = np.fromstring(str(row['score']), sep=';')
                if len(node_mass) != len(node_score):
                    raise FDRComputationError(
                        f"{ff} row {idx} has mismatched node_mass/score lengths: "
                        f"{len(node_mass)} != {len(node_score)}"
                    )
                mask = node_score>threshold_score
                node_mass, node_score = node_mass[mask], node_score[mask]
                if sum(node_score[1:-1]) < threshold_path_score:
                    continue
                node_mass = np.array2string(node_mass,separator=';',formatter={'float_kind': lambda x: f"{x:.4f}"},max_line_width=999999).strip('[]')
                node_score = np.array2string(node_score,separator=';',formatter={'float_kind': lambda x: f"{x:.4f}"},max_line_width=999999).strip('[]')
                # if (~mask).all() or mask.sum()<3: continue
                writer.writerow([row['scan'], node_mass, node_score])

if __name__ == "__main__":
    try:
        main()
    except FDRComputationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
