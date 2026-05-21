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
from RNovA_SeqFiller_Inference.utils.BasicClass import ResidueOnlyPeptide

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
    parser = argparse.ArgumentParser(description="Compute FDR stage 2 for sequence results.")
    parser.add_argument("target_directory", help="Folder with target *_rnova_denovo_seq.csv files")
    parser.add_argument("decoy_directory", help="Folder with decoy *.decoy_random0.40_rnova_denovo_seq.csv files")
    parser.add_argument(
        "--pattern",
        default="*rnova_denovo_seq.csv",
        help="Glob pattern for decoy CSV files",
    )
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
    all_csv_files = sorted(glob(os.path.join(decoy_directory, args.pattern)))
    if not all_csv_files:
        raise FDRComputationError(f"No decoy sequence CSV files found in {decoy_directory}")

    for ff in all_csv_files:
        logger.info(ff)
        sample = ff.replace('.mgf.decoy_random0.40_rnova_denovo_seq.csv', '')
        sample = sample.replace(str(decoy_directory) + os.sep, '')
        decoy_df = pd.read_csv(ff)
        target_csv = target_directory / f"{sample}_rnova_denovo_seq.csv"
        if not target_csv.exists():
            raise FDRComputationError(f"Missing target sequence CSV for sample {sample}: {target_csv}")
        target_df = pd.read_csv(str(target_csv))
        require_columns(target_df.columns, ("title", "sequence", "score"), source=str(target_csv))
        require_columns(decoy_df.columns, ("title", "sequence", "score"), source=str(ff))
        require_equal_lengths(len(target_df), len(decoy_df), source=sample)
        tagged = []
        for idx in range(len(decoy_df)):
            decoy = decoy_df.iloc[idx]
            target = target_df.iloc[idx]
            t_score = parse_score_series(
                target['score'],
                source=f"{target_csv} row {idx}",
                ignore_negative_inf=True,
            )
            d_score = parse_score_series(
                decoy['score'],
                source=f"{ff} row {idx}",
                ignore_negative_inf=True,
            )
            if sum(t_score) > sum(d_score):
                tagged += [(i, 0) for i in t_score]
            else:
                tagged += [(i, 1) for i in d_score]
        threshold_score, fdr, threshold_item = find_exact_fdr_threshold(
            tagged,
            label=f"{sample} amino-acid-level FDR",
        )
        logger.info("Amino Acid level fdr %s model score threshold %s", fdr, threshold_item)
        # whole path fdr
        tagged = []
        for idx in range(len(decoy_df)):
            decoy = decoy_df.iloc[idx]
            target = target_df.iloc[idx]
            t_score = parse_score_series(
                target['score'],
                source=f"{target_csv} row {idx}",
                ignore_negative_inf=True,
            )
            t_sum = sum([i for i in t_score if i > threshold_score])
            d_score = parse_score_series(
                decoy['score'],
                source=f"{ff} row {idx}",
                ignore_negative_inf=True,
            )
            d_sum = sum([i for i in d_score if i > threshold_score])
            if t_sum > d_sum:
                tagged.append((t_sum, 0))
            else:
                tagged.append((d_sum, 1))
        threshold_path_score, _, _ = find_exact_fdr_threshold(
            tagged,
            label=f"{sample} sequence path-level FDR",
        )
        path_fdr_result = target_directory / f"{sample}_rnova_denovo_path_001fdr.csv"
        if not path_fdr_result.exists():
            raise FDRComputationError(f"Missing path FDR result for sample {sample}: {path_fdr_result}")
        path_fdr_result = pd.read_csv(path_fdr_result)
        require_columns(
            path_fdr_result.columns,
            ("scan", "node_mass", "node_score"),
            source=str(target_directory / f"{sample}_rnova_denovo_path_001fdr.csv"),
        )
        path_fdr_result = {s:np.array(list(map(float,node.split(';'))))[1:] for s, node in zip(path_fdr_result['scan'],path_fdr_result['node_mass'])}
        # print(len(path_fdr_result))
        new_seq_list = {}
        new_score_list = {}

        for _, (i,seq, score) in target_df.iterrows():
            # seq = seq.replace('C[UniMod:4]','C|UniMod:4')
            # seq = seq.replace('M[UniMod:35]','M|UniMod:35')
            # seq = seq.replace('D[UniMod:734]','D|UniMod:734')
            # seq = seq.replace('E[UniMod:734]','E|UniMod:734')
            # print(i, path_fdr_result.keys())
            # print(stop)
            if i in path_fdr_result.keys():
                # print(seq)
                merge_flag = (np.abs(ResidueOnlyPeptide(seq).prefix_mass[None,:]-path_fdr_result[i][:,None])<0.02).any(0)
                # print(merge_flag)
                seq = ResidueOnlyPeptide(seq).sequence_residue_seq
                score = np.array(list(map(float,score.split(';'))))

                new_seq = []
                new_score = []
                seq_temp = []
                score_temp = 0
                for a,b,c in zip(merge_flag, seq, score):
                    if a:
                        seq_temp.append(b if len(b)==1 else f'{b[0]}[{b[2:]}]')
                        score_temp += c
                        score_temp = score_temp/len(seq_temp)
                        new_seq.append(seq_temp)
                        new_score.append(score_temp)
                        seq_temp = []
                        score_temp = 0
                    else:
                        seq_temp.append(b if len(b)==1 else f'{b[0]}[{b[2:]}]')
                        score_temp += c
                if not merge_flag.any():
                    new_seq.append(seq_temp)
                    new_score.append(score_temp)
                new_seq_list[i] = new_seq
                new_score_list[i] = new_score
            else:
                seq = ResidueOnlyPeptide(seq).sequence_residue_seq
                score = np.array(list(map(float,score.split(';'))))
                new_seq_list[i] = [[s if len(s)==1 else f'{s[0]}[{s[2:]}]' for s in seq]]
                new_score_list[i] = [score.sum()/len(seq)]

        output_path = target_directory / f"{sample}_rnova_denovo_seq001fdr.csv"
        with open(output_path, 'w', newline='') as fw:
            writer = csv.writer(fw)
            writer.writerow(['title', 'sequence', 'score'])
            for i in new_seq_list.keys():
                seq_per_psm, score_per_psm = new_seq_list[i], new_score_list[i]
                seq_temp, score_temp = [], []
                scores = 0
                aa_count = 0

                for seq_block, score in zip(seq_per_psm, score_per_psm):
                    pep = ''.join(seq_block)
                    # print(seq_block, score)
                    if score<threshold_score:
                        seq_mass = ResidueOnlyPeptide(''.join(seq_block)).total_residue_mass
                        seq_temp.append(f'{seq_mass}')
                        score_temp.append(f'{score}')
                    else:
                        seq_temp.append("".join(seq_block))
                        score_temp.append(f"{score}")
                        aa_count += len(''.join(seq_block))
                        scores += score
                if scores < threshold_path_score:
                    continue
                writer.writerow([i, ' '.join(seq_temp), ';'.join(score_temp)])

if __name__ == "__main__":
    try:
        main()
    except FDRComputationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
