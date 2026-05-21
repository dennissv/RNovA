import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
import re
from glob import glob
from pathlib import Path

from src.alignment import *
from src.fill_PTM import (
    UniModMassIndex,
    delta_to_unimod_candidates,
    load_unimod,
    parse_peptide_mods,
)
from src.DBSCAN1D import DBSCAN1D
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def build_cluster(
    in_csv: pd.DataFrame,
    *,
    cache_dir: Path | None = None,
    refresh_cache: bool = False,
    null_workers: int = 1,
    show_progress: bool = True,
):
    import numba
    from src.clustering import cdhit_style_cluster_numba_masstag

    temp = in_csv
    nodes_mass = numba.typed.List()
    for s in temp['node_mass']:
        # 假设 s 形如 "m0;m1;...;mk"
        t = np.fromstring(s, sep=';', dtype=np.float32)
        # t[1:] - t[:-1] 是相邻 mass 的差，按你原先的写法保留
        diffs = t[1:] - t[:-1]
        nodes_mass.append(diffs.astype(np.float32))

    logger.info("Total peptides: %s", len(nodes_mass))

    null_params = {
        "min_ngram": 6,
        "max_ngram": 15,
        "target_per_L": 50_000,
        "random_state": 42,
    }
    null_model = _load_or_build_null_distribution(
        temp,
        nodes_mass,
        null_params=null_params,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
        null_workers=null_workers,
        show_progress=show_progress,
    )
    clusters, reps = cdhit_style_cluster_numba_masstag(
        nodes_mass,
        null_model,
        p_thresh=1e-4,
        sim_threshold=0.7,
        L_min_use=8,
        show_progress=show_progress,
    )

    cluster_sizes = [len(c) for c in clusters]

    clusters_filt = [c for c in clusters if len(c) > 1]
    reps_filt = [r for r, c in zip(reps, clusters) if len(c) > 1]
    cluster_sizes_np = np.array([len(c) for c in clusters_filt], dtype=int)

    logger.info("Num clusters (size>1): %s", len(clusters_filt))
    if cluster_sizes_np.size > 0:
        logger.info("Cluster size stats: min=%s, max=%s, median=%s, sum=%s",
            cluster_sizes_np.min(),
            cluster_sizes_np.max(),
            np.median(cluster_sizes_np),
            cluster_sizes_np.sum(),
        )
    else:
        logger.info("All clusters are singletons.")

    return nodes_mass, clusters_filt


def _load_or_build_null_distribution(
    temp: pd.DataFrame,
    nodes_mass,
    *,
    null_params: dict,
    cache_dir: Path | None,
    refresh_cache: bool,
    null_workers: int,
    show_progress: bool,
):
    from src.clustering import nw_masstag_numba
    from src.null_background_ngram import build_null_distribution

    cache_path = _null_distribution_cache_path(cache_dir, temp, null_params) if cache_dir else None
    if cache_path is not None and cache_path.exists() and not refresh_cache:
        logger.info("Loading cached null distribution from %s", cache_path)
        with cache_path.open("rb") as handle:
            return pickle.load(handle)

    null_model = build_null_distribution(
        nodes_mass,
        score_fn=nw_masstag_numba,
        min_ngram=null_params["min_ngram"],
        max_ngram=null_params["max_ngram"],
        target_per_L=null_params["target_per_L"],
        random_state=null_params["random_state"],
        null_workers=null_workers,
        show_progress=show_progress,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(null_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
        logger.info("Cached null distribution at %s", cache_path)
    return null_model


def _null_distribution_cache_path(cache_dir: Path | None, temp: pd.DataFrame, null_params: dict) -> Path | None:
    if cache_dir is None:
        return None
    digest = hashlib.sha256(json.dumps(null_params, sort_keys=True).encode())
    for value in temp["node_mass"].astype(str):
        digest.update(b"\0")
        digest.update(value.encode())
    return cache_dir / f"null_distribution_{digest.hexdigest()[:20]}.pkl"


def masstag_to_delta_mass(nodes_mass, clusters_filt, temp):
    result = pd.DataFrame(columns=temp.columns)
    result["raw_sequence"] = []
    result["filled_sequence"] = []
    result["cluster_center"] = []

    for cluster in clusters_filt:
        token_seqs = []
        for idx in cluster:
            toks = masses_to_tokens_with_list(
                nodes_mass[idx],
                aa_list,
                aa_tol=0.1,
            )
            token_seqs.append(toks)
        center_idx = cluster[0]
        center = token_seqs[0]
        aligned_token_seqs = []
        for toks in token_seqs:
            aln_a, aln_b = nw_tokens(center, toks)
            aligned_token_seqs.append(aln_b)
        consensus = build_consensus_bases(aligned_token_seqs)
        filled = single_aa_fill(
            aligned_token_seqs,
            consensus,
            mass_zero_tol=0.02,
            single_aa_mass_tol=0.5,
            max_abs_delta=150.0,
        )

        filled2 = collapsed_block_fill(
            filled,
            block_mass_tol=0.5,
        )

        ungapped = ["".join(tok for tok in seq if tok != "-") for seq in filled2]
        raw = ["".join(tok for tok in seq if tok != "-") for seq in token_seqs]

        temp_psm = temp.iloc[cluster].copy()
        temp_psm["raw_sequence"] = raw
        temp_psm["filled_sequence"] = ungapped
        temp_psm["cluster_center"] = int(center_idx)
        result = pd.concat([result, temp_psm])

    result = result.reset_index()
    return result

def topk_ptm_annotation(filled_result, k=3):
    ptm_freq = dict()
    for row in filled_result.itertuples():
        pep = row.filled_sequence
        ptms = parse_peptide_mods(pep)
        for p in ptms:
            if p['delta_mass'] is not None:
                ptm = p['residue']+'['+str(p['delta_mass']) + ']'
                if ptm in ptm_freq.keys():
                    ptm_freq[ptm] += 1
                else:
                    ptm_freq[ptm] = 1
    return ptm_freq


def cluster_ptm_pairs(ptm_counts, eps=0.02, min_samples=1, mass_precision=3):
    grouped = {}
    residue_order = []
    leftovers = {}
    ptm_re = re.compile(r"^([A-Za-z])\[(.+)\]$")

    for ptm, count in ptm_counts.items():
        match = ptm_re.match(ptm)
        if not match:
            leftovers[ptm] = leftovers.get(ptm, 0) + count
            continue
        residue = match.group(1)
        try:
            mass = float(match.group(2))
        except ValueError:
            leftovers[ptm] = leftovers.get(ptm, 0) + count
            continue
        if residue not in grouped:
            grouped[residue] = {"masses": [], "counts": []}
            residue_order.append(residue)
        grouped[residue]["masses"].append(mass)
        grouped[residue]["counts"].append(count)

    clustered = {}
    for residue in residue_order:
        masses = np.asarray(grouped[residue]["masses"], dtype=float)
        counts = np.asarray(grouped[residue]["counts"], dtype=float)
        if masses.size == 0:
            continue
        db = DBSCAN1D(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(masses)
        for label in np.unique(labels):
            if label < 0:
                continue
            mask = labels == label
            center = masses[mask].mean()
            key = f"{residue}[{center:.{mass_precision}f}]"
            clustered[key] = clustered.get(key, 0) + counts[mask].sum()

    for ptm, count in leftovers.items():
        clustered[ptm] = clustered.get(ptm, 0) + count

    return clustered


def delta_mass_to_ptm(filled_result: pd.DataFrame, index: UniModMassIndex, k, tol=0.01):
    df_ptm = pd.DataFrame(columns=filled_result.columns.tolist() + ["mod_residue", "mod_name"])
    ptm_freq = dict()
    for row in filled_result.itertuples():
        pep = row.filled_sequence
        mods_residue = []
        mods_name = []
        seqfiller_ptms = []
        for mod in parse_peptide_mods(pep):
            if mod["unimod_id"] is not None:
                continue
            candidates = delta_to_unimod_candidates(
                index,
                mod["residue"],
                mod["delta_mass"],
                tol=tol,
            )
            if not candidates:
                continue
            candidate = candidates[0]
            mods_residue.append(f"{mod['residue']}{mod['index'] + 1}")
            mods_name.append(f"{candidate['title']}|UniMod:{candidate['unimod_id']}")
            seqfiller_ptms.append(_seqfiller_ptm_token(mod["residue"], candidate["mono_mass"]))
        if mods_residue:
            mods_residue_text = ";".join(mods_residue)
            mods_name_text = ";".join(mods_name)
            new_row = row._asdict()
            new_row["mod_residue"] = mods_residue_text
            new_row["mod_name"] = mods_name_text
            ptm = seqfiller_ptms[0]
            df_ptm = pd.concat([df_ptm, pd.DataFrame([new_row])], ignore_index=True)
            ptm_freq[ptm] = ptm_freq.get(ptm, 0) + 1
    return df_ptm, ptm_freq


def _seqfiller_ptm_token(residue, mono_mass):
    return f"{residue}[{float(mono_mass):.6f}]"

def parse_args():
    parser = argparse.ArgumentParser(description="Run the peptide clustering and optional UniMod annotation workflow.")
    parser.add_argument("pep_path", help="Glob pattern for input CSVs, e.g. /path/to/YBC*csv")
    parser.add_argument(
        "--topk-ptm",
        type=int,
        default=None,
        help="Compute top-k PTM frequencies from filled peptides",
    )
    parser.add_argument(
        "--use-unimod",
        action="store_true",
        help="Enable UniMod annotation (default)",
        default=False,
    )
    parser.add_argument(
        "--refresh-unimod",
        action="store_true",
        help="Refresh the cached UniMod XML before annotation",
    )
    parser.add_argument(
        "--refresh-workflow-cache",
        action="store_true",
        help="Rebuild cached workflow null distributions",
    )
    parser.add_argument(
        "--null-workers",
        type=int,
        default=1,
        help="Worker count for null-distribution sampling",
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
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    if args.topk_ptm is not None and args.topk_ptm <= 0:
        raise ValueError("--topk-ptm must be greater than 0")
    if args.null_workers <= 0:
        raise ValueError("--null-workers must be greater than 0")
    show_progress = _progress_enabled(args.progress)

    temp = []
    input_files = sorted(glob(args.pep_path))
    if not input_files:
        raise FileNotFoundError(f"No peptide path CSV files matched: {args.pep_path}")
    for f in input_files:
        temp_psm = pd.read_csv(f)
        temp_psm['file_name'] = os.path.basename(f)
        temp.append(temp_psm)
    temp = pd.concat(temp, ignore_index=True)

    pep_dir = Path(args.pep_path).parent
    output_path = str(pep_dir / "filled_peptides.csv")
    workflow_cache_dir = pep_dir / ".cache" / "rnova"

    # build sequence cluster
    nodes_mass, clusters_filt = build_cluster(
        temp,
        cache_dir=workflow_cache_dir,
        refresh_cache=args.refresh_workflow_cache,
        null_workers=args.null_workers,
        show_progress=show_progress,
    )

    # msa on clusters and fill mass tag gaps with delta masses + aa
    filled_result = masstag_to_delta_mass(
        nodes_mass,
        clusters_filt,
        temp,
    )

    # save filled result
    filled_result.to_csv(output_path, index=False)

    if args.topk_ptm is not None:
        topk = args.topk_ptm
    else:
        topk = 3
    if args.use_unimod:
        # load UniMod database
        mods = load_unimod(refresh=args.refresh_unimod)
        index = UniModMassIndex(mods)

        # annotate PTMs
        ptm_result, ptm_freq = delta_mass_to_ptm(filled_result, index, topk)
        # save PTM annotated result
        output_path_ptm = os.path.splitext(output_path)[0] + "_with_PTM.csv"
        ptm_result.to_csv(output_path_ptm, index=False)
    else:
        ptm_freq = topk_ptm_annotation(filled_result, topk)
        # print(topk_ptm)
    clustered_topk_ptm = cluster_ptm_pairs(ptm_freq, eps=0.02)
    ptm_freq_clustered = dict(sorted(clustered_topk_ptm.items(), key=lambda kv: kv[1], reverse=True))
    print("topk_ptm_annotation:", ";".join(list(ptm_freq_clustered.keys())[:topk]))


def _progress_enabled(progress: str) -> bool:
    if progress == "on":
        return True
    if progress == "off":
        return False
    return sys.stderr.isatty()

if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
