#!/usr/bin/env python3
"""
compute_iPLL_parallel.py
────────────────────────────────────────────────────────────────────────────
For every row in an input CSV that contains:
  • a PDB file path   (column name user‑specified)
  • a peptide/protein sequence (column name user‑specified)

Replace *one* chain in each PDB with that sequence, call ESM‑3, compute
the interface‑averaged pseudo log‑likelihood (iPLL), and write each result
immediately to a separate CSV for checkpointing. Multiple workers can
run in parallel and append safely.
"""
import os, csv, gc, argparse, sys, multiprocessing
from typing import Tuple, Optional
import torch
import pandas as pd
import MDAnalysis as mda
from esm.sdk.api import ESMProtein
from esm_utils import ESMUtils
from tqdm import tqdm
import warnings


# globals set in each worker
utils_global:      Optional[ESMUtils] = None
MODEL_NAME:        str              = ""
CHAIN_ID:          str              = ""
SEL1:              str              = ""
SEL2:              str              = ""
DISTANCE_CUTOFF:   float            = 1.0

# Suppress warnings from MDAnalysis about guessed elements
warnings.filterwarnings("ignore", message=".*elements were guessed from atom name", category=UserWarning)

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute iPLL for each (pdb, sequence) pair listed in a CSV, in parallel."
    )
    p.add_argument("csv", help="Input CSV file")
    p.add_argument("--pdb-col",   required=True, help="Column name containing PDB paths")
    p.add_argument("--seq-col",   required=True, help="Column name containing sequences")
    p.add_argument("--chain",     required=True, help="Chain ID to replace in each PDB")
    p.add_argument("--sel1", required=True, help="MDAnalysis selection for first of interface")
    p.add_argument("--sel2", required=True, help="MDAnalysis selection for second side of interface")
    p.add_argument("--key",       default=None,  help="ESM‑3 API key (or set $ESM3_KEY)")
    p.add_argument("--model",     default="esm3-large-multimer-2024-09",
                   help="ESM‑3 model name")
    p.add_argument("-o", "--out", default="iPLL_results.csv",
                   help="Output results CSV (default: iPLL_results.csv)")
    p.add_argument("-w", "--workers", type=int, default=4,
                   help="Number of parallel workers (default: CPU count)")
    p.add_argument("--sequence_of_concern", action="store_true", help="Set flag if you're looking at a concern (you can ask EvolutionaryScale to enable this for a virus project, for example)")
    return p.parse_args()

def chain_order(pdb_path: str):
    u = mda.Universe(pdb_path)
    order, prev = [], None
    for res in u.residues:
        cid = (res.segid.strip() or res.chainID.strip())
        if cid != prev:
            order.append(cid)
            prev = cid
    return order

def init_worker(esm_key: str, model_name: str,
                chain_id: str, sel1: str, sel2: str):
    """Initializer run in each process."""
    global utils_global, MODEL_NAME, CHAIN_ID, SEL1, SEL2
    MODEL_NAME = model_name
    CHAIN_ID   = chain_id
    SEL1       = sel1
    SEL2       = sel2
    utils_global = ESMUtils(esm3_api_token=esm_key, model_name=model_name)

def process_pair(task: Tuple[str,str]) -> Optional[Tuple[str,str,float]]:
    """Compute iPLL for one (pdb_path, sequence)."""
    pdb_path, sequence = task
    try:
        # check chain exists
        order = chain_order(pdb_path)
        if CHAIN_ID not in order:
            print(f"[WARN] chain {CHAIN_ID} not in {pdb_path}; skipping")
            return None
        chain_idx = order.index(CHAIN_ID)

        # load proteins
        prot_base, prot_complex = utils_global.get_protein_from_pdb(
            pdb_path, is_protein_complex=True
        )
        utils_global.protein = prot_complex
        base_split = prot_base.sequence.split("|")

        # interface mask
        int_mask = utils_global.define_interface(
            pdb_path=pdb_path,
            selection_1=SEL1,
            selection_2=SEL2,
            distance_cutoff_nm=DISTANCE_CUTOFF,
        )[1:-1]

        # build mutant sequence
        split = base_split.copy()
        split[chain_idx] = sequence
        complex_seq = "|".join(split)

        # call ESM‑3
        protein = ESMProtein(
            sequence=complex_seq,
            coordinates=prot_base.coordinates.clone(),
            potential_sequence_of_concern=args.sequence_of_concern,
        )

        try:
            _, nll, _ = utils_global.get_logits(protein)
        except Exception as e:
            print(f"[ERROR] get_logits failed for row {row_idx}: {e}")
            print("There may be an issue with your ESMProtein object:\n", protein)
            traceback.print_exc()
            return row_idx, None
        score = float((-nll).cpu().numpy()[int_mask].mean())

        # optional cleanup
        if MODEL_NAME == "local-gpu":
            torch.cuda.empty_cache()
            gc.collect()

        return (pdb_path, sequence, score)

    except Exception as e:
        print(f"[ERROR] {pdb_path}, {sequence}: {e}")
        return None

def main():
    args = parse_args()
    esm_key = args.key or os.getenv("ESM3_KEY")
    if not esm_key:
        sys.exit("Provide ESM‑3 key via --key or set $ESM3_KEY")

    # read input CSV
    df = pd.read_csv(args.csv)

    # Prepare or read checkpoint file
    seen = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                seen.add((row[args.pdb_col], row[args.seq_col]))
    else:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([args.pdb_col, args.seq_col, "iPLL"])

    # Build task list
    all_tasks = [
        (row[args.pdb_col], row[args.seq_col])
        for _, row in df.iterrows()
    ]
    todo = [t for t in all_tasks if t not in seen]
    if not todo:
        print("✅  All (pdb,sequence) pairs already processed.")
        return

    # workers = multiprocessing.cpu_count()
    workers = args.workers
    print(f"▶️  Processing {len(todo)} tasks with {workers} workers...")

    mgr  = multiprocessing.Manager()
    lock = mgr.Lock()
    pool = multiprocessing.Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(esm_key, args.model, args.chain, args.sel1, args.sel2)
    )

    # As each result arrives, append it under lock
    for result in tqdm(pool.imap_unordered(process_pair, todo), total=len(todo)):
        if not result:
            continue
        pdb_path, seq, score = result
        with lock:
            with open(args.out, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([pdb_path, seq, score])

    pool.close()
    pool.join()
    print(f"✅  Done — results in {args.out}")

if __name__ == "__main__":
    main()
