#!/usr/bin/env python3
"""
compute_iPLL.py
────────────────────────────────────────────────────────────────────────────
Compute interface‑averaged PLL (iPLL) scores with ESM‑3 for sequences in a
FASTA file (or a raw string). These are mutant sequences that replace one
chain in a PDB complex.

Example
-------
# From FASTA file
python compute_iPLL.py data/3gbn_ablh_fvar.pdb data/cr6261_3gbn_hc_lib.fasta \
    --chain H --sel1 "segid H or segid L" --sel2 "segid A or segid B" --workers 8 --sequence_of_concern

# From string
python compute_iPLL.py data/3gbn_ablh_fvar.pdb "EVQLVESGAEV..." \
    --chain H --sel1 "segid H or segid L" --sel2 "segid A or segid B" --workers 1
"""
import os, gc, csv, argparse, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set
import torch
import pandas as pd
from Bio import SeqIO
import MDAnalysis as mda
from tqdm import tqdm
from esm.sdk.api import ESMProtein
from esm_utils import ESMUtils

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute iPLL scores for sequences replacing one chain in a PDB complex.")
    p.add_argument("pdb", help="Reference complex PDB")
    p.add_argument("sequence", help="Either a path to FASTA file or a raw sequence string")
    p.add_argument("--chain", required=True, help="Chain ID to replace in the PDB")
    p.add_argument("--sel1", required=True, help="MDAnalysis selection for first of interface")
    p.add_argument("--sel2", required=True, help="MDAnalysis selection for second side of interface")
    p.add_argument("--model", default="esm3-large-multimer-2024-09", help="ESM-3 model name")
    p.add_argument("--key", default=None, help="ESM-3 API key (or set $ESM3_KEY)")
    p.add_argument("-w","--workers", type=int, default=4, help="Parallel workers (default: 4)")
    p.add_argument("-o","--out", default=None, help="Output CSV path")
    p.add_argument("--sequence_of_concern", action="store_true", help="Set flag if you're looking at a concern (you can ask EvolutionaryScale to enable this for a virus project, for example)")
    return p.parse_args()

def chain_order_from_pdb(pdb_path: str) -> List[str]:
    u = mda.Universe(pdb_path)
    order, prev = [], None
    for res in u.residues:
        cid = (res.segid.strip() or res.chainID.strip())
        if cid != prev:
            order.append(cid)
            prev = cid
    return order

def main():
    args = parse_args()
    pdb_path = args.pdb
    input_arg = args.sequence
    chain_id = args.chain
    sel1, sel2 = args.sel1, args.sel2
    model_name = args.model
    n_workers = args.workers

    esm_key = args.key or os.getenv("ESM3_KEY")
    if not esm_key:
        raise RuntimeError("No ESM-3 key provided. Use --key or export ESM3_KEY.")

    chain_order = chain_order_from_pdb(pdb_path)
    if chain_id not in chain_order:
        raise ValueError(f"Chain '{chain_id}' not found in PDB. Found: {chain_order}")
    chain_idx = chain_order.index(chain_id)

    # Interpret the sequence argument
    if os.path.exists(input_arg) and input_arg.endswith(".fasta"):
        records = list(SeqIO.parse(input_arg, "fasta"))
        seq_df = pd.DataFrame({
            "header": [r.id.strip() for r in records],
            "sequence": [str(r.seq) for r in records]
        })
        out_csv = args.out or os.path.join(os.path.dirname(input_arg), "iPLL_results.csv")
    else:
        seq_df = pd.DataFrame([{
            "header": "input_seq",
            "sequence": input_arg.strip()
        }])
        out_csv = args.out or "iPLL_result.csv"

    utils = ESMUtils(esm3_api_token=esm_key, model_name=model_name)
    protein_base, protein_complex = utils.get_protein_from_pdb(pdb_path, is_protein_complex=True)
    utils.protein = protein_complex
    base_split = protein_base.sequence.split("|")
    int_mask_full = utils.define_interface(pdb_path, sel1, sel2, distance_cutoff_nm=1.0)
    int_mask_no_special = int_mask_full[1:-1]

    done_set: Set[str] = set()
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        try:
            done_df = pd.read_csv(out_csv, usecols=["mutant"], dtype=str)
            done_set = set(done_df["mutant"].str.strip())
            print(f"[checkpoint] {len(done_set)} previously scored – skipping.")
        except Exception as e:
            print(f"[checkpoint] Failed to load checkpoint: {e}")

    to_run = [row for _, row in seq_df.iterrows() if row["header"] not in done_set]
    print(f"[INFO] Total: {len(seq_df)}, To process: {len(to_run)}")

    def score_row(row):
        hdr, new_seq = row["header"], row["sequence"]
        split = base_split.copy()
        split[chain_idx] = new_seq
        complex_seq = "|".join(split)
        protein = ESMProtein(sequence=complex_seq, coordinates=protein_base.coordinates.clone(), potential_sequence_of_concern=args.sequence_of_concern)
        try:
            _, nll, _ = utils.get_logits(protein)
        except Exception as e:
            print(f"[ERROR] get_logits failed for row {row_idx}: {e}")
            print("There may be an issue with your ESMProtein object:\n", protein)
            traceback.print_exc()
            return row_idx, None
        ipll = float((-nll).cpu().numpy()[int_mask_no_special].mean())
        if model_name == "local-gpu":
            torch.cuda.empty_cache(); gc.collect()
        return {"mutant": hdr, "iPLL": ipll, "sequence": new_seq}

    mode = "a" if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0 else "w"
    with open(out_csv, mode, newline="", buffering=1) as fh:
        writer = csv.DictWriter(fh, fieldnames=["mutant", "iPLL", "sequence"])
        if mode == "w":
            writer.writeheader()
        with ThreadPoolExecutor(max_workers=n_workers) as exe:
            futures = [exe.submit(score_row, row) for row in to_run]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                try:
                    writer.writerow(fut.result())
                except Exception as exc:
                    print("[WARN] worker raised:", exc)
    print(f"[DONE] Results in {out_csv}")

if __name__ == "__main__":
    main()
