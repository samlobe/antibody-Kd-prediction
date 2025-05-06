#!/usr/bin/env python3
"""
compute_iPLL.py
────────────────────────────────────────────────────────────────────────────
Compute interface‑averaged PLL (iPLL) scores with ESM‑3 for sequences in a
FASTA file that replace one chain in a PDB complex.

The script is checkpoint‑aware: rerunning appends only new mutants.

Example
-------
python compute_iPLL.py \
    data/3gbn_ablh_fvar.pdb \
    data/cr6261_3gbn_hc_lib.fasta \
    --chain H \
    --sel1 "segid H or segid L" \
    --sel2 "segid A or segid B"
"""
# ───────────────────────────────────────────────────────────────────────────
import os, gc, csv, argparse
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
    p = argparse.ArgumentParser(
        description="Compute iPLL scores for a FASTA replacing one chain "
                    "in a PDB complex."
    )
    p.add_argument("pdb", help="Reference complex PDB")
    p.add_argument("fasta", help="FASTA file with mutant sequences (must be full-chain sequences)")
    p.add_argument("--chain", required=True,
                   help="Chain ID in the PDB to replace with each FASTA sequence")

    p.add_argument("--sel1", required=True, help="MDAnalysis selection string defining first part of interface (e.g. binder)")
    p.add_argument("--sel2", required=True, help="MDAnalysis selection string defining second part of interface (e.g. target)")

    p.add_argument("--model", default="esm3-large-multimer-2024-09",
                   help="ESM‑3 model name (default: %(default)s)")
    p.add_argument("--key",   default=None,
                   help="ESM‑3 API key (defaults to env variable $ESM3_KEY)")
    p.add_argument("--workers", type=int, default=4,
                   help="Number of parallel API calls (default: %(default)s)")
    p.add_argument("--out", default=None,
                   help="Output CSV (default: <FASTA dir>/iPLL_results.csv)")
    return p.parse_args()


# ----------------------------- helpers -------------------------------------
def chain_order_from_pdb(pdb_path: str) -> List[str]:
    """Return list of chain IDs in the file order MDAnalysis sees."""
    u = mda.Universe(pdb_path)
    order, prev = [], None
    for res in u.residues:
        cid = (res.segid.strip() or res.chainID.strip())
        if cid != prev:
            order.append(cid)
            prev = cid
    return order

# ------------------------------ main ---------------------------------------
def main():
    args = parse_args()

    fasta_path = args.fasta
    pdb_path   = args.pdb
    chain_id   = args.chain
    sel1, sel2 = args.sel1, args.sel2
    model_name = args.model
    n_workers  = args.workers
    out_csv    = (
        args.out
        if args.out
        else os.path.join(os.path.dirname(fasta_path), "iPLL_RESULTS.csv")
    )

    # ESM‑3 key
    esm_key = args.key or os.getenv("ESM3_KEY")
    if not esm_key:
        raise RuntimeError(
            "No ESM‑3 key provided. Pass --key KEY or set environment variable $ESM3_KEY."
        )

    # ------------------- chain sanity check -------------------------------
    chain_order = chain_order_from_pdb(pdb_path)
    if chain_id not in chain_order:
        raise ValueError(
            f"Chain '{chain_id}' not found in PDB. Chains present: {chain_order}"
        )
    chain_idx = chain_order.index(chain_id)

    # ------------------- load FASTA --------------------------------------
    records = list(SeqIO.parse(fasta_path, "fasta"))
    seq_df = pd.DataFrame(
        {"header": [r.id.strip() for r in records],
         "sequence": [str(r.seq)  for r in records]}
    )

    # ------------------- set up ESM / PDB ---------------------------------
    utils = ESMUtils(esm3_api_token=esm_key, model_name=model_name)
    protein_base, protein_complex = utils.get_protein_from_pdb(
        pdb_path, is_protein_complex=True
    )
    utils.protein = protein_complex
    base_split = protein_base.sequence.split("|")

    # interface mask
    int_mask_full = utils.define_interface(
        pdb_path=pdb_path,
        selection_1=sel1,
        selection_2=sel2,
        distance_cutoff_nm=1.0,
    )
    int_mask_no_special = int_mask_full[1:-1]

    # ------------------- checkpointing ------------------------------------
    done_set: Set[str] = set()
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        try:
            done_df = pd.read_csv(out_csv, usecols=["mutant"], dtype=str)
            done_set = set(done_df["mutant"].str.strip())
            print(f"[checkpoint] {len(done_set)} mutants already scored – will skip them.")
        except Exception as e:
            print(f"[checkpoint] could not read existing CSV ({e}); proceeding without skip.")

    to_run = [row for _, row in seq_df.iterrows() if row["header"] not in done_set]
    print(f"[INFO] total mutants in FASTA: {len(seq_df)}")
    print(f"[INFO] mutants to process this run: {len(to_run)}")

    # ------------------- worker function ----------------------------------
    def score_row(row):
        hdr, new_seq = row["header"], row["sequence"]
        split = base_split.copy()
        split[chain_idx] = new_seq
        complex_seq = "|".join(split)

        protein = ESMProtein(
            sequence=complex_seq,
            coordinates=protein_base.coordinates.clone(),
            potential_sequence_of_concern=True,
        )
        _, nll, _ = utils.get_logits(protein)
        ipll = float((-nll).cpu().numpy()[int_mask_no_special].mean())

        if model_name == "local-gpu":
            torch.cuda.empty_cache(); gc.collect()
        return {"mutant": hdr, "iPLL": ipll, "sequence": new_seq}

    mode = "a" if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0 else "w" # write header if new file
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

    print(f"[DONE] results written to {out_csv}")

if __name__ == "__main__":
    main()
