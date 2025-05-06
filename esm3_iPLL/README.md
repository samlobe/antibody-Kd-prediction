# esm3 iPLL scoring

esm3-large-multimer (2024-09 model) appears to have an interface pseudologliklihood (iPLL) that is useful as a zero-shot antibody binding affinity predictor in some cases, and it has unique info from popular tools like esm-IF, proteinMPNN, and Rosetta ddG.

We provide a python script `compute_iPLL.py` that allows you to evlauate how mutations (provided in a fasta file) affect esm3's iPLL scores for protein structure that you provide (.pdb).

## Example

```
python compute_iPLL.py \
    data/3gbn_ablh_fvar.pdb \
    data/cr6261_3gbn_hc_lib.fasta \
    --chain H \
    --sel1 "segid H or segid L" \
    --sel2 "segid A or segid B"
```
Arguments:
- `pdb`     : Path to the reference PDB complex.
- `fasta`   : Path to FASTA file containing sequences to score.
- `--chain`   : Chain ID in the PDB file to replace with each FASTA sequence.
- `--sel1`    : MDAnalysis selection string for interface side 1 (e.g. binder).
- `--sel2`    : MDAnalysis selection string for interface side 2 (e.g. antigen).
- `--key`     : (optional) ESM-3 API key. Defaults to environment variable `$ESM3_KEY`.
- `--workers` : (optional) Number of parallel threads (defaults = 4, must be less than available processors on your machine).
- `--out`     : (optional) Output CSV path. Default: same dir as FASTA → `iPLL_RESULTS.csv`.

The fasta file must contain sequences for a full chain (e.g. antibody heavy chain) and this chain's ID must be specified (e.g. `--chain H`).

Use `--sel1` and `--sel2` to define the chains/residues for each side of the interface. Typically `--sel1` is the binder (e.g. antibody chains) and `--sel2` is the target protein, so for the complex in [PDB: 3GBN](https://www.rcsb.org/structure/3GBN) you would do `--sel1 segid H or segid L --sel2 segid A or segid B` which use [MDAnalysis selection language](https://docs.mdanalysis.org/stable/documentation_pages/selections.html). You are able to select residues liked `resid 10-20` or chains like `chainID B`, for example. 

We consider residues from each selection whose alpha carbons are within 1.0 nm of each other - 1.0 nm was tuned based on five antibody $K_D$ datasets (see ../images/esm3-large-multimer-2024-09_cutoff_sensitivity.png).

## Installation
In a fresh environment, install the following:
1. **ESM3** (has SDK to interact with ESM's models via forge):
   ```
   pip install esm3
   ```
2. **MDAnalysis** (for interface definition and structure parsing):
   ```
   pip install MDAnalysis
   ```
