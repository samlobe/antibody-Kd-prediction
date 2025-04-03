from Bio import SeqIO
from esm.sdk.api import GenerationConfig, ESMProtein
from esm_utils import ESMUtils  # <-- import from your updated utility file
from tqdm import tqdm

import numpy as np
import pandas as pd
import os
import torch
import gc
import time

from concurrent.futures import ThreadPoolExecutor, as_completed

ESM3_KEY = "ESM3_KEY"

# ESM3 API Key
try:
    esm3_token = os.environ[ESM3_KEY]
except KeyError:
    msg = f"Please set up an environment variable named '{ESM3_KEY}' for your ESM API key."
    raise KeyError(msg)

#############################
# Configuration parameters  #
#############################
N_WORKERS = 7
DATA_DIR = "./data"
MASK_DIR = "./data/masks"
folded_structures_path = os.path.join(DATA_DIR, "esm_folded_structures")
fasta_path = os.path.join(DATA_DIR, "cr6261_3gbn_hc_lib.fasta")
expr_path = os.path.join(DATA_DIR, "cr6261_exp_data.csv")
pdb_path = os.path.join(DATA_DIR, "3gbn_ablh_fvar.pdb")

hide_interface_structure = False
model_name = "esm3-large-multimer-2024-09"  # or "local-cpu", "local-gpu", etc.

contact_cutoff = 1.0
scores_path = os.path.join(DATA_DIR, f"PLL_singlePass_{model_name}.csv")
if hide_interface_structure:
    scores_path = os.path.join(DATA_DIR, f"PLL_singlePass_{model_name}_{contact_cutoff}_hidden_interface_struct.csv")
print(f"Scores will be saved to: {scores_path}")

interface_resids_mask_extended_path = os.path.join(MASK_DIR, f"interface_resid_mask_special_{contact_cutoff:.1f}.csv")
interface_resids_mask_path = os.path.join(MASK_DIR, f"interface_resid_mask_{contact_cutoff:.1f}.csv")

# Load FASTA data into dataframe
df_seq = pd.DataFrame(
    [{"genotype": str(r.id), "sequence": str(r.seq)} for r in SeqIO.parse(fasta_path, "fasta")]
)
df_seq['genotype'] = df_seq['genotype'].str.strip()

# Load experimental data into dataframe
df_expr = pd.read_csv(expr_path, dtype={"genotype": str})
df_expr['genotype'] = df_expr['genotype'].str.strip()

# Join FASTA sequences with experimental data
df = pd.merge(df_seq, df_expr, on="genotype", how="left")

# Retain columns of interest
cols_to_keep = [
    "genotype",
    "sequence",
    "h1_mean",
]
df = df[cols_to_keep]


print("Loading data from Shanker, et. al. 2024...")
print(df)

# import masks
interface_resids_mask = pd.read_csv(interface_resids_mask_path, index_col=0).values.squeeze()
interface_resids_mask_extended = pd.read_csv(interface_resids_mask_extended_path, index_col=0).values[1:-1].squeeze()

# Print shapes just as a sanity check
print(f'interface_resids_mask: {np.shape(interface_resids_mask)}')
print(f'interface_resids_mask_extended: {np.shape(interface_resids_mask_extended)}')

if model_name not in ["local-cpu", "local-gpu"]:
    print(f'Using private ESM3 model: {model_name}')

# Initialize ESMUtils once
esm_utils = ESMUtils(esm3_api_token=esm3_token, model_name=model_name)

# Get a baseline protein + protein_complex from the reference PDB
protein_base, protein_complex = esm_utils.get_protein_from_pdb(
    pdb_path=pdb_path, 
    is_protein_complex=True
)

# Prepare a new column `protein_complex_sequence` for each row
sequence_split = protein_base.sequence.split("|")
df["protein_complex_sequence"] = df["sequence"].apply(
    lambda x: "|".join(sequence_split[:2] + [x] + sequence_split[3:])
)

#######################################
# Checkpoint logic: partial results   #
#######################################
if os.path.exists(scores_path):
    df_done = pd.read_csv(scores_path, usecols=["genotype"], dtype={"genotype": str})
    done_set = set(df_done["genotype"])
    print(f"Found existing partial results with {len(done_set)} genotypes. Will skip those.")
else:
    done_set = set()


############################################
# Single-row logic in a separate function  #
############################################
def score_one_genotype(row) -> dict:
    """Compute the ESM-based scores for a single genotype row.

    Returns a dictionary with columns you will merge into your final DF.
    """
    genotype = row["genotype"]

    # Clone the base protein and override its fields
    protein = ESMProtein(
        sequence=row["protein_complex_sequence"], 
        coordinates=protein_base.coordinates.clone(),
        potential_sequence_of_concern=True
    )

    # Optionally hide structure for interface residues
    if hide_interface_structure:
        coords_tensor = protein.coordinates.clone()
        nan_matrix = torch.full(coords_tensor.shape[-2:], float('nan'), dtype=coords_tensor.dtype)
        extended_mask = torch.tensor(interface_resids_mask_extended, dtype=torch.bool)
        # hide only extended interface
        protein.coordinates[extended_mask] = nan_matrix

    # Get logits + negative log likelihoods
    logits, nll, _ = esm_utils.get_logits(protein=protein)

    # nll is tensor object, create a df of one row per token (token_0, token_1, etc)
    ll = nll.cpu().numpy() * -1
    # assign token indices to the nll values
    ll_df = pd.DataFrame(
        [ll],
        columns=[f"token_{i}" for i in range(nll.shape[0])]
    )
    ll_df["genotype"] = genotype
    ll_df = ll_df.set_index("genotype")

    output_dict = ll_df

    # Cleanup
    del logits
    if model_name == 'local-gpu':
        torch.cuda.empty_cache()
        gc.collect()

    return output_dict


########################################
# Parallel execution with Thread Pool  #
########################################
MAX_WORKERS = N_WORKERS  # Tweak as needed; 5 threads ~ 5 calls/sec if each call is ~1 second
futures = []
results = []

csv_exists = os.path.exists(scores_path)

columns = ["genotype"] + [f"token_{i}" for i in range(734)]  # 734 tokens in antibody complex

# We'll open the file in append mode and write headers only if it doesn't exist
csv_file_handle = open(scores_path, "a", buffering=1)
if not csv_exists:
    csv_file_handle.write(",".join(columns) + "\n")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submit tasks
    for i, row in df.iterrows():
        genotype = row["genotype"]
        if genotype in done_set:
            continue  # Already done
        future = executor.submit(score_one_genotype, row)
        futures.append(future)

    # As each future completes, append to CSV
    for f in tqdm(as_completed(futures), total=len(futures)):
        try:
            df_single_row = f.result()  # This is a 1-row DataFrame
            # Convert it to a dictionary of {column -> value}
            row_dict = df_single_row.reset_index().iloc[0].to_dict()

            # Build the CSV line using your known column order
            csv_line = ",".join(str(row_dict.get(col, "")) for col in columns)
            csv_file_handle.write(csv_line + "\n")

        except Exception as e:
            print(f"[WARNING] A thread raised an exception: {e}")


csv_file_handle.close()  # done writing partial results

print("Done!")
