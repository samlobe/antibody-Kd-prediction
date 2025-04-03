#%%
import pandas as pd
import numpy as np
import os
from Bio import SeqIO

csvs = ['cr6261_mpnn.csv','cr9114_mpnn.csv','g6_HC_mpnn.csv','g6_LC_mpnn.csv']
datasets = ['cr6261','cr9114','g6_hc','g6_lc']
fasta_files = [f'../fastas/{dataset}.fasta' for dataset in datasets]

# load sequence from fasta file
def load_fasta(fasta_file):
    """
    Load sequences from a FASTA file into a dictionary.
    
    Args:
        fasta_file (str): Path to the FASTA file.
        
    Returns:
        dict: Dictionary with sequence IDs as keys and sequences as values.
    """
    fasta_dict = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        fasta_dict[record.id] = str(record.seq)
    return fasta_dict

dfs = []
for i, csv in enumerate(csvs):
    # load mpnn scores
    df = pd.read_csv(f'mpnn_output/{csv}')
    # load fasta file
    fasta_dict = load_fasta(fasta_files[i])
    # reverse the dict
    fasta_dict = {v: k for k, v in fasta_dict.items()}
    
    # add sequence to dataframe
    df['mutant'] = df['seq'].map(fasta_dict)
    
    # # save to new csv
    df.to_csv(f'{datasets[i]}_mpnn.csv', index=False)
    
    # append to list
    dfs.append(df)
