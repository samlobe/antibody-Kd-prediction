#%%
import numpy as np
import mdtraj as md
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

peptide_path = 'data/3gbn_ablh_fvar.pdb'
pdb = md.load(peptide_path)
topology = pdb.topology

alpha_carbons = ([atom.index for atom in topology.atoms if atom.name == 'CA'])

# Get all pairs of alpha carbons
atom_pairs = list(combinations(alpha_carbons,2))
pairwise_distances = md.geometry.compute_distances(pdb, atom_pairs)[0]

# num_residues = pdb.n_residues  # Number of residues in the protein
num_residues = len(alpha_carbons)
atom_to_resid = {}
for i, atom in enumerate(alpha_carbons):
    atom_to_resid[atom] = i

# Initialize an empty 2D matrix for the distances
distance_matrix = np.zeros((num_residues, num_residues))

# Fill the distance matrix. Since the distances are symmetric, we mirror the values across the diagonal
for distance, (atom_index_i, atom_index_j) in zip(pairwise_distances, atom_pairs):
    residue_index_i = atom_to_resid[atom_index_i]
    residue_index_j = atom_to_resid[atom_index_j]
    
    # Populate the matrix, adjusting indices by -1 if necessary
    # Adjust the indices based on how your residues are indexed (0-based or 1-based)
    distance_matrix[residue_index_i][residue_index_j] = distance
    distance_matrix[residue_index_j][residue_index_i] = distance  # Mirror the distance for symmetry
    # print(distance_matrix)
    # print(np.shape(distance_matrix))
assert distance_matrix.shape == (num_residues, num_residues), f'Expected num_res x num_res array and got {np.size(distance_matrix)}'

# print the chain id
chain_order = ['A','B','H','L']
chain_ids = [atom.residue.chain.index for atom in topology.atoms if atom.name == 'CA']
unique_chain_ids = np.unique(chain_ids)

# dict from chain id to chain order
chain_id_to_chain_order = {}
for i, chain_id in enumerate(unique_chain_ids):
    chain_id_to_chain_order[chain_id] = chain_order[i]
chain_ids = [chain_id_to_chain_order[chain_id] for chain_id in chain_ids]

chain_counts = {chain_id: chain_ids.count(chain_id) for chain_id in chain_order}

# chain counts has the residues per chain (328, 173, 121, 109)
resids = []
for chain in chain_order:
    resids.append(np.arange(1, chain_counts[chain]+1))
# flatten the list
resids = [item for sublist in resids for item in sublist]
indices = [f'{chain}{resid}' for chain, resid in zip(chain_ids, resids)]


distance_df = pd.DataFrame(distance_matrix, columns=indices, index=indices)
cutoff = 1.0 # nm
contacts_df = distance_df < cutoff

# define the rectangular mask
chainH = np.array([chain == 'H' for chain in chain_ids])
chainL = np.array([chain == 'L' for chain in chain_ids])
chainHL = chainH | chainL
chainH_or_chainH = np.tile(chainH, (num_residues,1)) | np.tile(chainH, (num_residues,1)).T
chainL_or_chainL = np.tile(chainL, (num_residues,1)) | np.tile(chainL, (num_residues,1)).T
chainH_and_chainH = np.tile(chainH, (num_residues,1)) & np.tile(chainH, (num_residues,1)).T
chainL_and_chainL = np.tile(chainL, (num_residues,1)) & np.tile(chainL, (num_residues,1)).T
chainHL_and_chainHL = np.tile(chainHL, (num_residues,1)) & np.tile(chainHL, (num_residues,1)).T
rectangular_mask = chainH_or_chainH &~ chainHL_and_chainHL

# multiply the contact map with the rectangular mask
contact_mask = contacts_df & rectangular_mask

# plot the contact mask
plt.figure(figsize=(10,10))
plt.imshow(contact_mask, cmap='binary')
plt.xticks(np.arange(0,num_residues,20), indices[::20], rotation=90)
plt.yticks(np.arange(0,num_residues,20), indices[::20])
# put a vertical line between chains
plt.axvline(chain_counts['A'], color='white', linewidth=1)
plt.axvline(chain_counts['A']+chain_counts['B'], color='white', linewidth=1)
plt.axvline(chain_counts['A']+chain_counts['B']+chain_counts['H'], color='white', linewidth=1)
plt.axhline(chain_counts['A'], color='white', linewidth=1)
plt.axhline(chain_counts['A']+chain_counts['B'], color='white', linewidth=1)
plt.axhline(chain_counts['A']+chain_counts['B']+chain_counts['H'], color='white', linewidth=1)
plt.title('Contact Mask', fontsize=20)
print(np.shape(contact_mask))

# save this contact mask to a csv
contact_mask.to_csv(f'data/masks/contact_mask_{cutoff:.1f}.csv')

# make a copy of the contact mask that accounts for the extra tokens:
# 1 BOS, 3 chain breaks, 1 EOS

chain_breaks = [chain_counts['A'], chain_counts['A']+chain_counts['B'], chain_counts['A']+chain_counts['B']+chain_counts['H']]
# Number of additional tokens:
# 1 BOS + 3 chain breaks + 1 EOS = 5 extra tokens
num_extra_tokens = 5
new_size = num_residues + num_extra_tokens

new_contact_mask = np.zeros((new_size, new_size), dtype=bool)

chain_breaks = [0, chain_counts['A'], chain_counts['A']+chain_counts['B'], chain_counts['A']+chain_counts['B']+chain_counts['H'], num_residues]

# Create a mapping from old residue indices to new indices
res_map = [None]*num_residues
current_index = 1  # Start after BOS

for i in range(len(chain_breaks)-1):
    start = chain_breaks[i]
    end = chain_breaks[i+1]

    # Map residues in this segment
    for r in range(start, end):
        res_map[r] = current_index
        current_index += 1
    
    # Insert chain break token if not at the last segment
    # (We have 4 chains and thus 3 chain breaks)
    if i < len(chain_breaks)-2:
        current_index += 1

# Now copy the old matrix into the new matrix
old_contact_mask = contact_mask.values.astype(bool)  # if you're using DataFrame
for i in range(num_residues):
    for j in range(num_residues):
        new_contact_mask[res_map[i], res_map[j]] = old_contact_mask[i, j]

# 731x731 matrix now is 736x736
print(np.shape(new_contact_mask))
plt.figure(figsize=(10,10))
plt.imshow(new_contact_mask, cmap='binary')
plt.title('Contact Mask Extended\n(with 5 additional special tokens)', fontsize=20)

# create new indices including the extra tokens
chainA_indices = [f'A{resid}' for resid in range(1, chain_counts['A']+1)]
chainB_indices = [f'B{resid}' for resid in range(1, chain_counts['B']+1)]
chainH_indices = [f'H{resid}' for resid in range(1, chain_counts['H']+1)]
chainL_indices = [f'L{resid}' for resid in range(1, chain_counts['L']+1)]
new_indices = ['BOS'] + chainA_indices + ['chain_break1'] + chainB_indices + ['chain_break2'] + chainH_indices + ['chain_break2'] + chainL_indices + ['EOS']
num_tokens = len(new_indices)

# save this new contact mask to a csv
extended_contact_mask_df = pd.DataFrame(new_contact_mask, columns=new_indices, index=new_indices)
extended_contact_mask_df.to_csv(f'data/masks/contact_mask_special_{cutoff:.1f}.csv')

# now residue masks!
interface_resid_mask = np.zeros(num_residues, dtype=bool)

for i,res_contact_array in enumerate(contact_mask.index):
    if any(contact_mask[res_contact_array]):
        interface_resid_mask[i] = True
for i,res_contact_array in enumerate(contact_mask.columns):
    if any(contact_mask[res_contact_array]):
        interface_resid_mask[i] = True

print(f'Number of interface residues: {np.sum(interface_resid_mask)}')
print(f'(using cutoff of {cutoff} nm)')

# output interface_resid_mask to a csv
interface_resid_mask_df = pd.DataFrame(interface_resid_mask, columns=['Interface Residue'])
interface_resid_mask_df.to_csv(f'data/masks/interface_resid_mask_{cutoff:.1f}.csv')

interface_resid_mask_extended = extended_contact_mask_df.any(axis=0).values
interface_resid_mask_extended_df = pd.DataFrame(interface_resid_mask_extended, index=new_indices)
interface_resid_mask_extended_df.to_csv(f'data/masks/interface_resid_mask_special_{cutoff:.1f}.csv')

#%%