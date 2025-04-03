import pandas as pd
import numpy as np

# Input paths
pll_file = "data/PLL_singlePass_esm3-large-multimer-2024-09.csv"
interface_mask_file = "data/interface_resid_mask_special_1.0.csv" # 1nm cutoff

# Load PLL matrix
pll_df = pd.read_csv(pll_file, index_col=0).sort_index()
pll_df = pll_df.drop_duplicates().dropna(axis=1, how='all')

# Load mask and flatten it
mask = pd.read_csv(interface_mask_file, index_col=0)
interface_mask = mask.values[1:-1].flatten()

# Select only the interface columns
interface_pll = pll_df.loc[:, interface_mask.astype(bool)]
iPLL = interface_pll.mean(axis=1)

# Print or save the iPLL
print(iPLL.head())
iPLL.to_csv("iPLLs.csv")

#%%
# Optional: Load experimental data and compute correlation
try:
    exp_data_file = "data/experimental_data.csv"  # optional
    exp_data = pd.read_csv(exp_data_file, index_col=0).sort_index().squeeze()
    corr = iPLL.corr(exp_data, method='spearman')
    print(f"Spearman correlation with experimental data: {corr:.3f}")
except FileNotFoundError:
    print("Experimental data not found. Skipping correlation step.")

