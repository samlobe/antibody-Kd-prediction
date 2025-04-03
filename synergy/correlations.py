#%%
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Kd_data_scores.csv")
score_cols = ["MPNN_score", "esmIF_score", "esmIF_public_score", "iPLL_score", "ddG_score"]
df["group"] = df["dataset"].replace({"g6_HC": "g6", "g6_LC": "g6"})
standardized_df = df.copy()

# Standardize scores within each group
for group_name, group_df in df.groupby("group"):
    scaler = StandardScaler()
    standardized_scores = scaler.fit_transform(group_df[score_cols])
    standardized_df.loc[group_df.index, score_cols] = standardized_scores

# multiply mpnn and ddG score by -1 such that higher scores are better
standardized_df["MPNN_score"] = standardized_df["MPNN_score"] * -1
standardized_df["ddG_score"] = standardized_df["ddG_score"] * -1

#%%
cr6261_df = standardized_df[standardized_df["dataset"] == "cr6261"]
cr6261_df = cr6261_df.drop(columns=["dataset", "mutant", "h3_mean", "norm_binding","group"])
cr6261_df = cr6261_df.dropna()

correlation_matrix = cr6261_df.corr(method='spearman')
# replace esmIF_score with esmIF_shanker
correlation_matrix = correlation_matrix.rename(columns={"esmIF_score": "esmIF_shanker"})
correlation_matrix = correlation_matrix.rename(index={"esmIF_score": "esmIF_shanker"})
# replace esmIF_public_score with esmIF_score
correlation_matrix = correlation_matrix.rename(columns={"esmIF_public_score": "esmIF"})
correlation_matrix = correlation_matrix.rename(index={"esmIF_public_score": "esmIF"})
# replace iPLL_score with esm3-iPLL
correlation_matrix = correlation_matrix.rename(columns={"iPLL_score": "esm3-iPLL"})
correlation_matrix = correlation_matrix.rename(index={"iPLL_score": "esm3-iPLL"})
# replace ddG_score with ddG
correlation_matrix = correlation_matrix.rename(columns={"ddG_score": "ddG"})
correlation_matrix = correlation_matrix.rename(index={"ddG_score": "ddG"})
# replace MPNN_score with MPNN
correlation_matrix = correlation_matrix.rename(columns={"MPNN_score": "MPNN"})
correlation_matrix = correlation_matrix.rename(index={"MPNN_score": "MPNN"})
print(correlation_matrix)

correlation_matrix = correlation_matrix.iloc[2:, 2:]  # focus on features

plt.figure(figsize=(10, 8))
ax = sns.heatmap(correlation_matrix, 
            cmap='coolwarm', 
            annot=True, 
            fmt=".2f",
            annot_kws={"size": 14}, 
            xticklabels=True, 
            yticklabels=True,
            vmin=0, vmax=1,
            cbar_kws={})
plt.title("spearman correlations for zero-shot predictors:\ncr6261 dataset", fontsize=20)
plt.xticks(rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=14, rotation=0)

cbar = ax.collections[0].colorbar
cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
cbar.ax.tick_params(labelsize=14)

plt.tight_layout()
plt.show()
# %%
cr9114_df = standardized_df[standardized_df["dataset"] == "cr9114"]
cr9114_df = cr9114_df.drop(columns=["dataset", "mutant", "h9_mean", "norm_binding","group","ddG_score"])
cr9114_df = cr9114_df.dropna()

correlation_matrix = cr9114_df.corr(method='spearman')
# replace esmIF_score with esmIF_shanker
correlation_matrix = correlation_matrix.rename(columns={"esmIF_score": "esmIF_shanker_score"})
correlation_matrix = correlation_matrix.rename(index={"esmIF_score": "esmIF_shanker_score"})
# replace esmIF_public_score with esmIF_score
correlation_matrix = correlation_matrix.rename(columns={"esmIF_public_score": "esmIF_score"})
correlation_matrix = correlation_matrix.rename(index={"esmIF_public_score": "esmIF_score"})
# replace iPLL_score with esm3-iPLL
correlation_matrix = correlation_matrix.rename(columns={"iPLL_score": "esm3-iPLL_score"})
correlation_matrix = correlation_matrix.rename(index={"iPLL_score": "esm3-iPLL_score"})
print(correlation_matrix)
correlation_matrix = correlation_matrix.iloc[2:, 2:] # focus on features

plt.figure(figsize=(10, 8))
ax = sns.heatmap(correlation_matrix, 
            cmap='coolwarm', 
            annot=True, 
            fmt=".2f",  # format numbers to 2 decimal places
            annot_kws={"size": 14}, 
            xticklabels=True, 
            yticklabels=True,
            vmin=0, vmax=1,
            cbar_kws={})
plt.title("spearman correlations for cr9114 dataset", fontsize=20)
plt.xticks(rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=14, rotation=0)

cbar = ax.collections[0].colorbar
cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
cbar.ax.tick_params(labelsize=14)

plt.tight_layout()
plt.show()

#%%
g6_df = standardized_df[standardized_df["group"] == "g6"]
g6_df = g6_df.drop(columns=["dataset", "mutant", "h9_mean", "h1_mean", "h3_mean","group","ddG_score"])
g6_df = g6_df.dropna()

correlation_matrix = g6_df.corr(method='spearman') 
# replace esmIF_score with esmIF_shanker
correlation_matrix = correlation_matrix.rename(columns={"esmIF_score": "esmIF_shanker_score"})
correlation_matrix = correlation_matrix.rename(index={"esmIF_score": "esmIF_shanker_score"})
# replace esmIF_public_score with esmIF_score
correlation_matrix = correlation_matrix.rename(columns={"esmIF_public_score": "esmIF_score"})
correlation_matrix = correlation_matrix.rename(index={"esmIF_public_score": "esmIF_score"})
# replace iPLL_score with esm3-iPLL
correlation_matrix = correlation_matrix.rename(columns={"iPLL_score": "esm3-iPLL_score"})
correlation_matrix = correlation_matrix.rename(index={"iPLL_score": "esm3-iPLL_score"})
print(correlation_matrix)
correlation_matrix = correlation_matrix.iloc[1:, 1:] # focus on features

plt.figure(figsize=(10, 8))
ax = sns.heatmap(correlation_matrix, 
            cmap='coolwarm', 
            annot=True, 
            fmt=".2f",
            annot_kws={"size": 14}, 
            xticklabels=True, 
            yticklabels=True,
            vmin=0, vmax=1,
            cbar_kws={})
plt.title("spearman correlations for g6 dataset", fontsize=20)
plt.xticks(rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=14, rotation=0)

cbar = ax.collections[0].colorbar
cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) 
cbar.ax.tick_params(labelsize=14) 

plt.tight_layout()
plt.show()

# %%
