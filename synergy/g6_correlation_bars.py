#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

# Load and standardize
df = pd.read_csv("Kd_data_scores.csv")
score_cols = ["MPNN_score", "esmIF_score", "esmIF_public_score", "iPLL_score", "ddG_score"]
df["group"] = df["dataset"].replace({"g6_HC": "g6", "g6_LC": "g6"})

std_df = df.copy()
for name, grp in df.groupby("group"):
    scaler = StandardScaler()
    std_scores = scaler.fit_transform(grp[score_cols])
    std_df.loc[grp.index, score_cols] = std_scores

# Flip MPNN so that higher = better
std_df["MPNN_score"] = -std_df["MPNN_score"]

# Subset for g6 (both hc & lc)
g6_df = std_df[std_df["group"] == "g6"].copy()
g6_df.drop(columns=["dataset", "mutant", "h1_mean", "h3_mean", "h9_mean", "group",
                    "ddG_score", "esmIF_public_score"], inplace=True)
g6_df.dropna(inplace=True)

# Target: norm_binding (standardized)
y_raw = g6_df["norm_binding"].values.reshape(-1, 1)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y_raw).ravel()

# Define feature sets
single_sets = [
    (["MPNN_score"],        "MPNN"),
    (["iPLL_score"],        "esm3_iPLL"),
    (["esmIF_score"],       "esmIF"),
]
pair_sets = [
    (["MPNN_score", "iPLL_score"],   "MPNN + esm3_iPLL"),
    (["MPNN_score", "esmIF_score"],  "MPNN + esmIF"),
    (["iPLL_score", "esmIF_score"],  "esm3_iPLL + esmIF"),
]
triple_sets = [
    (["MPNN_score", "iPLL_score", "esmIF_score"], "MPNN + esm3_iPLL + esmIF"),
]
feature_sets = single_sets + pair_sets + triple_sets

# X positions
single_x = [0, 1, 2]
pair_x   = [3, 4, 5]
triple_x = [6]
x_positions = single_x + pair_x + triple_x

# Evaluate and plot
spearman_correlations = []
labels = []
for features, label in feature_sets:
    X = g6_df[features].values
    model = LinearRegression().fit(X, y)
    corr, _ = spearmanr(y, model.predict(X))
    spearman_correlations.append(corr)
    labels.append(label)
    if not np.all(model.coef_ > 0):
        print(f"WARNING: Coeffs not all positive: {label} -> {model.coef_}")

plt.figure(figsize=(10, 8))
plt.bar(x_positions, spearman_correlations, color="gray", width=0.8)
plt.xticks(x_positions, labels, rotation=45, ha='right', fontsize=14)
plt.ylabel("Spearman Correlation", fontsize=14)
plt.xlabel("Feature Sets", fontsize=14)
plt.title("Predicting Antibody Binding Affinity: g6 dataset (deep mutagenesis)", fontsize=16)
for gx in [2.5, 5.5]:
    plt.axvline(gx, color='black', linestyle='--', alpha=0.6)
plt.ylim(bottom=0)
plt.xticks(fontsize=16); plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()
