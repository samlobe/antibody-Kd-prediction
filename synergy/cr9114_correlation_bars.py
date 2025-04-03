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
df["group"] = df["dataset"].replace({"g6_hc": "g6", "g6_lc": "g6"})

standardized_df = df.copy()
for group_name, group_df in df.groupby("group"):
    scaler = StandardScaler()
    std_scores = scaler.fit_transform(group_df[score_cols])
    standardized_df.loc[group_df.index, score_cols] = std_scores

# Invert MPNN scores so that higher is better
standardized_df["MPNN_score"] = -standardized_df["MPNN_score"]

# Subset cr9114 data
cr9114_df = standardized_df[standardized_df["dataset"] == "cr9114"].copy()
cr9114_df.drop(columns=["dataset", "mutant", "h9_mean", "norm_binding", "group",
                        "ddG_score", "esmIF_public_score"], inplace=True)
cr9114_df.dropna(inplace=True)

# Standardize the target (example: "h1_mean")
y_raw = cr9114_df["h1_mean"].values.reshape(-1, 1)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y_raw).ravel()

# Define feature sets
single_sets = [
    (["MPNN_score"],      "MPNN"),
    (["iPLL_score"],      "esm3_iPLL"),
    (["esmIF_score"],     "esmIF"),
]
pair_sets = [
    (["MPNN_score", "iPLL_score"],    "MPNN + esm3_iPLL"),
    (["MPNN_score", "esmIF_score"],   "MPNN + esmIF"),
    (["iPLL_score", "esmIF_score"],   "esm3_iPLL + esmIF"),
]
triple_sets = [
    (["MPNN_score", "iPLL_score", "esmIF_score"],  "MPNN + esm3_iPLL + esmIF"),
]
feature_sets = single_sets + pair_sets + triple_sets

# Prepare bar positions
single_x = [0, 1, 2]
pair_x   = [3, 4, 5]
triple_x = [6]
x_positions = single_x + pair_x + triple_x

# Build and evaluate linear models
spearman_correlations = []
labels = []
for features, label in feature_sets:
    X = cr9114_df[features].values
    model = LinearRegression().fit(X, y)
    corr, _ = spearmanr(y, model.predict(X))
    spearman_correlations.append(corr)
    labels.append(label)
    # Check sign of coefficients
    if not np.all(model.coef_ > 0):
        print(f"WARNING: Coefficients not all positive: {label} -> {model.coef_}")

# Plot
plt.figure(figsize=(10, 8))
plt.bar(x_positions, spearman_correlations, color="gray", width=0.8)
plt.xticks(x_positions, labels, rotation=45, ha='right', fontsize=14)
plt.ylabel("Spearman Correlation", fontsize=16)
plt.xlabel("Feature Sets", fontsize=16)
plt.title("Predicting Antibody Binding Affinity: cr9114 dataset", fontsize=16)
for gap_x in [2.5, 5.5]:
    plt.axvline(x=gap_x, color='black', linestyle='--', alpha=0.6)
plt.ylim(bottom=0)
plt.xticks(fontsize=16); plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()
