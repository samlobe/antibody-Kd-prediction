#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

# Load dataset and standardize scores
df = pd.read_csv("Kd_data_scores.csv")
score_cols = ["MPNN_score", "esmIF_score", "esmIF_public_score", "iPLL_score", "ddG_score"]
df["group"] = df["dataset"].replace({"g6_HC": "g6", "g6_LC": "g6"})

std_df = df.copy()
for group_name, group_df in df.groupby("group"):
    scaler = StandardScaler()
    std_scores = scaler.fit_transform(group_df[score_cols])
    std_df.loc[group_df.index, score_cols] = std_scores

# Invert MPNN and ddG so that higher values are "better"
std_df["MPNN_score"] = -std_df["MPNN_score"]
std_df["ddG_score"]  = -std_df["ddG_score"]

# Subset cr6261 data and drop unused columns
cr6261_df = std_df[std_df["dataset"] == "cr6261"].copy()
cr6261_df.drop(columns=["dataset", "mutant", "h3_mean", "norm_binding", 
                        "group", "esmIF_public_score"], inplace=True)
cr6261_df.dropna(inplace=True)

# Define feature sets, positions, and color scheme
feature_sets = [
    # Single
    (["ddG_score"],        "ddG"),
    (["iPLL_score"],       "esm3_iPLL"),
    (["esmIF_score"],      "esmIF"),
    (["MPNN_score"],       "MPNN"),
    # Pairs
    (["ddG_score", "iPLL_score"],   "ddG + esm3_iPLL"),
    (["ddG_score", "esmIF_score"],  "ddG + esmIF"),
    (["ddG_score", "MPNN_score"],   "ddG + MPNN"),
    (["iPLL_score", "esmIF_score"], "esm3_iPLL + esmIF"),
    (["iPLL_score", "MPNN_score"],  "esm3_iPLL + MPNN"),
    (["esmIF_score", "MPNN_score"], "esmIF + MPNN"),
    # Triplets
    (["ddG_score", "iPLL_score", "esmIF_score"],    "ddG + esm3_iPLL + esmIF"),
    (["ddG_score", "iPLL_score", "MPNN_score"],     "ddG + esm3_iPLL + MPNN"),
    (["ddG_score", "esmIF_score", "MPNN_score"],    "ddG + esmIF + MPNN"),
    (["iPLL_score", "esmIF_score", "MPNN_score"],   "esm3_iPLL + esmIF + MPNN"),
    # All four
    (["ddG_score", "iPLL_score", "esmIF_score", "MPNN_score"], 
     "ddG + esm3_iPLL + esmIF + MPNN"),
]

single_x  = [0, 1, 2, 3]    # 4 singles
pair_x    = [4, 5, 6, 7, 8, 9]  # 6 pairs
triple_x  = [10, 11, 12, 13]    # 4 triples
all_x     = [14]                # 1 "all"
x_positions = single_x + pair_x + triple_x + all_x

def assign_color(feats):
    has_ddg = ("ddG_score" in feats)
    has_ipl = ("iPLL_score" in feats)
    if has_ddg and not has_ipl:
        return 'tab:blue'
    elif has_ipl and not has_ddg:
        return 'tab:red'
    elif has_ddg and has_ipl:
        return 'tab:purple'
    else:
        return 'gray'

# Function to run analysis for a given target, and produce a plot
def analyze_and_plot(df_subset, target_col, title_str, y_lim_bottom=0.0):
    # Standardize target
    y_raw = df_subset[target_col].values.reshape(-1, 1)
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y_raw).flatten()

    spearman_correlations = []
    labels = []
    bar_colors = []

    for feats, lbl in feature_sets:
        X = df_subset[feats].values
        model = LinearRegression().fit(X, y)
        corr, _ = spearmanr(y, model.predict(X))
        spearman_correlations.append(corr)
        labels.append(lbl)
        bar_colors.append(assign_color(feats))

        # are coefficients positive?
        if not np.all(model.coef_ > 0):
            print(f"WARNING: Coeffs not all positive: {lbl} -> {model.coef_}")

    plt.figure(figsize=(12, 8))
    plt.bar(x_positions, spearman_correlations, color=bar_colors, width=0.8)
    plt.xticks(x_positions, labels, rotation=45, ha='right', fontsize=14)
    plt.ylabel("Spearman Correlation", fontsize=16)
    plt.xlabel("Feature Sets", fontsize=16)
    plt.title(title_str, fontsize=20)
    for gap_x in [3.5, 9.5, 13.5]:
        plt.axvline(x=gap_x, color='black', linestyle='--', alpha=0.6)
    plt.ylim(bottom=y_lim_bottom)
    plt.xticks(fontsize=16); plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

# Produce two plots for cr6261: one for h1_mean, one for h9_mean
analyze_and_plot(cr6261_df, "h1_mean", "Predicting Binding Affinity: cr6261 binding H1 influenza", y_lim_bottom=0.4)
analyze_and_plot(cr6261_df, "h9_mean", "Predicting Binding Affinity: cr6261 binding H9 influenza", y_lim_bottom=0.3)
