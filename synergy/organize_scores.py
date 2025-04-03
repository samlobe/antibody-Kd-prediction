#%%
import pandas as pd
from pathlib import Path

# Set the base path for organization
base_path = Path(".")

# Dataset mapping with desired columns from exp_data
datasets = {
    "cr6261": {
        "exp_path": base_path / "exp_data" / "cr6261_exp_data.csv",
        "exp_cols": ["genotype", "h1_mean", "h9_mean"],
        "predictors": {
            "esmIF": base_path / "esmIF_scores" / "cr6261_esmIF.csv",
            "esmIF_public": base_path / "esmIF_scores" / "cr6261_esmIF_public.csv",
            "MPNN": base_path / "mpnn_scores" / "cr6261_mpnn.csv",
            "iPLL": base_path / "esm3-large-multimer_iPLL_scores" / "iPLL_cr6261_cutoff1.csv",
            "ddG": base_path / "ddG_scores" / "ddG_avg_cr6261.csv"
        }
    },
    "cr9114": {
        "exp_path": base_path / "exp_data" / "cr9114_exp_data.csv",
        "exp_cols": ["genotype", "h1_mean", "h3_mean"],
        "predictors": {
            "esmIF": base_path / "esmIF_scores" / "cr9114_esmIF.csv",
            "esmIF_public": base_path / "esmIF_scores" / "cr9114_esmIF_public.csv",
            "MPNN": base_path / "mpnn_scores" / "cr9114_mpnn.csv",
            "iPLL": base_path / "esm3-large-multimer_iPLL_scores" / "iPLL_cr9114_cutoff1.csv"
        }
    },
    "g6_HC": {
        "exp_path": base_path / "exp_data" / "g6_hc_exp_data.csv",
        "exp_cols": ["id", "norm_binding"],
        "predictors": {
            "esmIF": base_path / "esmIF_scores" / "g6_HC_esmIF.csv",
            "esmIF_public": base_path / "esmIF_scores" / "g6_HC_esmIF_public.csv",
            "MPNN": base_path / "mpnn_scores" / "g6_hc_mpnn.csv",
            "iPLL": base_path / "esm3-large-multimer_iPLL_scores" / "iPLL_g6_cutoff1.csv"
        }
    },
    "g6_LC": {
        "exp_path": base_path / "exp_data" / "g6_lc_exp_data.csv",
        "exp_cols": ["id", "norm_binding"],
        "predictors": {
            "esmIF": base_path / "esmIF_scores" / "g6_LC_esmIF.csv",
            "esmIF_public": base_path / "esmIF_scores" / "g6_LC_esmIF_public.csv",
            "MPNN": base_path / "mpnn_scores" / "g6_lc_mpnn.csv",
            "iPLL": base_path / "esm3-large-multimer_iPLL_scores" / "iPLL_g6_cutoff1.csv"
        }
    }
}

# Helper functions
def load_exp(path, cols):
    df = pd.read_csv(path, usecols=cols)
    df = df.rename(columns={cols[0]: "mutant"})
    df["mutant"] = df["mutant"].astype(str)
    return df

def load_pred(path, name):
    df = pd.read_csv(path)
    for col in ["mutant", "id", "seq", "genotype"]:
        if col in df.columns:
            df = df.rename(columns={col: "mutant"})
            break
    df["mutant"] = df["mutant"].astype(str)
    score_cols = [col for col in df.columns if col not in ["mutant", "seq"]]
    assert len(score_cols) == 1, f"Expected 1 score column in {path.name}, got {score_cols}"
    return df[["mutant", score_cols[0]]].rename(columns={score_cols[0]: f"{name}_score"})

# Build master DataFrame
all_dfs = []

for name, info in datasets.items():
    exp_df = load_exp(info["exp_path"], info["exp_cols"])
    for pred_name, pred_path in info["predictors"].items():
        pred_df = load_pred(pred_path, pred_name)
        exp_df = exp_df.merge(pred_df, on="mutant", how="left")
    exp_df["dataset"] = name
    all_dfs.append(exp_df)

final_df = pd.concat(all_dfs, ignore_index=True)
# put 'dataset' column on left
final_df = final_df[["dataset"] + [col for col in final_df.columns if col != "dataset"]]
# put h3_mean column after h1_mean
for i, col in enumerate(final_df.columns):
    if col == "h1_mean":
        h3_mean_index = i + 1
        break
final_df = final_df[[col for col in final_df.columns[:h3_mean_index]] + 
                ["h3_mean"] + 
                [col for col in final_df.columns[h3_mean_index:] if col != "h3_mean"]]
# put norm_binding column after h9_mean
for i, col in enumerate(final_df.columns):
    if col == "h9_mean":
        norm_binding_index = i + 1
        break
final_df = final_df[[col for col in final_df.columns[:norm_binding_index]] + 
                ["norm_binding"] + 
                [col for col in final_df.columns[norm_binding_index:] if col != "norm_binding"]]

# Save the final DataFrame
output_path = base_path / "Kd_data_scores.csv"
final_df.to_csv(output_path, index=False)