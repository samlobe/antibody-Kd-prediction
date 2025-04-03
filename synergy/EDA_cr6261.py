#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

df = pd.read_csv("Kd_data_scores.csv", dtype={"mutant": str})
score_cols = ["MPNN_score", "esmIF_score", "esmIF_public_score", "iPLL_score", "ddG_score"]

# Create a unified dataset label for g6_HC and g6_LC
df["group"] = df["dataset"].replace({"g6_hc": "g6", "g6_lc": "g6"})

standardized_df = df.copy()

for group_name, group_df in df.groupby("group"):
    scaler = StandardScaler()
    standardized_scores = scaler.fit_transform(group_df[score_cols])
    standardized_df.loc[group_df.index, score_cols] = standardized_scores

# multiply mpnn and ddG score by -1 such that higher scores are better
standardized_df["MPNN_score"] = standardized_df["MPNN_score"] * -1
standardized_df["ddG_score"] = standardized_df["ddG_score"] * -1

# for cr6261 dataset, find the correlation between each predictor and the experimental data
cr6261_df = standardized_df[standardized_df["dataset"] == "cr6261"]
cr6261_df = cr6261_df.drop(columns=["dataset", "h3_mean", "norm_binding","group"])
cr6261_df = cr6261_df.dropna()

title = "Antibody Binding Affinity dataset (cr6261, H1): ddG, esm3-iPLL, esmIF, MPNN"
X = cr6261_df[["MPNN_score", "esmIF_score", "iPLL_score", "ddG_score"]]

# title = "Linear Regression: ddG, esmIF, MPNN"
# X = cr6261_df[["MPNN_score", "esmIF_score", "ddG_score"]]

# title = "Linear Regression: esm3-iPLL, esmIF, MPNN"
# X = cr6261_df[["MPNN_score", "esmIF_score", "iPLL_score"]]

# title = "Linear Regression: ddG, esmIF, MPNN"
# X = cr6261_df[["MPNN_score", "esmIF_score", "ddG_score"]]

# title = "Linear Regression: esmIF, MPNN"
# X = cr6261_df[["MPNN_score", "esmIF_score"]]

title = "Antibody Binding Affinity dataset (cr6261, H1)\n Features: ddG, esm3-iPLL"
X = cr6261_df[["iPLL_score", "ddG_score"]]

# title = "Linear Regression: ddG, MPNN"
# X = cr6261_df[["MPNN_score", "ddG_score"]]

# title = "Linear Regression: ddG, esmIF"
# X = cr6261_df[["esmIF_score", "ddG_score"]]

# title = "Linear Regression: MPNN score"
# X = cr6261_df[["MPNN_score"]]

# title = "Linear Regression: esmIF log-likelihood"
# X = cr6261_df[["esmIF_score"]]

# title = "Linear Regression: esm3-large-multimer iPLL (single-pass)"
# X = cr6261_df[["iPLL_score"]]

# title = "Linear Regression: Rosetta ddG"
# X = cr6261_df[["ddG_score"]]

# title = "Linear Regression: MPNN, esmIF"
# X = cr6261_df[["MPNN_score", "esmIF_score"]]

# title = "Linear Regression: esm3-iPLL, esmIF"
# X = cr6261_df[["esmIF_score", "iPLL_score"]]

# title = "Linear Regression: esm3-iPLL, MPNN"
# X = cr6261_df[["MPNN_score", "iPLL_score"]]

y = cr6261_df["h1_mean"]
# standardize y
scaler = StandardScaler()
y = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

# Train Lasso over a range of alpha values and track MSE and coefficients
alphas = np.logspace(-3, 2, 50) 
coefs = []
losses = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    lasso.fit(X, y)
    
    # Collect coefficients
    coefs.append(lasso.coef_)
    
    # Predict on the same data to track the training MSE
    y_pred = lasso.predict(X)
    mse = mean_squared_error(y, y_pred)
    losses.append(mse)

coefs = np.array(coefs)

# Plot how the loss (MSE) changes with alpha
plt.figure()
plt.semilogx(alphas, losses, marker='o')
plt.title("LASSO: MSE vs. Regularization Strength")
plt.xlabel("Alpha (log scale)")
plt.ylabel("Mean Squared Error")
plt.show()

# Plot how the coefficients change as you vary alpha
plt.figure()
for i, col_name in enumerate(X.columns):
    plt.semilogx(alphas, coefs[:, i], label=col_name)

plt.title("LASSO: Coefficients vs. Regularization Strength")
plt.xlabel("Alpha (log scale)")
plt.ylabel("Coefficient Value")
plt.legend()
plt.show()

# do linear regression with just those predictors
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"MSE: {mse}")
print(f"R^2: {r2}")

# find spearman correlation between y and y_pred
corr, p_value = spearmanr(y, y_pred)
print(f"Spearman correlation: {corr}")

# Plot the predictions vs. actual values
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Actual Values (standardized)", fontsize=16)
plt.ylabel("Predicted Values (standardized)", fontsize=16)
plt.title(f'{title}\nSpearman correlation: {corr:.2f}', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#%%
# Add mutation count column
cr6261_df["mutation_count"] = cr6261_df.loc[cr6261_df.index, "mutant"].apply(lambda x: str(x).count("1"))
# Note: ddG of the fully mutated one (11 mutations) was one of the 125/1916 that didn't get scored by ddG

# Plot the predictions vs. actual values with color based on mutation count
plt.figure(figsize=(10, 6))
scatter = plt.scatter(y, y_pred, c=cr6261_df["mutation_count"], cmap="plasma", alpha=0.8)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Actual Values (standardized)", fontsize=16)
plt.ylabel("Predicted Values (standardized)", fontsize=16)
plt.title(f'{title}\nSpearman correlation: {corr:.2f}', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
cbar = plt.colorbar(scatter)
cbar.ax.tick_params(labelsize=16)
cbar.set_label("mutations from germline", fontsize=16)
plt.show()

#%%
# Get original (unstandardized) y values
y_actual = df.loc[cr6261_df.index, "h1_mean"].values

# Fit the model using standardized X, but keep y as original (unstandardized)
model = LinearRegression()
model.fit(X, y_actual)
y_pred_original = model.predict(X)

# Mutation count from original df
cr6261_df["mutation_count"] = df.loc[cr6261_df.index, "mutant"].apply(lambda x: str(x).count("1"))

# Spearman correlation on unstandardized data
corr, p_value = spearmanr(y_actual, y_pred_original)

# Plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(y_actual, y_pred_original, c=cr6261_df["mutation_count"], cmap="plasma", alpha=0.8, vmax=10)
plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], 'k--', lw=2)
plt.xlabel(r"Actual (– log $K_D$)", fontsize=16)
plt.ylabel(r"Predicted (– log $K_D$)", fontsize=16)
plt.title(f'{title}\nSpearman correlation: {corr:.2f}', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label("Mutations from Germline", fontsize=16)
plt.tight_layout()
plt.savefig("cr6261_ddG_esm3-iPLL.png", dpi=300)
plt.show()

# print the coefficients
coefficients = model.coef_
feature_names = X.columns
print("Coefficients:")
for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.4f}")

#%%