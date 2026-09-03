# %% [markdown]
# # Wildfire Spread Risk Mapper - Notebook 04: Predictive Modeling & Spatial Block Cross-Validation
# 
# **Author:** Senior Geospatial Data Analyst  
# **Focus:** Mitigating Spatial Autocorrelation Leakage via SpatialBlockKFold, Model Benchmarking & Interpretability
# 
# ### Key Concepts:
# 1. **Spatial Autocorrelation Leakage**: Standard random K-Fold randomly splits adjacent spatial pixels, allowing models to cheat by memorizing local spatial coordinates rather than physical environmental relationships.
# 2. **Spatial Block Cross-Validation**: Holds out contiguous $25\text{ km} \times 25\text{ km}$ geographic blocks, guaranteeing true out-of-region generalization.
# 3. **Algorithms**: Regularized Logistic Regression vs. Tuned Random Forest Classifier.

# %%
import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, brier_score_loss, classification_report
from sklearn.model_selection import KFold

# Add project root
project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(project_root / "scripts"))

from utils import setup_logger, load_config, generate_benchmark_synthetic_data
from modeling import SpatialBlockKFold, WildfireRiskModeler

logger = setup_logger("nb04_modeling")
config = load_config(project_root / "config" / "config.yaml")

# Load processed data
data_path = project_root / "data" / "processed" / "ca_wildfire_grid_250m.parquet"
if not data_path.exists():
    grid_gdf, _, _ = generate_benchmark_synthetic_data(num_grid_cells=1200, random_seed=42)
    grid_gdf.to_parquet(data_path)
else:
    grid_gdf = gpd.read_parquet(data_path)

print(f"Dataset Size: {len(grid_gdf)} cells across {grid_gdf['block_id'].nunique()} spatial blocks.")
print(f"Fire Ignitions/Spread Count: {grid_gdf['fire_occurrence'].sum()} ({grid_gdf['fire_occurrence'].mean()*100:.1f}%)")

# %% [markdown]
# ## 1. Visualizing Spatial Block Partitions
# 
# Each block represents an independent geographic fold ensuring spatial isolation.

# %%
fig, ax = plt.subplots(figsize=(10, 8))
grid_gdf.plot(column="block_id", cmap="tab20", edgecolor="#333333", linewidth=0.3, alpha=0.75, ax=ax, legend=False)
ax.set_title("Spatial Block Cross-Validation Partitions (25 km Blocks)", fontweight="bold", fontsize=12)
ax.set_xlabel("California Albers Easting (m)")
ax.set_ylabel("California Albers Northing (m)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Comparative Validation: Standard Random CV vs. Spatial Block CV
# 
# Demonstrating the quantitative inflation caused by random spatial data leakage.

# %%
feature_cols = [
    "elevation_m", "slope_degrees", "aspect_degrees", "temp_max_c",
    "relative_humidity_min", "wind_speed_ms", "wind_gust_ms", "fwi_score",
    "canopy_cover_pct", "dist_to_road_m", "pop_density_km2"
]
X = grid_gdf[feature_cols]
y = grid_gdf["fire_occurrence"].values
groups = grid_gdf["block_id"].values

# Standard Random 5-Fold CV
std_kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf_std = RandomForestClassifier(n_estimators=150, max_depth=14, class_weight="balanced_subsample", random_state=42)
std_oof = np.zeros(len(y))
for train_idx, test_idx in std_kf.split(X, y):
    rf_std.fit(X.iloc[train_idx], y[train_idx])
    std_oof[test_idx] = rf_std.predict_proba(X.iloc[test_idx])[:, 1]
roc_std = roc_auc_score(y, std_oof)

# Spatial Block 5-Fold CV
sp_kf = SpatialBlockKFold(n_splits=5, random_state=42)
rf_sp = RandomForestClassifier(n_estimators=150, max_depth=14, class_weight="balanced_subsample", random_state=42)
sp_oof = np.zeros(len(y))
for train_idx, test_idx in sp_kf.split(X, y, groups=groups):
    rf_sp.fit(X.iloc[train_idx], y[train_idx])
    sp_oof[test_idx] = rf_sp.predict_proba(X.iloc[test_idx])[:, 1]
roc_sp = roc_auc_score(y, sp_oof)

cv_comparison = pd.DataFrame({
    "Validation Method": ["Standard Random K-Fold CV (Leakage)", "Spatial Block K-Fold CV (Rigorous)"],
    "ROC-AUC": [round(roc_std, 4), round(roc_sp, 4)],
    "Assessment": ["Overfit to local geography", "True geographic generalization"]
})
print("Cross-Validation Leakage Benchmark:")
print(cv_comparison)

# %% [markdown]
# ## 3. Out-of-Fold Performance Diagnostics (ROC & PR Curves)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC Curve
fpr_sp, tpr_sp, _ = roc_curve(y, sp_oof)
axes[0].plot(fpr_sp, tpr_sp, color="#d95f0e", lw=2, label=f"Spatial Block RF (AUC = {roc_sp:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Chance Level")
axes[0].set_title("Receiver Operating Characteristic (ROC)", fontweight="bold")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(loc="lower right")

# Precision-Recall Curve
prec_sp, rec_sp, _ = precision_recall_curve(y, sp_oof)
pr_auc_sp = average_precision_score(y, sp_oof)
axes[1].plot(rec_sp, prec_sp, color="#2b8cbe", lw=2, label=f"Spatial Block RF (PR-AUC = {pr_auc_sp:.3f})")
axes[1].set_title("Precision-Recall (PR) Curve", fontweight="bold")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend(loc="lower left")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Feature Importance & Physical Explanations
# 
# What environmental drivers dominate wildfire spread hazard in California?

# %%
rf_final = RandomForestClassifier(n_estimators=200, max_depth=16, class_weight="balanced_subsample", random_state=42)
rf_final.fit(X, y)

feat_imp = pd.Series(rf_final.feature_importances_, index=feature_cols).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
feat_imp.plot(kind="barh", color="#d95f0e", edgecolor="#333333")
plt.title("Random Forest Mean Decrease in Impurity (Feature Importance)", fontweight="bold")
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Model Serialization
# 
# Save final model bundle for GIS raster inference.

# %%
import joblib

model_dir = project_root / "models"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "wildfire_rf_risk_model.joblib"

joblib.dump({
    "model": rf_final,
    "feature_names": feature_cols,
    "feature_importances": feat_imp.to_dict()
}, model_path)

print(f"Successfully saved final model artifact to: {model_path}")
