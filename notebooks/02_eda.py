# %% [markdown]
# # Wildfire Spread Risk Mapper - Notebook 02: Exploratory Spatial Data Analysis (ESDA)
# 
# **Author:** Senior Geospatial Data Analyst  
# **Focus:** Fire Perimeter Mapping, Seasonal Patterns, and Spatial Covariate Distributions
# 
# ### Key Research Questions:
# 1. How do fire ignitions and radiative intensity (FRP) vary across California's seasonal calendar?
# 2. What is the spatial distribution of fires across elevation and slope gradients?
# 3. How strongly do fire weather indices (FWI, RH, Wind) correlate with fire presence?
# 4. Does human proximity (distance to roads) exhibit a spatial decay relationship with ignition probability?

# %%
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10

# Add project root
project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(project_root / "scripts"))

from utils import load_config, setup_logger, generate_benchmark_synthetic_data

logger = setup_logger("nb02_eda")
config = load_config(project_root / "config" / "config.yaml")

# Load benchmark geospatial dataset
grid_gdf, firms_gdf, road_gdf = generate_benchmark_synthetic_data(num_grid_cells=1200, random_seed=42)
print(f"Loaded Grid Units: {len(grid_gdf)} | Positive Burns: {grid_gdf['fire_occurrence'].sum()}")
print(f"Loaded Satellite Detections: {len(firms_gdf)} points")

# %% [markdown]
# ## 1. Temporal Seasonality and Diurnal Radiative Intensity
# 
# In Mediterranean California climates, wildfire activity surges during late summer and autumn
# as live and dead fuel moisture reaches critical minima, often exacerbated by offshore wind events (Diablo/Santa Ana).

# %%
firms_gdf["acq_datetime"] = pd.to_datetime(firms_gdf["acq_date"])
firms_gdf["month"] = firms_gdf["acq_datetime"].dt.month_name()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot FRP distribution (Log-normal behavior)
sns.histplot(firms_gdf["frp_mw"], bins=30, kde=True, ax=axes[0], color="#d95f0e")
axes[0].set_title("Distribution of Fire Radiative Power (FRP in Megawatts)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("FRP (MW)")
axes[0].set_ylabel("Detection Count")

# Plot Confidence vs FRP
sns.boxplot(x=pd.cut(firms_gdf["confidence"], bins=[50, 75, 90, 100], labels=["Nominal (50-75)", "High (75-90)", "Extreme (90-100)"]),
            y="frp_mw", data=firms_gdf, ax=axes[1], palette="YlOrRd")
axes[1].set_title("Thermal Anomaly FRP by Sensor Confidence Tier", fontsize=12, fontweight="bold")
axes[1].set_xlabel("VIIRS Detection Confidence")
axes[1].set_ylabel("FRP (MW)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Topographic and Fuel Moisture Correlations
# 
# Analyzing how physical slope steepness and the Canadian Fire Weather Index influence fire occurrence.

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Slope vs Fire Occurrence
sns.boxplot(x="fire_occurrence", y="slope_degrees", data=grid_gdf, ax=axes[0], palette=["#2ca25f", "#de2d26"])
axes[0].set_xticklabels(["Unburned (0)", "Burned (1)"])
axes[0].set_title("Terrain Slope vs. Fire Spread", fontweight="bold")
axes[0].set_ylabel("Slope (Degrees)")

# FWI Score vs Fire Occurrence
sns.boxplot(x="fire_occurrence", y="fwi_score", data=grid_gdf, ax=axes[1], palette=["#2ca25f", "#de2d26"])
axes[1].set_xticklabels(["Unburned (0)", "Burned (1)"])
axes[1].set_title("Fire Weather Index (FWI) vs. Fire Spread", fontweight="bold")
axes[1].set_ylabel("FWI Score")

# Canopy Cover vs Fire Occurrence
sns.boxplot(x="fire_occurrence", y="canopy_cover_pct", data=grid_gdf, ax=axes[2], palette=["#2ca25f", "#de2d26"])
axes[2].set_xticklabels(["Unburned (0)", "Burned (1)"])
axes[2].set_title("Canopy Cover Fuel vs. Fire Spread", fontweight="bold")
axes[2].set_ylabel("Canopy Cover (%)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Anthropogenic Proximity: Distance-to-Roads Analysis
# 
# Human ignitions (power lines, catalytic converters, arson, escaped campfires) exhibit strong
# spatial concentration within 1,000 meters of transportation networks.

# %%
grid_gdf["dist_km"] = grid_gdf["dist_to_road_m"] / 1000.0
grid_gdf["dist_bin"] = pd.cut(grid_gdf["dist_km"], bins=[0, 2, 5, 10, 20, 50], labels=["0-2 km", "2-5 km", "5-10 km", "10-20 km", "20+ km"])

road_burn_rate = grid_gdf.groupby("dist_bin", observed=False)["fire_occurrence"].mean().reset_index()

plt.figure(figsize=(10, 4))
sns.barplot(x="dist_bin", y="fire_occurrence", data=road_burn_rate, palette="Reds_r")
plt.title("Empirical Wildfire Occurrence Rate by Distance to Primary Road Network", fontweight="bold", fontsize=12)
plt.xlabel("Distance to Road Corridor (km)")
plt.ylabel("Observed Fire Rate (Ignition / Spread)")
plt.ylim(0, max(road_burn_rate["fire_occurrence"]) * 1.3)
for idx, row in road_burn_rate.iterrows():
    plt.text(idx, row["fire_occurrence"] + 0.01, f"{row['fire_occurrence']*100:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Covariate Correlation Matrix
# 
# Inspecting collinearity among multi-scale predictors before model training.

# %%
num_cols = [
    "elevation_m", "slope_degrees", "temp_max_c", "relative_humidity_min",
    "wind_speed_ms", "fwi_score", "canopy_cover_pct", "dist_to_road_m", "fire_occurrence"
]
corr_matrix = grid_gdf[num_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title("Pearson Correlation Heatmap of Environmental & Anthropogenic Covariates", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Spatial Perimeter & Hotspot Landscape Visualization
# 
# Mapping grid units with burned occurrence and active satellite thermal points.

# %%
fig, ax = plt.subplots(figsize=(10, 10))

# Plot background grid colored by FWI
grid_gdf.plot(column="fwi_score", cmap="YlOrBr", alpha=0.6, edgecolor="#cccccc", linewidth=0.2, ax=ax, legend=True,
              legend_kwds={"label": "Fire Weather Index (FWI)", "orientation": "horizontal", "shrink": 0.6})

# Overlay burned cells
burned = grid_gdf[grid_gdf["fire_occurrence"] == 1]
burned.plot(ax=ax, facecolor="none", edgecolor="#b30000", linewidth=1.5, label="Observed Burn Event")

# Overlay roads
road_gdf.plot(ax=ax, color="#111111", linewidth=2.0, label="Highway Corridor")

# Overlay thermal detections
firms_gdf.plot(ax=ax, color="#ff0000", markersize=firms_gdf["frp_mw"] / 8.0, alpha=0.8, label="VIIRS Active Fires (Scaled by FRP)")

ax.set_title("Geospatial Landscape: Fire Weather Index, Burn Perimeters & VIIRS Hotspots (EPSG:3310)", fontweight="bold", fontsize=12)
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

print("Exploratory Spatial Data Analysis complete.")
