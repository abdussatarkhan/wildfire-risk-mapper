# %% [markdown]
# # Wildfire Spread Risk Mapper - Notebook 05: Cartographic Synthesis & 5-Tier Risk Classification
# 
# **Author:** Senior Geospatial Data Analyst  
# **Focus:** Landscape Inference, 5-Tier Hazard Discretization, Moran's I Verification & GeoTIFF Cartography
# 
# ### Objectives:
# 1. Generate full-coverage continuous wildfire spread probabilities across California landscape units.
# 2. Classify landscape cells into 5 authoritative hazard tiers (Low, Moderate, High, Very High, Extreme).
# 3. Verify spatial autocorrelation of risk surface using Global Moran's I.
# 4. Export dual-band GeoTIFF hazard surface and interactive GeoJSON summaries.

# %%
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root
project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(project_root / "scripts"))

from utils import setup_logger, load_config, save_geotiff
from risk_map_generation import RiskMapGenerator
from spatial_analysis import SpatialAutocorrelationAnalyzer

logger = setup_logger("nb05_risk_mapping")
config = load_config(project_root / "config" / "config.yaml")

# %% [markdown]
# ## 1. Load Trained Model and Predict Continuous Fire Probabilities

# %%
generator = RiskMapGenerator(config=config)
model, feature_names, grid_gdf = generator.load_model_and_data()

scored_gdf = generator.predict_risk_tiers(model, feature_names, grid_gdf)
print("Scored Dataset Sample:")
print(scored_gdf[["cell_id", "predicted_probability", "risk_tier", "risk_label", "action_guidance"]].head())

# %% [markdown]
# ## 2. Spatial Autocorrelation Verification (Moran's I)
# 
# A physically realistic wildfire hazard model must exhibit strong spatial continuity (high Moran's I),
# reflecting contiguous regional weather patterns and fuel complexes rather than high-frequency spatial noise.

# %%
moran_analyzer = SpatialAutocorrelationAnalyzer(k_neighbors=8)
moran_results = moran_analyzer.compute_global_morans_i(scored_gdf, attribute="predicted_probability")

print("Spatial Autocorrelation Audit:")
print(f"  Global Moran's I:     {moran_results['morans_i']}")
print(f"  Simulated p-value:    {moran_results['p_value_simulated']}")
print(f"  Z-Score:              {moran_results['z_score']}")
print(f"  Diagnostic Summary:   {moran_results['interpretation']}")

# %% [markdown]
# ## 3. Landscape Hazard Distribution & Acreage Exposure
# 
# Tabulate total land area and percentage allocation across risk tiers.

# %%
cell_area_km2 = (config["spatial"]["grid_resolution_meters"] ** 2) / 1e6
tier_summary = scored_gdf.groupby(["risk_tier", "risk_label"]).agg(
    Cell_Count=("cell_id", "count"),
    Mean_Prob=("predicted_probability", "mean"),
    Mean_FWI=("fwi_score", "mean")
).reset_index()

tier_summary["Area_km2"] = tier_summary["Cell_Count"] * cell_area_km2
tier_summary["Area_Acreage"] = tier_summary["Area_km2"] * 247.105
tier_summary["Percentage"] = np.round((tier_summary["Cell_Count"] / len(scored_gdf)) * 100, 1)

print("Landscape Hazard Allocation Table:")
print(tier_summary)

# %% [markdown]
# ## 4. Dual-Band GeoTIFF Raster Export
# 
# Export Band 1 (Probability) and Band 2 (Risk Tier) to `data/processed/california_wildfire_risk_250m.tif`.

# %%
out_tif = generator.export_risk_geotiff(scored_gdf)
print(f"Exported Dual-Band GeoTIFF to: {out_tif}")

# Verify exported raster
with rasterio.open(out_tif) as src:
    print(f"GeoTIFF Dimensions: {src.width}x{src.height} | Bands: {src.count} | CRS: {src.crs}")
    band1 = src.read(1)
    band2 = src.read(2)
    print(f"Band 1 (Probability) Range: [{band1[band1 > 0].min():.3f}, {band1.max():.3f}]")
    print(f"Band 2 (Risk Tiers) Unique:  {np.unique(band2)}")

# %% [markdown]
# ## 5. Publication-Ready Cartographic Map Render
# 
# Plot the 5-tier wildfire risk map with operational symbology.

# %%
tier_colors = {
    "Low": "#2ca25f",
    "Moderate": "#ffeb3b",
    "High": "#fe9929",
    "Very High": "#d95f0e",
    "Extreme": "#990000"
}

fig, ax = plt.subplots(figsize=(11, 10))
for label, color in tier_colors.items():
    subset = scored_gdf[scored_gdf["risk_label"] == label]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, edgecolor="#444444", linewidth=0.2, alpha=0.85, label=label)

legend_patches = [mpatches.Patch(color=color, label=f"{label} Hazard") for label, color in tier_colors.items()]
ax.legend(handles=legend_patches, loc="upper right", frameon=True, framealpha=0.9, title="Wildfire Hazard Tiers")

ax.set_title("California Wildfire Spread Risk Map (250m Grid, EPSG:3310)", fontsize=13, fontweight="bold")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
plt.tight_layout()

img_out = project_root / "images" / "california_wildfire_risk_map.png"
plt.savefig(img_out, dpi=200)
plt.show()
print(f"Saved publication map render to: {img_out}")

# %% [markdown]
# ## 6. Export Summary GeoJSON for Interactive Web Application
# 
# Export optimized WGS84 GeoJSON for Folium / Leaflet web deployment.

# %%
geojson_path = generator.export_geojson_summary(scored_gdf)
print(f"Interactive GeoJSON layer generated at: {geojson_path}")
print("Notebook 05 completed successfully.")
