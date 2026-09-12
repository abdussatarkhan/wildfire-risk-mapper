# %% [markdown]
# # Wildfire Spread Risk Mapper - Notebook 03: Raster Algebra, Topography & Zonal Statistics
# 
# **Author:** Senior Geospatial Data Analyst  
# **Focus:** Topographic Modeling via Rasterio, Terrain Derivatives & Zonal Metrics
# 
# ### Analytical Pipeline:
# 1. Inspect and read USGS 3DEP Digital Elevation Model (DEM) and LANDFIRE rasters.
# 2. Compute finite-difference gradient vectors to derive slope, aspect, and terrain ruggedness (TRI).
# 3. Resample and reproject continuous and categorical rasters to a standardized 250m grid.
# 4. Perform vector-raster zonal statistics across land management planning units.

### Terrain Derivation Mathematics (Horn's Method):
Given a $3 \times 3$ moving window of elevation cells $\begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix}$:
- Partial gradients:
  $$p = \frac{\partial z}{\partial x} = \frac{(c + 2f + i) - (a + 2d + g)}{8 \Delta x}, \quad q = \frac{\partial z}{\partial y} = \frac{(g + 2h + i) - (a + 2b + c)}{8 \Delta y}$$
- Slope angle: $\theta = \arctan\left(\sqrt{p^2 + q^2}\right) \times \frac{180^\circ}{\pi}$
- Topographic Roughness Index (TRI): $\text{TRI} = \sqrt{\frac{1}{8} \sum_{j \neq e} (z_j - z_e)^2}$

# %%
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt

# Add project root
project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(project_root / "scripts"))

from utils import setup_logger, load_config, save_geotiff
from raster_processing import RasterProcessor
from data_collection import LANDFIRERasterDownloader

logger = setup_logger("nb03_raster")
config = load_config(project_root / "config" / "config.yaml")

# Ensure reference rasters exist
raw_raster_dir = project_root / "data" / "raw" / "landfire"
lf = LANDFIRERasterDownloader(config=config)
raster_paths = lf.generate_reference_rasters(output_dir=raw_raster_dir)

# %% [markdown]
# ## 1. Ingest DEM and Inspect Geospatial Affine Transform
# 
# Read the elevation surface and verify spatial affine matrices.

# %%
dem_path = raster_paths["dem"]

with rasterio.open(dem_path) as src:
    dem_meta = src.meta
    dem_data = src.read(1)
    transform = src.transform
    crs = src.crs

print(f"DEM Grid Dimensions: {dem_data.shape} (Rows x Cols)")
print(f"Cell Resolution:     {src.res[0]:.1f}m x {src.res[1]:.1f}m")
print(f"CRS:                 {crs}")
print(f"Elevation Range:     {np.nanmin(dem_data):.1f}m to {np.nanmax(dem_data):.1f}m")
print(f"Affine Transform:\n{transform}")

# %% [markdown]
# ## 2. Raster Algebra: Topographic Derivations (Horn's Algorithm)
# 
# Derive slope, aspect, and local ruggedness index (TRI).
# Steep slopes accelerate convective heat transfer upwards, drastically increasing flame spread rates.

# %%
processor = RasterProcessor(target_crs="EPSG:3310", target_res_m=250.0)
proc_dir = project_root / "data" / "processed"
proc_dir.mkdir(parents=True, exist_ok=True)

topo_rasters = processor.compute_topography(dem_path=dem_path, output_dir=proc_dir)

with rasterio.open(topo_rasters["slope"]) as src:
    slope_data = src.read(1)

with rasterio.open(topo_rasters["aspect"]) as src:
    aspect_data = src.read(1)

with rasterio.open(topo_rasters["tri"]) as src:
    tri_data = src.read(1)

print("Topographic layers successfully derived.")

# %% [markdown]
# ## 3. Cartographic Surface Visualization
# 
# Visualizing DEM elevation, slope steepness, aspect direction, and ruggedness index.

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Elevation DEM
im0 = axes[0, 0].imshow(dem_data, cmap="terrain")
axes[0, 0].set_title("Digital Elevation Model (DEM, meters)", fontweight="bold")
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

# Slope
im1 = axes[0, 1].imshow(slope_data, cmap="YlOrRd")
axes[0, 1].set_title("Terrain Slope Angle (Degrees)", fontweight="bold")
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

# Aspect
im2 = axes[1, 0].imshow(aspect_data, cmap="twilight")
axes[1, 0].set_title("Terrain Aspect (0-360 Azimuth)", fontweight="bold")
plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

# TRI
im3 = axes[1, 1].imshow(tri_data, cmap="magma")
axes[1, 1].set_title("Terrain Ruggedness Index (TRI)", fontweight="bold")
plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Vector Zonal Statistics Extraction
# 
# Aggregating mean elevation and maximum slope across administrative or ecological units.

# %%
from utils import create_regular_grid

sample_zones = create_regular_grid((-50000.0, -100000.0, 50000.0, 0.0), resolution_m=15000.0, crs="EPSG:3310")
zones_with_stats = processor.compute_zonal_statistics(
    raster_path=topo_rasters["slope"],
    polygons_gdf=sample_zones,
    stat_prefix="slope",
    stats=("mean", "max", "std")
)

print("Sample Zonal Statistics Output:")
print(zones_with_stats[["cell_id", "slope_mean", "slope_max", "slope_std"]].head())

# %% [markdown]
# ## 5. Fuel Model Raster Inspection (LANDFIRE FBFM40)
# 
# Existing vegetation and Scott & Burgan fuel model distributions.

# %%
fbfm_path = raster_paths["fbfm40"]
with rasterio.open(fbfm_path) as src:
    fbfm_data = src.read(1)

unique_fuels, fuel_counts = np.unique(fbfm_data, return_counts=True)
fuel_df = pd.DataFrame({"Fuel_Code": unique_fuels, "Pixel_Count": fuel_counts})
fuel_labels = {
    102: "Grass (Low Load)",
    122: "Grass-Shrub (Moderate)",
    142: "Shrub / Chaparral (High)",
    161: "Timber Understory",
    181: "Timber Litter",
    99: "Non-burnable / Urban / Water"
}
fuel_df["Description"] = fuel_df["Fuel_Code"].map(fuel_labels)
fuel_df["Percent"] = np.round((fuel_df["Pixel_Count"] / fuel_df["Pixel_Count"].sum()) * 100, 2)
print("LANDFIRE FBFM40 Fuel Model Breakdown:")
print(fuel_df)

print("\nRaster analysis and terrain characterization completed.")
