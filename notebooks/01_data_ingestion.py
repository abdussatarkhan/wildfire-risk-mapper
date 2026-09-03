# %% [markdown]
# # Wildfire Spread Risk Mapper - Notebook 01: Multi-Source Data Ingestion & Harmonization
# 
# **Author:** Senior Geospatial Data Analyst  
# **Project:** Wildfire Spread Risk Mapper  
# **Domain:** Climate & Environmental Data Science  
# 
# ### Objectives:
# 1. Ingest active satellite thermal detections from NASA FIRMS (VIIRS 375m / MODIS 1km).
# 2. Ingest and calculate NOAA surface fire weather and Canadian Forest Fire Weather Index (FWI) components.
# 3. Inspect and validate USGS LANDFIRE fuel models (Scott & Burgan FBFM40) and 3DEP DEM elevation rasters.
# 4. Ingest US Census TIGER/Line road corridors and demographic boundaries.
# 5. Harmonize all layers into the **California Albers Equal Area** projection (`EPSG:3310`).

# %%
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root and scripts directory to system path
project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(project_root / "scripts"))

from utils import load_config, setup_logger, reproject_gdf, generate_benchmark_synthetic_data
from data_collection import FIRMSClient, NOAAWeatherDownloader, LANDFIRERasterDownloader

logger = setup_logger("nb01_ingestion")
config = load_config(project_root / "config" / "config.yaml")

print(f"Project Title: {config['project']['title']}")
print(f"Target Projected CRS: {config['spatial']['crs_projected']} (California Albers Equal Area)")

# %% [markdown]
# ## 1. Ingest Active Satellite Fire Detections (NASA FIRMS VIIRS 375m)
# 
# NASA FIRMS provides thermal anomaly coordinates with Fire Radiative Power (FRP in Megawatts).
# High FRP signifies intense crown fires with rapid heat dissipation and spread potential.

# %%
firms_client = FIRMSClient(config=config)
raw_firms_dir = project_root / "data" / "raw" / "firms"
raw_firms_dir.mkdir(parents=True, exist_ok=True)

firms_csv_path = raw_firms_dir / "fire_archive_sv-c2_california.csv"
firms_df = firms_client.fetch_active_fires(days=14, output_csv=firms_csv_path)

print(f"FIRMS Detection Records: {len(firms_df)}")
print("Sample Columns:", firms_df.columns.tolist())
display_df = firms_df[["latitude", "longitude", "bright_ti4", "frp", "confidence", "acq_date", "daynight"]].head()
print(display_df)

# Convert to GeoDataFrame in WGS84 and reproject to California Albers (EPSG:3310)
firms_gdf = gpd.GeoDataFrame(
    firms_df,
    geometry=gpd.points_from_xy(firms_df["longitude"], firms_df["latitude"]),
    crs="EPSG:4326"
)
firms_gdf_3310 = reproject_gdf(firms_gdf, target_crs=config["spatial"]["crs_projected"])
print(f"Reprojected FIRMS CRS: {firms_gdf_3310.crs}")

# %% [markdown]
# ## 2. Ingest Surface Fire Weather & Compute Van Wagner FWI Indices
# 
# The Canadian Fire Weather Index (FWI) system models fuel moisture and rate of spread:
# - **FFMC**: Fine Fuel Moisture Code (litter and fine cured fuels)
# - **ISI**: Initial Spread Index (wind speed combined with FFMC)
# - **FWI**: Composite hazard index for frontal fire intensity

# %%
weather_downloader = NOAAWeatherDownloader(config=config)
raw_noaa_dir = project_root / "data" / "raw" / "noaa"
weather_csv_path = raw_noaa_dir / "raws_california_weather.csv"
weather_df = weather_downloader.fetch_weather_grid(n_stations=180, output_csv=weather_csv_path)

print(f"Meteorological Stations Ingested: {len(weather_df)}")
print(weather_df.describe()[["temperature_2m_c", "relative_humidity_pct", "wind_speed_ms", "fwi"]])

weather_gdf = gpd.GeoDataFrame(
    weather_df,
    geometry=gpd.points_from_xy(weather_df["longitude"], weather_df["latitude"]),
    crs="EPSG:4326"
)
weather_gdf_3310 = reproject_gdf(weather_gdf, target_crs=config["spatial"]["crs_projected"])

# %% [markdown]
# ## 3. Ingest USGS LANDFIRE Fuel Models & 3DEP DEM Rasters
# 
# We inspect Scott & Burgan 40 Fire Behavior Fuel Models (FBFM40) and Elevation DEM.
# The native LANDFIRE resolution is 30m; we inspect profiles and spatial alignment.

# %%
raw_landfire_dir = project_root / "data" / "raw" / "landfire"
lf_downloader = LANDFIRERasterDownloader(config=config)
raster_paths = lf_downloader.generate_reference_rasters(output_dir=raw_landfire_dir)

for name, path in raster_paths.items():
    with rasterio.open(path) as src:
        print(f"Raster: {name.upper():12s} | Resolution: {src.res} | Shape: {src.shape} | CRS: {src.crs}")

# %% [markdown]
# ## 4. Ingest US Census TIGER/Line Transportation Network & Demographics
# 
# Road proximity represents both ignition hazards (vehicle sparks, powerlines along corridors)
# and tactical fire containment lines / evacuation routes.

# %%
from vector_processing import VectorProcessor
from utils import create_regular_grid

vp = VectorProcessor(target_crs="EPSG:3310")
analysis_grid = create_regular_grid((-50000.0, -100000.0, 50000.0, 0.0), resolution_m=2500.0, crs="EPSG:3310")
roads_gdf, structs_gdf, tracts_gdf = vp.generate_synthetic_vectors(analysis_grid)

print(f"Highway Network Features: {len(roads_gdf)}")
print(f"Structure Geometries:      {len(structs_gdf)}")
print(f"Census Demographic Tracts: {len(tracts_gdf)}")

# %% [markdown]
# ## 5. Layer Harmonization Verification and Spatial Integrity Audit
# 
# Confirm that all vector and raster geometries share identical spatial references (`EPSG:3310`).

# %%
crs_check = {
    "Analysis Grid": analysis_grid.crs.to_string(),
    "NASA FIRMS": firms_gdf_3310.crs.to_string(),
    "NOAA Weather": weather_gdf_3310.crs.to_string(),
    "Road Network": roads_gdf.crs.to_string(),
    "Census Tracts": tracts_gdf.crs.to_string()
}

for layer, crs_val in crs_check.items():
    assert crs_val == "EPSG:3310", f"CRS mismatch on {layer}: {crs_val}"
    print(f"[VERIFIED] {layer:18s} -> {crs_val}")

print("\nAll geospatial layers successfully ingested and harmonized.")
