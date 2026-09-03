"""
Utility functions and shared helpers for Wildfire Spread Risk Mapper.
Provides configuration loading, coordinate system transformations, spatial grid generation,
raster I/O helpers, and synthetic benchmark data generators for end-to-end execution.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union

import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point, Polygon, LineString
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS


def setup_logger(name: str = "wildfire_risk", log_level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standardized console/file logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(log_level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


LOGGER = setup_logger("wildfire_utils")


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Loads configuration YAML settings."""
    if config_path is None:
        # Default fallback to repo config
        base_dir = Path(__file__).resolve().parent.parent
        config_path = base_dir / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_project_root() -> Path:
    """Returns the absolute root path of the project repository."""
    return Path(__file__).resolve().parent.parent


def reproject_gdf(gdf: gpd.GeoDataFrame, target_crs: str = "EPSG:3310") -> gpd.GeoDataFrame:
    """Safely reprojects a GeoDataFrame to the specified target CRS."""
    if gdf.crs is None:
        LOGGER.warning("Input GeoDataFrame has no CRS set; assuming EPSG:4326 before transformation.")
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs.to_string() != target_crs:
        LOGGER.debug(f"Reprojecting GeoDataFrame from {gdf.crs.to_string()} to {target_crs}")
        return gdf.to_crs(target_crs)
    return gdf


def create_regular_grid(
    bbox: Tuple[float, float, float, float],
    resolution_m: float = 1000.0,
    crs: str = "EPSG:3310",
    block_size_km: float = 25.0
) -> gpd.GeoDataFrame:
    """
    Constructs a regular vector fishnet grid covering the given bounding box.
    Assigns each cell a unique cell_id, centroid coordinates, and a spatial block ID
    to support spatial block cross-validation (preventing spatial autocorrelation leakage).

    Parameters:
        bbox: (minx, miny, maxx, maxy) in target projected coordinates.
        resolution_m: cell width and height in meters (e.g. 250.0 or 1000.0).
        crs: target coordinate reference system (default EPSG:3310).
        block_size_km: block dimension for spatial cross-validation folding.

    Returns:
        GeoDataFrame of polygon grid cells with cell_id, block_id, row, col, and centroid.
    """
    minx, miny, maxx, maxy = bbox
    cols = int(np.ceil((maxx - minx) / resolution_m))
    rows = int(np.ceil((maxy - miny) / resolution_m))

    block_m = block_size_km * 1000.0

    cells = []
    cell_ids = []
    block_ids = []
    grid_rows = []
    grid_cols = []
    centroids_x = []
    centroids_y = []

    for i in range(cols):
        x0 = minx + i * resolution_m
        x1 = x0 + resolution_m
        block_col = int((x0 - minx) // block_m)

        for j in range(rows):
            y0 = miny + j * resolution_m
            y1 = y0 + resolution_m
            block_row = int((y0 - miny) // block_m)

            cell_poly = box(x0, y0, x1, y1)
            cell_id = f"CELL_{i:04d}_{j:04d}"
            block_id = f"BLOCK_{block_col:02d}_{block_row:02d}"

            cells.append(cell_poly)
            cell_ids.append(cell_id)
            block_ids.append(block_id)
            grid_cols.append(i)
            grid_rows.append(j)
            centroids_x.append((x0 + x1) / 2.0)
            centroids_y.append((y0 + y1) / 2.0)

    gdf = gpd.GeoDataFrame(
        {
            "cell_id": cell_ids,
            "block_id": block_ids,
            "grid_col": grid_cols,
            "grid_row": grid_rows,
            "centroid_x": centroids_x,
            "centroid_y": centroids_y,
            "geometry": cells
        },
        crs=crs
    )
    LOGGER.info(f"Generated regular grid with {len(gdf)} cells ({cols}x{rows}) in {crs}")
    return gdf


def save_geotiff(
    array: np.ndarray,
    output_path: Union[str, Path],
    transform: rasterio.Affine,
    crs: Union[str, CRS] = "EPSG:3310",
    nodata_val: float = -9999.0,
    dtype: np.dtype = np.float32
) -> Path:
    """
    Exports a 2D numpy array to a single-band GeoTIFF with geospatial metadata and LZW compression.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if array.ndim == 2:
        height, width = array.shape
        count = 1
        data = array[np.newaxis, :, :].astype(dtype)
    elif array.ndim == 3:
        count, height, width = array.shape
        data = array.astype(dtype)
    else:
        raise ValueError(f"Array must be 2D or 3D, got shape: {array.shape}")

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": dtype,
        "crs": CRS.from_user_input(crs) if isinstance(crs, str) else crs,
        "transform": transform,
        "nodata": nodata_val,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)

    LOGGER.info(f"Successfully wrote GeoTIFF to: {output_path} ({width}x{height}, count={count})")
    return output_path


def generate_benchmark_synthetic_data(
    num_grid_cells: int = 1200,
    random_seed: int = 42
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Generates realistic, physically consistent synthetic datasets for California wildfire risk testing:
    1. Grid cells with topographical, meteorological, vegetation, and distance covariates.
    2. Historical fire detections (NASA FIRMS mock).
    3. Road infrastructure network (TIGER mock).
    """
    np.random.seed(random_seed)

    # Sierra Nevada / Central California Albers coordinates: approx [minx=-50000, miny=-100000, maxx=100000, maxy=50000]
    minx, miny, maxx, maxy = -50000.0, -100000.0, 50000.0, 0.0
    res = 2500.0  # 2.5 km cell resolution for fast benchmark execution
    grid_gdf = create_regular_grid((minx, miny, maxx, maxy), resolution_m=res, crs="EPSG:3310", block_size_km=25.0)

    n = len(grid_gdf)
    cx = grid_gdf["centroid_x"].values
    cy = grid_gdf["centroid_y"].values

    # Elevation: higher towards eastern Sierra mountains (cx increasing)
    elevation_m = np.clip(300.0 + 0.02 * (cx - minx) + np.random.normal(0, 150, n), 100, 3800)
    # Slope: steepness correlates with terrain gradient
    slope_degrees = np.clip(np.abs(np.gradient(elevation_m)) * 0.15 + np.random.exponential(8, n), 0, 55)
    # Aspect: 0 to 360 azimuth
    aspect_degrees = np.random.uniform(0, 360, n)

    # Weather: hot dry interior vs cooler high elevation
    daily_max_temp_c = np.clip(38.0 - (elevation_m / 1000.0) * 6.5 + np.random.normal(0, 2.5, n), 15, 45)
    daily_min_rh = np.clip(18.0 + (elevation_m / 1000.0) * 4.0 - np.random.exponential(4, n), 5, 60)
    wind_speed_ms = np.clip(np.random.gamma(shape=3.5, scale=2.5, size=n), 1.0, 28.0)
    wind_gust_ms = wind_speed_ms * np.random.uniform(1.3, 1.9, n)

    # FWI Calculation (Empirical approximation)
    fwi_score = np.clip(
        (daily_max_temp_c * 0.8) + (wind_speed_ms * 1.6) - (daily_min_rh * 0.7) + np.random.normal(0, 3, n),
        0.0,
        100.0
    )

    # Vegetation & Fuel: Scott & Burgan fuel model groups (1=Grass, 2=Shrub, 3=Timber, 4=Slash, 0=Non-burnable)
    fuel_model_group = np.random.choice([1, 2, 3, 4, 0], size=n, p=[0.25, 0.35, 0.30, 0.05, 0.05])
    canopy_cover_pct = np.where(fuel_model_group == 3, np.random.uniform(40, 85, n),
                       np.where(fuel_model_group == 2, np.random.uniform(15, 45, n),
                       np.where(fuel_model_group == 1, np.random.uniform(5, 20, n), 0.0)))

    # Anthropogenic proximity: mock primary roads
    road_line = LineString([(-45000, -95000), (0, -50000), (45000, -5000)])
    road_gdf = gpd.GeoDataFrame({"road_name": ["State Hwy 41 Corridor"], "geometry": [road_line]}, crs="EPSG:3310")
    dist_to_road_m = grid_gdf.geometry.centroid.distance(road_line).values
    pop_density_km2 = np.clip(np.exp(8.0 - (dist_to_road_m / 6000.0)) + np.random.exponential(5, n), 0, 850)

    # True Fire Susceptibility Logit (Ground Truth function for benchmark realism)
    z = (
        -4.2
        + 0.045 * fwi_score
        + 0.06 * slope_degrees
        + 0.03 * canopy_cover_pct
        - 0.00015 * dist_to_road_m
        + (fuel_model_group == 2) * 1.1   # Shrub chaparral high risk
        + (fuel_model_group == 3) * 0.8   # Timber high risk
        - (fuel_model_group == 0) * 8.0   # Non-burnable
    )
    fire_probability = 1.0 / (1.0 + np.exp(-z))
    fire_occurrence = (np.random.uniform(0, 1, n) < fire_probability).astype(int)

    grid_gdf["elevation_m"] = np.round(elevation_m, 1)
    grid_gdf["slope_degrees"] = np.round(slope_degrees, 1)
    grid_gdf["aspect_degrees"] = np.round(aspect_degrees, 1)
    grid_gdf["temp_max_c"] = np.round(daily_max_temp_c, 1)
    grid_gdf["relative_humidity_min"] = np.round(daily_min_rh, 1)
    grid_gdf["wind_speed_ms"] = np.round(wind_speed_ms, 1)
    grid_gdf["wind_gust_ms"] = np.round(wind_gust_ms, 1)
    grid_gdf["fwi_score"] = np.round(fwi_score, 1)
    grid_gdf["fuel_model_group"] = fuel_model_group
    grid_gdf["canopy_cover_pct"] = np.round(canopy_cover_pct, 1)
    grid_gdf["dist_to_road_m"] = np.round(dist_to_road_m, 1)
    grid_gdf["pop_density_km2"] = np.round(pop_density_km2, 1)
    grid_gdf["fire_occurrence"] = fire_occurrence
    grid_gdf["true_prob"] = np.round(fire_probability, 4)

    # Synthesize FIRMS satellite detections from positive occurrences
    firms_pts = []
    firms_frp = []
    firms_confidence = []
    for idx, row in grid_gdf[grid_gdf["fire_occurrence"] == 1].iterrows():
        # Spawn 1 to 4 thermal detection points within burning cell
        num_pts = np.random.randint(1, 5)
        for _ in range(num_pts):
            pt_x = row.centroid_x + np.random.uniform(-res / 2, res / 2)
            pt_y = row.centroid_y + np.random.uniform(-res / 2, res / 2)
            firms_pts.append(Point(pt_x, pt_y))
            firms_frp.append(np.random.gamma(shape=2.0, scale=35.0))  # Fire Radiative Power (MW)
            firms_confidence.append(np.random.randint(60, 100))

    firms_gdf = gpd.GeoDataFrame(
        {
            "frp_mw": firms_frp,
            "confidence": firms_confidence,
            "sensor": ["VIIRS_SNPP"] * len(firms_pts),
            "acq_date": pd.date_range("2023-06-01", periods=len(firms_pts), freq="12h").strftime("%Y-%m-%d"),
            "geometry": firms_pts
        },
        crs="EPSG:3310"
    )

    LOGGER.info(
        f"Synthesized benchmark dataset: {len(grid_gdf)} cells ({grid_gdf['fire_occurrence'].sum()} burned), "
        f"{len(firms_gdf)} FIRMS detections, {len(road_gdf)} road corridors."
    )
    return grid_gdf, firms_gdf, road_gdf
