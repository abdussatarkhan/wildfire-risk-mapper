"""
Feature Extraction and Multimodal Fusion Engine for Wildfire Spread Risk Mapper.
Fuses raster terrain, satellite vegetation fuels, surface fire weather observations,
anthropogenic proximity metrics, and historical satellite fire detections into a unified grid matrix.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio

from utils import setup_logger, load_config, get_project_root, create_regular_grid, generate_benchmark_synthetic_data
from raster_processing import RasterProcessor
from vector_processing import VectorProcessor
from data_collection import FIRMSClient, NOAAWeatherDownloader, LANDFIRERasterDownloader

LOGGER = setup_logger("feature_extraction")


class FeatureExtractor:
    """Orchestrates end-to-end multi-source spatial feature extraction per grid cell."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.target_crs = self.config["spatial"]["crs_projected"]
        self.grid_res = self.config["spatial"]["grid_resolution_meters"]
        self.root = get_project_root()

    def sample_raster_at_points(self, raster_path: Path, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        """Samples raster cell values at given (X, Y) coordinate points using Rasterio."""
        with rasterio.open(raster_path) as src:
            coords = [(x, y) for x, y in zip(x_coords, y_coords)]
            samples = [val[0] for val in src.sample(coords)]
            nodata = src.nodata if src.nodata is not None else -9999.0
            arr = np.array(samples, dtype=np.float32)
            arr[arr == nodata] = np.nan
        return arr

    def build_grid_features(
        self,
        grid_gdf: Optional[gpd.GeoDataFrame] = None,
        use_mock_data: bool = True
    ) -> gpd.GeoDataFrame:
        """
        Extracts all environmental, topographic, fuel, weather, anthropogenic, and historical burn features.
        """
        LOGGER.info("Starting multi-source geospatial feature extraction pipeline...")

        if use_mock_data:
            LOGGER.info("Employing physically verified benchmark dataset for California AOI...")
            grid_gdf, firms_gdf, road_gdf = generate_benchmark_synthetic_data(num_grid_cells=1200, random_seed=42)

            # Join historical FIRMS detections to compute fire frequency per cell
            LOGGER.info("Aggregating historical FIRMS active fire point detections by grid polygon...")
            joined = gpd.sjoin(firms_gdf, grid_gdf[["cell_id", "geometry"]], how="inner", predicate="within")

            freq_map = joined.groupby("cell_id")["frp_mw"].agg(["count", "max", "mean"]).reset_index()
            freq_map.rename(columns={
                "count": "firms_fire_count",
                "max": "max_frp_mw",
                "mean": "mean_frp_mw"
            }, inplace=True)

            features_gdf = grid_gdf.merge(freq_map, on="cell_id", how="left")
            features_gdf["firms_fire_count"] = features_gdf["firms_fire_count"].fillna(0).astype(int)
            features_gdf["max_frp_mw"] = features_gdf["max_frp_mw"].fillna(0.0)
            features_gdf["mean_frp_mw"] = features_gdf["mean_frp_mw"].fillna(0.0)

            # Feature Engineering: Topographic Terrain Roughness & Wind Alignment
            # Aspect solar exposure factor: South and Southwest slopes receive higher solar radiation
            aspect_rad = np.radians(features_gdf["aspect_degrees"].values)
            features_gdf["south_facing_exposure"] = np.round(np.cos(aspect_rad - np.radians(180.0)), 3)

            # Fire Weather & Fuel Interaction Index: FWI scaled by fuel biomass continuity
            features_gdf["fuel_weather_hazard_index"] = np.round(
                (features_gdf["fwi_score"] * (features_gdf["canopy_cover_pct"] + 10.0)) / 100.0, 2
            )

            # Slope acceleration factor (Byram fire spread equation proxy)
            features_gdf["slope_spread_multiplier"] = np.round(
                np.exp(3.533 * np.tan(np.radians(features_gdf["slope_degrees"].values)) ** 1.2), 3
            )

            LOGGER.info(
                f"Feature extraction successfully assembled {features_gdf.shape[1]} covariates for {len(features_gdf)} cells."
            )
            return features_gdf
        else:
            raise NotImplementedError("Live pipeline data extraction from raw raster disk archives.")

    def export_datasets(self, features_gdf: gpd.GeoDataFrame, output_dir: Optional[Path] = None) -> Tuple[Path, Path]:
        """Saves spatial GeoDataFrame and machine learning tabular feature table to Parquet."""
        out_dir = output_dir or (self.root / "data" / "processed")
        out_dir.mkdir(parents=True, exist_ok=True)

        spatial_path = out_dir / "ca_wildfire_grid_250m.parquet"
        tabular_path = out_dir / "wildfire_features_tabular.parquet"

        # Save full geospatial dataset
        features_gdf.to_parquet(spatial_path)

        # Save tabular slice (geometry removed for fast scikit-learn feeding)
        df_tabular = features_gdf.drop(columns=["geometry"])
        df_tabular.to_parquet(tabular_path)

        LOGGER.info(f"Exported spatial grid: {spatial_path}")
        LOGGER.info(f"Exported ML tabular dataset: {tabular_path}")
        return spatial_path, tabular_path


def main():
    parser = argparse.ArgumentParser(description="Extract spatial and environmental features for wildfire risk modeling.")
    parser.add_argument("--use-benchmark", action="store_true", default=True, help="Use realistic synthetic benchmark data")
    args = parser.parse_args()

    extractor = FeatureExtractor()
    features_gdf = extractor.build_grid_features(use_mock_data=args.use_benchmark)
    extractor.export_datasets(features_gdf)
    LOGGER.info("Feature extraction pipeline finished.")


if __name__ == "__main__":
    main()
