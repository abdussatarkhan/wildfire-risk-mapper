"""
Risk Map Generation and Cartographic Synthesis Engine for Wildfire Spread Risk Mapper.
Performs:
1. Model inference across all landscape grid units to generate continuous fire spread probabilities.
2. 5-Tier risk classification (Low, Moderate, High, Very High, Extreme) based on state hazard standards.
3. Multi-band GeoTIFF raster export (Band 1: Float32 probability, Band 2: UInt8 risk tier).
4. Export of optimized GeoJSON summaries for interactive web mapping (EPSG:4326).
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin

from utils import setup_logger, load_config, get_project_root, save_geotiff, reproject_gdf

LOGGER = setup_logger("risk_map_generation")


class RiskMapGenerator:
    """Computes landscape hazard risk surfaces and produces GIS export artifacts."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.root = get_project_root()
        self.tier_config = self.config["risk_classification"]["tiers"]

    def load_model_and_data(
        self,
        model_path: Optional[Path] = None,
        data_path: Optional[Path] = None
    ) -> Tuple[Any, List[str], gpd.GeoDataFrame]:
        """Loads serialized model artifact and spatial grid dataframe."""
        m_path = model_path or (self.root / "models" / "wildfire_rf_risk_model.joblib")
        d_path = data_path or (self.root / "data" / "processed" / "ca_wildfire_grid_250m.parquet")

        if not m_path.exists() or not d_path.exists():
            LOGGER.warning("Model or data files not found. Triggering upstream modeling pipeline...")
            from modeling import WildfireRiskModeler
            from feature_extraction import FeatureExtractor
            extractor = FeatureExtractor()
            grid = extractor.build_grid_features(use_mock_data=True)
            extractor.export_datasets(grid)
            modeler = WildfireRiskModeler()
            modeler.evaluate_spatial_cv(grid, model_type="random_forest")
            modeler.train_final_model(grid, output_path=m_path)

        bundle = joblib.load(m_path)
        model = bundle["model"]
        feature_names = bundle["feature_names"]
        gdf = gpd.read_parquet(d_path)

        LOGGER.info(f"Loaded model and {len(gdf)} grid cells across {len(feature_names)} features.")
        return model, feature_names, gdf

    def predict_risk_tiers(
        self,
        model: Any,
        feature_names: List[str],
        gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Calculates predicted wildfire probabilities and discretizes into the 5 authoritative hazard tiers.
        """
        LOGGER.info("Computing fire risk probabilities across all grid units...")
        X = gdf[feature_names]
        probs = model.predict_proba(X)[:, 1]

        result_gdf = gdf.copy()
        result_gdf["predicted_probability"] = np.round(probs, 4)

        # Discretize into 5 risk tiers (1: Low to 5: Extreme)
        conditions = [
            probs < 0.20,
            (probs >= 0.20) & (probs < 0.40),
            (probs >= 0.40) & (probs < 0.60),
            (probs >= 0.60) & (probs < 0.80),
            probs >= 0.80
        ]
        tier_codes = [1, 2, 3, 4, 5]
        tier_names = ["Low", "Moderate", "High", "Very High", "Extreme"]
        tier_colors = ["#2ca25f", "#ffeb3b", "#fe9929", "#d95f0e", "#990000"]

        result_gdf["risk_tier"] = np.select(conditions, tier_codes, default=1)
        result_gdf["risk_label"] = np.select(conditions, tier_names, default="Low")
        result_gdf["risk_color"] = np.select(conditions, tier_colors, default="#2ca25f")

        # Action guidance from config
        guidance_map = {int(k): v["action_guidance"] for k, v in self.tier_config.items()}
        result_gdf["action_guidance"] = result_gdf["risk_tier"].map(guidance_map)

        # Tabulate area distribution
        counts = result_gdf["risk_label"].value_counts()
        LOGGER.info("=== LANDSCAPE WILDFIRE RISK DISTRIBUTION ===")
        for name, count in counts.items():
            pct = (count / len(result_gdf)) * 100
            LOGGER.info(f"  Tier {name:10s}: {count:5d} cells ({pct:5.1f}%)")

        return result_gdf

    def export_risk_geotiff(
        self,
        scored_gdf: gpd.GeoDataFrame,
        output_tif: Optional[Path] = None
    ) -> Path:
        """
        Exports the scored grid as a dual-band 250m GeoTIFF:
        - Band 1: Continuous Risk Probability (float32)
        - Band 2: Discrete Risk Tier (uint8)
        """
        out_tif = output_tif or (self.root / "data" / "processed" / "california_wildfire_risk_250m.tif")
        out_tif.parent.mkdir(parents=True, exist_ok=True)

        # Extract grid array dimensions from col/row
        max_col = scored_gdf["grid_col"].max() + 1
        max_row = scored_gdf["grid_row"].max() + 1

        prob_array = np.full((max_row, max_col), -9999.0, dtype=np.float32)
        tier_array = np.zeros((max_row, max_col), dtype=np.uint8)

        minx = scored_gdf.geometry.bounds["minx"].min()
        maxy = scored_gdf.geometry.bounds["maxy"].max()
        res = self.config["spatial"]["grid_resolution_meters"]

        for _, row in scored_gdf.iterrows():
            c = int(row["grid_col"])
            r = int(row["grid_row"])
            prob_array[r, c] = float(row["predicted_probability"])
            tier_array[r, c] = int(row["risk_tier"])

        transform = from_origin(minx, maxy, res, res)
        crs = self.config["spatial"]["crs_projected"]

        profile = {
            "driver": "GTiff",
            "height": max_row,
            "width": max_col,
            "count": 2,
            "dtype": np.float32,
            "crs": crs,
            "transform": transform,
            "nodata": -9999.0,
            "compress": "lzw"
        }

        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(prob_array, 1)
            dst.write(tier_array.astype(np.float32), 2)
            dst.set_band_description(1, "Wildfire_Spread_Probability_0_1")
            dst.set_band_description(2, "Risk_Tier_1_to_5")

        LOGGER.info(f"Wrote dual-band GeoTIFF hazard surface to: {out_tif}")
        return out_tif

    def export_geojson_summary(
        self,
        scored_gdf: gpd.GeoDataFrame,
        output_geojson: Optional[Path] = None,
        sample_limit: int = 1500
    ) -> Path:
        """
        Converts top risk units to WGS84 GeoJSON format for Leaflet/Folium web visualization.
        """
        out_json = output_geojson or (self.root / "data" / "processed" / "wildfire_risk_summary.geojson")
        out_json.parent.mkdir(parents=True, exist_ok=True)

        wgs84_gdf = reproject_gdf(scored_gdf, "EPSG:4326")

        cols_to_keep = [
            "cell_id", "block_id", "predicted_probability", "risk_tier", "risk_label",
            "risk_color", "fwi_score", "slope_degrees", "canopy_cover_pct",
            "dist_to_road_m", "action_guidance", "geometry"
        ]
        avail_cols = [c for c in cols_to_keep if c in wgs84_gdf.columns]

        export_gdf = wgs84_gdf[avail_cols].iloc[:sample_limit]
        export_gdf.to_file(out_json, driver="GeoJSON")

        LOGGER.info(f"Exported interactive GeoJSON layer ({len(export_gdf)} features) to: {out_json}")
        return out_json


def main():
    parser = argparse.ArgumentParser(description="Generate risk surfaces, classify tiers, and export GeoTIFF.")
    parser.add_argument("--output-tif", type=str, default="data/processed/california_wildfire_risk_250m.tif")
    parser.add_argument("--output-json", type=str, default="data/processed/wildfire_risk_summary.geojson")
    args = parser.parse_args()

    root = get_project_root()
    generator = RiskMapGenerator()

    model, feature_names, gdf = generator.load_model_and_data()
    scored_gdf = generator.predict_risk_tiers(model, feature_names, gdf)

    generator.export_risk_geotiff(scored_gdf, output_tif=root / args.output_tif)
    generator.export_geojson_summary(scored_gdf, output_geojson=root / args.output_json)
    LOGGER.info("Risk map generation workflow completed.")


if __name__ == "__main__":
    main()
