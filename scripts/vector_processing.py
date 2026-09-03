"""
Vector Processing Engine for Wildfire Spread Risk Mapper.
Performs spatial proximity analysis, anthropogenic hazard modeling, and areal interpolation:
1. Distance-to-road network (TIGER/Line primary & secondary transportation corridors).
2. Distance-to-structure / infrastructure assets.
3. Spatial population density via areal-weighted intersection with Census demographic tracts.
4. Wildland-Urban Interface (WUI) classification and zone attribution.
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
from shapely.geometry import Point, LineString, Polygon, box
from shapely.ops import nearest_points

from utils import setup_logger, load_config, get_project_root, reproject_gdf, create_regular_grid

LOGGER = setup_logger("vector_processing")


class VectorProcessor:
    """Performs spatial proximity calculations, spatial joins, and density interpolations."""

    def __init__(self, target_crs: str = "EPSG:3310"):
        self.target_crs = target_crs

    def compute_distance_to_roads(
        self,
        grid_gdf: gpd.GeoDataFrame,
        roads_gdf: gpd.GeoDataFrame,
        col_name: str = "dist_to_road_m"
    ) -> gpd.GeoDataFrame:
        """
        Computes minimum Euclidean distance (in meters) from each grid centroid to the nearest road corridor.
        Uses unary union of roads for rapid vectorized projection.
        """
        grid = reproject_gdf(grid_gdf, self.target_crs)
        roads = reproject_gdf(roads_gdf, self.target_crs)

        LOGGER.info(f"Computing Euclidean distances from {len(grid)} cells to road corridors ({len(roads)} lines)...")
        road_union = roads.geometry.unary_union

        # Measure distance from polygon centroid to nearest point on road network
        centroids = grid.geometry.centroid
        distances = centroids.distance(road_union)

        result_gdf = grid.copy()
        result_gdf[col_name] = np.round(distances.values, 2)
        LOGGER.info(f"Calculated {col_name}: mean={result_gdf[col_name].mean():.1f}m, min={result_gdf[col_name].min():.1f}m")
        return result_gdf

    def compute_distance_to_structures(
        self,
        grid_gdf: gpd.GeoDataFrame,
        structures_gdf: gpd.GeoDataFrame,
        col_name: str = "dist_to_structure_m"
    ) -> gpd.GeoDataFrame:
        """
        Computes distance from grid cells to nearest human infrastructure/building cluster.
        """
        grid = reproject_gdf(grid_gdf, self.target_crs)
        structs = reproject_gdf(structures_gdf, self.target_crs)

        LOGGER.info(f"Computing proximity to {len(structs)} structure geometries...")
        struct_union = structs.geometry.unary_union

        distances = grid.geometry.centroid.distance(struct_union)
        result_gdf = grid.copy()
        result_gdf[col_name] = np.round(distances.values, 2)
        return result_gdf

    def compute_population_density(
        self,
        grid_gdf: gpd.GeoDataFrame,
        census_tracts_gdf: gpd.GeoDataFrame,
        pop_col: str = "population",
        col_name: str = "population_density_km2"
    ) -> gpd.GeoDataFrame:
        """
        Estimates population density within each grid cell using areal-weighted overlay.
        Grid cell population is proportionally allocated based on intersection area with Census tracts.
        """
        grid = reproject_gdf(grid_gdf, self.target_crs)
        tracts = reproject_gdf(census_tracts_gdf, self.target_crs)

        LOGGER.info("Computing areal-weighted population density across grid units...")
        tracts = tracts.copy()
        tracts["tract_area_m2"] = tracts.geometry.area

        # Intersect grid and tracts
        overlay = gpd.overlay(grid[["cell_id", "geometry"]], tracts[["geometry", pop_col, "tract_area_m2"]], how="intersection")
        overlay["intersection_area_m2"] = overlay.geometry.area

        # Proportionate population allocated to intersection slice
        overlay["apportioned_pop"] = overlay[pop_col] * (overlay["intersection_area_m2"] / overlay["tract_area_m2"])

        # Aggregate back by cell_id
        cell_pop = overlay.groupby("cell_id")["apportioned_pop"].sum().reset_index()

        result_gdf = grid.merge(cell_pop, on="cell_id", how="left")
        result_gdf["apportioned_pop"] = result_gdf["apportioned_pop"].fillna(0.0)

        # Convert cell area (m^2) to km^2
        cell_area_km2 = result_gdf.geometry.area / 1e6
        result_gdf[col_name] = np.round(result_gdf["apportioned_pop"] / cell_area_km2, 2)
        result_gdf.drop(columns=["apportioned_pop"], inplace=True)

        LOGGER.info(f"Areal population density computed. Max={result_gdf[col_name].max():.1f} people/km²")
        return result_gdf

    def classify_wui_exposure(
        self,
        grid_gdf: gpd.GeoDataFrame,
        dist_road_col: str = "dist_to_road_m",
        pop_density_col: str = "population_density_km2",
        col_name: str = "wui_class"
    ) -> gpd.GeoDataFrame:
        """
        Classifies each cell into Wildland-Urban Interface (WUI) categories:
        - 0: Non-WUI (uninhabited wildland or high-density urban core)
        - 1: Intermix (low-to-moderate housing density embedded in wildland fuels)
        - 2: Interface (housing adjacent to wildland fuels within evacuation reach)
        """
        result_gdf = grid_gdf.copy()
        conditions = [
            (result_gdf[pop_density_col] >= 6.17) & (result_gdf[dist_road_col] <= 500.0),
            (result_gdf[pop_density_col] >= 2.0) & (result_gdf[dist_road_col] <= 2500.0),
        ]
        choices = [2, 1]  # 2=Interface, 1=Intermix
        result_gdf[col_name] = np.select(conditions, choices, default=0)
        LOGGER.info(f"WUI exposure classified: {result_gdf[col_name].value_counts().to_dict()}")
        return result_gdf

    def generate_synthetic_vectors(
        self,
        grid_gdf: gpd.GeoDataFrame
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Synthesizes mock road networks, structures, and census tracts if external raw files are absent."""
        LOGGER.info("Generating synthetic California infrastructure and demographic vectors...")
        bounds = grid_gdf.total_bounds
        minx, miny, maxx, maxy = bounds

        # 1. Main highway network lines
        road_geoms = [
            LineString([(minx + 10000, miny), (minx + 40000, (miny + maxy) / 2), (maxx - 15000, maxy)]),
            LineString([(minx, miny + 30000), (maxx, miny + 40000)]),
            LineString([(maxx - 20000, miny), ((minx + maxx) / 2, maxy)])
        ]
        roads_gdf = gpd.GeoDataFrame(
            {"road_id": [f"HWY_{i}" for i in range(len(road_geoms))], "type": ["Primary", "Secondary", "Secondary"], "geometry": road_geoms},
            crs=self.target_crs
        )

        # 2. Structures / housing nodes
        np.random.seed(55)
        num_structs = 180
        struct_x = np.random.uniform(minx + 5000, maxx - 5000, num_structs)
        struct_y = np.random.uniform(miny + 5000, maxy - 5000, num_structs)
        # Cluster structures closer to the first highway
        struct_pts = [Point(x, y).buffer(np.random.uniform(50, 150)) for x, y in zip(struct_x, struct_y)]
        structs_gdf = gpd.GeoDataFrame(
            {"structure_id": [f"STR_{i:04d}" for i in range(num_structs)], "geometry": struct_pts},
            crs=self.target_crs
        )

        # 3. Census Tracts polygons
        tract_boxes = []
        tract_pops = []
        nx, ny = 4, 4
        dx = (maxx - minx) / nx
        dy = (maxy - miny) / ny
        for i in range(nx):
            for j in range(ny):
                tract_boxes.append(box(minx + i * dx, miny + j * dy, minx + (i + 1) * dx, miny + (j + 1) * dy))
                tract_pops.append(np.random.randint(1200, 15000))

        tracts_gdf = gpd.GeoDataFrame(
            {"tract_id": [f"06019{i:04d}" for i in range(len(tract_boxes))], "population": tract_pops, "geometry": tract_boxes},
            crs=self.target_crs
        )

        return roads_gdf, structs_gdf, tracts_gdf


def main():
    parser = argparse.ArgumentParser(description="Process vector proximity, roads, and population density.")
    parser.add_argument("--output-parquet", type=str, default="data/processed/vector_features.parquet", help="Output Parquet")
    args = parser.parse_args()

    root = get_project_root()
    out_path = root / args.output_parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize processor
    vp = VectorProcessor(target_crs="EPSG:3310")

    # Generate benchmark grid
    grid = create_regular_grid((-50000.0, -100000.0, 50000.0, 0.0), resolution_m=2500.0, crs="EPSG:3310")
    roads_gdf, structs_gdf, tracts_gdf = vp.generate_synthetic_vectors(grid)

    grid = vp.compute_distance_to_roads(grid, roads_gdf)
    grid = vp.compute_distance_to_structures(grid, structs_gdf)
    grid = vp.compute_population_density(grid, tracts_gdf)
    grid = vp.classify_wui_exposure(grid)

    grid.to_parquet(out_path)
    LOGGER.info(f"Vector processing complete. Saved {len(grid)} spatial units to {out_path}")


if __name__ == "__main__":
    main()
