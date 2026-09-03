"""
Raster Processing Engine for Wildfire Spread Risk Mapper.
Handles:
1. Spatial coordinate transformation and reprojection to California Albers (EPSG:3310).
2. Raster resampling to consistent 250m analysis grid using bilinear and nearest-neighbor methods.
3. Topographic derivatives extraction: slope, aspect, and terrain ruggedness (TRI) from DEM.
4. Zonal statistics aggregation across vector spatial planning units.
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
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
from scipy.ndimage import uniform_filter

from utils import setup_logger, load_config, get_project_root, save_geotiff

LOGGER = setup_logger("raster_processing")


class RasterProcessor:
    """Core geospatial raster processing and transformation pipeline using Rasterio."""

    def __init__(self, target_crs: str = "EPSG:3310", target_res_m: float = 250.0):
        self.target_crs = target_crs
        self.target_res_m = float(target_res_m)

    def reproject_and_resample(
        self,
        src_path: Union[str, Path],
        dst_path: Union[str, Path],
        is_categorical: bool = False
    ) -> Path:
        """
        Reprojects an arbitrary raster to target CRS (default EPSG:3310) and resamples to target resolution.
        Uses nearest-neighbor interpolation for discrete classes (fuel models) and bilinear for continuous.
        """
        src_path = Path(src_path)
        dst_path = Path(dst_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        resample_alg = Resampling.nearest if is_categorical else Resampling.bilinear

        with rasterio.open(src_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs,
                self.target_crs,
                src.width,
                src.height,
                *src.bounds,
                resolution=(self.target_res_m, self.target_res_m)
            )

            kwargs = src.meta.copy()
            kwargs.update({
                "crs": self.target_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "nodata": src.nodata if src.nodata is not None else -9999.0,
                "compress": "lzw"
            })

            with rasterio.open(dst_path, "w", **kwargs) as dst:
                for band_idx in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=rasterio.band(dst, band_idx),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=self.target_crs,
                        resampling=resample_alg
                    )

        LOGGER.info(f"Reprojected and resampled: {src_path.name} -> {dst_path} (Method={resample_alg.name})")
        return dst_path

    def compute_topography(
        self,
        dem_path: Union[str, Path],
        output_dir: Union[str, Path]
    ) -> Dict[str, Path]:
        """
        Derives physical terrain covariates from DEM elevation:
        - Slope in degrees (Horn's 3x3 finite-difference algorithm)
        - Aspect in degrees (0 - 360 clockwise from North)
        - Terrain Ruggedness Index (TRI - standard deviation of local elevation)
        """
        dem_path = Path(dem_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata if src.nodata is not None else -9999.0
            cell_size_x = abs(transform.a)
            cell_size_y = abs(transform.e)

        valid_mask = dem != nodata

        # Compute gradient using central differences (Horn's method equivalent on regular grid)
        gy, gx = np.gradient(dem, cell_size_y, cell_size_x)

        # 1. Slope in degrees
        slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
        slope_deg = np.degrees(slope_rad)
        slope_deg[~valid_mask] = nodata

        # 2. Aspect in degrees: 0° (North), 90° (East), 180° (South), 270° (West)
        aspect_rad = np.arctan2(-gx, gy)
        aspect_deg = np.degrees(aspect_rad)
        aspect_deg = np.where(aspect_deg < 0, 90.0 - aspect_deg, 450.0 - aspect_deg) % 360.0
        aspect_deg[~valid_mask] = nodata

        # 3. Terrain Ruggedness Index (Local standard deviation in 3x3 window)
        mean = uniform_filter(dem, size=3, mode="reflect")
        mean_sq = uniform_filter(dem**2, size=3, mode="reflect")
        tri = np.sqrt(np.maximum(mean_sq - mean**2, 0.0))
        tri[~valid_mask] = nodata

        slope_path = output_dir / "terrain_slope_degrees_250m.tif"
        aspect_path = output_dir / "terrain_aspect_degrees_250m.tif"
        tri_path = output_dir / "terrain_ruggedness_index_250m.tif"

        save_geotiff(slope_deg, slope_path, transform, crs=crs, nodata_val=nodata)
        save_geotiff(aspect_deg, aspect_path, transform, crs=crs, nodata_val=nodata)
        save_geotiff(tri, tri_path, transform, crs=crs, nodata_val=nodata)

        LOGGER.info("Calculated topographic terrain variables: Slope, Aspect, and Ruggedness Index.")
        return {
            "slope": slope_path,
            "aspect": aspect_path,
            "tri": tri_path
        }

    def compute_zonal_statistics(
        self,
        raster_path: Union[str, Path],
        polygons_gdf: gpd.GeoDataFrame,
        stat_prefix: str = "val",
        stats: Tuple[str, ...] = ("mean", "max", "std")
    ) -> gpd.GeoDataFrame:
        """
        Computes summary statistics (mean, max, std) of raster pixels intersecting each polygon in GeoDataFrame.
        """
        raster_path = Path(raster_path)
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            transform = src.transform
            nodata = src.nodata if src.nodata is not None else -9999.0

            # Match polygon CRS with raster CRS
            if polygons_gdf.crs != src.crs:
                polygons_gdf = polygons_gdf.to_crs(src.crs)

            means = []
            maxs = []
            stds = []

            for geom in polygons_gdf.geometry:
                try:
                    mask = geometry_mask([geom], transform=transform, invert=True, out_shape=data.shape)
                    vals = data[mask]
                    vals = vals[vals != nodata]

                    if len(vals) > 0:
                        means.append(float(np.mean(vals)))
                        maxs.append(float(np.max(vals)))
                        stds.append(float(np.std(vals)))
                    else:
                        means.append(0.0)
                        maxs.append(0.0)
                        stds.append(0.0)
                except Exception:
                    means.append(0.0)
                    maxs.append(0.0)
                    stds.append(0.0)

        result_gdf = polygons_gdf.copy()
        if "mean" in stats:
            result_gdf[f"{stat_prefix}_mean"] = means
        if "max" in stats:
            result_gdf[f"{stat_prefix}_max"] = maxs
        if "std" in stats:
            result_gdf[f"{stat_prefix}_std"] = stds

        LOGGER.info(f"Aggregated zonal statistics for {len(result_gdf)} polygons from {raster_path.name}")
        return result_gdf


def main():
    parser = argparse.ArgumentParser(description="Process, reproject, and extract terrain variables from rasters.")
    parser.add_argument("--dem", type=str, default="data/raw/landfire/USGS_DEM_California_250m.tif", help="Path to raw DEM")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    args = parser.parse_args()

    root = get_project_root()
    dem_file = root / args.dem
    out_dir = root / args.output_dir

    if not dem_file.exists():
        LOGGER.warning(f"DEM file not found at {dem_file}. Generating synthetic LANDFIRE rasters first...")
        from data_collection import LANDFIRERasterDownloader
        downloader = LANDFIRERasterDownloader()
        downloader.generate_reference_rasters(dem_file.parent)

    processor = RasterProcessor(target_crs="EPSG:3310", target_res_m=250.0)
    topography_rasters = processor.compute_topography(dem_file, out_dir)
    LOGGER.info(f"Generated topographic raster suite: {list(topography_rasters.keys())}")


if __name__ == "__main__":
    main()
