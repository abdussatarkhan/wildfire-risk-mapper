# Raw Data Sources & Ingestion Protocols

This directory stores raw geospatial, satellite remote sensing, weather, and administrative datasets utilized by the **Wildfire Spread Risk Mapper**. Due to large file sizes and licensing/distribution terms, large raw files are not committed to Git and should be retrieved using the automated scripts in `scripts/data_collection.py` or manually via the institutional portals below.

---

## 1. NASA FIRMS (Fire Information for Resource Management System)

- **Provider**: NASA Earth Observing System Data and Information System (EOSDIS) / LANCE
- **Instruments**:
  - **VIIRS (Visible Infrared Imaging Radiometer Suite)** on Suomi-NPP and NOAA-20 (375m spatial resolution, I-band 3.75 µm thermal anomalies)
  - **MODIS (Moderate Resolution Imaging Spectroradiometer)** on Terra and Aqua (1km spatial resolution)
- **Coverage**: California / Western United States (2014 – Present, ~5M historical detection records)
- **API Endpoint**: `https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{AREA_COORDINATES}/{DAY_RANGE}`
- **Manual Download**:
  1. Visit the [NASA FIRMS Archive Download](https://firms.modaps.eosdis.nasa.gov/download/).
  2. Select Area: United States / California Bounding Box `[-124.5, 32.5, -114.1, 42.0]`.
  3. Select Sensors: VIIRS S-NPP (375m) & VIIRS NOAA-20 (375m).
  4. Date range: 2014-01-01 through 2024-12-31.
  5. Format: Shapefile (.shp) or CSV (`DL_FIRE_SV-C2_*.csv`).
  6. Place files in `data/raw/firms/`.

---

## 2. NOAA Fire Weather & Meteorological Grids

- **Provider**: National Oceanic and Atmospheric Administration (NOAA) / National Centers for Environmental Information (NCEI)
- **Products**:
  - **RTMA (Real-Time Mesoscale Analysis)**: 2.5 km surface temperature, relative humidity, wind speed, wind gust, dew point.
  - **RAWS (Remote Automated Weather Stations)**: Calibrated surface stations managed by USFS/BLM for Canadian Fire Weather Index (FWI) components:
    - FFMC (Fine Fuel Moisture Code)
    - DMC (Duff Moisture Code)
    - DC (Drought Code)
    - ISI (Initial Spread Index)
    - BUI (Buildup Index)
    - FWI (Fire Weather Index)
- **Automated Ingestion**:
  - Script: `python scripts/data_collection.py --source noaa --year 2023`
  - Saves tabular netCDF/GRIB2 / CSV files to `data/raw/noaa/`.

---

## 3. USGS LANDFIRE (Landscape Fire and Resource Management Planning Tools)

- **Provider**: US Geological Survey (USGS) & US Department of Agriculture Forest Service (USFS)
- **Resolution**: 30m native raster grid, CONUS extent
- **Required Layers**:
  - **FBFM40**: Scott and Burgan 40 Fire Behavior Fuel Models (surface fuel bed characteristics: timber litter, grass-shrub, slash-blowdown).
  - **EVT**: Existing Vegetation Type (ecological systems).
  - **CC**: Forest Canopy Cover (percentage 0–100%).
  - **CH**: Canopy Height (decimeters).
  - **CBH**: Canopy Base Height (decimeters to canopy fuel load).
  - **3DEP DEM**: Digital Elevation Model (USGS 3D Elevation Program 1 arc-second ~ 30m).
- **Download Instructions**:
  1. Access the [LANDFIRE Data Distribution Site (LFDDS)](https://www.landfire.gov/viewer/).
  2. Define Area of Interest (AOI): California bounding box or regional mega-fires (e.g., Camp Fire, Dixie Fire, August Complex).
  3. Select Version: LANDFIRE 2022 / 2023 Refresh (LF 2.3.0).
  4. Download GeoTIFF / LCP zip archives into `data/raw/landfire/`.

---

## 4. US Census Bureau TIGER/Line & Housing Density

- **Provider**: US Census Bureau Geography Division
- **Layers**:
  - **Primary & Secondary Roads (TIGER/Line)**: State highways, interstates, arterial roads representing human ignition vectors and fire suppression access corridors.
  - **Census Blocks & Tracts**: Decennial housing counts and resident population for computing spatial population density and Wildland-Urban Interface (WUI) exposure.
- **Download Instructions**:
  - FTP/HTTP: `https://www2.census.gov/geo/tiger/TIGER2023/ROADS/` and `https://www2.census.gov/geo/tiger/TIGER2023/TRACT/`
  - California FIPS: `06`
  - Unzip shapefiles into `data/raw/tiger/`.

---

## Directory Organization

```text
data/raw/
├── firms/
│   ├── fire_nrt_sv-c2_california.csv
│   └── fire_archive_sv-c2_california.csv
├── landfire/
│   ├── LF2022_FBFM40_CONUS.tif
│   ├── LF2022_EVT_CONUS.tif
│   ├── LF2022_CC_CONUS.tif
│   └── USGS_3DEP_California_DEM.tif
├── noaa/
│   ├── noaa_fwi_grid_daily.nc
│   └── raws_california_weather_2014_2024.csv
└── tiger/
    ├── tl_2023_06_prisecroads.shp
    └── tl_2023_06_tract.shp
```
