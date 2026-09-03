# External Reference Boundaries & Vector Layers

This directory houses authoritative boundary shapefiles, administrative geometries, and contextual spatial reference datasets used to mask, aggregate, and validate wildfire risk models across the state of California and the broader Western United States.

---

## Authoritative Reference Datasets

### 1. CAL FIRE Fire Perimeters (FRAP Database)
- **Source**: California Department of Forestry and Fire Protection (CAL FIRE) Fire and Resource Assessment Program (FRAP).
- **Format**: File Geodatabase (`.gdb`) or Shapefile (`.shp`).
- **Description**: Historic fire perimeters from 1878 to present, recording fire name, alarm date, containment date, cause, and total burned acreage.
- **Reference URL**: `https://frap.fire.ca.gov/frap-projects/fire-perimeters/`
- **Coordinate System**: NAD83 / California Albers (`EPSG:3310`).

### 2. SILVIS Wildland-Urban Interface (WUI) Maps
- **Source**: University of Wisconsin-Madison SILVIS Lab / USDA Forest Service.
- **Format**: Shapefile (`wui_ca_2020.shp`).
- **Description**: Classifies census blocks into Interface WUI (housing structures immediately adjacent to wildland fuels), Intermix WUI (structures intermingled with dense wildland vegetation), and Non-WUI.
- **Reference URL**: `http://silvis.forest.wisc.edu/data/wui-change/`

### 3. CPUC High Fire-Threat District (HFTD)
- **Source**: California Public Utilities Commission (CPUC).
- **Format**: GeoJSON (`cpuc_hftd_tier2_tier3.geojson`).
- **Description**: Designates Tier 2 (Elevated Risk) and Tier 3 (Extreme Risk) areas where enhanced utility wildfire mitigation and Public Safety Power Shutoff (PSPS) protocols are enacted.
- **Reference URL**: `https://www.cpuc.ca.gov/firethreatmaps/`

### 4. US Census California County Boundaries
- **Source**: US Census Bureau Cartographic Boundary Files.
- **Format**: GeoJSON (`ca_counties_cb_2020.geojson`).
- **Attributes**: `STATEFP`, `COUNTYFP`, `NAME`, `ALAND`, `AWATER`.

---

## Coordinate Reference System (CRS) Standards

All analytical operations are normalized to the **California Albers Equal Area** projection:
- **EPSG**: `3310`
- **Datum**: North American Datum 1983 (NAD83)
- **Ellipsoid**: GRS 1980
- **Units**: Meters
- **Standard Parallels**: 34°00'N, 40°30'N
- **Central Meridian**: 120°00'W
- **Latitude of Origin**: 0°00'N
- **False Easting**: 0.0 m
- **False Northing**: -4,000,000.0 m

Why `EPSG:3310`?
Equal-area projections preserve true surface area across California's 1,300 km length, eliminating spatial distortion in cell density, distance calculations, and perimeter-to-area metrics.

---

## Ingestion Workflow
Run the collection utility to fetch or synthesize reference boundaries:
```bash
python scripts/data_collection.py --source reference
```
