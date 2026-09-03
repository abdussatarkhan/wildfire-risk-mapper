# Wildfire Hazard Dashboards & Interactive Cartography

This directory outlines the interactive geospatial visualization architecture, operational dashboard interfaces, and GIS desktop integration protocols for the **Wildfire Spread Risk Mapper**.

---

## 1. Interactive Folium / Leaflet Web Application

The primary interactive application is located in `app/map_app.py`, generating `app/wildfire_risk_map.html`.

### Key Features:
- **Multi-Basemap Switcher**:
  - *Esri World Imagery*: High-resolution optical satellite basemap with terrain texture.
  - *CartoDB Positron*: High-contrast light cartographic canvas for clear hazard vector delineation.
  - *OpenStreetMap*: Authoritative road networks and municipal naming conventions.
- **Dynamic 250m Hazard Grid Choropleth**:
  - Color-coded by state hazard standard:
    - <span style="color:#2ca25f; font-weight:bold;">Low Tier 1</span> (`#2ca25f`): Spread prob < 0.20
    - <span style="color:#ffeb3b; font-weight:bold;">Moderate Tier 2</span> (`#ffeb3b`): Spread prob 0.20 – 0.40
    - <span style="color:#fe9929; font-weight:bold;">High Tier 3</span> (`#fe9929`): Spread prob 0.40 – 0.60
    - <span style="color:#d95f0e; font-weight:bold;">Very High Tier 4</span> (`#d95f0e`): Spread prob 0.60 – 0.80
    - <span style="color:#990000; font-weight:bold;">Extreme Tier 5</span> (`#990000`): Spread prob &ge; 0.80
- **Interactive Diagnostic Popups**:
  Clicking any grid cell reveals:
  - Grid Unit ID and 25km Planning Block ID
  - Calibrated Wildfire Spread Probability
  - Canadian Fire Weather Index (FWI)
  - Topographic Slope Angle (Degrees)
  - Canopy Fuel Cover Percentage (%)
  - Euclidean Distance to Road Network (Meters)
  - Specific Incident Directive (e.g. "Public Safety Power Shutoff warning")
- **Active Satellite Thermal Detections**:
  - NASA FIRMS VIIRS 375m thermal anomaly points with radius dynamically scaled by Fire Radiative Power (MW).
- **Measurement & Spatial Inspection Widgets**:
  - Fullscreen toggle, Leaflet MiniMap overview, and metric geodesic distance/area measurement tools.

### Launching the Dashboard:
```bash
# Generate the latest HTML map document
python app/map_app.py

# Launch interactive local HTTP server
python app/map_app.py --serve --port 8080
# Open http://localhost:8080/wildfire_risk_map.html in your browser
```

---

## 2. Desktop GIS Integration (QGIS & ArcGIS Pro)

The generated risk outputs are packaged in native OGC-standard GIS formats:

### 1. Dual-Band GeoTIFF (`data/processed/california_wildfire_risk_250m.tif`)
- **Band 1**: Continuous wildfire spread probability ($0.000$ to $1.000$, Float32).
  - *Recommended Styling*: Singleband pseudocolor with `YlOrRd` or `Magma` colormap.
- **Band 2**: Discrete Risk Tier ($1$ to $5$, UInt8).
  - *Recommended Styling*: Paletted / Unique values with Low (Green) to Extreme (Red) classification.

### 2. High-Performance GeoJSON (`data/processed/wildfire_risk_summary.geojson`)
- Reprojected to `EPSG:4326` (WGS84) for seamless ingestion into web GIS, ArcGIS Online, Mapbox, or QGIS vector styling.

---

## 3. Power BI & Tableau Business Intelligence Workflows

1. Connect to PostgreSQL / PostGIS database using ODBC/PostgreSQL connector.
2. Direct-query the `view_wui_critical_hazard_cells` and `wildfire_risk_surfaces` tables.
3. Build KPI cards:
   - Total California Acres in Extreme / Very High Hazard Zones.
   - Resident Population Exposed within 500m of Wildland-Urban Interface.
   - Road Corridors intersecting critical hazard zones.
4. Integrate Mapbox visual within Power BI using the GeoJSON summary.
