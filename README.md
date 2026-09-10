# Geospatial Wildfire Spread Risk & Hazard Mapping System

[![CI](https://github.com/abdussatarkhan/wildfire-risk-mapper/actions/workflows/ci.yml/badge.svg)](https://github.com/abdussatarkhan/wildfire-risk-mapper/actions)
[![GeoPandas](https://img.shields.io/badge/GIS-GeoPandas-2E7D32?style=for-the-badge&logo=python&logoColor=white)](https://geopandas.org/) [![Rasterio](https://img.shields.io/badge/Raster-Rasterio-00A8E8?style=for-the-badge)](https://rasterio.readthedocs.io/) [![Python](https://img.shields.io/badge/Python-Geospatial_AI-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Author](https://img.shields.io/badge/Author-Abdussatar-E50914?style=for-the-badge&logo=github&logoColor=white)](https://github.com/abdussatarkhan)

> **A geospatial predictive hazard model integrating satellite multispectral imagery (NDVI/EVI), meteorological indices (VPD, temperature, wind), and spatial autocorrelation (Moran's I) to forecast high-resolution wildfire ignition risks.**

---

## 🏛️ System Architecture

```mermaid
graph TD
    Satellite[Satellite Imagery: NDVI / Soil Moisture] --> GIS[GeoPandas & Rasterio Grid Processing]
    Weather[ERA5 Climate & Wind Vectors] --> GIS
    GIS --> SpatialML[Spatial Autocorrelation Moran's I & Random Forest]
    SpatialML --> HazardMap[Interactive Geospatial Risk Heatmap]
```

---

## 🌟 Key Features & Capabilities

- **Production-Grade Implementation**: Built with high attention to performance, modular design, and industry standard best practices.
- **Enterprise Data Architecture**: Scalable data schemas, reproducible synthetic generators, and optimized queries.
- **Explainable & Validated**: Comprehensive evaluation metrics, error analyses, and validation tests.
- **Comprehensive Tech Stack**: `Python` `GeoPandas` `Rasterio` `Shapely` `Scikit-Learn` `GIS`.

---

## 📊 Visual Preview & Analysis

<div align="center">

![wildfire-risk-mapper preview](images/wildfire_hazard_map.png)

</div>

<div align="center">

[![Daily Streak](https://img.shields.io/badge/Daily%20Streak-Active%20%F0%9F%94%A5-brightgreen?style=flat-square&logo=github)](https://github.com/abdussatarkhan)
[![Master Portfolio](https://img.shields.io/badge/Portfolio-50%2B%20Enterprise%20Projects-0e75b6?style=flat-square&logo=github)](https://github.com/abdussatarkhan/abdussatarkhan)
[![Author: Abdussatar](https://img.shields.io/badge/Author-Abdussatar-24292e?style=flat-square&logo=github)](https://github.com/abdussatarkhan)

</div>


---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/abdussatarkhan/wildfire-risk-mapper.git
cd wildfire-risk-mapper
```

### 2. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies (if requirements.txt exists)
pip install -r requirements.txt
```

---

## 🗺️ Roadmap & Upcoming Features

- [x] Satellite NDVI/EVI & ERA5 weather feature fusion
- [x] Spatial Autocorrelation (Moran's I) and hazard modeling
- [ ] High-resolution interactive Leaflet/Mapbox risk interface
- [ ] Automated daily MODIS / VIIRS satellite fire feed ingestion
- [ ] Wind spread physical vector simulation

---

## 👨‍💻 Author & Profile

Built and maintained by **Abdussatar** ([@abdussatarkhan](https://github.com/abdussatarkhan)).  
For technical discussions, collaboration, or queries, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/abdus-satar-5150813b5/) or [GitHub](https://github.com/abdussatarkhan).

---

## 📜 License

This project is licensed under the **MIT License** — see the LICENSE file for details.


---

<div align="center">

### 👨‍💻 Maintained by [Abdussatar (@abdussatarkhan)](https://github.com/abdussatarkhan)
Part of the **[Master Enterprise Data Analytics & AI Portfolio](https://github.com/abdussatarkhan/abdussatarkhan)**.

⭐ If you find this repository valuable, consider dropping a star! ⭐

</div>
